import copy

import torch
from torch import nn


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, outputs))


def normalize(state):
    offset = state.new_tensor([.1, 25, 25, 3.5, 0, .8, 0, 0])
    scale = state.new_tensor([.7, 20, 20, .7, 20, .2, .1, 1])
    return (state - offset) / scale


class Fuzzy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('rules', torch.tensor([[-1,-1,-1,-1],[0,-1,-1,-1],[-1,0,-1,-1],[-1,-1,0,-1],
            [-1,-1,-1,0],[1,0,-1,-1],[0,-1,0,-1],[-1,0,0,-1],[1,1,0,-1],[1,1,1,0],[2,1,1,2],[2,2,2,2]]))
        q = torch.tensor([.85,.2,.15,.1,.55,.3,.1,.08,.25,.5,.75,.95])
        self.beta = nn.Parameter(torch.logit(q))

    def forward(self, state):
        headroom = torch.stack([(45 - state[:, 1]) / 10, (4.2 - state[:, 3]) / .2,
                                state[:, 6] / .1, (state[:, 5] - .8) / .2], -1).clamp(0, 1)
        membership = (1 - (headroom[:, None] - self.rules.clamp_min(0)[None] * .5).abs() / .5).clamp_min(0)
        membership = torch.where(self.rules[None] < 0, torch.ones_like(membership), membership)
        firing = membership.prod(-1)
        weights = firing / (firing.sum(-1, keepdim=True) + 1e-8)
        return .2 + .8 * (weights * self.beta.sigmoid()).sum(-1), weights


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = network(8, 2)

    def forward(self, state, deterministic=False):
        mean, log_std = self.net(normalize(state)).chunk(2, -1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        logprob = (distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action[:, 0], logprob


class SAC:
    def __init__(self, variant='full'):
        self.variant = variant
        self.actor, self.fuzzy = Actor(), Fuzzy()
        self.reward = nn.ModuleList([network(9, 1), network(9, 1)])
        self.cost = network(9, 5)
        self.target_reward, self.target_cost = copy.deepcopy(self.reward), copy.deepcopy(self.cost)
        self.log_entropy = nn.Parameter(torch.zeros(()))
        self.dual = torch.zeros(5)
        parameters = list(self.actor.parameters())
        if variant != 'fixed_fuzzy':
            parameters += list(self.fuzzy.parameters())
        self.actor_opt = torch.optim.Adam(parameters, lr=3e-4)
        self.critic_opt = torch.optim.Adam(list(self.reward.parameters()) + list(self.cost.parameters()), lr=3e-4)
        self.entropy_opt = torch.optim.Adam([self.log_entropy], lr=3e-4)

    def action(self, state, safe, deterministic=False):
        action, logprob = self.actor(state, deterministic)
        requested = 10 * (action + 1)
        multiplier, firing = self.fuzzy(state)
        if self.variant in ('sac', 'lagrangian', 'no_fuzzy'):
            multiplier = torch.ones_like(multiplier)
        executed = requested * multiplier
        if self.variant not in ('sac', 'lagrangian', 'no_projection'):
            executed = torch.minimum(executed, safe)
        return executed, logprob, requested, firing

    def update(self, batch):
        state, executed, reward, costs, next_state, done, safe, next_safe = [torch.as_tensor(v, dtype=torch.float32) for v in batch]
        with torch.no_grad():
            next_current, next_logprob, _, _ = self.action(next_state, next_safe)
            next_input = torch.cat([normalize(next_state), (next_current / 10 - 1)[:, None]], 1)
            target_r = reward + .995 * (1 - done) * (torch.minimum(*[q(next_input)[:, 0] for q in self.target_reward]) - self.log_entropy.exp() * next_logprob)
            target_c = costs + .995 * (1 - done[:, None]) * self.target_cost(next_input)
        critic_input = torch.cat([normalize(state), (executed / 10 - 1)[:, None]], 1)
        critic_loss = sum((q(critic_input)[:, 0] - target_r).square().mean() for q in self.reward)
        critic_loss += (self.cost(critic_input) - target_c).square().mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        current, logprob, requested, _ = self.action(state, safe)
        actor_input = torch.cat([normalize(state), (current / 10 - 1)[:, None]], 1)
        objective = self.log_entropy.exp().detach() * logprob - torch.minimum(*[q(actor_input)[:, 0] for q in self.reward])
        if self.variant != 'sac':
            objective += (self.cost(actor_input) * self.dual).sum(-1)
        if self.variant not in ('sac', 'lagrangian', 'no_physics_penalty'):
            objective += 10 * ((requested - current) / 20).square()
        actor_loss = objective.mean()
        if self.variant not in ('sac', 'lagrangian', 'no_monotonic'):
            safer = state.clone()
            delta = torch.rand_like(state[:, :4])
            safer[:, 1] -= delta[:, 0] * 5
            safer[:, 3] -= delta[:, 1] * .1
            safer[:, 6] += delta[:, 2] * .05
            safer[:, 5] += delta[:, 3] * .1
            actor_loss += (self.fuzzy(state)[0] - self.fuzzy(safer)[0]).clamp_min(0).square().mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        entropy_loss = -(self.log_entropy * (logprob.detach() - 1.)).mean()
        self.entropy_opt.zero_grad()
        entropy_loss.backward()
        self.entropy_opt.step()
        with torch.no_grad():
            self.dual = (self.dual + 1e-4 * costs.mean(0)).clamp_min(0)
            for source, target in [(self.reward, self.target_reward), (self.cost, self.target_cost)]:
                for p, q in zip(source.parameters(), target.parameters()):
                    q.lerp_(p, .005)
        return {'critic_loss': float(critic_loss.detach()), 'actor_loss': float(actor_loss.detach()),
                'entropy': float(self.log_entropy.exp().detach()), 'dual': self.dual.tolist()}
