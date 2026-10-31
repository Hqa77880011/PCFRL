"""One SAC update; --plant additionally requires the real optional PyBaMM dependency."""
import argparse
import io
import json

import numpy as np
import torch

from pcfrl.method import SAC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plant', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = SAC()
    state = np.tile([.3, 30., 28., 3.8, 5., .9, .05, .2], (8, 1)).astype(np.float32)
    state[:, 1] += rng.uniform(-3, 3, 8)
    state[:, 3] += rng.uniform(-.05, .05, 8)
    next_state = state.copy()
    next_state[:, 0] += .001
    safe = np.full(8, 18., dtype=np.float32)
    batch = (state, np.full(8, 5.), -np.ones(8), np.full((8, 5), .01), next_state,
             np.zeros(8), safe, safe)
    before = {name: parameter.detach().clone() for name, parameter in agent.actor.named_parameters()}
    beta = agent.fuzzy.beta.detach().clone()
    results = agent.update(batch)
    assert all(np.isfinite(value) for name, value in results.items() if name != 'dual')
    assert any(not torch.equal(before[name], p) for name, p in agent.actor.named_parameters())
    assert agent.fuzzy.beta.grad is not None and agent.fuzzy.beta.grad.abs().sum() > 0
    assert not torch.equal(beta, agent.fuzzy.beta)
    assert agent.dual.min() > 0
    tensor = torch.tensor(state)
    current = agent.action(tensor, torch.full((8,), 2.), True)[0]
    assert bool((current <= 2.).all()) and bool((current >= 0).all())
    buffer = io.BytesIO()
    torch.save({'actor': agent.actor.state_dict(), 'fuzzy': agent.fuzzy.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = SAC()
    restored.actor.load_state_dict(saved['actor'])
    restored.fuzzy.load_state_dict(saved['fuzzy'])
    torch.testing.assert_close(current, restored.action(tensor, torch.full((8,), 2.), True)[0], rtol=0, atol=0)
    # The default rule has unit firing, and all-high headroom activates R12.
    high = torch.tensor([[.3, 25., 25., 3.5, 0., 1., .1, 0.]])
    multiplier, firing = agent.fuzzy(high)
    assert abs(float(firing[0, 0]) - .5) < 1e-7 and abs(float(firing[0, 11]) - .5) < 1e-7
    for variant in ['sac', 'fixed_fuzzy', 'no_projection']:
        control = SAC(variant)
        update = control.update(batch)
        assert np.isfinite(update['actor_loss'])
        if variant == 'fixed_fuzzy':
            np.testing.assert_allclose(control.fuzzy.beta.detach(), SAC().fuzzy.beta.detach())
    if args.plant:
        from pcfrl.plant import Battery
        battery = Battery('SPMe')
        initial = battery.state().copy()
        preview = battery.preview(.1)
        np.testing.assert_array_equal(initial, battery.state())
        state, reward, costs, done, info = battery.step(.1)
        assert np.isfinite(state).all() and np.isfinite(reward) and np.isfinite(costs).all()
        assert battery.elapsed == 1. and state[0] > initial[0]
        print('PASS: real SPMe initialization/preview/one transition')
    print('PASS: SAC actor/critic/fuzzy gradients and updates, nonnegative dual, rule arithmetic, safety cap, checkpoint equality, ablations')


if __name__ == '__main__':
    main()
