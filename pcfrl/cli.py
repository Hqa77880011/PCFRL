import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .method import SAC
from .plant import Battery


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    agent = SAC(args.variant)
    replay = deque(maxlen=args.capacity)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, episode, battery = [], 0, None
    for step in range(args.steps):
        if battery is None:
            battery = Battery('SPMe', float(rng.choice([10, 25, 40])), float(rng.choice([.8, .9, 1.])))
            state = battery.state()
            safe, failure = battery.safe_bound()
            if failure:
                raise RuntimeError('Initial state has no model-feasible current')
        with torch.no_grad():
            if step < args.warmup:
                requested = torch.tensor([rng.uniform(0, 20)], dtype=torch.float32)
                fuzzy = agent.fuzzy(torch.tensor(state[None]))[0]
                if args.variant in ('sac', 'lagrangian', 'no_fuzzy'):
                    fuzzy = torch.ones_like(fuzzy)
                current = requested * fuzzy
                if args.variant not in ('sac', 'lagrangian', 'no_projection'):
                    current = torch.minimum(current, torch.tensor([safe]))
                executed = float(current[0])
            else:
                executed = float(agent.action(torch.tensor(state[None]), torch.tensor([safe]))[0][0])
        try:
            next_state, reward, costs, done, info = battery.step(executed)
        except battery.pybamm.SolverError:
            history.append({'step': step, 'episode': episode, 'termination': 'solver_failure', 'requested_current_a': executed})
            episode += 1
            battery = None
            continue
        next_safe, failure = battery.safe_bound() if not done else (0., False)
        done = done or failure
        replay.append((state, executed, reward, costs, next_state, float(done), safe, next_safe))
        if len(replay) >= args.batch and step >= args.warmup:
            selected = rng.choice(len(replay), args.batch, replace=False)
            batch = tuple(np.asarray(v) for v in zip(*(replay[int(i)] for i in selected)))
            update = agent.update(batch)
            if step % 100 == 0:
                history.append({'step': step, **update})
        state, safe = next_state, next_safe
        if done:
            episode += 1
            battery = None
        if (step + 1) % args.checkpoint == 0 or step + 1 == args.steps:
            torch.save({'actor': agent.actor.state_dict(), 'fuzzy': agent.fuzzy.state_dict(),
                        'variant': args.variant, 'step': step + 1, 'seed': args.seed}, out / 'model.pt')
    (out / 'history.json').write_text(json.dumps(history, indent=2))


def evaluate(args):
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    checkpoint = torch.load(args.model, weights_only=False, map_location='cpu') if args.model else None
    agent = SAC(checkpoint['variant']) if checkpoint else None
    if agent:
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.fuzzy.load_state_dict(checkpoint['fuzzy'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for temperature in args.temperatures:
        for soh in args.sohs:
            battery = Battery(args.fidelity, temperature, soh, initial_soc=args.initial_soc)
            rows, previous_command, termination = [], 0., 'timeout'
            for step in range(args.max_seconds):
                state = battery.state()
                observed = state.copy()
                if args.noise:
                    observed += rng.normal(0, [ .01, .5, .5, .01, 0, 0, .015, 0])
                begin = time.perf_counter()
                safe, failure = battery.safe_bound()
                if failure:
                    termination = 'model_infeasible'
                    break
                if agent:
                    with torch.no_grad():
                        current, _, requested, firing = agent.action(torch.tensor(observed[None]), torch.tensor([safe]), True)
                    current, requested = float(current[0]), float(requested[0])
                    weights = firing[0].numpy()
                    rule = int(weights[1:].argmax() + 2) if weights[1:].max() > 0 else 1
                elif args.baseline == 'cccv':
                    requested = float(np.clip((4.2 - state[3]) * 100, 0, 10))
                    current, rule = min(requested, safe), 0
                else:
                    requested = 20 if state[0] < .4 else 12 if state[0] < .6 else 6
                    current, rule = min(requested, safe), 0
                latency = (time.perf_counter() - begin) * 1000
                commanded = previous_command if args.actuator_delay else current
                previous_command = current
                try:
                    next_state, reward, costs, done, info = battery.step(commanded)
                except battery.pybamm.SolverError:
                    termination = 'solver_failure'
                    break
                rows.append({'step': step, 'soc': float(next_state[0]), 'core_c': float(next_state[1]),
                             'surface_c': float(next_state[2]), 'voltage_v': float(next_state[3]),
                             'plating_margin_v': float(next_state[6]), 'requested_a': requested, 'safe_a': safe,
                             'current_a': commanded, 'rule': rule, 'reward': reward, 'violation': bool(costs.max() > 1e-8),
                             'latency_ms': latency, **info})
                if done:
                    termination = 'target' if battery.soc >= .8 - 1e-8 else 'timeout'
                    break
            name = f'T{temperature:g}_SOH{soh:g}'
            if rows:
                with open(out / f'{name}.csv', 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            violations = sum(row['violation'] for row in rows)
            success = termination == 'target' and violations == 0
            charge = sum(row['current_a'] / 3600 for row in rows)
            charger = sum(row['charger_wh'] for row in rows)
            summaries.append({'condition': name, 'termination': termination, 'success': success,
                              'charging_seconds': len(rows) if success else args.max_seconds,
                              'initial_soc': args.initial_soc, 'target_soc': .8, 'maximum_seconds': args.max_seconds,
                              'peak_core_c': max([row['core_c'] for row in rows] + [temperature]),
                              'energy_efficiency': sum(row['stored_wh'] for row in rows) / charger if charger > 0 else None,
                              'projected_fade_1000_percent': 100000 * sum(row['capacity_loss_ah'] for row in rows) / charge if charge > 0 else None,
                              'violation_rate': violations / len(rows) if rows else None,
                              'latency_median_ms': float(np.median([row['latency_ms'] for row in rows])) if rows else None,
                              'latency_p95_ms': float(np.quantile([row['latency_ms'] for row in rows], .95)) if rows else None,
                              'latency_max_ms': max([row['latency_ms'] for row in rows], default=None)})
            plt.plot([row['soc'] for row in rows], label=name)
    (out / 'metrics.json').write_text(json.dumps(summaries, indent=2))
    plt.xlabel('Control second')
    plt.ylabel('State of charge')
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out / 'charging.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['full', 'sac', 'lagrangian', 'no_fuzzy', 'no_projection', 'fixed_fuzzy', 'no_physics_penalty', 'no_monotonic'], default='full')
    for name, default in [('steps', 500000), ('warmup', 10000), ('capacity', 1000000), ('batch', 256), ('seed', 0), ('checkpoint', 10000)]:
        p.add_argument('--' + name, type=int, default=default)
    p = sub.add_parser('evaluate')
    p.add_argument('--output', required=True)
    p.add_argument('--model')
    p.add_argument('--baseline', choices=['cccv', 'mscc'], default='cccv')
    p.add_argument('--fidelity', choices=['SPMe', 'DFN'], default='DFN')
    p.add_argument('--temperatures', type=float, nargs='+', default=[10., 25., 40.])
    p.add_argument('--sohs', type=float, nargs='+', default=[.8, .9, 1.])
    p.add_argument('--initial-soc', type=float, default=.1)
    p.add_argument('--max-seconds', type=int, default=1800)
    p.add_argument('--noise', action='store_true')
    p.add_argument('--actuator-delay', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'evaluate' and (not 0 <= args.initial_soc < .8 or args.max_seconds < 1 or any(h <= 0 for h in args.sohs)):
        parser.error('initial-soc must be in [0,.8), max-seconds positive and SOH values positive')
    {'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == "__main__":
    main()
