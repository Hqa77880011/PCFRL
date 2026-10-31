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
        for fidelity in ['SPMe', 'DFN']:
            battery = Battery(fidelity)
            initial = battery.state().copy()
            initial_capacity = battery.scalar(battery.solution, 'Discharge capacity [A.h]')
            first = battery.preview(.1)
            second = battery.preview(.1)
            np.testing.assert_array_equal(initial, battery.state())
            assert battery.elapsed == battery.previous == battery.loss == 0.
            np.testing.assert_allclose(first[1], second[1], rtol=1e-6, atol=1e-8)
            state, reward, costs, done, info = battery.step(.1)
            assert np.isfinite(state).all() and np.isfinite(reward) and np.isfinite(costs).all()
            assert all(np.isfinite(value) for value in info.values()) and not done
            assert battery.elapsed == 1. and state[0] > initial[0]
            np.testing.assert_allclose(state[[3, 1, 6, 0]], first[1], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(battery.scalar(battery.solution, 'Current [A]'), -.1, atol=1e-12)
            capacity_change = battery.scalar(battery.solution, 'Discharge capacity [A.h]') - initial_capacity
            np.testing.assert_allclose(capacity_change, -.1 / 3600, rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(battery.soc - .1, -capacity_change / (5 * battery.soh), atol=1e-10)
            if fidelity == 'DFN':
                assert battery.simulation.model.options.negative['lithium plating'] == 'irreversible'
            else:
                # Exercise the real 41-current feasibility search without committing previews.
                committed = battery.state().copy()
                bound, failed = battery.safe_bound()
                assert not failed and 0 <= bound <= 20
                np.testing.assert_array_equal(committed, battery.state())
                _, value, _, _ = battery.preview(bound)
                assert value[0] <= 4.2 + 1e-6 and value[1] <= 45 + 1e-6
                assert value[2] >= -1e-6 and value[3] <= .8 + 1e-8
            print(f'PASS: real {fidelity} initialization, repeatable noncommitting preview, '
                  'one-second transition and electrochemical Coulomb counting', flush=True)
    print('PASS: SAC actor/critic/fuzzy gradients and updates, nonnegative dual, rule arithmetic, safety cap, checkpoint equality, ablations')


if __name__ == '__main__':
    main()
