# PCFRL

Implementation of **Physics-Constrained Fuzzy Reinforcement Learning for Safe and Interpretable Lithium-Ion Fast Charging**.

The controller combines a squashed-Gaussian SAC actor, the paper's twelve fixed-antecedent TSK rules with learned consequents, and a one-step current projection. Twin reward critics, five cost critics, entropy adaptation, nonnegative dual variables, projection-distance loss, and an ordered-headroom regularizer support learning.

## Install

```bash
python -m venv .venv
python -m pip install -e .
python -m pip install "pybamm>=25.4,<26"
```

Use Python 3.11 and activate the environment. PyBaMM supplies the SPMe/DFN electrochemical models; no measured cell dataset or model weights are bundled. Its [Simulation API](https://docs.pybamm.org/en/stable/source/api/simulation.html) provides state-preserving `step` calls. Charging current is positive in controller logs and converted to negative current at the PyBaMM boundary.

The plant begins with the Chen2020 NMC811/graphite parameter values. Missing degradation coefficients are supplied by OKane2022 before Chen2020 values overwrite shared keys. Isothermal electrochemistry is coupled explicitly to a two-node thermal model: core/surface capacitances 60/20 J K⁻¹ and core-surface/surface-ambient resistances 1.2/3 K W⁻¹. Core temperature supplies the electrochemical temperature at each one-second step. These thermal and degradation coefficients are implementation conventions and should be replaced with calibrated cell values for a device-specific study. SOH scales nominal capacity; the code does not claim a full aged-electrode parameter identification.

## Training and controls

```bash
pcfrl train --output outputs/full --steps 500000 --seed 0
pcfrl train --output outputs/sac --variant sac --seed 0
pcfrl train --output outputs/lagrangian --variant lagrangian --seed 0
pcfrl train --output outputs/no_fuzzy --variant no_fuzzy --seed 0
pcfrl train --output outputs/no_projection --variant no_projection --seed 0
pcfrl train --output outputs/fixed_fuzzy --variant fixed_fuzzy --seed 0
pcfrl train --output outputs/no_physics_penalty --variant no_physics_penalty --seed 0
pcfrl train --output outputs/no_monotonic --variant no_monotonic --seed 0
```

Training uses SPMe episodes over initial temperatures 10/25/40 °C and SOH 0.8/0.9/1.0, with SOC 0.1 to 0.8 and an 1800-second limit. Default networks are 256–256, Adam 3e-4, discount 0.995, Polyak coefficient 0.005, replay capacity one million, batch 256, and warm-up 10000. Zero cumulative safety budgets drive the dual update. Reward normalization uses one second, one mAh capacity loss, one Wh dissipation, and full-scale current slew. Weights are 1, 8, 0.5, 0.05, and a goal bonus of 120.

`sac` removes fuzzy processing, projection, and dual penalties. `lagrangian` retains cost critics and dual penalties but removes the fuzzy layer and shield. Other variants remove one named full-controller component. All variants share the actor/critic architecture. Solver failures are logged and reset the episode without inserting an invented physical transition into replay. Each run writes a periodic `model.pt` and sampled optimization history. Checkpoint frequency is `--checkpoint`; this is a fixed-budget final checkpoint, so use a separate declared validation campaign when choosing among checkpoints.

Projection evaluates 41 currents from zero to 20 A, keeps the feasible interval connected to zero, and uses twelve bisection steps when voltage, temperature, and plating constraints are monotone. Feasibility also prevents SOC overshoot. Every preview starts from the same stored physical state. Zero-current infeasibility and solver errors remain explicit failures. This direct implementation prioritizes the one-step physical definition over computational acceleration; the projection can dominate runtime.

## Blind evaluation and visualization

```bash
pcfrl evaluate --model outputs/full/model.pt --output outputs/full_dfn --fidelity DFN
pcfrl evaluate --output outputs/cccv --baseline cccv --fidelity DFN
pcfrl evaluate --output outputs/mscc --baseline mscc --fidelity DFN
pcfrl evaluate --model outputs/full/model.pt --output outputs/noise --noise
pcfrl evaluate --model outputs/full/model.pt --output outputs/delay --actuator-delay
```

CCCV uses a 10 A request tapered by voltage error; MSCC uses 20/12/6 A at SOC transitions 0.4/0.6. Both use the same physical projection. These are declared shielded engineering controls. A health-aware NMPC solver is not represented by either controller.

Nine condition CSVs retain requested/executed/safe current, dominant nondefault rule, SOC, voltage, core/surface temperature, plating margin, reward, safety event, capacity loss, and energy. `metrics.json` reports charging time, success, peak core temperature, energy efficiency, projected fade per 1000 equivalent full cycles, violation rate, and CPU decision median/p95/maximum latency. Failures retain 1800 seconds in charging-time summaries. `charging.png` overlays SOC trajectories.

Energy efficiency here uses terminal input energy minus modeled irreversible heat as the stored-energy estimate; it is not a separate equilibrium free-energy calculation. Projected fade is `1000 × lost capacity / charged throughput × 100%`, a linearized model projection. Latency includes the safety-model previews; episode rollout is outside the decision timer. Noise standard deviations are SOC 0.01, temperature 0.5 °C, voltage 10 mV, and estimated plating margin 15 mV. Actuator delay applies the previous current for one second. Run five seeds and aggregate matched seed/condition records before inference about performance. No training results are prefilled.
