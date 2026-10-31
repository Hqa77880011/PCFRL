# PCFRL

Implementation of **Physics-Constrained Fuzzy Reinforcement Learning for Safe and Interpretable Lithium-Ion Fast Charging**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[battery]"
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

The PyBaMM plant uses Chen2020 electrochemical parameters, OKane2022 degradation coefficients where required, and a two-node thermal model. This is a battery simulation without an external dataset requirement. Cell and thermal settings are in `pcfrl/plant.py`; evaluation supports SPMe and DFN.

## Training and inference

```bash
pcfrl train --output outputs/full --steps 500000 --seed 0
```

## Evaluation

```bash
pcfrl evaluate --model outputs/full/model.pt --output outputs/full_dfn --fidelity DFN --seed 0
```

## Ablations

Supported controls: `sac`, `lagrangian`, `no_fuzzy`, `no_projection`, `fixed_fuzzy`, `no_physics_penalty`, `no_monotonic`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
pcfrl train --output outputs/sac --steps 500000 --seed 0 --variant sac
```

## Outputs

Training writes `model.pt` and `history.json`. Evaluation writes `metrics.json` and `charging.png` under the selected output directory.

## Implementation scope and functional validation

The manuscript-based SAC, fixed-antecedent fuzzy controller and PyBaMM plant are reused. The paper's original replay buffers/checkpoints and full simulation logs are unavailable. No new training output is used as evidence for the manuscript's reported performance.

| Paper module / metric | Entry | Command / output |
| --- | --- | --- |
| Twelve TSK rules, actor and constrained SAC | `pcfrl/method.py` | `pcfrl train ... --variant full` |
| Electrochemical preview, thermal state, current projection | `pcfrl/plant.py` | real SPMe training / DFN evaluation |
| Success/time, violation/fade/efficiency/latency | `pcfrl/cli.py:evaluate` | per-condition CSV and `metrics.json` |
| CCCV / multistage controls | `evaluate` | `pcfrl evaluate --baseline cccv --output outputs/cccv --fidelity DFN` (use `mscc` for multistage) |
| Functional validation | `smoke.py` | `python smoke.py`; optional `python smoke.py --plant` |

`python smoke.py` passed on Python 3.13.5 / torch 2.7.1+cu118 CPU, seed 0 and one thread: real SAC update with finite actor/critic losses, nonzero fuzzy-consequent gradient/update, actor update, nonnegative duals, hand-checked rule weights, current cap, checkpoint prediction equality and SAC/fixed-fuzzy/no-projection controls. The replay batch is synthetic and makes no battery-performance claim. The local Python environment lacks optional PyBaMM; `--plant`, real SPMe/DFN transitions and all battery outcome metrics were **not run**. Install `.[battery]` in the documented environment before running that check. Parameter merging and thermal constants are implementation choices requiring validation against the intended cell protocol.
