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

Source inspection: Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
