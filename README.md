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

The PyBaMM plant uses Chen2020 electrochemical parameters, OKane2022 degradation coefficients where required, and a two-node thermal model. SPMe uses partially reversible plating during training; DFN evaluation uses irreversible plating. SOH rescales the nominal 5 Ah capacity and the Coulomb-counting denominator. This is a battery simulation without an external dataset requirement. Cell and thermal settings are in `pcfrl/plant.py`; evaluation supports SPMe and DFN.

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

## Controller and metrics

The controller combines twelve fixed-antecedent TSK rules, a SAC actor/critic and an electrochemical current projection. Evaluation reports charging success/time, constraint violations, capacity fade, efficiency and latency, with per-condition traces. `pcfrl evaluate --baseline cccv` selects constant-current/constant-voltage control; `--baseline mscc` selects multistage control.

## Windows CasADi runtime

If the CasADi installation path contains non-ASCII characters and its plugin loader reports error 126, use an ASCII directory junction with process-local `CASADIPATH`. From this repository in PowerShell:

```powershell
python -m venv --system-site-packages runs/pybamm-venv
./runs/pybamm-venv/Scripts/python.exe -m pip install "pybamm==25.12.2"
$pcfrlCasadi = Join-Path (Get-Location).Path 'runs/pybamm-venv/Lib/site-packages/casadi'
$pcfrlAlias = Join-Path $env:LOCALAPPDATA 'CodexRuntime/PCFRL/casadi-3.6.7'
New-Item -ItemType Directory -Path (Split-Path $pcfrlAlias) -Force | Out-Null
if (-not (Test-Path -LiteralPath $pcfrlAlias)) {
    New-Item -ItemType Junction -Path $pcfrlAlias -Target $pcfrlCasadi | Out-Null
} elseif ((Get-Item -LiteralPath $pcfrlAlias).Target -ne $pcfrlCasadi) {
    throw 'The runtime alias points to another installation'
}
$env:CASADIPATH = $pcfrlAlias
$env:PYBAMM_DISABLE_TELEMETRY = 'true'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
./runs/pybamm-venv/Scripts/python.exe -u smoke.py --plant
```

The isolated environment and its junction are local runtime files excluded from Git. An installation whose CasADi directory already has an ASCII-only path does not require the alias. The junction is a link to the runtime, so it must not be recursively cleaned as if it were a copy of the package.

## Complete episode and condition-matrix check

`evaluate` accepts explicit `--temperatures`, `--sohs`, `--initial-soc`,
and `--max-seconds`. Defaults are the 10/25/40 C by
0.8/0.9/1 SOH matrix, 10% initial SOC, 80% target and 1800-second limit.
Every condition produces a terminal status, trace and outcome metrics, including
failed-condition time penalties at that registered limit.

```bash
pcfrl evaluate --model outputs/full/model.pt --output outputs/condition_matrix --fidelity DFN --temperatures 10 25 40 --sohs 0.8 0.9 1 --initial-soc 0.1 --max-seconds 1800 --seed 0
python smoke.py --plant --episodes
```

`smoke.py --plant --episodes` uses a short SOC 0.7995-to-0.8 range and a 60-second limit to check solver integration, termination, condition aggregation, trace arithmetic and constraints. The SPMe check covers 25/40 C by SOH 0.8/1; the DFN check uses 25 C and SOH 1.

## Checks

```bash
python smoke.py
python smoke.py --plant
python smoke.py --plant --episodes
```

Synthetic replay checks cover SAC/fuzzy updates and checkpoint replay. Plant checks require PyBaMM and exercise actual SPMe/DFN integration.
