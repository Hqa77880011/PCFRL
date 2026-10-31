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

The PyBaMM plant uses Chen2020 electrochemical parameters, OKane2022 degradation coefficients where required, and a two-node thermal model. SPMe uses partially reversible plating during training; blind DFN evaluation uses irreversible plating as specified by the manuscript protocol. SOH rescales the nominal 5 Ah capacity and the Coulomb-counting denominator. This is a battery simulation without an external dataset requirement. Cell and thermal settings are in `pcfrl/plant.py`; evaluation supports SPMe and DFN.

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

`python smoke.py` passed on Python 3.13.5 / torch 2.7.1+cu118 CPU, seed 0 and one thread: real SAC update with finite actor/critic losses, nonzero fuzzy-consequent gradient/update, actor update, nonnegative duals, hand-checked rule weights, current cap, checkpoint prediction equality and SAC/fixed-fuzzy/no-projection controls. The replay batch is synthetic and makes no battery-performance claim. `python smoke.py --plant` also passed in a dedicated system-site-packages environment with PyBaMM 25.12.2, CasADi 3.6.7 and pybammsolvers 0.3.3, retaining NumPy 2.1.3 and the above CPU/Torch configuration. It initializes real SPMe and DFN models, checks repeatable previews without committing their state, executes a 0.1 A charging second, verifies the native negative-current convention and solver-integrated discharge-capacity change (-0.1/3600 Ah), and checks the DFN negative-electrode irreversible-plating option. SPMe additionally executes the real 41-current safe-bound search and checks its returned current against all one-step constraints. CI executes the same `--plant` check. Full charging episodes, condition-matrix outcome metrics and full policy training were not run. Parameter merging and thermal constants remain explicit implementation choices.

## Windows runtime used for the plant check

On the tested Windows machine, CasADi's plugin loader failed with error 126 when its DLL directory contained non-ASCII path components. Loading the same DLL through Windows' Unicode API succeeded. An ASCII directory junction plus process-local `CASADIPATH` resolved plugin loading; the global environment was not changed. From this repository in PowerShell, the complete local setup and check are:

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

The isolated environment and its junction are retained as a local runtime and are excluded from Git. An installation whose CasADi directory already has an ASCII-only path does not require the alias. The junction is a link to the runtime, so it must not be recursively cleaned as if it were a copy of the package.
