# Deterministic vs Stochastic Vulnerability Verification

## Overview

This verification compares risk outputs from the **OpenQuake Engine** (OQ) against
an independent calculation that applies **deterministic mean vulnerability curves**
to the same hazard inputs (ground motion fields, events, site mesh). The goal is
to quantify the difference between OQ's stochastic vulnerability treatment and a
deterministic mean-curve approach.

Three OQ risk demos from `demos/risk/` are used as test cases:

| Demo | Calculation mode | Description |
|---|---|---|
| **ScenarioRisk** | `scenario_risk` | Single rupture, 100 stochastic ground motion fields (Nepal) |
| **EventBasedRisk** | `event_based_risk` | Full stochastic event set with source model + logic tree (Nepal) |
| **Reinsurance** | `event_based_risk` | Same as EventBasedRisk with reinsurance layer on top |

Each demo uses both **structural** and **nonstructural** vulnerability components.

### Why a small difference is expected

The deterministic approach interpolates the **mean loss-ratio curve** directly at
each ground motion intensity value. OQ, by contrast, samples from a **lognormal
distribution** parameterised by (mean, CoV) at each intensity level and integrates
the full probability distribution.

This difference produces a systematic ~5–6% higher portfolio loss in the
deterministic calculation relative to OQ. The effect arises because OQ's
lognormal distribution, when truncated at loss ratios of 0 and 1, yields a lower
effective expected loss than the nominal mean curve — especially at high
intensities where the mean loss ratio approaches 1.0 and significant probability
mass falls above the truncation boundary.


## Test platform

All timings reported in this document were measured on an **Apple M4 Pro** processor
with **14 CPU cores** and **20 GPU cores** (Apple MacBook Pro, ARM64).

## Prerequisites

### 1. Clone the repository

```bash
git clone https://github.com/gem/oq-engine.git
cd oq-engine
```

The `oq_verification/` folder is already included at the repository root.

### 2. OpenQuake Engine environment

The OQ engine must be installed in a dedicated virtual environment. Follow the
official installation instructions at
[github.com/gem/oq-engine](https://github.com/gem/oq-engine).

The recommended method uses the bundled installer:

```bash
python3 install.py user
```

This creates a virtual environment with all OQ dependencies (numpy, scipy,
pandas, h5py, numba, matplotlib, etc.). Alternatively, for development:

```bash
python3 -m venv <venv-path>
source <venv-path>/bin/activate
pip install -e .
```

| Component | Version tested | Notes |
|---|---|---|
| **OpenQuake Engine** | 3.24.1 | Installed via `install.py` or `pip install -e .` |
| **Python** | 3.10+ | Python 3.10–3.12 recommended |

### 3. T-Risk and its dependencies

The verification compares OQ's stochastic vulnerability against T-Risk's
deterministic mean-curve approach. While the verification script itself runs
entirely within the OQ environment, a working T-Risk installation is required
to understand and extend the comparison.

T-Risk dependencies:

| Package | Notes |
|---|---|
| **numpy** | Already included in the OQ environment |
| **tensorflow** | Required by T-Risk's tensor engine |
| **matplotlib** | Already included in the OQ environment |

Install T-Risk's additional dependencies in a separate virtual environment:

```bash
python3 -m venv <trisk-venv-path>
source <trisk-venv-path>/bin/activate
pip install numpy tensorflow matplotlib
```

> **Note:** The `run_verification.py` script only requires the OQ environment
> to execute. The T-Risk environment is needed for running the T-Risk tensor
> engine directly.

### 4. Directory layout

After cloning, the relevant structure is:

```
oq-engine/
├── demos/risk/
│   ├── ScenarioRisk/
│   ├── EventBasedRisk/
│   └── Reinsurance/
├── oq_verification/              ← this folder
│   ├── run_verification.py       # Accuracy comparison
│   ├── run_gradient_verification.py  # Gradient AD vs FD verification
│   ├── run_benchmark_oq.py       # Benchmark Phase 1 (OQ env)
│   ├── run_benchmark_trisk.py    # Benchmark Phase 2 (T-Risk env)
│   ├── run_benchmark_batched.py  # Batched benchmark with .npz caching
│   ├── run_benchmark.sh          # Shell wrapper (both phases)
│   ├── library_oq_import.py
│   └── README.md
└── ...
```

The script resolves paths relative to its own location:
`OQ_ROOT = Path(__file__).resolve().parent.parent` (the oq-engine root).


## How to replicate

### Step 1: Clone the repository (if not already done)

```bash
git clone https://github.com/gem/oq-engine.git
cd oq-engine
```

### Step 2: Activate the OpenQuake environment

```bash
source <path-to-oq-virtualenv>/bin/activate
```

Verify the installation:

```bash
oq --version
# Expected: 3.24.1 (or compatible version)

python -c "from openquake.baselib import __version__; print(__version__)"
```

### Step 3: Navigate to the verification directory

```bash
cd oq_verification
```

### Step 4 (optional): Clean previous outputs

```bash
rm -rf outputs plots
```

### Step 5: Run the verification script

```bash
python run_verification.py
```

The script will:

1. **Run each OQ demo** — executes `oq run job.ini -e csv` in each demo directory,
   which performs the full OQ calculation and exports all results to CSV.
2. **Copy OQ outputs** — copies the exported CSVs into `outputs/<DemoName>/`.
3. **Compute deterministic losses** — for each demo:
   - Loads the exported ground motion fields (GMF), events, site mesh, and
     vulnerability XML files.
   - Maps exposure assets to OQ site IDs.
   - Builds a hazard intensity matrix (sites × events).
   - Applies deterministic mean loss-ratio curves from the vulnerability
     functions to compute per-asset, per-event losses.
   - Computes portfolio-level annualised losses using uniform event rates.
4. **Compare results** — computes event-level and asset-level comparisons,
   including MAE, RMSE, correlation, R², and portfolio ratio.
5. **Generate plots** — produces asset bar charts and event scatter plots in `plots/`.
6. **Generate a LaTeX report** — writes a `.tex` file (compilation optional).

**Expected runtime:** A few minutes total for all three demos.

### Step 6: Inspect the results

The primary summary file is:

```bash
cat outputs/multi_example_summary.csv
```

Key columns to check:
- `portfolio_ratio` — deterministic portfolio loss / OQ portfolio loss (expect ~1.05–1.06)
- `event_corr` — Pearson correlation of event losses (expect > 0.998)
- `event_r2` — R² of event losses (expect > 0.95)

Per-demo detailed outputs are in `outputs/<DemoName>/`:
- `summary.csv` — single-row summary for that demo
- `asset_comparison.csv` — per-asset loss comparison
- `event_comparison.csv` — per-event loss comparison

Plots are in `plots/`:
- `<DemoName>_asset_compare.png` — bar chart of top-40 assets by OQ loss
- `<DemoName>_event_scatter.png` — log-log scatter of event losses


## Expected results

Results from a verified run (April 2, 2026, OQ 3.24.1, macOS ARM64):

| Example | Portfolio ratio | Event correlation | Event R² | Assets | Events |
|---|---|---|---|---|---|
| ScenarioRisk | 1.0509 | 0.9989 | 0.9556 | 9,063 | 100 |
| EventBasedRisk | 1.0586 | 0.9998 | 0.9992 | 9,063 | 17,035 |
| Reinsurance | 1.0601 | 0.9998 | 0.9991 | 9,063 | 17,020 |

### Interpretation

- **Portfolio ratio ~1.05:** The deterministic approach computes ~5% higher losses
  than OQ. This is the expected consequence of using deterministic mean
  loss-ratio curves vs. lognormal-sampled loss ratios with truncation at [0, 1].
- **Correlation > 0.998:** Event-by-event losses are extremely well correlated,
  confirming both approaches process the same hazard inputs consistently.
- **R² > 0.95:** The variance in OQ event losses is well explained by the
  deterministic calculation, with the lower R² for ScenarioRisk (0.96 vs 0.999)
  due to having only 100 events vs ~17,000 for the event-based demos.
- **Asset ratio range:** Individual asset ratios vary (0.70–1.87 at extremes)
  due to asset-specific vulnerability curve shapes and ground motion levels, but
  the portfolio-level mean is stable at ~1.05–1.07.


## Output directory structure

After a successful run:

```
oq_verification/
├── outputs/
│   ├── multi_example_summary.csv          # Cross-demo summary (3 rows)
│   ├── ScenarioRisk/
│   │   ├── summary.csv                    # Per-demo summary
│   │   ├── asset_comparison.csv           # Asset-level comparison
│   │   ├── event_comparison.csv           # Event-level comparison
│   │   └── *_<calc_id>.csv                # Raw OQ exports
│   ├── EventBasedRisk/
│   │   └── ...
│   └── Reinsurance/
│       └── ...
├── plots/
│   ├── ScenarioRisk_asset_compare.png
│   ├── ScenarioRisk_event_scatter.png
│   ├── EventBasedRisk_asset_compare.png
│   ├── EventBasedRisk_event_scatter.png
│   ├── Reinsurance_asset_compare.png
│   └── Reinsurance_event_scatter.png
└── verification_report.tex                # LaTeX report (compile with pdflatex)
```

> **Note:** The `calc_id` in OQ export filenames increments with each run and
> will differ between runs.


## Runtime Benchmark: OpenQuake vs T-Risk

In addition to accuracy verification, this suite includes a **runtime benchmark**
comparing OpenQuake's risk computation time against T-Risk's tensor-based engine.

### Benchmark design

The benchmark isolates **risk-only** computation time for a fair comparison:

- **OpenQuake:** Hazard and risk are run separately using the `--hc` flag.
  Only the risk calculation wall-clock time is measured.
- **T-Risk:** Loads the same hazard CSV exports, builds tensorial arrays, and
  runs `TensorialRiskEngine.compute_loss_and_metrics()`. Both data loading and
  computation are timed separately.

Both engines start from the same hazard outputs, so the comparison is apple-to-apple.

### Demos benchmarked

| Demo | Mode | Notes |
|---|---|---|
| ScenarioRisk | `scenario_risk` | 100 GMFs, combined hazard+risk (cannot split) |
| EventBasedRisk | `event_based_risk` | ~17k events, hazard/risk split via `--hc` |
| Reinsurance | `event_based_risk` | Same as EventBasedRisk + reinsurance layer |
| EventBasedRisk_scaled | `event_based_risk` | 10× events (`ses_per_logic_tree_path=10`) |

### How to run the benchmark

There are two approaches:

#### Option A: Automated (local, requires both environments)

```bash
cd oq_verification
./run_benchmark.sh
```

This activates each virtual environment in turn and runs both phases.
Edit `run_benchmark.sh` to adjust the virtualenv paths if needed.

#### Option B: Manual (two separate steps)

**Phase 1** — OQ timing (in the OQ environment):

```bash
source <path-to-oq-env>/bin/activate
cd oq_verification
python run_benchmark_oq.py
```

This runs each demo, measures OQ risk-only time, exports CSVs and timing
data to `benchmark_outputs/`.

**Phase 2** — T-Risk timing (in the T-Risk environment):

```bash
source <path-to-trisk-env>/bin/activate
cd oq_verification
python run_benchmark_trisk.py
```

This reads the OQ exports, runs the TensorialRiskEngine, and produces the
final comparison summary and plots.

### Benchmark outputs

```
oq_verification/
├── benchmark_outputs/
│   ├── oq_timing.json                    # OQ timing data (Phase 1 → Phase 2)
│   ├── benchmark_summary.csv             # Final comparison table
│   ├── ScenarioRisk/
│   │   ├── *_<calc_id>.csv               # OQ CSV exports
│   │   ├── oq_performance_risk.txt       # oq show performance output
│   │   └── oq_performance_hazard.txt
│   ├── EventBasedRisk/
│   ├── EventBasedRisk_scaled/
│   └── Reinsurance/
├── benchmark_plots/
│   ├── benchmark_comparison.png          # Side-by-side bar chart with speedup
│   ├── benchmark_breakdown.png           # T-Risk load vs compute stacked bars
│   └── benchmark_scaling.png             # Events vs time (scaling behavior)
```

### Reading the results

The key output is `benchmark_outputs/benchmark_summary.csv` with columns:

| Column | Description |
|---|---|
| `name` | Demo name |
| `n_assets` | Number of exposure assets |
| `n_events` | Number of hazard events |
| `oq_risk_time_sec` | OQ risk-only wall-clock time |
| `trisk_load_time_sec` | T-Risk CSV loading + array building time |
| `trisk_compute_time_sec` | T-Risk `compute_loss_and_metrics()` time |
| `trisk_total_time_sec` | T-Risk total (load + compute) |
| `speedup` | OQ time / T-Risk time |


## Gradient Verification: T-Risk AD vs Finite Differences

This suite verifies T-Risk's automatic differentiation (AD) gradients against
independent central finite differences (FD) and analytical baselines.

### Gradients verified

| Gradient | Dimensions | Verification method | Max error |
|---|---|---|---|
| **Vulnerability** `∂AAL/∂C` | K×M = 6×8 = 48 params | Central FD (δ=1e-4) | 0.017% |
| **Exposure** `∂AAL/∂v` | N = 9,063 params | Analytical + FD (200 samples) | 0.00002% |

### How to run

In the T-Risk virtual environment:

```bash
source <path-to-trisk-env>/bin/activate
cd oq_verification
python run_gradient_verification.py
```

**Prerequisite:** `benchmark_outputs/ScenarioRisk/` must exist (from `run_benchmark_oq.py`).

### What it does

**Part A — Vulnerability gradient `∂AAL/∂C`:**
1. Loads ScenarioRisk demo data (9,063 assets, 100 events, 6 typologies × 8 IML points)
2. Computes T-Risk AD gradient via `gradient_wrt_vulnerability()` (single tape pass)
3. Computes central FD gradient using T-Risk-native numpy interpolation (96 perturbations)
4. Computes OQ-style FD gradient for cross-engine comparison
5. Runs convergence sweep (δ = 1e-3, 1e-4, 1e-5)

**Part B — Exposure gradient `∂AAL/∂v`:**
6. Computes T-Risk AD gradient via `gradient_wrt_exposure()` (single tape pass)
7. Computes analytical gradient: `∂AAL/∂v_i = Σ_q λ_q × MDR[i,q]` (exact for linear v)
8. Computes central FD on 200 random assets with relative perturbation (δ_rel=1e-6)
9. Reports per-typology exposure gradient statistics

**Part C — Cross-engine OQ exposure gradient (`run_gradient_oq.py`):**
10. Runs full OQ ScenarioRisk programmatically for baseline AAL
11. Perturbs 50 random assets in the exposure CSV (δ_rel=1%)
12. Computes central FD through the complete OQ pipeline (101 OQ runs)
13. Compares OQ FD gradients with T-Risk AD gradients

Run in the **OQ virtual environment**:
```bash
source <path-to-oq-env>/bin/activate
cd oq_verification
python run_gradient_oq.py
```

### Outputs

```
oq_verification/
├── gradient_verification_result.json     # Full results with gradient arrays
├── gradient_oq_exposure_result.json      # OQ FD exposure gradient (50 assets)
├── benchmark_plots/
│   ├── gradient_heatmaps.png             # AD vs FD side-by-side heatmaps
│   ├── gradient_scatter.png              # FD vs AD scatter plot
│   ├── gradient_scatter_oq.png           # OQ FD vs AD scatter (boundary divergence)
│   ├── gradient_convergence.png          # FD error vs δ
│   └── exposure_gradient.png             # AD vs analytical, AD vs FD, by-typology
```

### Expected results

**Vulnerability gradient:**

| Metric | Value |
|---|---|
| Max relative error (AD vs FD) | **0.017%** |
| Mean relative error | 0.012% |
| AD time | ~0.4s |
| FD time (96 evaluations) | ~2.1s |
| AD speedup | 5.8× |
| Verdict | **PASS** (threshold: < 1%) |

**Exposure gradient:**

| Metric | Value |
|---|---|
| Max error (AD vs analytical, N=9,063) | **0.00002%** |
| Max error (AD vs FD, sample=200) | 0.26% |
| AD time | ~0.02s |
| FD time (400 evaluations) | ~8.6s |
| Projected AD speedup (full N) | ~10,000× |
| Verdict | **PASS** |

**OQ cross-engine exposure gradient:**

| Metric | Value |
|---|---|
| OQ FD time per asset | 6.5s |
| Total FD time (50 assets) | 5.4 min |
| Projected full N (9,063) | **16.4 hours** |
| T-Risk AD time (all 9,063) | **0.02s** |
| Cross-engine speedup | ~3,000,000× |
| Non-zero OQ FD gradients | 24/50 (48%) |
| OQ FD precision issue | 52% zeros from float32 noise |

**Cross-engine note:** When comparing against OQ-style interpolation (which clips
GMVs to max IML instead of extrapolating), 10 boundary elements diverge by up to
56%. This is a genuine semantic difference, not an error — see the LaTeX report
for details.


## Troubleshooting

| Problem | Solution |
|---|---|
| `oq: command not found` | Activate the OQ virtual environment first |
| `ModuleNotFoundError: No module named 'library_oq_import'` | Run from the `oq_verification/` directory so Python finds the local module |
| `RuntimeError: OQ run failed` | Check OQ installation: `oq run --help`. Inspect the full stderr in the error message. |
| `FileNotFoundError: Missing export` | OQ may not have exported CSV files. Re-run with `oq run job.ini -e csv` manually in the demo directory. |
| Portfolio ratio far from 1.0 | Ratios of 1.05–1.06 are expected. Values outside 0.90–1.15 suggest a data or configuration issue. |
| `benchmark_outputs/ not found` | Run `run_benchmark_oq.py` first (Phase 1) before `run_benchmark_trisk.py` (Phase 2) |
| `T-Risk directory not found` | Ensure `T-Hazard/T-Risk/` is alongside `oq-engine/` in the workspace |
| `ModuleNotFoundError: tensorflow` | Activate the T-Risk virtual environment with TensorFlow installed |
