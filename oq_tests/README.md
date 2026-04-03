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
├── oq_verification/          ← this folder
│   ├── run_verification.py
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


## Troubleshooting

| Problem | Solution |
|---|---|
| `oq: command not found` | Activate the OQ virtual environment first |
| `ModuleNotFoundError: No module named 'library_oq_import'` | Run from the `oq_verification/` directory so Python finds the local module |
| `RuntimeError: OQ run failed` | Check OQ installation: `oq run --help`. Inspect the full stderr in the error message. |
| `FileNotFoundError: Missing export` | OQ may not have exported CSV files. Re-run with `oq run job.ini -e csv` manually in the demo directory. |
| Portfolio ratio far from 1.0 | Ratios of 1.05–1.06 are expected. Values outside 0.90–1.15 suggest a data or configuration issue. |
