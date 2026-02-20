# Multi-Example OQ-to-T-Risk Verification Log

**Date:** February 19, 2026  
**OQ Engine Version:** 3.24.1  
**Platform:** macOS (ARM64)

---

## Objective

Run three OpenQuake risk demo examples end-to-end, translate inputs/outputs via `library_oq_import.py`, recompute losses using the T-Risk mean-curve methodology, and compare against OQ reference outputs.

---

## Steps Performed

### 1. Script Review

Reviewed `run_multi_example_verification.py` and `library_oq_import.py`:

- **Pipeline:** Runs OQ → parses exports (GMF, events, sitemesh, vulnerability XML, exposure CSV) → maps assets to sites → builds hazard matrices → computes T-Risk losses via mean vulnerability curve interpolation → compares with OQ outputs.
- **Asset-to-site mapping:** Uses coordinate join with nearest-neighbour fallback.
- **Loss computation:** Deterministic mean-curve interpolation (`np.interp`) per asset/event, aggregated with uniform event rates (λ = 1/T).
- **Metrics:** Portfolio ratio, event-level correlation/R², MAE/RMSE, asset-level ratio statistics.
- **Conclusion:** Script is structurally correct. No code changes were required.

### 2. Examples Executed

| Example | Calc ID | Assets | Events | Job File |
|---|---|---|---|---|
| ScenarioRisk | 40 | 9,063 | 100 | `job.ini` |
| EventBasedRisk | 41 | 9,063 | 17,035 | `job.ini` |
| Reinsurance | 42 | 9,063 | 17,020 | `job.ini` |

All three ran successfully with no errors.

### 3. Verification Results

| Metric | ScenarioRisk | EventBasedRisk | Reinsurance |
|---|---|---|---|
| Portfolio ratio (T-Risk/OQ) | 1.0509 | 1.0587 | 1.0601 |
| Event correlation (ρ) | 0.9989 | 0.9998 | 0.9998 |
| Event R² | 0.9556 | 0.9992 | 0.9991 |
| Event MAE | 8.92×10⁸ | 1.80×10⁷ | 1.79×10⁷ |
| Event RMSE | 9.53×10⁸ | 2.15×10⁷ | 2.11×10⁷ |
| Asset ratio min | 0.913 | 0.747 | 0.704 |
| Asset ratio mean | 1.001 | 1.063 | 1.070 |
| Asset ratio max | 1.097 | 1.870 | 1.697 |

### 4. Discrepancy Analysis

The ~5–6% positive bias of T-Risk relative to OQ is attributed to:

1. **Deterministic vs. stochastic vulnerability** — T-Risk uses the mean curve directly; OQ samples from the beta distribution (bounded at [0,1]), producing slightly lower expected losses.
2. **Minimum-intensity filtering** — OQ suppresses ground motions below `minimum_intensity = 0.05g`; T-Risk processes all events including sub-threshold ones.
3. **Vulnerability curve clamping** — Curves start at IML = 0.0001 with mean LR = 0.0001; T-Risk clamps (rather than extrapolates to zero) at sub-threshold IMLs, adding a small cumulative bias.

### 5. LaTeX Report

Wrote and compiled a detailed verification report:

- **File:** `oq_multi_example_verification_report.tex`
- **PDF:** `oq_multi_example_verification_report.pdf` (676 KB, ~10 pages)
- **Contents:**
  - Introduction and objectives
  - Methodology (pipeline, loss formulation, error metrics)
  - Per-example descriptions (parameters, calculation mode, vulnerability components)
  - Results tables (summary + detailed metrics)
  - 6 figures (asset bar charts + event scatter plots for each example)
  - Discussion of discrepancy sources
  - Conclusions
  - Appendices (software versions, file manifest)

---

## Output Files

```
outputs/
├── multi_example_summary.csv
├── ScenarioRisk/
│   ├── summary.csv
│   ├── asset_comparison.csv
│   └── event_comparison.csv
├── EventBasedRisk/
│   ├── summary.csv
│   ├── asset_comparison.csv
│   └── event_comparison.csv
└── Reinsurance/
    ├── summary.csv
    ├── asset_comparison.csv
    └── event_comparison.csv

plots/
├── ScenarioRisk_asset_compare.png
├── ScenarioRisk_event_scatter.png
├── EventBasedRisk_asset_compare.png
├── EventBasedRisk_event_scatter.png
├── Reinsurance_asset_compare.png
└── Reinsurance_event_scatter.png

oq_multi_example_verification_report.tex
oq_multi_example_verification_report.pdf
```
