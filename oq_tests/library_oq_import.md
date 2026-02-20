# library_oq_import.py Documentation

## Purpose

`library_oq_import.py` is a small interoperability layer that standardizes how OpenQuake (OQ) CSV inputs/outputs are loaded and transformed for T-Risk workflows.

It solves three practical problems:

1. **OQ export naming variability** (e.g., `gmf-data_16.csv`, `gmf_data.csv`, `events.csv`).
2. **CSV format variability** (comment-prefixed metadata lines and slightly different column names).
3. **Direct translation to T-Risk arrays** (hazard matrix shape and uniform event-rate vector).

---

## Location

- [T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.py](T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.py)

Companion documentation file:
- [T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.md](T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.md)

---

## Dependencies

- Python 3.10+
- `pandas`
- `numpy`

No OpenQuake Python API import is required; this module consumes OQ CSV exports directly.

---

## Data Contract (Expected Inputs)

### OQ outputs directory (`oq_dir`)

The loader functions discover files via pattern matching (latest lexicographically sorted match):

- GMF data: `gmf-data_*.csv`, `gmf-data.csv`, `gmf_data_*.csv`, `gmf_data.csv`
- Events: `events_*.csv`, `events.csv`
- Site mesh: `sitemesh_*.csv`, `sitemesh.csv`
- Risk-by-event: `risk_by_event_*.csv`, `risk_by_event.csv`
- Average losses: `avg_losses-mean_*.csv`, `avg_losses-stats_*.csv`, `avg_losses-rlzs_*.csv`, plus non-suffixed variants
- Aggregate risk: `aggrisk-*_*.csv`, `aggrisk-*.csv`, `aggrisk_*.csv`, `aggrisk.csv`

### OQ inputs directory (`oq_inputs_dir`)

- Exposure model: `exposure_model.csv` (or custom filename)
- Vulnerability model (XML): `vulnerability_model.xml` (or matching `*vulnerability*.xml`)

### Required columns by operation

- Site-id inference: one of `custom_site_id` or `sid`
- Event-id inference: one of `event_id` or `eid`
- GMV inference: one of `gmv_PGA`, `gmv`, `gmv_sa`
- Coordinate join for exposure/site mapping: `lon`, `lat` (configurable)

### Required vulnerability XML elements

- `vulnerabilityFunction` nodes with attributes like `id` and `dist`
- `imls` values (with `imt` attribute, e.g. PGA)
- `meanLRs` values
- optional `covLRs` values

---

## API Reference

## 1) Low-level file and CSV helpers

### `read_oq_csv(path: str) -> pd.DataFrame`

Reads a CSV file while:

- removing lines starting with `#` (OQ metadata comments),
- trimming whitespace in column names.

Raises:
- standard I/O exceptions if the file is inaccessible.

Use this when you already know the exact CSV path.

---

## 2) OQ file discovery and typed loaders

### `load_oq_gmf_data(oq_dir: str) -> pd.DataFrame`
Loads GMF CSV from `oq_dir` using predefined patterns.

Raises:
- `FileNotFoundError` if no GMF file matches.

### `load_oq_events(oq_dir: str) -> pd.DataFrame`
Loads events CSV.

Raises:
- `FileNotFoundError` if no events file matches.

### `load_oq_sitemesh(oq_dir: str) -> pd.DataFrame`
Loads site mesh CSV.

Raises:
- `FileNotFoundError` if no sitemesh file matches.

### `load_oq_risk_by_event(oq_dir: str, required: bool = False) -> pd.DataFrame | None`
Loads risk-by-event CSV.

Behavior:
- if found: returns DataFrame,
- if not found and `required=False`: returns `None`,
- if not found and `required=True`: raises `FileNotFoundError`.

### `load_oq_avg_losses(oq_dir: str, required: bool = True) -> pd.DataFrame | None`
Loads average-loss CSV and normalizes structural-loss mean column to `aal_usd_per_yr` when possible.

Column normalization logic:
1. Try column containing both `structural` and `mean`.
2. Fallback to first column containing `structural`.
3. If found, rename to `aal_usd_per_yr`.

Behavior for missing file is controlled by `required` like above.

### `load_oq_aggrisk(oq_dir: str, required: bool = True) -> pd.DataFrame | None`
Loads aggregate-risk CSV with `required` behavior as above.

### `load_exposure_model(oq_inputs_dir: str, filename: str = 'exposure_model.csv') -> pd.DataFrame`
Loads exposure model from OQ input directory.

Raises:
- `FileNotFoundError` when file is missing.

### `load_oq_vulnerability_xml(oq_inputs_dir: str, filename: str | None = None) -> pd.DataFrame`
Loads OQ vulnerability XML and returns a normalized long DataFrame.

Returned columns:
- `vulnerability_id`
- `distribution`
- `imt`
- `point_index`
- `iml`
- `mean_lr`
- `cov_lr` (NaN when `covLRs` is absent)

Behavior:
- if `filename` is provided, loads that exact XML (absolute or relative to `oq_inputs_dir`),
- otherwise auto-discovers by vulnerability XML patterns.

Raises:
- `FileNotFoundError` if XML is not found,
- `ValueError` if XML curves are malformed (length mismatch or empty content).

---

## 3) Schema inference utilities

### `infer_site_id_column(df: pd.DataFrame) -> str`
Returns:
- `custom_site_id` if present,
- else `sid` if present.

Raises:
- `KeyError` if neither exists.

### `infer_event_id_column(df: pd.DataFrame) -> str`
Returns:
- `event_id` if present,
- else `eid` if present.

Raises:
- `KeyError` if neither exists.

### `infer_gmv_column(df: pd.DataFrame) -> str`
Returns first match among:
- `gmv_PGA`, `gmv`, `gmv_sa`.

Raises:
- `KeyError` if none exists.

---

## 4) OQ-to-T-Risk translation functions

### `map_exposure_to_oq_sites(exposure_df, sitemesh_df, lon_col='lon', lat_col='lat', strict=True) -> pd.DataFrame`

Performs coordinate join between exposure rows and OQ site mesh:

- Uses inferred site-id column from `sitemesh_df`.
- Adds site-id to exposure rows.
- Preserves exposure cardinality via left join.

`strict=True`:
- raises `RuntimeError` if any assets fail to map to a site id.

`strict=False`:
- allows unmapped rows (site-id can remain null).

### `build_hazard_matrix_from_oq(gmf_df, event_ids, site_ids, fillna=0.0) -> np.ndarray`

Builds the hazard matrix for T-Risk from OQ GMFs with deterministic ordering.

Process:
1. Infer event/site/gmv columns.
2. Pivot GMF table to matrix indexed by event, columns by site.
3. Reindex to provided `event_ids` and `site_ids` order.
4. Fill missing values with `fillna`.
5. Return transpose as `float32`.

Output shape:
- `(N_sites_or_assets, N_events)`

Typical interpretation:
- rows aligned with exposure assets mapped to OQ sites,
- columns aligned with event catalog order.

### `compute_catalog_years(events_df: pd.DataFrame) -> float`

Returns effective catalog duration $T$:

- if `year` column exists: `max(year) + 1`,
- else fallback: number of event rows.

### `compute_uniform_event_rates(events_df: pd.DataFrame) -> np.ndarray`

Returns uniform rate vector $\lambda_q$ of length `len(events_df)`:

$$
\lambda_q = \frac{1}{T}\,,\quad q=1,\dots,Q
$$

where $T$ is from `compute_catalog_years(events_df)`.

If event table is empty, returns empty `float32` array.

### `vulnerability_to_trisk_arrays(vulnerability_df, taxonomy_order=None, require_common_imls=True) -> tuple`

Translates normalized vulnerability rows into T-Risk-compatible arrays.

Returns:
- `x_grid` (shape `(M,)`): common IML grid
- `c_matrix_mean` (shape `(K, M)`): mean LR curves by taxonomy
- `taxonomy_to_index` (dict): taxonomy → row index in `c_matrix_mean`
- `c_matrix_cov` (shape `(K, M)`): cov LR curves by taxonomy

Requirements:
- input DataFrame should come from `load_oq_vulnerability_xml` or follow the same schema,
- when `require_common_imls=True`, all taxonomies must have identical IML grids.

Raises:
- `KeyError` for missing columns/taxonomies,
- `ValueError` for inconsistent IML grids.

---

## End-to-End Usage Example

```python
import numpy as np
from library_oq_import import (
    load_exposure_model,
    load_oq_events,
    load_oq_gmf_data,
    load_oq_risk_by_event,
    load_oq_sitemesh,
    map_exposure_to_oq_sites,
    infer_site_id_column,
    build_hazard_matrix_from_oq,
    compute_catalog_years,
    compute_uniform_event_rates,
)

oq_dir = ".../oq_outputs"
oq_inputs_dir = ".../oq_inputs"

exp = load_exposure_model(oq_inputs_dir)
gmf = load_oq_gmf_data(oq_dir)
events = load_oq_events(oq_dir)
sitemesh = load_oq_sitemesh(oq_dir)
rbe = load_oq_risk_by_event(oq_dir, required=True)

asset_site = map_exposure_to_oq_sites(exp, sitemesh)
site_col = infer_site_id_column(asset_site)

asset_ids = asset_site["id"].tolist()
site_ids = asset_site[site_col].tolist()
event_ids = events["event_id"].astype(int).values

H = build_hazard_matrix_from_oq(gmf, event_ids, site_ids)  # (N_assets, N_events)
T_years = compute_catalog_years(events)
lambdas = compute_uniform_event_rates(events)

print(H.shape, T_years, lambdas.shape)
```

## Vulnerability Translation Example

```python
from library_oq_import import load_oq_vulnerability_xml, vulnerability_to_trisk_arrays

oq_inputs_dir = ".../oq_inputs"
vuln_df = load_oq_vulnerability_xml(oq_inputs_dir)

x_grid, c_mean, tax_map, c_cov = vulnerability_to_trisk_arrays(vuln_df)

print(x_grid.shape)     # (M,)
print(c_mean.shape)     # (K, M)
print(tax_map)          # e.g. {'TypeA': 0, 'TypeB': 1}
```

---

## Error Handling Guidelines

Recommended call pattern:

- Use `required=True` for inputs that are mandatory for your workflow.
- Use `required=False` only for optional comparison/plot branches.
- Keep `strict=True` in site mapping for verification and production pipelines.

Typical failures and meaning:

- `FileNotFoundError`: expected OQ export missing.
- `KeyError`: schema mismatch (column naming not recognized).
- `RuntimeError` in mapping: exposure coordinates do not match OQ sites.

---

## Behavior Notes and Assumptions

1. **File selection policy**: if multiple files match a pattern, the lexicographically last one is used.
2. **No probabilistic vulnerability sampling here**: this module only handles import/translation to hazard/rate arrays.
3. **Missing GMFs after reindex are filled** (default `0.0`), which is often appropriate for absent site-event combinations.
4. **Catalog year fallback** without `year` assumes one event per unit time by count; if your catalog semantics differ, provide your own rate vector.

---

## Current In-Repo Consumers

The following scripts currently import this module:

- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_run_trisk_on_oq.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_run_trisk_on_oq.py)
- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare_trisk_on_oq.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare_trisk_on_oq.py)
- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare.py)
- [T-Hazard/end_to_end_verification/V2_event_based_risk/v2_decompose.py](T-Hazard/end_to_end_verification/V2_event_based_risk/v2_decompose.py)
- [T-Hazard/end_to_end_verification/V2_event_based_risk/examples/example4_vulnerability_translation.py](T-Hazard/end_to_end_verification/V2_event_based_risk/examples/example4_vulnerability_translation.py)

---

## Migration Complete (V2)

The V2 verification scripts now use `library_oq_import.py` as the shared OQ parsing and translation layer.

### Script coverage

- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_run_trisk_on_oq.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_run_trisk_on_oq.py)
    - Uses centralized loading for exposure, GMF, events, sitemesh, and risk-by-event.
    - Uses shared OQ→T-Risk transformation utilities for:
        - exposure-to-site mapping,
        - hazard matrix construction,
        - catalog years and uniform event rates.

- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare_trisk_on_oq.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare_trisk_on_oq.py)
    - Uses centralized OQ events and risk-by-event loaders for isolated consistency diagnostics.

- [T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare.py](T-Hazard/end_to_end_verification/V2_event_based_risk/V2_compare.py)
    - Uses centralized loaders for avg losses, aggregate risk, GMF data, and optional risk-by-event.

- [T-Hazard/end_to_end_verification/V2_event_based_risk/v2_decompose.py](T-Hazard/end_to_end_verification/V2_event_based_risk/v2_decompose.py)
    - Uses centralized exposure/GMF/sitemesh/risk-by-event loaders.
    - Uses schema inference helpers for robust site/event/GMV column handling.

### Result

- Version-specific OQ file naming and comment-line parsing are centralized.
- Remaining script logic focuses on analysis/metrics rather than CSV parsing details.
- Future OQ schema or filename updates should primarily require changes in one place:
    [T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.py](T-Hazard/end_to_end_verification/V2_event_based_risk/library_oq_import.py)

---

## Suggested Extension Points

If new OQ versions introduce additional column names or exports, extend:

- `DEFAULT_PATTERNS` for new filename families,
- `infer_*_column` functions for additional aliases,
- optional loader wrappers for new OQ report types.

Keep transformations deterministic by always reindexing against explicit event/site order before converting to arrays.
