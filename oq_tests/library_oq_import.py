#!/usr/bin/env python3
"""Utilities to import OpenQuake inputs/outputs and translate them for T-Risk workflows."""

from __future__ import annotations

import glob
import os
from io import StringIO
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


DEFAULT_PATTERNS = {
    'gmf': ['gmf-data_*.csv', 'gmf-data.csv', 'gmf_data_*.csv', 'gmf_data.csv'],
    'events': ['events_*.csv', 'events.csv'],
    'sitemesh': ['sitemesh_*.csv', 'sitemesh.csv'],
    'risk_by_event': ['risk_by_event_*.csv', 'risk_by_event.csv'],
    'avg_losses': [
        'avg_losses-mean_*.csv',
        'avg_losses-mean.csv',
        'avg_losses-rlz-*.csv',
        'avg_losses-stats_*.csv',
        'avg_losses-stats.csv',
        'avg_losses-rlzs_*.csv',
        'avg_losses-rlzs.csv',
    ],
    'aggrisk': ['aggrisk-*_*.csv', 'aggrisk-*.csv', 'aggrisk_*.csv', 'aggrisk.csv'],
    'vulnerability_xml': ['vulnerability_model.xml', 'vulnerability*.xml', '*vulnerability*.xml'],
}


def _find_latest_file(base_dir: str, patterns: Iterable[str]) -> str | None:
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(base_dir, pat)))
    return sorted(found)[-1] if found else None


def _read_oq_csv(path: str) -> pd.DataFrame:
    """Read OQ CSV while ignoring comment lines and normalizing column names."""
    with open(path, encoding='utf-8') as fh:
        lines = [line for line in fh if not line.startswith('#')]
    df = pd.read_csv(StringIO(''.join(lines)))
    df.columns = df.columns.str.strip()
    return df


def read_oq_csv(path: str) -> pd.DataFrame:
    return _read_oq_csv(path)


def load_oq_gmf_data(oq_dir: str) -> pd.DataFrame:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['gmf'])
    if not path:
        raise FileNotFoundError(f"OQ GMF file not found in {oq_dir}")
    return _read_oq_csv(path)


def load_oq_events(oq_dir: str) -> pd.DataFrame:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['events'])
    if not path:
        raise FileNotFoundError(f"OQ events file not found in {oq_dir}")
    return _read_oq_csv(path)


def load_oq_sitemesh(oq_dir: str) -> pd.DataFrame:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['sitemesh'])
    if not path:
        raise FileNotFoundError(f"OQ sitemesh file not found in {oq_dir}")
    return _read_oq_csv(path)


def load_oq_risk_by_event(oq_dir: str, required: bool = False) -> pd.DataFrame | None:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['risk_by_event'])
    if not path:
        if required:
            raise FileNotFoundError(f"OQ risk_by_event file not found in {oq_dir}")
        return None
    return _read_oq_csv(path)


def load_oq_avg_losses(oq_dir: str, required: bool = True) -> pd.DataFrame | None:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['avg_losses'])
    if not path:
        if required:
            raise FileNotFoundError(f"OQ avg_losses file not found in {oq_dir}")
        return None

    df = _read_oq_csv(path)
    struct_col = [c for c in df.columns if 'structural' in c.lower() and 'mean' in c.lower()]
    if not struct_col:
        struct_col = [c for c in df.columns if 'structural' in c.lower()]
    if struct_col:
        df = df.rename(columns={struct_col[0]: 'aal_usd_per_yr'})
    return df


def load_oq_aggrisk(oq_dir: str, required: bool = True) -> pd.DataFrame | None:
    path = _find_latest_file(oq_dir, DEFAULT_PATTERNS['aggrisk'])
    if not path:
        if required:
            raise FileNotFoundError(f"OQ aggrisk file not found in {oq_dir}")
        return None
    return _read_oq_csv(path)


def load_exposure_model(oq_inputs_dir: str, filename: str = 'exposure_model.csv') -> pd.DataFrame:
    path = os.path.join(oq_inputs_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exposure model not found: {path}")
    return pd.read_csv(path)


def load_oq_vulnerability_xml(oq_inputs_dir: str, filename: str | None = None) -> pd.DataFrame:
    """Load OQ vulnerability XML as a normalized long DataFrame.

    Returns columns:
    - vulnerability_id
    - distribution
    - imt
    - point_index
    - iml
    - mean_lr
    - cov_lr
    """
    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(oq_inputs_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vulnerability XML not found: {path}")
    else:
        path = _find_latest_file(oq_inputs_dir, DEFAULT_PATTERNS['vulnerability_xml'])
        if not path:
            raise FileNotFoundError(f"OQ vulnerability XML file not found in {oq_inputs_dir}")

    root = ET.parse(path).getroot()
    records = []

    for vuln_fun in root.findall('.//{*}vulnerabilityFunction'):
        vuln_id = vuln_fun.get('id')
        dist = vuln_fun.get('dist')

        imls_node = vuln_fun.find('{*}imls')
        means_node = vuln_fun.find('{*}meanLRs')
        covs_node = vuln_fun.find('{*}covLRs')

        if imls_node is None or means_node is None:
            continue

        imt = imls_node.get('imt', 'PGA')
        imls = [float(x) for x in (imls_node.text or '').split()]
        means = [float(x) for x in (means_node.text or '').split()]
        covs = [float(x) for x in (covs_node.text or '').split()] if covs_node is not None else []

        if len(imls) != len(means):
            raise ValueError(
                f"Vulnerability function '{vuln_id}' has inconsistent lengths: "
                f"len(imls)={len(imls)} vs len(meanLRs)={len(means)}"
            )
        if covs and len(covs) != len(imls):
            raise ValueError(
                f"Vulnerability function '{vuln_id}' has inconsistent covLRs length: "
                f"len(covLRs)={len(covs)} vs len(imls)={len(imls)}"
            )

        for i, (iml, mean) in enumerate(zip(imls, means)):
            cov = covs[i] if covs else np.nan
            records.append(
                {
                    'vulnerability_id': vuln_id,
                    'distribution': dist,
                    'imt': imt,
                    'point_index': i,
                    'iml': iml,
                    'mean_lr': mean,
                    'cov_lr': cov,
                }
            )

    if not records:
        raise ValueError(f"No vulnerabilityFunction nodes found in {path}")

    return pd.DataFrame(records)


def vulnerability_to_trisk_arrays(
    vulnerability_df: pd.DataFrame,
    taxonomy_order: list[str] | None = None,
    require_common_imls: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    """Translate normalized vulnerability data to T-Risk arrays.

    Returns:
    - x_grid: shape (M,), common IML grid
    - c_matrix_mean: shape (K, M), mean LR curves per taxonomy
    - taxonomy_to_index: mapping taxonomy -> row index in c_matrix_mean
    - c_matrix_cov: shape (K, M), cov LR curves per taxonomy (NaN if unavailable)
    """
    required_cols = {'vulnerability_id', 'point_index', 'iml', 'mean_lr'}
    missing = required_cols - set(vulnerability_df.columns)
    if missing:
        raise KeyError(f"Missing required vulnerability columns: {sorted(missing)}")

    if taxonomy_order is None:
        taxonomy_order = vulnerability_df['vulnerability_id'].drop_duplicates().tolist()

    curves_iml = {}
    curves_mean = {}
    curves_cov = {}

    for taxonomy in taxonomy_order:
        grp = vulnerability_df[vulnerability_df['vulnerability_id'] == taxonomy].sort_values('point_index')
        if grp.empty:
            raise KeyError(f"Taxonomy '{taxonomy}' not found in vulnerability DataFrame")
        curves_iml[taxonomy] = grp['iml'].to_numpy(dtype=np.float32)
        curves_mean[taxonomy] = grp['mean_lr'].to_numpy(dtype=np.float32)
        if 'cov_lr' in grp.columns:
            curves_cov[taxonomy] = grp['cov_lr'].to_numpy(dtype=np.float32)
        else:
            curves_cov[taxonomy] = np.full_like(curves_mean[taxonomy], np.nan, dtype=np.float32)

    ref_tax = taxonomy_order[0]
    x_grid = curves_iml[ref_tax]

    if require_common_imls:
        for taxonomy in taxonomy_order[1:]:
            if len(curves_iml[taxonomy]) != len(x_grid) or not np.allclose(curves_iml[taxonomy], x_grid):
                raise ValueError(
                    f"Taxonomy '{taxonomy}' has a different IML grid from '{ref_tax}'. "
                    "Set require_common_imls=False and harmonize externally if needed."
                )

    c_matrix_mean = np.vstack([curves_mean[taxonomy] for taxonomy in taxonomy_order]).astype(np.float32)
    c_matrix_cov = np.vstack([curves_cov[taxonomy] for taxonomy in taxonomy_order]).astype(np.float32)
    taxonomy_to_index = {taxonomy: i for i, taxonomy in enumerate(taxonomy_order)}

    return x_grid, c_matrix_mean, taxonomy_to_index, c_matrix_cov


def infer_site_id_column(df: pd.DataFrame) -> str:
    if 'custom_site_id' in df.columns:
        return 'custom_site_id'
    if 'sid' in df.columns:
        return 'sid'
    raise KeyError("No site-id column found. Expected one of ['custom_site_id', 'sid']")


def infer_event_id_column(df: pd.DataFrame) -> str:
    if 'event_id' in df.columns:
        return 'event_id'
    if 'eid' in df.columns:
        return 'eid'
    raise KeyError("No event-id column found. Expected one of ['event_id', 'eid']")


def infer_gmv_column(df: pd.DataFrame) -> str:
    for candidate in ('gmv_PGA', 'gmv', 'gmv_sa'):
        if candidate in df.columns:
            return candidate
    raise KeyError("No GMV column found. Tried ['gmv_PGA', 'gmv', 'gmv_sa']")


def map_exposure_to_oq_sites(
    exposure_df: pd.DataFrame,
    sitemesh_df: pd.DataFrame,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    strict: bool = True,
) -> pd.DataFrame:
    site_col = infer_site_id_column(sitemesh_df)
    asset_site = exposure_df.merge(sitemesh_df[[lon_col, lat_col, site_col]], on=[lon_col, lat_col], how='left')
    if strict and asset_site[site_col].isna().any():
        raise RuntimeError(f"Failed to map one or more assets to OQ site ids ({site_col}).")
    return asset_site


def build_hazard_matrix_from_oq(
    gmf_df: pd.DataFrame,
    event_ids: np.ndarray,
    site_ids: list,
    fillna: float = 0.0,
) -> np.ndarray:
    event_col = infer_event_id_column(gmf_df)
    site_col = infer_site_id_column(gmf_df)
    gmv_col = infer_gmv_column(gmf_df)

    pivot = gmf_df.pivot(index=event_col, columns=site_col, values=gmv_col)
    pivot = pivot.reindex(index=event_ids, columns=site_ids)
    if pivot.isna().any().any():
        pivot = pivot.fillna(fillna)
    return pivot.to_numpy(dtype=np.float32).T


def compute_catalog_years(events_df: pd.DataFrame) -> float:
    if 'year' in events_df.columns:
        return float(events_df['year'].max() + 1)
    return float(len(events_df))


def compute_uniform_event_rates(events_df: pd.DataFrame) -> np.ndarray:
    event_count = len(events_df)
    if event_count == 0:
        return np.zeros(0, dtype=np.float32)
    years = compute_catalog_years(events_df)
    return np.full(event_count, 1.0 / years, dtype=np.float32)


# ==========================================
# HAZARD CURVE LOADER (for classical risk)
# ==========================================

def load_oq_hazard_curves(csv_path: str, imt: str = 'PGA') -> tuple:
    """Load OQ hazard-curve CSV export and return arrays.

    OQ ``hcurves`` CSV has a header comment line listing the IML values, e.g.::

        #,, imt="PGA" sa_period=... iml=0.01 0.02 ...
        site_id, lon, lat, poe-0.01, poe-0.02, ...

    Parameters
    ----------
    csv_path : str
        Path to ``hcurves-*.csv`` file exported from OQ.
    imt : str
        Intensity measure type to extract (e.g. ``'PGA'``, ``'SA(0.3)'``).

    Returns
    -------
    site_ids : np.ndarray, shape (S,)
        Site IDs in the order they appear in the CSV.
    hazard_imls : np.ndarray, shape (L,), dtype float32
        IML levels for the hazard curves.
    hazard_poes : np.ndarray, shape (S, L), dtype float32
        PoE matrix — hazard_poes[s, l] = P(IML > imls[l]) at site s.
    lons : np.ndarray, shape (S,)
    lats : np.ndarray, shape (S,)
    """
    import re as _re

    # --- read and strip comment lines, extract IML header ---
    with open(csv_path, encoding='utf-8') as fh:
        raw_lines = fh.readlines()

    comment_lines = [l for l in raw_lines if l.startswith('#')]
    data_lines = [l for l in raw_lines if not l.startswith('#')]

    # Find IML values for the requested IMT in comment lines
    hazard_imls = None
    for cline in comment_lines:
        # OQ format: investigation_time=50, imt="PGA" ..., poe-0.01 poe-0.02 ...
        # or header with iml values
        if imt in cline:
            # Try to extract float sequences after the IMT mention
            floats = _re.findall(r'[\d.]+(?:[eE][+-]?\d+)?', cline.split(imt)[-1])
            if len(floats) > 2:
                hazard_imls = np.array([float(f) for f in floats], dtype=np.float32)
                break

    # Parse the data CSV
    df = pd.read_csv(StringIO(''.join(data_lines)))
    df.columns = df.columns.str.strip()

    # Identify PoE columns — they contain the imt string or start with "poe-"
    poe_cols = [c for c in df.columns if c.startswith('poe-') or c.startswith(f'{imt}-')]
    if not poe_cols:
        # Fallback: any numeric columns after lon/lat
        skip = {'custom_site_id', 'site_id', 'sid', 'lon', 'lat'}
        poe_cols = [c for c in df.columns if c not in skip]

    poe_matrix = df[poe_cols].to_numpy(dtype=np.float32)

    # If we couldn't parse IML values from comments, try from column names
    if hazard_imls is None or len(hazard_imls) != poe_matrix.shape[1]:
        imls_from_cols = []
        for c in poe_cols:
            nums = _re.findall(r'[\d.]+(?:[eE][+-]?\d+)?', c)
            if nums:
                imls_from_cols.append(float(nums[-1]))
        if len(imls_from_cols) == poe_matrix.shape[1]:
            hazard_imls = np.array(imls_from_cols, dtype=np.float32)
        else:
            raise ValueError(f"Cannot determine IML levels for IMT={imt} from {csv_path}")

    # Site IDs
    sid_col = 'custom_site_id' if 'custom_site_id' in df.columns else (
        'sid' if 'sid' in df.columns else 'site_id')
    site_ids = df[sid_col].to_numpy() if sid_col in df.columns else np.arange(len(df))
    lons = df['lon'].to_numpy(dtype=np.float64) if 'lon' in df.columns else np.zeros(len(df))
    lats = df['lat'].to_numpy(dtype=np.float64) if 'lat' in df.columns else np.zeros(len(df))

    return site_ids, hazard_imls, poe_matrix, lons, lats


# ==========================================
# FRAGILITY XML LOADER
# ==========================================

def load_oq_fragility_xml(oq_inputs_dir: str, filename: str | None = None) -> tuple:
    """Load OQ fragility model XML.

    Returns
    -------
    limit_states : list[str]
        Ordered limit-state names (e.g. ['slight', 'moderate', 'extreme', 'complete']).
    fragility_df : pd.DataFrame
        Long-form DataFrame with columns:
        taxonomy, imt, point_index, iml, + one column per limit state with PoE values.
    """
    patterns = ['structural_fragility_model.xml', '*fragility*.xml']
    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(oq_inputs_dir, filename)
    else:
        path = _find_latest_file(oq_inputs_dir, patterns)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Fragility XML not found: {filename or oq_inputs_dir}")

    root = ET.parse(path).getroot()

    # Extract limit states
    ls_node = root.find('.//{*}limitStates')
    if ls_node is None:
        raise ValueError("No <limitStates> found in fragility XML")
    limit_states = ls_node.text.strip().split()

    records = []
    for func in root.findall('.//{*}fragilityFunction'):
        taxonomy = func.get('id')
        imls_node = func.find('{*}imls')
        if imls_node is None:
            continue
        imt = imls_node.get('imt', 'PGA')
        imls = [float(x) for x in (imls_node.text or '').split()]

        poe_dict = {}
        for ls in limit_states:
            poe_node = func.find(f'{{*}}poes[@ls="{ls}"]')
            if poe_node is not None:
                poe_dict[ls] = [float(x) for x in (poe_node.text or '').split()]
            else:
                poe_dict[ls] = [0.0] * len(imls)

        for i, iml in enumerate(imls):
            row = {'taxonomy': taxonomy, 'imt': imt, 'point_index': i, 'iml': iml}
            for ls in limit_states:
                row[ls] = poe_dict[ls][i]
            records.append(row)

    if not records:
        raise ValueError(f"No fragilityFunction nodes found in {path}")

    return limit_states, pd.DataFrame(records)


def fragility_to_trisk_arrays(
    limit_states: list,
    fragility_df: pd.DataFrame,
    taxonomy_order: list | None = None,
) -> tuple:
    """Convert fragility DataFrame to T-Risk tensor arrays.

    Returns
    -------
    x_grid : np.ndarray, shape (M,)
    F_tensor : np.ndarray, shape (K, D, M)
        F[k,d,m] = P(exceeding limit-state d | IML = x_grid[m]) for typology k.
    taxonomy_to_index : dict[str, int]
    """
    D = len(limit_states)
    if taxonomy_order is None:
        taxonomy_order = fragility_df['taxonomy'].drop_duplicates().tolist()
    K = len(taxonomy_order)

    ref = fragility_df[fragility_df['taxonomy'] == taxonomy_order[0]].sort_values('point_index')
    x_grid = ref['iml'].to_numpy(dtype=np.float32)
    M = len(x_grid)

    F_tensor = np.zeros((K, D, M), dtype=np.float32)
    for k, tax in enumerate(taxonomy_order):
        grp = fragility_df[fragility_df['taxonomy'] == tax].sort_values('point_index')
        for d, ls in enumerate(limit_states):
            F_tensor[k, d, :] = grp[ls].to_numpy(dtype=np.float32)

    taxonomy_to_index = {tax: i for i, tax in enumerate(taxonomy_order)}
    return x_grid, F_tensor, taxonomy_to_index


# ==========================================
# CONSEQUENCE CSV LOADER
# ==========================================

def load_oq_consequence_csv(csv_path: str, limit_states: list) -> tuple:
    """Load OQ consequence CSV.

    Parameters
    ----------
    csv_path : str
        Path to consequences CSV file.
    limit_states : list[str]
        Ordered limit state names matching the fragility model.

    Returns
    -------
    consequence_ratios : dict[str, np.ndarray]
        Mapping taxonomy → array of shape (D+1,) where index 0 = no-damage (always 0),
        indices 1..D = consequence ratios for each limit state.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Identify taxonomy column
    tax_col = 'risk_id' if 'risk_id' in df.columns else 'taxonomy'

    result = {}
    for _, row in df.iterrows():
        tax = row[tax_col]
        ratios = [0.0]  # no-damage ratio = 0
        for ls in limit_states:
            ratios.append(float(row.get(ls, 0.0)))
        result[tax] = np.array(ratios, dtype=np.float32)

    return result


def consequence_to_trisk_array(consequence_dict: dict, taxonomy_order: list) -> np.ndarray:
    """Convert consequence dict to tensor array.

    Returns
    -------
    consequence_ratios : np.ndarray, shape (K, D+1)
    """
    K = len(taxonomy_order)
    D_plus_1 = len(next(iter(consequence_dict.values())))
    arr = np.zeros((K, D_plus_1), dtype=np.float32)
    for k, tax in enumerate(taxonomy_order):
        if tax in consequence_dict:
            arr[k] = consequence_dict[tax]
    return arr
