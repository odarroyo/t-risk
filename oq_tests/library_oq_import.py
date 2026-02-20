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
