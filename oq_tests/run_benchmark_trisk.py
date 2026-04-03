#!/usr/bin/env python3
"""
Benchmark Phase 2: T-Risk timing against OpenQuake.

Reads CSV exports and OQ timing from benchmark_outputs/ (produced by
run_benchmark_oq.py), then times the TensorialRiskEngine on the same
hazard inputs.

Requires: T-Risk environment (tensorial_engine venv with tensorflow).
"""
import configparser
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Paths --
BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / 'benchmark_outputs'
PLOT_DIR = BASE_DIR / 'benchmark_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OQ_ROOT = BASE_DIR.parent

# -- T-Risk import --
TRISK_DIR = Path(__file__).resolve().parents[2] / 'T-Hazard' / 'T-Risk'
if not TRISK_DIR.exists():
    # Fallback: try relative to workspace
    TRISK_DIR = OQ_ROOT.parent / 'T-Hazard' / 'T-Risk'

if TRISK_DIR.exists():
    sys.path.insert(0, str(TRISK_DIR))
else:
    print(f'WARNING: T-Risk directory not found at {TRISK_DIR}')
    print('Make sure T-Hazard/T-Risk is alongside oq-engine in the workspace.')
    sys.exit(1)

from tensor_engine import TensorialRiskEngine  # noqa: E402

# -- Local library --
from library_oq_import import (  # noqa: E402
    compute_uniform_event_rates,
    infer_event_id_column,
    infer_site_id_column,
    load_exposure_model,
    load_oq_vulnerability_xml,
    map_exposure_to_oq_sites,
)

LOSS_COMPONENTS = ('structural', 'nonstructural')


def find_csv(directory: Path, prefix: str) -> Path:
    """Find the first CSV matching prefix in directory."""
    candidates = sorted(directory.glob(f'{prefix}*.csv'))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv in {directory}")
    return candidates[0]


def get_vuln_files(example_dir: Path, job_file: str) -> dict:
    """Read vulnerability file references from job.ini."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(example_dir / job_file)
    vuln_map = {}
    if parser.has_section('vulnerability'):
        for comp in LOSS_COMPONENTS:
            key = f'{comp}_vulnerability_file'
            if parser.has_option('vulnerability', key):
                vuln_map[comp] = parser.get('vulnerability', key)
    return vuln_map


def map_assets_sites_tolerant(exposure, sitemesh):
    """Map assets to sites with fallback to nearest-neighbor."""
    site_col = infer_site_id_column(sitemesh)

    def _nearest(df):
        out = df.copy()
        miss = out[site_col].isna()
        if not miss.any():
            return out
        sm_lon = sitemesh['lon'].to_numpy(dtype=np.float64)
        sm_lat = sitemesh['lat'].to_numpy(dtype=np.float64)
        sm_sid = sitemesh[site_col].to_numpy()
        lon_u = out.loc[miss, 'lon'].to_numpy(dtype=np.float64)
        lat_u = out.loc[miss, 'lat'].to_numpy(dtype=np.float64)
        for i in range(len(lon_u)):
            d2 = (sm_lon - lon_u[i]) ** 2 + (sm_lat - lat_u[i]) ** 2
            out.loc[out.index[miss][i], site_col] = sm_sid[int(np.argmin(d2))]
        return out

    try:
        mapped = map_exposure_to_oq_sites(exposure, sitemesh, strict=False)
        return _nearest(mapped)
    except RuntimeError:
        exp = exposure.copy()
        sm = sitemesh.copy()
        exp['lon_r'] = exp['lon'].round(5)
        exp['lat_r'] = exp['lat'].round(5)
        sm['lon_r'] = sm['lon'].round(5)
        sm['lat_r'] = sm['lat'].round(5)
        mapped = exp.merge(sm[['lon_r', 'lat_r', site_col]],
                           on=['lon_r', 'lat_r'], how='left')
        mapped = mapped.drop(columns=['lon_r', 'lat_r'])
        return _nearest(mapped)


def benchmark_trisk(demo_name: str, source_demo: str) -> dict:
    """
    Time T-Risk computation for one demo.
    Returns timing dict with data-loading and computation times.
    """
    example_dir = OQ_ROOT / 'demos' / 'risk' / source_demo
    bench_out = BENCH_DIR / demo_name

    if not bench_out.exists():
        raise FileNotFoundError(f"benchmark_outputs/{demo_name} not found. "
                                f"Run run_benchmark_oq.py first.")

    # ---- Phase A: Data Loading (timed) ----
    t_load_start = time.perf_counter()

    events = pd.read_csv(find_csv(bench_out, 'events'), comment='#')
    gmf = pd.read_csv(find_csv(bench_out, 'gmf-data'), comment='#')
    sitemesh = pd.read_csv(find_csv(bench_out, 'sitemesh'), comment='#')

    exposure = load_exposure_model(str(example_dir), filename='exposure_model.csv')
    asset_site = map_assets_sites_tolerant(exposure, sitemesh)
    site_col = infer_site_id_column(asset_site)
    asset_site = asset_site[asset_site[site_col].notna()].copy()

    event_col = infer_event_id_column(events)
    event_ids = events[event_col].astype(int).values
    site_ids = asset_site[site_col].tolist()

    vuln_files = get_vuln_files(example_dir, 'job.ini')

    # Build hazard matrix (N x Q) for each IMT, accumulate vulnerability
    gmv_cols = [c for c in gmf.columns if c.startswith('gmv_')]
    first_imt = gmv_cols[0].replace('gmv_', '')

    # Pivot GMF into hazard matrix
    def build_H(imt):
        col = f'gmv_{imt}'
        ecol = infer_event_id_column(gmf)
        scol = infer_site_id_column(gmf)
        pivot = gmf.pivot(index=ecol, columns=scol, values=col)
        pivot = pivot.reindex(index=event_ids, columns=site_ids).fillna(0.0)
        return pivot.to_numpy(dtype=np.float32).T  # (N, Q)

    # Collect vulnerability curves and build arrays
    all_vuln_dfs = {}
    for comp, vf in vuln_files.items():
        if comp not in asset_site.columns:
            continue
        values = asset_site[comp].astype(float).values
        if 'number' in asset_site.columns:
            values = values * asset_site['number'].astype(float).values
        if np.all(values == 0):
            continue
        vuln_df = load_oq_vulnerability_xml(str(example_dir), filename=vf)
        all_vuln_dfs[comp] = (vuln_df, values)

    # Build TensorialRiskEngine inputs
    # Merge vulnerability curves across components: we sum losses component by component
    # For each component, run a separate engine instance and sum results

    lambdas = compute_uniform_event_rates(events)

    # Pre-build per-component engine inputs
    engine_inputs = []
    h_cache = {}
    for comp, (vuln_df, values) in all_vuln_dfs.items():
        tax = asset_site['taxonomy'].astype(str).values
        unique_tax = vuln_df['vulnerability_id'].unique()
        # Get the IMT from the vulnerability
        imt_values = vuln_df['imt'].dropna().unique().tolist()
        imt = imt_values[0] if imt_values else 'PGA'

        if imt not in h_cache:
            h_cache[imt] = build_H(imt)
        H = h_cache[imt]

        # Build vulnerability matrix C (K x M) and typology index u (N,)
        x_grid = np.sort(vuln_df[vuln_df['vulnerability_id'] == unique_tax[0]]
                         ['iml'].to_numpy(dtype=np.float32))
        tax_to_idx = {t: i for i, t in enumerate(unique_tax)}
        K = len(unique_tax)
        M = len(x_grid)
        C = np.zeros((K, M), dtype=np.float32)
        for t in unique_tax:
            rows = vuln_df[vuln_df['vulnerability_id'] == t].sort_values('point_index')
            curve = rows['mean_lr'].to_numpy(dtype=np.float32)
            if len(curve) == M:
                C[tax_to_idx[t]] = curve

        u = np.array([tax_to_idx.get(t, 0) for t in tax], dtype=np.int32)
        v = values.astype(np.float32)

        engine_inputs.append({
            'comp': comp, 'v': v, 'u': u, 'C': C,
            'x_grid': x_grid, 'H': H, 'lambdas': lambdas.astype(np.float32),
        })

    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start

    # ---- Phase B: T-Risk Computation (timed) ----
    t_compute_start = time.perf_counter()

    total_loss_per_event = np.zeros(len(event_ids), dtype=np.float64)
    total_aal_per_asset = np.zeros(len(site_ids), dtype=np.float64)

    for inp in engine_inputs:
        engine = TensorialRiskEngine(
            v=inp['v'], u=inp['u'], C=inp['C'],
            x_grid=inp['x_grid'], H=inp['H'], lambdas=inp['lambdas']
        )
        J_matrix, metrics = engine.compute_loss_and_metrics()
        total_loss_per_event += np.array(metrics['loss_per_event'], dtype=np.float64)
        total_aal_per_asset += np.array(metrics['aal_per_asset'], dtype=np.float64)

    t_compute_end = time.perf_counter()
    compute_time = t_compute_end - t_compute_start

    total_time = load_time + compute_time

    result = {
        'name': demo_name,
        'n_assets': len(site_ids),
        'n_events': len(event_ids),
        'trisk_load_time_sec': round(load_time, 4),
        'trisk_compute_time_sec': round(compute_time, 4),
        'trisk_total_time_sec': round(total_time, 4),
        'n_components': len(engine_inputs),
    }
    print(f'  Load: {load_time:.3f}s  Compute: {compute_time:.3f}s  '
          f'Total: {total_time:.3f}s')

    return result


def make_benchmark_plots(summary_df: pd.DataFrame):
    """Generate comparison bar chart and scaling chart."""

    # --- Bar chart: OQ risk vs T-Risk total per demo ---
    fig, ax = plt.subplots(figsize=(10, 5))
    demos = summary_df['name'].tolist()
    x = np.arange(len(demos))
    w = 0.35

    oq_times = summary_df['oq_risk_time_sec'].values
    tr_times = summary_df['trisk_total_time_sec'].values

    bars_oq = ax.bar(x - w / 2, oq_times, w, label='OpenQuake (risk only)',
                     color='steelblue')
    bars_tr = ax.bar(x + w / 2, tr_times, w, label='T-Risk (load + compute)',
                     color='tomato')

    # Add speedup labels
    for i in range(len(demos)):
        if tr_times[i] > 0:
            speedup = oq_times[i] / tr_times[i]
            ypos = max(oq_times[i], tr_times[i]) * 1.05
            ax.text(x[i], ypos, f'{speedup:.1f}×', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(demos, rotation=15, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Benchmark: OpenQuake vs T-Risk')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'benchmark_comparison.png', dpi=220)
    plt.close(fig)

    # --- Stacked bar: T-Risk load vs compute ---
    fig, ax = plt.subplots(figsize=(10, 5))
    load_times = summary_df['trisk_load_time_sec'].values
    comp_times = summary_df['trisk_compute_time_sec'].values

    ax.bar(x, load_times, w, label='T-Risk data loading', color='sandybrown')
    ax.bar(x, comp_times, w, bottom=load_times, label='T-Risk computation',
           color='tomato')
    ax.bar(x + w, oq_times, w, label='OpenQuake risk-only', color='steelblue')

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(demos, rotation=15, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Breakdown: T-Risk (load + compute) vs OpenQuake')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'benchmark_breakdown.png', dpi=220)
    plt.close(fig)

    # --- Scaling chart: events vs time (if multiple event counts) ---
    eb = summary_df[summary_df['name'].str.startswith('EventBased')].copy()
    if len(eb) > 1:
        eb = eb.sort_values('n_events')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eb['n_events'], eb['oq_risk_time_sec'], 'o-',
                label='OpenQuake', color='steelblue', markersize=8)
        ax.plot(eb['n_events'], eb['trisk_total_time_sec'], 's-',
                label='T-Risk', color='tomato', markersize=8)
        ax.set_xlabel('Number of events')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Scaling: Runtime vs Event Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / 'benchmark_scaling.png', dpi=220)
        plt.close(fig)

    print(f'Plots saved to: {PLOT_DIR}/')


def main():
    print('=' * 60)
    print('T-Risk Benchmark - Phase 2 (T-Risk timing)')
    print('=' * 60)

    # Load OQ timing data
    timing_path = BENCH_DIR / 'oq_timing.json'
    if not timing_path.exists():
        print(f'ERROR: {timing_path} not found.')
        print('Run run_benchmark_oq.py first to generate OQ timing data.')
        sys.exit(1)

    with open(timing_path) as f:
        oq_timing = json.load(f)

    # Demo name → source demo mapping
    demo_sources = {
        'ScenarioRisk': 'ScenarioRisk',
        'EventBasedRisk': 'EventBasedRisk',
        'Reinsurance': 'Reinsurance',
        'EventBasedRisk_scaled': 'EventBasedRisk',
    }

    results = []
    for oq_entry in oq_timing:
        name = oq_entry['name']
        source = demo_sources.get(name, name.split('_scaled')[0])
        print(f'\n--- {name} ---')

        try:
            tr_result = benchmark_trisk(name, source)
            tr_result['oq_risk_time_sec'] = oq_entry['risk_time_sec']
            if tr_result['trisk_total_time_sec'] > 0:
                tr_result['speedup'] = round(
                    oq_entry['risk_time_sec'] / tr_result['trisk_total_time_sec'], 2
                )
            else:
                tr_result['speedup'] = float('inf')
            results.append(tr_result)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

    if not results:
        print('\nNo benchmarks completed successfully.')
        sys.exit(1)

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = BENCH_DIR / 'benchmark_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    # Generate plots
    make_benchmark_plots(summary_df)

    # Print summary
    print(f'\n{"=" * 60}')
    print(f'Benchmark summary saved to: {summary_path}')
    print(f'\n{"Demo":<28} {"OQ risk(s)":>10} {"T-Risk(s)":>10} {"Speedup":>8}')
    print('-' * 60)
    for r in results:
        print(f'{r["name"]:<28} {r["oq_risk_time_sec"]:>10.2f} '
              f'{r["trisk_total_time_sec"]:>10.2f} {r["speedup"]:>7.1f}×')


if __name__ == '__main__':
    main()
