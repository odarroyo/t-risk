#!/usr/bin/env python3
"""
Batched T-Risk Benchmark for Large Event Sets.

Runs TensorialRiskEngine on the EventBasedRisk_scaled demo using
event-batching to avoid GPU OOM. Compares against the OQ timing
from run_benchmark_oq.py.

Strategy: partition the Q events into batches of size B, run the
engine on each batch, and accumulate loss_per_event and aal_per_asset.

Requires: T-Risk environment (tensorial_engine venv with tensorflow).
"""
import configparser
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths --
BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / 'benchmark_outputs'
OQ_ROOT = BASE_DIR.parent

# -- T-Risk import --
TRISK_DIR = Path(__file__).resolve().parents[2] / 'T-Hazard' / 'T-Risk'
if not TRISK_DIR.exists():
    TRISK_DIR = OQ_ROOT.parent / 'T-Hazard' / 'T-Risk'
if TRISK_DIR.exists():
    sys.path.insert(0, str(TRISK_DIR))
else:
    print(f'ERROR: T-Risk directory not found at {TRISK_DIR}')
    sys.exit(1)

from tensor_engine import TensorialRiskEngine  # noqa: E402

from library_oq_import import (  # noqa: E402
    compute_uniform_event_rates,
    infer_event_id_column,
    infer_site_id_column,
    load_exposure_model,
    load_oq_vulnerability_xml,
    map_exposure_to_oq_sites,
)

LOSS_COMPONENTS = ('structural', 'nonstructural')
DEFAULT_BATCH_SIZE = 20000  # events per batch


def find_csv(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f'{prefix}*.csv'))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv in {directory}")
    return candidates[0]


def get_vuln_files(example_dir: Path, job_file: str) -> dict:
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


def save_arrays_npz(cache_path: Path, component_inputs: list,
                    lambdas: np.ndarray, N: int, Q: int):
    """Save all pre-built arrays to a compressed .npz cache file."""
    data = {'lambdas': lambdas, 'N': np.array(N), 'Q': np.array(Q),
            'n_components': np.array(len(component_inputs))}
    for ci, inp in enumerate(component_inputs):
        data[f'v_{ci}'] = inp['v']
        data[f'u_{ci}'] = inp['u']
        data[f'C_{ci}'] = inp['C']
        data[f'x_grid_{ci}'] = inp['x_grid']
        data[f'H_{ci}'] = inp['H_full']
        data[f'comp_{ci}'] = np.array(inp['comp'], dtype=object)
    np.savez(cache_path, **data)


def load_arrays_npz(cache_path: Path) -> tuple:
    """Load pre-built arrays from .npz cache. Returns (component_inputs, lambdas, N, Q)."""
    d = np.load(cache_path, allow_pickle=True)
    lambdas = d['lambdas']
    N = int(d['N'])
    Q = int(d['Q'])
    nc = int(d['n_components'])
    component_inputs = []
    for ci in range(nc):
        component_inputs.append({
            'comp': str(d[f'comp_{ci}']),
            'v': d[f'v_{ci}'],
            'u': d[f'u_{ci}'],
            'C': d[f'C_{ci}'],
            'x_grid': d[f'x_grid_{ci}'],
            'H_full': d[f'H_{ci}'],
        })
    return component_inputs, lambdas, N, Q


def load_from_csv(bench_out: Path, example_dir: Path) -> tuple:
    """Load data from CSV exports, build arrays. Returns (component_inputs, lambdas, N, Q)."""
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
    N = len(site_ids)
    Q = len(event_ids)

    vuln_files = get_vuln_files(example_dir, 'job.ini')

    def build_H(imt):
        col = f'gmv_{imt}'
        ecol = infer_event_id_column(gmf)
        scol = infer_site_id_column(gmf)
        pivot = gmf.pivot(index=ecol, columns=scol, values=col)
        pivot = pivot.reindex(index=event_ids, columns=site_ids).fillna(0.0)
        return pivot.to_numpy(dtype=np.float32).T

    lambdas = compute_uniform_event_rates(events)

    h_cache = {}
    component_inputs = []
    for comp, vf in vuln_files.items():
        if comp not in asset_site.columns:
            continue
        values = asset_site[comp].astype(float).values
        if 'number' in asset_site.columns:
            values = values * asset_site['number'].astype(float).values
        if np.all(values == 0):
            continue

        vuln_df = load_oq_vulnerability_xml(str(example_dir), filename=vf)
        tax = asset_site['taxonomy'].astype(str).values
        unique_tax = vuln_df['vulnerability_id'].unique()

        imt_values = vuln_df['imt'].dropna().unique().tolist()
        imt = imt_values[0] if imt_values else 'PGA'
        if imt not in h_cache:
            h_cache[imt] = build_H(imt)
        H_full = h_cache[imt]

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

        component_inputs.append({
            'comp': comp, 'v': v, 'u': u, 'C': C,
            'x_grid': x_grid, 'H_full': H_full,
        })

    return component_inputs, lambdas, N, Q


def main():
    batch_size = DEFAULT_BATCH_SIZE
    no_cache = False

    # Parse args: [batch_size] [--no-cache]
    args = sys.argv[1:]
    if '--no-cache' in args:
        no_cache = True
        args.remove('--no-cache')
    if args:
        batch_size = int(args[0])

    demo_name = 'EventBasedRisk_scaled'
    source_demo = 'EventBasedRisk'
    bench_out = BENCH_DIR / demo_name
    example_dir = OQ_ROOT / 'demos' / 'risk' / source_demo

    print('=' * 60)
    print(f'Batched T-Risk Benchmark: {demo_name}')
    print(f'Batch size: {batch_size} events')
    print('=' * 60)

    if not bench_out.exists():
        print(f'ERROR: {bench_out} not found. Run run_benchmark_oq.py first.')
        sys.exit(1)

    # ---- Load OQ timing ----
    with open(BENCH_DIR / 'oq_timing.json') as f:
        oq_timing = json.load(f)
    oq_entry = next(e for e in oq_timing if e['name'] == demo_name)
    oq_risk_time = oq_entry['risk_time_sec']

    # ---- Phase A: Data Loading (timed) ----
    cache_path = bench_out / 'trisk_arrays.npz'
    use_cache = cache_path.exists() and not no_cache

    t_load_start = time.perf_counter()

    if use_cache:
        print(f'\nLoading from .npz cache: {cache_path.name}')
        component_inputs, lambdas, N, Q = load_arrays_npz(cache_path)
        load_source = 'npz'
    else:
        print(f'\nLoading from CSV exports (first run)...')
        component_inputs, lambdas, N, Q = load_from_csv(bench_out, example_dir)
        load_source = 'csv'
        # Save cache for next run
        print(f'Saving .npz cache: {cache_path.name}')
        t_save_start = time.perf_counter()
        save_arrays_npz(cache_path, component_inputs, lambdas, N, Q)
        t_save_end = time.perf_counter()
        print(f'Cache saved in {t_save_end - t_save_start:.2f}s '
              f'({cache_path.stat().st_size / 1e6:.1f} MB)')

    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start

    print(f'\nData loaded: N={N}, Q={Q}, components={len(component_inputs)} '
          f'(source: {load_source})')
    print(f'Load time: {load_time:.3f}s')
    print(f'Full matrix size: {N} x {Q} = {N*Q:,} elements '
          f'({N*Q*4/1e9:.2f} GB per component)')

    # ---- Phase B: Batched Computation (timed) ----
    n_batches = math.ceil(Q / batch_size)
    print(f'\nBatching: {n_batches} batches of up to {batch_size} events')
    print(f'Per-batch matrix: {N} x {batch_size} = {N*batch_size:,} elements '
          f'({N*batch_size*4/1e9:.3f} GB)')

    t_compute_start = time.perf_counter()

    total_loss_per_event = np.zeros(Q, dtype=np.float64)
    total_aal_per_asset = np.zeros(N, dtype=np.float64)

    for ci, inp in enumerate(component_inputs):
        print(f'\n  Component {ci+1}/{len(component_inputs)}: {inp["comp"]}')
        H_full = inp['H_full']
        lambdas_f32 = lambdas.astype(np.float32)

        for b in range(n_batches):
            q_start = b * batch_size
            q_end = min((b + 1) * batch_size, Q)
            batch_q = q_end - q_start

            H_batch = H_full[:, q_start:q_end]           # (N, batch_q)
            lam_batch = lambdas_f32[q_start:q_end]        # (batch_q,)

            engine = TensorialRiskEngine(
                v=inp['v'], u=inp['u'], C=inp['C'],
                x_grid=inp['x_grid'], H=H_batch, lambdas=lam_batch
            )
            J_batch, metrics = engine.compute_loss_and_metrics()

            total_loss_per_event[q_start:q_end] += np.array(
                metrics['loss_per_event'], dtype=np.float64)
            total_aal_per_asset += np.array(
                metrics['aal_per_asset'], dtype=np.float64)

            if (b + 1) % 3 == 0 or b == n_batches - 1:
                print(f'    Batch {b+1}/{n_batches} done '
                      f'(events {q_start}–{q_end})')

    t_compute_end = time.perf_counter()
    compute_time = t_compute_end - t_compute_start
    total_time = load_time + compute_time

    # ---- Summary ----
    print(f'\n{"=" * 60}')
    print(f'RESULTS: {demo_name} (batched, B={batch_size})')
    print(f'{"=" * 60}')
    print(f'  Assets (N):         {N:,}')
    print(f'  Events (Q):         {Q:,}')
    print(f'  Batch size (B):     {batch_size:,}')
    print(f'  Number of batches:  {n_batches}')
    print(f'  Components:         {len(component_inputs)}')
    print(f'')
    print(f'  T-Risk load time:    {load_time:.3f} s')
    print(f'  T-Risk compute time: {compute_time:.3f} s')
    print(f'  T-Risk total time:   {total_time:.3f} s')
    print(f'')
    print(f'  OQ risk-only time:   {oq_risk_time:.3f} s')
    if total_time > 0:
        speedup = oq_risk_time / total_time
        print(f'  Speedup (OQ/T-Risk): {speedup:.2f}x')
    print(f'')
    print(f'  Portfolio AAL:       {float(total_aal_per_asset.sum()):.2f}')
    print(f'  Total event losses:  {float(total_loss_per_event.sum()):.2f}')

    # Save results
    result = {
        'demo': demo_name,
        'n_assets': N,
        'n_events': Q,
        'batch_size': batch_size,
        'n_batches': n_batches,
        'n_components': len(component_inputs),
        'load_source': load_source,
        'load_time_sec': round(load_time, 4),
        'compute_time_sec': round(compute_time, 4),
        'total_time_sec': round(total_time, 4),
        'oq_risk_time_sec': oq_risk_time,
        'speedup': round(oq_risk_time / total_time, 2) if total_time > 0 else None,
        'portfolio_aal': float(total_aal_per_asset.sum()),
    }

    suffix = '_npz' if load_source == 'npz' else '_csv'
    out_path = BENCH_DIR / f'batched_benchmark_result{suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResult saved to: {out_path}')


if __name__ == '__main__':
    main()
