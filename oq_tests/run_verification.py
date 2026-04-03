#!/usr/bin/env python3
import configparser
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OQ_ROOT = Path(__file__).resolve().parent.parent
BASE_OUT = Path(__file__).resolve().parent
OUT_DIR = BASE_OUT / 'outputs'
PLOT_DIR = BASE_OUT / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

from library_oq_import import (
    compute_uniform_event_rates,
    infer_event_id_column,
    infer_site_id_column,
    load_exposure_model,
    load_oq_vulnerability_xml,
    map_exposure_to_oq_sites,
)

EXAMPLES = [
    ('ScenarioRisk', 'job.ini'),
    ('EventBasedRisk', 'job.ini'),
    ('Reinsurance', 'job.ini'),
]

LOSS_COMPONENTS = ('structural', 'nonstructural')


def run_oq_example(example_dir: Path, job_file: str) -> int:
    cmd = ['oq', 'run', job_file, '-e', 'csv']
    proc = subprocess.run(cmd, cwd=example_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"OQ run failed for {example_dir.name}:\n{proc.stdout}\n{proc.stderr}")

    text = proc.stdout + '\n' + proc.stderr
    m = re.search(r'calc_(\d+)\.hdf5', text)
    if not m:
        raise RuntimeError(f"Cannot parse calc ID for {example_dir.name}\n{text}")
    return int(m.group(1))


def find_exported_file(example_dir: Path, calc_id: int, prefix: str) -> Path:
    candidates = sorted(example_dir.glob(f'{prefix}*_{calc_id}.csv'))
    if not candidates:
        raise FileNotFoundError(f"Missing export {prefix}*_{calc_id}.csv in {example_dir}")
    return candidates[0]


def get_loss_components_from_job(example_dir: Path, job_file: str):
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


def map_assets_sites_tolerant(exposure: pd.DataFrame, sitemesh: pd.DataFrame) -> pd.DataFrame:
    site_col = infer_site_id_column(sitemesh)

    def _nearest_assign(df: pd.DataFrame) -> pd.DataFrame:
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
        return _nearest_assign(mapped)
    except RuntimeError:
        exp = exposure.copy()
        sm = sitemesh.copy()
        exp['lon_r'] = exp['lon'].round(5)
        exp['lat_r'] = exp['lat'].round(5)
        sm['lon_r'] = sm['lon'].round(5)
        sm['lat_r'] = sm['lat'].round(5)
        mapped = exp.merge(sm[['lon_r', 'lat_r', site_col]], on=['lon_r', 'lat_r'], how='left')
        mapped = mapped.drop(columns=['lon_r', 'lat_r'])
        return _nearest_assign(mapped)


def select_oq_event_loss(rbe: pd.DataFrame, components):
    event_col = infer_event_id_column(rbe)
    if 'loss' not in rbe.columns:
        raise KeyError('risk_by_event export has no loss column')

    if 'loss_type' in rbe.columns:
        use = rbe[rbe['loss_type'].isin(list(components))].copy()
        if len(use) == 0:
            use = rbe.copy()
    else:
        use = rbe.copy()

    oq_event = use.groupby(event_col, as_index=False)['loss'].sum().rename(columns={event_col: 'event_id', 'loss': 'loss_oq'})
    return oq_event


def select_oq_asset_loss(avg_losses: pd.DataFrame, components):
    if 'asset_id' not in avg_losses.columns:
        raise KeyError('avg_losses export has no asset_id column')

    cols = [c for c in components if c in avg_losses.columns]
    if not cols:
        fallback = [c for c in avg_losses.columns if c.lower().startswith('structural')]
        cols = fallback[:1]
        if not cols:
            raise KeyError('No comparable loss columns found in avg_losses')

    out = avg_losses[['asset_id'] + cols].copy()
    out['loss_oq_asset'] = out[cols].sum(axis=1)
    return out[['asset_id', 'loss_oq_asset']]


def build_hazard_matrix_for_imt(gmf: pd.DataFrame, event_ids: np.ndarray, site_ids: list, imt: str) -> np.ndarray:
    event_col = infer_event_id_column(gmf)
    site_col = infer_site_id_column(gmf)
    gmv_col = f'gmv_{imt}'
    if gmv_col not in gmf.columns:
        raise KeyError(f'Missing GMV column {gmv_col} in GMF data')

    pivot = gmf.pivot(index=event_col, columns=site_col, values=gmv_col)
    pivot = pivot.reindex(index=event_ids, columns=site_ids).fillna(0.0)
    return pivot.to_numpy(dtype=np.float32).T


def compute_det_from_oq(example_name: str, example_dir: Path, calc_id: int, vuln_files: dict):
    events = pd.read_csv(find_exported_file(example_dir, calc_id, 'events'), comment='#')
    gmf = pd.read_csv(find_exported_file(example_dir, calc_id, 'gmf-data'), comment='#')
    sitemesh = pd.read_csv(find_exported_file(example_dir, calc_id, 'sitemesh'), comment='#')
    rbe = pd.read_csv(find_exported_file(example_dir, calc_id, 'risk_by_event'), comment='#')
    avg_losses = pd.read_csv(find_exported_file(example_dir, calc_id, 'avg_losses'), comment='#')

    exposure_csv = example_dir / 'exposure_model.csv'
    if not exposure_csv.exists():
        raise FileNotFoundError(f'{example_name}: missing exposure_model.csv')
    exposure = load_exposure_model(str(example_dir), filename='exposure_model.csv')

    asset_site = map_assets_sites_tolerant(exposure, sitemesh)
    site_col = infer_site_id_column(asset_site)
    asset_site = asset_site[asset_site[site_col].notna()].copy()
    if asset_site.empty:
        raise RuntimeError(f'{example_name}: no assets mapped to OQ sitemesh')
    event_col = infer_event_id_column(events)

    event_ids = events[event_col].astype(int).values
    site_ids = asset_site[site_col].tolist()
    gmv_cols = [c for c in gmf.columns if c.startswith('gmv_')]
    if not gmv_cols:
        raise KeyError(f'{example_name}: no gmv_* columns in GMF data')
    first_imt = gmv_cols[0].replace('gmv_', '')
    h0 = build_hazard_matrix_for_imt(gmf, event_ids, site_ids, first_imt)

    det_total = np.zeros_like(h0, dtype=np.float64)
    used_components = []
    h_cache = {}

    for comp, vf in vuln_files.items():
        if comp not in asset_site.columns:
            continue
        values = asset_site[comp].astype(float).values
        if 'number' in asset_site.columns:
            values = values * asset_site['number'].astype(float).values
        if np.all(values == 0):
            continue

        vuln_df = load_oq_vulnerability_xml(str(example_dir), filename=vf)
        component_loss = np.zeros_like(h0, dtype=np.float64)
        tax = asset_site['taxonomy'].astype(str).values
        for i in range(h0.shape[0]):
            vtax = vuln_df[vuln_df['vulnerability_id'] == tax[i]].sort_values('point_index')
            if vtax.empty:
                continue
            imt_values = vtax['imt'].dropna().unique().tolist()
            imt = imt_values[0] if imt_values else 'PGA'
            if imt not in h_cache:
                h_cache[imt] = build_hazard_matrix_for_imt(gmf, event_ids, site_ids, imt)
            h_imt = h_cache[imt]

            x_grid = vtax['iml'].to_numpy(dtype=np.float32)
            curve = vtax['mean_lr'].to_numpy(dtype=np.float32)
            mdr = np.interp(h_imt[i], x_grid, curve, left=curve[0], right=curve[-1])
            component_loss[i] = values[i] * mdr

        det_total += component_loss
        used_components.append(comp)

    if not used_components:
        raise RuntimeError(f'{example_name}: no matching loss components found')

    lambdas = compute_uniform_event_rates(events)
    det_asset = (det_total * lambdas[np.newaxis, :]).sum(axis=1)
    det_event = det_total.sum(axis=0)

    det_asset_df = pd.DataFrame({'asset_id': asset_site['id'].astype(str).values, 'loss_det_asset': det_asset})
    det_event_df = pd.DataFrame({'event_id': event_ids, 'loss_det': det_event})

    oq_event_df = select_oq_event_loss(rbe, used_components)
    oq_asset_df = select_oq_asset_loss(avg_losses, used_components)

    oq_event_aligned = det_event_df[['event_id']].merge(oq_event_df, on='event_id', how='left').fillna(0.0)
    oq_port = float(np.sum(oq_event_aligned['loss_oq'].values * lambdas))
    oq_asset_sum = float(oq_asset_df['loss_oq_asset'].sum())
    if oq_asset_sum > 0:
        oq_asset_df['loss_oq_asset'] = oq_asset_df['loss_oq_asset'] * (oq_port / oq_asset_sum)

    evt_cmp = det_event_df.merge(oq_event_df, on='event_id', how='left').fillna(0.0)
    ast_cmp = det_asset_df.merge(oq_asset_df, on='asset_id', how='left').fillna(0.0)

    evt_cmp['abs_diff'] = (evt_cmp['loss_det'] - evt_cmp['loss_oq']).abs()
    ast_cmp['abs_diff'] = (ast_cmp['loss_det_asset'] - ast_cmp['loss_oq_asset']).abs()
    ast_cmp['ratio'] = np.where(ast_cmp['loss_oq_asset'] > 0, ast_cmp['loss_det_asset'] / ast_cmp['loss_oq_asset'], np.nan)

    mae = float(np.mean(np.abs(evt_cmp['loss_det'] - evt_cmp['loss_oq'])))
    rmse = float(np.sqrt(np.mean((evt_cmp['loss_det'] - evt_cmp['loss_oq']) ** 2)))
    corr = float(np.corrcoef(evt_cmp['loss_det'].values, evt_cmp['loss_oq'].values)[0, 1]) if len(evt_cmp) > 1 else np.nan
    ss_res = float(np.sum((evt_cmp['loss_det'] - evt_cmp['loss_oq']) ** 2))
    ss_tot = float(np.sum((evt_cmp['loss_oq'] - evt_cmp['loss_oq'].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    det_port = float(np.sum(det_event * lambdas))

    summary = {
        'example': example_name,
        'calc_id': calc_id,
        'n_assets': int(h0.shape[0]),
        'n_events': int(h0.shape[1]),
        'components_used': '+'.join(used_components),
        'portfolio_det': det_port,
        'portfolio_oq': oq_port,
        'portfolio_ratio': (det_port / oq_port) if oq_port > 0 else np.nan,
        'event_mae': mae,
        'event_rmse': rmse,
        'event_corr': corr,
        'event_r2': r2,
        'asset_ratio_min': float(np.nanmin(ast_cmp['ratio'])),
        'asset_ratio_mean': float(np.nanmean(ast_cmp['ratio'])),
        'asset_ratio_max': float(np.nanmax(ast_cmp['ratio'])),
    }

    return summary, ast_cmp, evt_cmp


def make_plots(example_name: str, ast_cmp: pd.DataFrame, evt_cmp: pd.DataFrame):
    plot_assets = ast_cmp.sort_values('loss_oq_asset', ascending=False).head(40).copy()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(plot_assets))
    w = 0.38
    ax.bar(x - w / 2, plot_assets['loss_oq_asset'], w, label='OQ (stochastic)', color='steelblue')
    ax.bar(x + w / 2, plot_assets['loss_det_asset'], w, label='Deterministic vulnerability', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_assets['asset_id'], rotation=90, fontsize=6)
    ax.set_ylabel('Loss metric (annualized/average)')
    ax.set_title(f'{example_name}: Asset comparison (top 40 by OQ loss)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f'{example_name}_asset_compare.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    eps = 1.0
    ax.scatter(evt_cmp['loss_oq'] + eps, evt_cmp['loss_det'] + eps, s=8, alpha=0.35, color='darkgreen')
    lo = float(min((evt_cmp['loss_oq'] + eps).min(), (evt_cmp['loss_det'] + eps).min()))
    hi = float(max((evt_cmp['loss_oq'] + eps).max(), (evt_cmp['loss_det'] + eps).max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('OQ event loss + 1')
    ax.set_ylabel('Deterministic event loss + 1')
    ax.set_title(f'{example_name}: Event loss scatter')
    ax.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f'{example_name}_event_scatter.png', dpi=220)
    plt.close(fig)


def write_report_tex(summaries: pd.DataFrame):
    lines = []
    lines.append('\\documentclass[11pt,a4paper]{article}')
    lines.append('\\usepackage[margin=1in]{geometry}')
    lines.append('\\usepackage{booktabs}')
    lines.append('\\usepackage{graphicx}')
    lines.append('\\usepackage{float}')
    lines.append('\\usepackage{hyperref}')
    lines.append('\\title{Multi-Example Deterministic vs Stochastic Vulnerability Verification}')
    lines.append('\\author{Automated verification}')
    lines.append('\\date{\\today}')
    lines.append('\\begin{document}')
    lines.append('\\maketitle')
    lines.append('\\tableofcontents')
    lines.append('\\section{Objective}')
    lines.append('Run several OQ-engine risk examples end-to-end, apply deterministic mean vulnerability curves to the same hazard inputs, and compare the resulting losses against OQ stochastic vulnerability outputs.')
    lines.append('\\section{Summary Table}')
    lines.append('\\begin{table}[H]\\centering')
    lines.append('\\caption{Per-example verification summary}')
    lines.append('\\begin{tabular}{lrrrrrr}')
    lines.append('\\toprule')
    lines.append('Example & Ratio & Corr & $R^2$ & Asset min & Asset mean & Asset max \\\\')
    lines.append('\\midrule')
    for _, r in summaries.iterrows():
        lines.append(f"{r['example']} & {r['portfolio_ratio']:.6f} & {r['event_corr']:.6f} & {r['event_r2']:.6f} & {r['asset_ratio_min']:.6f} & {r['asset_ratio_mean']:.6f} & {r['asset_ratio_max']:.6f} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    for _, r in summaries.iterrows():
        ex = r['example']
        lines.append(f'\\section{{{ex}}}')
        lines.append('\\begin{itemize}')
        lines.append(f"\\item Calculation ID: {int(r['calc_id'])}")
        lines.append(f"\\item Components used: \\texttt{{{r['components_used']}}}")
        lines.append(f"\\item Assets: {int(r['n_assets'])}, Events: {int(r['n_events'])}")
        lines.append(f"\\item Portfolio ratio (deterministic/OQ): {r['portfolio_ratio']:.6f}")
        lines.append(f"\\item Event MAE: {r['event_mae']:.4f}, RMSE: {r['event_rmse']:.4f}, Corr: {r['event_corr']:.6f}, $R^2$: {r['event_r2']:.6f}")
        lines.append('\\end{itemize}')
        lines.append('\\begin{figure}[H]\\centering')
        lines.append(f"\\includegraphics[width=0.92\\textwidth]{{plots/{ex}_asset_compare.png}}")
        lines.append(f"\\caption{{{ex}: asset-level comparison}}")
        lines.append('\\end{figure}')
        lines.append('\\begin{figure}[H]\\centering')
        lines.append(f"\\includegraphics[width=0.70\\textwidth]{{plots/{ex}_event_scatter.png}}")
        lines.append(f"\\caption{{{ex}: event-level scatter (log-log)}}")
        lines.append('\\end{figure}')

    lines.append('\\end{document}')
    (BASE_OUT / 'verification_report.tex').write_text('\n'.join(lines), encoding='utf-8')


def main():
    summaries = []

    for name, job in EXAMPLES:
        example_dir = OQ_ROOT / 'demos' / 'risk' / name
        print(f'\n=== Running {name} ({job}) ===')
        calc_id = run_oq_example(example_dir, job)
        print(f'calc_id={calc_id}')

        out_sub = OUT_DIR / name
        out_sub.mkdir(parents=True, exist_ok=True)

        for csv_file in example_dir.glob(f'*_{calc_id}.csv'):
            shutil.copy2(csv_file, out_sub / csv_file.name)

        vuln_files = get_loss_components_from_job(example_dir, job)
        summary, ast_cmp, evt_cmp = compute_det_from_oq(name, example_dir, calc_id, vuln_files)

        pd.DataFrame([summary]).to_csv(out_sub / 'summary.csv', index=False)
        ast_cmp.to_csv(out_sub / 'asset_comparison.csv', index=False)
        evt_cmp.to_csv(out_sub / 'event_comparison.csv', index=False)

        make_plots(name, ast_cmp, evt_cmp)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / 'multi_example_summary.csv', index=False)
    write_report_tex(summary_df)

    print('\nCompleted multi-example verification.')
    print(summary_df[['example', 'portfolio_ratio', 'event_corr', 'event_r2']])


if __name__ == '__main__':
    main()
