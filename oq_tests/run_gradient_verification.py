#!/usr/bin/env python3
"""
Gradient Verification: T-Risk AD vs OQ-based Finite Differences.

Computes:
  A) ∂(AAL)/∂C (vulnerability gradient, K×M matrix)
  B) ∂(AAL)/∂v (exposure gradient, N-vector)

Two verification methods:
  1. T-Risk automatic differentiation  (single GradientTape pass)
  2. Central finite differences (numpy reimplementation of interpolation)

Uses the ScenarioRisk demo (K=6 typologies, M=8 IML points, Q=100 events,
N≈9000 assets).

Requires: T-Risk environment (tensorial_engine venv with tensorflow).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
    TRISK_DIR = OQ_ROOT.parent / 'T-Hazard' / 'T-Risk'
if TRISK_DIR.exists():
    sys.path.insert(0, str(TRISK_DIR))
else:
    print(f'ERROR: T-Risk directory not found at {TRISK_DIR}')
    sys.exit(1)

from tensor_engine import TensorialRiskEngine  # noqa: E402

# -- OQ-equivalent interpolation (no OQ import needed) --
from scipy.interpolate import interp1d  # noqa: E402

# -- Local library --
from library_oq_import import (  # noqa: E402
    compute_uniform_event_rates,
    infer_event_id_column,
    infer_site_id_column,
    load_exposure_model,
    load_oq_vulnerability_xml,
    vulnerability_to_trisk_arrays,
)

DEMO_NAME = 'ScenarioRisk'
DEMO_DIR_NAME = 'ScenarioRisk'


# ── Helpers ─────────────────────────────────────────────────────────────

def find_csv(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f'{prefix}*.csv'))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv in {directory}")
    return candidates[0]


def map_assets_sites_tolerant(exposure, sitemesh):
    from library_oq_import import map_exposure_to_oq_sites
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


def compute_aal_oq(v, u, H, lambdas, taxonomy_order, x_grid, C_matrix):
    """
    Compute portfolio AAL using OQ-equivalent interpolation.

    Replicates OQ's VulnerabilityFunction.interpolate():
     - scipy.interpolate.interp1d (linear) on (x_grid, C[k,:])
     - GMVs clipped to max(x_grid)
     - Below min(x_grid) → loss ratio = 0
    Then: AAL = Σ_q λ_q × Σ_i v[i] × lr(H[i,q])
    """
    N, Q = H.shape
    K, M = C_matrix.shape

    # Build interpolators per typology (same as OQ's _mlr_i1d)
    interpolators = []
    for k in range(K):
        f = interp1d(x_grid.astype(np.float64),
                     C_matrix[k].astype(np.float64),
                     kind='linear', fill_value='extrapolate')
        interpolators.append(f)

    # Compute loss ratios for all (N, Q)
    loss_ratios = np.zeros((N, Q), dtype=np.float64)
    for i in range(N):
        k = u[i]
        gmvs = H[i, :].astype(np.float64)
        # OQ clipping: cap at max IML
        gmvs_clipped = np.minimum(gmvs, x_grid[-1])
        # Only interpolate where >= min IML
        ok = gmvs_clipped >= x_grid[0]
        if ok.any():
            loss_ratios[i, ok] = interpolators[k](gmvs_clipped[ok])

    # J[i,q] = v[i] * lr[i,q]
    J = v[:, None].astype(np.float64) * loss_ratios

    # AAL = Σ_q λ_q * Σ_i J[i,q]
    loss_per_event = J.sum(axis=0)  # (Q,)
    aal = np.dot(lambdas.astype(np.float64), loss_per_event)
    return aal


def compute_aal_trisk_numpy(v, u, H, lambdas, x_grid, C_matrix):
    """
    Compute portfolio AAL matching T-Risk's exact interpolation logic.

    Replicates TensorialRiskEngine's probabilistic_loss_matrix():
     - searchsorted to find grid index, clip to [0, M-2]
     - Linear interpolation using alpha weight
     - NO clipping of GMVs to max IML (extrapolates via last segment)
    This is the numpy equivalent of T-Risk's TensorFlow computation.
    """
    N, Q = H.shape
    M = len(x_grid)
    x_grid_64 = x_grid.astype(np.float64)
    C_64 = C_matrix.astype(np.float64)
    H_64 = H.astype(np.float64)

    # Vectorized: process all (N, Q) at once
    H_flat = H_64.ravel()  # (N*Q,)
    idx = np.searchsorted(x_grid_64, H_flat, side='right') - 1
    idx = np.clip(idx, 0, M - 2)

    x_lower = x_grid_64[idx]
    x_upper = x_grid_64[idx + 1]
    alpha = (H_flat - x_lower) / (x_upper - x_lower + 1e-8)

    u_repeated = np.repeat(u, Q)  # (N*Q,)

    c_lower = C_64[u_repeated, idx]
    c_upper = C_64[u_repeated, idx + 1]

    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_matrix = mdr_flat.reshape(N, Q)

    # J[i,q] = v[i] * mdr[i,q]
    J = v[:, None].astype(np.float64) * mdr_matrix

    # AAL = Σ_q λ_q * Σ_i J[i,q]
    loss_per_event = J.sum(axis=0)
    aal = np.dot(lambdas.astype(np.float64), loss_per_event)
    return aal


def fd_gradient_vulnerability(v, u, H, lambdas, taxonomy_order, x_grid,
                              C_matrix, delta=1e-4, method='trisk'):
    """Central finite-difference gradient ∂AAL/∂C.

    method='trisk': use T-Risk-native numpy interpolation (for AD verification)
    method='oq':    use OQ-equivalent interpolation (for cross-engine comparison)
    """
    K, M = C_matrix.shape
    grad = np.zeros((K, M), dtype=np.float64)

    if method == 'trisk':
        aal_fn = lambda C: compute_aal_trisk_numpy(v, u, H, lambdas, x_grid, C)
    else:
        aal_fn = lambda C: compute_aal_oq(v, u, H, lambdas, taxonomy_order,
                                           x_grid, C)

    for k in range(K):
        for m in range(M):
            C_plus = C_matrix.copy()
            C_minus = C_matrix.copy()
            C_plus[k, m] += delta
            C_minus[k, m] -= delta

            aal_plus = aal_fn(C_plus)
            aal_minus = aal_fn(C_minus)
            grad[k, m] = (aal_plus - aal_minus) / (2.0 * delta)

    return grad


def convergence_sweep(v, u, H, lambdas, taxonomy_order, x_grid, C_matrix,
                      ad_grad, deltas=(1e-3, 1e-4, 1e-5), method='trisk'):
    """Run FD at multiple delta values to check convergence toward AD."""
    results = {}
    for d in deltas:
        fd_grad = fd_gradient_vulnerability(v, u, H, lambdas, taxonomy_order,
                                            x_grid, C_matrix, delta=d,
                                            method=method)
        abs_diff = np.abs(fd_grad - ad_grad)
        denom = np.maximum(np.abs(ad_grad), 1e-10)
        rel_err = abs_diff / denom
        sig = np.abs(ad_grad) > 1e-6 * np.abs(ad_grad).max()
        results[d] = {
            'fd_grad': fd_grad,
            'max_rel_err': float(rel_err[sig].max()) if sig.any() else 0.0,
            'mean_rel_err': float(rel_err[sig].mean()) if sig.any() else 0.0,
            'median_rel_err': float(np.median(rel_err[sig])) if sig.any() else 0.0,
        }
    return results


# ── Exposure gradient helpers ───────────────────────────────────────────

def compute_mdr_matrix_trisk_numpy(u, H, x_grid, C_matrix):
    """Compute MDR matrix (N, Q) using T-Risk-native interpolation.

    Returns the mean damage ratio for each (asset, event) pair.
    """
    N, Q = H.shape
    M = len(x_grid)
    x_grid_64 = x_grid.astype(np.float64)
    C_64 = C_matrix.astype(np.float64)
    H_64 = H.astype(np.float64)

    H_flat = H_64.ravel()
    idx = np.searchsorted(x_grid_64, H_flat, side='right') - 1
    idx = np.clip(idx, 0, M - 2)

    x_lower = x_grid_64[idx]
    x_upper = x_grid_64[idx + 1]
    alpha = (H_flat - x_lower) / (x_upper - x_lower + 1e-8)

    u_repeated = np.repeat(u, Q)
    c_lower = C_64[u_repeated, idx]
    c_upper = C_64[u_repeated, idx + 1]

    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    return mdr_flat.reshape(N, Q)


def analytical_gradient_exposure(u, H, lambdas, x_grid, C_matrix):
    """Analytical exposure gradient: ∂AAL/∂v_i = Σ_q λ_q × MDR[i,q].

    Since v enters AAL linearly (J = v * MDR), the gradient is exact.
    """
    mdr = compute_mdr_matrix_trisk_numpy(u, H, x_grid, C_matrix)
    lambdas_64 = lambdas.astype(np.float64)
    return mdr @ lambdas_64  # (N,)


def fd_gradient_exposure(v, u, H, lambdas, x_grid, C_matrix,
                         asset_indices, delta_rel=1e-6):
    """Central FD gradient ∂AAL/∂v for a subset of assets.

    Uses relative perturbation: δ_i = delta_rel × max(|v[i]|, 1.0) to
    avoid catastrophic cancellation on large exposure values.
    Only perturbs assets in asset_indices to keep runtime reasonable.
    Returns: grad array of shape (len(asset_indices),)
    """
    grad = np.zeros(len(asset_indices), dtype=np.float64)
    for j, i in enumerate(asset_indices):
        h = delta_rel * max(abs(float(v[i])), 1.0)
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += h
        v_minus[i] -= h

        aal_plus = compute_aal_trisk_numpy(v_plus, u, H, lambdas, x_grid,
                                           C_matrix)
        aal_minus = compute_aal_trisk_numpy(v_minus, u, H, lambdas, x_grid,
                                            C_matrix)
        grad[j] = (aal_plus - aal_minus) / (2.0 * h)
    return grad


# ── Plotting ────────────────────────────────────────────────────────────

def plot_gradient_heatmaps(ad_grad, fd_grad, taxonomy_order, x_grid):
    """Side-by-side heatmaps of AD vs FD gradients."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(np.abs(ad_grad).max(), np.abs(fd_grad).max())
    vmin = -vmax

    im0 = axes[0].imshow(ad_grad, aspect='auto', cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
    axes[0].set_title('T-Risk AD: ∂AAL/∂C')
    axes[0].set_xlabel('IML index (m)')
    axes[0].set_ylabel('Typology (k)')
    axes[0].set_yticks(range(len(taxonomy_order)))
    axes[0].set_yticklabels(taxonomy_order, fontsize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fd_grad, aspect='auto', cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
    axes[1].set_title('FD: ∂AAL/∂C')
    axes[1].set_xlabel('IML index (m)')
    axes[1].set_ylabel('Typology (k)')
    axes[1].set_yticks(range(len(taxonomy_order)))
    axes[1].set_yticklabels(taxonomy_order, fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Relative error heatmap
    denom = np.maximum(np.abs(ad_grad), 1e-10)
    rel_err = np.abs(fd_grad - ad_grad) / denom
    # Mask where AD is negligible
    mask = np.abs(ad_grad) < 1e-6 * np.abs(ad_grad).max()
    rel_err_display = rel_err.copy()
    rel_err_display[mask] = np.nan

    im2 = axes[2].imshow(rel_err_display * 100, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=5)
    axes[2].set_title('Relative Error (%)')
    axes[2].set_xlabel('IML index (m)')
    axes[2].set_ylabel('Typology (k)')
    axes[2].set_yticks(range(len(taxonomy_order)))
    axes[2].set_yticklabels(taxonomy_order, fontsize=8)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    path = PLOT_DIR / 'gradient_heatmaps.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'  Heatmaps saved: {path}')


def plot_gradient_scatter(ad_grad, fd_grad):
    """Scatter plot: FD vs AD gradient values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ad_flat = ad_grad.flatten()
    fd_flat = fd_grad.flatten()

    ax.scatter(ad_flat, fd_flat, alpha=0.7, edgecolors='k', linewidths=0.5,
               s=60, c='steelblue')

    lim = max(np.abs(ad_flat).max(), np.abs(fd_flat).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='y = x')

    ax.set_xlabel('T-Risk AD gradient')
    ax.set_ylabel('FD gradient')
    ax.set_title('∂AAL/∂C: T-Risk AD vs Finite-Difference')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    path = PLOT_DIR / 'gradient_scatter.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'  Scatter saved: {path}')


def plot_convergence(conv_results, ad_grad):
    """Plot FD convergence toward AD as delta decreases."""
    deltas = sorted(conv_results.keys())
    max_errs = [conv_results[d]['max_rel_err'] * 100 for d in deltas]
    mean_errs = [conv_results[d]['mean_rel_err'] * 100 for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deltas, max_errs, 'o-', label='Max relative error', color='tomato',
            markersize=8)
    ax.plot(deltas, mean_errs, 's-', label='Mean relative error', color='steelblue',
            markersize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Perturbation δ')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('FD Convergence: Relative Error vs δ')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    path = PLOT_DIR / 'gradient_convergence.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'  Convergence saved: {path}')


def plot_exposure_gradient(ad_grad_v, analytical_grad_v, fd_grad_v,
                           fd_indices, taxonomy_labels):
    """Plots for exposure gradient verification."""
    # 1. AD vs Analytical scatter (all N assets)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.scatter(analytical_grad_v, ad_grad_v, alpha=0.3, s=15, c='steelblue',
               edgecolors='none')
    lim = max(np.abs(analytical_grad_v).max(), np.abs(ad_grad_v).max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='y = x')
    ax.set_xlabel('Analytical ∂AAL/∂v')
    ax.set_ylabel('T-Risk AD ∂AAL/∂v')
    ax.set_title(f'Exposure Gradient: AD vs Analytical (N={len(ad_grad_v)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. AD vs FD scatter (sampled assets)
    ax = axes[1]
    ad_sampled = ad_grad_v[fd_indices]
    ax.scatter(ad_sampled, fd_grad_v, alpha=0.7, s=40, c='darkorange',
               edgecolors='k', linewidths=0.3)
    lim = max(np.abs(ad_sampled).max(), np.abs(fd_grad_v).max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='y = x')
    ax.set_xlabel('T-Risk AD ∂AAL/∂v')
    ax.set_ylabel('FD ∂AAL/∂v')
    ax.set_title(f'Exposure Gradient: AD vs FD (sample={len(fd_indices)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Gradient distribution by typology
    ax = axes[2]
    unique_labels = sorted(set(taxonomy_labels))
    data_per_tax = [ad_grad_v[np.array(taxonomy_labels) == t]
                    for t in unique_labels]
    short_labels = [t[:12] for t in unique_labels]
    bp = ax.boxplot(data_per_tax, labels=short_labels, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('∂AAL/∂v')
    ax.set_title('Exposure Gradient by Typology')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = PLOT_DIR / 'exposure_gradient.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'  Exposure gradient plots saved: {path}')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import configparser

    print('=' * 60)
    print('Gradient Verification: T-Risk AD vs OQ Finite-Difference')
    print('=' * 60)

    example_dir = OQ_ROOT / 'demos' / 'risk' / DEMO_DIR_NAME
    bench_out = BENCH_DIR / DEMO_NAME

    if not bench_out.exists():
        print(f'ERROR: {bench_out} not found. Run run_benchmark_oq.py first.')
        sys.exit(1)

    # ── 1. Load data ────────────────────────────────────────────────────
    print('\n[1] Loading ScenarioRisk data ...')
    import pandas as pd
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

    # Read vulnerability file from job.ini
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(example_dir / 'job.ini')
    vuln_file = parser.get('vulnerability', 'structural_vulnerability_file',
                           fallback=None)
    if not vuln_file:
        print('ERROR: structural_vulnerability_file not found in job.ini')
        sys.exit(1)

    vuln_df = load_oq_vulnerability_xml(str(example_dir), filename=vuln_file)
    x_grid, C_matrix, tax_to_idx, _ = vulnerability_to_trisk_arrays(vuln_df)
    taxonomy_order = list(tax_to_idx.keys())

    # Build hazard matrix H (N x Q) — structural uses PGA
    ecol = infer_event_id_column(gmf)
    scol = infer_site_id_column(gmf)
    # Determine IMT column
    gmv_cols = [c for c in gmf.columns if c.startswith('gmv_')]
    imt_col = gmv_cols[0] if gmv_cols else 'gmv_PGA'
    pivot = gmf.pivot(index=ecol, columns=scol, values=imt_col)
    pivot = pivot.reindex(index=event_ids, columns=site_ids).fillna(0.0)
    H = pivot.to_numpy(dtype=np.float32).T  # (N, Q)

    # Build exposure vector v and typology index u
    v_values = asset_site['structural'].astype(float).values
    if 'number' in asset_site.columns:
        v_values = v_values * asset_site['number'].astype(float).values
    v = v_values.astype(np.float32)
    u = np.array([tax_to_idx.get(t, 0) for t in asset_site['taxonomy'].astype(str)],
                 dtype=np.int32)

    lambdas = compute_uniform_event_rates(events).astype(np.float32)

    K, M = C_matrix.shape
    N, Q = H.shape
    print(f'  N={N} assets, Q={Q} events, K={K} typologies, M={M} IML points')
    print(f'  Taxonomies: {taxonomy_order}')
    print(f'  x_grid: {x_grid}')
    print(f'  Total FD perturbations: {K*M*2}')

    # ── 2. T-Risk AD gradient ──────────────────────────────────────────
    print('\n[2] Computing T-Risk AD gradient ...')
    t0 = time.perf_counter()
    engine = TensorialRiskEngine(v=v, u=u, C=C_matrix, x_grid=x_grid,
                                  H=H, lambdas=lambdas)
    grad_C_ad, metrics = engine.gradient_wrt_vulnerability()
    ad_grad = grad_C_ad.numpy().astype(np.float64)
    aal_trisk = float(metrics['aal_portfolio'].numpy())
    t_ad = time.perf_counter() - t0
    print(f'  AAL (T-Risk): {aal_trisk:,.2f}')
    print(f'  AD time: {t_ad:.3f}s')
    print(f'  AD grad shape: {ad_grad.shape}')
    print(f'  AD grad range: [{ad_grad.min():.4e}, {ad_grad.max():.4e}]')

    # ── 3. T-Risk-native FD (primary verification) ────────────────────
    print('\n[3] Computing T-Risk-native FD gradient (δ=1e-4) ...')
    t0 = time.perf_counter()

    aal_trisk_np = compute_aal_trisk_numpy(v, u, H, lambdas, x_grid, C_matrix)
    print(f'  AAL (T-Risk numpy): {aal_trisk_np:,.2f}')
    print(f'  AAL diff (TF vs numpy): {abs(aal_trisk - aal_trisk_np):.4e}')

    fd_grad_trisk = fd_gradient_vulnerability(v, u, H, lambdas, taxonomy_order,
                                               x_grid, C_matrix, delta=1e-4,
                                               method='trisk')
    t_fd_trisk = time.perf_counter() - t0
    print(f'  FD time: {t_fd_trisk:.3f}s')
    print(f'  FD grad range: [{fd_grad_trisk.min():.4e}, {fd_grad_trisk.max():.4e}]')

    # ── 4. Primary comparison: AD vs T-Risk FD ─────────────────────────
    print('\n[4] Comparing AD vs T-Risk-native FD ...')

    def compute_errors(ad, fd, label):
        abs_diff = np.abs(fd - ad)
        denom = np.maximum(np.abs(ad), 1e-10)
        rel_err = abs_diff / denom
        sig_mask = np.abs(ad) > 1e-3 * np.abs(ad).max()
        n_sig = int(sig_mask.sum())
        if n_sig > 0:
            max_rel = float(rel_err[sig_mask].max())
            mean_rel = float(rel_err[sig_mask].mean())
            median_rel = float(np.median(rel_err[sig_mask]))
        else:
            max_rel = mean_rel = median_rel = 0.0
        passed = max_rel < 0.01
        print(f'  [{label}] Significant elements: {n_sig}/{K*M}')
        print(f'  [{label}] Max relative error:    {max_rel*100:.4f}%')
        print(f'  [{label}] Mean relative error:   {mean_rel*100:.4f}%')
        print(f'  [{label}] Median relative error: {median_rel*100:.4f}%')
        print(f'  [{label}] RESULT: {"PASS ✓" if passed else "FAIL ✗"}')
        return rel_err, sig_mask, n_sig, max_rel, mean_rel, median_rel, passed

    rel_err_trisk, sig_mask, n_sig, max_rel_trisk, mean_rel_trisk, \
        median_rel_trisk, passed_trisk = compute_errors(
            ad_grad, fd_grad_trisk, 'T-Risk FD')

    # Per-typology for T-Risk FD
    print('\n  Per-typology max relative error (T-Risk FD):')
    per_typology_trisk = {}
    for k, tax in enumerate(taxonomy_order):
        row_sig = sig_mask[k]
        if row_sig.any():
            err_k = float(rel_err_trisk[k][row_sig].max())
        else:
            err_k = 0.0
        per_typology_trisk[tax] = err_k
        print(f'    {tax:30s}: {err_k*100:.4f}%')

    # ── 5. OQ FD (cross-engine comparison) ─────────────────────────────
    print('\n[5] Computing OQ-style FD gradient (δ=1e-4) ...')
    t0 = time.perf_counter()
    aal_oq = compute_aal_oq(v, u, H, lambdas, taxonomy_order, x_grid, C_matrix)
    print(f'  AAL (OQ): {aal_oq:,.2f}')
    print(f'  AAL diff (T-Risk vs OQ): {abs(aal_trisk - aal_oq):.4e}')

    fd_grad_oq = fd_gradient_vulnerability(v, u, H, lambdas, taxonomy_order,
                                            x_grid, C_matrix, delta=1e-4,
                                            method='oq')
    t_fd_oq = time.perf_counter() - t0
    print(f'  FD time: {t_fd_oq:.3f}s')

    _, _, _, max_rel_oq, mean_rel_oq, median_rel_oq, passed_oq = \
        compute_errors(ad_grad, fd_grad_oq, 'OQ FD')

    # Identify boundary elements causing OQ divergence
    oq_abs_diff = np.abs(fd_grad_oq - ad_grad)
    oq_denom = np.maximum(np.abs(ad_grad), 1e-10)
    oq_rel_err = oq_abs_diff / oq_denom
    boundary_mask = oq_rel_err > 0.05  # >5% error
    n_boundary = int(boundary_mask.sum())
    if n_boundary > 0:
        print(f'\n  Boundary divergence: {n_boundary} elements with >5% error')
        for k in range(K):
            for m in range(M):
                if boundary_mask[k, m]:
                    print(f'    C[{taxonomy_order[k]}, IML={x_grid[m]:.4f}]: '
                          f'AD={ad_grad[k,m]:.4e}, OQ_FD={fd_grad_oq[k,m]:.4e}, '
                          f'err={oq_rel_err[k,m]*100:.1f}%')
        print('  Cause: OQ clips GMVs to max(IML); T-Risk extrapolates past it.')

    # ── 6. Convergence sweep (T-Risk FD) ───────────────────────────────
    print('\n[6] Convergence sweep (δ = 1e-3, 1e-4, 1e-5) ...')
    t0 = time.perf_counter()
    conv = convergence_sweep(v, u, H, lambdas, taxonomy_order, x_grid,
                              C_matrix, ad_grad, method='trisk')
    t_conv = time.perf_counter() - t0
    print(f'  Sweep time: {t_conv:.3f}s')
    for d in sorted(conv.keys()):
        r = conv[d]
        print(f'  δ={d:.0e}: max_err={r["max_rel_err"]*100:.4f}%, '
              f'mean_err={r["mean_rel_err"]*100:.4f}%')

    # ── 7. Plots ───────────────────────────────────────────────────────
    print('\n[7] Generating plots ...')
    # Primary plots: AD vs T-Risk FD
    plot_gradient_heatmaps(ad_grad, fd_grad_trisk, taxonomy_order, x_grid)
    plot_gradient_scatter(ad_grad, fd_grad_trisk)
    plot_convergence(conv, ad_grad)

    # Secondary: AD vs OQ FD scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ad_flat = ad_grad.flatten()
    oq_flat = fd_grad_oq.flatten()
    ax.scatter(ad_flat, oq_flat, alpha=0.7, edgecolors='k', linewidths=0.5,
               s=60, c='darkorange', label='OQ FD')
    lim = max(np.abs(ad_flat).max(), np.abs(oq_flat).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='y = x')
    ax.set_xlabel('T-Risk AD gradient')
    ax.set_ylabel('OQ FD gradient')
    ax.set_title('∂AAL/∂C: T-Risk AD vs OQ FD (boundary divergence)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    path = PLOT_DIR / 'gradient_scatter_oq.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f'  OQ scatter saved: {path}')

    # ════════════════════════════════════════════════════════════════════
    # PART B: EXPOSURE GRADIENT  ∂AAL/∂v
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXPOSURE GRADIENT: ∂AAL/∂v')
    print('=' * 60)

    # ── B1. T-Risk AD exposure gradient ────────────────────────────────
    print('\n[B1] Computing T-Risk AD exposure gradient ...')
    t0 = time.perf_counter()
    grad_v_ad_tf, _ = engine.gradient_wrt_exposure()
    ad_grad_v = grad_v_ad_tf.numpy().astype(np.float64)
    t_ad_v = time.perf_counter() - t0
    print(f'  AD time: {t_ad_v:.3f}s')
    print(f'  AD grad shape: {ad_grad_v.shape}')
    print(f'  AD grad range: [{ad_grad_v.min():.6e}, {ad_grad_v.max():.6e}]')
    print(f'  Non-zero: {(ad_grad_v != 0).sum()}/{N}')

    # ── B2. Analytical gradient (exact for linear v) ──────────────────
    print('\n[B2] Computing analytical exposure gradient ...')
    t0 = time.perf_counter()
    analytical_v = analytical_gradient_exposure(u, H, lambdas, x_grid, C_matrix)
    t_analytical = time.perf_counter() - t0
    print(f'  Analytical time: {t_analytical:.3f}s')

    # Compare AD vs analytical
    sig_v = np.abs(ad_grad_v) > 1e-6 * np.abs(ad_grad_v).max()
    n_sig_v = int(sig_v.sum())
    denom_v = np.maximum(np.abs(analytical_v), 1e-15)
    rel_err_v_analytical = np.abs(ad_grad_v - analytical_v) / denom_v

    max_rel_v_ana = float(rel_err_v_analytical[sig_v].max()) if sig_v.any() else 0.0
    mean_rel_v_ana = float(rel_err_v_analytical[sig_v].mean()) if sig_v.any() else 0.0
    median_rel_v_ana = float(np.median(rel_err_v_analytical[sig_v])) if sig_v.any() else 0.0
    passed_v_ana = max_rel_v_ana < 0.01

    print(f'  Significant assets: {n_sig_v}/{N}')
    print(f'  Max relative error (AD vs analytical): {max_rel_v_ana*100:.6f}%')
    print(f'  Mean relative error:                   {mean_rel_v_ana*100:.6f}%')
    print(f'  Median relative error:                 {median_rel_v_ana*100:.6f}%')
    print(f'  RESULT: {"PASS ✓" if passed_v_ana else "FAIL ✗"}')

    # ── B3. FD verification on random sample ──────────────────────────
    N_SAMPLE = min(200, N)
    rng = np.random.default_rng(42)
    fd_sample_idx = rng.choice(N, size=N_SAMPLE, replace=False)
    fd_sample_idx.sort()

    print(f'\n[B3] Computing FD exposure gradient (sample={N_SAMPLE}, δ_rel=1e-6) ...')
    t0 = time.perf_counter()
    # Use float64 v for FD precision
    v_64 = v.astype(np.float64)
    fd_grad_v = fd_gradient_exposure(v_64, u, H, lambdas, x_grid, C_matrix,
                                     fd_sample_idx, delta_rel=1e-6)
    t_fd_v = time.perf_counter() - t0
    print(f'  FD time: {t_fd_v:.3f}s ({N_SAMPLE * 2} evaluations)')

    # Compare AD vs FD on sample
    ad_sampled = ad_grad_v[fd_sample_idx]
    sig_sample = np.abs(ad_sampled) > 1e-6 * np.abs(ad_grad_v).max()
    n_sig_sample = int(sig_sample.sum())
    denom_sample = np.maximum(np.abs(ad_sampled), 1e-15)
    rel_err_v_fd = np.abs(fd_grad_v - ad_sampled) / denom_sample

    max_rel_v_fd = float(rel_err_v_fd[sig_sample].max()) if sig_sample.any() else 0.0
    mean_rel_v_fd = float(rel_err_v_fd[sig_sample].mean()) if sig_sample.any() else 0.0
    median_rel_v_fd = float(np.median(rel_err_v_fd[sig_sample])) if sig_sample.any() else 0.0
    passed_v_fd = max_rel_v_fd < 0.01

    print(f'  Significant in sample: {n_sig_sample}/{N_SAMPLE}')
    print(f'  Max relative error (AD vs FD): {max_rel_v_fd*100:.6f}%')
    print(f'  Mean relative error:           {mean_rel_v_fd*100:.6f}%')
    print(f'  Median relative error:         {median_rel_v_fd*100:.6f}%')
    print(f'  RESULT: {"PASS ✓" if passed_v_fd else "FAIL ✗"}')

    # Per-typology exposure gradient stats
    taxonomy_labels = [taxonomy_order[u[i]] for i in range(N)]
    print('\n  Per-typology exposure gradient stats:')
    per_typology_exposure = {}
    for k, tax in enumerate(taxonomy_order):
        mask_k = (np.array([u[i] for i in range(N)]) == k)
        if mask_k.any():
            vals = ad_grad_v[mask_k]
            per_typology_exposure[tax] = {
                'count': int(mask_k.sum()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
            print(f'    {tax:30s}: n={mask_k.sum():5d}, '
                  f'mean={vals.mean():.6e}, std={vals.std():.6e}')

    # ── B4. Exposure gradient plots ───────────────────────────────────
    print('\n[B4] Generating exposure gradient plots ...')
    plot_exposure_gradient(ad_grad_v, analytical_v, fd_grad_v,
                           fd_sample_idx, taxonomy_labels)

    # ── 8. Save results ────────────────────────────────────────────────
    result = {
        'demo': DEMO_NAME,
        'N': int(N), 'Q': int(Q), 'K': int(K), 'M': int(M),
        'taxonomies': taxonomy_order,
        'x_grid': x_grid.tolist(),
        'aal_trisk': aal_trisk,
        'aal_trisk_numpy': aal_trisk_np,
        'aal_oq': aal_oq,
        'vulnerability_gradient': {
            'ad_time_sec': round(t_ad, 4),
            'fd_trisk_time_sec': round(t_fd_trisk, 4),
            'fd_oq_time_sec': round(t_fd_oq, 4),
            'convergence_time_sec': round(t_conv, 4),
            'delta': 1e-4,
            'primary_verification': {
                'method': 'T-Risk native FD',
                'n_significant': int(n_sig),
                'max_rel_error': round(max_rel_trisk, 8),
                'mean_rel_error': round(mean_rel_trisk, 8),
                'median_rel_error': round(median_rel_trisk, 8),
                'passed': bool(passed_trisk),
                'per_typology_max_rel_error': {
                    t: round(e, 8) for t, e in per_typology_trisk.items()
                },
            },
            'cross_engine_comparison': {
                'method': 'OQ-style FD',
                'max_rel_error': round(max_rel_oq, 8),
                'mean_rel_error': round(mean_rel_oq, 8),
                'median_rel_error': round(median_rel_oq, 8),
                'passed': bool(passed_oq),
                'n_boundary_divergent': n_boundary,
                'note': 'OQ clips GMVs to max(IML); T-Risk extrapolates. '
                        'Divergence at last IML point is expected.',
            },
            'convergence': {
                str(d): {k2: round(v2, 8) for k2, v2 in r.items()
                         if k2 != 'fd_grad'}
                for d, r in conv.items()
            },
            'ad_grad': ad_grad.tolist(),
            'fd_grad_trisk': fd_grad_trisk.tolist(),
            'fd_grad_oq': fd_grad_oq.tolist(),
        },
        'exposure_gradient': {
            'ad_time_sec': round(t_ad_v, 4),
            'analytical_time_sec': round(t_analytical, 4),
            'fd_time_sec': round(t_fd_v, 4),
            'fd_sample_size': N_SAMPLE,
            'delta': 1e-4,
            'ad_vs_analytical': {
                'n_significant': int(n_sig_v),
                'max_rel_error': round(max_rel_v_ana, 8),
                'mean_rel_error': round(mean_rel_v_ana, 8),
                'median_rel_error': round(median_rel_v_ana, 8),
                'passed': bool(passed_v_ana),
            },
            'ad_vs_fd': {
                'n_significant_in_sample': int(n_sig_sample),
                'max_rel_error': round(max_rel_v_fd, 8),
                'mean_rel_error': round(mean_rel_v_fd, 8),
                'median_rel_error': round(median_rel_v_fd, 8),
                'passed': bool(passed_v_fd),
            },
            'per_typology_stats': per_typology_exposure,
            'ad_grad_v_summary': {
                'min': float(ad_grad_v.min()),
                'max': float(ad_grad_v.max()),
                'mean': float(ad_grad_v.mean()),
                'std': float(ad_grad_v.std()),
            },
        },
    }

    out_path = BASE_DIR / 'gradient_verification_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    # ── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  AAL (T-Risk TF):     {aal_trisk:>16,.2f}')
    print(f'  AAL (T-Risk numpy):  {aal_trisk_np:>16,.2f}')
    print(f'  AAL (OQ-style):      {aal_oq:>16,.2f}')
    print()
    print('  ── VULNERABILITY GRADIENT ∂AAL/∂C ──')
    print(f'  PRIMARY (AD vs T-Risk FD):')
    print(f'    Max rel. error:    {max_rel_trisk*100:>10.4f}%')
    print(f'    Verdict:           {"PASS ✓" if passed_trisk else "FAIL ✗"}')
    print(f'  CROSS-ENGINE (AD vs OQ FD):')
    print(f'    Max rel. error:    {max_rel_oq*100:>10.4f}%')
    print(f'    Boundary elements: {n_boundary}')
    print()
    print('  ── EXPOSURE GRADIENT ∂AAL/∂v ──')
    print(f'  AD vs ANALYTICAL:')
    print(f'    Max rel. error:    {max_rel_v_ana*100:>10.6f}%')
    print(f'    Verdict:           {"PASS ✓" if passed_v_ana else "FAIL ✗"}')
    print(f'  AD vs FD (sample={N_SAMPLE}):')
    print(f'    Max rel. error:    {max_rel_v_fd*100:>10.6f}%')
    print(f'    Verdict:           {"PASS ✓" if passed_v_fd else "FAIL ✗"}')
    print()
    print('  ── TIMINGS ──')
    print(f'  Vuln AD:             {t_ad:>10.3f}s')
    print(f'  Vuln FD (T-Risk):    {t_fd_trisk:>10.3f}s  ({K*M*2} evals)')
    print(f'  Vuln FD (OQ):        {t_fd_oq:>10.3f}s')
    print(f'  Expo AD:             {t_ad_v:>10.3f}s')
    print(f'  Expo Analytical:     {t_analytical:>10.3f}s')
    print(f'  Expo FD (sample):    {t_fd_v:>10.3f}s  ({N_SAMPLE*2} evals)')
    print(f'  AD speedup (vuln):   {t_fd_trisk/t_ad:>10.1f}×')
    print('=' * 60)

    passed_all = passed_trisk and passed_v_ana and passed_v_fd
    return 0 if passed_all else 1


if __name__ == '__main__':
    sys.exit(main())
