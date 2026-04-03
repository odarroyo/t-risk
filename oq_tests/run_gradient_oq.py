#!/usr/bin/env python3
"""
OQ Exposure Gradient via Finite Differences.

Computes ∂(AAL)/∂v for a sample of assets by perturbing the exposure
CSV and re-running OQ ScenarioRisk.

Requires: OQ environment (oq-env).
Run AFTER run_gradient_verification.py (which produces T-Risk AD results).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

OQ_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = OQ_ROOT / 'demos' / 'risk' / 'ScenarioRisk'

N_SAMPLE = 50  # number of assets to perturb
DELTA_REL = 1e-2  # relative perturbation size (needs to be large enough
                   # to overcome float32 precision in OQ's AAL)


def run_oq_and_get_aal(job_ini: str, exposure_xml: str = None) -> float:
    """Run OQ ScenarioRisk and return portfolio structural AAL.

    If exposure_xml is provided, overrides the exposure input.
    """
    from openquake.calculators.base import run_calc
    from openquake.commonlib.readinput import get_params

    params = get_params(job_ini)
    if exposure_xml is not None:
        params['inputs']['exposure'] = [exposure_xml]
    calc = run_calc(params)

    aal = float(calc.datastore['avg_losses-rlzs/structural'][:].sum())
    calc.datastore.close()
    return aal


def main():
    print('=' * 60)
    print('OQ Exposure Gradient: Finite Differences')
    print('=' * 60)

    # ── 1. Read original exposure ──────────────────────────────────
    print('\n[1] Loading exposure data ...')
    expo_csv = DEMO_DIR / 'exposure_model.csv'
    expo_df = pd.read_csv(expo_csv)
    # Cast structural to float to avoid pandas integer truncation
    expo_df['structural'] = expo_df['structural'].astype(float)
    N = len(expo_df)
    print(f'  Assets: {N}')
    print(f'  Structural range: [{expo_df["structural"].min()}, '
          f'{expo_df["structural"].max()}]')

    job_ini = str(DEMO_DIR / 'job.ini')

    # ── 2. Baseline AAL ───────────────────────────────────────────
    print('\n[2] Computing baseline AAL ...')
    t0 = time.perf_counter()
    aal_base = run_oq_and_get_aal(job_ini)
    t_base = time.perf_counter() - t0
    print(f'  Baseline AAL: {aal_base:,.2f}')
    print(f'  Time: {t_base:.2f}s')

    # ── 3. Select random sample of assets ─────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, size=N_SAMPLE, replace=False)
    sample_idx.sort()
    print(f'\n[3] Selected {N_SAMPLE} random assets for FD')

    # ── 4. Central FD for each sampled asset ──────────────────────
    print(f'\n[4] Running FD ({N_SAMPLE * 2} OQ calculations) ...')
    grad_oq = np.zeros(N_SAMPLE, dtype=np.float64)
    times = []

    # Create a temp directory for modified exposure files
    tmpdir = Path(tempfile.mkdtemp(prefix='oq_grad_'))
    # Copy the exposure XML (it references CSV by relative name)
    expo_xml_orig = DEMO_DIR / 'exposure_model.xml'
    expo_xml_tmp = tmpdir / 'exposure_model.xml'
    shutil.copy(expo_xml_orig, expo_xml_tmp)
    csv_tmp = tmpdir / 'exposure_model.csv'

    try:
        for j, i in enumerate(sample_idx):
            t_start = time.perf_counter()
            base_val = float(expo_df.loc[i, 'structural'])
            h = DELTA_REL * max(abs(base_val), 1.0)

            # Forward perturbation
            df_plus = expo_df.copy()
            df_plus.loc[i, 'structural'] = base_val + h
            df_plus.to_csv(csv_tmp, index=False)
            aal_plus = run_oq_and_get_aal(job_ini, str(expo_xml_tmp))

            # Backward perturbation
            df_minus = expo_df.copy()
            df_minus.loc[i, 'structural'] = base_val - h
            df_minus.to_csv(csv_tmp, index=False)
            aal_minus = run_oq_and_get_aal(job_ini, str(expo_xml_tmp))

            grad_oq[j] = (aal_plus - aal_minus) / (2.0 * h)
            elapsed = time.perf_counter() - t_start
            times.append(elapsed)

            if (j + 1) % 10 == 0 or j == 0:
                eta = np.mean(times) * (N_SAMPLE - j - 1)
                print(f'  [{j+1:3d}/{N_SAMPLE}] asset={i:5d}, '
                      f'v={base_val:.0f}, ∂AAL/∂v={grad_oq[j]:.6f}, '
                      f'time={elapsed:.1f}s, ETA={eta/60:.1f}min')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    t_total = sum(times)
    print(f'\n  Total FD time: {t_total:.1f}s ({t_total/60:.1f} min)')
    print(f'  Mean per asset: {np.mean(times):.2f}s')

    # ── 5. Compare with T-Risk AD (if available) ──────────────────
    print('\n[5] Comparing with T-Risk AD ...')
    trisk_file = BASE_DIR / 'gradient_verification_result.json'
    if trisk_file.exists():
        with open(trisk_file) as f:
            trisk_data = json.load(f)

        # T-Risk AD gradient for the sampled assets
        # Note: OQ may aggregate assets differently. We compare by
        # checking if the grad values are in the same ballpark.
        expo_grad = trisk_data.get('exposure_gradient', {})
        ad_summary = expo_grad.get('ad_grad_v_summary', {})
        print(f'  T-Risk AD grad range: [{ad_summary.get("min", "?"):.6f}, '
              f'{ad_summary.get("max", "?"):.6f}]')
        print(f'  OQ FD grad range:     [{grad_oq.min():.6f}, '
              f'{grad_oq.max():.6f}]')
        print(f'  T-Risk AD mean:       {ad_summary.get("mean", "?"):.6f}')
        print(f'  OQ FD mean:           {grad_oq.mean():.6f}')
    else:
        print('  T-Risk results not found. Run run_gradient_verification.py first.')

    # ── 6. Save results ───────────────────────────────────────────
    result = {
        'demo': 'ScenarioRisk',
        'N_total': N,
        'N_sample': N_SAMPLE,
        'delta_rel': DELTA_REL,
        'baseline_aal': aal_base,
        'baseline_time_sec': round(t_base, 2),
        'total_fd_time_sec': round(t_total, 2),
        'mean_time_per_asset_sec': round(float(np.mean(times)), 2),
        'projected_full_time_sec': round(float(np.mean(times)) * N, 2),
        'sample_indices': sample_idx.tolist(),
        'oq_fd_grad': grad_oq.tolist(),
        'oq_fd_grad_summary': {
            'min': float(grad_oq.min()),
            'max': float(grad_oq.max()),
            'mean': float(grad_oq.mean()),
            'std': float(grad_oq.std()),
        },
    }

    out_path = BASE_DIR / 'gradient_oq_exposure_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    # ── Summary ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Baseline AAL:        {aal_base:>16,.2f}')
    print(f'  Assets sampled:      {N_SAMPLE}')
    print(f'  OQ runs:             {N_SAMPLE * 2 + 1}')
    print(f'  Total time:          {t_total/60:.1f} min')
    print(f'  Mean per asset:      {np.mean(times):.2f}s')
    print(f'  Projected full N:    {float(np.mean(times)) * N / 3600:.1f} hours')
    print(f'  Grad range:          [{grad_oq.min():.6f}, {grad_oq.max():.6f}]')
    print(f'  Grad mean:           {grad_oq.mean():.6f}')
    print('=' * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
