#!/usr/bin/env python3
"""
Benchmark Phase 1: OpenQuake risk-only timing.

Runs each demo in two stages:
  1. Hazard calculation → stores calc_id
  2. Risk calculation (--hc <hazard_id>) → wall-clock risk-only time

Exports CSV outputs + timing JSON to benchmark_outputs/ for the
T-Risk benchmark script (run_benchmark_trisk.py) to consume.

Requires: OQ environment (oq-env).
"""
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

OQ_ROOT = Path(__file__).resolve().parent.parent
BASE_OUT = Path(__file__).resolve().parent
BENCH_DIR = BASE_OUT / 'benchmark_outputs'

# ---------- demo definitions ----------
# (name, job_file, calc_mode, can_split_haz_risk, scaled_params)
DEMOS = [
    {
        'name': 'ScenarioRisk',
        'job': 'job.ini',
        'mode': 'scenario_risk',
        'can_split': False,          # scenario jobs have hazard embedded
    },
    {
        'name': 'EventBasedRisk',
        'job': 'job.ini',
        'mode': 'event_based_risk',
        'can_split': True,
    },
    {
        'name': 'Reinsurance',
        'job': 'job.ini',
        'mode': 'event_based_risk',
        'can_split': False,          # reinsurance layer needs combined run
    },
    {
        'name': 'EventBasedRisk_scaled',
        'job': 'job.ini',
        'mode': 'event_based_risk',
        'can_split': True,
        'source_demo': 'EventBasedRisk',
        'overrides': {'ses_per_logic_tree_path': '10'},
    },
]


def parse_calc_id(output: str) -> int:
    m = re.search(r'calc_(\d+)\.hdf5', output)
    if not m:
        raise RuntimeError(f"Cannot parse calc ID from output:\n{output}")
    return int(m.group(1))


def run_oq(args: list, cwd: Path) -> tuple:
    """Run an oq command, return (stdout+stderr, elapsed_seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ['oq'] + args, cwd=cwd,
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    text = proc.stdout + '\n' + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"oq {' '.join(args)} failed:\n{text}")
    return text, elapsed


def get_performance(calc_id: int) -> str:
    proc = subprocess.run(
        ['oq', 'show', 'performance', str(calc_id)],
        capture_output=True, text=True
    )
    return proc.stdout


def write_split_ini(example_dir: Path, job_file: str,
                    overrides: dict = None):
    """
    Read the original job.ini, write a hazard-only and risk-only INI
    into the example_dir (so relative paths to data files resolve correctly).
    Returns (hazard_ini_path, risk_ini_path).
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(example_dir / job_file)

    # Determine original mode
    orig_mode = parser.get('general', 'calculation_mode')

    # ---- hazard INI ----
    haz_parser = configparser.ConfigParser()
    haz_parser.optionxform = str

    if 'event_based' in orig_mode:
        haz_mode = 'event_based'
    elif 'scenario' in orig_mode:
        haz_mode = 'scenario'
    elif 'classical' in orig_mode:
        haz_mode = 'classical'
    else:
        haz_mode = orig_mode.replace('_risk', '').replace('_damage', '')

    # Copy all sections from original, skipping risk_calculation only
    # (keep vulnerability so OQ can infer IMTs when not explicitly set)
    for sec in parser.sections():
        if sec == 'risk_calculation':
            continue
        if not haz_parser.has_section(sec):
            haz_parser.add_section(sec)
        for k, v in parser.items(sec):
            if sec == 'general' and k == 'calculation_mode':
                haz_parser.set(sec, k, haz_mode)
            else:
                haz_parser.set(sec, k, v)

    # Apply overrides to hazard
    if overrides:
        for k, v in overrides.items():
            for sec in haz_parser.sections():
                if haz_parser.has_option(sec, k):
                    haz_parser.set(sec, k, v)

    haz_path = example_dir / '_bench_job_hazard.ini'
    with open(haz_path, 'w') as f:
        haz_parser.write(f)

    # ---- risk INI ----
    risk_parser = configparser.ConfigParser()
    risk_parser.optionxform = str

    for sec in parser.sections():
        if not risk_parser.has_section(sec):
            risk_parser.add_section(sec)
        for k, v in parser.items(sec):
            risk_parser.set(sec, k, v)

    # Apply overrides to risk
    if overrides:
        for k, v in overrides.items():
            for sec in risk_parser.sections():
                if risk_parser.has_option(sec, k):
                    risk_parser.set(sec, k, v)

    risk_path = example_dir / '_bench_job_risk.ini'
    with open(risk_path, 'w') as f:
        risk_parser.write(f)

    return haz_path, risk_path


def benchmark_split(demo: dict, example_dir: Path, out_dir: Path) -> dict:
    """Benchmark by splitting hazard + risk into two OQ runs."""
    overrides = demo.get('overrides', None)

    haz_ini, risk_ini = write_split_ini(example_dir, demo['job'], overrides)

    try:
        print(f'  [hazard] Running {haz_ini.name} ...')
        haz_text, haz_time = run_oq(
            ['run', haz_ini.name, '-e', 'csv'], cwd=example_dir
        )
        haz_id = parse_calc_id(haz_text)
        print(f'  [hazard] calc_id={haz_id}  time={haz_time:.2f}s')

        print(f'  [risk]   Running {risk_ini.name} --hc {haz_id} ...')
        risk_text, risk_time = run_oq(
            ['run', risk_ini.name, '--hc', str(haz_id), '-e', 'csv'], cwd=example_dir
        )
        risk_id = parse_calc_id(risk_text)
        print(f'  [risk]   calc_id={risk_id}  time={risk_time:.2f}s')

        # Collect exported files from both hazard and risk runs
        for csv_file in example_dir.glob(f'*_{risk_id}.csv'):
            shutil.copy2(csv_file, out_dir / csv_file.name)
        for csv_file in example_dir.glob(f'*_{haz_id}.csv'):
            shutil.copy2(csv_file, out_dir / csv_file.name)

        perf_risk = get_performance(risk_id)
        perf_haz = get_performance(haz_id)
    finally:
        # Clean up temp INI files from demo directory
        haz_ini.unlink(missing_ok=True)
        risk_ini.unlink(missing_ok=True)

    return {
        'name': demo['name'],
        'hazard_calc_id': haz_id,
        'risk_calc_id': risk_id,
        'hazard_time_sec': round(haz_time, 4),
        'risk_time_sec': round(risk_time, 4),
        'performance_risk': perf_risk,
        'performance_hazard': perf_haz,
    }


def benchmark_combined(demo: dict, example_dir: Path, out_dir: Path) -> dict:
    """Benchmark when hazard/risk can't be split (e.g. ScenarioRisk)."""
    print(f'  [combined] Running {demo["job"]} ...')
    text, total_time = run_oq(
        ['run', demo['job'], '-e', 'csv'], cwd=example_dir
    )
    calc_id = parse_calc_id(text)
    print(f'  [combined] calc_id={calc_id}  time={total_time:.2f}s')

    for csv_file in example_dir.glob(f'*_{calc_id}.csv'):
        shutil.copy2(csv_file, out_dir / csv_file.name)

    perf = get_performance(calc_id)

    return {
        'name': demo['name'],
        'hazard_calc_id': None,
        'risk_calc_id': calc_id,
        'hazard_time_sec': None,
        'risk_time_sec': round(total_time, 4),
        'performance_risk': perf,
        'performance_hazard': None,
    }


def count_events_assets(out_dir: Path) -> tuple:
    """Count events and assets from exported CSVs."""
    n_events, n_assets = 0, 0
    events_files = sorted(out_dir.glob('events_*.csv'))
    if events_files:
        import pandas as pd
        ev = pd.read_csv(events_files[0], comment='#')
        n_events = len(ev)
    avg_files = sorted(out_dir.glob('avg_losses*_*.csv'))
    if avg_files:
        import pandas as pd
        al = pd.read_csv(avg_files[0], comment='#')
        n_assets = len(al)
    return n_events, n_assets


def main():
    print('=' * 60)
    print('OpenQuake Benchmark - Phase 1 (OQ risk-only timing)')
    print('=' * 60)

    results = []

    for demo in DEMOS:
        name = demo['name']
        source = demo.get('source_demo', name)
        example_dir = OQ_ROOT / 'demos' / 'risk' / source

        if not example_dir.exists():
            print(f'\nSKIPPING {name}: {example_dir} not found')
            continue

        print(f'\n--- {name} ---')
        out_dir = BENCH_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if demo['can_split']:
            result = benchmark_split(demo, example_dir, out_dir)
        else:
            result = benchmark_combined(demo, example_dir, out_dir)

        n_events, n_assets = count_events_assets(out_dir)
        result['n_events'] = n_events
        result['n_assets'] = n_assets

        results.append(result)
        print(f'  => OQ risk time: {result["risk_time_sec"]:.2f}s  '
              f'(events={n_events}, assets={n_assets})')

    # Save timing JSON for T-Risk benchmark to read
    timing_path = BENCH_DIR / 'oq_timing.json'
    timing_data = []
    for r in results:
        timing_data.append({
            'name': r['name'],
            'hazard_calc_id': r['hazard_calc_id'],
            'risk_calc_id': r['risk_calc_id'],
            'hazard_time_sec': r['hazard_time_sec'],
            'risk_time_sec': r['risk_time_sec'],
            'n_events': r['n_events'],
            'n_assets': r['n_assets'],
        })
    with open(timing_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    # Save performance text
    for r in results:
        perf_dir = BENCH_DIR / r['name']
        if r['performance_risk']:
            (perf_dir / 'oq_performance_risk.txt').write_text(r['performance_risk'])
        if r.get('performance_hazard'):
            (perf_dir / 'oq_performance_hazard.txt').write_text(r['performance_hazard'])

    print(f'\n{"=" * 60}')
    print(f'OQ timing saved to: {timing_path}')
    print(f'CSV outputs saved to: {BENCH_DIR}/')
    print(f'\nSummary:')
    print(f'{"Demo":<30} {"Risk time (s)":>14} {"Events":>8} {"Assets":>8}')
    print('-' * 62)
    for r in results:
        print(f'{r["name"]:<30} {r["risk_time_sec"]:>14.2f} '
              f'{r["n_events"]:>8} {r["n_assets"]:>8}')
    print(f'\nNext step: run run_benchmark_trisk.py in the T-Risk environment.')


if __name__ == '__main__':
    main()
