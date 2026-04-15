#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability benchmark for the Tensorial Risk Engine.

This script extends the basic SoftwareX example into a reproducible benchmark
that addresses reviewer requests for:
- CPU versus GPU comparisons
- Separate timings for forward and backward passes
- Scaling studies with increasing numbers of assets and events
- A concise discussion of memory growth

Outputs:
- CSV table with all benchmark runs
- Publication-oriented figures saved under
  `figures_response_software_x_scalability/`
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

_cache_root = Path.cwd() / "figures_response_software_x_scalability"
os.environ["MPLCONFIGDIR"] = str(_cache_root / ".mplconfig")
os.environ["XDG_CACHE_HOME"] = str(_cache_root / ".cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensor_engine import TensorialRiskEngine, generate_synthetic_portfolio


OUTPUT_DIR = Path("figures_response_software_x_scalability")
CSV_PATH = OUTPUT_DIR / "software_x_scalability_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "software_x_scalability_summary.txt"


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def detect_devices() -> Dict[str, str]:
    devices = {"cpu": "/CPU:0"}
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        devices["gpu"] = "/GPU:0"
    return devices


def build_engine(
    n_assets: int,
    n_events: int,
    n_typologies: int,
    n_curve_points: int,
    lambda_distribution: str,
    seed: int,
) -> TensorialRiskEngine:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    v, u, c_matrix, x_grid, h_matrix, lambdas = generate_synthetic_portfolio(
        n_assets=n_assets,
        n_events=n_events,
        n_typologies=n_typologies,
        n_curve_points=n_curve_points,
        lambda_distribution=lambda_distribution,
    )
    return TensorialRiskEngine(v, u, c_matrix, x_grid, h_matrix, lambdas)


def sync_tensor(tensor: tf.Tensor) -> float:
    return float(tf.reduce_sum(tf.cast(tensor, tf.float32)).numpy())


def time_forward(engine: TensorialRiskEngine, device_name: str) -> float:
    with tf.device(device_name):
        start = time.perf_counter()
        _, metrics = engine.compute_loss_and_metrics()
        _ = float(metrics["aal_portfolio"].numpy())
        end = time.perf_counter()
    return end - start


def time_forward_backward_vulnerability(
    engine: TensorialRiskEngine, device_name: str
) -> Dict[str, float]:
    with tf.device(device_name):
        with tf.GradientTape() as tape:
            forward_start = time.perf_counter()
            _, metrics = engine.compute_loss_and_metrics()
            aal = metrics["aal_portfolio"]
            _ = float(aal.numpy())
            forward_end = time.perf_counter()

        backward_start = time.perf_counter()
        grad_c = tape.gradient(aal, engine.C)
        if grad_c is None:
            raise RuntimeError("Gradient w.r.t. vulnerability matrix returned None.")
        _ = sync_tensor(grad_c)
        backward_end = time.perf_counter()

    return {
        "forward_seconds": forward_end - forward_start,
        "backward_seconds": backward_end - backward_start,
        "total_seconds": backward_end - forward_start,
    }


def time_forward_backward_full(
    engine: TensorialRiskEngine, device_name: str
) -> Dict[str, float]:
    with tf.device(device_name):
        with tf.GradientTape() as tape:
            forward_start = time.perf_counter()
            _, metrics = engine.compute_loss_and_metrics()
            aal = metrics["aal_portfolio"]
            _ = float(aal.numpy())
            forward_end = time.perf_counter()

        backward_start = time.perf_counter()
        gradients = tape.gradient(aal, [engine.H, engine.C, engine.v, engine.lambdas])
        if any(grad is None for grad in gradients):
            raise RuntimeError("One or more gradients returned None in full analysis.")
        _ = sum(sync_tensor(grad) for grad in gradients)
        backward_end = time.perf_counter()

    return {
        "forward_seconds": forward_end - forward_start,
        "backward_seconds": backward_end - backward_start,
        "total_seconds": backward_end - forward_start,
    }


def median_or_nan(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan


def benchmark_case(
    n_assets: int,
    n_events: int,
    device_label: str,
    device_name: str,
    warmup_runs: int,
    timed_runs: int,
    n_typologies: int,
    n_curve_points: int,
    lambda_distribution: str,
    seed: int,
) -> Dict[str, float]:
    for warmup_idx in range(warmup_runs):
        engine = build_engine(
            n_assets, n_events, n_typologies, n_curve_points, lambda_distribution, seed + warmup_idx
        )
        _ = time_forward(engine, device_name)
        _ = time_forward_backward_vulnerability(engine, device_name)
        _ = time_forward_backward_full(engine, device_name)

    forward_times = []
    vulnerability_forward_times = []
    vulnerability_backward_times = []
    vulnerability_total_times = []
    full_forward_times = []
    full_backward_times = []
    full_total_times = []

    for run_idx in range(timed_runs):
        engine = build_engine(
            n_assets,
            n_events,
            n_typologies,
            n_curve_points,
            lambda_distribution,
            seed + 100 + run_idx,
        )
        forward_times.append(time_forward(engine, device_name))

        vuln_times = time_forward_backward_vulnerability(engine, device_name)
        vulnerability_forward_times.append(vuln_times["forward_seconds"])
        vulnerability_backward_times.append(vuln_times["backward_seconds"])
        vulnerability_total_times.append(vuln_times["total_seconds"])

        full_times = time_forward_backward_full(engine, device_name)
        full_forward_times.append(full_times["forward_seconds"])
        full_backward_times.append(full_times["backward_seconds"])
        full_total_times.append(full_times["total_seconds"])

    memory = estimate_memory_bytes(n_assets, n_events, n_typologies, n_curve_points)

    return {
        "n_assets": n_assets,
        "n_events": n_events,
        "device": device_label,
        "forward_only_seconds": median_or_nan(forward_times),
        "backward_vulnerability_forward_seconds": median_or_nan(vulnerability_forward_times),
        "backward_vulnerability_backward_seconds": median_or_nan(vulnerability_backward_times),
        "backward_vulnerability_total_seconds": median_or_nan(vulnerability_total_times),
        "backward_full_forward_seconds": median_or_nan(full_forward_times),
        "backward_full_backward_seconds": median_or_nan(full_backward_times),
        "backward_full_total_seconds": median_or_nan(full_total_times),
        "hazard_bytes": memory["hazard_bytes"],
        "loss_matrix_bytes": memory["loss_matrix_bytes"],
        "grad_h_bytes": memory["grad_h_bytes"],
        "forward_working_set_bytes": memory["forward_working_set_bytes"],
        "full_backward_working_set_bytes": memory["full_backward_working_set_bytes"],
    }


def estimate_memory_bytes(
    n_assets: int, n_events: int, n_typologies: int, n_curve_points: int
) -> Dict[str, int]:
    float32_bytes = 4
    int32_bytes = 4

    hazard_bytes = n_assets * n_events * float32_bytes
    loss_matrix_bytes = n_assets * n_events * float32_bytes
    grad_h_bytes = n_assets * n_events * float32_bytes
    exposure_bytes = n_assets * float32_bytes
    typology_bytes = n_assets * int32_bytes
    lambdas_bytes = n_events * float32_bytes
    curve_bytes = n_typologies * n_curve_points * float32_bytes
    grid_bytes = n_curve_points * float32_bytes

    forward_working_set_bytes = (
        hazard_bytes
        + loss_matrix_bytes
        + exposure_bytes
        + typology_bytes
        + lambdas_bytes
        + curve_bytes
        + grid_bytes
    )
    full_backward_working_set_bytes = forward_working_set_bytes + grad_h_bytes + curve_bytes

    return {
        "hazard_bytes": hazard_bytes,
        "loss_matrix_bytes": loss_matrix_bytes,
        "grad_h_bytes": grad_h_bytes,
        "forward_working_set_bytes": forward_working_set_bytes,
        "full_backward_working_set_bytes": full_backward_working_set_bytes,
    }


def write_csv(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(
    rows: Sequence[Dict[str, float]], study: str, device: str | None = None
) -> List[Dict[str, float]]:
    filtered = [row for row in rows if row["study"] == study]
    if device is not None:
        filtered = [row for row in filtered if row["device"] == device]
    return sorted(filtered, key=lambda item: item["x_value"])


def plot_scaling_results(rows: Sequence[Dict[str, float]], devices: Iterable[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    studies = {
        "asset_scaling": ("Number of assets", "Assets"),
        "event_scaling": ("Number of events", "Events"),
    }

    metrics_to_plot = [
        ("forward_only_seconds", "Forward Only Time [s]", "software_x_scalability_forward.png"),
        (
            "backward_vulnerability_backward_seconds",
            "Backward Time wrt Vulnerability [s]",
            "software_x_scalability_backward_vulnerability.png",
        ),
        (
            "backward_full_backward_seconds",
            "Backward Time Full Gradient [s]",
            "software_x_scalability_backward_full.png",
        ),
        (
            "full_backward_working_set_bytes",
            "Estimated Working Memory Full Backward [bytes]",
            "software_x_scalability_memory.png",
        ),
    ]

    for metric_name, y_label, filename in metrics_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, (study_key, (x_label, title_label)) in zip(axes, studies.items()):
            for device in devices:
                study_rows = filter_rows(rows, study_key, device)
                if not study_rows:
                    continue
                x_values = [row["x_value"] for row in study_rows]
                y_values = [row[metric_name] for row in study_rows]
                axis.plot(x_values, y_values, marker="o", linewidth=2, label=device.upper())

            axis.set_xlabel(x_label, fontsize=12)
            axis.set_ylabel(y_label, fontsize=12)
            axis.set_title(f"{title_label} Scaling", fontsize=13, fontweight="bold")
            axis.grid(True, alpha=0.3)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if "gpu" in set(devices):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        speedup_specs = [
            ("forward_only_seconds", "GPU Speedup Forward Only [x]"),
            ("backward_full_total_seconds", "GPU Speedup Full Forward+Backward [x]"),
        ]
        for axis, (metric_name, y_label) in zip(axes, speedup_specs):
            for study_key, (x_label, title_label) in studies.items():
                cpu_rows = filter_rows(rows, study_key, "cpu")
                gpu_rows = filter_rows(rows, study_key, "gpu")
                paired = zip(cpu_rows, gpu_rows)
                x_values = []
                speedups = []
                for cpu_row, gpu_row in paired:
                    if cpu_row["x_value"] != gpu_row["x_value"]:
                        continue
                    x_values.append(cpu_row["x_value"])
                    speedups.append(cpu_row[metric_name] / gpu_row[metric_name])
                if x_values:
                    axis.plot(x_values, speedups, marker="o", linewidth=2, label=title_label)

            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("Scaling variable", fontsize=12)
            axis.set_ylabel(y_label, fontsize=12)
            axis.set_title(y_label, fontsize=13, fontweight="bold")
            axis.grid(True, alpha=0.3)
            axis.set_xscale("log")
            axis.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "software_x_scalability_speedup.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_summary(rows: Sequence[Dict[str, float]], output_path: Path, devices: Iterable[str]) -> None:
    device_list = list(devices)
    lines = [
        "SoftwareX scalability benchmark summary",
        "=======================================",
        "",
        f"Devices benchmarked: {', '.join(device_list).upper()}",
        "",
        "Memory interpretation:",
        "- The dominant tensors scale as O(NQ).",
        "- H, J_matrix, and grad_H each require approximately 4*N*Q bytes in float32.",
        "- Full backward mode is more memory-intensive than forward-only because autodiff retains intermediates and, when requested, materializes grad_H.",
        "",
    ]

    for study in ("asset_scaling", "event_scaling"):
        study_rows = filter_rows(rows, study)
        if not study_rows:
            continue
        lines.append(f"{study}:")
        for row in study_rows:
            lines.append(
                f"- {row['device'].upper()} | N={row['n_assets']}, Q={row['n_events']} | "
                f"forward={row['forward_only_seconds']:.4f}s | "
                f"full backward total={row['backward_full_total_seconds']:.4f}s | "
                f"est. full memory={format_bytes(row['full_backward_working_set_bytes'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def enrich_rows(raw_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    enriched = []
    for row in raw_rows:
        enriched_row = dict(row)
        if row["study"] == "asset_scaling":
            enriched_row["x_value"] = row["n_assets"]
        else:
            enriched_row["x_value"] = row["n_events"]
        enriched.append(enriched_row)
    return enriched


def run_benchmark(args: argparse.Namespace) -> List[Dict[str, float]]:
    devices = detect_devices()
    selected_devices = [args.device] if args.device != "all" else list(devices.keys())

    asset_rows = []
    for device_label in selected_devices:
        for n_assets in args.asset_sizes:
            print(f"[asset_scaling] device={device_label} assets={n_assets} events={args.fixed_events}")
            row = benchmark_case(
                n_assets=n_assets,
                n_events=args.fixed_events,
                device_label=device_label,
                device_name=devices[device_label],
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                n_typologies=args.n_typologies,
                n_curve_points=args.n_curve_points,
                lambda_distribution=args.lambda_distribution,
                seed=args.seed,
            )
            row["study"] = "asset_scaling"
            asset_rows.append(row)

    event_rows = []
    for device_label in selected_devices:
        for n_events in args.event_sizes:
            print(f"[event_scaling] device={device_label} assets={args.fixed_assets} events={n_events}")
            row = benchmark_case(
                n_assets=args.fixed_assets,
                n_events=n_events,
                device_label=device_label,
                device_name=devices[device_label],
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                n_typologies=args.n_typologies,
                n_curve_points=args.n_curve_points,
                lambda_distribution=args.lambda_distribution,
                seed=args.seed,
            )
            row["study"] = "event_scaling"
            event_rows.append(row)

    return enrich_rows(asset_rows + event_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scalability benchmark for the Tensorial Risk Engine."
    )
    parser.add_argument(
        "--asset-sizes",
        type=parse_int_list,
        default=[100, 500, 1000, 5000, 10000],
        help="Comma-separated asset sizes for the asset scaling study.",
    )
    parser.add_argument(
        "--event-sizes",
        type=parse_int_list,
        default=[100, 500, 1000, 5000, 10000, 20000],
        help="Comma-separated event sizes for the event scaling study.",
    )
    parser.add_argument(
        "--fixed-assets",
        type=int,
        default=1000,
        help="Fixed number of assets for the event scaling study.",
    )
    parser.add_argument(
        "--fixed-events",
        type=int,
        default=2000,
        help="Fixed number of events for the asset scaling study.",
    )
    parser.add_argument("--n-typologies", type=int, default=5)
    parser.add_argument("--n-curve-points", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lambda-distribution",
        type=str,
        default="exponential",
        choices=["uniform", "exponential"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="all",
        choices=["all", "cpu", "gpu"],
        help="Device to benchmark. 'all' uses every available supported device.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    available_devices = detect_devices()
    if args.device == "gpu" and "gpu" not in available_devices:
        raise RuntimeError("GPU benchmarking requested, but no TensorFlow GPU was detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    write_csv(rows, CSV_PATH)
    plot_scaling_results(rows, sorted(set(row["device"] for row in rows)))
    write_summary(rows, SUMMARY_PATH, sorted(set(row["device"] for row in rows)))

    print(f"Saved benchmark table to {CSV_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
