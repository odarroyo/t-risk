#!/usr/bin/env python3
"""
Standalone Bogotá T-Risk hardware benchmark.

Run this script inside a TensorFlow environment. It assumes the input files are
in the same directory as this script unless paths are provided explicitly.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


DEFAULT_GROUPS = ("SA_0p1", "SA_0p3", "SA_0p6", "SA_1p0")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(script_dir / "bogota_trisk_inputs.npz"))
    parser.add_argument("--hazard", default=str(script_dir / "bogota_hazard_chia.npz"))
    parser.add_argument("--out", default=str(script_dir / "trisk_hardware_benchmark_summary.json"))
    parser.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["forward", "vulnerability", "exposure", "hazard"],
        choices=("forward", "vulnerability", "exposure", "hazard", "all"),
        help="Benchmark modes. 'all' expands to every mode.",
    )
    parser.add_argument("--taxonomy-mode", choices=("most_common", "all"), default="most_common")
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def expand_modes(modes: list[str]) -> list[str]:
    if "all" in modes:
        return ["forward", "vulnerability", "exposure", "hazard"]
    return list(dict.fromkeys(modes))


def time_call(fn: Callable, warmup: int, repeat: int) -> tuple[object, dict]:
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, {
        "times_s": times,
        "min_s": float(np.min(times)),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "repeat": repeat,
        "warmup": warmup,
    }


def safe_float(x) -> float:
    return float(x.numpy() if hasattr(x, "numpy") else x)


@tf.function
def probabilistic_loss_matrix(v: tf.Tensor, u: tf.Tensor, c: tf.Tensor, x_grid: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
    n_assets = tf.shape(h)[0]
    n_events = tf.shape(h)[1]
    n_points = tf.shape(x_grid)[0]

    h_flat = tf.reshape(h, [-1])
    h_eval = tf.clip_by_value(h_flat, x_grid[0], x_grid[-1])
    valid_iml = h_flat >= x_grid[0]

    idx = tf.searchsorted(x_grid, h_eval, side="right") - 1
    idx = tf.clip_by_value(idx, 0, n_points - 2)
    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    alpha = (h_eval - x_lower) / (x_upper - x_lower + 1e-8)

    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, n_events])
    u_flat = tf.reshape(u_repeated, [-1])
    c_flat = tf.reshape(c, [-1])
    lower = tf.gather(c_flat, u_flat * n_points + idx)
    upper = tf.gather(c_flat, u_flat * n_points + idx + 1)
    mdr = (1.0 - alpha) * lower + alpha * upper
    mdr = tf.where(valid_iml, mdr, tf.zeros_like(mdr))
    return tf.expand_dims(v, 1) * tf.reshape(mdr, [n_assets, n_events])


@tf.function
def risk_metrics(j_matrix: tf.Tensor, lambdas: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    aal_per_asset = tf.linalg.matvec(j_matrix, lambdas)
    loss_per_event = tf.reduce_sum(j_matrix, axis=0)
    aal_portfolio = tf.reduce_sum(aal_per_asset)
    return aal_portfolio, aal_per_asset, loss_per_event


def load_group(inputs, hazard, group: str, max_assets: int | None, max_events: int | None) -> dict:
    v = inputs[f"{group}_v"].astype(np.float32)
    u = inputs[f"{group}_u"].astype(np.int32)
    c = inputs[f"{group}_C"].astype(np.float32)
    x_grid = inputs[f"{group}_x_grid"].astype(np.float32)
    h = hazard[f"H_{group}"].astype(np.float32)
    lambdas = hazard["lambdas"].astype(np.float32)
    labels = inputs[f"{group}_taxonomy_labels"].astype(str)

    if max_assets is not None:
        v = v[:max_assets]
        u = u[:max_assets]
        h = h[:max_assets, :]
    if max_events is not None:
        h = h[:, :max_events]
        lambdas = lambdas[:max_events]

    return {"v": v, "u": u, "C": c, "x_grid": x_grid, "H": h, "lambdas": lambdas, "labels": labels}


def choose_taxonomies(u: np.ndarray, taxonomy_mode: str) -> list[int]:
    present = np.flatnonzero(np.bincount(u))
    if taxonomy_mode == "all":
        return [int(x) for x in present]
    counts = np.bincount(u)
    return [int(np.argmax(counts))]


def main() -> int:
    args = parse_args()
    modes = expand_modes(args.modes)

    load_t0 = time.perf_counter()
    inputs = np.load(args.inputs, allow_pickle=True)
    hazard = np.load(args.hazard, allow_pickle=True)
    load_runtime_s = time.perf_counter() - load_t0

    summary = {
        "runner": "run_trisk_hardware_benchmark.py",
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "physical_devices": [str(d) for d in tf.config.list_physical_devices()],
        "inputs": str(Path(args.inputs).resolve()),
        "hazard": str(Path(args.hazard).resolve()),
        "load_runtime_s": load_runtime_s,
        "modes": modes,
        "groups": {},
    }

    for group in args.groups:
        data = load_group(inputs, hazard, group, args.max_assets, args.max_events)
        v = tf.Variable(data["v"], dtype=tf.float32)
        u = tf.constant(data["u"], dtype=tf.int32)
        c = tf.Variable(data["C"], dtype=tf.float32)
        x_grid = tf.constant(data["x_grid"], dtype=tf.float32)
        h = tf.Variable(data["H"], dtype=tf.float32)
        lambdas = tf.constant(data["lambdas"], dtype=tf.float32)

        group_summary = {
            "n_assets": int(data["H"].shape[0]),
            "n_events": int(data["H"].shape[1]),
            "n_taxonomies": int(data["C"].shape[0]),
        }

        def forward_fn():
            j = probabilistic_loss_matrix(v, u, c, x_grid, h)
            return risk_metrics(j, lambdas)

        if "forward" in modes:
            forward_result, timing = time_call(forward_fn, args.warmup, args.repeat)
            aal_portfolio, aal_per_asset, loss_per_event = forward_result
            group_summary["forward"] = {
                "portfolio_aal": safe_float(aal_portfolio),
                "timing": timing,
                "aal_per_asset_shape": list(aal_per_asset.shape),
                "loss_per_event_shape": list(loss_per_event.shape),
            }

        if "vulnerability" in modes:
            def vuln_grad_fn():
                with tf.GradientTape() as tape:
                    j = probabilistic_loss_matrix(v, u, c, x_grid, h)
                    aal, _, _ = risk_metrics(j, lambdas)
                return tape.gradient(aal, c)

            grad_c, timing = time_call(vuln_grad_fn, args.warmup, args.repeat)
            grad_c_np = grad_c.numpy()
            selected = []
            for taxonomy_id in choose_taxonomies(data["u"], args.taxonomy_mode):
                curve_grad = grad_c_np[taxonomy_id]
                selected.append(
                    {
                        "taxonomy_id": taxonomy_id,
                        "taxonomy_label": str(data["labels"][taxonomy_id]),
                        "asset_count": int(np.sum(data["u"] == taxonomy_id)),
                        "curve_l2_norm": float(np.linalg.norm(curve_grad)),
                        "curve_max_abs": float(np.max(np.abs(curve_grad))),
                    }
                )
            group_summary["vulnerability_gradient"] = {
                "gradient_shape": list(grad_c_np.shape),
                "timing": timing,
                "selected_taxonomies": selected,
            }

        if "exposure" in modes:
            def exposure_grad_fn():
                with tf.GradientTape() as tape:
                    j = probabilistic_loss_matrix(v, u, c, x_grid, h)
                    aal, _, _ = risk_metrics(j, lambdas)
                return tape.gradient(aal, v)

            grad_v, timing = time_call(exposure_grad_fn, args.warmup, args.repeat)
            group_summary["exposure_gradient"] = {
                "gradient_shape": list(grad_v.shape),
                "timing": timing,
                "l2_norm": float(np.linalg.norm(grad_v.numpy())),
            }

        if "hazard" in modes:
            def hazard_grad_fn():
                with tf.GradientTape() as tape:
                    j = probabilistic_loss_matrix(v, u, c, x_grid, h)
                    aal, _, _ = risk_metrics(j, lambdas)
                return tape.gradient(aal, h)

            grad_h, timing = time_call(hazard_grad_fn, args.warmup, args.repeat)
            group_summary["hazard_gradient"] = {
                "gradient_shape": list(grad_h.shape),
                "timing": timing,
                "l2_norm": float(np.linalg.norm(grad_h.numpy())),
            }

        summary["groups"][group] = group_summary
        tf.keras.backend.clear_session()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2))
    print(out_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
