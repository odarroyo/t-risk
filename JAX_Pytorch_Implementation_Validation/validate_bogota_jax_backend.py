#!/usr/bin/env python3
"""
Validate the JAX Tensorial Risk Engine backend against TensorFlow on Bogotá data.

The script compares forward metrics and automatic-differentiation gradients
using the same ``bogota_trisk_inputs.npz`` and ``bogota_hazard_chia.npz`` files.
JAX timings are reported as eager, first JIT call, and steady-state JIT because
the first JIT call includes XLA compilation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np


DEFAULT_GROUPS = ("SA_0p1", "SA_0p3", "SA_0p6", "SA_1p0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="bogota_trisk_inputs.npz")
    parser.add_argument("--hazard", default="bogota_hazard_chia.npz")
    parser.add_argument("--out", default="JAX_validation/bogota_jax_backend_validation.json")
    parser.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    parser.add_argument("--max-assets", type=int, default=2000)
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["forward", "vulnerability", "exposure", "hazard", "lambdas"],
        choices=("forward", "vulnerability", "exposure", "hazard", "lambdas", "all"),
    )
    parser.add_argument(
        "--jax-device",
        default="auto",
        choices=("auto", "cpu", "gpu", "tpu"),
        help="JAX platform. Default 'auto' uses JAX's default device.",
    )
    return parser.parse_args()


def expand_modes(modes: List[str]) -> List[str]:
    if "all" in modes:
        return ["forward", "vulnerability", "exposure", "hazard", "lambdas"]
    return list(dict.fromkeys(modes))


def load_group(
    inputs: Any,
    hazard: Any,
    group: str,
    max_assets: Optional[int],
    max_events: Optional[int],
) -> Dict[str, np.ndarray]:
    v = inputs[f"{group}_v"].astype(np.float32)
    u = inputs[f"{group}_u"].astype(np.int32)
    c = inputs[f"{group}_C"].astype(np.float32)
    x_grid = inputs[f"{group}_x_grid"].astype(np.float32)
    h = hazard[f"H_{group}"].astype(np.float32)
    lambdas = hazard["lambdas"].astype(np.float32)

    if max_assets is not None:
        v = v[:max_assets]
        u = u[:max_assets]
        h = h[:max_assets, :]
    if max_events is not None:
        h = h[:, :max_events]
        lambdas = lambdas[:max_events]

    return {"v": v, "u": u, "C": c, "x_grid": x_grid, "H": h, "lambdas": lambdas}


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    diff = candidate - reference
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    diff_flat = diff.reshape(-1)
    ref_norm = float(np.linalg.norm(ref_flat) + 1e-12)
    cand_norm = float(np.linalg.norm(cand_flat) + 1e-12)
    ref_centered = ref_flat - float(np.mean(ref_flat))
    cand_centered = cand_flat - float(np.mean(cand_flat))
    corr_den = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered) + 1e-12)
    cosine_den = float(ref_norm * cand_norm + 1e-12)
    return {
        "shape": list(reference.shape),
        "max_abs_error": float(np.max(np.abs(diff_flat))) if diff_flat.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(diff_flat))) if diff_flat.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(diff_flat) / ref_norm),
        "cosine_similarity": float(np.dot(ref_flat, cand_flat) / cosine_den),
        "correlation": float(np.dot(ref_centered, cand_centered) / corr_den),
        "reference_l2_norm": float(np.linalg.norm(ref_flat)),
        "candidate_l2_norm": float(np.linalg.norm(cand_flat)),
    }


def tf_to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x.numpy())


def jax_to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def block_until_ready_tree(value: Any) -> Any:
    import jax

    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def choose_jax_device(jax_module: Any, requested: str) -> Optional[Any]:
    if requested == "auto":
        return None
    devices = jax_module.devices(requested)
    if not devices:
        raise RuntimeError(f"Requested JAX platform {requested!r}, but no device is available.")
    return devices[0]


def timed_call(fn: Callable[[], Any], repeat: int = 1) -> Dict[str, Any]:
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = block_until_ready_tree(fn())
        times.append(time.perf_counter() - t0)
    return {
        "times_s": [float(t) for t in times],
        "min_s": float(np.min(times)),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "result": result,
    }


def strip_result(timing: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in timing.items() if key != "result"}


def main() -> int:
    args = parse_args()
    modes = expand_modes(args.modes)

    try:
        import tensorflow as tf
        import jax
    except ImportError as exc:
        print(f"This validator requires TensorFlow and JAX. Missing import: {exc}", file=sys.stderr)
        return 2

    from tensor_engine import TensorialRiskEngine as TensorFlowRiskEngine
    from tensor_engine_jax import (
        TensorialRiskEngine as JAXRiskEngine,
        compute_loss_and_metrics_pure,
        portfolio_aal,
    )

    jax_device = choose_jax_device(jax, args.jax_device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = np.load(args.inputs, allow_pickle=True)
    hazard = np.load(args.hazard, allow_pickle=True)

    summary: Dict[str, Any] = {
        "runner": "validate_bogota_jax_backend.py",
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tensorflow_physical_devices": [str(d) for d in tf.config.list_physical_devices()],
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_requested_device": args.jax_device,
        "jax_selected_device": str(jax_device) if jax_device is not None else "default",
        "inputs": str(Path(args.inputs).resolve()),
        "hazard": str(Path(args.hazard).resolve()),
        "max_assets": args.max_assets,
        "max_events": args.max_events,
        "repeat": args.repeat,
        "modes": modes,
        "groups": {},
    }

    print(f"TensorFlow devices: {summary['tensorflow_physical_devices']}")
    print(f"JAX backend: {summary['jax_default_backend']}; devices: {summary['jax_devices']}")

    grad_fns = {
        "vulnerability": jax.grad(portfolio_aal, argnums=2),
        "exposure": jax.grad(portfolio_aal, argnums=0),
        "hazard": jax.grad(portfolio_aal, argnums=4),
        "lambdas": jax.grad(portfolio_aal, argnums=5),
    }
    jit_forward = jax.jit(compute_loss_and_metrics_pure)
    jit_grad_fns = {name: jax.jit(fn) for name, fn in grad_fns.items()}

    for group in args.groups:
        data = load_group(inputs, hazard, group, args.max_assets, args.max_events)
        tf_engine = TensorFlowRiskEngine(data["v"], data["u"], data["C"], data["x_grid"], data["H"], data["lambdas"])
        jax_engine = JAXRiskEngine(
            data["v"],
            data["u"],
            data["C"],
            data["x_grid"],
            data["H"],
            data["lambdas"],
            device=jax_device,
        )

        group_summary: Dict[str, Any] = {
            "n_assets": int(data["H"].shape[0]),
            "n_events": int(data["H"].shape[1]),
            "n_typologies": int(data["C"].shape[0]),
        }

        if "forward" in modes:
            t0 = time.perf_counter()
            tf_j, tf_metrics = tf_engine.compute_loss_and_metrics()
            tf_runtime = time.perf_counter() - t0

            eager = timed_call(jax_engine.compute_loss_and_metrics, repeat=args.repeat)
            first_jit = timed_call(
                lambda: jit_forward(
                    jax_engine.v, jax_engine.u, jax_engine.C, jax_engine.x_grid, jax_engine.H, jax_engine.lambdas
                ),
                repeat=1,
            )
            steady_jit = timed_call(
                lambda: jit_forward(
                    jax_engine.v, jax_engine.u, jax_engine.C, jax_engine.x_grid, jax_engine.H, jax_engine.lambdas
                ),
                repeat=args.repeat,
            )
            jax_j, jax_metrics = steady_jit["result"]

            group_summary["forward"] = {
                "tensorflow_runtime_s": float(tf_runtime),
                "jax_eager_timing": strip_result(eager),
                "jax_first_jit_call_timing": strip_result(first_jit),
                "jax_steady_jit_timing": strip_result(steady_jit),
                "loss_matrix": compare_arrays(tf_to_numpy(tf_j), jax_to_numpy(jax_j)),
                "aal_per_asset": compare_arrays(
                    tf_to_numpy(tf_metrics["aal_per_asset"]),
                    jax_to_numpy(jax_metrics["aal_per_asset"]),
                ),
                "loss_per_event": compare_arrays(
                    tf_to_numpy(tf_metrics["loss_per_event"]),
                    jax_to_numpy(jax_metrics["loss_per_event"]),
                ),
                "aal_portfolio_tf": float(tf_metrics["aal_portfolio"].numpy()),
                "aal_portfolio_jax": float(np.asarray(jax_metrics["aal_portfolio"])),
            }

        for mode in ("vulnerability", "exposure", "hazard", "lambdas"):
            if mode not in modes:
                continue
            tf_method = getattr(tf_engine, f"gradient_wrt_{mode}")
            t0 = time.perf_counter()
            tf_grad, _ = tf_method()
            tf_runtime = time.perf_counter() - t0

            fn = grad_fns[mode]
            jit_fn = jit_grad_fns[mode]
            args_tuple = (
                jax_engine.v,
                jax_engine.u,
                jax_engine.C,
                jax_engine.x_grid,
                jax_engine.H,
                jax_engine.lambdas,
            )
            eager = timed_call(lambda: fn(*args_tuple), repeat=args.repeat)
            first_jit = timed_call(lambda: jit_fn(*args_tuple), repeat=1)
            steady_jit = timed_call(lambda: jit_fn(*args_tuple), repeat=args.repeat)

            group_summary[f"gradient_{mode}"] = {
                "tensorflow_runtime_s": float(tf_runtime),
                "jax_eager_timing": strip_result(eager),
                "jax_first_jit_call_timing": strip_result(first_jit),
                "jax_steady_jit_timing": strip_result(steady_jit),
                "comparison": compare_arrays(tf_to_numpy(tf_grad), jax_to_numpy(steady_jit["result"])),
            }

        summary["groups"][group] = group_summary
        print(
            f"{group}: forward rel-L2="
            f"{group_summary.get('forward', {}).get('loss_matrix', {}).get('relative_l2_error', 'not-run')}"
        )

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
