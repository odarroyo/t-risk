#!/usr/bin/env python3
"""
Standalone Bogotá OpenQuake risk-library hardware benchmark.

Run this script inside an OpenQuake environment. It uses direct in-memory calls
to OpenQuake risk-library functions; it does not use `oq engine --run`.
Input files are expected in the same directory as this script unless paths are
provided explicitly.
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
from openquake.risklib.scientific import VulnerabilityFunction


DEFAULT_GROUPS = ("SA_0p1", "SA_0p3", "SA_0p6", "SA_1p0")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(script_dir / "bogota_trisk_inputs.npz"))
    parser.add_argument("--hazard", default=str(script_dir / "bogota_hazard_chia.npz"))
    parser.add_argument("--out", default=str(script_dir / "openquake_hardware_benchmark_summary.json"))
    parser.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["forward", "vulnerability", "exposure"],
        choices=("forward", "vulnerability", "exposure", "hazard", "all"),
        help="Benchmark modes. 'all' expands to every mode. Hazard FD can take minutes.",
    )
    parser.add_argument("--taxonomy-mode", choices=("most_common", "all"), default="most_common")
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--fd-rel-eps", type=float, default=0.01)
    parser.add_argument("--fd-abs-eps", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
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


def group_imt(group: str) -> str:
    return group.replace("_", "(").replace("p", ".") + ")"


def make_vulnerability(curve_id: str, imt: str, x_grid: np.ndarray, curve: np.ndarray) -> VulnerabilityFunction:
    vf = VulnerabilityFunction(
        vf_id=curve_id,
        imt=imt,
        imls=x_grid.astype(np.float64),
        mean_loss_ratios=curve.astype(np.float64),
        covs=np.zeros_like(curve, dtype=np.float64),
        distribution="LN",
    )
    vf.init()
    return vf


def oq_mean_loss_ratios(vf: VulnerabilityFunction, gmvs: np.ndarray) -> np.ndarray:
    gmvs = np.asarray(gmvs, dtype=np.float64)
    gmvs_clip = np.minimum(gmvs, vf.imls[-1])
    out = np.zeros_like(gmvs_clip, dtype=np.float64)
    ok = gmvs_clip >= vf.imls[0]
    if np.any(ok):
        out[ok] = vf._mlr_i1d(gmvs_clip[ok])
    return out


def load_group(inputs, hazard, group: str, max_assets: int | None, max_events: int | None) -> dict:
    v = inputs[f"{group}_v"].astype(np.float64)
    u = inputs[f"{group}_u"].astype(np.int32)
    c = inputs[f"{group}_C"].astype(np.float64)
    x_grid = inputs[f"{group}_x_grid"].astype(np.float64)
    h = hazard[f"H_{group}"].astype(np.float64)
    lambdas = hazard["lambdas"].astype(np.float64)
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


def compute_forward(data: dict, group: str) -> dict:
    v = data["v"]
    u = data["u"]
    c = data["C"]
    x_grid = data["x_grid"]
    h = data["H"]
    lambdas = data["lambdas"]
    labels = data["labels"]
    imt = group_imt(group)

    loss_per_event = np.zeros(h.shape[1], dtype=np.float64)
    aal_per_asset = np.zeros(h.shape[0], dtype=np.float64)
    for taxonomy_id, label in enumerate(labels):
        idx = np.flatnonzero(u == taxonomy_id)
        if idx.size == 0:
            continue
        vf = make_vulnerability(str(label), imt, x_grid, c[taxonomy_id])
        mdr = oq_mean_loss_ratios(vf, h[idx, :])
        loss = v[idx, None] * mdr
        loss_per_event += loss.sum(axis=0)
        aal_per_asset[idx] = loss @ lambdas
    return {
        "portfolio_aal": float(loss_per_event @ lambdas),
        "aal_per_asset": aal_per_asset,
        "loss_per_event": loss_per_event,
    }


def vulnerability_fd_gradient(data: dict, group: str, taxonomy_id: int, fd_rel_eps: float, fd_abs_eps: float) -> np.ndarray:
    v = data["v"]
    u = data["u"]
    c = data["C"]
    x_grid = data["x_grid"]
    h = data["H"]
    lambdas = data["lambdas"]
    labels = data["labels"]
    idx = np.flatnonzero(u == taxonomy_id)
    imt = group_imt(group)
    grad = np.zeros(c.shape[1], dtype=np.float64)
    if idx.size == 0:
        return grad

    base_vf = make_vulnerability(str(labels[taxonomy_id]), imt, x_grid, c[taxonomy_id])
    base_mdr = oq_mean_loss_ratios(base_vf, h[idx, :])
    base_aal = float(((v[idx, None] * base_mdr) @ lambdas).sum())

    for point_id in range(c.shape[1]):
        curve = c[taxonomy_id].copy()
        delta = max(fd_abs_eps, fd_rel_eps * abs(curve[point_id]))
        curve[point_id] += delta
        pert_vf = make_vulnerability(str(labels[taxonomy_id]), imt, x_grid, curve)
        pert_mdr = oq_mean_loss_ratios(pert_vf, h[idx, :])
        pert_aal = float(((v[idx, None] * pert_mdr) @ lambdas).sum())
        grad[point_id] = (pert_aal - base_aal) / delta
    return grad


def exposure_fd_gradient(data: dict, group: str) -> np.ndarray:
    u = data["u"]
    c = data["C"]
    x_grid = data["x_grid"]
    h = data["H"]
    lambdas = data["lambdas"]
    labels = data["labels"]
    imt = group_imt(group)
    grad = np.zeros(h.shape[0], dtype=np.float64)
    for taxonomy_id, label in enumerate(labels):
        idx = np.flatnonzero(u == taxonomy_id)
        if idx.size == 0:
            continue
        vf = make_vulnerability(str(label), imt, x_grid, c[taxonomy_id])
        mdr = oq_mean_loss_ratios(vf, h[idx, :])
        grad[idx] = mdr @ lambdas
    return grad


def hazard_fd_gradient(data: dict, group: str, fd_rel_eps: float, fd_abs_eps: float) -> np.ndarray:
    v = data["v"]
    u = data["u"]
    c = data["C"]
    x_grid = data["x_grid"]
    h = data["H"]
    lambdas = data["lambdas"]
    labels = data["labels"]
    imt = group_imt(group)
    grad = np.zeros_like(h, dtype=np.float64)
    vf_cache = {
        taxonomy_id: make_vulnerability(str(label), imt, x_grid, c[taxonomy_id])
        for taxonomy_id, label in enumerate(labels)
    }

    for asset_id in range(h.shape[0]):
        vf = vf_cache[int(u[asset_id])]
        base_mdr = oq_mean_loss_ratios(vf, h[asset_id, :])
        for event_id in range(h.shape[1]):
            base_h = h[asset_id, event_id]
            delta = max(fd_abs_eps, fd_rel_eps * abs(base_h))
            pert_h = min(float(x_grid[-1]), base_h + delta)
            if pert_h == base_h:
                pert_h = base_h + delta
            pert_mdr = oq_mean_loss_ratios(vf, np.array([pert_h], dtype=np.float64))[0]
            grad[asset_id, event_id] = v[asset_id] * (pert_mdr - base_mdr[event_id]) * lambdas[event_id] / max(
                pert_h - base_h, 1e-12
            )
    return grad


def main() -> int:
    args = parse_args()
    modes = expand_modes(args.modes)

    load_t0 = time.perf_counter()
    inputs = np.load(args.inputs, allow_pickle=True)
    hazard = np.load(args.hazard, allow_pickle=True)
    load_runtime_s = time.perf_counter() - load_t0

    summary = {
        "runner": "run_openquake_hardware_benchmark.py",
        "execution_mode": "direct in-memory calls to OpenQuake risk-library functions, not oq engine exports",
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": str(Path(args.inputs).resolve()),
        "hazard": str(Path(args.hazard).resolve()),
        "load_runtime_s": load_runtime_s,
        "modes": modes,
        "fd_rel_eps": args.fd_rel_eps,
        "fd_abs_eps": args.fd_abs_eps,
        "groups": {},
    }

    for group in args.groups:
        data = load_group(inputs, hazard, group, args.max_assets, args.max_events)
        group_summary = {
            "n_assets": int(data["H"].shape[0]),
            "n_events": int(data["H"].shape[1]),
            "n_taxonomies": int(data["C"].shape[0]),
        }

        if "forward" in modes:
            forward, timing = time_call(lambda: compute_forward(data, group), args.warmup, args.repeat)
            group_summary["forward"] = {
                "portfolio_aal": forward["portfolio_aal"],
                "timing": timing,
                "aal_per_asset_shape": list(forward["aal_per_asset"].shape),
                "loss_per_event_shape": list(forward["loss_per_event"].shape),
            }

        if "vulnerability" in modes:
            selected = []
            for taxonomy_id in choose_taxonomies(data["u"], args.taxonomy_mode):
                grad, timing = time_call(
                    lambda tid=taxonomy_id: vulnerability_fd_gradient(
                        data, group, tid, args.fd_rel_eps, args.fd_abs_eps
                    ),
                    args.warmup,
                    args.repeat,
                )
                selected.append(
                    {
                        "taxonomy_id": taxonomy_id,
                        "taxonomy_label": str(data["labels"][taxonomy_id]),
                        "asset_count": int(np.sum(data["u"] == taxonomy_id)),
                        "gradient_shape": list(grad.shape),
                        "curve_l2_norm": float(np.linalg.norm(grad)),
                        "curve_max_abs": float(np.max(np.abs(grad))),
                        "timing": timing,
                    }
                )
            group_summary["vulnerability_gradient_fd"] = {"selected_taxonomies": selected}

        if "exposure" in modes:
            grad, timing = time_call(lambda: exposure_fd_gradient(data, group), args.warmup, args.repeat)
            group_summary["exposure_gradient_fd"] = {
                "gradient_shape": list(grad.shape),
                "l2_norm": float(np.linalg.norm(grad)),
                "timing": timing,
            }

        if "hazard" in modes:
            grad, timing = time_call(
                lambda: hazard_fd_gradient(data, group, args.fd_rel_eps, args.fd_abs_eps),
                args.warmup,
                args.repeat,
            )
            group_summary["hazard_gradient_fd"] = {
                "gradient_shape": list(grad.shape),
                "l2_norm": float(np.linalg.norm(grad)),
                "timing": timing,
            }

        summary["groups"][group] = group_summary

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2))
    print(out_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
