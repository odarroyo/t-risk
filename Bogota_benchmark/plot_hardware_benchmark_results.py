#!/usr/bin/env python3
"""
Create plots from standalone Bogotá hardware benchmark JSON files.

Assumes the T-Risk and OpenQuake summary JSON files are stored in the same
directory. The script discovers them automatically unless paths are supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_SPECS = {
    "forward": ("forward", "forward", "Forward"),
    "vulnerability": ("vulnerability_gradient", "vulnerability_gradient_fd", "Vulnerability Gradient"),
    "exposure": ("exposure_gradient", "exposure_gradient_fd", "Exposure Gradient"),
    "hazard": ("hazard_gradient", "hazard_gradient_fd", "Hazard Gradient"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory containing trisk_hardware_benchmark_summary.json and openquake_hardware_benchmark_summary.json.",
    )
    parser.add_argument("--trisk-json", default=None)
    parser.add_argument("--openquake-json", default=None)
    parser.add_argument("--out-dir", default=None, help="Defaults to <results-dir>/figures.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_json(results_dir: Path, explicit: str | None, prefix: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    candidates = sorted(results_dir.glob(f"{prefix}*benchmark*.json"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix} benchmark JSON found in {results_dir}")
    return candidates[0].resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_group(group: str) -> str:
    return group.replace("SA_", "SA(").replace("p", ".") + "s)"


def timing_mean(entry: dict | None) -> float:
    if not entry:
        return float("nan")
    timing = entry.get("timing")
    if not timing:
        return float("nan")
    return float(timing.get("mean_s", np.nan))


def vulnerability_timing_mean(entry: dict | None) -> float:
    if not entry:
        return float("nan")
    selected = entry.get("selected_taxonomies", [])
    if not selected:
        return float("nan")
    times = [timing_mean(item) for item in selected]
    times = [x for x in times if np.isfinite(x)]
    return float(np.sum(times)) if times else float("nan")


def collect_rows(trisk: dict, oq: dict) -> list[dict]:
    groups = sorted(set(trisk.get("groups", {})) | set(oq.get("groups", {})))
    rows = []
    for group in groups:
        tg = trisk.get("groups", {}).get(group, {})
        og = oq.get("groups", {}).get(group, {})
        row = {
            "group": group,
            "label": format_group(group),
            "n_assets": tg.get("n_assets", og.get("n_assets")),
            "n_events": tg.get("n_events", og.get("n_events")),
        }
        for metric, (t_key, o_key, _) in METRIC_SPECS.items():
            if metric == "vulnerability":
                t_time = timing_mean(tg.get(t_key))
                o_time = vulnerability_timing_mean(og.get(o_key))
            else:
                t_time = timing_mean(tg.get(t_key))
                o_time = timing_mean(og.get(o_key))
            row[f"{metric}_trisk_s"] = t_time
            row[f"{metric}_openquake_s"] = o_time
            row[f"{metric}_speedup"] = o_time / t_time if np.isfinite(t_time) and t_time > 0 else float("nan")
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "group",
        "n_assets",
        "n_events",
        "forward_trisk_s",
        "forward_openquake_s",
        "forward_speedup",
        "vulnerability_trisk_s",
        "vulnerability_openquake_s",
        "vulnerability_speedup",
        "exposure_trisk_s",
        "exposure_openquake_s",
        "exposure_speedup",
        "hazard_trisk_s",
        "hazard_openquake_s",
        "hazard_speedup",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_metric_runtimes(rows: list[dict], out_dir: Path) -> list[str]:
    paths = []
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36

    for metric, (_, _, title) in METRIC_SPECS.items():
        trisk_times = np.array([row[f"{metric}_trisk_s"] for row in rows], dtype=float)
        oq_times = np.array([row[f"{metric}_openquake_s"] for row in rows], dtype=float)
        if not np.any(np.isfinite(trisk_times)) or not np.any(np.isfinite(oq_times)):
            continue

        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.bar(x - width / 2, trisk_times, width=width, color="#0B6E4F", label="T-Risk")
        ax.bar(x + width / 2, oq_times, width=width, color="#C97C10", label="OpenQuake")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Runtime (s, log scale)")
        ax.set_title(f"{title} Runtime by IM Group")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{metric}_runtime_by_im.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_speedup_summary(rows: list[dict], out_dir: Path) -> str:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    available = []
    for metric, (_, _, title) in METRIC_SPECS.items():
        vals = np.array([row[f"{metric}_speedup"] for row in rows], dtype=float)
        if np.any(np.isfinite(vals)):
            available.append((metric, title, vals))

    width = min(0.8 / max(len(available), 1), 0.22)
    offsets = (np.arange(len(available)) - (len(available) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#5E548E", "#7B2CBF", "#2A9D8F", "#B33A3A"]
    for i, (metric, title, vals) in enumerate(available):
        ax.bar(x + offsets[i], vals, width=width, label=title, color=colors[i % len(colors)])
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("OpenQuake / T-Risk Runtime (x, log scale)")
    ax.set_title("Runtime Speedup by IM Group")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "speedup_summary_by_im.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_hazard_pair_scaling(rows: list[dict], out_dir: Path) -> str | None:
    valid = [
        row
        for row in rows
        if np.isfinite(row.get("hazard_trisk_s", np.nan)) and np.isfinite(row.get("hazard_openquake_s", np.nan))
    ]
    if not valid:
        return None
    pairs = np.array([row["n_assets"] * row["n_events"] for row in valid], dtype=float)
    trisk = np.array([row["hazard_trisk_s"] for row in valid], dtype=float)
    oq = np.array([row["hazard_openquake_s"] for row in valid], dtype=float)
    labels = [row["label"] for row in valid]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(pairs, trisk, s=80, color="#0B6E4F", label="T-Risk")
    ax.scatter(pairs, oq, s=80, color="#C97C10", label="OpenQuake")
    for p, t, o, label in zip(pairs, trisk, oq, labels):
        ax.annotate(label, (p, max(t, o)), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hazard Gradient Size (assets x events)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Hazard-Gradient Runtime Scaling")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "hazard_gradient_runtime_scaling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_hazard_time_share(rows: list[dict], out_dir: Path) -> str | None:
    valid = [
        row
        for row in rows
        if np.isfinite(row.get("hazard_trisk_s", np.nan)) and np.isfinite(row.get("hazard_openquake_s", np.nan))
    ]
    if not valid:
        return None
    labels = [row["label"] for row in valid]
    trisk = np.array([row["hazard_trisk_s"] for row in valid], dtype=float)
    oq = np.array([row["hazard_openquake_s"] for row in valid], dtype=float)
    totals = trisk + oq
    x = np.arange(len(valid))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x, trisk / totals * 100, color="#0B6E4F", label="T-Risk")
    ax.bar(x, oq / totals * 100, bottom=trisk / totals * 100, color="#C97C10", label="OpenQuake")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of Combined Hazard-Gradient Runtime (%)")
    ax.set_title("Hazard-Gradient Runtime Share")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "hazard_gradient_runtime_share.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir / "figures"
    ensure_dir(out_dir)

    trisk_path = find_json(results_dir, args.trisk_json, "trisk")
    oq_path = find_json(results_dir, args.openquake_json, "openquake")
    trisk = load_json(trisk_path)
    oq = load_json(oq_path)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )

    rows = collect_rows(trisk, oq)
    csv_path = out_dir / "hardware_benchmark_timing_summary.csv"
    write_csv(rows, csv_path)

    figures = []
    figures.extend(plot_metric_runtimes(rows, out_dir))
    figures.append(plot_speedup_summary(rows, out_dir))
    hazard_scaling = plot_hazard_pair_scaling(rows, out_dir)
    hazard_share = plot_hazard_time_share(rows, out_dir)
    if hazard_scaling:
        figures.append(hazard_scaling)
    if hazard_share:
        figures.append(hazard_share)

    summary = {
        "trisk_json": str(trisk_path),
        "openquake_json": str(oq_path),
        "csv": str(csv_path),
        "figures": figures,
        "rows": rows,
    }
    summary_path = out_dir / "hardware_benchmark_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
