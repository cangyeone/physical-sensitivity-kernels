#!/usr/bin/env python3
"""Regenerate GJI review figures from production fair-comparison tables.

The evaluation scripts create the numerical CSV/JSON products.  This plotting
script is intentionally table-driven so manuscript figures can be regenerated
without re-running model inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/fair_di_comparison/production"
GJI_FIG_DIR = ROOT / "gji_dnn_posterior_inversion/figures"

COLORS = {
    "DI-Strong": "#0072B2",
    "DI-Weak": "#009E73",
    "DET-Strong": "#56B4E9",
    "DET-Weak": "#E69F00",
    "IND-FWD": "#D55E00",
    "raw": "#0072B2",
    "temperature_scaled": "#D55E00",
}
REGIMES = ["in-prior", "boundary", "out-of-prior"]
REGIME_LABELS = ["In-prior", "Boundary", "Out-of-prior"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def finish(fig: plt.Figure, stem: str, *extra_dirs: Path) -> None:
    for out_dir in (GJI_FIG_DIR, *extra_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary() -> None:
    rows = read_csv(RESULT_DIR / "fair_di_metrics.csv")
    metrics = [
        ("vs_mae", r"$V_S$ MAE (km s$^{-1}$)"),
        ("pred_disp_mae", r"Dispersion MAE (km s$^{-1}$)"),
        ("coverage_16_84_mean", "16--84% coverage"),
        ("boundary_pull_fraction", "Pull-in fraction"),
    ]
    methods = ["DI-Strong", "DI-Weak"]
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.0))
    x = np.arange(len(REGIMES))
    width = 0.36
    for ax, (metric, ylabel) in zip(axes, metrics):
        for j, method in enumerate(methods):
            vals = []
            lo = []
            hi = []
            for regime in REGIMES:
                row = next(r for r in rows if r["method"] == method and r["test_set"] == regime)
                vals.append(as_float(row, metric))
                lo_key = f"{metric}_ci_low"
                hi_key = f"{metric}_ci_high"
                if lo_key in row and row[lo_key]:
                    lo.append(vals[-1] - as_float(row, lo_key))
                    hi.append(as_float(row, hi_key) - vals[-1])
                else:
                    lo.append(0.0)
                    hi.append(0.0)
            pos = x + (j - 0.5) * width
            ax.bar(pos, vals, width=width, color=COLORS[method], alpha=0.85, label=method if ax is axes[0] else None)
            if any(v > 0 for v in lo + hi):
                ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
        if metric == "coverage_16_84_mean":
            ax.axhline(0.68, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_LABELS, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    finish(fig, "fair_di_metric_summary", ROOT / "figures/fair_di_comparison/production")


def plot_reliability() -> None:
    rows = read_csv(RESULT_DIR / "calibration/calibration_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6), sharex=True, sharey=True)
    for i, method in enumerate(["DI-Strong", "DI-Weak"]):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            ax.plot([0, 100], [0, 1], color="#333333", linestyle="--", linewidth=0.8, label="Nominal" if i == 0 and j == 0 else None)
            for scale, linestyle, marker in [("raw", "-", "o"), ("temperature_scaled", ":", "s")]:
                subset = [
                    r
                    for r in rows
                    if r["method"] == method and r["test_set"] == regime and r["scale_label"] == scale
                ]
                subset.sort(key=lambda r: as_float(r, "nominal_percent"))
                ax.plot(
                    [as_float(r, "nominal_percent") for r in subset],
                    [as_float(r, "coverage_mean") for r in subset],
                    color=COLORS[scale],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    label=scale.replace("_", " ") if i == 0 and j == 0 else None,
                )
            ax.set_title(f"{method}, {REGIME_LABELS[j]}")
            ax.grid(color="#E5E5E5", linewidth=0.6)
            if i == 1:
                ax.set_xlabel("Nominal interval (%)")
            if j == 0:
                ax.set_ylabel("Held-out coverage")
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish(fig, "fair_calibration_reliability", ROOT / "figures/fair_di_comparison/production/calibration")


def plot_missing_band() -> None:
    rows = read_csv(RESULT_DIR / "missing_band/missing_band_uncertainty.csv")
    labels = [r["window"].replace("-", "\n") for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.9))
    specs = [
        ("valid_fraction_mean", "Valid period fraction", "#999999"),
        ("vs_mae", r"$V_S$ MAE (km s$^{-1}$)", "#0072B2"),
        ("vs_spread_mean", r"$V_S$ posterior std (km s$^{-1}$)", "#009E73"),
        ("coverage_vs", r"$V_S$ 16--84% coverage", "#D55E00"),
    ]
    for ax, (key, ylabel, color) in zip(axes, specs):
        ax.bar(x, [as_float(r, key) for r in rows], color=color, alpha=0.85)
        if key == "coverage_vs":
            ax.axhline(0.68, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    finish(fig, "fair_missing_band_uncertainty", ROOT / "figures/fair_di_comparison/production/missing_band")


def plot_noise() -> None:
    rows = read_csv(RESULT_DIR / "noise/noise_sensitivity.csv")
    rows = [r for r in rows if r["test_set"] == "in-prior"]
    metrics = [
        ("vs_mae", r"$V_S$ MAE (km s$^{-1}$)"),
        ("coverage_16_84_mean", "16--84% coverage"),
        ("posterior_std_vs", r"$V_S$ posterior std (km s$^{-1}$)"),
        ("temperature_scale_68", r"Temperature $\tau_{68}$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.0))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in ["DI-Strong", "DI-Weak"]:
            subset = [r for r in rows if r["method"] == method]
            subset.sort(key=lambda r: as_float(r, "noise_sigma_km_s"))
            ax.plot(
                [as_float(r, "noise_sigma_km_s") for r in subset],
                [as_float(r, metric) for r in subset],
                color=COLORS[method],
                marker="o",
                linewidth=1.7,
                label=method if ax is axes[0] else None,
            )
        if metric == "coverage_16_84_mean":
            ax.axhline(0.68, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"Added noise $\sigma$ (km s$^{-1}$)")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E5E5E5", linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    finish(fig, "fair_noise_sensitivity", ROOT / "figures/fair_di_comparison/production/noise")


def plot_baselines() -> None:
    rows = read_csv(RESULT_DIR / "baselines/baseline_metrics.csv")
    methods = ["DET-Strong", "DET-Weak", "IND-FWD"]
    metrics = [("vs_mae", r"$V_S$ MAE (km s$^{-1}$)"), ("pred_disp_mae", r"Dispersion MAE (km s$^{-1}$)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(REGIMES))
    width = 0.24
    for ax, (metric, ylabel) in zip(axes, metrics):
        for j, method in enumerate(methods):
            vals = []
            for regime in REGIMES:
                row = next(r for r in rows if r["method"] == method and r["test_set"] == regime)
                vals.append(as_float(row, metric))
            ax.bar(x + (j - 1) * width, vals, width=width, color=COLORS[method], alpha=0.85, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_LABELS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.15), loc="upper left")
    fig.tight_layout()
    finish(fig, "fair_baseline_metric_summary", ROOT / "figures/fair_di_comparison/production/baselines")


def main() -> None:
    plot_metric_summary()
    plot_reliability()
    plot_missing_band()
    plot_noise()
    plot_baselines()
    print(f"Wrote review figures to {GJI_FIG_DIR}")


if __name__ == "__main__":
    main()
