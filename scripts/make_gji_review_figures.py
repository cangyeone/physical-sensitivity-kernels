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
from matplotlib.tri import Triangulation


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


def plot_example_profiles() -> None:
    path = RESULT_DIR / "fair_di_diagnostics.npz"
    if not path.exists():
        return
    data = np.load(path)
    depth = np.linspace(0.0, 400.0, data["DI_Weak_in_prior_target"].shape[-1])
    regimes = ["in_prior", "boundary", "out_of_prior"]
    labels = ["In-prior", "Boundary", "Out-of-prior"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.8), sharey=True)
    for ax, regime, label in zip(axes, regimes, labels):
        target = data[f"DI_Weak_{regime}_target"][:, 1, :]
        samples = data[f"DI_Weak_{regime}_samples"][:, :, 1, :]
        median = np.median(samples, axis=1)
        q16 = np.quantile(samples, 0.16, axis=1)
        q84 = np.quantile(samples, 0.84, axis=1)
        mae = np.mean(np.abs(median - target), axis=1)
        idx = int(np.argsort(mae)[len(mae) // 2])
        ax.fill_betweenx(depth, q16[idx], q84[idx], color=COLORS["DI-Weak"], alpha=0.20, linewidth=0)
        ax.plot(target[idx], depth, color="#111111", linewidth=1.6, label="Truth" if ax is axes[0] else None)
        ax.plot(median[idx], depth, color=COLORS["DI-Weak"], linewidth=1.8, label="Posterior median" if ax is axes[0] else None)
        ax.set_title(label)
        ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
        ax.grid(color="#E5E5E5", linewidth=0.6)
        ax.set_xlim(1.8, 6.4)
    axes[0].set_ylabel("Depth (km)")
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish(fig, "fair_di_example_profiles", ROOT / "figures/fair_di_comparison/production")


def _field_meta_to_dicts(meta: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "subarray": int(row["subarray"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "period_min": float(row["period_min"]),
            "period_max": float(row["period_max"]),
            "n_periods_used": int(row["n_periods_used"]),
            "rayleigh_mean_km_s": float(row["rayleigh_mean_km_s"]),
        }
        for row in meta
    ]


def plot_field_volume() -> None:
    path = ROOT / "field_masw_results_fair_weak/bayan_obo_masw_dnn_posterior_volume.npz"
    if not path.exists():
        return
    data = np.load(path, allow_pickle=True)
    period = data["period_s"]
    disp = data["disp"]
    mask = data["mask"]
    depth = data["depth_km"]
    meta = _field_meta_to_dicts(data["meta"])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i in range(len(disp)):
        ok = mask[i, 1].astype(bool)
        ax.plot(period[ok], disp[i, 1, ok], color="0.25", linewidth=0.45, alpha=0.25)
    ray = np.where(mask[:, 1, :].astype(bool), disp[:, 1, :], np.nan)
    valid = np.isfinite(ray).any(axis=0)
    ax.plot(period[valid], np.nanmedian(ray[:, valid], axis=0), color="#D55E00", linewidth=2.0, label="Median")
    ax.set_xlim(float(period[valid].min()), float(period[valid].max()))
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Rayleigh phase velocity (km s$^{-1}$)")
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    finish(fig, "fair_field_dispersion_qc", ROOT / "figures/fair_di_comparison/production/field")

    def slice_grid(field: np.ndarray, label: str, stem: str) -> None:
        lon = np.asarray([m["lon"] for m in meta])
        lat = np.asarray([m["lat"] for m in meta])
        tri = Triangulation(lon, lat)
        depths = [5.0, 10.0, 20.0, 40.0]
        idxs = [int(np.argmin(np.abs(depth - z))) for z in depths]
        scale = np.concatenate([field[:, iz] for iz in idxs])
        vmin = float(np.nanpercentile(scale, 2))
        vmax = float(np.nanpercentile(scale, 98))
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8), sharex=True, sharey=True)
        last = None
        for ax, z, iz in zip(axes.ravel(), depths, idxs):
            last = ax.tricontourf(tri, field[:, iz], levels=18, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.tricontour(tri, field[:, iz], colors="k", linewidths=0.25, alpha=0.35)
            ax.scatter(lon, lat, s=5, c="k", alpha=0.35)
            ax.set_title(f"{z:g} km")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#E5E5E5", linewidth=0.5)
        for ax in axes[-1]:
            ax.set_xlabel("Longitude")
        for ax in axes[:, 0]:
            ax.set_ylabel("Latitude")
        cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.90, pad=0.02)
        cbar.set_label(label)
        finish(fig, stem, ROOT / "figures/fair_di_comparison/production/field")

    slice_grid(data["median"][:, 1, :], r"Posterior median $V_S$ (km s$^{-1}$)", "fair_field_vs_median_slices")
    slice_grid(data["std"][:, 1, :], r"Posterior std $V_S$ (km s$^{-1}$)", "fair_field_vs_std_slices")


def plot_field_summary() -> None:
    summary_path = RESULT_DIR / "field/field_summary.csv"
    comparison_path = RESULT_DIR / "field/field_reference_comparison.csv"
    if not summary_path.exists():
        return
    summary = read_csv(summary_path)
    depth = np.asarray([as_float(r, "depth_km") for r in summary])
    vs = np.asarray([as_float(r, "vs_median_mean") for r in summary])
    std = np.asarray([as_float(r, "posterior_std_mean") for r in summary])
    fig, ax = plt.subplots(figsize=(3.4, 5.2))
    ax.plot(vs, depth, color=COLORS["DI-Weak"], label="Field mean median")
    ax.fill_betweenx(depth, vs - std, vs + std, color=COLORS["DI-Weak"], alpha=0.18, label="Mean posterior std")
    ax.invert_yaxis()
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    finish(fig, "fair_field_summary_vs_depth", ROOT / "figures/fair_di_comparison/production/field")

    if comparison_path.exists():
        comparison = read_csv(comparison_path)
        if comparison:
            depth = np.asarray([as_float(r, "depth_km") for r in comparison])
            mae = np.asarray([as_float(r, "vs_difference_mae_km_s") for r in comparison])
            fig, ax = plt.subplots(figsize=(3.4, 5.2))
            ax.plot(mae, depth, color=COLORS["IND-FWD"])
            ax.invert_yaxis()
            ax.set_xlabel(r"Reference difference MAE (km s$^{-1}$)")
            ax.set_ylabel("Depth (km)")
            ax.grid(color="#E5E5E5", linewidth=0.6)
            fig.tight_layout()
            finish(fig, "fair_field_reference_difference", ROOT / "figures/fair_di_comparison/production/field")


def main() -> None:
    plot_metric_summary()
    plot_example_profiles()
    plot_reliability()
    plot_missing_band()
    plot_noise()
    plot_baselines()
    plot_field_volume()
    plot_field_summary()
    print(f"Wrote review figures to {GJI_FIG_DIR}")


if __name__ == "__main__":
    main()
