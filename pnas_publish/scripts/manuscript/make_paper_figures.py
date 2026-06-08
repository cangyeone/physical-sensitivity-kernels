#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch, Polygon
import numpy as np
import torch
import torch.nn.functional as F


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
OUT = OVERLEAF / "figures"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.generate_data import compute_phase_dispersion

PDF_METADATA = {
    "Creator": "make_paper_figures.py",
    "CreationDate": datetime(2026, 6, 3, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 6, 3, tzinfo=timezone.utc),
}

FULL_WIDTH_IN = 7.1
HALF_WIDTH_IN = 3.55
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#111111",
    "gray": "#6E6E6E",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.8,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.25,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax, grid=True):
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.2")
    if grid:
        ax.grid(color="0.88", lw=0.55, alpha=1.0)


def panel_label(ax, label, text=None, x=0.025, y=0.965):
    body = f"({label})" if text is None else f"({label}) {text}"
    ax.text(
        x,
        y,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        zorder=10,
    )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_epoch_log(path):
    rows = []
    pat = re.compile(
        r"\[Epoch\s+(\d+)/(\d+)\].*?"
        r"train: loss=([0-9.]+), flow=([0-9.]+), rec=([0-9.]+), mae=([0-9.]+), "
        r"vp=([0-9.]+), vs=([0-9.]+), rho=([0-9.]+).*?"
        r"val: loss=([0-9.]+), flow=([0-9.]+), sample_mae=([0-9.]+), "
        r"sample_vp=([0-9.]+), sample_vs=([0-9.]+), sample_rho=([0-9.]+)"
    )
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        vals = list(map(float, m.groups()))
        rows.append(
            {
                "epoch": int(vals[0]),
                "epochs": int(vals[1]),
                "train_loss": vals[2],
                "train_flow": vals[3],
                "train_rec": vals[4],
                "train_mae": vals[5],
                "train_vp": vals[6],
                "train_vs": vals[7],
                "train_rho": vals[8],
                "val_loss": vals[9],
                "val_flow": vals[10],
                "sample_mae": vals[11],
                "sample_vp": vals[12],
                "sample_vs": vals[13],
                "sample_rho": vals[14],
            }
        )
    return rows


def restore_model(module, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = module.Disp2StructCRF(
        H=len(ckpt["depth_grid"]),
        T=59,
        profile_channels=3,
        cond_base_channels=cfg["cond_base_channels"],
        cond_dim=cfg["cond_dim"],
        flow_hidden=cfg["flow_hidden"],
        time_dim=cfg["time_dim"],
        dropout=0.0,
        reference_profile=ckpt["reference_profile"],
        profile_scale=ckpt["profile_scale"],
        depth_grid=ckpt["depth_grid"],
        period_minmax=tuple(float(x) for x in ckpt["period_minmax"].tolist()),
        disp_mean=ckpt["disp_mean"],
        disp_scale=ckpt["disp_scale"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def collect_dataset(module, n=64):
    ds = module.SurfaceWaveDataset(
        n_samples=n,
        z_max_km=150.0,
        z_max_num=256,
        dz_km=0.5,
        seed=2026 + 1_000_000,
    )
    models, disps, masks = [], [], []
    idx = 0
    while len(models) < n:
        try:
            m, d, mask = ds[idx]
        except Exception:
            idx += 1
            continue
        models.append(m)
        disps.append(d)
        masks.append(mask)
        idx += 1
    return torch.stack(models), torch.stack(disps), torch.stack(masks)


def roughness(profile):
    d2 = profile[..., 2:] - 2.0 * profile[..., 1:-1] + profile[..., :-2]
    return d2.abs().mean(dim=-1)


def nominal_coverage(samples, target, qs):
    cover = []
    for q in qs:
        lo = 50.0 - q / 2.0
        hi = 50.0 + q / 2.0
        qlo = torch.quantile(samples, lo / 100.0, dim=1)
        qhi = torch.quantile(samples, hi / 100.0, dim=1)
        cover.append(((target >= qlo) & (target <= qhi)).float().mean(dim=(0, 2)).numpy())
    return np.asarray(cover)


def scaled_samples_about_median(samples, scale):
    median = samples.median(dim=1, keepdim=True).values
    return median + scale * (samples - median)


def interval_coverage_by_example(samples, target, nominal_percent=68.0, scale=1.0):
    scaled = scaled_samples_about_median(samples, scale)
    lo = 50.0 - nominal_percent / 2.0
    hi = 50.0 + nominal_percent / 2.0
    qlo = torch.quantile(scaled, lo / 100.0, dim=1)
    qhi = torch.quantile(scaled, hi / 100.0, dim=1)
    return ((target >= qlo) & (target <= qhi)).float().mean(dim=2)


def mean_interval_coverage(samples, target, nominal_percent=68.0, scale=1.0):
    return float(interval_coverage_by_example(samples, target, nominal_percent, scale).mean())


def fit_temperature_scale(samples, target, nominal_percent=68.0, target_coverage=None):
    target_coverage = nominal_percent / 100.0 if target_coverage is None else target_coverage
    if mean_interval_coverage(samples, target, nominal_percent, scale=1.0) >= target_coverage:
        return 1.0

    lo, hi = 1.0, 1.5
    while mean_interval_coverage(samples, target, nominal_percent, scale=hi) < target_coverage and hi < 16.0:
        lo = hi
        hi *= 1.5

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if mean_interval_coverage(samples, target, nominal_percent, scale=mid) < target_coverage:
            lo = mid
        else:
            hi = mid
    return float(hi)


def bootstrap_ci(values, n_boot=2000, seed=2026):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    boot = np.empty((n_boot, arr.shape[1]), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = arr[idx].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return lo, hi


def channel_ci_dict(lo, hi):
    return {
        "Vp": [float(lo[0]), float(hi[0])],
        "Vs": [float(lo[1]), float(hi[1])],
        "rho": [float(lo[2]), float(hi[2])],
    }


def savefig(name):
    path = OUT / name
    metadata = PDF_METADATA if path.suffix.lower() == ".pdf" else None
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    plt.close()
    return str(path)


def fig_workflow():
    rng = np.random.default_rng(2026)
    fig = plt.figure(figsize=(6.65, 5.05))

    def panel(rect, label, title, facecolor):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.006, 0.006),
                0.988,
                0.988,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                linewidth=0.7,
                edgecolor="0.72",
                facecolor=facecolor,
                zorder=-5,
            )
        )
        ax.text(
            0.025,
            0.965,
            label,
            ha="left",
            va="top",
            fontsize=8.7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="0.75", linewidth=0.5),
        )
        ax.text(0.118, 0.965, title, ha="left", va="top", fontsize=7.5, fontweight="bold")
        return ax

    def mini_axis(rect):
        ax = fig.add_axes(rect)
        ax.tick_params(direction="out", length=2.2, width=0.55, pad=1.0, labelsize=5.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("0.35")
        ax.grid(color="0.9", lw=0.35)
        return ax

    def in_panel(base, x, y, w, h):
        return [base[0] + x * base[2], base[1] + y * base[3], w * base[2], h * base[3]]

    def draw_profiles(ax, x0, y0, w, h, alpha=1.0):
        depth = np.linspace(0, 1, 90)
        bases = [0.18, 0.50, 0.80]
        colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
        labels = [r"$V_P$", r"$V_S$", r"$\rho$"]
        for j, (base, color, lab) in enumerate(zip(bases, colors, labels)):
            for k in range(4):
                noise = 0.015 * rng.standard_normal(depth.size)
                curve = base + 0.11 * np.tanh((depth - 0.25 - 0.04 * j) * 5) + noise.cumsum() / 70
                ax.plot(x0 + w * curve, y0 + h * (1 - depth), color=color, lw=0.55, alpha=0.18 * alpha)
            curve = base + 0.11 * np.tanh((depth - 0.25 - 0.04 * j) * 5)
            ax.plot(x0 + w * curve, y0 + h * (1 - depth), color=color, lw=1.0, alpha=alpha)
            ax.text(x0 + w * base, y0 - 0.055 * h, lab, ha="center", va="top", fontsize=5.9)
        ax.plot([x0 + 0.02 * w, x0 + 0.02 * w], [y0, y0 + h], color="0.72", lw=0.45)
        ax.text(x0 - 0.010, y0 + 0.50 * h, "Depth", ha="right", va="center", rotation=90, fontsize=5.8)

    def arrow(ax, x0, y0, x1, y1, color="0.18", lw=0.8):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    a_rect = [0.015, 0.665, 0.505, 0.315]
    b_rect = [0.525, 0.665, 0.255, 0.315]
    c_rect = [0.785, 0.665, 0.200, 0.315]
    d_rect = [0.015, 0.375, 0.675, 0.280]
    e_rect = [0.695, 0.375, 0.290, 0.280]
    f_rect = [0.015, 0.065, 0.970, 0.295]

    axa = panel(a_rect, "A", "Prior and simulator (offline)", "#f6fbff")
    axa.text(0.055, 0.80, "Tectonic\nprior", fontsize=5.9, fontweight="bold", linespacing=0.90)
    classes = [
        ("Oceanic", "#7db7c7"),
        ("Shield", "#d9c989"),
        ("Platform", "#8fc46c"),
        ("Orogen", "#c98f5a"),
        ("Rift", "#d86b5f"),
    ]
    for i, (name, color) in enumerate(classes):
        y = 0.68 - i * 0.105
        axa.add_patch(plt.Rectangle((0.045, y), 0.040, 0.065, facecolor=color, edgecolor="0.55", lw=0.4))
        for _ in range(3):
            xs = np.linspace(0.049, 0.081, 18)
            ys = y + 0.018 + 0.035 * rng.random() + 0.006 * np.sin(np.linspace(0, 2 * np.pi, 18) + rng.random())
            axa.plot(xs, ys, color="0.35", lw=0.28, alpha=0.55)
        axa.text(0.095, y + 0.033, name, va="center", fontsize=5.7)
    axa.text(0.190, 0.80, "1-D elastic\ndraws", fontsize=5.9, fontweight="bold", linespacing=0.90)
    draw_profiles(axa, 0.175, 0.16, 0.250, 0.54)
    arrow(axa, 0.445, 0.43, 0.500, 0.43)
    axa.text(0.505, 0.80, "Surface-wave\nsimulator", fontsize=5.9, fontweight="bold", linespacing=0.90)
    axa.add_patch(plt.Rectangle((0.515, 0.27), 0.250, 0.310, facecolor="#dfd0bd", edgecolor="0.35", lw=0.55))
    axa.add_patch(plt.Rectangle((0.515, 0.56), 0.250, 0.035, facecolor="#b8d8d8", edgecolor="0.35", lw=0.4))
    for x in np.linspace(0.545, 0.735, 6):
        axa.plot(x, 0.595, marker="^", color="0.15", ms=3.0, mfc="white", mew=0.7)
    axa.plot(0.640, 0.635, marker="*", color=OKABE_ITO["orange"], ms=7.0, mec=OKABE_ITO["vermillion"])
    for width, color in zip(np.linspace(0.08, 0.25, 5), [OKABE_ITO["vermillion"], OKABE_ITO["blue"], OKABE_ITO["purple"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]):
        axa.add_patch(Arc((0.640, 0.58), width, 0.56 * width / 0.25, theta1=200, theta2=340, lw=0.75, color=color, alpha=0.78))
    axa.text(0.530, 0.315, "triangles: stations\nstar: virtual source", fontsize=5.1, color="0.25")
    arrow(axa, 0.775, 0.43, 0.810, 0.43)
    axa.text(0.805, 0.80, "Complete\ndispersion", fontsize=5.9, fontweight="bold", linespacing=0.90)
    for k, (yy, color, lab) in enumerate([(0.52, "red", "Rayleigh"), (0.22, "blue", "Love")]):
        ax = mini_axis(in_panel(a_rect, 0.815, yy, 0.155, 0.205))
        t = np.logspace(0, 2, 80)
        base = 4.8 - 0.75 * np.log10(t) + (0.15 if k == 0 else -0.55)
        ax.plot(t, base, color=color, lw=0.85)
        ax.plot(t, base + 0.25, color=color, lw=0.55, ls="--", alpha=0.8)
        ax.plot(t, base - 0.25, color=color, lw=0.55, ls="--", alpha=0.8)
        ax.set_xscale("log")
        ax.set_ylim(2.0, 5.8)
        ax.set_yticks([2, 4, 6])
        ax.set_xticks([1, 10, 100])
        if k == 1:
            ax.set_xlabel("Period (s)", fontsize=5.5)
        ax.set_ylabel("c", fontsize=5.5)
        ax.text(0.92, 0.86, lab, transform=ax.transAxes, ha="right", va="top", fontsize=5.3, color=color)

    axb = panel(b_rect, "B", "Observation process", "#fbfbff")
    axb.text(0.105, 0.80, "Cross-\ncorrelations", fontsize=5.8, fontweight="bold", linespacing=0.90)
    axcc = mini_axis(in_panel(b_rect, 0.095, 0.18, 0.245, 0.56))
    freq = np.linspace(0, 1, 100)
    for lag in np.linspace(-1, 1, 23):
        wiggle = 0.045 * np.sin(28 * freq + 8 * lag) * np.exp(-((freq - 0.47 - 0.20 * lag) ** 2) / 0.10)
        axcc.plot(freq + wiggle, np.full_like(freq, lag), color="0.15", lw=0.38, alpha=0.75)
    axcc.set_xticks([0.2, 0.8])
    axcc.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$"], fontsize=5.0)
    axcc.set_yticks([-1, 0, 1])
    axcc.set_yticklabels(["-100", "0", "100"], fontsize=5.0)
    axcc.set_xlabel("Frequency", fontsize=5.3)
    axcc.set_ylabel("Lag time", fontsize=5.3)
    arrow(axb, 0.385, 0.45, 0.455, 0.45)
    axb.text(0.475, 0.80, "Dispersion\nextraction", fontsize=5.8, fontweight="bold", linespacing=0.90)
    axex = mini_axis(in_panel(b_rect, 0.475, 0.18, 0.400, 0.56))
    periods = np.array([1.2, 1.6, 2.3, 3.4, 5.2, 8.0, 12.5, 19, 31, 50])
    ray = 5.2 - 0.52 * np.log10(periods) + 0.10 * rng.standard_normal(periods.size)
    lov = 4.6 - 0.62 * np.log10(periods) + 0.06 * rng.standard_normal(periods.size)
    obs = np.ones(periods.size, dtype=bool)
    obs[[3, 7]] = False
    axex.scatter(periods[obs], ray[obs], s=10, facecolors="white", edgecolors="red", linewidths=0.6, label="Rayleigh")
    axex.scatter(periods[obs], lov[obs], s=10, facecolors="white", edgecolors="blue", linewidths=0.6, label="Love")
    axex.scatter(periods[~obs], ray[~obs], marker="x", s=14, color="0.2", lw=0.7)
    axex.scatter(periods[~obs], lov[~obs], marker="x", s=14, color="0.2", lw=0.7)
    axex.set_xscale("log")
    axex.set_xlabel("Period (s)", fontsize=5.4)
    axex.set_ylabel("Phase velocity", fontsize=5.4)
    axex.set_ylim(2.2, 5.8)
    axex.legend(loc="lower left", fontsize=4.8, frameon=True, borderpad=0.25)

    axc = panel(c_rect, "C", "Noisy observations", "#fffdf8")
    axc.text(0.210, 0.80, "Add heteroscedastic\nnoise", fontsize=5.8, fontweight="bold", linespacing=0.90)
    axno = mini_axis(in_panel(c_rect, 0.145, 0.18, 0.770, 0.56))
    p = np.array([1.2, 1.6, 2.3, 3.2, 5.0, 7.6, 11, 17, 25, 38, 58])
    r = 5.05 - 0.55 * np.log10(p)
    l = 4.42 - 0.62 * np.log10(p)
    er = 0.12 + 0.04 * np.log10(p)
    el = 0.10 + 0.035 * np.log10(p)
    axno.errorbar(p, r + 0.08 * rng.standard_normal(p.size), yerr=er, color="red", marker="o", ms=2.4, lw=0.65, mfc="white", label="Rayleigh")
    axno.errorbar(p, l + 0.06 * rng.standard_normal(p.size), yerr=el, color="blue", marker="o", ms=2.4, lw=0.65, mfc="white", label="Love")
    axno.set_xscale("log")
    axno.set_xlabel("Period (s)", fontsize=5.4)
    axno.set_ylabel("Phase velocity", fontsize=5.4)
    axno.set_ylim(2.2, 5.8)
    axno.legend(loc="upper right", fontsize=4.9, frameon=True)
    axc.text(0.185, 0.085, "irregular periods + noise", fontsize=5.5)

    axd = panel(d_rect, "D", "Amortized posterior sampler: conditional rectified flow (training)", "#fbf8ff")
    axd.text(0.060, 0.79, "Training pairs", fontsize=6.0, fontweight="bold")
    draw_profiles(axd, 0.045, 0.25, 0.130, 0.40, alpha=0.78)
    axd.text(0.050, 0.68, r"${\bf m}^{(i)}$", fontsize=5.8)
    axd.plot([0.205, 0.235, 0.265], [0.48, 0.48, 0.48], "k.", ms=2.4)
    axd.add_patch(plt.Rectangle((0.285, 0.26), 0.075, 0.38, facecolor="white", edgecolor="0.6", lw=0.5))
    for color, offset in [("red", 0.18), ("blue", 0.00)]:
        x = np.linspace(0.295, 0.350, 8)
        y = 0.58 - offset - 0.10 * np.linspace(0, 1, 8)
        axd.errorbar(x, y, yerr=0.010, color=color, marker="o", ms=1.8, lw=0.55, mfc="white")
    axd.text(0.287, 0.68, r"${\bf d}^{(i)}$", fontsize=5.8)
    axd.plot([0.385, 0.415, 0.445], [0.48, 0.48, 0.48], "k.", ms=2.4)
    axd.text(0.505, 0.78, r"Conditional path   $p_\theta({\bf x}_t|{\bf d})$", fontsize=6.0, fontweight="bold", ha="center")
    axd.text(0.500, 0.665, r"$d{\bf x}_t/dt=v_\theta({\bf x}_t,t,{\bf d}),\quad t\in[0,1]$", fontsize=6.3, ha="center")
    axd.text(0.405, 0.43, r"${\bf x}_0\sim N(0,I)$", fontsize=5.9)
    cloud_x = np.linspace(0.470, 0.790, 5)
    cloud_colors = ["#6f3fb3", OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["orange"], "red"]
    for i, (cx, color) in enumerate(zip(cloud_x, cloud_colors)):
        pts = rng.normal(size=(52, 2))
        pts /= np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 0.25)
        axd.scatter(cx + 0.020 * pts[:, 0], 0.45 + 0.070 * pts[:, 1], s=2.2, color=color, alpha=0.65, lw=0)
        if i < len(cloud_x) - 1:
            arrow(axd, cx + 0.035, 0.45, cloud_x[i + 1] - 0.035, 0.45, lw=0.65)
    axd.plot([0.450, 0.812], [0.27, 0.27], color="0.2", lw=0.55)
    axd.text(0.450, 0.235, r"$t=0$", fontsize=5.8, ha="center")
    axd.text(0.812, 0.235, r"$t=1$", fontsize=5.8, ha="center")
    arrow(axd, 0.820, 0.45, 0.865, 0.45)
    draw_profiles(axd, 0.875, 0.25, 0.095, 0.40, alpha=0.75)
    axd.text(0.868, 0.68, "posterior\nsamples", fontsize=5.6, linespacing=0.90)
    axd.add_patch(
        FancyBboxPatch(
            (0.130, 0.055),
            0.760,
            0.070,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="#fff5df",
            edgecolor="#c9a65a",
            linewidth=0.55,
        )
    )
    axd.text(0.510, 0.090, "Network is trained by a rectified-flow objective with physics regularization.", ha="center", va="center", fontsize=5.9)

    axe = panel(e_rect, "E", "Inference for one location", "#f8fcf6")
    for i, y in enumerate([0.70, 0.46, 0.23], 1):
        axe.plot(0.075, y, marker="o", ms=9, color="#2c6b63")
        axe.text(0.075, y, str(i), color="white", ha="center", va="center", fontsize=7.0, fontweight="bold")
    axin = mini_axis(in_panel(e_rect, 0.265, 0.595, 0.360, 0.210))
    axin.errorbar(p[::2], r[::2], yerr=er[::2], color="red", marker="o", ms=2, lw=0.5, mfc="white")
    axin.errorbar(p[::2], l[::2], yerr=el[::2], color="blue", marker="o", ms=2, lw=0.5, mfc="white")
    axin.set_xscale("log")
    axin.set_xticks([])
    axin.set_yticks([])
    axe.text(0.335, 0.835, r"noisy input ${\bf d}^*$", fontsize=5.7)
    axe.add_patch(
        FancyBboxPatch(
            (0.245, 0.410),
            0.390,
            0.125,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor="0.88",
            edgecolor="0.55",
            linewidth=0.55,
        )
    )
    axe.text(0.440, 0.472, r"trained CRF" + "\n" + r"$v_\theta({\bf x}_t,t,{\bf d}^*)$", fontsize=5.8, ha="center", va="center")
    arrow(axe, 0.440, 0.590, 0.440, 0.535, lw=0.65)
    arrow(axe, 0.440, 0.410, 0.440, 0.350, lw=0.65)
    draw_profiles(axe, 0.270, 0.135, 0.315, 0.185, alpha=0.80)
    axe.text(0.635, 0.285, "median\n16-84% interval\ndensities\npredictive checks", fontsize=5.8, va="top")

    axf = panel(f_rect, "F", "Regional posterior products (concept)", "#fffdf8")
    axf.text(0.045, 0.80, "Candidate\nregion", fontsize=5.9, fontweight="bold", linespacing=0.90)
    axf.add_patch(plt.Rectangle((0.035, 0.22), 0.155, 0.50, facecolor="#e1d6b9", edgecolor="0.45", lw=0.5))
    for _ in range(18):
        xs = np.linspace(0.040, 0.185, 80)
        ys = 0.24 + 0.46 * rng.random() + 0.035 * np.sin(np.linspace(0, 2 * np.pi, 80) + rng.random() * 6)
        axf.plot(xs, ys, color="#7aa889", lw=0.35, alpha=0.50)
    locs = rng.uniform([0.055, 0.30], [0.170, 0.67], size=(25, 2))
    axf.scatter(locs[:, 0], locs[:, 1], s=5, color=OKABE_ITO["vermillion"], edgecolor="white", lw=0.3)
    axf.scatter(rng.uniform(0.045, 0.180, 16), rng.uniform(0.25, 0.70, 16), marker="^", s=10, color="white", edgecolor="0.25", lw=0.45)
    axf.text(0.235, 0.80, "Example\nposterior profiles", fontsize=5.9, fontweight="bold", linespacing=0.90)
    for row, color in enumerate([OKABE_ITO["vermillion"], OKABE_ITO["orange"], OKABE_ITO["blue"]]):
        ymid = 0.62 - row * 0.17
        axf.plot(0.215, ymid, marker="o", ms=6.8, color=color)
        axf.text(0.215, ymid, str(row + 1), ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
        for col in range(3):
            x0 = 0.245 + col * 0.052
            y = np.linspace(ymid - 0.060, ymid + 0.060, 36)
            x = x0 + 0.010 * np.tanh(np.linspace(-2, 2, 36)) + 0.002 * rng.standard_normal(36)
            axf.fill_betweenx(y, x - 0.009, x + 0.009, color=color, alpha=0.18, lw=0)
            axf.plot(x, y, color="0.15", lw=0.55)
    axf.text(0.432, 0.78, r"Depth slices of posterior median $V_S$", fontsize=6.1, fontweight="bold")
    cmap = plt.get_cmap("viridis")
    for k, depth_lab in enumerate(["10", "50", "100", "150"]):
        x0 = 0.430 + k * 0.075
        y0 = 0.32
        grid = rng.normal(size=(6, 6)).cumsum(axis=0).cumsum(axis=1)
        grid = (grid - grid.min()) / (grid.max() - grid.min())
        for iy in range(6):
            for ix in range(6):
                axf.add_patch(plt.Rectangle((x0 + ix * 0.010, y0 + iy * 0.032), 0.010, 0.032, facecolor=cmap(0.15 + 0.70 * grid[iy, ix]), edgecolor="none"))
        axf.add_patch(plt.Rectangle((x0, y0), 0.060, 0.192, facecolor="none", edgecolor="0.45", lw=0.35))
        axf.text(x0 + 0.030, y0 + 0.215, f"{depth_lab} km", fontsize=5.2, ha="center")
        axf.scatter(x0 + rng.uniform(0.006, 0.054, 7), y0 + rng.uniform(0.010, 0.180, 7), s=3.0, color="white", edgecolor="0.25", lw=0.2)
    axf.text(0.725, 0.78, r"Posterior std of $V_S$", fontsize=6.1, fontweight="bold")
    cmap2 = plt.get_cmap("magma")
    for k, depth_lab in enumerate(["10", "50", "100"]):
        x0 = 0.725 + k * 0.063
        y0 = 0.32
        grid = rng.random((6, 6))
        for iy in range(6):
            for ix in range(6):
                axf.add_patch(plt.Rectangle((x0 + ix * 0.0085, y0 + iy * 0.032), 0.0085, 0.032, facecolor=cmap2(0.10 + 0.65 * grid[iy, ix]), edgecolor="none"))
        axf.add_patch(plt.Rectangle((x0, y0), 0.051, 0.192, facecolor="none", edgecolor="0.45", lw=0.35))
        axf.text(x0 + 0.026, y0 + 0.215, f"{depth_lab} km", fontsize=5.2, ha="center")
    axf.text(0.915, 0.78, "P(LVZ)", fontsize=6.1, fontweight="bold")
    x0, y0 = 0.910, 0.32
    grid = np.linspace(0, 1, 36).reshape(6, 6)
    for iy in range(6):
        for ix in range(6):
            axf.add_patch(plt.Rectangle((x0 + ix * 0.010, y0 + iy * 0.032), 0.010, 0.032, facecolor=plt.get_cmap("turbo")(grid[iy, ix]), edgecolor="none"))
    axf.add_patch(plt.Rectangle((x0, y0), 0.060, 0.192, facecolor="none", edgecolor="0.45", lw=0.35))
    axf.text(0.500, 0.075, "Panel F illustrates possible products after validated regional deployment; it is not field-data validation in this manuscript.", ha="center", fontsize=5.7, color="0.25")

    legend = fig.add_axes([0.015, 0.010, 0.970, 0.045])
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    legend.axis("off")
    entries = [
        ("#f6fbff", "data generation"),
        ("#fbf8ff", "learning"),
        ("#f8fcf6", "inference"),
        ("#fffdf8", "deployment concept"),
    ]
    x = 0.015
    for color, text in entries:
        legend.add_patch(FancyBboxPatch((x, 0.30), 0.030, 0.40, boxstyle="round,pad=0.01", facecolor=color, edgecolor="0.70", linewidth=0.45))
        legend.text(x + 0.040, 0.50, text, va="center", fontsize=5.9)
        x += 0.170
    legend.plot(0.700, 0.50, marker="o", mfc="white", mec="red", ms=4, lw=0)
    legend.text(0.715, 0.50, "Rayleigh", va="center", fontsize=5.9, color="red")
    legend.plot(0.790, 0.50, marker="o", mfc="white", mec="blue", ms=4, lw=0)
    legend.text(0.805, 0.50, "Love", va="center", fontsize=5.9, color="blue")
    legend.plot(0.885, 0.50, marker="x", color="0.15", ms=4, lw=0)
    legend.text(0.900, 0.50, "missing", va="center", fontsize=5.9)
    legend.errorbar([0.955], [0.50], yerr=[0.18], color="0.2", lw=0.7)
    legend.text(0.963, 0.50, "noise", va="center", fontsize=5.9)
    savefig("fig01_workflow.pdf")


def fig_control_points(model, profiles, highlight_index=0):
    depth = model.depth_grid.cpu().numpy()
    cp = model.control_depth_grid.cpu().numpy()
    profile_np = profiles.detach().cpu().numpy() if torch.is_tensor(profiles) else np.asarray(profiles)
    if profile_np.ndim == 2:
        profile_np = profile_np[None, ...]
    highlight_index = min(max(int(highlight_index), 0), profile_np.shape[0] - 1)
    vs_profiles = profile_np[:, 1, :]
    highlight_vs = vs_profiles[highlight_index]
    plt.figure(figsize=(FULL_WIDTH_IN, 3.55))
    ax1 = plt.subplot(1, 2, 1)
    for vs in vs_profiles[:100]:
        ax1.plot(vs, depth, color=OKABE_ITO["blue"], alpha=0.115, lw=0.65)
    ax1.plot(highlight_vs, depth, color=OKABE_ITO["black"], lw=1.35, label="Highlighted profile")
    ax1.invert_yaxis()
    ax1.set_ylim(depth.max(), depth.min())
    ax1.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax1.set_ylabel("Depth (km)")
    panel_label(ax1, "a", f"{min(100, len(vs_profiles))} synthetic profiles")
    ax1.legend(frameon=False, loc="lower right")
    style_axis(ax1)
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    ax2.plot(highlight_vs, depth, color=OKABE_ITO["black"], lw=1.45, label="Full profile")
    ax2.plot(
        highlight_vs[model.control_indices.cpu().numpy()],
        cp,
        "o-",
        color=OKABE_ITO["vermillion"],
        ms=3.1,
        lw=1.05,
        label="Depth-control values",
    )
    ax2.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    panel_label(ax2, "b", f"One profile; {len(cp)} controls")
    ax2.legend(frameon=False, loc="lower right")
    style_axis(ax2)
    plt.tight_layout()
    savefig("fig02_control_points.pdf")


def fig_training_history(rows):
    e = np.array([r["epoch"] for r in rows])
    plt.figure(figsize=(FULL_WIDTH_IN, 4.65))
    ax = plt.subplot(2, 1, 1)
    ax.plot(e, [r["train_loss"] for r in rows], label="Training", color=OKABE_ITO["blue"], lw=1.4)
    ax.plot(e, [r["val_loss"] for r in rows], label="Validation", color=OKABE_ITO["orange"], lw=1.4, ls="--")
    ax.set_ylabel("Loss")
    panel_label(ax, "a", "Rectified-flow objective")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax)
    ax = plt.subplot(2, 1, 2)
    ax.plot(e, [r["sample_vp"] for r in rows], label="$V_P$", color=OKABE_ITO["blue"], lw=1.35)
    ax.plot(e, [r["sample_vs"] for r in rows], label="$V_S$", color=OKABE_ITO["green"], lw=1.35, ls="--")
    ax.plot(e, [r["sample_rho"] for r in rows], label="$\\rho$", color=OKABE_ITO["purple"], lw=1.35, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation sample MAE")
    panel_label(ax, "b", "Sample error")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_axis(ax)
    plt.tight_layout()
    savefig("fig03_training_history.pdf")


def load_ak135_profile(depth_grid):
    rows = []
    path = files("obspy.taup.data").joinpath("ak135.tvel")
    for line in path.read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    z_nodes, vp_nodes, vs_nodes, rho_nodes = arr.T

    depth = np.asarray(depth_grid, dtype=np.float64)
    vp = np.empty_like(depth)
    vs = np.empty_like(depth)
    rho = np.empty_like(depth)
    for i, z in enumerate(depth):
        idx = np.searchsorted(z_nodes, z, side="right") - 1
        idx = int(np.clip(idx, 0, len(z_nodes) - 2))
        while idx + 1 < len(z_nodes) - 1 and z_nodes[idx + 1] == z_nodes[idx]:
            idx += 1
        z0, z1 = z_nodes[idx], z_nodes[idx + 1]
        if z1 <= z0:
            frac = 0.0
        else:
            frac = (z - z0) / (z1 - z0)
        vp[i] = vp_nodes[idx] + frac * (vp_nodes[idx + 1] - vp_nodes[idx])
        vs[i] = vs_nodes[idx] + frac * (vs_nodes[idx + 1] - vs_nodes[idx])
        rho[i] = rho_nodes[idx] + frac * (rho_nodes[idx + 1] - rho_nodes[idx])

    return torch.from_numpy(np.stack([vp, vs, rho]).astype(np.float32))


def dispersion_from_profile(depth, profile, periods):
    depth_np = np.asarray(depth, dtype=np.float64)
    profile_np = profile.detach().cpu().numpy() if torch.is_tensor(profile) else np.asarray(profile)
    rayleigh = compute_phase_dispersion(
        depth_np,
        profile_np[0],
        profile_np[1],
        profile_np[2],
        periods=periods,
        modes=(0,),
        wave="rayleigh",
    )[0]
    love = compute_phase_dispersion(
        depth_np,
        profile_np[0],
        profile_np[1],
        profile_np[2],
        periods=periods,
        modes=(0,),
        wave="love",
    )[0]
    disp = np.stack([periods, rayleigh.velocity, love.velocity], axis=0).astype(np.float32)
    mask = np.ones_like(disp, dtype=np.float32)
    return torch.from_numpy(disp), torch.from_numpy(mask)


def save_posterior_sample_archive(
    cp_model,
    cp_ckpt,
    args,
    target,
    disp_batch,
    mask_batch,
    samples,
    ak135_profile,
    ak135_disp,
    ak135_mask,
    ak135_samples,
):
    """Save the posterior samples used directly in Figures 4--6."""
    np.savez_compressed(
        OUT / "posterior_figure_samples.npz",
        depth_km=cp_model.depth_grid.detach().cpu().numpy(),
        channel_names=np.asarray(["Vp", "Vs", "rho"]),
        channel_units=np.asarray(["km s^-1", "km s^-1", "g/cm^3"]),
        synthetic_example_index=np.asarray(0, dtype=np.int64),
        synthetic_target=target[0].detach().cpu().numpy(),
        synthetic_dispersion=disp_batch[0].detach().cpu().numpy(),
        synthetic_mask=mask_batch[0].detach().cpu().numpy(),
        synthetic_posterior_samples=samples[0].detach().cpu().numpy(),
        ak135_target=ak135_profile.detach().cpu().numpy(),
        ak135_dispersion=ak135_disp.detach().cpu().numpy(),
        ak135_mask=ak135_mask.detach().cpu().numpy(),
        ak135_posterior_samples=ak135_samples.detach().cpu().numpy(),
        posterior_samples=np.asarray(args.posterior_samples, dtype=np.int64),
        sampling_steps=np.asarray(args.sampling_steps, dtype=np.int64),
        evaluation_seed=np.asarray(2026, dtype=np.int64),
        checkpoint_epoch=np.asarray(int(cp_ckpt["epoch"]), dtype=np.int64),
        checkpoint_global_step=np.asarray(int(cp_ckpt["global_step"]), dtype=np.int64),
    )


def fig_posterior_profiles(model, true_profile, samples, ak135_profile=None, ak135_samples=None):
    depth = model.depth_grid.cpu().numpy()
    labels = ["$V_P$", "$V_S$", "$\\rho$"]
    units = [r"km s$^{-1}$", r"km s$^{-1}$", "g cm$^{-3}$"]
    cases = [("Held-out synthetic", true_profile, samples)]
    if ak135_profile is not None and ak135_samples is not None:
        cases.append(("ak135 standard model", ak135_profile, ak135_samples))
    plt.figure(figsize=(FULL_WIDTH_IN, 3.35 * len(cases)))
    letters = iter("abcdef")
    for row, (case_name, case_true, case_samples) in enumerate(cases):
        true_np = case_true.cpu().numpy()
        s = case_samples.cpu().numpy()
        med = np.median(s, axis=0)
        q16 = np.percentile(s, 16, axis=0)
        q84 = np.percentile(s, 84, axis=0)
        for i in range(3):
            ax = plt.subplot(len(cases), 3, row * 3 + i + 1)
            for k in range(min(24, s.shape[0])):
                ax.plot(s[k, i], depth, color=OKABE_ITO["vermillion"], alpha=0.16, lw=0.75)
            ax.fill_betweenx(depth, q16[i], q84[i], color=OKABE_ITO["vermillion"], alpha=0.22, label="16-84%")
            ax.plot(true_np[i], depth, color=OKABE_ITO["black"], lw=1.55, label="Target")
            ax.plot(med[i], depth, "--", color=OKABE_ITO["vermillion"], lw=1.55, label="Median")
            ax.invert_yaxis()
            ax.set_xlabel(f"{labels[i]} ({units[i]})")
            panel_text = f"{case_name}; {labels[i]}"
            panel_label(ax, next(letters), panel_text)
            if i == 0:
                ax.set_ylabel("Depth (km)")
                if row == 0:
                    ax.legend(frameon=False, loc="lower left")
            style_axis(ax)
    plt.tight_layout()
    savefig("fig04_posterior_profiles.pdf")


def fig_posterior_density_vs(model, true_profile, samples):
    depth = model.depth_grid.cpu().numpy()
    s = samples[:, 1].cpu().numpy()
    true_vs = true_profile[1].cpu().numpy()
    med = np.median(s, axis=0)
    vmin = min(s.min(), true_vs.min()) - 0.05
    vmax = max(s.max(), true_vs.max()) + 0.05
    bins = np.linspace(vmin, vmax, 90)
    hist = []
    centers = 0.5 * (bins[:-1] + bins[1:])
    for iz in range(s.shape[1]):
        h, _ = np.histogram(s[:, iz], bins=bins, density=True)
        hist.append(h)
    hist = np.asarray(hist).T
    plt.figure(figsize=(HALF_WIDTH_IN, 5.35))
    ax = plt.gca()
    mesh = ax.pcolormesh(centers, depth, hist.T, shading="auto", cmap="viridis")
    ax.plot(true_vs, depth, color="white", lw=2.25, label="Target")
    ax.plot(true_vs, depth, color=OKABE_ITO["black"], lw=1.15)
    ax.plot(med, depth, color=OKABE_ITO["orange"], ls="--", lw=1.75, label="Median")
    ax.invert_yaxis()
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025)
    cbar.set_label("Posterior density")
    cbar.ax.tick_params(labelsize=7.4, width=0.6, length=2.5)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid=False)
    plt.tight_layout()
    savefig("fig05_posterior_density_vs.pdf")


def fig_uncertainty(model, samples):
    depth = model.depth_grid.cpu().numpy()
    std = samples.std(dim=0, unbiased=False).cpu().numpy()
    labels = ["$V_P$", "$V_S$", "$\\rho$"]
    units = [r"km s$^{-1}$", r"km s$^{-1}$", "g cm$^{-3}$"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    plt.figure(figsize=(FULL_WIDTH_IN, 3.9))
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        ax.plot(std[i], depth, color=colors[i], lw=1.55)
        ax.fill_betweenx(depth, 0, std[i], color=colors[i], alpha=0.18)
        ax.invert_yaxis()
        ax.set_xlabel(f"Posterior std. ({units[i]})")
        panel_label(ax, "abc"[i], labels[i])
        if i == 0:
            ax.set_ylabel("Depth (km)")
        style_axis(ax)
    plt.tight_layout()
    savefig("fig06_posterior_uncertainty.pdf")


def fig_coverage(qs, coverage, calibrated_coverage=None, temperature_scale=None):
    labels = ["$V_P$", "$V_S$", "$\\rho$"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    styles = ["-", "--", ":"]
    markers = ["o", "s", "^"]
    if calibrated_coverage is None:
        plt.figure(figsize=(HALF_WIDTH_IN, 3.65))
        axes = [plt.gca()]
        panels = [(axes[0], coverage, "Raw posterior")]
    else:
        plt.figure(figsize=(FULL_WIDTH_IN, 3.55))
        axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
        panels = [
            (axes[0], coverage, "Raw posterior"),
            (axes[1], calibrated_coverage, f"Scaled posterior (s={temperature_scale:.2f})"),
        ]
    for label, (ax, panel_coverage, title) in zip("ab", panels):
        ax.plot(qs, qs / 100.0, color=OKABE_ITO["black"], ls="--", lw=1.1, label="Ideal")
        for i in range(3):
            ax.plot(
                qs,
                panel_coverage[:, i],
                marker=markers[i],
                ms=3.5,
                lw=1.25,
                ls=styles[i],
                color=colors[i],
                label=labels[i],
            )
        ax.set_xlabel("Nominal credible interval (%)")
        panel_label(ax, label, title)
        ax.set_ylim(0, 1.02)
        style_axis(ax)
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(frameon=False, loc="lower right")
    plt.tight_layout()
    savefig("fig08_coverage.pdf")


def sample_median_profiles(model, disp_batch, mask_batch, num_samples, num_steps, batch_size):
    preds = []
    with torch.no_grad():
        for i in range(0, disp_batch.size(0), batch_size):
            out = model.predict(
                disp_batch[i : i + batch_size],
                mask_batch[i : i + batch_size],
                num_samples=num_samples,
                num_steps=num_steps,
                reduce="median",
            )
            preds.append(out["profile_mu"])
    return torch.cat(preds, dim=0)


def summarize_point_estimate(pred, target):
    err = pred - target
    mae_by_example = err.abs().mean(dim=2)
    rmse_by_example = torch.sqrt(err.pow(2).mean(dim=2))
    rough_by_example = roughness(pred)
    return {
        "mae": mae_by_example.mean(dim=0),
        "rmse": rmse_by_example.mean(dim=0),
        "roughness": rough_by_example.mean(dim=0),
        "mean_mae": err.abs().mean(),
        "mean_rmse": torch.sqrt(err.pow(2).mean(dim=(1, 2))).mean(),
        "mean_roughness": rough_by_example.mean(),
        "per_example_mae": mae_by_example,
        "per_example_rmse": rmse_by_example,
        "per_example_roughness": rough_by_example,
    }


def fig_dense_vs_control(cp_model, dense, cp, true_profile):
    depth = cp_model.depth_grid.cpu().numpy()
    true_np = true_profile.cpu().numpy()
    dense_np = dense.cpu().numpy()
    cp_np = cp.cpu().numpy()
    labels = ["$V_P$", "$V_S$", "$\\rho$"]
    units = [r"km s$^{-1}$", r"km s$^{-1}$", "g cm$^{-3}$"]
    plt.figure(figsize=(FULL_WIDTH_IN, 3.9))
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        ax.plot(true_np[i], depth, color=OKABE_ITO["black"], lw=1.45, label="Target")
        ax.plot(dense_np[i], depth, color=OKABE_ITO["purple"], lw=1.2, ls=":", label="Dense RF")
        ax.plot(cp_np[i], depth, color=OKABE_ITO["vermillion"], lw=1.45, ls="--", label="Depth-control RF")
        ax.invert_yaxis()
        ax.set_xlabel(f"{labels[i]} ({units[i]})")
        panel_label(ax, "abc"[i], labels[i])
        if i == 0:
            ax.set_ylabel("Depth (km)")
            ax.legend(frameon=False, loc="lower left")
        style_axis(ax)
    plt.tight_layout()
    savefig("fig09_dense_vs_control.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate manuscript figures and validation metrics.")
    parser.add_argument("--n-eval", type=int, default=256, help="Number of held-out synthetic examples.")
    parser.add_argument("--posterior-samples", type=int, default=16, help="Posterior samples per example.")
    parser.add_argument("--sampling-steps", type=int, default=24, help="Euler steps for rectified-flow sampling.")
    parser.add_argument("--batch-size", type=int, default=8, help="Sampling batch size.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates for 95 percent intervals.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)

    mod12 = load_module("disp_inv_train_v12", ROOT / "disp_inv_train.v1.2.py")
    mod11 = load_module("disp_inv_train_v11", ROOT / "disp_inv_train.v1.1.py")

    cp_model, cp_ckpt = restore_model(mod12, ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    dense_model, _ = restore_model(mod11, ROOT / "ckpt/disp2struct_crf.v1.1/best.pt")

    model_batch, disp_batch, mask_batch = collect_dataset(mod12, n=args.n_eval)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    with torch.no_grad():
        samples = []
        bs = args.batch_size
        for i in range(0, target.size(0), bs):
            out = cp_model.sample(
                disp_batch[i : i + bs],
                mask_batch[i : i + bs],
                num_samples=args.posterior_samples,
                num_steps=args.sampling_steps,
            )
            samples.append(out["profile_samples"])
        samples = torch.cat(samples, dim=0)

    median = samples.median(dim=1).values
    err = (median - target).abs()
    per_example_mae = err.mean(dim=2)
    per_example_rmse = torch.sqrt((median - target).pow(2).mean(dim=2))
    per_example_std = samples.std(dim=1, unbiased=False).mean(dim=2)
    rmse = per_example_rmse.mean(dim=0)
    mae = per_example_mae.mean(dim=0)
    std_mean = per_example_std.mean(dim=0)
    q16 = torch.quantile(samples, 0.16, dim=1)
    q84 = torch.quantile(samples, 0.84, dim=1)
    per_example_coverage68 = ((target >= q16) & (target <= q84)).float().mean(dim=2)
    coverage68 = per_example_coverage68.mean(dim=0)
    qs = np.array([20, 40, 60, 68, 80, 90])
    cov = nominal_coverage(samples, target, qs)
    mae_lo, mae_hi = bootstrap_ci(per_example_mae.numpy(), args.bootstrap, seed=2026)
    rmse_lo, rmse_hi = bootstrap_ci(per_example_rmse.numpy(), args.bootstrap, seed=2027)
    std_lo, std_hi = bootstrap_ci(per_example_std.numpy(), args.bootstrap, seed=2028)
    cov_lo, cov_hi = bootstrap_ci(per_example_coverage68.numpy(), args.bootstrap, seed=2029)
    mean_err_by_example = err.mean(dim=(1, 2)).numpy()
    mean_rmse_by_example = torch.sqrt((median - target).pow(2).mean(dim=(1, 2))).numpy()
    mean_cov_by_example = per_example_coverage68.mean(dim=1).numpy()
    mean_mae_ci = bootstrap_ci(mean_err_by_example, args.bootstrap, seed=2030)
    mean_rmse_ci = bootstrap_ci(mean_rmse_by_example, args.bootstrap, seed=2031)
    mean_cov_ci = bootstrap_ci(mean_cov_by_example, args.bootstrap, seed=2032)

    split_index = target.size(0) // 2
    calibration_samples = samples[:split_index]
    calibration_target = target[:split_index]
    test_samples = samples[split_index:]
    test_target = target[split_index:]
    temperature_scale = fit_temperature_scale(calibration_samples, calibration_target, nominal_percent=68.0)
    raw_test_cov = nominal_coverage(test_samples, test_target, qs)
    scaled_test_samples = scaled_samples_about_median(test_samples, temperature_scale)
    scaled_test_cov = nominal_coverage(scaled_test_samples, test_target, qs)
    raw_test_coverage68_by_example = interval_coverage_by_example(test_samples, test_target, 68.0, scale=1.0)
    scaled_test_coverage68_by_example = interval_coverage_by_example(test_samples, test_target, 68.0, scale=temperature_scale)
    raw_test_coverage68 = raw_test_coverage68_by_example.mean(dim=0)
    scaled_test_coverage68 = scaled_test_coverage68_by_example.mean(dim=0)
    raw_test_cov_lo, raw_test_cov_hi = bootstrap_ci(raw_test_coverage68_by_example.numpy(), args.bootstrap, seed=2033)
    scaled_test_cov_lo, scaled_test_cov_hi = bootstrap_ci(scaled_test_coverage68_by_example.numpy(), args.bootstrap, seed=2034)
    raw_test_mean_cov_ci = bootstrap_ci(raw_test_coverage68_by_example.mean(dim=1).numpy(), args.bootstrap, seed=2035)
    scaled_test_mean_cov_ci = bootstrap_ci(scaled_test_coverage68_by_example.mean(dim=1).numpy(), args.bootstrap, seed=2036)

    dense_median = sample_median_profiles(
        dense_model,
        disp_batch,
        mask_batch,
        num_samples=args.posterior_samples,
        num_steps=args.sampling_steps,
        batch_size=args.batch_size,
    )
    cp_summary = summarize_point_estimate(median, target)
    dense_summary = summarize_point_estimate(dense_median, target)
    true_roughness = roughness(target).mean(dim=0)
    roughness_reduction = dense_summary["roughness"] / cp_summary["roughness"].clamp_min(1.0e-8)
    mean_roughness_reduction = dense_summary["mean_roughness"] / cp_summary["mean_roughness"].clamp_min(1.0e-8)

    periods = disp_batch[0, 0].cpu().numpy().astype(np.float64)
    ak135_profile = load_ak135_profile(cp_model.depth_grid.cpu().numpy())
    ak135_disp, ak135_mask = dispersion_from_profile(cp_model.depth_grid.cpu().numpy(), ak135_profile, periods)
    with torch.no_grad():
        ak135_out = cp_model.sample(
            ak135_disp.unsqueeze(0).float(),
            ak135_mask.unsqueeze(0).float(),
            num_samples=args.posterior_samples,
            num_steps=args.sampling_steps,
        )
        ak135_samples = ak135_out["profile_samples"][0]
    ak135_median = ak135_samples.median(dim=0).values
    ak135_err = (ak135_median - ak135_profile).abs()
    ak135_mae = ak135_err.mean(dim=1)
    ak135_rmse = torch.sqrt((ak135_median - ak135_profile).pow(2).mean(dim=1))
    ak135_q16 = torch.quantile(ak135_samples, 0.16, dim=0)
    ak135_q84 = torch.quantile(ak135_samples, 0.84, dim=0)
    ak135_coverage68 = ((ak135_profile >= ak135_q16) & (ak135_profile <= ak135_q84)).float().mean(dim=1)
    ak135_std = ak135_samples.std(dim=0, unbiased=False).mean(dim=1)

    rows = parse_epoch_log(ROOT / "ckpt/disp2struct_crf.v1.2_cp/train.log")
    save_posterior_sample_archive(
        cp_model,
        cp_ckpt,
        args,
        target,
        disp_batch,
        mask_batch,
        samples,
        ak135_profile,
        ak135_disp,
        ak135_mask,
        ak135_samples,
    )
    fig_dense_vs_control(cp_model, dense_median[0], median[0], target[0])
    fig_workflow()
    fig_control_points(cp_model, target[:100])
    fig_training_history(rows)
    fig_posterior_profiles(cp_model, target[0], samples[0], ak135_profile, ak135_samples)
    fig_posterior_density_vs(cp_model, target[0], samples[0])
    fig_uncertainty(cp_model, samples[0])
    fig_coverage(qs, raw_test_cov, scaled_test_cov, temperature_scale)

    metrics = {
        "checkpoint_epoch": int(cp_ckpt["epoch"]),
        "checkpoint_global_step": int(cp_ckpt["global_step"]),
        "best_val_loss": float(cp_ckpt["best_val_loss"]),
        "n_eval": int(target.size(0)),
        "num_posterior_samples": int(samples.size(1)),
        "num_sampling_steps": int(args.sampling_steps),
        "evaluation_seed": 2026,
        "mae": {"Vp": float(mae[0]), "Vs": float(mae[1]), "rho": float(mae[2]), "mean": float(err.mean())},
        "rmse": {"Vp": float(rmse[0]), "Vs": float(rmse[1]), "rho": float(rmse[2]), "mean": float(mean_rmse_by_example.mean())},
        "mean_posterior_std": {"Vp": float(std_mean[0]), "Vs": float(std_mean[1]), "rho": float(std_mean[2])},
        "coverage_16_84": {"Vp": float(coverage68[0]), "Vs": float(coverage68[1]), "rho": float(coverage68[2]), "mean": float(coverage68.mean())},
        "ci95": {
            "mae": {**channel_ci_dict(mae_lo, mae_hi), "mean": [float(mean_mae_ci[0][0]), float(mean_mae_ci[1][0])]},
            "rmse": {**channel_ci_dict(rmse_lo, rmse_hi), "mean": [float(mean_rmse_ci[0][0]), float(mean_rmse_ci[1][0])]},
            "mean_posterior_std": channel_ci_dict(std_lo, std_hi),
            "coverage_16_84": {**channel_ci_dict(cov_lo, cov_hi), "mean": [float(mean_cov_ci[0][0]), float(mean_cov_ci[1][0])]},
        },
        "split_temperature_calibration": {
            "method": "Single scalar scale fitted on the first half of evaluation examples to target mean pointwise 68 percent coverage and evaluated on the second half.",
            "calibration_examples": int(calibration_target.size(0)),
            "test_examples": int(test_target.size(0)),
            "nominal_percent": 68.0,
            "temperature_scale": float(temperature_scale),
            "raw_test_coverage_16_84": {
                "Vp": float(raw_test_coverage68[0]),
                "Vs": float(raw_test_coverage68[1]),
                "rho": float(raw_test_coverage68[2]),
                "mean": float(raw_test_coverage68.mean()),
            },
            "scaled_test_coverage_16_84": {
                "Vp": float(scaled_test_coverage68[0]),
                "Vs": float(scaled_test_coverage68[1]),
                "rho": float(scaled_test_coverage68[2]),
                "mean": float(scaled_test_coverage68.mean()),
            },
            "ci95": {
                "raw_test_coverage_16_84": {
                    **channel_ci_dict(raw_test_cov_lo, raw_test_cov_hi),
                    "mean": [float(raw_test_mean_cov_ci[0][0]), float(raw_test_mean_cov_ci[1][0])],
                },
                "scaled_test_coverage_16_84": {
                    **channel_ci_dict(scaled_test_cov_lo, scaled_test_cov_hi),
                    "mean": [float(scaled_test_mean_cov_ci[0][0]), float(scaled_test_mean_cov_ci[1][0])],
                },
            },
        },
        "ak135_standard_model": {
            "source": "ak135.tvel distributed with ObsPy, based on Kennett et al. (1995).",
            "mask": "Full Rayleigh and Love period band, 2-60 s.",
            "mae": {
                "Vp": float(ak135_mae[0]),
                "Vs": float(ak135_mae[1]),
                "rho": float(ak135_mae[2]),
                "mean": float(ak135_err.mean()),
            },
            "rmse": {
                "Vp": float(ak135_rmse[0]),
                "Vs": float(ak135_rmse[1]),
                "rho": float(ak135_rmse[2]),
                "mean": float(torch.sqrt((ak135_median - ak135_profile).pow(2).mean())),
            },
            "coverage_16_84": {
                "Vp": float(ak135_coverage68[0]),
                "Vs": float(ak135_coverage68[1]),
                "rho": float(ak135_coverage68[2]),
                "mean": float(ak135_coverage68.mean()),
            },
            "mean_posterior_std": {
                "Vp": float(ak135_std[0]),
                "Vs": float(ak135_std[1]),
                "rho": float(ak135_std[2]),
            },
        },
        "control_points": int(cp_model.Nc),
        "depth_nodes": int(cp_model.H),
        "periods": int(cp_model.T),
        "parameters": int(sum(p.numel() for p in cp_model.parameters())),
        "dense_control_ablation": {
            "dense_mae": {
                "Vp": float(dense_summary["mae"][0]),
                "Vs": float(dense_summary["mae"][1]),
                "rho": float(dense_summary["mae"][2]),
                "mean": float(dense_summary["mean_mae"]),
            },
            "control_mae": {
                "Vp": float(cp_summary["mae"][0]),
                "Vs": float(cp_summary["mae"][1]),
                "rho": float(cp_summary["mae"][2]),
                "mean": float(cp_summary["mean_mae"]),
            },
            "dense_rmse": {
                "Vp": float(dense_summary["rmse"][0]),
                "Vs": float(dense_summary["rmse"][1]),
                "rho": float(dense_summary["rmse"][2]),
                "mean": float(dense_summary["mean_rmse"]),
            },
            "control_rmse": {
                "Vp": float(cp_summary["rmse"][0]),
                "Vs": float(cp_summary["rmse"][1]),
                "rho": float(cp_summary["rmse"][2]),
                "mean": float(cp_summary["mean_rmse"]),
            },
            "true_roughness": {"Vp": float(true_roughness[0]), "Vs": float(true_roughness[1]), "rho": float(true_roughness[2])},
            "dense_roughness": {
                "Vp": float(dense_summary["roughness"][0]),
                "Vs": float(dense_summary["roughness"][1]),
                "rho": float(dense_summary["roughness"][2]),
                "mean": float(dense_summary["mean_roughness"]),
            },
            "control_roughness": {
                "Vp": float(cp_summary["roughness"][0]),
                "Vs": float(cp_summary["roughness"][1]),
                "rho": float(cp_summary["roughness"][2]),
                "mean": float(cp_summary["mean_roughness"]),
            },
            "roughness_reduction_factor": {
                "Vp": float(roughness_reduction[0]),
                "Vs": float(roughness_reduction[1]),
                "rho": float(roughness_reduction[2]),
                "mean": float(mean_roughness_reduction),
            },
        },
        "latest_logged_epoch": rows[-1] if rows else None,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
