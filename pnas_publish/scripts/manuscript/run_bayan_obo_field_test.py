#!/usr/bin/env python3
"""Run a Bayan Obo field-data stress test for the trained inversion sampler."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
DATA = ROOT / "Bayan_Obo_Dataset"
OUT = OVERLEAF / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PDF_METADATA = {
    "Creator": "run_bayan_obo_field_test.py",
    "CreationDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
}

FULL_WIDTH_IN = 7.1
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
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.72,
        "lines.linewidth": 1.15,
        "xtick.major.width": 0.62,
        "ytick.major.width": 0.62,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax, grid=True):
    ax.tick_params(direction="out", length=2.8, width=0.62, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.72)
        spine.set_color("0.25")
    if grid:
        ax.grid(color="0.88", lw=0.5, alpha=1.0)


def panel_label(ax, label, text=None, x=0.025, y=0.97):
    body = f"({label})" if text is None else f"({label}) {text}"
    ax.text(
        x,
        y,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
        zorder=20,
    )


def load_training_module(path: Path):
    # The field-test script only needs the model class. Injecting a minimal
    # SurfaceWaveDataset stub avoids requiring disba in environments used only
    # for posterior sampling from the checkpoint.
    fake = types.ModuleType("utils.generate_data")

    class SurfaceWaveDataset:  # pragma: no cover - import shim only
        pass

    fake.SurfaceWaveDataset = SurfaceWaveDataset
    old_module = sys.modules.get("utils.generate_data")
    sys.modules["utils.generate_data"] = fake
    try:
        spec = importlib.util.spec_from_file_location("disp_inv_train_v12_field", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if old_module is None:
            sys.modules.pop("utils.generate_data", None)
        else:
            sys.modules["utils.generate_data"] = old_module
    return module


def restore_model(train_module, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = train_module.Disp2StructCRF(
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
        control_indices=ckpt["control_indices"],
        period_minmax=tuple(float(x) for x in ckpt["period_minmax"].tolist()),
        disp_mean=ckpt["disp_mean"],
        disp_scale=ckpt["disp_scale"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def subarray_centers():
    lonmin, lonmax = 109.0, 111.5
    latmin, latmax = 41.15, 42.43
    dlon, dlat = 1.0 / 6.0, 1.0 / 8.0
    lon = np.arange(lonmin + dlon / 2.0, lonmax - dlon / 2.0 + 1e-9, dlon)
    lat = np.arange(latmin + dlat / 2.0, latmax - dlat / 2.0 + 1e-9, dlat)
    x, y = np.meshgrid(lon, lat)
    return x.ravel(order="F"), y.ravel(order="F")


def parse_surf96_dispersion(path: Path, period_min=4.0, period_max=26.0):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SURF96":
            continue
        try:
            period = float(parts[5])
            velocity = float(parts[6])
        except ValueError:
            continue
        if not (math.isfinite(period) and math.isfinite(velocity)):
            continue
        if period_min <= period <= period_max and 1.5 <= velocity <= 6.0:
            rows.append((period, velocity))
    if not rows:
        return np.array([]), np.array([])
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    periods = []
    velocities = []
    for period in np.unique(arr[:, 0]):
        periods.append(period)
        velocities.append(float(np.median(arr[arr[:, 0] == period, 1])))
    return np.asarray(periods), np.asarray(velocities)


def load_masw_inputs(data_dir: Path):
    disp_dir = data_dir / "Subarray-Based MASW" / "disp_curves_subarray_cut"
    files = sorted(
        disp_dir.glob("disp_subarray*.in"),
        key=lambda path: int(re.search(r"(\d+)", path.name).group(1)),
    )
    centers_lon, centers_lat = subarray_centers()
    if len(files) != len(centers_lon):
        raise RuntimeError(f"Expected {len(centers_lon)} subarray files, found {len(files)}.")

    model_periods = np.arange(2.0, 61.0, 1.0, dtype=np.float32)
    observed_periods = np.arange(4.0, 27.0, 1.0, dtype=np.float32)
    disp = np.zeros((len(files), 3, model_periods.size), dtype=np.float32)
    mask = np.zeros_like(disp)
    disp[:, 0, :] = model_periods[None, :]
    mask[:, 0, :] = 1.0

    curves = np.full((len(files), observed_periods.size), np.nan, dtype=np.float32)
    file_indices = []
    for row, path in enumerate(files):
        periods, velocities = parse_surf96_dispersion(path)
        if periods.size < 5:
            continue
        values = np.interp(observed_periods, periods, velocities, left=np.nan, right=np.nan)
        valid = np.isfinite(values)
        if valid.sum() < 5:
            continue
        grid_indices = (observed_periods[valid] - 2.0).astype(int)
        disp[row, 1, grid_indices] = values[valid]
        mask[row, 1, grid_indices] = 1.0
        curves[row, valid] = values[valid]
        file_indices.append(row + 1)

    keep = mask[:, 1, :].sum(axis=1) >= 5
    return {
        "period_grid": model_periods,
        "observed_periods": observed_periods,
        "dispersion": disp[keep],
        "mask": mask[keep],
        "curves": curves[keep],
        "lon": centers_lon[keep],
        "lat": centers_lat[keep],
        "subarray_index": np.asarray(file_indices, dtype=int),
    }


def load_stations(data_dir: Path):
    path = data_dir / "Subarray-Based MASW" / "station_location.txt"
    raw = np.loadtxt(path, dtype=str)
    return raw[:, 0].astype(float), raw[:, 1].astype(float), raw[:, 2]


def load_surf96_vs(data_dir: Path):
    path = data_dir / "S-Wave_Velocity_Result" / "image.3D_BY_SV"
    arr = np.loadtxt(path)
    lon_values = np.unique(arr[:, 0])
    lat_values = np.unique(arr[:, 1])
    depth_values = np.unique(arr[:, 2])
    grid_by_depth = arr[:, 3].reshape(depth_values.size, lat_values.size, lon_values.size)
    grid = np.moveaxis(grid_by_depth, 0, -1)
    return lon_values, lat_values, depth_values, grid


def nearest_surf96_profiles(lon, lat, data_dir: Path):
    lon_grid, lat_grid, depth, vs_grid = load_surf96_vs(data_dir)
    profiles = np.zeros((len(lon), depth.size), dtype=float)
    for idx, (x, y) in enumerate(zip(lon, lat)):
        ix = int(np.argmin(np.abs(lon_grid - x)))
        iy = int(np.argmin(np.abs(lat_grid - y)))
        profiles[idx] = vs_grid[iy, ix, :]
    return depth, profiles


def run_sampler(model, disp, mask, samples, steps, batch_size, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sample_blocks = []
    median_blocks = []
    std_blocks = []
    for start in range(0, disp.shape[0], batch_size):
        stop = min(start + batch_size, disp.shape[0])
        d = torch.from_numpy(disp[start:stop]).float()
        q = torch.from_numpy(mask[start:stop]).float()
        with torch.no_grad():
            out = model.sample(
                d,
                q,
                num_samples=samples,
                num_steps=steps,
                temperature=1.0,
                generator=generator,
            )
        sample_blocks.append(out["profile_samples"].detach().cpu().numpy())
        median_blocks.append(out["profile_median"].detach().cpu().numpy())
        std_blocks.append(out["profile_std"].detach().cpu().numpy())
    return {
        "samples": np.concatenate(sample_blocks, axis=0),
        "median": np.concatenate(median_blocks, axis=0),
        "std": np.concatenate(std_blocks, axis=0),
    }


def interp_profiles_to_depth(profile, model_depth, target_depth):
    out = np.zeros((profile.shape[0], target_depth.size), dtype=float)
    for idx in range(profile.shape[0]):
        out[idx] = np.interp(target_depth, model_depth, profile[idx])
    return out


def finite_corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def save_outputs(field, depth, posterior, surf_depth, surf_vs, args):
    samples = posterior["samples"]
    median = posterior["median"]
    std = posterior["std"]
    vs_median = median[:, 1, :]
    vs_std = std[:, 1, :]
    vs_surf_interp = interp_profiles_to_depth(vs_median, depth, surf_depth)
    residual = vs_surf_interp - surf_vs
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Bayan Obo field-data stress test for the amortized posterior sampler.",
        "data_root": str(DATA),
        "subarrays_used": int(field["dispersion"].shape[0]),
        "rayleigh_period_min_s": float(field["observed_periods"][0]),
        "rayleigh_period_max_s": float(field["observed_periods"][-1]),
        "rayleigh_period_count": int(field["observed_periods"].size),
        "love_observations": 0,
        "posterior_samples": int(args.posterior_samples),
        "euler_steps": int(args.euler_steps),
        "comparison_reference": "Nearest-neighbor conventional SURF96 Vs model from Bayan_Obo_Dataset/S-Wave_Velocity_Result; not treated as ground truth.",
        "surf96_depth_min_km": float(surf_depth.min()),
        "surf96_depth_max_km": float(surf_depth.max()),
        "surf96_vs_mae_km_s": float(np.nanmean(np.abs(residual))),
        "surf96_vs_bias_km_s": float(np.nanmean(residual)),
        "surf96_vs_rmse_km_s": float(np.sqrt(np.nanmean(residual**2))),
        "surf96_vs_corr": finite_corr(vs_surf_interp.ravel(), surf_vs.ravel()),
        "posterior_vs_std_mean_0p5_9km": float(np.nanmean(interp_profiles_to_depth(vs_std, depth, surf_depth))),
        "notes": [
            "Only Rayleigh MASW subarray dispersion curves are used; Love observations are masked.",
            "The trained sampler is not recalibrated for Bayan Obo observational errors.",
            "The field result is a transfer/stress test of the synthetic-prior posterior surrogate.",
        ],
    }
    np.savez(
        OUT / "bayan_obo_field_results.npz",
        lon=field["lon"],
        lat=field["lat"],
        subarray_index=field["subarray_index"],
        period_grid=field["period_grid"],
        observed_periods=field["observed_periods"],
        dispersion=field["dispersion"],
        mask=field["mask"],
        curves=field["curves"],
        depth=depth,
        posterior_samples=samples,
        posterior_median=median,
        posterior_std=std,
        surf96_depth=surf_depth,
        surf96_vs=surf_vs,
        posterior_vs_on_surf96_depth=vs_surf_interp,
    )
    (OUT / "bayan_obo_field_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def plot_field_figure(field, depth, posterior, surf_depth, surf_vs, summary):
    samples = posterior["samples"]
    median = posterior["median"]
    std = posterior["std"]
    lon = field["lon"]
    lat = field["lat"]
    periods = field["observed_periods"]
    curves = field["curves"]
    vs_median = median[:, 1, :]
    vs_std = std[:, 1, :]
    vs_surf_interp = interp_profiles_to_depth(vs_median, depth, surf_depth)

    target_lon, target_lat = np.median(lon), np.median(lat)
    example = int(np.argmin((lon - target_lon) ** 2 + (lat - target_lat) ** 2))
    depth_limit = depth <= 40.0
    surf_limit = surf_depth <= 9.0
    d5_index = int(np.argmin(np.abs(depth - 5.0)))

    station_lon, station_lat, _ = load_stations(DATA)

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 6.15))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.965, wspace=0.62, hspace=0.48)
    ax = axes[0, 0]
    ax.scatter(station_lon, station_lat, marker="^", s=10, facecolor="white", edgecolor="0.35", lw=0.45, label="Stations")
    ax.scatter(lon, lat, s=12, color=OKABE_ITO["vermillion"], edgecolor="white", lw=0.25, label="Subarrays")
    ax.scatter(lon[example], lat[example], s=34, color=OKABE_ITO["blue"], edgecolor="white", lw=0.5, zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.25)
    panel_label(ax, "a", "Bayan Obo array")
    style_axis(ax)

    ax = axes[0, 1]
    for row in range(curves.shape[0]):
        ax.plot(periods, curves[row], color="0.68", lw=0.45, alpha=0.22)
    ax.plot(periods, curves[example], color=OKABE_ITO["black"], lw=1.25, label=f"Subarray {field['subarray_index'][example]}")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Rayleigh phase velocity (km/s)")
    ax.set_xlim(3.5, 26.5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    panel_label(ax, "b", "field dispersion")
    style_axis(ax)

    ax = axes[0, 2]
    for sample in samples[example, : min(samples.shape[1], 32), 1, :]:
        ax.plot(sample[depth_limit], depth[depth_limit], color=OKABE_ITO["blue"], alpha=0.11, lw=0.55)
    q16 = np.quantile(samples[example, :, 1, :], 0.16, axis=0)
    q84 = np.quantile(samples[example, :, 1, :], 0.84, axis=0)
    ax.fill_betweenx(depth[depth_limit], q16[depth_limit], q84[depth_limit], color=OKABE_ITO["sky"], alpha=0.24, lw=0)
    ax.plot(vs_median[example, depth_limit], depth[depth_limit], color=OKABE_ITO["vermillion"], lw=1.45, label="Posterior median")
    ax.plot(surf_vs[example, surf_limit], surf_depth[surf_limit], color=OKABE_ITO["black"], lw=1.25, ls="--", label="SURF96 reference")
    ax.invert_yaxis()
    ax.set_ylim(40, 0)
    ax.set_xlabel("$V_S$ (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    panel_label(ax, "c", "example posterior")
    style_axis(ax)

    ax = axes[1, 0]
    sc = ax.scatter(lon, lat, c=vs_median[:, d5_index], s=25, cmap="viridis", edgecolor="0.15", lw=0.18)
    ax.scatter(station_lon, station_lat, marker="^", s=7, facecolor="none", edgecolor="0.35", lw=0.35)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cax = inset_axes(ax, width="4.8%", height="58%", loc="lower right", borderpad=0.85)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("$V_S$ (km/s)")
    panel_label(ax, "d", "median $V_S$ at 5 km")
    style_axis(ax)

    ax = axes[1, 1]
    sc = ax.scatter(lon, lat, c=vs_std[:, d5_index], s=25, cmap="magma_r", edgecolor="0.15", lw=0.18)
    ax.scatter(station_lon, station_lat, marker="^", s=7, facecolor="none", edgecolor="0.35", lw=0.35)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cax = inset_axes(ax, width="4.8%", height="58%", loc="lower right", borderpad=0.85)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Std($V_S$) (km/s)")
    panel_label(ax, "e", "posterior spread at 5 km")
    style_axis(ax)

    ax = axes[1, 2]
    x = surf_vs.ravel()
    y = vs_surf_interp.ravel()
    ax.scatter(x, y, s=8, color=OKABE_ITO["blue"], alpha=0.36, edgecolor="none")
    lo = min(np.nanmin(x), np.nanmin(y)) - 0.05
    hi = max(np.nanmax(x), np.nanmax(y)) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.9, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SURF96 $V_S$, 0.5--9 km (km/s)")
    ax.set_ylabel("Posterior $V_S$, 0.5--9 km (km/s)")
    text = f"MAE 0.5-9 km = {summary['surf96_vs_mae_km_s']:.2f} km/s\nr = {summary['surf96_vs_corr']:.2f}"
    ax.text(0.05, 0.08, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.0)
    panel_label(ax, "f", "SURF96 comparison")
    style_axis(ax)

    fig.savefig(OUT / "fig07_bayan_obo_field_test.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    parser.add_argument("--train-script", type=Path, default=ROOT / "disp_inv_train.v1.2.py")
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--euler-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Missing Bayan Obo data directory: {args.data_dir}")

    train_module = load_training_module(args.train_script)
    model, _ = restore_model(train_module, args.checkpoint)
    field = load_masw_inputs(args.data_dir)
    posterior = run_sampler(
        model,
        field["dispersion"],
        field["mask"],
        samples=args.posterior_samples,
        steps=args.euler_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    depth = model.depth_grid.detach().cpu().numpy()
    surf_depth, surf_vs = nearest_surf96_profiles(field["lon"], field["lat"], args.data_dir)
    summary = save_outputs(field, depth, posterior, surf_depth, surf_vs, args)
    plot_field_figure(field, depth, posterior, surf_depth, surf_vs, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
