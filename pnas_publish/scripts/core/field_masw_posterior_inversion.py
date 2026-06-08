#!/usr/bin/env python3
"""Run DNN posterior inversion on Bayan Obo subarray MASW dispersion curves.

The MASW directory contains Rayleigh-wave phase-velocity picks for 150
subarrays. This script converts those picks to the five-channel DNN condition
used by ``disp_inv_train.v1.2.py``:

    [period, Rayleigh velocity, Love velocity] with Rayleigh/Love masks.

Only Rayleigh data are available here, so the Love channel is masked out. The
result is a field demonstration volume, not a calibrated field-data posterior:
the current network was trained on a mask-only synthetic benchmark without an
explicit observational-error model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.tri import Triangulation


ROOT = Path(__file__).resolve().parents[1]
MASW_DIR = ROOT / "Bayan_Obo_Dataset" / "Subarray-Based MASW"
DISP_DIR = MASW_DIR / "disp_curves_subarray_cut"
DEFAULT_CKPT = ROOT / "ckpt" / "disp2struct_crf.v1.2_cp_weak" / "best.pt"


def load_training_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    path = ROOT / "disp_inv_train.v1.3.py"
    if not path.exists():
        path = ROOT / "disp_inv_train.v1.2.py"
    spec = importlib.util.spec_from_file_location("disp_inv_train_v12", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device):
    module = load_training_module()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = module.Disp2StructCRF(
        H=int(cfg.get("z_max_num", ckpt["reference_profile"].shape[-1])),
        T=59,
        profile_channels=3,
        cond_base_channels=int(cfg.get("cond_base_channels", 64)),
        cond_dim=int(cfg.get("cond_dim", 256)),
        flow_hidden=int(cfg.get("flow_hidden", 1024)),
        time_dim=int(cfg.get("time_dim", 64)),
        dropout=float(cfg.get("dropout", 0.1)),
        reference_profile=ckpt["reference_profile"],
        profile_scale=ckpt["profile_scale"],
        depth_grid=ckpt["depth_grid"],
        control_indices=ckpt["control_indices"],
        period_minmax=tuple(float(x) for x in ckpt["period_minmax"].tolist()),
        disp_mean=ckpt["disp_mean"],
        disp_scale=ckpt["disp_scale"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def subarray_centres() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the MATLAB subarray indexing used by build_grids.m."""
    lonmin, lonmax, dlon = 109.0, 111.5, 1.0 / 6.0
    latmin, latmax, dlat = 41.15, 42.43, 1.0 / 8.0
    lon = np.arange(lonmin + dlon / 2.0, lonmax - dlon / 2.0 + 1e-9, dlon)
    lat = np.arange(latmin + dlat / 2.0, latmax - dlat / 2.0 + 1e-9, dlat)
    x, y = np.meshgrid(lon, lat)
    return x.ravel(order="F"), y.ravel(order="F")


def parse_subarray_index(path: Path) -> int:
    m = re.search(r"disp_curve_subarray(\d+)\.txt$", path.name)
    if not m:
        raise ValueError(f"Cannot parse subarray index from {path}")
    return int(m.group(1))


def read_dispersion_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected dispersion format in {path}")
    freq_hz = arr[:, 0]
    vel_km_s = arr[:, 1] / 1000.0
    ok = np.isfinite(freq_hz) & np.isfinite(vel_km_s) & (freq_hz > 0.0) & (vel_km_s > 0.0)
    period_s = 1.0 / freq_hz[ok]
    vel_km_s = vel_km_s[ok]
    order = np.argsort(period_s)
    return period_s[order], vel_km_s[order]


def build_field_conditions(
    period_grid: np.ndarray,
    min_period: float,
    max_period: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    centre_lon, centre_lat = subarray_centres()
    files = sorted(DISP_DIR.glob("disp_curve_subarray*.txt"), key=parse_subarray_index)
    disp_rows = []
    mask_rows = []
    meta = []
    for path in files:
        idx = parse_subarray_index(path)
        period_obs, ray_obs = read_dispersion_curve(path)
        in_range = (
            (period_grid >= period_obs.min())
            & (period_grid <= period_obs.max())
            & (period_grid >= min_period)
            & (period_grid <= max_period)
        )
        ray_interp = np.zeros_like(period_grid, dtype=np.float32)
        ray_interp[in_range] = np.interp(period_grid[in_range], period_obs, ray_obs).astype(np.float32)

        disp = np.zeros((3, len(period_grid)), dtype=np.float32)
        mask = np.zeros((3, len(period_grid)), dtype=np.float32)
        disp[0] = period_grid
        disp[1] = ray_interp
        mask[0] = 1.0
        mask[1] = in_range.astype(np.float32)
        # Love channel remains zero and masked out.
        disp_rows.append(disp)
        mask_rows.append(mask)
        meta.append(
            {
                "subarray": idx,
                "lon": float(centre_lon[idx - 1]),
                "lat": float(centre_lat[idx - 1]),
                "period_min": float(period_obs.min()),
                "period_max": float(period_obs.max()),
                "n_periods_used": int(in_range.sum()),
                "rayleigh_mean_km_s": float(ray_interp[in_range].mean()) if in_range.any() else float("nan"),
            }
        )
    return np.stack(disp_rows), np.stack(mask_rows), meta


def batched(iterable: Iterable[int], n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def run_inference(
    model,
    disp: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    batch_size: int,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    all_samples = []
    with torch.no_grad():
        for idx in batched(range(len(disp)), batch_size):
            d = torch.from_numpy(disp[idx]).to(device)
            m = torch.from_numpy(mask[idx]).to(device)
            out = model.sample(
                d,
                m,
                num_samples=num_samples,
                num_steps=num_steps,
                temperature=1.0,
                generator=generator,
            )
            all_samples.append(out["profile_samples"].cpu())
    samples = torch.cat(all_samples, dim=0).numpy()
    return {
        "samples": samples,
        "median": np.median(samples, axis=1),
        "q16": np.quantile(samples, 0.16, axis=1),
        "q84": np.quantile(samples, 0.84, axis=1),
        "std": np.std(samples, axis=1),
    }


def write_point_summary(path: Path, meta: list[dict], depth: np.ndarray, stats: dict) -> None:
    depth_targets = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    depth_indices = [int(np.argmin(np.abs(depth - z))) for z in depth_targets]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subarray",
                "lon",
                "lat",
                "depth_km",
                "vs_median_km_s",
                "vs_q16_km_s",
                "vs_q84_km_s",
                "vs_std_km_s",
                "n_periods_used",
            ]
        )
        for i, row in enumerate(meta):
            for z, iz in zip(depth_targets, depth_indices):
                writer.writerow(
                    [
                        row["subarray"],
                        f"{row['lon']:.6f}",
                        f"{row['lat']:.6f}",
                        f"{z:.1f}",
                        f"{stats['median'][i, 1, iz]:.5f}",
                        f"{stats['q16'][i, 1, iz]:.5f}",
                        f"{stats['q84'][i, 1, iz]:.5f}",
                        f"{stats['std'][i, 1, iz]:.5f}",
                        row["n_periods_used"],
                    ]
                )


def plot_dispersion_qc(path: Path, period_grid: np.ndarray, disp: np.ndarray, mask: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i in range(len(disp)):
        ok = mask[i, 1].astype(bool)
        ax.plot(period_grid[ok], disp[i, 1, ok], color="0.25", lw=0.45, alpha=0.25)
    ray = np.where(mask[:, 1, :].astype(bool), disp[:, 1, :], np.nan)
    valid = np.isfinite(ray).any(axis=0)
    ax.plot(period_grid[valid], np.nanmedian(ray[:, valid], axis=0), color="#d62728", lw=2.0, label="median")
    ax.set_xlim(float(period_grid[valid].min()), float(period_grid[valid].max()))
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Rayleigh phase velocity (km s$^{-1}$)")
    ax.set_title("Bayan Obo subarray MASW dispersion picks", fontsize=12)
    ax.grid(True, color="0.90", lw=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_slice_grid(path: Path, meta: list[dict], depth: np.ndarray, field: np.ndarray, label: str, depths: list[float]) -> None:
    lon = np.array([m["lon"] for m in meta])
    lat = np.array([m["lat"] for m in meta])
    tri = Triangulation(lon, lat)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8), sharex=True, sharey=True)
    vals_for_scale = []
    depth_indices = []
    for z in depths:
        iz = int(np.argmin(np.abs(depth - z)))
        depth_indices.append(iz)
        vals_for_scale.append(field[:, iz])
    vals_for_scale = np.concatenate(vals_for_scale)
    vmin = float(np.nanpercentile(vals_for_scale, 2))
    vmax = float(np.nanpercentile(vals_for_scale, 98))
    last = None
    for ax, z, iz in zip(axes.ravel(), depths, depth_indices):
        last = ax.tricontourf(tri, field[:, iz], levels=18, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.tricontour(tri, field[:, iz], colors="k", linewidths=0.25, alpha=0.35)
        ax.scatter(lon, lat, s=5, c="k", alpha=0.35)
        ax.set_title(f"{z:g} km", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.92", lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Longitude")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.90, pad=0.02)
    cbar.set_label(label)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "field_masw_results")
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--num-samples", "--posterior-samples", dest="num_samples", type=int, default=32)
    parser.add_argument("--num-steps", "--sampling-steps", dest="num_steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-period", "--period-min", dest="min_period", type=float, default=2.0)
    parser.add_argument("--max-period", "--period-max", dest="max_period", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    period_grid = np.linspace(2.0, 60.0, 59).astype(np.float32)
    disp, mask, meta = build_field_conditions(period_grid, args.min_period, args.max_period)
    if not meta:
        raise RuntimeError(f"No dispersion files found in {DISP_DIR}")

    model, ckpt = load_model(args.checkpoint, device)
    stats = run_inference(
        model=model,
        disp=disp,
        mask=mask,
        device=device,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    depth = model.depth_grid.detach().cpu().numpy()

    meta_dtype = [
        ("subarray", "i4"),
        ("lon", "f8"),
        ("lat", "f8"),
        ("period_min", "f4"),
        ("period_max", "f4"),
        ("n_periods_used", "i4"),
        ("rayleigh_mean_km_s", "f4"),
    ]
    meta_arr = np.array([tuple(row[k] for k, _ in meta_dtype) for row in meta], dtype=meta_dtype)
    np.savez_compressed(
        args.out_dir / "bayan_obo_masw_dnn_posterior_volume.npz",
        depth_km=depth,
        period_s=period_grid,
        disp=disp,
        mask=mask,
        meta=meta_arr,
        posterior_samples=stats["samples"],
        median=stats["median"],
        q16=stats["q16"],
        q84=stats["q84"],
        std=stats["std"],
        checkpoint=str(args.checkpoint),
        checkpoint_epoch=np.asarray(int(ckpt.get("epoch", -1))),
        checkpoint_global_step=np.asarray(int(ckpt.get("global_step", -1))),
        note="Field demonstration: Rayleigh-only MASW, Love masked out, no explicit observational-error likelihood.",
    )
    write_point_summary(args.out_dir / "bayan_obo_masw_vs_depth_summary.csv", meta, depth, stats)
    plot_dispersion_qc(args.out_dir / "field_dispersion_qc.png", period_grid, disp, mask)
    plot_slice_grid(
        args.out_dir / "field_vs_median_slices.png",
        meta,
        depth,
        stats["median"][:, 1, :],
        r"Posterior median $V_S$ (km s$^{-1}$)",
        [5.0, 10.0, 20.0, 40.0],
    )
    plot_slice_grid(
        args.out_dir / "field_vs_std_slices.png",
        meta,
        depth,
        stats["std"][:, 1, :],
        r"Posterior std $V_S$ (km s$^{-1}$)",
        [5.0, 10.0, 20.0, 40.0],
    )
    if args.fig_dir is not None:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "field_dispersion_qc",
            "field_vs_median_slices",
            "field_vs_std_slices",
        ):
            for suffix in (".png", ".pdf"):
                src = args.out_dir / f"{name}{suffix}"
                if src.exists():
                    shutil.copyfile(src, args.fig_dir / f"fair_{name}{suffix}")

    n_used = np.array([m["n_periods_used"] for m in meta])
    print(f"[done] subarrays={len(meta)} device={device} samples={args.num_samples} steps={args.num_steps}")
    print(f"[done] periods used per subarray: min={n_used.min()} median={np.median(n_used):.0f} max={n_used.max()}")
    print(f"[done] outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
