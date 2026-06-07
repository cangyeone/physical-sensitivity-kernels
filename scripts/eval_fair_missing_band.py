#!/usr/bin/env python3
"""Missing-period-band uncertainty diagnostic for a fair DI checkpoint."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_period_window(disp: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = mask.copy()
    keep = (disp[:, 0, :] >= lo) & (disp[:, 0, :] <= hi)
    out[:, 1:3, :] *= keep[:, None, :].astype(out.dtype)
    out[:, 0, :] = (out[:, 1:3, :].sum(axis=1) > 0).astype(out.dtype)
    return out


def apply_random_windows(disp: np.ndarray, mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = mask.copy()
    widths = np.zeros(len(mask), dtype=np.float32)
    for i in range(len(mask)):
        periods = disp[i, 0]
        width = rng.uniform(15.0, 45.0)
        start = rng.uniform(float(periods.min()), float(periods.max() - width))
        end = start + width
        widths[i] = width
        keep = (periods >= start) & (periods <= end)
        valid = np.where(keep)[0]
        if len(valid):
            holes = rng.integers(0, 7)
            if holes:
                keep[rng.choice(valid, size=min(holes, len(valid)), replace=False)] = False
        keep &= rng.random(len(periods)) >= 0.04
        if not keep.any():
            keep[np.argmin(np.abs(periods - 0.5 * (start + end)))] = True
        out[i, 1:3, :] *= keep[None, :].astype(out.dtype)
        out[i, 0, :] = (out[i, 1:3, :].sum(axis=0) > 0).astype(out.dtype)
    return out, widths


def summarize_window(name: str, samples: np.ndarray, target: np.ndarray, mask: np.ndarray, base_mask: np.ndarray, random_width=None):
    pred = np.median(samples, axis=1)
    q16 = np.quantile(samples, 0.16, axis=1)
    q84 = np.quantile(samples, 0.84, axis=1)
    inside = (target >= q16) & (target <= q84)
    spread = samples.std(axis=1)
    valid_fraction = mask[:, 1:3, :].sum(axis=(1, 2)) / max(float(base_mask[:, 1:3, :].shape[1] * base_mask.shape[2]), 1.0)
    vs_mae = np.abs(pred[:, 1, :] - target[:, 1, :]).mean(axis=1)
    vs_spread = spread[:, 1, :].mean(axis=1)
    row = {
        "window": name,
        "n": int(len(target)),
        "valid_fraction_mean": float(valid_fraction.mean()),
        "valid_fraction_std": float(valid_fraction.std()),
        "vs_mae": float(vs_mae.mean()),
        "vs_spread_mean": float(vs_spread.mean()),
        "coverage_vs": float(inside[:, 1, :].mean()),
        "coverage_mean": float(inside.mean()),
    }
    if random_width is not None:
        row["random_width_mean_s"] = float(np.mean(random_width))
        row["corr_valid_fraction_vs_spread"] = float(np.corrcoef(valid_fraction, vs_spread)[0, 1])
        row["corr_valid_fraction_vs_mae"] = float(np.corrcoef(valid_fraction, vs_mae)[0, 1])
    return row, {"valid_fraction": valid_fraction, "vs_mae": vs_mae, "vs_spread": vs_spread}


def plot(rows: list[dict], per_example: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["window"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, metric, ylabel, color in zip(
        axes,
        ["vs_mae", "vs_spread_mean", "coverage_vs"],
        ["$V_S$ MAE (km s$^{-1}$)", "$V_S$ posterior std", "$V_S$ coverage"],
        ["#3b82c4", "#e69f00", "#6aa84f"],
    ):
        ax.bar(x, [r[metric] for r in rows], color=color, alpha=0.82)
        if metric == "coverage_vs":
            ax.axhline(0.68, color="#222222", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "missing_band_uncertainty.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "missing_band_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if "random-window" in per_example:
        r = per_example["random-window"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        axes[0].scatter(r["valid_fraction"], r["vs_spread"], s=12, alpha=0.7)
        axes[0].set_ylabel("$V_S$ posterior std")
        axes[1].scatter(r["valid_fraction"], r["vs_mae"], s=12, alpha=0.7)
        axes[1].set_ylabel("$V_S$ MAE")
        for ax in axes:
            ax.set_xlabel("Valid period fraction")
            ax.grid(color="#e5e5e5", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "missing_band_random_scatter.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "missing_band_random_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/fair_di_weak_full_seed642026/best.pt")
    p.add_argument("--prior", choices=("strong", "weak"), default="weak")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/missing_band")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production/missing_band")
    p.add_argument("--n-eval", type=int, default=1024)
    p.add_argument("--posterior-samples", type=int, default=64)
    p.add_argument("--euler-steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=642026)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    fair = import_from_path("fair_eval_for_missing_band", ROOT / "scripts/eval_fair_di_comparison.py")
    boundary = import_from_path("prior_boundary_for_missing_band", ROOT / "scripts/eval_prior_boundary_effect.py")
    dataset_mod = import_from_path(
        f"{args.prior}_generator_for_missing_band",
        ROOT / "utils" / ("generate_data.py" if args.prior == "strong" else "generate_data_weak_prior.py"),
    )
    model, _ = boundary.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.ckpt, device)
    models_full, disp, base_mask = boundary.dataset_to_arrays(
        dataset_mod.SurfaceWaveDataset(args.n_eval, z_max_km=150.0, z_max_num=256, dz_km=0.5, seed=args.seed + 70_000)
    )
    target = models_full[:, 1:4, :].astype(np.float32)
    windows = {
        "full-2-60": base_mask.astype(np.float32),
        "p2-40": apply_period_window(disp, base_mask, 2.0, 40.0),
        "p10-30": apply_period_window(disp, base_mask, 10.0, 30.0),
        "p15-45": apply_period_window(disp, base_mask, 15.0, 45.0),
    }
    random_mask, random_width = apply_random_windows(disp, base_mask, args.seed + 71_000)
    windows["random-window"] = random_mask
    rows = []
    per_example = {}
    for name, mask in windows.items():
        samples = boundary.direct_samples(model, disp, mask, device, args.posterior_samples, args.euler_steps, args.batch_size)
        row, diag = summarize_window(name, samples, target, mask, base_mask, random_width if name == "random-window" else None)
        rows.append(row)
        per_example[name] = diag
    write_csv(args.out_dir / "missing_band_uncertainty.csv", rows)
    write_json(args.out_dir / "missing_band_uncertainty.json", {"protocol": vars(args), "rows": rows})
    np.savez_compressed(args.out_dir / "missing_band_per_example.npz", **{f"{k}_{kk}": vv for k, d in per_example.items() for kk, vv in d.items()})
    plot(rows, per_example, args.fig_dir)
    print(f"Wrote missing-band diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
