#!/usr/bin/env python3
"""Evaluation-time phase-velocity noise sensitivity for fair DI checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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
CHANNELS = ("vp", "vs", "rho")


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


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
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_noise(disp: np.ndarray, mask: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    noisy = disp.copy()
    if sigma == 0.0:
        return noisy
    rng = np.random.default_rng(seed)
    keep = mask[:, 1:3, :] > 0.5
    perturb = rng.normal(0.0, sigma, size=noisy[:, 1:3, :].shape).astype(np.float32)
    noisy[:, 1:3, :] = np.where(keep, noisy[:, 1:3, :] + perturb, noisy[:, 1:3, :])
    noisy[:, 1:3, :] = np.clip(noisy[:, 1:3, :], 0.1, None)
    return noisy.astype(np.float32)


def fit_temperature(samples: np.ndarray, target: np.ndarray) -> float:
    def cov(scale: float) -> float:
        median = np.median(samples, axis=1, keepdims=True)
        scaled = median + scale * (samples - median)
        q16 = np.quantile(scaled, 0.16, axis=1)
        q84 = np.quantile(scaled, 0.84, axis=1)
        return float(((target >= q16) & (target <= q84)).mean())

    if cov(1.0) >= 0.68:
        return 1.0
    lo, hi = 1.0, 1.5
    while cov(hi) < 0.68 and hi < 32.0:
        lo = hi
        hi *= 1.5
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if cov(mid) < 0.68:
            lo = mid
        else:
            hi = mid
    return float(hi)


def summarize(method: str, regime: str, sigma: float, samples: np.ndarray, target: np.ndarray, fair, strong_mod, disp, mask, args):
    pred = np.median(samples, axis=1)
    if args.skip_dispersion_residuals:
        disp_vec = [np.empty((0,), dtype=np.float32) for _ in range(len(target))]
    else:
        disp_vec = fair.dispersion_residual_vectors(fair.boundary_mod, strong_mod, pred, disp, mask)
    empty_envelope = {"lo": np.full_like(target[0], -np.inf), "hi": np.full_like(target[0], np.inf)}
    base = fair.metrics_for_indices(np.arange(len(target)), pred, samples, target, disp_vec, empty_envelope)
    spread = samples.std(axis=1)
    row = {
        "method": method,
        "test_set": regime,
        "noise_sigma_km_s": sigma,
        "n": int(len(target)),
        "posterior_samples": int(samples.shape[1]),
        "euler_steps": int(args.euler_steps),
        "temperature_scale_68": fit_temperature(samples[: max(1, len(target) // 2)], target[: max(1, len(target) // 2)]),
        "posterior_std_mean": float(spread.mean()),
        "posterior_std_vs": float(spread[:, 1, :].mean()),
    }
    row.update(
        {
            k: v
            for k, v in base.items()
            if not k.startswith("target_outside") and not k.startswith("pred_inside") and k != "boundary_pull_fraction"
        }
    )
    return row


def plot_noise(rows: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({r["method"] for r in rows})
    regimes = ["in-prior", "boundary", "out-of-prior"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
    for ax, metric, ylabel in zip(axes, ["vs_mae", "coverage_vs", "posterior_std_vs"], ["$V_S$ MAE", "$V_S$ coverage", "$V_S$ posterior std"]):
        for method in methods:
            subset = [r for r in rows if r["method"] == method and r["test_set"] == "in-prior"]
            subset = sorted(subset, key=lambda r: float(r["noise_sigma_km_s"]))
            ax.plot([r["noise_sigma_km_s"] for r in subset], [r[metric] for r in subset], marker="o", label=method)
        ax.set_xlabel("Noise sigma (km/s)")
        ax.set_ylabel(ylabel)
        ax.grid(color="#e5e5e5", linewidth=0.5)
    axes[0].legend(frameon=False)
    fig.suptitle("In-prior noise sensitivity")
    fig.tight_layout()
    fig.savefig(fig_dir / "fair_noise_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fair_noise_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strong-ckpt", type=Path, default=ROOT / "ckpt/fair_di_strong_full_seed642026/best.pt")
    p.add_argument("--weak-ckpt", type=Path, default=ROOT / "ckpt/fair_di_weak_full_seed642026/best.pt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/noise")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production/noise")
    p.add_argument("--noise-sigma-km-s", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10])
    p.add_argument("--n-eval", type=int, default=1024)
    p.add_argument("--posterior-samples", type=int, default=64)
    p.add_argument("--euler-steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=642026)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-dispersion-residuals", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    fair = import_from_path("fair_eval_for_noise", ROOT / "scripts/eval_fair_di_comparison.py")
    fair.boundary_mod = import_from_path("prior_boundary_for_noise", ROOT / "scripts/eval_prior_boundary_effect.py")
    strong_mod = import_from_path("strong_generator_for_noise", ROOT / "utils/generate_data.py")
    strong_model, _ = fair.boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.strong_ckpt, device)
    weak_model, _ = fair.boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.weak_ckpt, device)
    test_sets = fair.make_test_sets(fair.boundary_mod, strong_mod, argparse.Namespace(n_test=args.n_eval, seed=args.seed))
    rows = []
    for method, model in (("DI-Strong", strong_model), ("DI-Weak", weak_model)):
        for regime, (target, disp, mask) in test_sets.items():
            for j, sigma in enumerate(args.noise_sigma_km_s):
                noisy = add_noise(disp, mask, sigma, args.seed + 1000 * j + len(method))
                samples = fair.boundary_mod.direct_samples(model, noisy, mask, device, args.posterior_samples, args.euler_steps, args.batch_size)
                rows.append(summarize(method, regime, sigma, samples, target, fair, strong_mod, noisy, mask, args))
    write_csv(args.out_dir / "noise_sensitivity.csv", rows)
    write_json(args.out_dir / "noise_sensitivity.json", {"protocol": vars(args), "rows": rows})
    plot_noise(rows, args.fig_dir)
    print(f"Wrote noise sensitivity diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
