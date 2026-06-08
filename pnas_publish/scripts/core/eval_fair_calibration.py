#!/usr/bin/env python3
"""Calibration diagnostics for matched fair DI-Strong and DI-Weak checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
NOMINALS = (50.0, 68.0, 90.0)
DEPTH_BINS = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0), (100.0, 128.0))


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
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scaled_samples(samples: np.ndarray, scale: float) -> np.ndarray:
    median = np.median(samples, axis=1, keepdims=True)
    return median + scale * (samples - median)


def interval_coverage(samples: np.ndarray, target: np.ndarray, nominal: float, scale: float = 1.0) -> np.ndarray:
    s = scaled_samples(samples, scale)
    lo = (50.0 - nominal / 2.0) / 100.0
    hi = (50.0 + nominal / 2.0) / 100.0
    qlo = np.quantile(s, lo, axis=1)
    qhi = np.quantile(s, hi, axis=1)
    return ((target >= qlo) & (target <= qhi)).astype(np.float32)


def mean_coverage(samples: np.ndarray, target: np.ndarray, nominal: float, scale: float) -> float:
    return float(interval_coverage(samples, target, nominal, scale).mean())


def fit_temperature(samples: np.ndarray, target: np.ndarray, nominal: float = 68.0) -> float:
    target_cov = nominal / 100.0
    if mean_coverage(samples, target, nominal, 1.0) >= target_cov:
        return 1.0
    lo, hi = 1.0, 1.5
    while mean_coverage(samples, target, nominal, hi) < target_cov and hi < 32.0:
        lo = hi
        hi *= 1.5
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if mean_coverage(samples, target, nominal, mid) < target_cov:
            lo = mid
        else:
            hi = mid
    return float(hi)


def rank_histogram(samples: np.ndarray, target: np.ndarray, bins: int = 20) -> Dict[str, object]:
    ranks = (samples <= target[:, None, :, :]).mean(axis=1).reshape(-1)
    hist, edges = np.histogram(ranks, bins=bins, range=(0.0, 1.0))
    return {"bin_edges": edges.tolist(), "counts": hist.astype(int).tolist()}


def depth_bin_rows(method: str, regime: str, samples: np.ndarray, target: np.ndarray, scale: float) -> List[Dict[str, object]]:
    depth = np.arange(target.shape[-1], dtype=np.float32) * 0.5
    rows = []
    cov = interval_coverage(samples, target, 68.0, scale)
    std = scaled_samples(samples, scale).std(axis=1)
    for lo, hi in DEPTH_BINS:
        keep = (depth >= lo) & (depth < hi)
        if not keep.any():
            continue
        for idx, channel in enumerate(CHANNELS):
            rows.append(
                {
                    "method": method,
                    "test_set": regime,
                    "depth_min_km": lo,
                    "depth_max_km": hi,
                    "channel": channel,
                    "coverage_68": float(cov[:, idx, :][:, keep].mean()),
                    "posterior_std_mean": float(std[:, idx, :][:, keep].mean()),
                }
            )
    return rows


def reliability_rows(method: str, regime: str, samples: np.ndarray, target: np.ndarray, scale: float, label: str) -> List[Dict[str, object]]:
    rows = []
    for nominal in NOMINALS:
        cov = interval_coverage(samples, target, nominal, scale)
        row = {
            "method": method,
            "test_set": regime,
            "scale_label": label,
            "temperature_scale": scale,
            "nominal_percent": nominal,
            "coverage_mean": float(cov.mean()),
        }
        for i, channel in enumerate(CHANNELS):
            row[f"coverage_{channel}"] = float(cov[:, i, :].mean())
        rows.append(row)
    return rows


def evaluate_method(method: str, model, test_sets, device, args, fair_mod) -> tuple[list[dict], list[dict], list[dict], dict]:
    metric_rows: List[dict] = []
    depth_rows: List[dict] = []
    rank_rows: List[dict] = []
    temp_summary: Dict[str, object] = {}
    for regime, (target, disp, mask) in test_sets.items():
        samples = fair_mod.boundary_mod.direct_samples(
            model,
            disp,
            mask,
            device,
            n_samples=args.posterior_samples,
            steps=args.euler_steps,
            batch_size=args.batch_size,
        )
        if args.calibration_examples >= len(target):
            raise ValueError("--calibration-examples must be smaller than the examples per regime")
        rng = np.random.default_rng(args.seed + sum(ord(c) for c in f"{method}:{regime}"))
        perm = rng.permutation(len(target))
        cal_idx = perm[: args.calibration_examples]
        test_idx = perm[args.calibration_examples :]
        temp = fit_temperature(samples[cal_idx], target[cal_idx], nominal=68.0)
        temp_summary[f"{method}_{regime}"] = {
            "temperature_scale": temp,
            "calibration_examples": int(len(cal_idx)),
            "test_examples": int(len(test_idx)),
        }
        metric_rows.extend(reliability_rows(method, regime, samples[test_idx], target[test_idx], 1.0, "raw"))
        metric_rows.extend(reliability_rows(method, regime, samples[test_idx], target[test_idx], temp, "temperature_scaled"))
        depth_rows.extend(
            {**row, "scale_label": "raw"}
            for row in depth_bin_rows(method, regime, samples[test_idx], target[test_idx], 1.0)
        )
        depth_rows.extend(
            {**row, "scale_label": "temperature_scaled"}
            for row in depth_bin_rows(method, regime, samples[test_idx], target[test_idx], temp)
        )
        for label, scale in (("raw", 1.0), ("temperature_scaled", temp)):
            hist = rank_histogram(scaled_samples(samples[test_idx], scale), target[test_idx], bins=args.rank_bins)
            for b, count in enumerate(hist["counts"]):
                rank_rows.append(
                    {
                        "method": method,
                        "test_set": regime,
                        "scale_label": label,
                        "bin_left": hist["bin_edges"][b],
                        "bin_right": hist["bin_edges"][b + 1],
                        "count": count,
                    }
                )
    return metric_rows, depth_rows, rank_rows, temp_summary


def plot_reliability(rows: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({r["method"] for r in rows})
    regimes = ["in-prior", "boundary", "out-of-prior"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(10.5, 3.2), sharex=True, sharey=True)
    for ax, regime in zip(axes, regimes):
        ax.plot([0, 100], [0, 1], color="#222222", linestyle="--", linewidth=0.8)
        for method in methods:
            for label, style in (("raw", "-"), ("temperature_scaled", ":")):
                subset = [r for r in rows if r["method"] == method and r["test_set"] == regime and r["scale_label"] == label]
                subset = sorted(subset, key=lambda r: float(r["nominal_percent"]))
                if not subset:
                    continue
                ax.plot(
                    [float(r["nominal_percent"]) for r in subset],
                    [float(r["coverage_mean"]) for r in subset],
                    marker="o",
                    linestyle=style,
                    label=f"{method} {label}" if regime == regimes[0] else None,
                )
        ax.set_title(regime)
        ax.set_xlabel("Nominal interval (%)")
        ax.grid(color="#e5e5e5", linewidth=0.5)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fair_calibration_reliability.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fair_calibration_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strong-ckpt", type=Path, default=ROOT / "ckpt/fair_di_strong_full_seed642026/best.pt")
    p.add_argument("--weak-ckpt", type=Path, default=ROOT / "ckpt/fair_di_weak_full_seed642026/best.pt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/calibration")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production/calibration")
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--calibration-examples", type=int, default=1024)
    p.add_argument("--posterior-samples", type=int, default=64)
    p.add_argument("--euler-steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--rank-bins", type=int, default=20)
    p.add_argument("--seed", type=int, default=642026)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    fair = import_from_path("fair_eval_for_calibration", ROOT / "scripts/eval_fair_di_comparison.py")
    fair.boundary_mod = import_from_path("prior_boundary_for_calibration", ROOT / "scripts/eval_prior_boundary_effect.py")
    strong_mod = import_from_path("strong_generator_for_calibration", ROOT / "utils/generate_data.py")
    strong_model, _ = fair.boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.strong_ckpt, device)
    weak_model, _ = fair.boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.weak_ckpt, device)
    test_sets = fair.make_test_sets(fair.boundary_mod, strong_mod, argparse.Namespace(n_test=args.n_eval, seed=args.seed))

    rows: list[dict] = []
    depth_rows: list[dict] = []
    rank_rows: list[dict] = []
    temps: dict = {}
    for method, model in (("DI-Strong", strong_model), ("DI-Weak", weak_model)):
        r, d, rank, temp = evaluate_method(method, model, test_sets, device, args, fair)
        rows.extend(r)
        depth_rows.extend(d)
        rank_rows.extend(rank)
        temps.update(temp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "calibration_metrics.csv", rows)
    write_csv(args.out_dir / "depth_binned_coverage.csv", depth_rows)
    write_csv(args.out_dir / "rank_diagnostics.csv", rank_rows)
    write_json(args.out_dir / "temperature_scaling.json", temps)
    write_json(args.out_dir / "calibration_protocol.json", vars(args))
    plot_reliability(rows, args.fig_dir)
    print(f"Wrote calibration diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
