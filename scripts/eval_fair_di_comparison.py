#!/usr/bin/env python3
"""Fair DI-Strong versus DI-Weak evaluation.

This script evaluates two direct-inversion checkpoints trained with matched
budgets. It writes new fair-comparison outputs and intentionally does not
overwrite the earlier prior-boundary diagnostic products.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
DEFAULT_STRONG_CKPT = ROOT / "ckpt" / "fair_di_strong_full_seed642026" / "best.pt"
DEFAULT_WEAK_CKPT = ROOT / "ckpt" / "fair_di_weak_full_seed642026" / "best.pt"
DEFAULT_OUT_DIR = ROOT / "results" / "fair_di_comparison"
DEFAULT_FIG_DIR = ROOT / "figures" / "fair_di_comparison"
CHANNELS = ("vp", "vs", "rho")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def tensor_to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    return value


def load_checkpoint_config(path: Path) -> Dict[str, object]:
    ckpt = torch.load(path, map_location="cpu")
    return tensor_to_jsonable(ckpt.get("config", {}))


def make_test_sets(boundary_mod, strong_mod, args) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    in_models, in_disp, in_mask = boundary_mod.dataset_to_arrays(
        boundary_mod.strong_dataset(strong_mod, args.n_test, args.seed + 100)
    )
    in_profiles = in_models[:, 1:4, :].astype(np.float32)
    boundary_profiles, boundary_disp, boundary_mask = boundary_mod.parametric_dataset(
        strong_mod, "boundary", args.n_test, args.seed + 200, in_profiles.shape[-1], periods
    )
    out_profiles, out_disp, out_mask = boundary_mod.parametric_dataset(
        strong_mod, "out-of-prior", args.n_test, args.seed + 300, in_profiles.shape[-1], periods
    )
    return {
        "in-prior": (in_profiles, in_disp.astype(np.float32), in_mask.astype(np.float32)),
        "boundary": (boundary_profiles.astype(np.float32), boundary_disp.astype(np.float32), boundary_mask.astype(np.float32)),
        "out-of-prior": (out_profiles.astype(np.float32), out_disp.astype(np.float32), out_mask.astype(np.float32)),
    }


def prior_envelope_fast(strong_mod, n: int, seed: int, n_depth: int, dz_km: float = 0.5) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    profiles = []
    z_max_km = (n_depth - 1) * dz_km
    for _ in range(n):
        depth, vs, vp, rho, _ = strong_mod.sample_global_1d_model(
            z_max_km=z_max_km,
            dz_km=dz_km,
            rng=rng,
        )
        if len(depth) != n_depth:
            depth_target = np.arange(n_depth, dtype=np.float32) * dz_km
            vp = np.interp(depth_target, depth, vp)
            vs = np.interp(depth_target, depth, vs)
            rho = np.interp(depth_target, depth, rho)
        profiles.append(np.stack([vp, vs, rho]).astype(np.float32))
    models = np.stack(profiles)
    return {
        "lo": np.quantile(models, 0.01, axis=0).astype(np.float32),
        "hi": np.quantile(models, 0.99, axis=0).astype(np.float32),
        "mean": models.mean(axis=0).astype(np.float32),
    }


def dispersion_residual_vectors(boundary_mod, strong_mod, pred: np.ndarray, disp: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    periods = disp[0, 0, :]
    vectors: List[np.ndarray] = []
    for i in range(len(pred)):
        pred_disp = boundary_mod.compute_dispersion(strong_mod, pred[i], periods)
        if pred_disp is None:
            vectors.append(np.empty((0,), dtype=np.float32))
            continue
        wave_mask = mask[i, 1:3, :].astype(bool)
        vectors.append((pred_disp[1:3, :] - disp[i, 1:3, :])[wave_mask].astype(np.float32))
    return vectors


def metrics_for_indices(
    idx: np.ndarray,
    pred: np.ndarray,
    samples: np.ndarray,
    target: np.ndarray,
    disp_vectors: List[np.ndarray],
    envelope: Dict[str, np.ndarray],
) -> Dict[str, float]:
    p = pred[idx]
    s = samples[idx]
    t = target[idx]
    err = p - t
    out: Dict[str, float] = {}
    for i, name in enumerate(CHANNELS):
        out[f"{name}_mae"] = float(np.mean(np.abs(err[:, i, :])))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err[:, i, :] ** 2)))
    q16 = np.quantile(s, 0.16, axis=1)
    q84 = np.quantile(s, 0.84, axis=1)
    inside = (t >= q16) & (t <= q84)
    out["coverage_16_84_mean"] = float(inside.mean())
    for i, name in enumerate(CHANNELS):
        out[f"coverage_{name}"] = float(inside[:, i, :].mean())

    lo = envelope["lo"][None, :, :]
    hi = envelope["hi"][None, :, :]
    target_outside = (t < lo) | (t > hi)
    pred_inside = (p >= lo) & (p <= hi)
    near_lo = np.abs(p - lo) < 0.05 * np.maximum(hi - lo, 1e-3)
    near_hi = np.abs(p - hi) < 0.05 * np.maximum(hi - lo, 1e-3)
    outside_count = float(target_outside.sum())
    out["target_outside_fraction"] = float(target_outside.mean())
    out["pred_inside_given_target_outside"] = float((pred_inside & target_outside).sum() / max(outside_count, 1.0))
    out["boundary_pull_fraction"] = float(((near_lo | near_hi) & target_outside).sum() / max(outside_count, 1.0))

    residual_parts = [disp_vectors[int(i)] for i in idx if len(disp_vectors[int(i)])]
    if residual_parts:
        r = np.concatenate(residual_parts)
        out["pred_disp_mae"] = float(np.mean(np.abs(r)))
        out["pred_disp_rmse"] = float(np.sqrt(np.mean(r**2)))
    else:
        out["pred_disp_mae"] = math.nan
        out["pred_disp_rmse"] = math.nan
    return out


def bootstrap_ci(
    pred: np.ndarray,
    samples: np.ndarray,
    target: np.ndarray,
    disp_vectors: List[np.ndarray],
    envelope: Dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(target)
    boot_rows: Dict[str, List[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        row = metrics_for_indices(idx, pred, samples, target, disp_vectors, envelope)
        for key, value in row.items():
            if math.isfinite(float(value)):
                boot_rows.setdefault(key, []).append(float(value))
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in boot_rows.items()
        if values
    }


def stable_offset(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100_000


def evaluate_model(label: str, model, test_sets, boundary_mod, strong_mod, envelope, device, args):
    rows = []
    diagnostics = {}
    for test_name, (target, disp, mask) in test_sets.items():
        tic = time.time()
        samples = boundary_mod.direct_samples(
            model,
            disp,
            mask,
            device,
            n_samples=args.posterior_samples,
            steps=args.euler_steps,
            batch_size=args.batch_size,
        )
        pred = np.median(samples, axis=1)
        if args.skip_dispersion_residuals:
            disp_vec = [np.empty((0,), dtype=np.float32) for _ in range(len(target))]
        else:
            disp_vec = dispersion_residual_vectors(boundary_mod, strong_mod, pred, disp, mask)
        base = metrics_for_indices(np.arange(len(target)), pred, samples, target, disp_vec, envelope)
        ci = bootstrap_ci(
            pred,
            samples,
            target,
            disp_vec,
            envelope,
            n_boot=args.bootstrap,
            seed=args.seed + stable_offset(f"{label}:{test_name}"),
        )
        row: Dict[str, object] = {
            "method": label,
            "test_set": test_name,
            "n": int(len(target)),
            "posterior_samples": int(args.posterior_samples),
            "euler_steps": int(args.euler_steps),
            "runtime_s": float(time.time() - tic),
            "bootstrap_n": int(args.bootstrap),
        }
        row.update(base)
        for key, (lo, hi) in ci.items():
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        rows.append(row)
        diagnostics[f"{label}_{test_name}"] = {
            "target": target,
            "pred": pred,
            "samples": samples,
            "mask": mask,
            "disp": disp,
        }
    return rows, diagnostics


def plot_metric_summary(rows: List[Dict[str, object]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tests = ["in-prior", "boundary", "out-of-prior"]
    methods = ["DI-Strong", "DI-Weak"]
    colors = {"DI-Strong": "#3b82c4", "DI-Weak": "#6aa84f"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, metric, title in zip(
        axes,
        ["vs_mae", "coverage_vs", "pred_disp_mae"],
        [r"$V_S$ MAE (km s$^{-1}$)", r"$V_S$ 16--84% coverage", "Dispersion residual MAE"],
    ):
        width = 0.35
        x = np.arange(len(tests))
        for j, method in enumerate(methods):
            vals = []
            yerr_low = []
            yerr_high = []
            for test in tests:
                row = next(r for r in rows if r["method"] == method and r["test_set"] == test)
                val = float(row[metric])
                vals.append(val)
                lo = float(row.get(f"{metric}_ci_low", val))
                hi = float(row.get(f"{metric}_ci_high", val))
                yerr_low.append(max(0.0, val - lo))
                yerr_high.append(max(0.0, hi - val))
            ax.bar(x + (j - 0.5) * width, vals, width=width, color=colors[method], alpha=0.78, label=method)
            if any(v > 0 for v in yerr_low + yerr_high):
                ax.errorbar(
                    x + (j - 0.5) * width,
                    vals,
                    yerr=np.vstack([yerr_low, yerr_high]),
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.8,
                    capsize=2.5,
                )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["In-prior", "Boundary", "Out-of-prior"], rotation=20, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fair_di_metric_summary.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fair_di_metric_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_profiles(diagnostics: Dict[str, Dict[str, np.ndarray]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tests = ["in-prior", "boundary", "out-of-prior"]
    methods = ["DI-Strong", "DI-Weak"]
    colors = {"DI-Strong": "#3b82c4", "DI-Weak": "#6aa84f"}
    depth = np.arange(next(iter(diagnostics.values()))["target"].shape[-1], dtype=np.float32) * 0.5
    fig, axes = plt.subplots(len(tests), len(methods) + 1, figsize=(8.2, 7.0), sharey=True, sharex=True)
    for r, test in enumerate(tests):
        first = diagnostics[f"DI-Strong_{test}"]
        axes[r, 0].plot(first["target"][0, 1], depth, color="#111111", linewidth=1.2)
        axes[r, 0].set_ylabel(f"{test}\nDepth (km)", fontsize=9)
        axes[r, 0].set_title("Truth" if r == 0 else "", fontsize=9)
        axes[r, 0].invert_yaxis()
        for c, method in enumerate(methods, start=1):
            diag = diagnostics[f"{method}_{test}"]
            samples = diag["samples"][0, :, 1, :]
            pred = diag["pred"][0, 1, :]
            q16 = np.quantile(samples, 0.16, axis=0)
            q84 = np.quantile(samples, 0.84, axis=0)
            axes[r, c].fill_betweenx(depth, q16, q84, color=colors[method], alpha=0.22, linewidth=0)
            axes[r, c].plot(pred, depth, color=colors[method], linewidth=1.2)
            axes[r, c].plot(diag["target"][0, 1], depth, color="#111111", linewidth=0.7, alpha=0.7)
            axes[r, c].set_title(method if r == 0 else "", fontsize=9)
    for ax in axes.ravel():
        ax.grid(color="#e5e5e5", linewidth=0.5)
        ax.set_xlim(0.0, 6.2)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$V_S$ (km s$^{-1}$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fair_di_example_profiles.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fair_di_example_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics(path: Path, diagnostics: Dict[str, Dict[str, np.ndarray]]) -> None:
    arrays = {}
    for diag_name, diag in diagnostics.items():
        prefix = diag_name.replace("-", "_").replace(" ", "_")
        for key, value in diag.items():
            arrays[f"{prefix}_{key}"] = value
    np.savez_compressed(path, **arrays)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    parser.add_argument("--weak-ckpt", type=Path, default=DEFAULT_WEAK_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--n-envelope", type=int, default=2048)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--euler-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=642026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-dispersion-residuals",
        action="store_true",
        help="Skip forward-solver recomputation of prediction dispersion residuals for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    boundary_mod = import_from_path("prior_boundary_eval_for_fair", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    strong_mod = import_from_path("strong_generator_for_fair", ROOT / "utils" / "generate_data.py")

    missing = [str(p) for p in (args.strong_ckpt, args.weak_ckpt) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    strong_model, strong_cfg = boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.strong_ckpt, device)
    weak_model, weak_cfg = boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.weak_ckpt, device)
    if strong_model is None or weak_model is None:
        raise RuntimeError("Could not load both direct-inversion models")

    test_sets = make_test_sets(boundary_mod, strong_mod, args)
    first_target = next(iter(test_sets.values()))[0]
    envelope = prior_envelope_fast(strong_mod, args.n_envelope, args.seed + 400, first_target.shape[-1])
    rows = []
    diagnostics = {}
    strong_rows, strong_diag = evaluate_model("DI-Strong", strong_model, test_sets, boundary_mod, strong_mod, envelope, device, args)
    weak_rows, weak_diag = evaluate_model("DI-Weak", weak_model, test_sets, boundary_mod, strong_mod, envelope, device, args)
    rows.extend(strong_rows)
    rows.extend(weak_rows)
    diagnostics.update(strong_diag)
    diagnostics.update(weak_diag)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fair_di_metrics.csv", rows)
    write_json(args.out_dir / "fair_di_metrics.json", rows)
    write_json(
        args.out_dir / "fair_di_protocol.json",
        {
            "created_unix_time": time.time(),
            "device": str(device),
            "strong_checkpoint": str(args.strong_ckpt),
            "weak_checkpoint": str(args.weak_ckpt),
            "strong_config": tensor_to_jsonable(strong_cfg),
            "weak_config": tensor_to_jsonable(weak_cfg),
            "n_test": args.n_test,
            "n_envelope": args.n_envelope,
            "posterior_samples": args.posterior_samples,
            "euler_steps": args.euler_steps,
            "bootstrap": args.bootstrap,
            "skip_dispersion_residuals": bool(args.skip_dispersion_residuals),
            "seed": args.seed,
            "test_sets": sorted(test_sets.keys()),
            "intended_difference": "structural_prior_generator_only",
        },
    )
    save_diagnostics(args.out_dir / "fair_di_diagnostics.npz", diagnostics)
    plot_metric_summary(rows, args.fig_dir)
    plot_example_profiles(diagnostics, args.fig_dir)
    print(f"Wrote fair DI comparison to {args.out_dir}")
    print(f"Wrote fair DI figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
