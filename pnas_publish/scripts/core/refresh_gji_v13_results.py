#!/usr/bin/env python3
"""Refresh GJI manuscript statistics and field figures with v1.3 checkpoints.

This script is intentionally conservative: it waits for completed v1.3
checkpoints, recomputes synthetic diagnostics, evaluates the relationship
between missing period bands and posterior uncertainty, and reruns the Bayan
Obo field demonstration using the v1.3 weak-prior checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GJI = ROOT / "gji_dnn_posterior_inversion"
GJI_RESULTS = GJI / "results"
GJI_FIGURES = GJI / "figures"
V13_STRONG = ROOT / "ckpt" / "disp2struct_crf.v1.3_cp" / "best.pt"
V13_WEAK = ROOT / "ckpt" / "disp2struct_crf.v1.3_cp_weak" / "best.pt"

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


def checkpoint_epoch(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        ckpt = torch.load(path, map_location="cpu")
        return int(ckpt.get("epoch", -1))
    except Exception:
        return -1


def completed_epoch(path: Path) -> int:
    latest = path.with_name("latest.pt")
    return max(checkpoint_epoch(path), checkpoint_epoch(latest))


def wait_for_checkpoints(strong_ckpt: Path, weak_ckpt: Path, strong_epoch: int, weak_epoch: int, poll_s: int) -> None:
    while True:
        se = completed_epoch(strong_ckpt)
        we = completed_epoch(weak_ckpt)
        print(f"[wait] strong epoch={se}/{strong_epoch}; weak epoch={we}/{weak_epoch}", flush=True)
        if se >= strong_epoch and we >= weak_epoch:
            return
        time.sleep(poll_s)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        p = disp[i, 0]
        width = rng.uniform(15.0, 45.0)
        start = rng.uniform(float(p.min()), float(p.max() - width))
        end = start + width
        widths[i] = width
        keep = (p >= start) & (p <= end)
        hole_count = rng.integers(0, 7)
        valid = np.where(keep)[0]
        if len(valid) and hole_count:
            keep[rng.choice(valid, size=min(hole_count, len(valid)), replace=False)] = False
        point_keep = rng.random(len(p)) >= 0.04
        keep &= point_keep
        if not keep.any():
            keep[valid[len(valid) // 2]] = True
        out[i, 1:3, :] *= keep[None, :].astype(out.dtype)
        out[i, 0, :] = (out[i, 1:3, :].sum(axis=0) > 0).astype(out.dtype)
    return out, widths


def evaluate_missing_band(boundary, model, dataset_mod, device: torch.device, args) -> tuple[list[dict], dict]:
    models_full, disp, base_mask = boundary.dataset_to_arrays(
        dataset_mod.SurfaceWaveDataset(
            n_samples=args.n_missing,
            z_max_km=150.0,
            z_max_num=256,
            dz_km=0.5,
            seed=args.seed + 70_000,
        )
    )
    target = models_full[:, 1:4, :]
    windows = {
        "full-2-60": base_mask,
        "p2-40": apply_period_window(disp, base_mask, 2.0, 40.0),
        "p10-30": apply_period_window(disp, base_mask, 10.0, 30.0),
        "p15-45": apply_period_window(disp, base_mask, 15.0, 45.0),
    }
    random_mask, random_width = apply_random_windows(disp, base_mask, args.seed + 71_000)
    windows["random-window"] = random_mask

    rows = []
    per_example = {}
    for name, mask in windows.items():
        samples = boundary.direct_samples(
            model,
            disp,
            mask,
            device,
            n_samples=args.posterior_samples,
            steps=args.sampling_steps,
            batch_size=args.batch_size,
        )
        pred = np.median(samples, axis=1)
        q16 = np.quantile(samples, 0.16, axis=1)
        q84 = np.quantile(samples, 0.84, axis=1)
        inside = (target >= q16) & (target <= q84)
        spread = np.std(samples, axis=1)
        valid_fraction = mask[:, 1:3, :].sum(axis=(1, 2)) / max(float(base_mask[:, 1:3, :].shape[1] * base_mask.shape[2]), 1.0)
        per_vs_mae = np.abs(pred[:, 1, :] - target[:, 1, :]).mean(axis=1)
        per_vs_spread = spread[:, 1, :].mean(axis=1)
        row = {
            "window": name,
            "n": len(target),
            "valid_fraction_mean": float(valid_fraction.mean()),
            "vs_mae": float(per_vs_mae.mean()),
            "vs_spread_mean": float(per_vs_spread.mean()),
            "coverage_vs": float(inside[:, 1, :].mean()),
            "coverage_mean": float(inside.mean()),
        }
        if name == "random-window":
            row["random_width_mean_s"] = float(random_width.mean())
            row["corr_valid_fraction_vs_spread"] = float(np.corrcoef(valid_fraction, per_vs_spread)[0, 1])
            row["corr_valid_fraction_vs_mae"] = float(np.corrcoef(valid_fraction, per_vs_mae)[0, 1])
        rows.append(row)
        per_example[name] = {
            "valid_fraction": valid_fraction,
            "vs_mae": per_vs_mae,
            "vs_spread": per_vs_spread,
        }
    return rows, per_example


def plot_missing_band(path: Path, rows: list[dict], per_example: dict) -> None:
    labels = [r["window"] for r in rows]
    x = np.arange(len(rows))
    vs_mae = [r["vs_mae"] for r in rows]
    spread = [r["vs_spread_mean"] for r in rows]
    cov = [r["coverage_vs"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].bar(x, vs_mae, color="#4c78a8")
    axes[0].set_ylabel("$V_S$ MAE (km/s)")
    axes[1].bar(x, spread, color="#f58518")
    axes[1].set_ylabel("mean posterior std $V_S$ (km/s)")
    axes[2].bar(x, cov, color="#54a24b")
    axes[2].axhline(0.68, color="k", lw=0.8, ls="--")
    axes[2].set_ylabel("16--84% coverage $V_S$")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    r = per_example.get("random-window")
    if r:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.scatter(r["valid_fraction"], r["vs_spread"], s=18, alpha=0.6)
        ax.set_xlabel("valid Rayleigh/Love fraction")
        ax.set_ylabel("mean posterior std $V_S$ (km/s)")
        ax.grid(True, color="0.9")
        fig.tight_layout()
        scatter = path.with_name(path.stem + "_scatter.png")
        fig.savefig(scatter, dpi=300)
        fig.savefig(scatter.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def run_field_refresh(args) -> None:
    out_dir = ROOT / "field_masw_results_v13_p2_40"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "field_masw_posterior_inversion.py"),
        "--checkpoint",
        str(args.weak_ckpt),
        "--out-dir",
        str(out_dir),
        "--num-samples",
        str(args.field_samples),
        "--num-steps",
        str(args.field_steps),
        "--batch-size",
        str(args.batch_size),
        "--max-period",
        "40",
        "--device",
        args.device,
    ]
    print("[field]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    for name in [
        "field_dispersion_qc.pdf",
        "field_vs_median_slices.pdf",
        "field_vs_std_slices.pdf",
    ]:
        src = out_dir / name
        # Overwrite the manuscript-facing field figure names so Figure 9 is
        # replaced by the v1.3 field inversion without requiring a LaTeX edit.
        shutil.copyfile(src, GJI_FIGURES / name.replace("field_", "field_masw_"))
        shutil.copyfile(src, GJI_FIGURES / name.replace("field_", "field_masw_v13_"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-ckpt", type=Path, default=V13_STRONG)
    parser.add_argument("--weak-ckpt", type=Path, default=V13_WEAK)
    parser.add_argument("--expected-strong-epoch", type=int, default=24)
    parser.add_argument("--expected-weak-epoch", type=int, default=12)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--poll-s", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-validation", type=int, default=256)
    parser.add_argument("--n-boundary", type=int, default=64)
    parser.add_argument("--n-envelope", type=int, default=512)
    parser.add_argument("--n-missing", type=int, default=128)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--field-samples", type=int, default=32)
    parser.add_argument("--field-steps", type=int, default=24)
    args = parser.parse_args()

    if args.wait:
        wait_for_checkpoints(
            args.strong_ckpt,
            args.weak_ckpt,
            args.expected_strong_epoch,
            args.expected_weak_epoch,
            args.poll_s,
        )

    GJI_RESULTS.mkdir(parents=True, exist_ok=True)
    GJI_FIGURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = choose_device(args.device)

    boundary = import_from_path("eval_prior_boundary_v13_refresh", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    strong_mod = import_from_path("generate_data_v13_refresh", ROOT / "utils" / "generate_data.py")
    weak_mod = import_from_path("generate_weak_v13_refresh", ROOT / "utils" / "generate_data_weak_prior.py")

    envelope = boundary.prior_envelope(strong_mod, args.n_envelope, args.seed + 10)
    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    in_models_full, in_disp, in_mask = boundary.dataset_to_arrays(boundary.strong_dataset(strong_mod, args.n_boundary, args.seed + 20))
    test_sets = {
        "in-prior": (in_models_full[:, 1:4, :], in_disp, in_mask),
        "boundary": boundary.parametric_dataset(strong_mod, "boundary", args.n_boundary, args.seed + 30, 256, periods),
        "out-of-prior": boundary.parametric_dataset(strong_mod, "out-of-prior", args.n_boundary, args.seed + 40, 256, periods),
    }

    eval_args = SimpleNamespace(
        posterior_samples=args.posterior_samples,
        sampling_steps=args.sampling_steps,
        batch_size=args.batch_size,
    )

    strong_model, _ = boundary.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.strong_ckpt, device)
    weak_model, _ = boundary.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.weak_ckpt, device)

    strong_rows, strong_diag = boundary.evaluate_direct("DI-Strong-v1.3", strong_model, test_sets, strong_mod, envelope, device, eval_args)
    weak_rows, weak_diag = boundary.evaluate_direct("DI-Weak-v1.3", weak_model, test_sets, strong_mod, envelope, device, eval_args)
    write_csv(GJI_RESULTS / "prior_boundary_strong_v13.csv", strong_rows)
    write_csv(GJI_RESULTS / "prior_boundary_weak_v13.csv", weak_rows)
    write_csv(GJI_RESULTS / "prior_boundary_v13_combined.csv", strong_rows + weak_rows)
    boundary.plot_direct_examples(GJI_FIGURES / "direct_prior_boundary_examples_v13.png", weak_diag)

    missing_rows, missing_examples = evaluate_missing_band(boundary, weak_model, weak_mod, device, args)
    write_csv(GJI_RESULTS / "missing_band_uncertainty_v13.csv", missing_rows)
    plot_missing_band(GJI_FIGURES / "missing_band_uncertainty_v13.png", missing_rows, missing_examples)

    run_field_refresh(args)

    summary_lines = ["# v1.3 result refresh", ""]
    summary_lines.append(f"Strong checkpoint: `{args.strong_ckpt}` epoch {checkpoint_epoch(args.strong_ckpt)}")
    summary_lines.append(f"Weak checkpoint: `{args.weak_ckpt}` epoch {checkpoint_epoch(args.weak_ckpt)}")
    summary_lines.append("")
    summary_lines.append("## Prior-boundary rows")
    for row in strong_rows + weak_rows:
        summary_lines.append(
            f"- {row['method']} {row['test_set']}: Vs MAE={row['vs_mae']:.3f}, "
            f"disp MAE={row['pred_disp_mae']:.3f}, cov={row['coverage_16_84_mean']:.3f}, "
            f"pull-in={row['pred_inside_given_target_outside']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append("## Missing-band rows")
    for row in missing_rows:
        summary_lines.append(
            f"- {row['window']}: valid={row['valid_fraction_mean']:.3f}, "
            f"Vs MAE={row['vs_mae']:.3f}, spread={row['vs_spread_mean']:.3f}, covVs={row['coverage_vs']:.3f}"
        )
    (GJI_RESULTS / "v13_refresh_summary.md").write_text("\n".join(summary_lines) + "\n")
    print("[done] wrote v1.3 GJI refresh outputs", flush=True)


if __name__ == "__main__":
    main()
