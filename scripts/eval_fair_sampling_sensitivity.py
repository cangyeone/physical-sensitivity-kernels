#!/usr/bin/env python3
"""Posterior sample-count and Euler-step sensitivity for the fair DI-Weak model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def dispersion_mae(fair, boundary, strong_mod, pred: np.ndarray, disp: np.ndarray, mask: np.ndarray) -> float:
    vectors = fair.dispersion_residual_vectors(boundary, strong_mod, pred, disp, mask)
    parts = [v for v in vectors if len(v)]
    if not parts:
        return float("nan")
    residual = np.concatenate(parts)
    return float(np.mean(np.abs(residual)))


def evaluate_combo(
    fair,
    boundary,
    strong_mod,
    model,
    device: torch.device,
    regime: str,
    target: np.ndarray,
    disp: np.ndarray,
    mask: np.ndarray,
    posterior_samples: int,
    euler_steps: int,
    batch_size: int,
    compute_dispersion: bool,
) -> dict[str, object]:
    t0 = time.time()
    samples = boundary.direct_samples(
        model,
        disp,
        mask,
        device,
        n_samples=posterior_samples,
        steps=euler_steps,
        batch_size=batch_size,
    )
    runtime_s = time.time() - t0
    pred = np.median(samples, axis=1)
    q16 = np.quantile(samples, 0.16, axis=1)
    q84 = np.quantile(samples, 0.84, axis=1)
    inside = (target >= q16) & (target <= q84)
    row: dict[str, object] = {
        "method": "DI-Weak",
        "test_set": regime,
        "n": int(len(target)),
        "posterior_samples": int(posterior_samples),
        "euler_steps": int(euler_steps),
        "vs_mae": float(np.abs(pred[:, 1, :] - target[:, 1, :]).mean()),
        "coverage_16_84_mean": float(inside.mean()),
        "coverage_vs": float(inside[:, 1, :].mean()),
        "posterior_std_mean": float(samples.std(axis=1).mean()),
        "posterior_std_vs": float(samples[:, :, 1, :].std(axis=1).mean()),
        "runtime_s": float(runtime_s),
    }
    if compute_dispersion:
        row["pred_disp_mae"] = dispersion_mae(fair, boundary, strong_mod, pred, disp, mask)
    else:
        row["pred_disp_mae"] = float("nan")
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/fair_di_weak_full_seed642026/best.pt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/sampling_sensitivity")
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--posterior-samples", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--euler-steps", type=int, nargs="+", default=[12, 24, 48])
    p.add_argument("--regimes", nargs="+", default=["in-prior", "boundary"], choices=["in-prior", "boundary", "out-of-prior"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=642026)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-dispersion-residuals", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    fair = import_from_path("fair_eval_for_sampling_sensitivity", ROOT / "scripts/eval_fair_di_comparison.py")
    boundary = import_from_path("prior_boundary_for_sampling_sensitivity", ROOT / "scripts/eval_prior_boundary_effect.py")
    strong_mod = import_from_path("strong_generator_for_sampling_sensitivity", ROOT / "utils/generate_data.py")
    model, _ = boundary.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.ckpt, device)
    eval_args = argparse.Namespace(n_test=args.n_test, n_envelope=512, seed=args.seed)
    test_sets = fair.make_test_sets(boundary, strong_mod, eval_args)
    rows: list[dict[str, object]] = []
    for regime in args.regimes:
        target, disp, mask = test_sets[regime]
        for samples in args.posterior_samples:
            for steps in args.euler_steps:
                row = evaluate_combo(
                    fair=fair,
                    boundary=boundary,
                    strong_mod=strong_mod,
                    model=model,
                    device=device,
                    regime=regime,
                    target=target,
                    disp=disp,
                    mask=mask,
                    posterior_samples=samples,
                    euler_steps=steps,
                    batch_size=args.batch_size,
                    compute_dispersion=not args.skip_dispersion_residuals,
                )
                rows.append(row)
                print(
                    f"{regime} samples={samples} steps={steps} "
                    f"vs_mae={row['vs_mae']:.3f} cov={row['coverage_16_84_mean']:.3f} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    write_csv(args.out_dir / "sampling_sensitivity.csv", rows)
    write_json(args.out_dir / "sampling_sensitivity.json", {"protocol": vars(args), "rows": rows})
    print(f"Wrote sampling sensitivity to {args.out_dir}")


if __name__ == "__main__":
    main()
