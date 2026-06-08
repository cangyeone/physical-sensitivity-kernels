#!/usr/bin/env python3
"""Evaluate sensitivity to posterior sample count and Euler sampling steps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
OUT = OVERLEAF / "figures"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_paper_figures import collect_dataset, load_module, restore_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sampling-configuration sensitivity check.")
    parser.add_argument("--n-eval", type=int, default=1024, help="Held-out examples used for the sensitivity check.")
    parser.add_argument("--batch-size", type=int, default=8, help="Sampling batch size.")
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--step-counts", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--baseline-samples", type=int, default=16)
    parser.add_argument("--baseline-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUT / "sampling_sensitivity.json")
    return parser.parse_args()


def sample_profiles(model, disp_batch, mask_batch, num_samples, num_steps, batch_size):
    samples = []
    with torch.no_grad():
        for i in range(0, disp_batch.size(0), batch_size):
            out = model.sample(
                disp_batch[i : i + batch_size],
                mask_batch[i : i + batch_size],
                num_samples=num_samples,
                num_steps=num_steps,
            )
            samples.append(out["profile_samples"])
    return torch.cat(samples, dim=0)


def summarize(samples, target):
    median = samples.median(dim=1).values
    err = (median - target).abs()
    q16 = torch.quantile(samples, 0.16, dim=1)
    q84 = torch.quantile(samples, 0.84, dim=1)
    coverage = ((target >= q16) & (target <= q84)).float().mean(dim=(0, 2))
    std = samples.std(dim=1, unbiased=False).mean(dim=(0, 2))
    return {
        "mae": {
            "Vp": float(err[:, 0].mean()),
            "Vs": float(err[:, 1].mean()),
            "rho": float(err[:, 2].mean()),
            "mean": float(err.mean()),
        },
        "coverage_16_84": {
            "Vp": float(coverage[0]),
            "Vs": float(coverage[1]),
            "rho": float(coverage[2]),
            "mean": float(coverage.mean()),
        },
        "mean_posterior_std": {
            "Vp": float(std[0]),
            "Vs": float(std[1]),
            "rho": float(std[2]),
        },
    }


def run_config(model, target, disp_batch, mask_batch, args, num_samples, num_steps):
    torch.manual_seed(args.seed + 1000 * int(num_samples) + int(num_steps))
    start = time.perf_counter()
    samples = sample_profiles(model, disp_batch, mask_batch, num_samples, num_steps, args.batch_size)
    elapsed = time.perf_counter() - start
    summary = summarize(samples, target)
    summary.update(
        {
            "posterior_samples": int(num_samples),
            "sampling_steps": int(num_steps),
            "elapsed_seconds": float(elapsed),
            "curves_per_second": float(target.size(0) / elapsed),
            "posterior_profiles_per_second": float(target.size(0) * num_samples / elapsed),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mod12 = load_module("disp_inv_train_v12", ROOT / "disp_inv_train.v1.2.py")
    model, ckpt = restore_model(mod12, ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    model_batch, disp_batch, mask_batch = collect_dataset(mod12, n=args.n_eval)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    seen = set()
    configs = []
    for n_samples in args.sample_counts:
        configs.append(("sample_count", n_samples, args.baseline_steps))
    for n_steps in args.step_counts:
        configs.append(("step_count", args.baseline_samples, n_steps))

    results = []
    for family, n_samples, n_steps in configs:
        key = (n_samples, n_steps)
        if key in seen:
            continue
        seen.add(key)
        item = run_config(model, target, disp_batch, mask_batch, args, n_samples, n_steps)
        item["family"] = family
        item["is_baseline"] = n_samples == args.baseline_samples and n_steps == args.baseline_steps
        results.append(item)

    baseline = next(item for item in results if item["is_baseline"])
    for item in results:
        item["delta_mean_mae_vs_baseline"] = float(item["mae"]["mean"] - baseline["mae"]["mean"])
        item["delta_mean_coverage_vs_baseline"] = float(item["coverage_16_84"]["mean"] - baseline["coverage_16_84"]["mean"])

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_global_step": int(ckpt["global_step"]),
        "n_eval": int(args.n_eval),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "baseline": {
            "posterior_samples": int(args.baseline_samples),
            "sampling_steps": int(args.baseline_steps),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
