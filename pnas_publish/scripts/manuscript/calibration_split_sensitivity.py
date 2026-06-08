#!/usr/bin/env python3
"""Evaluate split-to-split stability of scalar posterior-temperature calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from make_paper_figures import (
    ROOT,
    collect_dataset,
    fit_temperature_scale,
    interval_coverage_by_example,
    load_module,
    restore_model,
)


def channel_mean(values: torch.Tensor) -> dict[str, float]:
    return {
        "Vp": float(values[:, 0].mean()),
        "Vs": float(values[:, 1].mean()),
        "rho": float(values[:, 2].mean()),
        "mean": float(values.mean()),
    }


def sample_profiles(model, disp: torch.Tensor, mask: torch.Tensor, samples: int, steps: int, batch_size: int) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, disp.size(0), batch_size):
            batch = model.sample(
                disp[start : start + batch_size],
                mask[start : start + batch_size],
                num_samples=samples,
                num_steps=steps,
            )
            outputs.append(batch["profile_samples"])
    return torch.cat(outputs, dim=0)


def summarize_split(
    samples: torch.Tensor,
    target: torch.Tensor,
    split_seed: int,
    calibration_examples: int,
    nominal_percent: float,
) -> dict:
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(target.size(0))
    calibration_index = torch.as_tensor(permutation[:calibration_examples], dtype=torch.long)
    test_index = torch.as_tensor(permutation[calibration_examples:], dtype=torch.long)

    temperature = fit_temperature_scale(samples[calibration_index], target[calibration_index], nominal_percent=nominal_percent)
    raw_test = interval_coverage_by_example(samples[test_index], target[test_index], nominal_percent, scale=1.0)
    scaled_test = interval_coverage_by_example(samples[test_index], target[test_index], nominal_percent, scale=temperature)

    return {
        "split_seed": int(split_seed),
        "calibration_examples": int(calibration_index.numel()),
        "test_examples": int(test_index.numel()),
        "temperature_scale": float(temperature),
        "raw_test_coverage_16_84": channel_mean(raw_test),
        "scaled_test_coverage_16_84": channel_mean(scaled_test),
        "coverage_gain_mean": float(scaled_test.mean() - raw_test.mean()),
    }


def summarize_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-split posterior-temperature calibration stability diagnostics.")
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--nominal-percent", type=float, default=68.0)
    parser.add_argument("--sample-seed", type=int, default=2041)
    parser.add_argument("--split-seeds", nargs="+", default=["2030", "2031", "2032", "2033", "2034"])
    parser.add_argument("--output", type=Path, default=Path("figures/calibration_split_sensitivity.json"))
    args = parser.parse_args()

    split_seeds = [int(value) for value in args.split_seeds]
    if not split_seeds:
        raise ValueError("At least one split seed is required.")
    if args.calibration_examples <= 0 or args.calibration_examples >= args.n_eval:
        raise ValueError("--calibration-examples must be between 1 and n_eval - 1.")

    torch.manual_seed(args.sample_seed)
    np.random.seed(args.sample_seed)

    module = load_module("disp_inv_train_v12_calibration_split_sensitivity", ROOT / "disp_inv_train.v1.2.py")
    model, checkpoint = restore_model(module, ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    model_batch, disp_batch, mask_batch = collect_dataset(module, n=args.n_eval)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    samples = sample_profiles(
        model,
        disp_batch,
        mask_batch,
        samples=args.posterior_samples,
        steps=args.sampling_steps,
        batch_size=args.batch_size,
    )

    splits = [
        summarize_split(samples, target, seed, args.calibration_examples, args.nominal_percent)
        for seed in split_seeds
    ]
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Archive diagnostic for split-to-split stability of scalar posterior-temperature calibration.",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "n_eval": int(args.n_eval),
        "posterior_samples": int(args.posterior_samples),
        "sampling_steps": int(args.sampling_steps),
        "batch_size": int(args.batch_size),
        "sample_seed": int(args.sample_seed),
        "nominal_percent": float(args.nominal_percent),
        "split_seeds": split_seeds,
        "splits": splits,
        "summary": {
            "temperature_scale": summarize_distribution([item["temperature_scale"] for item in splits]),
            "raw_test_coverage_16_84_mean": summarize_distribution([item["raw_test_coverage_16_84"]["mean"] for item in splits]),
            "scaled_test_coverage_16_84_mean": summarize_distribution([item["scaled_test_coverage_16_84"]["mean"] for item in splits]),
            "coverage_gain_mean": summarize_distribution([item["coverage_gain_mean"] for item in splits]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
