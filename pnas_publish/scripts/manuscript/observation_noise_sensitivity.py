#!/usr/bin/env python3
"""Evaluate inversion sensitivity to synthetic dispersion-pick noise."""

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
    scaled_samples_about_median,
)


def parse_noise_levels(values: list[str]) -> list[float]:
    levels = [float(value) for value in values]
    if any(level < 0.0 for level in levels):
        raise argparse.ArgumentTypeError("Noise levels must be non-negative.")
    return levels


def add_dispersion_noise(disp: torch.Tensor, mask: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    noisy = disp.clone()
    if sigma == 0.0:
        return noisy
    generator = torch.Generator(device=noisy.device)
    generator.manual_seed(seed)
    velocity_mask = mask[:, 1:3, :] > 0.5
    perturbation = torch.randn(noisy[:, 1:3, :].shape, generator=generator, device=noisy.device) * sigma
    noisy[:, 1:3, :] = torch.where(velocity_mask, noisy[:, 1:3, :] + perturbation, noisy[:, 1:3, :])
    noisy[:, 1:3, :] = torch.clamp(noisy[:, 1:3, :], min=0.1)
    return noisy


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


def summarize_samples(samples: torch.Tensor, target: torch.Tensor) -> dict:
    median = samples.median(dim=1).values
    err = (median - target).abs()
    per_example_mae = err.mean(dim=2)
    q16 = torch.quantile(samples, 0.16, dim=1)
    q84 = torch.quantile(samples, 0.84, dim=1)
    per_example_coverage = ((target >= q16) & (target <= q84)).float().mean(dim=2)
    split = target.size(0) // 2
    temp = fit_temperature_scale(samples[:split], target[:split], nominal_percent=68.0)
    raw_test_coverage = interval_coverage_by_example(samples[split:], target[split:], 68.0, scale=1.0)
    scaled_test_samples = scaled_samples_about_median(samples[split:], temp)
    scaled_test_coverage = interval_coverage_by_example(scaled_test_samples, target[split:], 68.0, scale=1.0)
    return {
        "mae": {
            "Vp": float(per_example_mae[:, 0].mean()),
            "Vs": float(per_example_mae[:, 1].mean()),
            "rho": float(per_example_mae[:, 2].mean()),
            "mean": float(per_example_mae.mean()),
        },
        "coverage_16_84": {
            "Vp": float(per_example_coverage[:, 0].mean()),
            "Vs": float(per_example_coverage[:, 1].mean()),
            "rho": float(per_example_coverage[:, 2].mean()),
            "mean": float(per_example_coverage.mean()),
        },
        "split_temperature": {
            "calibration_examples": int(split),
            "test_examples": int(target.size(0) - split),
            "temperature_scale": float(temp),
            "raw_test_coverage_16_84_mean": float(raw_test_coverage.mean()),
            "scaled_test_coverage_16_84_mean": float(scaled_test_coverage.mean()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic observation-noise sensitivity for the inversion sampler.")
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-km-s", nargs="+", default=["0.0", "0.02", "0.05", "0.10"])
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--output", type=Path, default=Path("figures/observation_noise_sensitivity.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    noise_levels = parse_noise_levels(args.noise_km_s)

    module = load_module("disp_inv_train_v12_noise_sensitivity", ROOT / "disp_inv_train.v1.2.py")
    model, checkpoint = restore_model(module, ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    model_batch, disp_batch, mask_batch = collect_dataset(module, n=args.n_eval)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    results = []
    for level_index, sigma in enumerate(noise_levels):
        condition = add_dispersion_noise(disp_batch, mask_batch, sigma, seed=args.seed + level_index)
        torch.manual_seed(args.seed + 10_000 + level_index)
        samples = sample_profiles(
            model,
            condition,
            mask_batch,
            samples=args.posterior_samples,
            steps=args.sampling_steps,
            batch_size=args.batch_size,
        )
        summary = summarize_samples(samples, target)
        results.append(
            {
                "noise_sigma_km_s": float(sigma),
                "description": "Independent Gaussian noise added to observed Rayleigh/Love phase-velocity entries; period channel and masks are unchanged.",
                **summary,
            }
        )

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Archive diagnostic for sensitivity of the trained posterior sampler to synthetic dispersion-pick noise.",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "n_eval": int(args.n_eval),
        "posterior_samples": int(args.posterior_samples),
        "sampling_steps": int(args.sampling_steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "noise_model": "Additive zero-mean Gaussian phase-velocity noise in km/s on observed Rayleigh and Love entries only.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
