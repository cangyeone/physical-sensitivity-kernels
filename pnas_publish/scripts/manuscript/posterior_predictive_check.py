#!/usr/bin/env python3
"""Posterior-predictive dispersion checks for archived Figure 4 examples."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.generate_data import compute_phase_dispersion


WAVE_ROWS = {"Rayleigh": 1, "Love": 2}


def forward_dispersion(depth: np.ndarray, profile: np.ndarray, periods: np.ndarray) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for label, wave in (("Rayleigh", "rayleigh"), ("Love", "love")):
        result = compute_phase_dispersion(
            depth,
            profile[0],
            profile[1],
            profile[2],
            periods=periods,
            modes=(0,),
            wave=wave,
        )[0]
        predictions[label] = np.asarray(result.velocity, dtype=np.float64)
    return predictions


def summarize_wave(
    observed: np.ndarray,
    predicted_samples: np.ndarray,
    observed_mask: np.ndarray,
) -> dict[str, Any]:
    keep = observed_mask > 0.5
    n_observed = int(keep.sum())
    if n_observed == 0:
        return {
            "observed_periods": 0,
            "coverage_16_84": None,
            "median_abs_residual_km_s": None,
            "mean_abs_residual_km_s": None,
            "rmse_km_s": None,
            "mean_interval_width_km_s": None,
            "bias_km_s": None,
        }

    pred = predicted_samples[:, keep]
    obs = observed[keep]
    q16 = np.quantile(pred, 0.16, axis=0)
    q50 = np.quantile(pred, 0.50, axis=0)
    q84 = np.quantile(pred, 0.84, axis=0)
    residual = q50 - obs
    return {
        "observed_periods": n_observed,
        "coverage_16_84": float(np.mean((obs >= q16) & (obs <= q84))),
        "median_abs_residual_km_s": float(np.median(np.abs(residual))),
        "mean_abs_residual_km_s": float(np.mean(np.abs(residual))),
        "rmse_km_s": float(np.sqrt(np.mean(residual**2))),
        "mean_interval_width_km_s": float(np.mean(q84 - q16)),
        "bias_km_s": float(np.mean(residual)),
    }


def summarize_case(name: str, archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    depth = np.asarray(archive["depth_km"], dtype=np.float64)
    periods = np.asarray(archive[f"{name}_dispersion"][0], dtype=np.float64)
    observed = np.asarray(archive[f"{name}_dispersion"], dtype=np.float64)
    mask = np.asarray(archive[f"{name}_mask"], dtype=np.float64)
    samples = np.asarray(archive[f"{name}_posterior_samples"], dtype=np.float64)

    predictions_by_wave = {wave: [] for wave in WAVE_ROWS}
    failed_samples: list[int] = []
    for index, sample in enumerate(samples):
        try:
            predicted = forward_dispersion(depth, sample, periods)
        except Exception:
            failed_samples.append(index)
            continue
        for wave in WAVE_ROWS:
            predictions_by_wave[wave].append(predicted[wave])

    wave_summaries: dict[str, dict[str, Any]] = {}
    for wave, row in WAVE_ROWS.items():
        if predictions_by_wave[wave]:
            predicted_stack = np.stack(predictions_by_wave[wave], axis=0)
            wave_summaries[wave] = summarize_wave(observed[row], predicted_stack, mask[row])
        else:
            wave_summaries[wave] = {
                "observed_periods": int((mask[row] > 0.5).sum()),
                "coverage_16_84": None,
                "median_abs_residual_km_s": None,
                "mean_abs_residual_km_s": None,
                "rmse_km_s": None,
                "mean_interval_width_km_s": None,
                "bias_km_s": None,
            }

    observed_counts = [summary["observed_periods"] for summary in wave_summaries.values()]
    weighted_values: dict[str, float] = {}
    for key in ("coverage_16_84", "median_abs_residual_km_s", "mean_abs_residual_km_s", "rmse_km_s", "mean_interval_width_km_s"):
        numer = 0.0
        denom = 0
        for count, summary in zip(observed_counts, wave_summaries.values()):
            value = summary[key]
            if value is None or count == 0:
                continue
            numer += float(value) * count
            denom += count
        weighted_values[key] = float(numer / denom) if denom else None

    return {
        "posterior_samples_requested": int(samples.shape[0]),
        "posterior_samples_forward_modeled": int(samples.shape[0] - len(failed_samples)),
        "failed_sample_indices": failed_samples,
        "period_min_s": float(periods.min()),
        "period_max_s": float(periods.max()),
        "waves": wave_summaries,
        "weighted_observed_periods": int(sum(observed_counts)),
        "weighted_summary": weighted_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute posterior-predictive dispersion checks for archived examples.")
    parser.add_argument("--samples", type=Path, default=Path("figures/posterior_figure_samples.npz"))
    parser.add_argument("--output", type=Path, default=Path("figures/posterior_predictive_check.json"))
    args = parser.parse_args()

    archive = np.load(args.samples)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_samples": str(args.samples),
        "purpose": "Posterior-predictive dispersion check for the archived Figure 4 synthetic and ak135 examples.",
        "method": "Each archived posterior velocity sample is forward-modeled to fundamental-mode Rayleigh and Love phase velocity over the stored 2-60 s period grid; observed periods are selected by the stored mask.",
        "cases": {
            "synthetic": summarize_case("synthetic", archive),
            "ak135": summarize_case("ak135", archive),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    failures = [
        (case_name, case["failed_sample_indices"])
        for case_name, case in result["cases"].items()
        if case["failed_sample_indices"]
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
