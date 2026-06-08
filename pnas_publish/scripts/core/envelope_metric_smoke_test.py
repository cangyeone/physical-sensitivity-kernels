#!/usr/bin/env python3
"""Smoke tests for prior-envelope pull-in metrics.

The manuscript reports conditional pull-in:

    P(prediction inside strong-prior envelope | target outside envelope)

This script checks the edge cases that can otherwise make the metric easy to
misread. It is intentionally lightweight and does not depend on model
checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvelopeMetrics:
    target_outside_fraction: float
    pred_inside_given_target_outside: float
    target_outside_count: int


def envelope_metrics(pred: np.ndarray, target: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> EnvelopeMetrics:
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    pred = np.asarray(pred)
    target = np.asarray(target)
    target_outside = (target < lo) | (target > hi)
    pred_inside = (pred >= lo) & (pred <= hi)
    count = int(target_outside.sum())
    # Match the production metric: if no target nodes are outside the envelope,
    # the conditional numerator is zero and the protected denominator is one.
    conditional = float((pred_inside & target_outside).sum() / max(float(count), 1.0))
    return EnvelopeMetrics(
        target_outside_fraction=float(target_outside.mean()),
        pred_inside_given_target_outside=conditional,
        target_outside_count=count,
    )


def assert_close(value: float, expected: float, label: str) -> None:
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: got {value}, expected {expected}")


def main() -> None:
    lo = np.zeros((1, 4), dtype=float)
    hi = np.ones((1, 4), dtype=float)

    cases = [
        (
            "all target nodes inside envelope",
            np.array([[0.2, 0.3, 0.4, 0.5]]),
            np.array([[0.2, 0.3, 0.4, 0.5]]),
            0.0,
            0.0,
            0,
        ),
        (
            "all target nodes outside, predictions inside",
            np.array([[0.5, 0.5, 0.5, 0.5]]),
            np.array([[-1.0, 2.0, -0.5, 1.5]]),
            1.0,
            1.0,
            4,
        ),
        (
            "all target nodes outside, predictions outside",
            np.array([[-1.2, 2.2, -0.2, 1.2]]),
            np.array([[-1.0, 2.0, -0.5, 1.5]]),
            1.0,
            0.0,
            4,
        ),
        (
            "mixed target support and mixed predictions",
            np.array([[0.5, 1.2, 0.5, -0.1]]),
            np.array([[0.5, 2.0, -0.5, 0.5]]),
            0.5,
            0.5,
            2,
        ),
    ]

    for label, pred, target, outside_fraction, conditional_pull, outside_count in cases:
        metrics = envelope_metrics(pred, target, lo, hi)
        assert_close(metrics.target_outside_fraction, outside_fraction, f"{label} outside fraction")
        assert_close(metrics.pred_inside_given_target_outside, conditional_pull, f"{label} conditional pull-in")
        if metrics.target_outside_count != outside_count:
            raise AssertionError(f"{label} outside count: got {metrics.target_outside_count}, expected {outside_count}")

    print("envelope metric smoke tests passed")


if __name__ == "__main__":
    main()

