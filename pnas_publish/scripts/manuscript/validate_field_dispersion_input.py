#!/usr/bin/env python3
"""Validate and tensorize a field dispersion CSV for the inversion sampler."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "wave",
    "mode",
    "period_s",
    "phase_velocity_km_s",
    "uncertainty_km_s",
    "source",
    "notes",
]

WAVE_ALIASES = {
    "rayleigh": "Rayleigh",
    "r": "Rayleigh",
    "love": "Love",
    "l": "Love",
}

PERIOD_MIN = 2.0
PERIOD_MAX = 60.0
PERIOD_STEP = 1.0
MIN_TOTAL_OBSERVATIONS = 5
MIN_RECOMMENDED_PER_WAVE = 5


class Collector:
    def __init__(self, template_mode: bool) -> None:
        self.template_mode = template_mode
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add(self, message: str) -> None:
        if self.template_mode:
            self.warnings.append(message)
        else:
            self.errors.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def parse_float(value: str, path: str, collector: Collector) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        collector.add(f"{path} must be numeric.")
        return None
    if not math.isfinite(parsed):
        collector.add(f"{path} must be finite.")
        return None
    return parsed


def normalize_wave(value: str, path: str, collector: Collector) -> str | None:
    wave = WAVE_ALIASES.get(value.strip().casefold()) if isinstance(value, str) else None
    if wave is None:
        collector.add(f"{path} must be Rayleigh or Love.")
    return wave


def parse_mode(value: str, path: str, collector: Collector) -> int | None:
    text = value.strip().casefold() if isinstance(value, str) else ""
    if text in {"fundamental", "fundamental-mode", "f0"}:
        return 0
    try:
        mode = int(text)
    except ValueError:
        collector.add(f"{path} must be 0/fundamental for the trained model.")
        return None
    if mode != 0:
        collector.add(f"{path} is mode {mode}; the trained model expects fundamental mode 0.")
    return mode


def period_grid() -> list[float]:
    count = int(round((PERIOD_MAX - PERIOD_MIN) / PERIOD_STEP)) + 1
    return [PERIOD_MIN + PERIOD_STEP * i for i in range(count)]


def read_rows(path: Path, collector: Collector) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except FileNotFoundError:
        collector.error(f"Missing {path}.")
        return [], []
    except csv.Error as exc:
        collector.error(f"Could not parse {path}: {exc}")
        return [], []
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        collector.error("Missing required columns: " + ", ".join(missing))
    return rows, columns


def validate_rows(rows: list[dict[str, str]], collector: Collector, tolerance: float) -> dict[str, Any]:
    grid = period_grid()
    counts = {"Rayleigh": 0, "Love": 0}
    periods_by_wave = {"Rayleigh": [], "Love": []}
    tensor_records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    if not rows:
        collector.error("CSV must contain at least one dispersion row.")

    for row_index, row in enumerate(rows, start=2):
        path = f"row {row_index}"
        wave = normalize_wave(row.get("wave", ""), f"{path}.wave", collector)
        mode = parse_mode(row.get("mode", ""), f"{path}.mode", collector)
        period = parse_float(row.get("period_s", ""), f"{path}.period_s", collector)
        velocity = parse_float(row.get("phase_velocity_km_s", ""), f"{path}.phase_velocity_km_s", collector)
        uncertainty = parse_float(row.get("uncertainty_km_s", ""), f"{path}.uncertainty_km_s", collector)
        source = (row.get("source") or "").strip()

        if not source:
            collector.add(f"{path}.source must identify the data source or processing chain.")
        if period is None or velocity is None or uncertainty is None or wave is None or mode is None:
            continue

        if period < PERIOD_MIN or period > PERIOD_MAX:
            collector.add(f"{path}.period_s={period:g} is outside the trained {PERIOD_MIN:g}-{PERIOD_MAX:g} s band.")
        nearest_index = min(range(len(grid)), key=lambda idx: abs(grid[idx] - period))
        nearest_period = grid[nearest_index]
        if abs(nearest_period - period) > tolerance:
            collector.add(
                f"{path}.period_s={period:g} is not within {tolerance:g} s of the model grid; nearest is {nearest_period:g} s."
            )
        if velocity <= 0.0:
            collector.add(f"{path}.phase_velocity_km_s must be positive.")
        elif velocity < 1.0 or velocity > 7.0:
            collector.warn(f"{path}.phase_velocity_km_s={velocity:g} is outside the broad 1-7 km/s plausibility range.")
        if uncertainty <= 0.0:
            collector.add(f"{path}.uncertainty_km_s must be positive.")
        elif uncertainty > 1.0:
            collector.warn(f"{path}.uncertainty_km_s={uncertainty:g} is unusually large; verify units.")

        key = (wave, nearest_index)
        if key in seen:
            collector.add(f"{path} duplicates {wave} at model-grid period {nearest_period:g} s.")
        seen.add(key)
        counts[wave] += 1
        periods_by_wave[wave].append(period)
        tensor_records.append(
            {
                "wave": wave,
                "mode": mode,
                "period_s": period,
                "grid_period_s": nearest_period,
                "grid_index": nearest_index,
                "phase_velocity_km_s": velocity,
                "uncertainty_km_s": uncertainty,
                "source": source,
                "notes": (row.get("notes") or "").strip(),
            }
        )

    total = sum(counts.values())
    if total < MIN_TOTAL_OBSERVATIONS:
        collector.add(f"At least {MIN_TOTAL_OBSERVATIONS} total observations are required; found {total}.")
    for wave, count in counts.items():
        if 0 < count < MIN_RECOMMENDED_PER_WAVE:
            collector.warn(f"{wave} has {count} observations; {MIN_RECOMMENDED_PER_WAVE}+ is recommended.")
    if counts["Rayleigh"] == 0 and counts["Love"] == 0:
        collector.error("At least one of Rayleigh or Love must be present.")
    if counts["Rayleigh"] == 0 or counts["Love"] == 0:
        collector.warn("Only one wave type is present; this is allowed but should be justified in the manuscript.")

    return {
        "period_grid_s": grid,
        "counts": counts,
        "periods_by_wave": periods_by_wave,
        "tensor_records": tensor_records,
        "period_tolerance_s": tolerance,
    }


def write_npz(summary: dict[str, Any], output: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for --output-npz.") from exc

    grid = np.asarray(summary["period_grid_s"], dtype=np.float32)
    disp = np.zeros((3, grid.size), dtype=np.float32)
    mask = np.zeros((3, grid.size), dtype=np.float32)
    uncertainty = np.zeros((3, grid.size), dtype=np.float32)
    disp[0, :] = grid
    mask[0, :] = 1.0
    wave_to_channel = {"Rayleigh": 1, "Love": 2}
    for item in summary["tensor_records"]:
        channel = wave_to_channel[item["wave"]]
        index = int(item["grid_index"])
        disp[channel, index] = float(item["phase_velocity_km_s"])
        mask[channel, index] = 1.0
        uncertainty[channel, index] = float(item["uncertainty_km_s"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        period_s=grid,
        dispersion=disp,
        mask=mask,
        uncertainty_km_s=uncertainty,
        channel_names=np.asarray(["period", "Rayleigh", "Love"]),
        channel_units=np.asarray(["s", "km/s", "km/s"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a field dispersion CSV and optionally write model-input tensors.")
    parser.add_argument("path", nargs="?", help="CSV file to validate.")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Validate the committed template shape while treating data limitations as warnings.",
    )
    parser.add_argument("--period-tolerance", type=float, default=0.05, help="Allowed offset from the 2-60 s integer model grid.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--output-npz", type=Path, help="Optional NPZ output with dispersion, mask, and uncertainty tensors.")
    args = parser.parse_args()

    collector = Collector(template_mode=args.template)
    path = Path(args.path or ("field_dispersion_input.template.csv" if args.template else "field_dispersion_input.csv"))
    rows, columns = read_rows(path, collector)
    summary = validate_rows(rows, collector, tolerance=args.period_tolerance)
    result = {
        "ok": not collector.errors,
        "mode": "template" if args.template else "final",
        "path": str(path),
        "required_columns": REQUIRED_COLUMNS,
        "columns": columns,
        "period_band_s": [PERIOD_MIN, PERIOD_MAX],
        "period_step_s": PERIOD_STEP,
        "counts": summary["counts"],
        "period_tolerance_s": args.period_tolerance,
        "errors": collector.errors,
        "warnings": collector.warnings,
    }

    if result["ok"] and args.output_npz:
        try:
            write_npz(summary, args.output_npz)
        except RuntimeError as exc:
            result["ok"] = False
            result["errors"].append(str(exc))
        else:
            result["output_npz"] = str(args.output_npz)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
