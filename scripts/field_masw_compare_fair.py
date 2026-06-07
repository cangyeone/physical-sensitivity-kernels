#!/usr/bin/env python3
"""Compare fair DNN Bayan Obo field outputs against available references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def summarize_field(rows: list[dict]) -> list[dict]:
    depths = sorted({float(r["depth_km"]) for r in rows})
    out = []
    for z in depths:
        subset = [r for r in rows if float(r["depth_km"]) == z]
        vs = np.array([float(r["vs_median_km_s"]) for r in subset])
        std = np.array([float(r["vs_std_km_s"]) for r in subset])
        n_periods = np.array([float(r["n_periods_used"]) for r in subset])
        out.append(
            {
                "depth_km": z,
                "n_subarrays": int(len(subset)),
                "vs_median_mean": float(vs.mean()),
                "vs_median_std_spatial": float(vs.std()),
                "vs_median_min": float(vs.min()),
                "vs_median_max": float(vs.max()),
                "posterior_std_mean": float(std.mean()),
                "n_periods_used_mean": float(n_periods.mean()),
            }
        )
    return out


def compare_reference(dnn_rows: list[dict], ref_rows: list[dict]) -> list[dict]:
    ref = {(int(r["subarray"]), float(r["depth_km"])): r for r in ref_rows}
    depths = sorted({float(r["depth_km"]) for r in dnn_rows})
    rows = []
    for z in depths:
        dnn_vals = []
        ref_vals = []
        diffs = []
        for row in dnn_rows:
            key = (int(row["subarray"]), float(row["depth_km"]))
            if key not in ref or float(row["depth_km"]) != z:
                continue
            dv = float(row["vs_median_km_s"])
            rv = float(ref[key]["vs_median_km_s"])
            dnn_vals.append(dv)
            ref_vals.append(rv)
            diffs.append(dv - rv)
        if not diffs:
            continue
        dnn_arr = np.asarray(dnn_vals)
        ref_arr = np.asarray(ref_vals)
        diff = np.asarray(diffs)
        corr = float(np.corrcoef(dnn_arr, ref_arr)[0, 1]) if len(diff) > 1 and dnn_arr.std() > 0 and ref_arr.std() > 0 else float("nan")
        rows.append(
            {
                "depth_km": z,
                "n_matched": int(len(diff)),
                "vs_difference_mean_km_s": float(diff.mean()),
                "vs_difference_mae_km_s": float(np.abs(diff).mean()),
                "vs_difference_rmse_km_s": float(np.sqrt(np.mean(diff**2))),
                "spatial_correlation": corr,
            }
        )
    return rows


def plot_summary(summary: list[dict], comparison: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray([r["depth_km"] for r in summary])
    vs = np.asarray([r["vs_median_mean"] for r in summary])
    std = np.asarray([r["posterior_std_mean"] for r in summary])
    fig, ax = plt.subplots(figsize=(3.4, 5.2))
    ax.plot(vs, depth, color="#3b82c4", label="field mean median")
    ax.fill_betweenx(depth, vs - std, vs + std, color="#3b82c4", alpha=0.18, label="mean posterior std")
    ax.invert_yaxis()
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.grid(color="#e5e5e5", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "field_summary_vs_depth.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "field_summary_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if comparison:
        depth = np.asarray([r["depth_km"] for r in comparison])
        mae = np.asarray([r["vs_difference_mae_km_s"] for r in comparison])
        fig, ax = plt.subplots(figsize=(3.4, 5.2))
        ax.plot(mae, depth, color="#d55e00")
        ax.invert_yaxis()
        ax.set_xlabel(r"Reference difference MAE (km s$^{-1}$)")
        ax.set_ylabel("Depth (km)")
        ax.grid(color="#e5e5e5", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "field_reference_difference.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "field_reference_difference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dnn-dir", type=Path, required=True)
    p.add_argument("--masw-dir", type=Path, default=ROOT / "Bayan_Obo_Dataset/Subarray-Based MASW")
    p.add_argument("--reference-summary", type=Path, default=ROOT / "field_masw_results_v13_p2_40/bayan_obo_masw_vs_depth_summary.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/field")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production/field")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dnn_summary = args.dnn_dir / "bayan_obo_masw_vs_depth_summary.csv"
    if not dnn_summary.exists():
        raise FileNotFoundError(f"Missing DNN field summary: {dnn_summary}")
    dnn_rows = read_rows(dnn_summary)
    summary = summarize_field(dnn_rows)
    comparison = []
    if args.reference_summary.exists() and args.reference_summary.resolve() != dnn_summary.resolve():
        comparison = compare_reference(dnn_rows, read_rows(args.reference_summary))
    write_csv(args.out_dir / "field_summary.csv", summary)
    write_csv(args.out_dir / "field_reference_comparison.csv", comparison)
    write_json(
        args.out_dir / "field_posterior_predictive.json",
        {
            "dnn_dir": args.dnn_dir,
            "masw_dir": args.masw_dir,
            "reference_summary": args.reference_summary if args.reference_summary.exists() else None,
            "interpretation": "Field comparison is a workflow diagnostic unless an independently trusted reference model is supplied.",
            "summary": summary,
            "reference_comparison": comparison,
        },
    )
    plot_summary(summary, comparison, args.fig_dir)
    print(f"Wrote field comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
