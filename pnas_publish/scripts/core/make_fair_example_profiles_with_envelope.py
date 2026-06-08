#!/usr/bin/env python3
"""Regenerate fair DI example profiles with strong-prior envelope shading.

Inputs are archived production diagnostics, so this script does not retrain or
resample posterior profiles. It only estimates the empirical strong-prior
envelope and redraws the example-profile figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
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


def strong_prior_envelope(n: int, seed: int, n_depth: int, dz_km: float = 0.5) -> dict[str, np.ndarray]:
    strong_mod = import_from_path("strong_generator_for_envelope_figure", ROOT / "utils" / "generate_data.py")
    rng = np.random.default_rng(seed)
    profiles = []
    z_max_km = (n_depth - 1) * dz_km
    depth_target = np.arange(n_depth, dtype=np.float32) * dz_km
    for _ in range(n):
        depth, vs, vp, rho, _ = strong_mod.sample_global_1d_model(
            z_max_km=z_max_km,
            dz_km=dz_km,
            rng=rng,
        )
        if len(depth) != n_depth:
            vp = np.interp(depth_target, depth, vp)
            vs = np.interp(depth_target, depth, vs)
            rho = np.interp(depth_target, depth, rho)
        profiles.append(np.stack([vp, vs, rho]).astype(np.float32))
    arr = np.stack(profiles)
    return {
        "lo": np.quantile(arr, 0.01, axis=0).astype(np.float32),
        "hi": np.quantile(arr, 0.99, axis=0).astype(np.float32),
        "median": np.quantile(arr, 0.50, axis=0).astype(np.float32),
    }


def load_diag(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def draw(diagnostics: dict[str, np.ndarray], envelope: dict[str, np.ndarray], output: Path) -> None:
    tests = ["in_prior", "boundary", "out_of_prior"]
    test_labels = {"in_prior": "In-prior", "boundary": "Boundary", "out_of_prior": "Out-of-prior"}
    methods = ["DI_Strong", "DI_Weak"]
    method_labels = {"DI_Strong": "DI-Strong", "DI_Weak": "DI-Weak"}
    colors = {"DI_Strong": "#3b82c4", "DI_Weak": "#6aa84f"}
    depth = np.arange(envelope["lo"].shape[-1], dtype=np.float32) * 0.5

    fig, axes = plt.subplots(len(tests), len(methods) + 1, figsize=(8.4, 7.1), sharex=True, sharey=True)
    env_lo = envelope["lo"][1]
    env_hi = envelope["hi"][1]
    env_med = envelope["median"][1]

    for row, test in enumerate(tests):
        target = diagnostics[f"DI_Strong_{test}_target"][0, 1]
        ax = axes[row, 0]
        ax.fill_betweenx(depth, env_lo, env_hi, color="#d9d9d9", alpha=0.55, linewidth=0, label="Strong-prior 1-99%")
        ax.plot(env_med, depth, color="#777777", linewidth=0.75, linestyle=":", label="Strong-prior median")
        ax.plot(target, depth, color="#111111", linewidth=1.35, label="True")
        ax.set_ylabel(f"{test_labels[test]}\nDepth (km)", fontsize=9)
        ax.set_title("Target", fontsize=9)
        ax.invert_yaxis()

        for col, method in enumerate(methods, start=1):
            ax = axes[row, col]
            samples = diagnostics[f"{method}_{test}_samples"][0, :, 1, :]
            pred = diagnostics[f"{method}_{test}_pred"][0, 1]
            target = diagnostics[f"{method}_{test}_target"][0, 1]
            q16 = np.quantile(samples, 0.16, axis=0)
            q84 = np.quantile(samples, 0.84, axis=0)
            ax.fill_betweenx(depth, env_lo, env_hi, color="#d9d9d9", alpha=0.45, linewidth=0)
            ax.fill_betweenx(depth, q16, q84, color=colors[method], alpha=0.24, linewidth=0)
            ax.plot(target, depth, color="#111111", linewidth=0.85, alpha=0.85)
            ax.plot(pred, depth, color=colors[method], linewidth=1.35)
            ax.set_title(method_labels[method], fontsize=9)

    for ax in axes.ravel():
        ax.grid(color="#e6e6e6", linewidth=0.5)
        ax.set_xlim(0.0, 6.2)
        ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$V_S$ (km s$^{-1}$)", fontsize=9)

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="none", alpha=0.55, label="Strong-prior 1-99%"),
        Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.9, label="Strong-prior median"),
        Line2D([0], [0], color="#111111", linewidth=1.2, label="True"),
        Line2D([0], [0], color=colors["DI_Strong"], linewidth=1.35, label="DI-Strong median"),
        Line2D([0], [0], color=colors["DI_Weak"], linewidth=1.35, label="DI-Weak median"),
        Patch(facecolor=colors["DI_Strong"], edgecolor="none", alpha=0.24, label="16-84% interval"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.95, bottom=0.13, wspace=0.08, hspace=0.10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "results/fair_di_comparison/production/fair_di_diagnostics.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/fair_di_comparison/production/fair_di_example_profiles.pdf")
    parser.add_argument("--n-envelope", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=642426)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = load_diag(args.diagnostics)
    n_depth = diagnostics["DI_Strong_in_prior_target"].shape[-1]
    envelope = strong_prior_envelope(args.n_envelope, args.seed, n_depth)
    draw(diagnostics, envelope, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
