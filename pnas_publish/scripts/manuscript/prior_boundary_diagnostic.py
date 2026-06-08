#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prior-support diagnostic for the surface-wave posterior sampler.

The diagnostic separates three questions that are easy to conflate in the
manuscript:

1. What model-space envelope is implied by the strong tectonic training prior?
2. How do direct amortized inversion models behave on samples inside and outside
   that envelope?
3. Which additional checkpoints are required before claiming a strong-prior
   versus weak-prior or forward-control inversion comparison?

Run on this workstation, for example:

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  /Users/anaconda3/bin/python overleaf_inversion_paper/disp_inv_scripts/prior_boundary_diagnostic.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
FIG_DIR = OVERLEAF / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = SCRIPTS / "prior_boundary_diagnostic_outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PDF_METADATA = {
    "Creator": "prior_boundary_diagnostic.py",
    "CreationDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
}

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6E6E6E",
    "black": "#111111",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.8,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.25,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax, grid: bool = True) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.2")
    if grid:
        ax.grid(color="0.88", lw=0.55, alpha=1.0)


def panel_label(ax, label: str, text: Optional[str] = None) -> None:
    body = f"({label})" if text is None else f"({label}) {text}"
    ax.text(
        0.025,
        0.975,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        zorder=10,
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_dataset(
    dataset_module,
    n: int,
    seed: int,
    z_max_km: float = 150.0,
    z_max_num: int = 256,
    dz_km: float = 0.5,
    max_scan_factor: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ds = dataset_module.SurfaceWaveDataset(
        n_samples=max(n * max_scan_factor, n),
        z_max_km=z_max_km,
        z_max_num=z_max_num,
        dz_km=dz_km,
        seed=seed,
    )
    models, disps, masks = [], [], []
    idx = 0
    while len(models) < n and idx < len(ds):
        try:
            model, disp, mask = ds[idx]
        except Exception:
            idx += 1
            continue
        models.append(model)
        disps.append(disp)
        masks.append(mask)
        idx += 1
    if len(models) < n:
        raise RuntimeError(f"Only collected {len(models)} valid samples out of requested n={n}.")
    return torch.stack(models), torch.stack(disps), torch.stack(masks)


def restore_direct_model(module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    init_sig = inspect.signature(module.Disp2StructCRF.__init__)
    kwargs: Dict[str, Any] = {
        "H": int(len(ckpt["depth_grid"])),
        "T": 59,
        "profile_channels": 3,
        "cond_base_channels": int(cfg.get("cond_base_channels", 64)),
        "cond_dim": int(cfg.get("cond_dim", 256)),
        "flow_hidden": int(cfg.get("flow_hidden", 1024)),
        "time_dim": int(cfg.get("time_dim", 64)),
        "dropout": 0.0,
        "reference_profile": ckpt["reference_profile"],
        "profile_scale": ckpt["profile_scale"],
        "depth_grid": ckpt["depth_grid"],
        "period_minmax": tuple(float(x) for x in ckpt["period_minmax"].tolist()),
        "disp_mean": ckpt["disp_mean"],
        "disp_scale": ckpt["disp_scale"],
    }
    if "control_indices" in init_sig.parameters and "control_indices" in ckpt:
        kwargs["control_indices"] = ckpt["control_indices"]
    model = module.Disp2StructCRF(**kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def roughness(profile: torch.Tensor) -> torch.Tensor:
    if profile.size(-1) < 3:
        return torch.zeros(profile.shape[:-1], dtype=profile.dtype, device=profile.device)
    d2 = profile[..., 2:] - 2.0 * profile[..., 1:-1] + profile[..., :-2]
    return d2.abs().mean(dim=-1)


def make_envelope(prior_profiles: torch.Tensor, lo_q: float, hi_q: float) -> Dict[str, torch.Tensor]:
    return {
        "lo": torch.quantile(prior_profiles, lo_q, dim=0),
        "hi": torch.quantile(prior_profiles, hi_q, dim=0),
        "median": torch.quantile(prior_profiles, 0.50, dim=0),
        "mean": prior_profiles.mean(dim=0),
        "std": prior_profiles.std(dim=0, unbiased=False),
    }


def support_scores(profiles: torch.Tensor, envelope: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    lo = envelope["lo"].to(profiles.device)
    hi = envelope["hi"].to(profiles.device)
    span = (hi - lo).clamp_min(1.0e-6)
    below = (lo.unsqueeze(0) - profiles).clamp_min(0.0) / span.unsqueeze(0)
    above = (profiles - hi.unsqueeze(0)).clamp_min(0.0) / span.unsqueeze(0)
    violation = below + above
    outside = violation > 0.0
    return {
        "outside_fraction": outside.float().mean(dim=(1, 2)),
        "outside_fraction_vs": outside[:, 1, :].float().mean(dim=1),
        "mean_violation": violation.mean(dim=(1, 2)),
        "mean_violation_vs": violation[:, 1, :].mean(dim=1),
    }


def prior_pull_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    envelope: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    lo = envelope["lo"].to(target.device).unsqueeze(0)
    hi = envelope["hi"].to(target.device).unsqueeze(0)
    target_outside = (target < lo) | (target > hi)
    if int(target_outside.sum().item()) == 0:
        return {
            "target_outside_count": 0.0,
            "pred_inside_given_target_outside": float("nan"),
            "mean_abs_target_violation": float("nan"),
            "mean_abs_pred_violation": float("nan"),
            "boundary_pull_fraction": float("nan"),
        }

    target_violation = torch.where(target < lo, lo - target, torch.where(target > hi, target - hi, torch.zeros_like(target)))
    pred_violation = torch.where(pred < lo, lo - pred, torch.where(pred > hi, pred - hi, torch.zeros_like(pred)))
    target_abs = target_violation[target_outside].abs()
    pred_abs = pred_violation[target_outside].abs()
    pred_inside = ((pred >= lo) & (pred <= hi) & target_outside).float().sum() / target_outside.float().sum()
    ratio = pred_abs.mean() / target_abs.mean().clamp_min(1.0e-8)
    return {
        "target_outside_count": float(target_outside.float().sum().item()),
        "pred_inside_given_target_outside": float(pred_inside.item()),
        "mean_abs_target_violation": float(target_abs.mean().item()),
        "mean_abs_pred_violation": float(pred_abs.mean().item()),
        "boundary_pull_fraction": float((1.0 - ratio).clamp(min=-10.0, max=10.0).item()),
    }


def summarize_prediction(
    pred: torch.Tensor,
    samples: torch.Tensor,
    target: torch.Tensor,
    support: Dict[str, torch.Tensor],
    envelope: Dict[str, torch.Tensor],
    in_support_threshold: float,
) -> Dict[str, Any]:
    groups = {
        "all": torch.ones(target.size(0), dtype=torch.bool),
        "inside_strong_prior_envelope": support["outside_fraction"] <= in_support_threshold,
        "outside_strong_prior_envelope": support["outside_fraction"] > in_support_threshold,
    }
    out: Dict[str, Any] = {}
    abs_err = (pred - target).abs()
    sq_err = (pred - target).pow(2)
    coverage = {}
    if samples.numel() > 0 and samples.size(1) >= 2:
        qlo = torch.quantile(samples, 0.16, dim=1)
        qhi = torch.quantile(samples, 0.84, dim=1)
        coverage_tensor = ((target >= qlo) & (target <= qhi)).float()
    else:
        coverage_tensor = torch.full_like(target, float("nan"))

    for name, mask in groups.items():
        n = int(mask.sum().item())
        if n == 0:
            out[name] = {"n": 0}
            coverage[name] = {"n": 0}
            continue
        mae = abs_err[mask].mean(dim=(0, 2))
        rmse = sq_err[mask].mean(dim=(0, 2)).sqrt()
        rough = roughness(pred[mask]).mean(dim=0)
        cov = coverage_tensor[mask].mean(dim=(0, 2))
        pull = prior_pull_metrics(pred[mask], target[mask], envelope)
        out[name] = {
            "n": n,
            "mae": {
                "Vp": float(mae[0].item()),
                "Vs": float(mae[1].item()),
                "rho": float(mae[2].item()),
                "mean": float(mae.mean().item()),
            },
            "rmse": {
                "Vp": float(rmse[0].item()),
                "Vs": float(rmse[1].item()),
                "rho": float(rmse[2].item()),
                "mean": float(rmse.mean().item()),
            },
            "roughness": {
                "Vp": float(rough[0].item()),
                "Vs": float(rough[1].item()),
                "rho": float(rough[2].item()),
                "mean": float(rough.mean().item()),
            },
            "coverage_16_84": {
                "Vp": float(cov[0].item()),
                "Vs": float(cov[1].item()),
                "rho": float(cov[2].item()),
                "mean": float(cov.mean().item()),
            },
            "prior_pull": pull,
        }
    return out


@torch.no_grad()
def sample_model(
    model,
    disp: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_samples: int,
    num_steps: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    medians: List[torch.Tensor] = []
    samples: List[torch.Tensor] = []
    for start in range(0, disp.size(0), batch_size):
        stop = min(start + batch_size, disp.size(0))
        disp_b = disp[start:stop].to(device)
        mask_b = mask[start:stop].to(device)
        out = model.predict(
            disp_b,
            mask_b,
            num_samples=num_samples,
            num_steps=num_steps,
            temperature=temperature,
            reduce="median",
        )
        medians.append(out["profile_mu"].detach().cpu())
        samples.append(out["profile_samples"].detach().cpu())
    return torch.cat(medians, dim=0), torch.cat(samples, dim=0)


def serializable_support_summary(support: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        "outside_fraction_mean": float(support["outside_fraction"].mean().item()),
        "outside_fraction_median": float(support["outside_fraction"].median().item()),
        "outside_fraction_vs_mean": float(support["outside_fraction_vs"].mean().item()),
        "outside_fraction_vs_median": float(support["outside_fraction_vs"].median().item()),
        "mean_violation_mean": float(support["mean_violation"].mean().item()),
        "mean_violation_vs_mean": float(support["mean_violation_vs"].mean().item()),
    }


def plot_diagnostic(
    result: Dict[str, Any],
    envelope: Dict[str, torch.Tensor],
    prior_profiles: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    depth = result["depth_grid_km"]
    depth_arr = np.asarray(depth, dtype=float)
    lo_vs = envelope["lo"][1].numpy()
    hi_vs = envelope["hi"][1].numpy()
    med_vs = envelope["median"][1].numpy()

    fig = plt.figure(figsize=(7.1, 5.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], wspace=0.30, hspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    strong = prior_profiles["strong"][:, 1, :].numpy()
    weak = prior_profiles["weak"][:, 1, :].numpy()
    for row in strong[: min(80, strong.shape[0])]:
        ax0.plot(row, depth_arr, color="0.72", lw=0.45, alpha=0.20)
    for row in weak[: min(80, weak.shape[0])]:
        ax0.plot(row, depth_arr, color=COLORS["orange"], lw=0.45, alpha=0.12)
    ax0.fill_betweenx(depth_arr, lo_vs, hi_vs, color=COLORS["blue"], alpha=0.16, lw=0, label="strong-prior 1-99%")
    ax0.plot(med_vs, depth_arr, color=COLORS["blue"], lw=1.4, label="strong-prior median")
    ax0.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_ylim(depth_arr[-1], depth_arr[0])
    ax0.set_title("Training-prior support")
    style_axis(ax0)
    ax0.legend(loc="lower right", frameon=True, borderpad=0.35)
    panel_label(ax0, "a")

    bins = np.linspace(0, 1.0, 26)
    for family, color, label in [
        ("strong", COLORS["blue"], "strong test"),
        ("weak", COLORS["orange"], "weak test"),
    ]:
        values = np.asarray(result["test_sets"][family]["support"]["outside_fraction"], dtype=float)
        ax1.hist(values, bins=bins, histtype="step", lw=1.4, color=color, density=False, label=label)
    thr = float(result["settings"]["in_support_threshold"])
    ax1.axvline(thr, color="0.25", lw=0.9, ls="--", label=f"threshold={thr:g}")
    ax1.set_xlabel("Fraction of full-grid parameters outside strong-prior envelope")
    ax1.set_ylabel("Count")
    ax1.set_title("In-support versus out-of-support split")
    style_axis(ax1)
    ax1.legend(loc="upper right", frameon=True, borderpad=0.35)
    panel_label(ax1, "b")

    direct = result.get("direct_models", {})
    rows = []
    for model_name, model_result in direct.items():
        if model_result.get("status") != "ok":
            continue
        for family in ["strong", "weak"]:
            fam_result = model_result["test_sets"].get(family)
            if not fam_result:
                continue
            summary = fam_result["summary"]
            for group, group_label in [
                ("inside_strong_prior_envelope", "inside"),
                ("outside_strong_prior_envelope", "outside"),
            ]:
                if summary.get(group, {}).get("n", 0) == 0:
                    continue
                rows.append(
                    {
                        "model": model_result["short_label"],
                        "family": family,
                        "group": group_label,
                        "vs_mae": summary[group]["mae"]["Vs"],
                        "pull": summary[group]["prior_pull"]["boundary_pull_fraction"],
                    }
                )

    if rows:
        models = list(dict.fromkeys(r["model"] for r in rows))
        conditions = [
            ("strong", "inside", "strong in", COLORS["blue"], ""),
            ("strong", "outside", "strong out", COLORS["blue"], "//"),
            ("weak", "inside", "weak in", COLORS["orange"], ""),
            ("weak", "outside", "weak out", COLORS["orange"], "//"),
        ]
        lookup = {(r["model"], r["family"], r["group"]): r for r in rows}
        x = np.arange(len(models), dtype=float)
        width = 0.17
        offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * width
        for offset, (family, group, label, color, hatch) in zip(offsets, conditions):
            heights = [
                lookup.get((model, family, group), {}).get("vs_mae", np.nan)
                for model in models
            ]
            ax2.bar(x + offset, heights, width=width, color=color, alpha=0.86, hatch=hatch, label=label)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylabel(r"$V_S$ MAE (km s$^{-1}$)")
        ax2.set_title("Direct inversion error by support class")
        ax2.legend(loc="upper left", frameon=True, borderpad=0.30, ncol=2, columnspacing=0.8, handlelength=1.2)
    else:
        ax2.text(0.5, 0.5, "No direct-model results", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    style_axis(ax2)
    panel_label(ax2, "c")

    pull_rows = [r for r in rows if r["family"] == "weak" and r["group"] == "outside" and np.isfinite(r["pull"])]
    if pull_rows:
        x = np.arange(len(pull_rows))
        ax3.bar(x, [r["pull"] for r in pull_rows], color=COLORS["vermillion"], alpha=0.86)
        ax3.axhline(0.0, color="0.25", lw=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([r["model"] for r in pull_rows], rotation=25, ha="right")
        ax3.set_ylabel("Boundary-pull fraction")
        ax3.set_ylim(min(-0.1, min(r["pull"] for r in pull_rows) - 0.05), max(1.0, max(r["pull"] for r in pull_rows) + 0.05))
        ax3.set_title("Weak-prior out-of-support pullback")
    else:
        ax3.text(0.5, 0.5, "No weak out-of-support pull metric", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    style_axis(ax3)
    panel_label(ax3, "d")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.035, metadata=PDF_METADATA)
    plt.close(fig)


def tensor_list(values: torch.Tensor) -> List[float]:
    return [float(x) for x in values.detach().cpu().reshape(-1).tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prior-boundary behavior of direct inversion samplers.")
    parser.add_argument("--n-prior", type=int, default=512, help="Samples used to estimate the strong-prior envelope.")
    parser.add_argument("--n-eval", type=int, default=64, help="Strong and weak test samples used for inversion diagnostics.")
    parser.add_argument("--num-samples", type=int, default=8, help="Posterior samples per example.")
    parser.add_argument("--num-steps", type=int, default=16, help="Euler steps for rectified-flow sampling.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for direct-model sampling.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lo-q", type=float, default=0.01)
    parser.add_argument("--hi-q", type=float, default=0.99)
    parser.add_argument("--in-support-threshold", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run.")
    parser.add_argument("--out-json", default=str(DIAG_DIR / "prior_boundary_diagnostic.json"))
    parser.add_argument("--out-figure", default=str(DIAG_DIR / "prior_boundary_diagnostic.pdf"))
    parser.add_argument("--dense-strong-ckpt", default=str(ROOT / "ckpt/disp2struct_crf.v1.1/best.pt"))
    parser.add_argument("--control-strong-ckpt", default=str(ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt"))
    parser.add_argument("--dense-weak-ckpt", default=str(ROOT / "ckpt/disp2struct_crf.v1.1_weak/best.pt"))
    parser.add_argument("--control-weak-ckpt", default=str(ROOT / "ckpt/disp2struct_crf.v1.2_cp_weak/best.pt"))
    parser.add_argument("--forward-ckpt", default=str(ROOT / "ckpt/struct2disp_transformer.v1.1.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_prior = min(args.n_prior, 128)
        args.n_eval = min(args.n_eval, 12)
        args.num_samples = min(args.num_samples, 4)
        args.num_steps = min(args.num_steps, 8)
        args.batch_size = min(args.batch_size, 2)

    device = select_device(args.device)
    print(f"[device] {device}")

    strong_data_mod = load_module("strong_prior_data", ROOT / "utils/generate_data.py")
    weak_data_mod = load_module("weak_prior_data", ROOT / "utils/generate_data_weak_prior.py")

    dense_mod = load_module("disp_inv_v11", ROOT / "disp_inv_train.v1.1.py")
    control_mod = load_module("disp_inv_v12", ROOT / "disp_inv_train.v1.2.py")

    print(f"[collect] strong prior envelope n={args.n_prior}")
    strong_prior_models, _, _ = collect_dataset(strong_data_mod, args.n_prior, seed=args.seed + 10_000)
    strong_prior_profiles = strong_prior_models[:, 1:4, :]
    envelope = make_envelope(strong_prior_profiles, args.lo_q, args.hi_q)

    print(f"[collect] strong/weak evaluation sets n={args.n_eval}")
    strong_models, strong_disp, strong_mask = collect_dataset(strong_data_mod, args.n_eval, seed=args.seed + 20_000)
    weak_models, weak_disp, weak_mask = collect_dataset(weak_data_mod, args.n_eval, seed=args.seed + 30_000)

    test_data = {
        "strong": {
            "models": strong_models,
            "disp": strong_disp,
            "mask": strong_mask,
            "target": strong_models[:, 1:4, :],
        },
        "weak": {
            "models": weak_models,
            "disp": weak_disp,
            "mask": weak_mask,
            "target": weak_models[:, 1:4, :],
        },
    }

    support_by_family = {
        family: support_scores(data["target"], envelope)
        for family, data in test_data.items()
    }

    direct_specs = [
        {
            "name": "dense_strong_checkpoint",
            "short_label": "dense",
            "module": dense_mod,
            "ckpt": Path(args.dense_strong_ckpt),
            "training_prior": "strong",
        },
        {
            "name": "control_strong_checkpoint",
            "short_label": "control",
            "module": control_mod,
            "ckpt": Path(args.control_strong_ckpt),
            "training_prior": "strong",
        },
        {
            "name": "dense_weak_checkpoint",
            "short_label": "dense-weak",
            "module": dense_mod,
            "ckpt": Path(args.dense_weak_ckpt),
            "training_prior": "weak",
        },
        {
            "name": "control_weak_checkpoint",
            "short_label": "control-weak",
            "module": control_mod,
            "ckpt": Path(args.control_weak_ckpt),
            "training_prior": "weak",
        },
    ]

    direct_results: Dict[str, Any] = {}
    for spec in direct_specs:
        if not spec["ckpt"].exists():
            direct_results[spec["name"]] = {
                "status": "skipped",
                "reason": f"checkpoint not found: {spec['ckpt']}",
                "short_label": spec["short_label"],
                "training_prior": spec["training_prior"],
            }
            print(f"[skip] {spec['name']} missing checkpoint")
            continue

        print(f"[eval] {spec['name']} -> {spec['ckpt']}")
        model, ckpt = restore_direct_model(spec["module"], spec["ckpt"], device=device)
        family_results = {}
        for family, data in test_data.items():
            pred, samples = sample_model(
                model,
                data["disp"],
                data["mask"],
                device=device,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                temperature=args.temperature,
            )
            summary = summarize_prediction(
                pred=pred,
                samples=samples,
                target=data["target"],
                support=support_by_family[family],
                envelope=envelope,
                in_support_threshold=args.in_support_threshold,
            )
            family_results[family] = {"summary": summary}
        direct_results[spec["name"]] = {
            "status": "ok",
            "short_label": spec["short_label"],
            "training_prior": spec["training_prior"],
            "checkpoint": str(spec["ckpt"]),
            "checkpoint_config": ckpt.get("config", {}),
            "test_sets": family_results,
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "n_prior": args.n_prior,
            "n_eval": args.n_eval,
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "device": str(device),
            "temperature": args.temperature,
            "strong_prior_envelope_quantiles": [args.lo_q, args.hi_q],
            "in_support_threshold": args.in_support_threshold,
        },
        "interpretation": {
            "learned_quantity": (
                "The evaluated direct models are posterior surrogates under their training prior-predictive "
                "distribution. Existing local checkpoints are strong-prior checkpoints unless a weak-prior "
                "checkpoint path is supplied."
            ),
            "boundary_pull_fraction": (
                "Computed on target nodes outside the strong-prior envelope. Values near 1 indicate that "
                "predictions are pulled back inside the strong-prior envelope; values near 0 indicate that "
                "predicted out-of-envelope magnitude matches the target out-of-envelope magnitude."
            ),
        },
        "depth_grid_km": tensor_list(strong_prior_models[0, 0, :]),
        "test_sets": {},
        "direct_models": direct_results,
        "forward_control_point": {
            "status": "skipped",
            "reason": (
                "Forward control-point inversion requires a trained struct2disp checkpoint. "
                f"Expected path: {args.forward_ckpt}"
                if not Path(args.forward_ckpt).exists()
                else "Checkpoint exists, but this diagnostic records direct-model prior support only unless a "
                "forward-control benchmark is added explicitly."
            ),
            "checkpoint_exists": Path(args.forward_ckpt).exists(),
            "checkpoint": args.forward_ckpt,
        },
    }

    for family, data in test_data.items():
        support = support_by_family[family]
        result["test_sets"][family] = {
            "n": int(data["target"].size(0)),
            "support_summary": serializable_support_summary(support),
            "support": {
                "outside_fraction": tensor_list(support["outside_fraction"]),
                "outside_fraction_vs": tensor_list(support["outside_fraction_vs"]),
                "mean_violation": tensor_list(support["mean_violation"]),
                "mean_violation_vs": tensor_list(support["mean_violation_vs"]),
            },
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    plot_diagnostic(
        result=result,
        envelope=envelope,
        prior_profiles={
            "strong": strong_prior_profiles,
            "weak": test_data["weak"]["target"],
        },
        out_path=Path(args.out_figure),
    )
    print(f"[write] {args.out_figure}")

    print(json.dumps({k: result["test_sets"][k]["support_summary"] for k in ["strong", "weak"]}, indent=2))


if __name__ == "__main__":
    main()
