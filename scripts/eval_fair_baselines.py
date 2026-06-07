#!/usr/bin/env python3
"""Evaluate deterministic DNN and learned-forward optimization baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
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


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_det_model(path: Path, device: torch.device):
    direct_mod = import_from_path("disp_inv_train_v13_det_eval", ROOT / "disp_inv_train.v1.3.py")
    train_det = import_from_path("train_deterministic_di_fair_eval", ROOT / "scripts/train_deterministic_di_fair.py")
    ckpt = torch.load(path, map_location="cpu")
    model = train_det.DeterministicControlPointDNN(direct_mod, ckpt["normalization"], ckpt["config_yaml"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def det_predict(model, disp: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, len(disp), batch_size):
        end = min(len(disp), start + batch_size)
        out = model(torch.from_numpy(disp[start:end]).to(device), torch.from_numpy(mask[start:end]).to(device))
        preds.append(out["profile"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_det(label, model, test_sets, ev, strong_mod, envelope, device, args):
    rows = []
    diagnostics = {}
    for name, (target, disp, mask) in test_sets.items():
        tic = time.time()
        pred = det_predict(model, disp, mask, device, args.batch_size)
        row = {
            "method": label,
            "test_set": name,
            "n": int(len(target)),
            "runtime_s": float(time.time() - tic),
            "coverage_16_84_mean": float("nan"),
            "coverage_vp": float("nan"),
            "coverage_vs": float("nan"),
            "coverage_rho": float("nan"),
        }
        row.update(ev.mae_rmse(pred, target))
        row.update(ev.profile_roughness(pred))
        row.update(ev.dispersion_residuals(strong_mod, pred, disp, mask))
        row.update(ev.prior_pull_metrics(pred, target, envelope))
        rows.append(row)
        diagnostics[f"{label}_{name}"] = {"target": target, "pred": pred}
    return rows, diagnostics


def plot_baseline_summary(rows: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tests = ["in-prior", "boundary", "out-of-prior"]
    methods = sorted({r["method"] for r in rows})
    x = np.arange(len(tests))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for j, method in enumerate(methods):
        vals = []
        for test in tests:
            match = [r for r in rows if r["method"] == method and r["test_set"] == test]
            vals.append(float(match[0]["vs_mae"]) if match else np.nan)
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(["In-prior", "Boundary", "Out-of-prior"])
    ax.set_ylabel("$V_S$ MAE (km/s)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_metric_summary.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "baseline_metric_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fair-results", type=Path, default=None)
    p.add_argument("--det-strong", type=Path, default=ROOT / "ckpt/det_di_strong_full_seed642026/best.pt")
    p.add_argument("--det-weak", type=Path, default=ROOT / "ckpt/det_di_weak_full_seed642026/best.pt")
    p.add_argument("--ind-fwd", type=Path, default=ROOT / "ckpt/struct2disp_cpmlp.prior_boundary_v3.pt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/fair_di_comparison/production/baselines")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production/baselines")
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--n-envelope", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--posterior-samples", type=int, default=64)
    p.add_argument("--euler-steps", type=int, default=24)
    p.add_argument("--n-forward-eval", type=int, default=128)
    p.add_argument("--forward-inv-steps", type=int, default=220)
    p.add_argument("--forward-inv-lr", type=float, default=0.04)
    p.add_argument("--indirect-multistarts", type=int, default=6)
    p.add_argument("--seed", type=int, default=642026)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    ev = import_from_path("prior_boundary_for_baselines", ROOT / "scripts/eval_prior_boundary_effect.py")
    fair = import_from_path("fair_eval_for_baselines", ROOT / "scripts/eval_fair_di_comparison.py")
    strong_mod = import_from_path("strong_generator_for_baselines", ROOT / "utils/generate_data.py")
    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    in_full, in_disp, in_mask = ev.dataset_to_arrays(ev.strong_dataset(strong_mod, args.n_test, args.seed + 20))
    test_sets = {
        "in-prior": (in_full[:, 1:4, :], in_disp, in_mask),
        "boundary": ev.parametric_dataset(strong_mod, "boundary", args.n_test, args.seed + 30, in_full.shape[-1], periods),
        "out-of-prior": ev.parametric_dataset(strong_mod, "out-of-prior", args.n_test, args.seed + 40, in_full.shape[-1], periods),
    }
    envelope = fair.prior_envelope_fast(strong_mod, args.n_envelope, args.seed + 10, in_full[:, 1:4, :].shape[-1])
    rows = []
    diagnostics = {}
    for label, ckpt in (("DET-Strong", args.det_strong), ("DET-Weak", args.det_weak)):
        if ckpt.exists():
            r, d = evaluate_det(label, load_det_model(ckpt, device), test_sets, ev, strong_mod, envelope, device, args)
            rows.extend(r)
            diagnostics.update(d)
    if args.ind_fwd.exists():
        fwd = ev.load_forward_model(args.ind_fwd, device)
        ind_args = SimpleNamespace(
            n_forward_eval=args.n_forward_eval,
            indirect_multistarts=args.indirect_multistarts,
            forward_inv_steps=args.forward_inv_steps,
            forward_inv_lr=args.forward_inv_lr,
            seed=args.seed,
        )
        r, _ = ev.evaluate_forward_iterative(fwd, test_sets, strong_mod, envelope, device, ind_args)
        rows.extend(r)
    write_csv(args.out_dir / "baseline_metrics.csv", rows)
    write_json(args.out_dir / "baseline_metrics.json", {"protocol": vars(args), "rows": rows})
    if rows:
        plot_baseline_summary(rows, args.fig_dir)
    print(f"Wrote baseline metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
