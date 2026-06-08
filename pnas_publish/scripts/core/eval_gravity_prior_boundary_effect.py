#!/usr/bin/env python3
"""Prior-boundary diagnostic for learned 1-D gravity inversion.

This is a compact companion experiment to the surface-wave prior-boundary
diagnostic.  It uses the existing strong and weak synthetic Earth-model priors
but changes the observation operator to a simple gravity forward model.

The target model is density rho(z).  Direct inversion maps gravity anomalies to
density profiles.  Indirect inversion trains a neural forward surrogate from
density control points to gravity anomalies and then optimizes control points.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_control_indices(n_depth: int) -> List[int]:
    fine = list(range(0, min(20, n_depth), 2))
    mid = list(range(20, min(100, n_depth), 10))
    deep = list(range(100, n_depth, 32))
    return sorted(set(fine + mid + deep + [n_depth - 1]))


def interp_controls_np(depth: np.ndarray, cp_depth: np.ndarray, cp_values: np.ndarray) -> np.ndarray:
    out = []
    for values in cp_values:
        out.append(np.interp(depth, cp_depth, values))
    return np.stack(out, axis=0).astype(np.float32)


def interp_controls_torch(depth: torch.Tensor, cp_depth: torch.Tensor, cp_values: torch.Tensor) -> torch.Tensor:
    right = torch.searchsorted(cp_depth, depth).clamp(1, len(cp_depth) - 1)
    left = right - 1
    dl = cp_depth[left]
    dr = cp_depth[right]
    w = (depth - dl) / torch.clamp(dr - dl, min=1e-6)
    return (1.0 - w)[None, :] * cp_values[:, left] + w[None, :] * cp_values[:, right]


def gravity_kernel(depth_km: np.ndarray, offsets_km: np.ndarray, radius_km: float = 12.0, dz_km: float = 0.5) -> np.ndarray:
    """Point-mass approximation to vertical gravity from stacked cylindrical cells.

    Returns kernel K so that g_mgal = K @ delta_rho_g_cm3.
    """
    g_si = 6.67430e-11
    z_m = (depth_km + 0.5 * dz_km + 0.25).astype(np.float64) * 1000.0
    r_m = offsets_km.astype(np.float64)[:, None] * 1000.0
    area_m2 = math.pi * (radius_km * 1000.0) ** 2
    dz_m = dz_km * 1000.0
    volume_m3 = area_m2 * dz_m
    # density is supplied in g/cm^3, hence *1000 kg/m^3.
    kernel = g_si * 1000.0 * volume_m3 * z_m[None, :] / np.maximum(r_m**2 + z_m[None, :] ** 2, 1.0) ** 1.5
    return (kernel * 1e5).astype(np.float32)  # mGal


def smooth_profile(x: np.ndarray, passes: int = 2) -> np.ndarray:
    y = x.astype(np.float64).copy()
    for _ in range(passes):
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    return y.astype(np.float32)


def sample_strong_rho(strong_mod, n: int, seed: int, n_depth: int, dz: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    depth = None
    rho = []
    for _ in range(n):
        d, _, _, r, _ = strong_mod.sample_global_1d_model(z_max_km=150.0, dz_km=dz, rng=rng)
        depth = d[:n_depth].astype(np.float32)
        rho.append(r[:n_depth].astype(np.float32))
    return depth, np.stack(rho)


def sample_weak_rho(weak_mod, n: int, seed: int, n_depth: int, dz: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    depth = None
    rho = []
    for _ in range(n):
        d, _, _, r, _ = weak_mod.sample_weak_prior_1d_model(z_max_km=150.0, dz_km=dz, rng=rng)
        depth = d[:n_depth].astype(np.float32)
        rho.append(r[:n_depth].astype(np.float32))
    return depth, np.stack(rho)


def make_parametric_rho(kind: str, n: int, seed: int, depth: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    zmax = float(depth[-1])
    for _ in range(n):
        if kind == "boundary":
            sed = rng.choice([rng.uniform(0.0, 1.0), rng.uniform(10.0, 14.0)])
            moho = rng.choice([rng.uniform(8.0, 14.0), rng.uniform(62.0, 76.0)])
            mantle = rng.choice([rng.uniform(3.08, 3.16), rng.uniform(3.36, 3.48)])
            crust = rng.choice([rng.uniform(2.50, 2.62), rng.uniform(2.92, 3.02)])
            anomaly_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.12)
        elif kind == "out-of-prior":
            sed = rng.uniform(15.0, 26.0)
            moho = rng.choice([rng.uniform(4.0, 8.0), rng.uniform(82.0, 100.0)])
            mantle = rng.choice([rng.uniform(2.95, 3.08), rng.uniform(3.50, 3.65)])
            crust = rng.choice([rng.uniform(2.30, 2.48), rng.uniform(3.03, 3.18)])
            anomaly_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.16, 0.28)
        else:
            raise ValueError(kind)
        sed = min(float(sed), max(0.0, float(moho) - 2.0))
        moho = min(float(moho), zmax - 5.0)
        mid = max(sed + 1.0, 0.55 * moho)
        sed_rho = rng.uniform(1.75, 2.25) if kind == "out-of-prior" else rng.uniform(2.05, 2.35)
        lower_crust = crust + rng.uniform(0.05, 0.20)
        knots = np.array([0.0, sed, mid, moho, zmax], dtype=np.float32)
        vals = np.array([sed_rho, crust, lower_crust, mantle, mantle + rng.uniform(-0.03, 0.05)], dtype=np.float32)
        rho = np.interp(depth, knots, vals).astype(np.float32)
        center = rng.uniform(8.0, min(zmax - 5.0, 105.0))
        width = rng.uniform(5.0, 20.0) if kind == "out-of-prior" else rng.uniform(8.0, 28.0)
        rho += anomaly_amp * np.exp(-0.5 * ((depth - center) / width) ** 2).astype(np.float32)
        rho += rng.normal(0.0, 0.012 if kind == "boundary" else 0.018, size=depth.shape).astype(np.float32)
        out.append(smooth_profile(np.clip(rho, 1.6, 3.75), passes=3))
    return np.stack(out).astype(np.float32)


def gravity_obs(rho: np.ndarray, kernel: np.ndarray, ref_rho: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    g = (rho - ref_rho[None, :]) @ kernel.T
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        g = g + rng.normal(0.0, noise_std, size=g.shape).astype(np.float32)
    return g.astype(np.float32)


def prior_envelope(rho: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "lo": np.quantile(rho, 0.01, axis=0).astype(np.float32),
        "hi": np.quantile(rho, 0.99, axis=0).astype(np.float32),
        "mean": rho.mean(axis=0).astype(np.float32),
    }


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256, layers: int = 4, dropout: float = 0.05):
        super().__init__()
        blocks: List[nn.Module] = []
        dim = n_in
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden
        blocks.append(nn.Linear(dim, n_out))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_direct_ensemble(
    train_g: np.ndarray,
    train_cp: np.ndarray,
    seeds: Iterable[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> Tuple[List[MLP], Dict[str, torch.Tensor]]:
    x_mean = torch.from_numpy(train_g.mean(axis=0, keepdims=True).astype(np.float32))
    x_std = torch.from_numpy((train_g.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    y_mean = torch.from_numpy(train_cp.mean(axis=0, keepdims=True).astype(np.float32))
    y_std = torch.from_numpy((train_cp.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    x = ((torch.from_numpy(train_g) - x_mean) / x_std).float()
    y = ((torch.from_numpy(train_cp) - y_mean) / y_std).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    models: List[MLP] = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = MLP(train_g.shape[1], train_cp.shape[1], hidden=256, layers=4, dropout=0.04).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=8e-5)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        model.eval()
        models.append(model)
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return models, stats


@torch.no_grad()
def predict_direct(models: List[MLP], stats: Dict[str, torch.Tensor], g: np.ndarray, device: torch.device) -> np.ndarray:
    x = ((torch.from_numpy(g) - stats["x_mean"]) / stats["x_std"]).float().to(device)
    samples = []
    for model in models:
        y = model(x).cpu() * stats["y_std"] + stats["y_mean"]
        samples.append(y.numpy())
    return np.stack(samples, axis=1).astype(np.float32)


def train_forward_surrogate(cp: np.ndarray, g: np.ndarray, device: torch.device, epochs: int, batch_size: int):
    x_mean = torch.from_numpy(cp.mean(axis=0, keepdims=True).astype(np.float32))
    x_std = torch.from_numpy((cp.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    y_mean = torch.from_numpy(g.mean(axis=0, keepdims=True).astype(np.float32))
    y_std = torch.from_numpy((g.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    x = ((torch.from_numpy(cp) - x_mean) / x_std).float()
    y = ((torch.from_numpy(g) - y_mean) / y_std).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = MLP(cp.shape[1], g.shape[1], hidden=256, layers=4, dropout=0.03).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=8e-5)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model, {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def optimize_indirect_one(
    fwd_model: MLP,
    fwd_stats: Dict[str, torch.Tensor],
    obs_g: np.ndarray,
    init_cps: List[np.ndarray],
    device: torch.device,
    steps: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y = torch.from_numpy(obs_g[None, :]).float().to(device)
    y_mean = fwd_stats["y_mean"].to(device)
    y_std = fwd_stats["y_std"].to(device)
    x_mean = fwd_stats["x_mean"].to(device)
    x_std = fwd_stats["x_std"].to(device)
    lo = torch.tensor(1.55, dtype=torch.float32, device=device)
    hi = torch.tensor(3.80, dtype=torch.float32, device=device)
    preds = []
    losses = []
    for init in init_cps:
        init_t = torch.from_numpy(init).float().to(device).clamp(float(lo) + 1e-3, float(hi) - 1e-3)
        p = ((init_t - lo) / (hi - lo)).clamp(1e-4, 1.0 - 1e-4)
        raw = torch.nn.Parameter(torch.log(p) - torch.log1p(-p))
        opt = torch.optim.Adam([raw], lr=lr)
        best_loss = float("inf")
        best_cp = None
        for _ in range(steps):
            cp = lo + (hi - lo) * torch.sigmoid(raw)
            pred_n = fwd_model(((cp[None, :] - x_mean) / torch.clamp(x_std, min=1e-6)))
            pred = pred_n * y_std + y_mean
            data_loss = F.mse_loss(pred, y)
            smooth_loss = torch.mean((cp[2:] - 2.0 * cp[1:-1] + cp[:-2]) ** 2)
            loss = data_loss + 2e-2 * smooth_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if float(data_loss.detach().cpu()) < best_loss:
                best_loss = float(data_loss.detach().cpu())
                best_cp = cp.detach().cpu().numpy().astype(np.float32)
        preds.append(best_cp)
        losses.append(best_loss)
    order = np.argsort(losses)
    return np.stack(preds), np.stack(preds)[order[0]]


def mae_rmse(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    return {
        "rho_mae": float(np.mean(np.abs(err))),
        "rho_rmse": float(np.sqrt(np.mean(err**2))),
    }


def coverage(samples: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    q16 = np.quantile(samples, 0.16, axis=1)
    q84 = np.quantile(samples, 0.84, axis=1)
    inside = (target >= q16) & (target <= q84)
    return {"coverage_16_84": float(inside.mean())}


def pull_metrics(pred: np.ndarray, target: np.ndarray, envelope: Dict[str, np.ndarray]) -> Dict[str, float]:
    lo = envelope["lo"][None, :]
    hi = envelope["hi"][None, :]
    outside = (target < lo) | (target > hi)
    inside = (pred >= lo) & (pred <= hi)
    count = float(outside.sum())
    return {
        "target_outside_fraction": float(outside.mean()),
        "pred_inside_given_target_outside": float((inside & outside).sum() / max(count, 1.0)),
    }


def roughness(profile: np.ndarray) -> Dict[str, float]:
    d2 = profile[:, 2:] - 2.0 * profile[:, 1:-1] + profile[:, :-2]
    return {"rho_roughness": float(np.mean(np.abs(d2)))}


def gravity_residual(pred: np.ndarray, target_g: np.ndarray, kernel: np.ndarray, ref_rho: np.ndarray) -> Dict[str, float]:
    pred_g = gravity_obs(pred, kernel, ref_rho, noise_std=0.0, seed=0)
    res = pred_g - target_g
    return {
        "gravity_mae_mgal": float(np.mean(np.abs(res))),
        "gravity_rmse_mgal": float(np.sqrt(np.mean(res**2))),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_examples(
    path: Path,
    diagnostics: Dict[str, Dict[str, np.ndarray]],
    depth: np.ndarray,
    offsets: np.ndarray,
    kernel: np.ndarray,
    ref_rho: np.ndarray,
) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.0),
        gridspec_kw={"width_ratios": [0.95, 1.05], "wspace": 0.34, "hspace": 0.34},
    )
    colors = {"DI-Strong": "#b2182b", "DI-Weak": "#2166ac", "IND-FWD": "#1b7837"}
    labels = ["a", "b", "c", "d", "e", "f"]
    for row, set_name in enumerate(["in-prior", "boundary", "out-of-prior"]):
        ax = axes[row, 0]
        target = diagnostics["target"][set_name][0]
        ax.plot(target, depth, "k-", lw=1.5, label="true")
        for method in ["DI-Strong", "DI-Weak", "IND-FWD"]:
            pred = diagnostics[method][set_name][0]
            ax.plot(pred, depth, lw=1.2, ls="--", color=colors[method], label=method)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.set_title(f"({labels[2 * row]}) {set_name}: density model", loc="left", fontsize=8)
        ax.set_ylabel("Depth (km)")
        ax.set_xlim(1.55, 3.75)

        axg = axes[row, 1]
        true_g = gravity_obs(target[None, :], kernel, ref_rho, 0.0, 0)[0]
        axg.plot(offsets, true_g, "k-", lw=1.4, label="true")
        for method in ["DI-Strong", "DI-Weak", "IND-FWD"]:
            pred = diagnostics[method][set_name][0]
            pred_g = gravity_obs(pred[None, :], kernel, ref_rho, 0.0, 0)[0]
            axg.plot(offsets, pred_g, lw=1.2, ls="--", color=colors[method], label=method)
        axg.grid(True, alpha=0.25)
        axg.set_title(f"({labels[2 * row + 1]}) gravity response", loc="left", fontsize=8)
        axg.set_ylabel("Gravity (mGal)")
    axes[-1, 0].set_xlabel(r"Density (g cm$^{-3}$)")
    axes[-1, 1].set_xlabel("Offset (km)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metrics(path: Path, rows: List[Dict]) -> None:
    methods = ["DI-Strong", "DI-Weak", "IND-FWD"]
    sets = ["in-prior", "boundary", "out-of-prior"]
    colors = {"DI-Strong": "#b2182b", "DI-Weak": "#2166ac", "IND-FWD": "#1b7837"}
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes.ravel()
    metrics = [
        ("rho_mae", r"Density MAE (g cm$^{-3}$)", "(a) Model error"),
        ("gravity_mae_mgal", "Gravity MAE (mGal)", "(b) Data residual"),
        ("coverage_16_84", "16--84% coverage", "(c) Ensemble coverage"),
        ("pred_inside_given_target_outside", "Pull-in fraction", "(d) Prior-envelope pull-in"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        x = np.arange(len(sets))
        width = 0.24
        for i, method in enumerate(methods):
            vals = []
            for set_name in sets:
                match = [r for r in rows if r["method"] == method and r["test_set"] == set_name]
                vals.append(float(match[0].get(metric, math.nan)) if match else math.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, color=colors[method], edgecolor="0.25", linewidth=0.25, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(["In", "Boundary", "Out"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=96)
    p.add_argument("--n-envelope", type=int, default=1024)
    p.add_argument("--n-indirect", type=int, default=30)
    p.add_argument("--direct-ensemble", type=int, default=6)
    p.add_argument("--direct-epochs", type=int, default=120)
    p.add_argument("--forward-epochs", type=int, default=160)
    p.add_argument("--indirect-steps", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=6060)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = min(args.n_train, 512)
        args.n_test = min(args.n_test, 24)
        args.n_envelope = min(args.n_envelope, 256)
        args.n_indirect = min(args.n_indirect, 8)
        args.direct_ensemble = min(args.direct_ensemble, 3)
        args.direct_epochs = min(args.direct_epochs, 20)
        args.forward_epochs = min(args.forward_epochs, 30)
        args.indirect_steps = min(args.indirect_steps, 40)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(1)
    device = choose_device(args.device)
    print(f"[info] device={device}")

    strong_mod = import_from_path("gravity_generate_strong", ROOT / "utils" / "generate_data.py")
    weak_mod = import_from_path("gravity_generate_weak", ROOT / "utils" / "generate_data_weak_prior.py")
    n_depth = 256
    dz = 0.5
    offsets = np.linspace(2.0, 80.0, 17, dtype=np.float32)

    depth, env_rho = sample_strong_rho(strong_mod, args.n_envelope, args.seed + 1, n_depth, dz)
    envelope = prior_envelope(env_rho)
    ref_rho = envelope["mean"]
    kernel = gravity_kernel(depth, offsets, radius_km=3.0, dz_km=dz)
    cp_idx = select_control_indices(n_depth)
    cp_depth = depth[cp_idx]

    _, strong_train = sample_strong_rho(strong_mod, args.n_train, args.seed + 10, n_depth, dz)
    _, weak_train = sample_weak_rho(weak_mod, args.n_train, args.seed + 20, n_depth, dz)
    strong_g = gravity_obs(strong_train, kernel, ref_rho, args.noise_std, args.seed + 11)
    weak_g = gravity_obs(weak_train, kernel, ref_rho, args.noise_std, args.seed + 21)
    strong_cp = strong_train[:, cp_idx]
    weak_cp = weak_train[:, cp_idx]

    print("[info] training direct strong ensemble")
    strong_models, strong_stats = train_direct_ensemble(
        strong_g, strong_cp, range(args.seed, args.seed + args.direct_ensemble), device, args.direct_epochs, args.batch_size
    )
    print("[info] training direct weak ensemble")
    weak_models, weak_stats = train_direct_ensemble(
        weak_g, weak_cp, range(args.seed + 100, args.seed + 100 + args.direct_ensemble), device, args.direct_epochs, args.batch_size
    )

    print("[info] building gravity test sets")
    _, in_rho = sample_strong_rho(strong_mod, args.n_test, args.seed + 30, n_depth, dz)
    boundary_rho = make_parametric_rho("boundary", args.n_test, args.seed + 40, depth)
    out_rho = make_parametric_rho("out-of-prior", args.n_test, args.seed + 50, depth)
    test_sets = {
        "in-prior": in_rho,
        "boundary": boundary_rho,
        "out-of-prior": out_rho,
    }
    test_g = {name: gravity_obs(rho, kernel, ref_rho, args.noise_std, args.seed + 60 + i) for i, (name, rho) in enumerate(test_sets.items())}

    print("[info] training gravity forward surrogate for indirect inversion")
    fwd_cp = np.concatenate([strong_cp[: args.n_train // 2], weak_cp[: args.n_train // 2]], axis=0)
    fwd_g = gravity_obs(interp_controls_np(depth, cp_depth, fwd_cp), kernel, ref_rho, args.noise_std, args.seed + 70)
    fwd_model, fwd_stats = train_forward_surrogate(fwd_cp, fwd_g, device, args.forward_epochs, args.batch_size)

    rows: List[Dict] = []
    diagnostics: Dict[str, Dict[str, np.ndarray]] = {"target": {}}
    diagnostics["DI-Strong"] = {}
    diagnostics["DI-Weak"] = {}
    diagnostics["IND-FWD"] = {}
    for name, target in test_sets.items():
        diagnostics["target"][name] = target
        g = test_g[name]
        for method, models, stats in [("DI-Strong", strong_models, strong_stats), ("DI-Weak", weak_models, weak_stats)]:
            tic = time.time()
            cp_samples = predict_direct(models, stats, g, device)
            samples = np.stack([interp_controls_np(depth, cp_depth, cp_samples[:, i, :]) for i in range(cp_samples.shape[1])], axis=1)
            pred = np.median(samples, axis=1)
            row = {"method": method, "test_set": name, "n": int(len(target)), "runtime_s": float(time.time() - tic)}
            row.update(mae_rmse(pred, target))
            row.update(coverage(samples, target))
            row.update(gravity_residual(pred, g, kernel, ref_rho))
            row.update(pull_metrics(pred, target, envelope))
            row.update(roughness(pred))
            rows.append(row)
            diagnostics[method][name] = pred

        tic = time.time()
        n_ind = min(args.n_indirect, len(target))
        preds = []
        ensembles = []
        for i in range(n_ind):
            rng = np.random.default_rng(args.seed + 1000 + i)
            init_cps = [
                envelope["mean"][cp_idx],
                strong_train[int(rng.integers(0, len(strong_train))), cp_idx],
                weak_train[int(rng.integers(0, len(weak_train))), cp_idx],
                make_parametric_rho("out-of-prior", 1, args.seed + 2000 + i, depth)[0, cp_idx],
            ]
            ens_cp, best_cp = optimize_indirect_one(
                fwd_model, fwd_stats, g[i], init_cps, device, args.indirect_steps, lr=0.04
            )
            ensembles.append(interp_controls_np(depth, cp_depth, ens_cp))
            preds.append(np.interp(depth, cp_depth, best_cp).astype(np.float32))
        pred = np.stack(preds)
        ens = np.stack(ensembles)
        row = {"method": "IND-FWD", "test_set": name, "n": int(n_ind), "runtime_s": float(time.time() - tic)}
        row.update(mae_rmse(pred, target[:n_ind]))
        row.update(coverage(ens, target[:n_ind]))
        row.update(gravity_residual(pred, g[:n_ind], kernel, ref_rho))
        row.update(pull_metrics(pred, target[:n_ind], envelope))
        row.update(roughness(pred))
        row["indirect_spread_mean"] = float(np.std(ens, axis=1).mean())
        rows.append(row)
        diagnostics["IND-FWD"][name] = pred

    metrics_path = RESULTS_DIR / "gravity_prior_boundary_metrics.csv"
    write_csv(metrics_path, rows)
    plot_examples(FIGURES_DIR / "gravity_prior_boundary_examples.png", diagnostics, depth, offsets, kernel, ref_rho)
    plot_metrics(FIGURES_DIR / "gravity_prior_boundary_metric_comparison.png", rows)

    lines = [
        "# Simulated Gravity Prior-Boundary Summary",
        "",
        "This companion diagnostic uses the existing strong/weak Earth-model priors but replaces the observation operator with a simulated gravity forward model. The inferred state is density rho(z).",
        "",
        "| Method | Test set | N | Rho MAE | Gravity MAE | Coverage | Pull-in | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['test_set']} | {int(r['n'])} | {r['rho_mae']:.3f} | "
            f"{r['gravity_mae_mgal']:.3f} | {r['coverage_16_84']:.3f} | "
            f"{r['pred_inside_given_target_outside']:.3f} | {r['runtime_s']:.1f} |"
        )
    lines += [
        "",
        "Interpretation: gravity inversion is intentionally non-unique. These rows diagnose prior-boundary behavior rather than defining a calibrated field-data posterior.",
    ]
    (RESULTS_DIR / "gravity_prior_boundary_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {metrics_path}")


if __name__ == "__main__":
    main()
