#!/usr/bin/env python3
"""Minimal prior-boundary diagnostic for direct and forward-surrogate inversion.

This script intentionally reuses the existing project modules and checkpoints.
It is designed as a small diagnostic, not a definitive benchmark.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DEFAULT_STRONG_CKPT = ROOT / "ckpt" / "disp2struct_crf.v1.2_cp" / "best.pt"
DEFAULT_WEAK_CKPT = ROOT / "ckpt" / "disp2struct_crf.v1.2_cp_weak" / "best.pt"
DEFAULT_FWD_CKPT = ROOT / "ckpt" / "struct2disp_cpmlp.prior_boundary_v3.pt"
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


class NormalizedForwardSurrogate(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, stats: Dict[str, torch.Tensor]):
        super().__init__()
        self.base = base
        self.register_buffer("x_mean", stats["x_mean"].float())
        self.register_buffer("x_scale", stats["x_scale"].float())
        self.register_buffer("y_mean", stats["y_mean"].float())
        self.register_buffer("y_scale", stats["y_scale"].float())

    def forward(self, x: torch.Tensor, periods: Optional[torch.Tensor] = None):
        xn = (x - self.x_mean) / torch.clamp(self.x_scale, min=1e-6)
        mu_n, logvar = self.base(xn, periods=periods)
        mu = mu_n * self.y_scale + self.y_mean
        return mu, logvar


class ControlPointForwardMLP(torch.nn.Module):
    def __init__(
        self,
        n_depth: int,
        n_periods: int,
        control_indices: List[int],
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        y_mean: torch.Tensor,
        y_scale: torch.Tensor,
        hidden: int = 512,
        layers: int = 5,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_depth = n_depth
        self.n_periods = n_periods
        self.register_buffer("control_indices", torch.tensor(control_indices, dtype=torch.long))
        self.register_buffer("x_mean", x_mean.float())
        self.register_buffer("x_scale", x_scale.float())
        self.register_buffer("y_mean", y_mean.float())
        self.register_buffer("y_scale", y_scale.float())
        n_in = 3 * len(control_indices)
        n_out = 2 * n_periods
        blocks = []
        dim = n_in
        for _ in range(layers):
            blocks += [torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            dim = hidden
        blocks.append(torch.nn.Linear(dim, n_out))
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, periods: Optional[torch.Tensor] = None):
        cp = x[:, 1:4, :].index_select(-1, self.control_indices).reshape(x.shape[0], -1)
        cp_n = (cp - self.x_mean) / torch.clamp(self.x_scale, min=1e-6)
        y_n = self.net(cp_n)
        y = y_n * self.y_scale + self.y_mean
        y = y.reshape(x.shape[0], 2, self.n_periods)
        logvar = torch.zeros_like(y)
        return y, logvar


def strong_dataset(strong_mod, n: int, seed: int):
    return strong_mod.SurfaceWaveDataset(n_samples=n, z_max_km=150.0, z_max_num=256, dz_km=0.5, seed=seed)


def dataset_to_arrays(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    models, dispersions, masks = [], [], []
    for i in range(len(dataset)):
        model, disp, mask = dataset[i]
        models.append(model.numpy())
        dispersions.append(disp.numpy())
        masks.append(mask.numpy())
    return np.stack(models), np.stack(dispersions), np.stack(masks)


def brocher_rho(strong_mod, vp: np.ndarray) -> np.ndarray:
    if hasattr(strong_mod, "brocher_rho_from_vp"):
        return strong_mod.brocher_rho_from_vp(vp)
    rho = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    return np.clip(rho, 1.2, 3.8)


def smooth(x: np.ndarray, passes: int = 2) -> np.ndarray:
    y = x.astype(np.float64).copy()
    for _ in range(passes):
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    return y.astype(np.float32)


def make_parametric_profile(
    strong_mod,
    rng: np.random.Generator,
    kind: str,
    n_depth: int,
    dz: float,
) -> np.ndarray:
    depth = np.arange(n_depth, dtype=np.float32) * dz
    zmax = float(depth[-1])
    if kind == "boundary":
        moho = rng.choice([rng.uniform(8.0, 15.0), rng.uniform(60.0, 75.0)])
        sediment = rng.choice([rng.uniform(0.0, 1.5), rng.uniform(9.0, 14.0)])
        mantle_vs = rng.choice([rng.uniform(3.95, 4.15), rng.uniform(5.00, 5.25)])
        lvz_amp = rng.uniform(0.10, 0.18)
        lvz_center = rng.choice([rng.uniform(45.0, 65.0), rng.uniform(110.0, 145.0)])
        lvz_width = rng.uniform(18.0, 50.0)
    elif kind == "out-of-prior":
        moho = rng.choice([rng.uniform(5.0, 10.0), rng.uniform(78.0, 95.0)])
        sediment = rng.uniform(12.0, 24.0)
        mantle_vs = rng.choice([rng.uniform(3.45, 3.85), rng.uniform(5.25, 5.75)])
        lvz_amp = rng.uniform(0.18, 0.32)
        lvz_center = rng.choice([rng.uniform(25.0, 45.0), rng.uniform(135.0, 175.0)])
        lvz_width = rng.uniform(25.0, 80.0)
    else:
        raise ValueError(kind)

    sediment = min(float(sediment), max(0.0, float(moho) - 3.0))
    moho = min(float(moho), zmax - 10.0)
    lvz_center = min(float(lvz_center), zmax - 2.0)
    crust_mid = max(sediment + 1.0, moho * 0.55)

    sed_vs0 = rng.uniform(0.18, 0.65) if kind == "out-of-prior" else rng.uniform(0.35, 0.90)
    sed_vs1 = rng.uniform(1.0, 2.1)
    upper_crust = rng.uniform(2.5, 3.45)
    lower_crust = rng.uniform(3.35, 4.15)
    knots = np.array([0.0, sediment, crust_mid, moho, zmax], dtype=np.float32)
    values = np.array([sed_vs0, sed_vs1, upper_crust, lower_crust, mantle_vs], dtype=np.float32)
    order = np.argsort(knots)
    vs = np.interp(depth, knots[order], values[order]).astype(np.float32)

    lvz = lvz_amp * mantle_vs * np.exp(-0.5 * ((depth - lvz_center) / lvz_width) ** 2)
    vs = vs - lvz * (depth > moho)
    vs += rng.normal(0.0, 0.025 if kind == "boundary" else 0.04, size=vs.shape).astype(np.float32)
    vs = smooth(np.clip(vs, 0.15, 5.85), passes=3)

    ratio = np.where(depth < sediment, rng.uniform(1.85, 2.15), rng.uniform(1.68, 1.88))
    vp = vs * ratio + rng.normal(0.0, 0.03, size=vs.shape).astype(np.float32)
    vp = smooth(np.maximum(vp, vs + 0.20), passes=2)
    vp = np.clip(vp, 0.8, 10.5).astype(np.float32)
    rho = brocher_rho(strong_mod, vp).astype(np.float32)
    rho = smooth(np.clip(rho, 1.2, 3.8), passes=2)
    return np.stack([vp, vs, rho]).astype(np.float32)


def compute_dispersion(strong_mod, profile: np.ndarray, periods: np.ndarray) -> Optional[np.ndarray]:
    depth = (np.arange(profile.shape[-1], dtype=np.float64) * 0.5).astype(np.float64)
    vp, vs, rho = [np.asarray(profile[i], dtype=np.float64) for i in range(3)]
    periods64 = np.asarray(periods, dtype=np.float64)
    try:
        ray_out = strong_mod.compute_phase_dispersion(depth, vp, vs, rho, periods64, modes=(0,), wave="rayleigh")
        love_out = strong_mod.compute_phase_dispersion(depth, vp, vs, rho, periods64, modes=(0,), wave="love")
        ray = ray_out[0].velocity.astype(np.float32)
        love = love_out[0].velocity.astype(np.float32)
    except Exception:
        return None
    if np.any(~np.isfinite(ray)) or np.any(~np.isfinite(love)):
        return None
    disp = np.stack([periods, ray, love]).astype(np.float32)
    return disp


def parametric_dataset(
    strong_mod,
    kind: str,
    n: int,
    seed: int,
    n_depth: int,
    periods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    models, dispersions = [], []
    attempts = 0
    while len(models) < n and attempts < n * 40:
        attempts += 1
        model = make_parametric_profile(strong_mod, rng, kind, n_depth=n_depth, dz=0.5)
        disp = compute_dispersion(strong_mod, model, periods)
        if disp is None:
            continue
        models.append(model)
        dispersions.append(disp)
    if len(models) < n:
        raise RuntimeError(f"Only generated {len(models)} valid {kind} samples after {attempts} attempts")
    masks = np.ones((n, 3, len(periods)), dtype=np.float32)
    return np.stack(models), np.stack(dispersions), masks


def prior_envelope(strong_mod, n: int, seed: int) -> Dict[str, np.ndarray]:
    models, _, _ = dataset_to_arrays(strong_dataset(strong_mod, n, seed))
    models = models[:, 1:4, :]
    return {
        "lo": np.quantile(models, 0.01, axis=0).astype(np.float32),
        "hi": np.quantile(models, 0.99, axis=0).astype(np.float32),
        "mean": models.mean(axis=0).astype(np.float32),
    }


def load_direct_model(module_path: Path, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        return None, None
    direct_mod = import_from_path(f"direct_inv_{ckpt_path.parent.name}", module_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    kwargs = {
        "H": cfg.get("z_max_num", ckpt.get("depth_grid", torch.arange(256)).numel()),
        "T": int(ckpt.get("period_minmax", torch.tensor([2.0, 60.0])).numel() * 0 + cfg.get("n_periods", 59)),
        "profile_channels": 3,
        "cond_base_channels": cfg.get("cond_base_channels", 64),
        "cond_dim": cfg.get("cond_dim", 256),
        "flow_hidden": cfg.get("flow_hidden", 1024),
        "time_dim": cfg.get("time_dim", 64),
        "dropout": cfg.get("dropout", 0.1),
        "reference_profile": ckpt.get("reference_profile"),
        "profile_scale": ckpt.get("profile_scale"),
        "depth_grid": ckpt.get("depth_grid"),
        "control_indices": ckpt.get("control_indices"),
        "period_minmax": tuple(float(x) for x in ckpt.get("period_minmax", torch.tensor([2.0, 60.0])).reshape(-1).tolist()),
        "disp_mean": ckpt.get("disp_mean"),
        "disp_scale": ckpt.get("disp_scale"),
    }
    model = direct_mod.Disp2StructCRF(**kwargs)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def direct_samples(
    model,
    disp: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    n_samples: int,
    steps: int,
    batch_size: int,
) -> np.ndarray:
    all_samples: List[np.ndarray] = []
    for start in range(0, len(disp), batch_size):
        end = min(len(disp), start + batch_size)
        y = torch.from_numpy(disp[start:end]).to(device)
        m = torch.from_numpy(mask[start:end]).to(device)
        batch_samples = []
        for _ in range(n_samples):
            try:
                out = model.sample(y, mask=m, num_samples=1, num_steps=steps)
                sample = out["profile_samples"][:, 0, :, :]
            except TypeError:
                sample = model.sample(y, mask=m, n_steps=steps)
            batch_samples.append(sample.detach().cpu().numpy())
        all_samples.append(np.stack(batch_samples, axis=1))
    return np.concatenate(all_samples, axis=0)


def mae_rmse(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    names = ["vp", "vs", "rho"]
    out: Dict[str, float] = {}
    err = pred - target
    for i, name in enumerate(names):
        out[f"{name}_mae"] = float(np.mean(np.abs(err[:, i, :])))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err[:, i, :] ** 2)))
    return out


def profile_roughness(profile: np.ndarray) -> Dict[str, float]:
    names = ["vp", "vs", "rho"]
    d2 = profile[:, :, 2:] - 2.0 * profile[:, :, 1:-1] + profile[:, :, :-2]
    out = {"roughness_mean": float(np.mean(np.abs(d2)))}
    for i, name in enumerate(names):
        out[f"{name}_roughness"] = float(np.mean(np.abs(d2[:, i, :])))
    return out


def coverage(samples: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    q16 = np.quantile(samples, 0.16, axis=1)
    q84 = np.quantile(samples, 0.84, axis=1)
    inside = (target >= q16) & (target <= q84)
    return {
        "coverage_16_84_mean": float(inside.mean()),
        "coverage_vp": float(inside[:, 0, :].mean()),
        "coverage_vs": float(inside[:, 1, :].mean()),
        "coverage_rho": float(inside[:, 2, :].mean()),
    }


def prior_pull_metrics(pred: np.ndarray, target: np.ndarray, envelope: Dict[str, np.ndarray]) -> Dict[str, float]:
    lo = envelope["lo"][None, :, :]
    hi = envelope["hi"][None, :, :]
    target_outside = (target < lo) | (target > hi)
    pred_inside = (pred >= lo) & (pred <= hi)
    pred_at_boundary = (np.abs(pred - lo) < 0.05 * np.maximum(hi - lo, 1e-3)) | (
        np.abs(pred - hi) < 0.05 * np.maximum(hi - lo, 1e-3)
    )
    outside_count = float(target_outside.sum())
    return {
        "target_outside_fraction": float(target_outside.mean()),
        "pred_inside_given_target_outside": float((pred_inside & target_outside).sum() / max(outside_count, 1.0)),
        "boundary_pull_fraction": float((pred_at_boundary & target_outside).sum() / max(outside_count, 1.0)),
    }


def dispersion_residuals(strong_mod, pred: np.ndarray, disp: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    periods = disp[0, 0, :]
    residuals = []
    for i in range(len(pred)):
        pred_disp = compute_dispersion(strong_mod, pred[i], periods)
        if pred_disp is None:
            continue
        wave_mask = mask[i, 1:3, :].astype(bool)
        residuals.append((pred_disp[1:3, :] - disp[i, 1:3, :])[wave_mask])
    if not residuals:
        return {"pred_disp_mae": math.nan, "pred_disp_rmse": math.nan}
    r = np.concatenate(residuals)
    return {"pred_disp_mae": float(np.mean(np.abs(r))), "pred_disp_rmse": float(np.sqrt(np.mean(r**2)))}


def evaluate_direct(
    label: str,
    model,
    test_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    strong_mod,
    envelope: Dict[str, np.ndarray],
    device: torch.device,
    args,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    rows = []
    diagnostics = {}
    for name, (target, disp, mask) in test_sets.items():
        tic = time.time()
        samples = direct_samples(model, disp, mask, device, args.posterior_samples, args.sampling_steps, args.batch_size)
        pred = np.median(samples, axis=1)
        row: Dict[str, float] = {
            "method": label,
            "test_set": name,
            "n": int(len(target)),
            "status": "ok",
            "runtime_s": float(time.time() - tic),
        }
        row.update(mae_rmse(pred, target))
        row.update(profile_roughness(pred))
        row.update(coverage(samples, target))
        row.update(dispersion_residuals(strong_mod, pred, disp, mask))
        row.update(prior_pull_metrics(pred, target, envelope))
        rows.append(row)
        diagnostics[name] = {
            "target": target,
            "pred": pred,
            "samples": samples,
            "vs_bias": (pred[:, 1, :] - target[:, 1, :]).mean(axis=0),
        }
    return rows, diagnostics


def select_control_indices(n_depth: int) -> List[int]:
    fine = list(range(0, min(20, n_depth), 2))
    mid = list(range(20, min(100, n_depth), 10))
    deep = list(range(100, n_depth, 32))
    idx = sorted(set(fine + mid + deep + [n_depth - 1]))
    return idx


def load_forward_model(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    if cfg.get("model_type") == "control_point_mlp":
        stats = {k: torch.as_tensor(v) for k, v in ckpt["normalization"].items()}
        model = ControlPointForwardMLP(
            n_depth=cfg.get("H", cfg.get("n_depth", 256)),
            n_periods=cfg.get("T", cfg.get("n_periods", 59)),
            control_indices=cfg["control_indices"],
            x_mean=stats["x_mean"],
            x_scale=stats["x_scale"],
            y_mean=stats["y_mean"],
            y_scale=stats["y_scale"],
            hidden=cfg.get("hidden", 512),
            layers=cfg.get("layers", 5),
            dropout=cfg.get("dropout", 0.05),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model_mod = import_from_path("struct2disp_transformer_prior_boundary", ROOT / "models" / "struct2disp_transformer.py")
        model = model_mod.Struct2DispTransformer(
            H=cfg.get("H", cfg.get("n_depth", 256)),
            T=cfg.get("T", cfg.get("n_periods", 59)),
            C_in=cfg.get("C_in", cfg.get("d_struct", 4)),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_enc_layers=cfg.get("num_enc_layers", cfg.get("num_encoder_layers", 2)),
            num_dec_layers=cfg.get("num_dec_layers", cfg.get("num_decoder_layers", 2)),
            dim_ff=cfg.get("dim_ff", cfg.get("dim_feedforward", 256)),
            dropout=cfg.get("dropout", 0.05),
            period_minmax=tuple(cfg.get("period_minmax", (2.0, 60.0))),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "normalization" in ckpt:
            stats = {k: torch.as_tensor(v) for k, v in ckpt["normalization"].items()}
            model = NormalizedForwardSurrogate(model, stats)
    model.to(device).eval()
    return model


def make_forward_training_arrays(
    strong_mod,
    n: int,
    seed: int,
    periods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_strong = max(1, int(round(n * 0.50)))
    n_boundary = max(1, int(round(n * 0.25)))
    n_out = max(1, n - n_strong - n_boundary)
    strong_models, strong_disp, _ = dataset_to_arrays(strong_dataset(strong_mod, n_strong, seed))
    boundary_profiles, boundary_disp, _ = parametric_dataset(
        strong_mod, "boundary", n_boundary, seed + 1_000, strong_models.shape[-1], periods
    )
    outside_profiles, outside_disp, _ = parametric_dataset(
        strong_mod, "out-of-prior", n_out, seed + 2_000, strong_models.shape[-1], periods
    )
    depth = np.arange(strong_models.shape[-1], dtype=np.float32) * 0.5
    boundary_models = np.concatenate(
        [np.broadcast_to(depth[None, None, :], (n_boundary, 1, len(depth))), boundary_profiles], axis=1
    )
    outside_models = np.concatenate(
        [np.broadcast_to(depth[None, None, :], (n_out, 1, len(depth))), outside_profiles], axis=1
    )
    x = np.concatenate([strong_models, boundary_models, outside_models], axis=0).astype(np.float32)
    disp = np.concatenate([strong_disp, boundary_disp, outside_disp], axis=0).astype(np.float32)
    labels = np.array(["in-prior"] * n_strong + ["boundary"] * n_boundary + ["out-of-prior"] * n_out)
    return x, disp[:, 0, :].astype(np.float32), disp[:, 1:3, :].astype(np.float32), labels


@torch.no_grad()
def forward_surrogate_residual_rows(
    fwd_model,
    test_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    prefix: str = "surrogate",
) -> List[Dict[str, float]]:
    rows = []
    for name, (profile, disp, mask) in test_sets.items():
        depth = np.arange(profile.shape[-1], dtype=np.float32) * 0.5
        x_np = np.concatenate([np.broadcast_to(depth[None, None, :], (len(profile), 1, len(depth))), profile], axis=1)
        preds = []
        for start in range(0, len(profile), 32):
            end = min(len(profile), start + 32)
            x = torch.from_numpy(x_np[start:end]).to(device)
            periods = torch.from_numpy(disp[start:end, 0, :]).to(device)
            pred, _ = fwd_model(x, periods=periods)
            preds.append(pred.detach().cpu().numpy())
        pred_np = np.concatenate(preds, axis=0)
        residual = pred_np - disp[:, 1:3, :]
        wave_mask = mask[:, 1:3, :].astype(bool)
        ray = residual[:, 0, :][mask[:, 1, :].astype(bool)]
        love = residual[:, 1, :][mask[:, 2, :].astype(bool)]
        both = residual[wave_mask]
        rows.append(
            {
                "method": prefix,
                "test_set": name,
                "n": int(len(profile)),
                "surrogate_disp_mae": float(np.mean(np.abs(both))),
                "surrogate_disp_rmse": float(np.sqrt(np.mean(both**2))),
                "surrogate_rayleigh_mae": float(np.mean(np.abs(ray))) if len(ray) else math.nan,
                "surrogate_love_mae": float(np.mean(np.abs(love))) if len(love) else math.nan,
            }
        )
    return rows


def train_tiny_forward_surrogate(
    strong_mod,
    ckpt_path: Path,
    device: torch.device,
    n_train: int,
    epochs: int,
    seed: int,
    batch_size: int,
):
    periods_grid = np.linspace(2.0, 60.0, 59).astype(np.float32)
    x, periods, y, labels = make_forward_training_arrays(strong_mod, n_train, seed, periods_grid)
    n_val = min(512, max(96, n_train // 5))
    x_val, p_val, y_val, labels_val = make_forward_training_arrays(strong_mod, n_val, seed + 100_000, periods_grid)
    control_indices = select_control_indices(x.shape[-1])
    x_cp = x[:, 1:4, :][:, :, control_indices].reshape(len(x), -1)
    y_flat = y.reshape(len(y), -1)
    x_mean = torch.from_numpy(x_cp.mean(axis=0, keepdims=True).astype(np.float32))
    x_scale = torch.from_numpy((x_cp.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    y_mean = torch.from_numpy(y_flat.mean(axis=0, keepdims=True).astype(np.float32))
    y_scale = torch.from_numpy((y_flat.std(axis=0, keepdims=True) + 1e-6).astype(np.float32))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(periods), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
    )
    cfg = {
        "model_type": "control_point_mlp",
        "H": int(x.shape[-1]),
        "T": int(y.shape[-1]),
        "control_indices": [int(i) for i in control_indices],
        "hidden": 512,
        "layers": 5,
        "dropout": 0.05,
        "period_minmax": (2.0, 60.0),
    }
    normalization = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }
    model = ControlPointForwardMLP(
        n_depth=cfg["H"],
        n_periods=cfg["T"],
        control_indices=control_indices,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=8e-5)
    model.train()
    xv = torch.from_numpy(x_val).float().to(device)
    pv = torch.from_numpy(p_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    ys = y_scale.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        n_seen = 0
        for xb, pb, yb in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(xb, periods=pb)
            loss = torch.mean(((pred - yb).reshape(len(xb), -1) / torch.clamp(ys, min=1e-6)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n_seen += len(xb)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred_raw, _ = model(xv, periods=pv)
            val_mae = float(torch.mean(torch.abs(pred_raw - yv)).detach().cpu())
            train_loss = total / max(n_seen, 1)
        model.train()
        history.append({"epoch": epoch, "train_norm_mse": train_loss, "val_disp_mae": val_mae})
        print(f"[forward epoch {epoch:03d}/{epochs:03d}] train_norm_mse={train_loss:.5f} val_disp_mae={val_mae:.4f}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "normalization": normalization,
            "training_history": history,
            "training_distribution": {
                "n_train": int(n_train),
                "n_val": int(n_val),
                "mix": "50% in-prior, 25% boundary, 25% out-of-prior",
            },
        },
        ckpt_path,
    )
    wrapped = model.to(device).eval()
    val_sets = {}
    for label in ["in-prior", "boundary", "out-of-prior"]:
        idx = np.where(labels_val == label)[0]
        if len(idx) == 0:
            continue
        masks = np.ones((len(idx), 3, y_val.shape[-1]), dtype=np.float32)
        depth = x_val[idx, 0:1, :]
        profiles = x_val[idx, 1:4, :]
        disp = np.concatenate([p_val[idx, None, :], y_val[idx]], axis=1)
        val_sets[label] = (profiles, disp, masks)
    write_csv(RESULTS_DIR / "forward_surrogate_validation.csv", forward_surrogate_residual_rows(wrapped, val_sets, device))
    return wrapped


def torch_interp_controls(depth: torch.Tensor, cp_depth: torch.Tensor, cp_values: torch.Tensor) -> torch.Tensor:
    right = torch.searchsorted(cp_depth, depth).clamp(1, len(cp_depth) - 1)
    left = right - 1
    dl = cp_depth[left]
    dr = cp_depth[right]
    w = (depth - dl) / torch.clamp(dr - dl, min=1e-6)
    return (1.0 - w)[None, :] * cp_values[:, left] + w[None, :] * cp_values[:, right]


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(torch.clamp(x, min=1e-4)))


def inverse_sigmoid_bounds(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    p = ((x - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def forward_iterative_invert_one(
    fwd_model,
    obs_disp: np.ndarray,
    obs_mask: np.ndarray,
    init_profile: np.ndarray,
    device: torch.device,
    steps: int,
    lr: float,
) -> Tuple[np.ndarray, float]:
    n_depth = init_profile.shape[-1]
    depth_np = np.arange(n_depth, dtype=np.float32) * 0.5
    cp_idx = select_control_indices(n_depth)
    cp_depth = torch.tensor(depth_np[cp_idx], dtype=torch.float32, device=device)
    depth = torch.tensor(depth_np, dtype=torch.float32, device=device)
    init_cp = torch.tensor(init_profile[:, cp_idx], dtype=torch.float32, device=device)
    lo = torch.tensor([[0.60], [0.10], [1.10]], dtype=torch.float32, device=device)
    hi = torch.tensor([[10.80], [5.95], [3.95]], dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(inverse_sigmoid_bounds(init_cp, lo, hi))
    opt = torch.optim.Adam([raw], lr=lr)
    periods = torch.tensor(obs_disp[None, 0, :], dtype=torch.float32, device=device)
    y = torch.tensor(obs_disp[None, 1:3, :], dtype=torch.float32, device=device)
    m = torch.tensor(obs_mask[None, 1:3, :], dtype=torch.float32, device=device)
    depth_chan = depth[None, None, :]
    best_loss = float("inf")
    best_full = None
    for _ in range(steps):
        cp = lo + (hi - lo) * torch.sigmoid(raw)
        full = torch_interp_controls(depth, cp_depth, cp)
        vp, vs, rho = full[0], full[1], full[2]
        x = torch.cat([depth_chan, full[None, :, :]], dim=1)
        pred, _ = fwd_model(x, periods=periods)
        data_loss = (((pred - y) * m) ** 2).sum() / torch.clamp(m.sum(), min=1.0)
        smooth_loss = ((cp[:, 2:] - 2.0 * cp[:, 1:-1] + cp[:, :-2]) ** 2).mean()
        phys_loss = (
            F.relu(vs - vp + 0.05).pow(2).mean()
            + F.relu(0.10 - vs).pow(2).mean()
            + F.relu(1.20 - rho).pow(2).mean()
            + F.relu(rho - 3.80).pow(2).mean()
        )
        soft_bound = (
            F.relu(0.80 - vp).pow(2).mean()
            + F.relu(vp - 10.50).pow(2).mean()
            + F.relu(0.15 - vs).pow(2).mean()
            + F.relu(vs - 5.80).pow(2).mean()
            + F.relu(1.20 - rho).pow(2).mean()
            + F.relu(rho - 3.80).pow(2).mean()
        )
        loss = data_loss + 5e-3 * smooth_loss + 5.0 * phys_loss + 1e-2 * soft_bound
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        current = float(data_loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_full = full.detach()
    with torch.no_grad():
        cp = lo + (hi - lo) * torch.sigmoid(raw)
        full = torch_interp_controls(depth, cp_depth, cp)
        x = torch.cat([depth_chan, full[None, :, :]], dim=1)
        pred, _ = fwd_model(x, periods=periods)
        data_loss = (((pred - y) * m) ** 2).sum() / torch.clamp(m.sum(), min=1.0)
        if float(data_loss.detach().cpu()) < best_loss or best_full is None:
            best_loss = float(data_loss.detach().cpu())
            best_full = full.detach()
    return best_full.cpu().numpy().astype(np.float32), best_loss


def evaluate_forward_iterative(
    fwd_model,
    test_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    strong_mod,
    envelope: Dict[str, np.ndarray],
    device: torch.device,
    args,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    rows = []
    diagnostics = {}
    for name, (target_all, disp_all, mask_all) in test_sets.items():
        n = min(args.n_forward_eval, len(target_all))
        target = target_all[:n]
        disp = disp_all[:n]
        mask = mask_all[:n]
        tic = time.time()
        preds = []
        ensembles = []
        best_losses = []
        for i in range(n):
            init_profiles = make_indirect_initial_profiles(
                strong_mod,
                envelope,
                target.shape[-1],
                args.seed + 7_000 + i,
                max(1, args.indirect_multistarts),
            )
            sample_preds = []
            sample_losses = []
            for init in init_profiles:
                p_i, l_i = forward_iterative_invert_one(
                    fwd_model,
                    disp[i],
                    mask[i],
                    init,
                    device,
                    args.forward_inv_steps,
                    args.forward_inv_lr,
                )
                sample_preds.append(p_i)
                sample_losses.append(l_i)
            sample_preds_np = np.stack(sample_preds)
            best_idx = int(np.argmin(sample_losses))
            preds.append(sample_preds_np[best_idx])
            ensembles.append(sample_preds_np)
            best_losses.append(float(sample_losses[best_idx]))
        pred = np.stack(preds)
        ens = np.stack(ensembles)
        q16 = np.quantile(ens, 0.16, axis=1)
        q84 = np.quantile(ens, 0.84, axis=1)
        inside = (target >= q16) & (target <= q84)
        spread = np.std(ens, axis=1)
        row: Dict[str, float] = {
            "method": "IND-FWD",
            "test_set": name,
            "n": int(n),
            "status": "ok",
            "runtime_s": float(time.time() - tic),
            "coverage_16_84_mean": float(inside.mean()),
            "coverage_vp": float(inside[:, 0, :].mean()),
            "coverage_vs": float(inside[:, 1, :].mean()),
            "coverage_rho": float(inside[:, 2, :].mean()),
            "indirect_spread_vs_mean": float(spread[:, 1, :].mean()),
            "indirect_best_surrogate_mse": float(np.mean(best_losses)),
        }
        row.update(mae_rmse(pred, target))
        row.update(profile_roughness(pred))
        row.update(dispersion_residuals(strong_mod, pred, disp, mask))
        row.update(prior_pull_metrics(pred, target, envelope))
        rows.append(row)
        diagnostics[name] = {
            "target": target,
            "pred": pred,
            "ensemble": ens,
            "vs_bias": (pred[:, 1, :] - target[:, 1, :]).mean(axis=0),
        }
    return rows, diagnostics


def make_indirect_initial_profiles(
    strong_mod,
    envelope: Dict[str, np.ndarray],
    n_depth: int,
    seed: int,
    count: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    inits = [envelope["mean"].astype(np.float32)]
    if count >= 2:
        perturb = envelope["mean"] + rng.normal(0.0, 0.35, size=envelope["mean"].shape).astype(np.float32)
        perturb[0] = np.maximum(perturb[0], perturb[1] + 0.2)
        perturb[1] = np.clip(perturb[1], 0.15, 5.8)
        perturb[2] = np.clip(perturb[2], 1.2, 3.8)
        inits.append(perturb.astype(np.float32))
    if count >= 3:
        depth = np.arange(n_depth, dtype=np.float32) * 0.5
        cp = np.array([0.0, 5.0, 20.0, 60.0, depth[-1]], dtype=np.float32)
        vs_cp = rng.uniform([0.2, 1.2, 2.6, 3.5, 3.8], [1.1, 2.8, 4.2, 5.6, 5.8]).astype(np.float32)
        vs = smooth(np.interp(depth, cp, vs_cp).astype(np.float32), passes=3)
        vp = smooth(vs * rng.uniform(1.65, 2.05) + rng.normal(0.0, 0.04, size=n_depth), passes=2)
        vp = np.maximum(vp, vs + 0.2).astype(np.float32)
        rho = brocher_rho(strong_mod, vp).astype(np.float32)
        inits.append(np.stack([vp, vs, rho]).astype(np.float32))
    while len(inits) < count:
        kind = "out-of-prior" if len(inits) % 2 == 0 else "boundary"
        inits.append(make_parametric_profile(strong_mod, rng, kind, n_depth=n_depth, dz=0.5))
    return inits[:count]


def indirect_uncertainty_diagnostic(
    fwd_model,
    test_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    strong_mod,
    envelope: Dict[str, np.ndarray],
    device: torch.device,
    args,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    rows = []
    diagnostics = {}
    for name, (target_all, disp_all, mask_all) in test_sets.items():
        n = min(args.indirect_uncertainty_samples, len(target_all))
        if n <= 0:
            continue
        target = target_all[:n]
        ensembles = []
        tic = time.time()
        for i in range(n):
            init_profiles = make_indirect_initial_profiles(
                strong_mod,
                envelope,
                target.shape[-1],
                args.seed + 1000 + i,
                args.indirect_multistarts,
            )
            preds = []
            for init in init_profiles:
                pred_i, _ = forward_iterative_invert_one(
                    fwd_model,
                    disp_all[i],
                    mask_all[i],
                    init,
                    device,
                    max(20, args.indirect_uncertainty_steps),
                    args.forward_inv_lr,
                )
                preds.append(pred_i)
            ensembles.append(np.stack(preds))
        ens = np.stack(ensembles)  # [N, S, 3, H]
        q16 = np.quantile(ens, 0.16, axis=1)
        q84 = np.quantile(ens, 0.84, axis=1)
        inside = (target >= q16) & (target <= q84)
        spread = np.std(ens, axis=1)
        med = np.median(ens, axis=1)
        row = {
            "method": "IND-FWD-uncertainty",
            "test_set": name,
            "n": int(n),
            "status": "ok",
            "runtime_s": float(time.time() - tic),
            "indirect_multistarts": int(args.indirect_multistarts),
            "indirect_coverage_16_84_mean": float(inside.mean()),
            "indirect_coverage_vs": float(inside[:, 1, :].mean()),
            "indirect_spread_vs_mean": float(spread[:, 1, :].mean()),
            "indirect_spread_mean": float(spread.mean()),
        }
        row.update(mae_rmse(med, target))
        row.update(profile_roughness(med))
        rows.append(row)
        diagnostics[name] = {
            "target": target,
            "ensemble": ens,
            "median": med,
            "spread_vs_depth": spread[:, 1, :].mean(axis=0),
            "coverage_vs_depth": inside[:, 1, :].mean(axis=0),
        }
    return rows, diagnostics


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def plot_direct_examples(path: Path, diagnostics: Dict[str, Dict[str, np.ndarray]]) -> None:
    depth = np.arange(next(iter(diagnostics.values()))["target"].shape[-1]) * 0.5
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.3), sharey=True)
    channels = [("Vp", 0), ("Vs", 1), ("rho", 2)]
    for r, (set_name, diag) in enumerate(diagnostics.items()):
        target = diag["target"][0]
        pred = diag["pred"][0]
        samples = diag.get("samples")
        for c, (label, idx) in enumerate(channels):
            ax = axes[r, c]
            if samples is not None:
                q16 = np.quantile(samples[0, :, idx, :], 0.16, axis=0)
                q84 = np.quantile(samples[0, :, idx, :], 0.84, axis=0)
                ax.fill_betweenx(depth, q16, q84, color="#d62728", alpha=0.16, lw=0)
            ax.plot(target[idx], depth, "k-", lw=1.5, label="true")
            ax.plot(pred[idx], depth, color="#d62728", lw=1.5, ls="--", label="median")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"{set_name}\nDepth (km)")
            if r == 2:
                ax.set_xlabel("km/s" if idx < 2 else "g/cm3")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "make_paper_figures.py"})
    plt.close(fig)


def plot_bias(path: Path, bias_by_method: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.7), sharey=True)
    colors = {"DI-Strong": "#d62728", "DI-Weak": "#1f77b4", "IND-FWD": "#2ca02c"}
    for ax, set_name in zip(axes, ["in-prior", "boundary", "out-of-prior"]):
        for method, diags in bias_by_method.items():
            if set_name not in diags:
                continue
            bias = diags[set_name]["vs_bias"]
            depth = np.arange(len(bias)) * 0.5
            ax.plot(bias, depth, lw=1.6, label=method, color=colors.get(method))
        ax.axvline(0.0, color="0.3", lw=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.set_title(set_name)
        ax.set_xlabel("Vs bias (km/s)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "make_paper_figures.py"})
    plt.close(fig)


def plot_forward_examples(path: Path, diagnostics: Dict[str, Dict[str, np.ndarray]]) -> None:
    depth = np.arange(next(iter(diagnostics.values()))["target"].shape[-1]) * 0.5
    fig, axes = plt.subplots(3, 1, figsize=(3.8, 6.8), sharex=False, sharey=True)
    for ax, (set_name, diag) in zip(axes, diagnostics.items()):
        ax.plot(diag["target"][0, 1], depth, "k-", lw=1.5, label="true")
        ax.plot(diag["pred"][0, 1], depth, color="#2ca02c", lw=1.5, ls="--", label="IND-FWD")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.set_title(set_name)
        ax.set_xlabel("Vs (km/s)")
        ax.set_ylabel("Depth (km)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "make_paper_figures.py"})
    plt.close(fig)


def plot_indirect_uncertainty(path: Path, diagnostics: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), sharey=True)
    for set_name, diag in diagnostics.items():
        depth = np.arange(diag["spread_vs_depth"].shape[-1]) * 0.5
        axes[0].plot(diag["spread_vs_depth"], depth, lw=1.6, label=set_name)
        axes[1].plot(diag["coverage_vs_depth"], depth, lw=1.6, label=set_name)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Depth (km)")
    axes[0].set_xlabel("Multi-start Vs spread (km/s)")
    axes[0].set_title("Indirect ensemble spread")
    axes[1].axvline(0.68, color="0.3", lw=0.8, ls="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Pointwise coverage")
    axes[1].set_title("16-84% coverage")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "make_paper_figures.py"})
    plt.close(fig)


def skipped_rows(method: str, test_sets: Iterable[str], reason: str) -> List[Dict]:
    return [
        {
            "method": method,
            "test_set": name,
            "n": 0,
            "status": reason,
            "runtime_s": math.nan,
        }
        for name in test_sets
    ]


def build_summary(path: Path, rows: List[Dict], run_command: str) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [
        "# Prior-Boundary Diagnostic Summary",
        "",
        "This is a minimal synthetic diagnostic, not a final benchmark. The direct-inversion results test whether an amortized learned inverse mapping tends to return structures toward the support of its training prior when the target model is near or outside that support.",
        "",
        f"Run command: `{run_command}`",
        "",
        "## Main Metrics",
        "",
        "| Method | Test set | n | Vs MAE | Vs RMSE | Disp MAE | Coverage | Target outside | Pred inside given outside | Runtime (s) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def fmt(key):
            val = r.get(key, math.nan)
            if isinstance(val, str):
                return val
            try:
                if math.isnan(float(val)):
                    return "NA"
            except Exception:
                return str(val)
            return f"{float(val):.4f}"

        coverage_key = "coverage_16_84_mean" if "coverage_16_84_mean" in r else "indirect_coverage_16_84_mean"
        lines.append(
            f"| {r.get('method')} | {r.get('test_set')} | {r.get('n')} | {fmt('vs_mae')} | {fmt('vs_rmse')} | "
            f"{fmt('pred_disp_mae')} | {fmt(coverage_key)} | {fmt('target_outside_fraction')} | "
            f"{fmt('pred_inside_given_target_outside')} | {fmt('runtime_s')} | {r.get('status')} |"
        )
    lines += [
        "",
        "## Conservative Interpretation",
        "",
        "1. DI-Strong should be interpreted primarily as a posterior surrogate under the synthetic prior and simulator used during training. Its in-prior behavior is the relevant baseline.",
        "2. Boundary and out-of-prior rows diagnose whether predictions are pulled back toward the strong-prior support. A large `pred_inside_given_target_outside` indicates prior-boundary collapse rather than successful extrapolation.",
        "3. DI-Weak is only comparable if a weak-prior checkpoint is supplied or trained. A skipped row means the current project did not contain that checkpoint at run time.",
        "4. IND-FWD is a preliminary control-point inversion through a forward surrogate. It may reduce the prior-boundary bias of direct inverse mappings, but it remains dependent on the surrogate training domain, control-point parameterization, optimization bounds, initialization, and regularization.",
        "5. Any result intended for the paper should be rerun with larger test sets, more posterior samples, an independently trained weak-prior model, and a validated forward surrogate.",
        "",
        "## Paper-Ready Use",
        "",
        "The strongest paper-facing result from this diagnostic is the prior-support caveat: amortized direct inversion can be accurate inside the synthetic prior while becoming biased near or outside the prior support. The weak-prior and forward-surrogate rows should be presented as preliminary unless they are rerun at larger scale.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--n-envelope", type=int, default=512)
    p.add_argument("--posterior-samples", type=int, default=16)
    p.add_argument("--sampling-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    p.add_argument("--weak-ckpt", type=Path, default=DEFAULT_WEAK_CKPT)
    p.add_argument("--forward-ckpt", type=Path, default=DEFAULT_FWD_CKPT)
    p.add_argument("--train-forward-if-missing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-forward-train", type=int, default=4096)
    p.add_argument("--forward-epochs", type=int, default=20)
    p.add_argument("--n-forward-eval", type=int, default=50)
    p.add_argument("--forward-inv-steps", type=int, default=220)
    p.add_argument("--forward-inv-lr", type=float, default=0.04)
    p.add_argument("--indirect-uncertainty-samples", type=int, default=8)
    p.add_argument("--indirect-multistarts", type=int, default=6)
    p.add_argument("--indirect-uncertainty-steps", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_test = min(args.n_test, 24)
        args.n_envelope = min(args.n_envelope, 192)
        args.posterior_samples = min(args.posterior_samples, 4)
        args.sampling_steps = min(args.sampling_steps, 8)
        args.n_forward_train = min(args.n_forward_train, 128)
        args.forward_epochs = min(args.forward_epochs, 2)
        args.n_forward_eval = min(args.n_forward_eval, 6)
        args.forward_inv_steps = min(args.forward_inv_steps, 60)
        args.indirect_uncertainty_samples = min(args.indirect_uncertainty_samples, 2)
        args.indirect_multistarts = min(args.indirect_multistarts, 3)
        args.indirect_uncertainty_steps = min(args.indirect_uncertainty_steps, 30)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(1)
    device = choose_device(args.device)

    strong_mod = import_from_path("prior_boundary_generate_data", ROOT / "utils" / "generate_data.py")
    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    print(f"[info] device={device}, n_test={args.n_test}, posterior_samples={args.posterior_samples}")
    print("[info] building strong-prior envelope")
    envelope = prior_envelope(strong_mod, args.n_envelope, args.seed + 10)

    print("[info] generating test sets")
    in_models_full, in_disp, in_mask = dataset_to_arrays(strong_dataset(strong_mod, args.n_test, args.seed + 20))
    in_models = in_models_full[:, 1:4, :]
    boundary = parametric_dataset(strong_mod, "boundary", args.n_test, args.seed + 30, in_models.shape[-1], periods)
    outside = parametric_dataset(strong_mod, "out-of-prior", args.n_test, args.seed + 40, in_models.shape[-1], periods)
    test_sets = {
        "in-prior": (in_models, in_disp, in_mask),
        "boundary": boundary,
        "out-of-prior": outside,
    }

    all_rows: List[Dict] = []
    bias_diags: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    print(f"[info] evaluating DI-Strong: {args.strong_ckpt}")
    strong_model, _ = load_direct_model(ROOT / "disp_inv_train.v1.2.py", args.strong_ckpt, device)
    if strong_model is None:
        strong_rows = skipped_rows("DI-Strong", test_sets.keys(), "skipped_missing_checkpoint")
        strong_diag = {}
    else:
        strong_rows, strong_diag = evaluate_direct("DI-Strong", strong_model, test_sets, strong_mod, envelope, device, args)
        write_csv(RESULTS_DIR / "prior_boundary_strong.csv", strong_rows)
        plot_direct_examples(FIGURES_DIR / "prior_boundary_strong_examples.png", strong_diag)
        shutil.copyfile(FIGURES_DIR / "prior_boundary_strong_examples.png", FIGURES_DIR / "direct_prior_boundary_examples.png")
        shutil.copyfile(FIGURES_DIR / "prior_boundary_strong_examples.pdf", FIGURES_DIR / "direct_prior_boundary_examples.pdf")
        bias_diags["DI-Strong"] = strong_diag
    all_rows.extend(strong_rows)

    print(f"[info] evaluating DI-Weak if available: {args.weak_ckpt}")
    weak_model, _ = load_direct_model(ROOT / "disp_inv_train.v1.2.py", args.weak_ckpt, device)
    if weak_model is None:
        weak_rows = skipped_rows("DI-Weak", test_sets.keys(), "skipped_missing_checkpoint")
        write_csv(RESULTS_DIR / "prior_boundary_weak.csv", weak_rows)
    else:
        weak_rows, weak_diag = evaluate_direct("DI-Weak", weak_model, test_sets, strong_mod, envelope, device, args)
        write_csv(RESULTS_DIR / "prior_boundary_weak.csv", weak_rows)
        bias_diags["DI-Weak"] = weak_diag
    all_rows.extend(weak_rows)

    print(f"[info] loading/training forward surrogate: {args.forward_ckpt}")
    fwd_model = load_forward_model(args.forward_ckpt, device)
    if fwd_model is None and args.train_forward_if_missing:
        fwd_model = train_tiny_forward_surrogate(
            strong_mod,
            args.forward_ckpt,
            device,
            args.n_forward_train,
            args.forward_epochs,
            args.seed + 50,
            max(4, args.batch_size),
        )
    if fwd_model is None:
        fwd_rows = skipped_rows("IND-FWD", test_sets.keys(), "skipped_missing_forward_surrogate")
        uncertainty_rows = skipped_rows("IND-FWD-uncertainty", test_sets.keys(), "skipped_missing_forward_surrogate")
    else:
        write_csv(
            RESULTS_DIR / "forward_surrogate_test_residuals.csv",
            forward_surrogate_residual_rows(fwd_model, test_sets, device, prefix="FWD-surrogate"),
        )
        fwd_rows, fwd_diag = evaluate_forward_iterative(fwd_model, test_sets, strong_mod, envelope, device, args)
        write_csv(RESULTS_DIR / "prior_boundary_forward_iterative.csv", fwd_rows)
        plot_forward_examples(FIGURES_DIR / "forward_iterative_examples.png", fwd_diag)
        shutil.copyfile(FIGURES_DIR / "forward_iterative_examples.png", FIGURES_DIR / "indirect_forward_inversion_examples.png")
        shutil.copyfile(FIGURES_DIR / "forward_iterative_examples.pdf", FIGURES_DIR / "indirect_forward_inversion_examples.pdf")
        bias_diags["IND-FWD"] = fwd_diag
        uncertainty_rows, uncertainty_diag = indirect_uncertainty_diagnostic(
            fwd_model, test_sets, strong_mod, envelope, device, args
        )
        write_csv(RESULTS_DIR / "prior_boundary_indirect_uncertainty.csv", uncertainty_rows)
        plot_indirect_uncertainty(FIGURES_DIR / "indirect_uncertainty_diagnostics.png", uncertainty_diag)
    all_rows.extend(fwd_rows)
    all_rows.extend(uncertainty_rows)

    if bias_diags:
        plot_bias(FIGURES_DIR / "prior_boundary_bias_vs_depth.png", bias_diags)
        shutil.copyfile(FIGURES_DIR / "prior_boundary_bias_vs_depth.png", FIGURES_DIR / "vs_bias_vs_depth.png")
        shutil.copyfile(FIGURES_DIR / "prior_boundary_bias_vs_depth.pdf", FIGURES_DIR / "vs_bias_vs_depth.pdf")
    write_csv(RESULTS_DIR / "prior_boundary_all_methods.csv", all_rows)
    write_csv(RESULTS_DIR / "prior_boundary_metrics.csv", all_rows)
    build_summary(RESULTS_DIR / "prior_boundary_summary.md", all_rows, " ".join(sys.argv))
    shutil.copyfile(RESULTS_DIR / "prior_boundary_summary.md", RESULTS_DIR / "direct_vs_indirect_summary.md")
    print(f"[done] wrote {RESULTS_DIR / 'prior_boundary_summary.md'}")


if __name__ == "__main__":
    main()
