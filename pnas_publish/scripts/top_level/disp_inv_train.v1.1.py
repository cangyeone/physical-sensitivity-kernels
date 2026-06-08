#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conditional Rectified Flow inversion training:
    dispersion curves -> 1D velocity model

This script keeps the existing SurfaceWaveDataset/DataLoader contract:
    model_batch: [B, 4, H] = [depth, Vp, Vs, rho]
    disp_batch : [B, 3, T] = [period, Rayleigh phase velocity, Love phase velocity]
    mask_batch : [B, 3, T]

The CRF target is only [Vp, Vs, rho]. The depth grid is fixed and saved as a
buffer, then attached back to predictions as [depth, Vp, Vs, rho].
"""

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.generate_data import SurfaceWaveDataset

plt.switch_backend("Agg")


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def exists(x) -> bool:
    return x is not None


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = ConvGNAct(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_gn_groups(out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    if t.ndim == 2:
        t = t[:, 0]
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0, np.log(10000.0), half, device=t.device, dtype=t.dtype) * -1.0
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DispersionEncoder(nn.Module):
    """
    Encodes masked dispersion observations.

    Input channels:
      0: normalized period
      1: normalized Rayleigh velocity, zeroed where invalid
      2: normalized Love velocity, zeroed where invalid
      3: Rayleigh mask
      4: Love mask
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 64,
        cond_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4

        self.stem = ConvGNAct(in_channels, c1, kernel_size=7, stride=1)
        self.backbone = nn.Sequential(
            ResidualBlock(c1, c1, stride=1, kernel_size=7),
            ResidualBlock(c1, c2, stride=2, kernel_size=7),
            ResidualBlock(c2, c2, stride=1, kernel_size=5, dilation=2),
            ResidualBlock(c2, c3, stride=2, kernel_size=7),
            ResidualBlock(c3, c3, stride=1, kernel_size=5, dilation=2),
            ResidualBlock(c3, c4, stride=2, kernel_size=5),
            ResidualBlock(c4, c4, stride=1, kernel_size=5, dilation=2),
        )

        self.out_dim = cond_dim
        self.head = nn.Sequential(
            nn.Linear(c4 * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(self.stem(x))
        feat = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(feat)


class Disp2StructCRF(nn.Module):
    """
    Conditional Rectified Flow from masked dispersion curves to velocity profiles.
    """

    def __init__(
        self,
        H: int,
        T: int,
        profile_channels: int = 3,
        cond_base_channels: int = 64,
        cond_dim: int = 256,
        flow_hidden: int = 1024,
        time_dim: int = 64,
        dropout: float = 0.1,
        reference_profile: Optional[torch.Tensor] = None,
        profile_scale: Optional[torch.Tensor] = None,
        depth_grid: Optional[torch.Tensor] = None,
        period_minmax: Tuple[float, float] = (2.0, 60.0),
        disp_mean: Optional[torch.Tensor] = None,
        disp_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.H = H
        self.T = T
        self.profile_channels = profile_channels
        self.output_dim = profile_channels * H
        self.time_dim = time_dim

        if reference_profile is None:
            reference_profile = torch.zeros(profile_channels, H, dtype=torch.float32)
        if profile_scale is None:
            profile_scale = torch.ones(profile_channels, H, dtype=torch.float32)
        if depth_grid is None:
            depth_grid = torch.arange(H, dtype=torch.float32)
        if disp_mean is None:
            disp_mean = torch.zeros(2, dtype=torch.float32)
        if disp_scale is None:
            disp_scale = torch.ones(2, dtype=torch.float32)

        self.register_buffer(
            "reference_profile",
            torch.as_tensor(reference_profile, dtype=torch.float32).reshape(profile_channels, H),
        )
        self.register_buffer(
            "profile_scale",
            torch.as_tensor(profile_scale, dtype=torch.float32).reshape(profile_channels, H).clamp_min(1e-4),
        )
        self.register_buffer(
            "depth_grid",
            torch.as_tensor(depth_grid, dtype=torch.float32).reshape(H),
        )
        self.register_buffer(
            "period_minmax",
            torch.tensor(period_minmax, dtype=torch.float32).reshape(2),
        )
        self.register_buffer(
            "disp_mean",
            torch.as_tensor(disp_mean, dtype=torch.float32).reshape(2),
        )
        self.register_buffer(
            "disp_scale",
            torch.as_tensor(disp_scale, dtype=torch.float32).reshape(2).clamp_min(1e-4),
        )

        self.encoder = DispersionEncoder(
            in_channels=5,
            base_channels=cond_base_channels,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        self.flow_net = nn.Sequential(
            nn.Linear(cond_dim + self.output_dim + time_dim, flow_hidden),
            nn.LayerNorm(flow_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden, flow_hidden),
            nn.LayerNorm(flow_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden, self.output_dim),
        )

    def format_condition(self, disp: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if disp.ndim != 3 or disp.size(1) < 3:
            raise ValueError(f"Expected disp shape [B,3,T], got {tuple(disp.shape)}")
        if disp.size(-1) != self.T:
            raise ValueError(f"Expected T={self.T}, got T={disp.size(-1)}")

        periods = disp[:, 0, :]
        wave = disp[:, 1:3, :]

        if mask is None:
            wave_mask = torch.ones_like(wave)
        else:
            if mask.ndim != 3 or mask.size(1) < 3:
                raise ValueError(f"Expected mask shape [B,3,T], got {tuple(mask.shape)}")
            wave_mask = mask[:, 1:3, :].to(dtype=wave.dtype)

        pmin, pmax = self.period_minmax[0], self.period_minmax[1]
        period_norm = (periods - pmin) / (pmax - pmin).clamp_min(1e-6)
        period_norm = period_norm * 2.0 - 1.0

        wave_norm = (wave - self.disp_mean.view(1, 2, 1)) / self.disp_scale.view(1, 2, 1)
        wave_norm = wave_norm * wave_mask

        return torch.cat([period_norm.unsqueeze(1), wave_norm, wave_mask], dim=1)

    def encode(self, disp: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(self.format_condition(disp, mask))

    def profile_to_z(self, profile: torch.Tensor) -> torch.Tensor:
        return (profile - self.reference_profile.view(1, self.profile_channels, self.H)) / self.profile_scale.view(
            1, self.profile_channels, self.H
        )

    def z_to_profile(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.profile_scale.view(1, self.profile_channels, self.H) + self.reference_profile.view(
            1, self.profile_channels, self.H
        )

    def attach_depth(self, profile: torch.Tensor) -> torch.Tensor:
        if profile.ndim == 3:
            depth = self.depth_grid.view(1, 1, self.H).expand(profile.size(0), -1, -1)
            return torch.cat([depth, profile], dim=1)
        if profile.ndim == 4:
            depth = self.depth_grid.view(1, 1, 1, self.H).expand(profile.size(0), profile.size(1), -1, -1)
            return torch.cat([depth, profile], dim=2)
        raise ValueError(f"Expected profile [B,C,H] or [B,S,C,H], got {tuple(profile.shape)}")

    def flow_velocity(self, cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1).expand(x_t.size(0))
        if t.ndim > 1:
            t = t.reshape(-1)
        t_emb = sinusoidal_time_embedding(t.to(dtype=x_t.dtype), dim=self.time_dim)
        x_flat = x_t.reshape(x_t.size(0), -1)
        velocity = self.flow_net(torch.cat([cond, x_flat, t_emb], dim=1))
        return velocity.view_as(x_t)

    def forward(
        self,
        disp: torch.Tensor,
        mask: Optional[torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cond = self.encode(disp, mask)
        return {"flow_velocity": self.flow_velocity(cond, x_t, t)}

    @torch.no_grad()
    def sample(
        self,
        disp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        num_steps: int = 32,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()

        cond = self.encode(disp, mask)
        batch_size = cond.size(0)
        cond_rep = cond[:, None, :].expand(batch_size, num_samples, cond.size(1)).reshape(
            batch_size * num_samples, -1
        )
        z = torch.randn(
            batch_size * num_samples,
            self.profile_channels,
            self.H,
            device=cond.device,
            dtype=cond.dtype,
            generator=generator,
        ) * float(temperature)

        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = torch.full((batch_size * num_samples,), step * dt, device=cond.device, dtype=cond.dtype)
            velocity = self.flow_velocity(cond_rep, z, t)
            z = z + dt * velocity

        z_samples = z.view(batch_size, num_samples, self.profile_channels, self.H)
        profile_samples = self.z_to_profile(z_samples.reshape(-1, self.profile_channels, self.H)).view_as(z_samples)
        model_samples = self.attach_depth(profile_samples)

        if was_training:
            self.train()

        return {
            "z_samples": z_samples,
            "profile_samples": profile_samples,
            "model_samples": model_samples,
            "profile_mean": profile_samples.mean(dim=1),
            "profile_median": profile_samples.median(dim=1).values,
            "profile_std": profile_samples.std(dim=1, unbiased=False),
            "model_mean": self.attach_depth(profile_samples.mean(dim=1)),
            "model_median": self.attach_depth(profile_samples.median(dim=1).values),
        }

    @torch.no_grad()
    def predict(
        self,
        disp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        num_steps: int = 32,
        temperature: float = 1.0,
        reduce: str = "median",
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.sample(
            disp=disp,
            mask=mask,
            num_samples=num_samples,
            num_steps=num_steps,
            temperature=temperature,
            generator=generator,
        )
        if reduce == "median":
            profile_mu = out["profile_median"]
            model_mu = out["model_median"]
        elif reduce == "mean":
            profile_mu = out["profile_mean"]
            model_mu = out["model_mean"]
        else:
            raise ValueError("reduce must be 'median' or 'mean'")
        out["profile_mu"] = profile_mu
        out["model_mu"] = model_mu
        return out


def smooth_l1_mean(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="mean")


def slope_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    if pred.size(-1) < 2:
        return pred.new_tensor(0.0)
    return smooth_l1_mean(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1], beta=beta)


def curvature_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    if pred.size(-1) < 3:
        return pred.new_tensor(0.0)
    pred_d2 = pred[..., 2:] - 2.0 * pred[..., 1:-1] + pred[..., :-2]
    target_d2 = target[..., 2:] - 2.0 * target[..., 1:-1] + target[..., :-2]
    return smooth_l1_mean(pred_d2, target_d2, beta=beta)


def compute_total_loss(
    model: Disp2StructCRF,
    disp: torch.Tensor,
    mask: torch.Tensor,
    target_profile: torch.Tensor,
    lambda_rec: float = 0.5,
    lambda_slope: float = 0.05,
    lambda_curvature: float = 0.01,
    flow_beta: float = 0.5,
) -> Dict[str, torch.Tensor]:
    target_z = model.profile_to_z(target_profile)
    noise = torch.randn_like(target_z)
    t = torch.rand(target_z.size(0), 1, 1, device=target_z.device, dtype=target_z.dtype)

    x_t = (1.0 - t) * noise + t * target_z
    target_velocity = target_z - noise

    outputs = model(disp=disp, mask=mask, x_t=x_t, t=t.reshape(-1))
    pred_velocity = outputs["flow_velocity"]

    flow_loss = smooth_l1_mean(pred_velocity, target_velocity, beta=flow_beta)
    pred_z_1 = x_t + (1.0 - t) * pred_velocity
    rec_loss = smooth_l1_mean(pred_z_1, target_z, beta=flow_beta)
    slope = slope_loss(pred_z_1, target_z)
    curvature = curvature_loss(pred_z_1, target_z)

    total = flow_loss + lambda_rec * rec_loss + lambda_slope * slope + lambda_curvature * curvature
    return {
        "loss": total,
        "flow_loss": flow_loss,
        "rec_loss": rec_loss,
        "slope_loss": slope,
        "curvature_loss": curvature,
        "pred_profile_train": model.z_to_profile(pred_z_1).detach(),
    }


@torch.no_grad()
def profile_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    err = pred - target
    mae_ch = err.abs().mean(dim=(0, 2))
    rmse_ch = torch.sqrt(err.pow(2).mean(dim=(0, 2)))
    return {
        "mae": err.abs().mean().item(),
        "rmse": torch.sqrt(err.pow(2).mean()).item(),
        "vp_mae": mae_ch[0].item(),
        "vs_mae": mae_ch[1].item(),
        "rho_mae": mae_ch[2].item(),
        "vp_rmse": rmse_ch[0].item(),
        "vs_rmse": rmse_ch[1].item(),
        "rho_rmse": rmse_ch[2].item(),
    }


def move_to_device(batch, device: torch.device):
    return tuple(x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in batch)


@torch.no_grad()
def estimate_training_stats(
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = 64,
    profile_scale_floors: Tuple[float, float, float] = (0.05, 0.05, 0.02),
) -> Dict[str, torch.Tensor]:
    profile_sum = None
    profile_sq_sum = None
    profile_count = 0

    disp_sum = torch.zeros(2, device=device)
    disp_sq_sum = torch.zeros(2, device=device)
    disp_count = torch.zeros(2, device=device)

    period_min = None
    period_max = None
    depth_grid = None
    H = None
    T = None
    processed_batches = 0

    for batch_idx, batch in enumerate(loader):
        if exists(max_batches) and batch_idx >= max_batches:
            break

        model_batch, disp_batch, mask_batch = move_to_device(batch, device)
        model_batch = model_batch.float()
        disp_batch = disp_batch.float()
        mask_batch = mask_batch.float()

        if H is None:
            H = model_batch.size(-1)
            T = disp_batch.size(-1)
            depth_grid = model_batch[0, 0].detach().clone()
            profile_sum = torch.zeros(3, H, device=device)
            profile_sq_sum = torch.zeros(3, H, device=device)

        profile = model_batch[:, 1:4, :]
        profile_sum += profile.sum(dim=0)
        profile_sq_sum += profile.pow(2).sum(dim=0)
        profile_count += profile.size(0)

        periods = disp_batch[:, 0, :]
        cur_min = periods.min()
        cur_max = periods.max()
        period_min = cur_min if period_min is None else torch.minimum(period_min, cur_min)
        period_max = cur_max if period_max is None else torch.maximum(period_max, cur_max)

        wave = disp_batch[:, 1:3, :]
        wave_mask = mask_batch[:, 1:3, :]
        disp_sum += (wave * wave_mask).sum(dim=(0, 2))
        disp_sq_sum += (wave.pow(2) * wave_mask).sum(dim=(0, 2))
        disp_count += wave_mask.sum(dim=(0, 2))
        processed_batches += 1

    if profile_count == 0:
        raise RuntimeError("Cannot estimate stats from an empty loader.")

    reference_profile = profile_sum / float(profile_count)
    profile_var = (profile_sq_sum / float(profile_count) - reference_profile.pow(2)).clamp_min(0.0)
    profile_scale = torch.sqrt(profile_var)
    floors = torch.tensor(profile_scale_floors, device=device, dtype=profile_scale.dtype).view(3, 1)
    profile_scale = profile_scale.clamp_min(floors)

    disp_mean = disp_sum / disp_count.clamp_min(1.0)
    disp_var = (disp_sq_sum / disp_count.clamp_min(1.0) - disp_mean.pow(2)).clamp_min(0.0)
    disp_scale = torch.sqrt(disp_var).clamp_min(0.05)

    return {
        "H": torch.tensor(H),
        "T": torch.tensor(T),
        "reference_profile": reference_profile.detach().cpu(),
        "profile_scale": profile_scale.detach().cpu(),
        "depth_grid": depth_grid.detach().cpu(),
        "period_minmax": torch.stack([period_min, period_max]).detach().cpu(),
        "disp_mean": disp_mean.detach().cpu(),
        "disp_scale": disp_scale.detach().cpu(),
        "stats_batches": torch.tensor(processed_batches),
    }


def plot_profile_preview(
    depth: torch.Tensor,
    true_profile: torch.Tensor,
    pred_profile: torch.Tensor,
    save_path: str,
    title: str,
):
    depth_np = depth.detach().cpu().numpy()
    true_np = true_profile.detach().cpu().numpy()
    pred_np = pred_profile.detach().cpu().numpy()

    labels = ["Vp", "Vs", "rho"]
    xlabels = ["km/s", "km/s", "g/cm^3"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 6), sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(true_np[i], depth_np, "k-", lw=1.8, label="true")
        ax.plot(pred_np[i], depth_np, "r--", lw=1.8, label="pred")
        ax.set_xlabel(f"{labels[i]} ({xlabels[i]})")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Depth (km)")
            ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_log_line(log_path: Optional[str], message: str):
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def plot_inversion_result_and_uncertainty(
    depth: torch.Tensor,
    true_profile: torch.Tensor,
    profile_samples: torch.Tensor,
    result_path: str,
    uncertainty_path: str,
    title: str,
):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    os.makedirs(os.path.dirname(uncertainty_path), exist_ok=True)

    depth_np = depth.detach().cpu().numpy()
    true_np = true_profile.detach().cpu().numpy()
    samples_np = profile_samples.detach().cpu().numpy()

    median_np = np.median(samples_np, axis=0)
    q16_np = np.percentile(samples_np, 16.0, axis=0)
    q84_np = np.percentile(samples_np, 84.0, axis=0)
    std_np = np.std(samples_np, axis=0)

    labels = ["Vp", "Vs", "rho"]
    xlabels = ["km/s", "km/s", "g/cm^3"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    for i, ax in enumerate(axes):
        ax.fill_betweenx(depth_np, q16_np[i], q84_np[i], color="tab:red", alpha=0.20, label="16-84%")
        ax.plot(true_np[i], depth_np, "k-", lw=1.8, label="true")
        ax.plot(median_np[i], depth_np, color="tab:red", linestyle="--", lw=1.8, label="median")
        ax.set_xlabel(f"{labels[i]} ({xlabels[i]})")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Depth (km)")
            ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(std_np[i], depth_np, color="tab:blue", lw=1.8)
        ax.fill_betweenx(depth_np, 0.0, std_np[i], color="tab:blue", alpha=0.16)
        ax.set_xlabel(f"std({labels[i]}) ({xlabels[i]})")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Depth (km)")
    axes[0].invert_yaxis()
    fig.suptitle(title + " uncertainty")
    fig.tight_layout()
    fig.savefig(uncertainty_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one_epoch(
    model: Disp2StructCRF,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 5.0,
    lambda_rec: float = 0.5,
    lambda_slope: float = 0.05,
    lambda_curvature: float = 0.01,
    flow_beta: float = 0.5,
    sample_metrics: bool = False,
    eval_num_samples: int = 4,
    eval_num_steps: int = 24,
    global_step: int = 0,
    epoch: int = 0,
    log_every_steps: int = 0,
    log_path: Optional[str] = None,
    plot_every_steps: int = 0,
    plot_num_samples: int = 16,
    fig_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "flow_loss": 0.0,
        "rec_loss": 0.0,
        "slope_loss": 0.0,
        "curvature_loss": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "vp_mae": 0.0,
        "vs_mae": 0.0,
        "rho_mae": 0.0,
        "sample_mae": 0.0,
        "sample_rmse": 0.0,
        "sample_vp_mae": 0.0,
        "sample_vs_mae": 0.0,
        "sample_rho_mae": 0.0,
    }
    n_batches = 0
    n_sample_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        model_batch, disp_batch, mask_batch = move_to_device(batch, device)
        target_profile = model_batch[:, 1:4, :].float()
        disp_batch = disp_batch.float()
        mask_batch = mask_batch.float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                loss_dict = compute_total_loss(
                    model=model,
                    disp=disp_batch,
                    mask=mask_batch,
                    target_profile=target_profile,
                    lambda_rec=lambda_rec,
                    lambda_slope=lambda_slope,
                    lambda_curvature=lambda_curvature,
                    flow_beta=flow_beta,
                )
                loss = loss_dict["loss"]

            if is_train:
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if exists(grad_clip):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if exists(grad_clip):
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                global_step += 1

        metrics = profile_metrics(loss_dict["pred_profile_train"], target_profile)
        totals["mae"] += metrics["mae"]
        totals["rmse"] += metrics["rmse"]
        totals["vp_mae"] += metrics["vp_mae"]
        totals["vs_mae"] += metrics["vs_mae"]
        totals["rho_mae"] += metrics["rho_mae"]

        if is_train and log_every_steps > 0 and global_step > 0 and global_step % log_every_steps == 0:
            step_msg = (
                f"[Step {global_step:08d}] epoch={epoch:03d} batch={batch_idx:05d}/{len(loader):05d} "
                f"loss={loss_dict['loss'].item():.6f} flow={loss_dict['flow_loss'].item():.6f} "
                f"rec={loss_dict['rec_loss'].item():.6f} slope={loss_dict['slope_loss'].item():.6f} "
                f"curv={loss_dict['curvature_loss'].item():.6f} mae={metrics['mae']:.5f} "
                f"vp_mae={metrics['vp_mae']:.5f} vs_mae={metrics['vs_mae']:.5f} "
                f"rho_mae={metrics['rho_mae']:.5f}"
            )
            print(step_msg)
            write_log_line(log_path, step_msg)

        if sample_metrics:
            pred = model.predict(
                disp=disp_batch,
                mask=mask_batch,
                num_samples=eval_num_samples,
                num_steps=eval_num_steps,
                reduce="median",
            )["profile_mu"]
            sample = profile_metrics(pred, target_profile)
            totals["sample_mae"] += sample["mae"]
            totals["sample_rmse"] += sample["rmse"]
            totals["sample_vp_mae"] += sample["vp_mae"]
            totals["sample_vs_mae"] += sample["vs_mae"]
            totals["sample_rho_mae"] += sample["rho_mae"]
            n_sample_batches += 1

        if (
            is_train
            and plot_every_steps > 0
            and fig_dir is not None
            and global_step > 0
            and global_step % plot_every_steps == 0
        ):
            sample_out = model.sample(
                disp=disp_batch[:1],
                mask=mask_batch[:1],
                num_samples=max(2, plot_num_samples),
                num_steps=eval_num_steps,
            )
            plot_inversion_result_and_uncertainty(
                depth=model.depth_grid,
                true_profile=target_profile[0],
                profile_samples=sample_out["profile_samples"][0],
                result_path=os.path.join(fig_dir, "profiles", f"inv_profile_step_{global_step:08d}.pdf"),
                uncertainty_path=os.path.join(fig_dir, "uncertainty", f"inv_uncertainty_step_{global_step:08d}.pdf"),
                title=f"Step={global_step}, loss={loss.item():.4f}",
            )

        for key in ["loss", "flow_loss", "rec_loss", "slope_loss", "curvature_loss"]:
            totals[key] += loss_dict[key].item()
        n_batches += 1

    denom = max(n_batches, 1)
    out = {k: v / denom for k, v in totals.items()}
    if n_sample_batches > 0:
        for key in ["sample_mae", "sample_rmse", "sample_vp_mae", "sample_vs_mae", "sample_rho_mae"]:
            out[key] = totals[key] / n_sample_batches
    return out, global_step


@dataclass
class TrainConfig:
    n_train: int = 100_000
    n_val: int = 2_048
    z_max_km: float = 150.0
    z_max_num: int = 256
    dz_km: float = 0.5
    tectonic_type: Optional[str] = None

    batch_size: int = 64
    num_workers: int = 36
    seed: int = 2026
    stats_batches: int = 64

    save_dir: str = "ckpt/disp2struct_crf.v1.1"
    fig_dir: str = "tfig_inv"
    resume: bool = True

    cond_base_channels: int = 64
    cond_dim: int = 256
    flow_hidden: int = 1024
    time_dim: int = 64
    dropout: float = 0.1

    epochs: int = 200
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    use_amp: bool = True

    lambda_rec: float = 0.5
    lambda_slope: float = 0.05
    lambda_curvature: float = 0.01
    flow_beta: float = 0.5

    val_every: int = 1
    eval_num_samples: int = 4
    eval_num_steps: int = 24
    log_every_steps: int = 100
    log_filename: str = "train.log"
    plot_every_steps: int = 100
    plot_num_samples: int = 16
    device: str = default_device()


def save_checkpoint(
    path: str,
    model: Disp2StructCRF,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: TrainConfig,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": asdict(cfg),
            "model_name": "Disp2StructCRF",
            "reference_profile": model.reference_profile.detach().cpu(),
            "profile_scale": model.profile_scale.detach().cpu(),
            "depth_grid": model.depth_grid.detach().cpu(),
            "period_minmax": model.period_minmax.detach().cpu(),
            "disp_mean": model.disp_mean.detach().cpu(),
            "disp_scale": model.disp_scale.detach().cpu(),
        },
        path,
    )


def maybe_resume(
    ckpt_path: str,
    model: Disp2StructCRF,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
) -> Tuple[int, int, float]:
    if not os.path.exists(ckpt_path):
        return 1, 0, float("inf")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if isinstance(ckpt, dict):
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    print(f"Loaded checkpoint: {ckpt_path}")
    return start_epoch, global_step, best_val_loss


def make_loader(dataset: SurfaceWaveDataset, cfg: TrainConfig, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )


def train_disp2struct_crf(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
) -> Disp2StructCRF:
    if cfg is None:
        cfg = TrainConfig()

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, cfg.log_filename)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print("Device:", device)

    print(f"Estimating normalization stats from {cfg.stats_batches} training batches...")
    stats = estimate_training_stats(
        train_loader,
        device=device,
        max_batches=cfg.stats_batches,
    )
    H = int(stats["H"].item())
    T = int(stats["T"].item())
    period_minmax = tuple(float(x) for x in stats["period_minmax"].tolist())

    model = Disp2StructCRF(
        H=H,
        T=T,
        profile_channels=3,
        cond_base_channels=cfg.cond_base_channels,
        cond_dim=cfg.cond_dim,
        flow_hidden=cfg.flow_hidden,
        time_dim=cfg.time_dim,
        dropout=cfg.dropout,
        reference_profile=stats["reference_profile"],
        profile_scale=stats["profile_scale"],
        depth_grid=stats["depth_grid"],
        period_minmax=period_minmax,
        disp_mean=stats["disp_mean"],
        disp_scale=stats["disp_scale"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    latest_path = os.path.join(cfg.save_dir, "latest.pt")
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    if cfg.resume:
        start_epoch, global_step, best_val_loss = maybe_resume(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    print("========== Disp2Struct CRF Training Start ==========")
    print(f"H/T           : {H}/{T}")
    print(f"period range  : {period_minmax[0]:.2f} - {period_minmax[1]:.2f} s")
    print(f"params        : {sum(p.numel() for p in model.parameters())}")
    print(f"train batches : {len(train_loader)}")
    print(f"val batches   : {len(val_loader) if val_loader is not None else 0}")
    print(f"save_dir      : {cfg.save_dir}")
    write_log_line(
        log_path,
        (
            f"========== Disp2Struct CRF Training Start | start_epoch={start_epoch} "
            f"global_step={global_step} H={H} T={T} params={sum(p.numel() for p in model.parameters())} =========="
        ),
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_stats, global_step = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=cfg.use_amp,
            grad_clip=cfg.grad_clip,
            lambda_rec=cfg.lambda_rec,
            lambda_slope=cfg.lambda_slope,
            lambda_curvature=cfg.lambda_curvature,
            flow_beta=cfg.flow_beta,
            sample_metrics=False,
            eval_num_samples=cfg.eval_num_samples,
            eval_num_steps=cfg.eval_num_steps,
            global_step=global_step,
            epoch=epoch,
            log_every_steps=cfg.log_every_steps,
            log_path=log_path,
            plot_every_steps=cfg.plot_every_steps,
            plot_num_samples=cfg.plot_num_samples,
            fig_dir=cfg.fig_dir,
        )

        val_stats = None
        if val_loader is not None and epoch % cfg.val_every == 0:
            val_stats, _ = run_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=cfg.use_amp,
                grad_clip=None,
                lambda_rec=cfg.lambda_rec,
                lambda_slope=cfg.lambda_slope,
                lambda_curvature=cfg.lambda_curvature,
                flow_beta=cfg.flow_beta,
                sample_metrics=True,
                eval_num_samples=cfg.eval_num_samples,
                eval_num_steps=cfg.eval_num_steps,
                global_step=global_step,
                epoch=epoch,
                log_every_steps=0,
                log_path=None,
                plot_every_steps=0,
                plot_num_samples=cfg.plot_num_samples,
                fig_dir=None,
            )

        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        msg = (
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] time={dt:.1f}s lr={lr_now:.2e} | "
            f"train: loss={train_stats['loss']:.5f}, flow={train_stats['flow_loss']:.5f}, "
            f"rec={train_stats['rec_loss']:.5f}, mae={train_stats['mae']:.4f}, "
            f"vp={train_stats['vp_mae']:.4f}, vs={train_stats['vs_mae']:.4f}, "
            f"rho={train_stats['rho_mae']:.4f}"
        )
        if val_stats is not None:
            msg += (
                f" | val: loss={val_stats['loss']:.5f}, flow={val_stats['flow_loss']:.5f}, "
                f"sample_mae={val_stats['sample_mae']:.4f}, sample_vp={val_stats['sample_vp_mae']:.4f}, "
                f"sample_vs={val_stats['sample_vs_mae']:.4f}, sample_rho={val_stats['sample_rho_mae']:.4f}"
            )
        print(msg)
        write_log_line(log_path, msg)

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        score = val_stats["loss"] if val_stats is not None else train_stats["loss"]
        if score < best_val_loss:
            best_val_loss = score
            best_path = os.path.join(cfg.save_dir, "best.pt")
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            print(f"  -> Best model saved to {best_path}")

    print("========== Disp2Struct CRF Training Done ==========")
    print(f"Best score: {best_val_loss:.6f}")
    return model


def main():
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    train_ds = SurfaceWaveDataset(
        n_samples=cfg.n_train,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        tectonic_type=cfg.tectonic_type,
        seed=cfg.seed,
    )
    val_ds = SurfaceWaveDataset(
        n_samples=cfg.n_val,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        tectonic_type=cfg.tectonic_type,
        seed=cfg.seed + 1_000_000,
    )

    train_loader = make_loader(train_ds, cfg=cfg, shuffle=True, device=device)
    val_loader = make_loader(val_ds, cfg=cfg, shuffle=False, device=device)

    train_disp2struct_crf(train_loader, val_loader=val_loader, cfg=cfg)


if __name__ == "__main__":
    main()
