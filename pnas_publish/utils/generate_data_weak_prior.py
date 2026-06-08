#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weak-prior ensemble generator for global 1D Vs/Vp/rho models.

Design philosophy
-----------------
This generator does NOT impose tectonic classes, explicit crust/mantle layering,
or a prescribed LVZ. Instead, it constructs broad, physically plausible 1D models
using:
  1) weak depth-dependent bounds,
  2) random control points (knots),
  3) smooth interpolation,
  4) correlated random perturbations.

The goal is to reduce prior imprint when training neural surrogates for
surface-wave forward modeling and sensitivity-kernel analysis.

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# Utilities
# ============================================================

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    if window <= 1:
        return x.copy()
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(xpad, kernel, mode="valid")


def _gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Simple Gaussian smoothing implemented with numpy only."""
    if sigma_samples <= 0:
        return x.copy()

    half = max(3, int(math.ceil(4.0 * sigma_samples)))
    grid = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_samples) ** 2)
    kernel /= kernel.sum()

    xpad = np.pad(x, (half, half), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    max_tries: int = 100,
) -> float:
    """Sample from a truncated normal by rejection."""
    for _ in range(max_tries):
        x = rng.normal(mean, std)
        if low <= x <= high:
            return float(x)
    return float(np.clip(mean, low, high))


def _piecewise_linear_depth_bounds(depth_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weak, broad physical bounds for Vs(z), in km/s.
    These are intentionally wide and do NOT encode explicit crust/mantle templates.

    Control points chosen to allow broad global variability while preventing
    pathological profiles.
    """
    z_ctrl = np.array([0.0, 3.0, 8.0, 20.0, 40.0, 80.0, 150.0], dtype=float)

    # Very broad lower and upper envelopes
    vs_min_ctrl = np.array([0.15, 0.25, 1.5, 2.5, 3.0, 3.6, 3.8], dtype=float)
    vs_max_ctrl = np.array([4.0,  4.2,  4.5, 4.8, 5.1, 5.4, 5.6], dtype=float)

    vs_min = np.interp(depth_km, z_ctrl, vs_min_ctrl)
    vs_max = np.interp(depth_km, z_ctrl, vs_max_ctrl)
    return vs_min, vs_max


def _depth_dependent_sigma(depth_km: np.ndarray) -> np.ndarray:
    """
    Std amplitude of random perturbations for Vs(z).
    Larger near crust/mantle transition depths and moderate in upper mantle,
    but still broad and weakly prescribed.
    """
    z_ctrl = np.array([0.0, 5.0, 15.0, 40.0, 80.0, 150.0], dtype=float)
    sig_ctrl = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10], dtype=float)
    return np.interp(depth_km, z_ctrl, sig_ctrl)


def _depth_dependent_corrlen_km(depth_km: np.ndarray) -> np.ndarray:
    """
    Correlation length scale in km.
    Shorter at shallow depth, longer at greater depth.
    """
    z_ctrl = np.array([0.0, 5.0, 15.0, 40.0, 80.0, 150.0], dtype=float)
    L_ctrl = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 18.0], dtype=float)
    return np.interp(depth_km, z_ctrl, L_ctrl)


def brocher_vp_from_vs(vs: np.ndarray) -> np.ndarray:
    """
    Brocher (2005)-style empirical Vp(Vs), km/s.
    Input Vs in km/s, output Vp in km/s.

    Polynomial is widely used in crustal seismology. For very low Vs near surface,
    clip to avoid pathological values.
    """
    vs_clip = np.clip(vs, 0.2, 5.5)
    vp = (
        0.9409
        + 2.0947 * vs_clip
        - 0.8206 * vs_clip**2
        + 0.2683 * vs_clip**3
        - 0.0251 * vs_clip**4
    )
    return vp


def brocher_rho_from_vp(vp: np.ndarray) -> np.ndarray:
    """
    Brocher (2005)-style rho(Vp), g/cm^3.
    Input Vp in km/s, output density in g/cm^3.
    """
    vp_clip = np.clip(vp, 1.0, 8.5)
    rho = (
        1.6612 * vp_clip
        - 0.4721 * vp_clip**2
        + 0.0671 * vp_clip**3
        - 0.0043 * vp_clip**4
        + 0.000106 * vp_clip**5
    )
    return rho


def _vpvs_ratio_profile(depth_km: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Weakly depth-dependent Vp/Vs ratio with smooth random variability.
    This avoids strong layer-dependent parameterization.
    """
    z_ctrl = np.array([0.0, 5.0, 20.0, 60.0, 150.0], dtype=float)
    mean_ctrl = np.array([1.90, 1.82, 1.76, 1.80, 1.84], dtype=float)

    mean_profile = np.interp(depth_km, z_ctrl, mean_ctrl)

    # Smooth random perturbation
    noise = rng.normal(0.0, 0.03, size=depth_km.shape)
    noise = _gaussian_smooth(noise, sigma_samples=6.0)

    vpvs = mean_profile + noise
    vpvs = np.clip(vpvs, 1.65, 2.00)
    return vpvs


def _make_random_knots(
    z_max_km: float,
    rng: np.random.Generator,
    n_knots_range: Tuple[int, int],
) -> np.ndarray:
    """
    Random knot depths for spline/background construction.
    More density near shallow depths but still broad.
    """
    n_knots = int(rng.integers(n_knots_range[0], n_knots_range[1] + 1))

    # Draw from a mixture: some shallow-biased, some uniform
    u1 = rng.uniform(0.0, 1.0, size=n_knots * 2)
    shallow = (u1**1.8) * z_max_km
    uniform = rng.uniform(0.0, z_max_km, size=n_knots * 2)
    mix_mask = rng.random(n_knots * 2) < 0.6
    z_raw = np.where(mix_mask, shallow, uniform)

    z = np.unique(np.round(z_raw, 2))
    z = z[(z > 0.5) & (z < z_max_km - 0.5)]

    if len(z) < n_knots:
        extra = np.linspace(2.0, z_max_km - 2.0, n_knots)
        z = np.unique(np.concatenate([z, extra]))

    z = np.sort(z)[:n_knots]
    z = np.concatenate([[0.0], z, [z_max_km]])
    return z


def _sample_vs_background_from_knots(
    knot_depths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample broad Vs values at knot depths under weak bounds.
    Includes only a weak encouragement of overall deepening, not a hard layer template.
    """
    vs_min, vs_max = _piecewise_linear_depth_bounds(knot_depths)

    # Weak background trend: deeper tends to be faster, but not strictly monotonic
    trend = np.interp(
        knot_depths,
        [0.0, 10.0, 30.0, 80.0, knot_depths[-1]],
        [2.0, 2.8, 3.6, 4.3, 4.7],
    )

    # Depth-dependent allowable spread
    spread = np.interp(
        knot_depths,
        [0.0, 5.0, 20.0, 50.0, 150.0],
        [1.0, 0.9, 0.7, 0.6, 0.5],
    )

    vs_knots = trend + rng.normal(0.0, spread, size=knot_depths.shape)

    # Allow some shallow very low values occasionally
    if rng.random() < 0.35:
        shallow_mask = knot_depths < 5.0
        vs_knots[shallow_mask] -= rng.uniform(0.3, 1.2)

    # Allow broad local low-velocity or high-velocity tendencies naturally via knots
    if rng.random() < 0.5:
        center = rng.uniform(10.0, knot_depths[-1] - 10.0)
        width = rng.uniform(8.0, 35.0)
        amp = rng.uniform(-0.5, 0.5)
        vs_knots += amp * np.exp(-0.5 * ((knot_depths - center) / width) ** 2)

    vs_knots = np.clip(vs_knots, vs_min, vs_max)
    return vs_knots


def _interpolate_profile(
    z_knots: np.ndarray,
    y_knots: np.ndarray,
    depth_km: np.ndarray,
) -> np.ndarray:
    """
    Smooth monotone-preserving interpolation if scipy is available; otherwise linear.
    """
    if HAS_SCIPY and len(z_knots) >= 3:
        f = PchipInterpolator(z_knots, y_knots, extrapolate=True)
        y = f(depth_km)
    else:
        y = np.interp(depth_km, z_knots, y_knots)
    return y


def _add_correlated_perturbations(
    base_vs: np.ndarray,
    depth_km: np.ndarray,
    dz_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add weak-to-moderate correlated perturbations without prescribing specific structures.
    """
    sig = _depth_dependent_sigma(depth_km)
    L_km = _depth_dependent_corrlen_km(depth_km)

    # Use a few band-limited random fields with different scales
    n = len(depth_km)
    noise1 = rng.normal(0.0, 1.0, size=n)
    noise2 = rng.normal(0.0, 1.0, size=n)
    noise3 = rng.normal(0.0, 1.0, size=n)

    # Smooth at different effective scales
    s1 = _gaussian_smooth(noise1, sigma_samples=max(1.0, 4.0 / dz_km))
    s2 = _gaussian_smooth(noise2, sigma_samples=max(1.0, 10.0 / dz_km))
    s3 = _gaussian_smooth(noise3, sigma_samples=max(1.0, 20.0 / dz_km))

    mixed = 0.50 * s1 + 0.35 * s2 + 0.15 * s3
    mixed = mixed / (np.std(mixed) + 1e-8)

    # Depth-dependent amplitude
    pert = sig * mixed

    # Small number of smooth local anomalies, no fixed preferred depth
    n_lumps = int(rng.integers(0, 4))
    for _ in range(n_lumps):
        center = rng.uniform(0.0, depth_km[-1])
        width = rng.uniform(5.0, 30.0)
        amp = rng.uniform(-0.35, 0.35)
        pert += amp * np.exp(-0.5 * ((depth_km - center) / width) ** 2)

    out = base_vs + pert
    return out


def _enforce_weak_physical_consistency(
    vs: np.ndarray,
    depth_km: np.ndarray,
    dz_km: float,
) -> np.ndarray:
    """
    Weak cleanup:
      - clip to broad physical bounds
      - smooth slightly
      - limit pathological local zig-zag
    This is intentionally weak, not a hard geologic template.
    """
    vs_min, vs_max = _piecewise_linear_depth_bounds(depth_km)
    vs = np.clip(vs, vs_min, vs_max)

    # Light smoothing
    vs = _gaussian_smooth(vs, sigma_samples=max(1.0, 2.0 / dz_km))

    # Limit extremely sharp point-to-point jumps, but allow local reversals
    max_jump = np.interp(depth_km[:-1], [0.0, 20.0, 80.0, 150.0], [0.45, 0.30, 0.18, 0.12])
    out = vs.copy()
    for i in range(1, len(out)):
        dv = out[i] - out[i - 1]
        if dv > max_jump[i - 1]:
            out[i] = out[i - 1] + max_jump[i - 1]
        elif dv < -max_jump[i - 1]:
            out[i] = out[i - 1] - max_jump[i - 1]

    out = np.clip(out, vs_min, vs_max)
    return out


# ============================================================
# Main public API
# ============================================================

@dataclass
class WeakPriorModelMeta:
    z_max_km: float
    dz_km: float
    n_knots: int
    knot_depths_km: np.ndarray
    knot_vs_km_s: np.ndarray
    has_scipy: bool
    notes: str


def sample_weak_prior_1d_model(
    z_max_km: float = 150.0,
    dz_km: float = 0.5,
    n_knots_range: Tuple[int, int] = (6, 12),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate a weak-prior 1D Earth model ensemble sample.

    Returns
    -------
    depth_km : (N,) ndarray
    vs       : (N,) ndarray, km/s
    vp       : (N,) ndarray, km/s
    rho      : (N,) ndarray, g/cm^3
    meta     : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    depth_km = np.arange(0.0, z_max_km + dz_km, dz_km)

    # 1) Random smooth background from sparse control points
    z_knots = _make_random_knots(z_max_km, rng, n_knots_range)
    vs_knots = _sample_vs_background_from_knots(z_knots, rng)
    base_vs = _interpolate_profile(z_knots, vs_knots, depth_km)

    # 2) Add correlated perturbations
    vs = _add_correlated_perturbations(base_vs, depth_km, dz_km, rng)

    # 3) Weak cleanup only
    vs = _enforce_weak_physical_consistency(vs, depth_km, dz_km)

    # 4) Vp/Vs profile with weak random depth dependence
    vpvs = _vpvs_ratio_profile(depth_km, rng)
    vp = vs * vpvs

    # Optional weak smoothing to keep Vp realistic
    vp = _gaussian_smooth(vp, sigma_samples=max(1.0, 2.0 / dz_km))
    vp = np.clip(vp, 1.0, 12.5)

    # 5) Density
    rho = brocher_rho_from_vp(vp)
    rho = np.clip(rho, 1.0, 3.8)

    meta_obj = WeakPriorModelMeta(
        z_max_km=float(z_max_km),
        dz_km=float(dz_km),
        n_knots=int(len(z_knots)),
        knot_depths_km=z_knots.copy(),
        knot_vs_km_s=vs_knots.copy(),
        has_scipy=HAS_SCIPY,
        notes=(
            "Weak-prior model: no explicit tectonic class, no prescribed Moho, "
            "no explicit LVZ, broad depth-dependent bounds only."
        ),
    )

    meta = {
        "z_max_km": meta_obj.z_max_km,
        "dz_km": meta_obj.dz_km,
        "n_knots": meta_obj.n_knots,
        "knot_depths_km": meta_obj.knot_depths_km,
        "knot_vs_km_s": meta_obj.knot_vs_km_s,
        "has_scipy": meta_obj.has_scipy,
        "notes": meta_obj.notes,
    }
    return depth_km, vs, vp, rho, meta


# ============================================================
# Dataset helper
# ============================================================

def generate_weak_prior_dataset(
    n_models: int,
    z_max_km: float = 150.0,
    dz_km: float = 0.5,
    n_knots_range: Tuple[int, int] = (6, 12),
    seed: int = 1234,
) -> Dict[str, np.ndarray]:
    """
    Generate a dataset of weak-prior 1D models.

    Returns a dict of stacked arrays:
      depth_km : (N,)
      vs       : (M, N)
      vp       : (M, N)
      rho      : (M, N)
    """
    rng = np.random.default_rng(seed)

    depth_km_ref = None
    vs_list = []
    vp_list = []
    rho_list = []

    for _ in range(n_models):
        depth_km, vs, vp, rho, _ = sample_weak_prior_1d_model(
            z_max_km=z_max_km,
            dz_km=dz_km,
            n_knots_range=n_knots_range,
            rng=rng,
        )
        if depth_km_ref is None:
            depth_km_ref = depth_km
        vs_list.append(vs)
        vp_list.append(vp)
        rho_list.append(rho)

    return {
        "depth_km": depth_km_ref,
        "vs": np.stack(vs_list, axis=0),
        "vp": np.stack(vp_list, axis=0),
        "rho": np.stack(rho_list, axis=0),
    }


# ============================================================
# Quick diagnostics
# ============================================================

def summarize_dataset(ds: Dict[str, np.ndarray]) -> None:
    """Print simple summary statistics."""
    depth_km = ds["depth_km"]
    vs = ds["vs"]
    vp = ds["vp"]
    rho = ds["rho"]

    print("=== Weak-prior dataset summary ===")
    print(f"n_models = {vs.shape[0]}")
    print(f"n_depth  = {vs.shape[1]}")
    print(f"z_max_km = {depth_km[-1]:.1f}")
    print(f"dz_km    = {depth_km[1] - depth_km[0]:.3f}")
    print()
    print(f"Vs range : {vs.min():.3f} - {vs.max():.3f} km/s")
    print(f"Vp range : {vp.min():.3f} - {vp.max():.3f} km/s")
    print(f"Rho range: {rho.min():.3f} - {rho.max():.3f} g/cm^3")
    print()

    for z0 in [0, 5, 10, 20, 40, 70, 100, 150]:
        idx = np.argmin(np.abs(depth_km - z0))
        print(
            f"Depth {depth_km[idx]:6.1f} km | "
            f"Vs mean={vs[:, idx].mean():.3f}, std={vs[:, idx].std():.3f} | "
            f"Vp mean={vp[:, idx].mean():.3f}, std={vp[:, idx].std():.3f}"
        )


# ============================================================
# Optional plotting
# ============================================================

def demo_plot(n_show: int = 12, seed: int = 1234) -> None:
    """Quick visualization for manual inspection."""
    import matplotlib.pyplot as plt

    ds = generate_weak_prior_dataset(n_models=n_show, seed=seed)
    depth_km = ds["depth_km"]
    vs = ds["vs"]
    vp = ds["vp"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    for i in range(min(n_show, vs.shape[0])):
        axes[0].plot(vs[i], depth_km, alpha=0.8, lw=1.2)
        axes[1].plot(vp[i], depth_km, alpha=0.8, lw=1.2)

    axes[0].invert_yaxis()
    axes[0].set_xlabel("Vs (km/s)")
    axes[1].set_xlabel("Vp (km/s)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title("Weak-prior ensemble: Vs")
    axes[1].set_title("Weak-prior ensemble: Vp")
    plt.tight_layout()
    plt.show()


def convert_depth_profile_to_layers(depth_km, vp, vs, rho):
    """
    将节点形式的 (depth_km, vp, vs, rho) 转换为层模型
    返回:
        h (km)   - 每一层的厚度
        vp_l, vs_l, rho_l - 每层的参数
    """
    h = np.diff(depth_km)        # 每层厚度
    vp_l = vp[:-1]
    vs_l = vs[:-1]
    rho_l = rho[:-1]
    return h, vp_l, vs_l, rho_l

def profile_to_velocity_model(depth_km, vp, vs, rho):
    """
    将节点形式的剖面 (depth_km, vp, vs, rho)
    转换为 disba 所需的 velocity_model:
        [thickness, Vp, Vs, density] (每行一层)

    这里简单地把相邻两个深度点之间当作一层。
    最后一层的厚度就用最后一个间距近似。
    """
    depth_km = np.asarray(depth_km)
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    rho = np.asarray(rho)

    # 每层厚度 = 相邻深度差
    thick = np.diff(depth_km)
    # 给最底层再加一个厚度（这里取和倒数第二层一样）
    last_thick = thick[-1] if len(thick) > 0 else 5.0
    thick = np.concatenate([thick, [last_thick]])

    velocity_model = np.column_stack([thick, vp, vs, rho])
    return velocity_model

import numpy as np

def profile_to_velocity_model_v2(depth_km, vp, vs, rho):
    depth_km = np.asarray(depth_km)
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    rho = np.asarray(rho)

    thick = np.diff(depth_km)
    if len(thick) == 0:
        raise ValueError("depth_km 至少需要 2 个点")

    # 最底层再追加一个厚度，当作“半空间近似”
    last_thick = thick[-1]
    thick = np.concatenate([thick, [last_thick]])

    velocity_model = np.column_stack([thick, vp, vs, rho])
    return velocity_model


from disba import PhaseDispersion

def compute_phase_dispersion(
    depth_km,
    vp,
    vs,
    rho,
    periods=None,
    modes=(0,),        # 要哪些阶：0=基阶
    wave="rayleigh",   # "rayleigh" or "love"
):
    """
    使用 disba.PhaseDispersion 计算相速度频散。

    返回：
      result = { mode: namedtuple(period, velocity, mode, wave, type) }
    """
    if periods is None:
        periods = np.logspace(np.log10(5.0), np.log10(100.0), 40)

    velocity_model = profile_to_velocity_model(depth_km, vp, vs, rho)
    # thickness, Vp, Vs, rho
    pd = PhaseDispersion(*velocity_model.T)

    out = {}
    for m in modes:
        out[m] = pd(periods, mode=m, wave=wave)

    return out


def plot_dispersion(periods, c, U, title="Dispersion"):
    plt.figure(figsize=(5,6))
    plt.plot(periods, c, "b-", lw=2, label="Phase velocity")
    plt.plot(periods, U, "r--", lw=2, label="Group velocity")
    plt.xlabel("Period (s)")
    plt.ylabel("Velocity (km/s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_phase_dispersion(r0, l0=None, title="Phase velocity dispersion"):
    plt.figure(figsize=(5,6))
    plt.plot(r0.period, r0.velocity, "b-", lw=2, label="Rayleigh 0")
    if l0 is not None:
        plt.plot(l0.period, l0.velocity, "r--", lw=2, label="Love 0")
    plt.xlabel("Period (s)")
    plt.ylabel("Phase velocity (km/s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()



import torch
from torch.utils.data import Dataset

# disba 的异常类型（不同版本位置略有差异，这样写最稳）
try:
    from disba import DispersionError
except Exception:
    # 某些版本把错误藏在 _exception 里
    try:
        from disba._exception import DispersionError
    except Exception:
        DispersionError = Exception  # 实在不行就退化成普通 Exception


class SurfaceWaveDataset(Dataset):
    """
    使用 sample_global_1d_model + compute_phase_dispersion
    动态生成训练样本的 Dataset。

    每个样本：
      - model: [4, H]  (depth, vp, vs, rho)
      - disp:  [3, T]  (period, c_Rayleigh, c_Love)
    """

    def __init__(
        self,
        n_samples: int,
        z_max_km: float = 150.0,
        z_max_num: int = 250, 
        dz_km: float = 0.5,
        periods: np.ndarray | None = None,
        tectonic_type: str | None = None,
        max_tries: int = 10,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.z_max_km = z_max_km
        self.dz_km = dz_km
        self.tectonic_type = tectonic_type
        self.max_tries = max_tries
        self.z_max_num = z_max_num
        # 周期：2–60 s，步长 1 s
        if periods is None:
            self.periods = np.arange(2.0, 61.0, 1.0, dtype=float)
        else:
            self.periods = np.asarray(periods, dtype=float)

        # 独立 RNG（注意：这里只存种子，真正用的时候重新建 Generator，避免 DataLoader 多进程冲突）
        self._seed = seed
        self._base_rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def _get_rng(self, idx: int) -> np.random.Generator:
        """
        为了在 DataLoader shuffle / 多进程下还能可复现，
        用全局 seed + idx 再生成一个独立 rng。
        """
        if self._seed is None:
            # 用户没传 seed，就用一个“漂移的” rng
            return self._base_rng
        # 有 seed，就派生一个新的
        return np.random.default_rng(self._seed + idx)

    def _one_sample(self, rng: np.random.Generator):
        """
        生成一个样本（单次尝试）：
        返回:
          model: torch.float32, [4, H]
          disp:  torch.float32, [3, T]
        这里不做异常捕获，异常在 __getitem__ 里处理。
        """
        # 1) 生成速度模型
        depth_km, vs, vp, rho, meta = sample_weak_prior_1d_model(
            z_max_km=self.z_max_km,
            dz_km=self.dz_km,
            rng=rng,
        )
        #print(depth_km)
        depth_km, vp, vs, rho = depth_km[:self.z_max_num], vp[:self.z_max_num], vs[:self.z_max_num], rho[:self.z_max_num]

        # 2) 计算 Rayleigh 基阶
        rayleigh = compute_phase_dispersion(
            depth_km, vp, vs, rho,
            periods=self.periods,
            modes=(0,),
            wave="rayleigh",
        )
        r0 = rayleigh[0]   # namedtuple(period, velocity, mode, wave, type)

        # 3) 计算 Love 基阶
        love = compute_phase_dispersion(
            depth_km, vp, vs, rho,
            periods=self.periods,
            modes=(0,),
            wave="love",
        )
        l0 = love[0]

        # 4) 组装输入 / 输出张量
        depth_km = depth_km.astype(np.float32)
        vp = vp.astype(np.float32)
        vs = vs.astype(np.float32)
        rho = rho.astype(np.float32)

        T = r0.period.astype(np.float32)
        c_R = r0.velocity.astype(np.float32)
        c_L = l0.velocity.astype(np.float32)

        # [4, H] : depth, vp, vs, rho
        model = np.stack([depth_km, vp, vs, rho], axis=0)
        model = torch.from_numpy(model)  # float32

        # [3, T] : period, Rayleigh, Love
        disp = np.stack([T, c_R, c_L], axis=0)
        disp = torch.from_numpy(disp)    # float32



        # ============================================================
        # 生成 mask（[3, T]），和 disp 相乘后把“没有值”的地方置零
        # ============================================================
        nT = disp.shape[1]

        # ---------------------------------------------------------
        # 1) 随机确定频散有效长度 min_len（占比 20%-80%）
        # ---------------------------------------------------------
        p = rng.uniform(0.2, 0.8)              # 随机占比
        min_len = max(3, int(p * nT))          # 至少 3 个点，避免太短

        # 随机起点 bidx
        bidx = int(rng.integers(0, nT - min_len))
        # 随机终点 eidx
        eidx = int(rng.integers(bidx + min_len, nT))  # eidx 不包含

        # ---------------------------------------------------------
        # 2) 基础 mask：只在 [bidx, eidx) 有 1
        # ---------------------------------------------------------
        mask = np.zeros((3, nT), dtype=np.float32)
        mask[:, bidx:eidx] = 1.0

        # ---------------------------------------------------------
        # 3) 在 [bidx, eidx) 内随机挖洞（0~3 个点）
        # ---------------------------------------------------------
        n_holes = int(rng.integers(0, 4))
        for _ in range(n_holes):
            hole_idx = int(rng.integers(bidx, eidx))
            mask[:, hole_idx] = 0.0

        # ---------------------------------------------------------
        # 4) 40% 没 Rayleigh，40% 没 Love，20% 都有
        # ---------------------------------------------------------
        r = rng.random()
        if r < 0.4:
            mask[1, :] = 0.0     # 没有 Rayleigh
        elif r < 0.8:
            mask[2, :] = 0.0     # 没有 Love
        # else: 20% 全留着

        # 转回 torch
        mask = torch.from_numpy(mask)  # float32





        return model, disp, mask 

    def __getitem__(self, idx):
        """
        一个样本的采样逻辑：
          - 拿一个 rng
          - 循环尝试最多 max_tries 次
          - 如果 compute_phase_dispersion / disba 内部报错（DispersionError 等），
            就丢弃当前速度模型，重新生成一条再算，直到成功。
        """
        rng = self._get_rng(idx)

        last_error = None
        for attempt in range(self.max_tries):
            try:
                model, disp, mask = self._one_sample(rng)
                return model, disp, mask
            except DispersionError as e:
                # 典型：failed to find root for fundamental mode
                last_error = e
                continue
            except Exception as e:
                # 其它偶发错误（比如某些极端模型数值不稳定）
                last_error = e
                continue

        # 如果多次都失败，就明确抛一个 RuntimeError，方便你在外层定位问题
        raise RuntimeError(
            f"Failed to generate a valid sample after {self.max_tries} tries. "
            f"Last error: {repr(last_error)}"
        )


if __name__ == "__main__":
    ds = SurfaceWaveDataset(
        n_samples=100000,
        z_max_km=150.0,
        dz_km=0.5,
        seed=2025,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    import matplotlib.pyplot as plt
    for b, (model, disp) in enumerate(loader):
        print("model:", model.shape)  # [B, 4, H]
        print("disp :", disp.shape)   # [B, 3, T]  (T ≈ 59)
        # 看一眼第一条频散
        if b == 0:
            T = disp[0, 0].numpy()
            cR = disp[0, 1].numpy()
            cL = disp[0, 2].numpy()

            plt.figure(figsize=(5,6))
            plt.subplot(1, 2, 1)
            plt.plot(model[0, 1].numpy(), model[0, 0].numpy(), "b-", label="Vp")
            plt.ylim(120, 0)
            plt.subplot(1, 2, 2)
            plt.plot(T, cR, "b-", label="Rayleigh 0")
            plt.plot(T, cL, "r--", label="Love 0")
            plt.xlabel("Period (s)")
            plt.ylabel("Phase velocity (km/s)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title("Example sample from SurfaceWaveDataset")
            plt.show()

        if b > 1:
            break
