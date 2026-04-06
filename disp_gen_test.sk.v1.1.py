#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 替换：用 Transformer
from models.struct2disp_transformer import Struct2DispTransformer
from utils.generate_data import SurfaceWaveDataset
import random 
import time 

def seed_everything(seed=2026):
    """
    一键固定所有随机种子，确保实验可复现。
    """
    # 1. 基本 Python 与 Numpy 种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. PyTorch 种子 (CPU & 所有 GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 3. 彻底解决 GPU 算子不确定性 (关键！)
    # 牺牲少量训练速度，换取 100% 的结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. 针对 PyTorch 2.0+ 的最新设置
    # 确保在使用 torch.use_deterministic_algorithms 时能抛出不可确定操作的错误
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"✅ 所有随机种子已固定为: {seed}")

# 建议在全局作用域直接调用


plt.switch_backend("Agg")
def smooth_1d_along_depth(arr_2d: torch.Tensor, win: int = 9) -> torch.Tensor:
    """对 [NT, H] 的核在 depth(H) 方向做滑动平均平滑。"""
    assert arr_2d.ndim == 2
    if win <= 1:
        return arr_2d
    x = arr_2d[None, None, :, :]  # [1,1,NT,H]
    x = F.avg_pool2d(x, kernel_size=(1, win), stride=1, padding=(0, win // 2))
    return x[0, 0]


def normalize_each_period(arr_2d: torch.Tensor, mode: str = "l1abs") -> torch.Tensor:
    """对每个周期(每行)做归一化，突出“形状”而非绝对幅值。"""
    if mode == "none":
        return arr_2d
    if mode == "l1abs":
        denom = arr_2d.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        return arr_2d / denom
    raise ValueError(f"Unknown normalize mode: {mode}")


@torch.no_grad()
def _disable_inplace(model: nn.Module):
    # Transformer 基本不会有 inplace，但保留也没坏处
    for m in model.modules():
        if hasattr(m, "inplace"):
            try:
                m.inplace = False
            except Exception:
                pass


def compute_dcdvs_full_jacobian(model: nn.Module, x0: torch.Tensor, periods: torch.Tensor):
    """
    torch.func 版本：K = ∂c/∂Vs (c 使用 Transformer 输出的 mu)

    输入:
      x0: [N,4,H]
      periods: [NT] or [N,NT]
    输出:
      K:  [N,2,NT,H]
      y:  [N,2,NT]
      vs: [N,H]
    """
    from torch.func import functional_call, vmap, jacrev, jacfwd

    device = x0.device
    N, C, H = x0.shape
    assert C == 4

    # periods -> [N,NT]
    if periods.ndim == 1:
        periods_in = periods.unsqueeze(0).expand(N, -1).to(device)
    else:
        periods_in = periods.to(device)
        assert periods_in.shape[0] == N

    # 参数/缓冲区（functional_call 需要）
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # vs as independent variable
    vs0 = x0[:, 2, :].detach()  # [N,H]

    # 单样本函数：输入 vs:[H], x_base:[4,H], p:[NT] -> y_flat:[2*NT]
    def f_single(vs_1d, x_base_4h, p_1d, params, buffers):
        # 替换 Vs 通道
        x = x_base_4h.clone()
        x[2, :] = vs_1d

        mu, _ = functional_call(
            model, (params, buffers),
            (x.unsqueeze(0),),  # x: [1,4,H]
            {"periods": p_1d.unsqueeze(0)}  # periods: [1,NT]
        )
        # mu: [1,2,NT] -> [2*NT]
        return mu.squeeze(0).reshape(-1)

    # jacobian w.r.t. vs_1d: [2*NT, H]
    jac_single = jacrev(f_single, argnums=0)

    # vmap over batch:
    # vs0: [N,H], x0: [N,4,H], periods_in: [N,NT]
    K_flat = vmap(jac_single, in_dims=(0, 0, 0, None, None))(
        vs0, x0.detach(), periods_in.detach(), params, buffers
    )  # [N, 2*NT, H]

    # 同时拿到 y（mu），避免再循环 forward
    y_flat = vmap(f_single, in_dims=(0, 0, 0, None, None))(
        vs0, x0.detach(), periods_in.detach(), params, buffers
    )  # [N, 2*NT]

    NT = periods_in.shape[1]
    y = y_flat.view(N, 2, NT)              # [N,2,NT]
    K = K_flat.view(N, 2, NT, H)           # [N,2,NT,H]

    return K, y.detach(), vs0.detach()


# -------------------------
# disba part
# -------------------------
def _interp1d_to_depth(depth_src_km: np.ndarray, k_src: np.ndarray, depth_tgt_km: np.ndarray) -> np.ndarray:
    """把 disba 的 kernel 插值到你的深度网格上。"""
    depth_src_km = np.asarray(depth_src_km, dtype=float)
    k_src = np.asarray(k_src, dtype=float)
    depth_tgt_km = np.asarray(depth_tgt_km, dtype=float)
    return np.interp(depth_tgt_km, depth_src_km, k_src, left=k_src[0], right=k_src[-1])


def disba_vs_phase_sensitivity(
    depth_km: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    periods_s: np.ndarray,
    wave: str,
    mode: int = 0,
    assume_units_kms_gcc: bool = True,
):
    """
    用 disba 计算 phase velocity 对 Vs 的敏感核 (parameter="velocity_s")。
    返回:
      K_disba: [NT,H]  已插值到 depth_km
    """
    try:
        from disba import PhaseSensitivity
    except Exception as e:
        raise RuntimeError(
            "disba 未安装或不可用。请先安装：pip install disba[full]\n"
            f"原始错误: {e}"
        )

    depth_km = np.asarray(depth_km, dtype=float)
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    periods_s = np.asarray(periods_s, dtype=float)

    # disba 期望 vp/vs: km/s, rho: g/cm^3
    if not assume_units_kms_gcc:
        vp = vp / 1000.0
        vs = vs / 1000.0
        rho = rho / 1000.0

    H = depth_km.size
    if H < 2:
        raise ValueError("depth_km 长度太短，至少2")

    dz = np.diff(depth_km)
    dz0 = float(np.median(dz))
    thickness = np.full(H, dz0, dtype=float)
    thickness[-1] = 0.0  # 半空间

    ps = PhaseSensitivity(thickness, vp, vs, rho)

    k_list = []
    depth_out = None

    for T in periods_s:
        out = ps(float(T), mode=mode, wave=wave, parameter="velocity_s")
        depth_out = np.asarray(out.depth, dtype=float)
        k_list.append(np.asarray(out.kernel, dtype=float))

    K_disba = np.stack([_interp1d_to_depth(depth_out, k, depth_km) for k in k_list], axis=0)  # [NT,H]
    return K_disba




def plot_kernels_with_disba_multiperoid(
    depth_km: torch.Tensor,          # [H]
    periods_s: torch.Tensor,         # [NT]
    K_log_dz: torch.Tensor,          # [2,NT,H]  (NN: ∂lnc/∂lnVs * dz)
    K_disba_vs: torch.Tensor,        # [2,NT,H]
    out_prefix: str,
    smooth_win: int = 9,
    norm_mode: str = "l1abs",
    period_indices: list[int] | None = None,   # e.g. [3, 8, 18, 28]
    max_depth_km: float = 80.0,
    dpi: int = 400,
    seed: int = 7, 
):
    """
    GRL-style multi-period comparison:
      - nrow x 2 layout
      - left column: Rayleigh
      - right column: Love
      - each row: one period
      - shared legend at bottom
      - black/gray print-friendly style
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    # ---------------- style ----------------
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["legend.fontsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["axes.linewidth"] = 0.8

    # ---------------- smooth + normalize ----------------
    K_R = smooth_1d_along_depth(K_log_dz[0], win=smooth_win)
    K_L = smooth_1d_along_depth(K_log_dz[1], win=smooth_win)
    K_Rn = normalize_each_period(K_R, mode=norm_mode)
    K_Ln = normalize_each_period(K_L, mode=norm_mode)

    D_R = smooth_1d_along_depth(K_disba_vs[0], win=smooth_win)
    D_L = smooth_1d_along_depth(K_disba_vs[1], win=smooth_win)
    D_Rn = normalize_each_period(D_R, mode=norm_mode)
    D_Ln = normalize_each_period(D_L, mode=norm_mode)

    depth_np = depth_km.detach().cpu().numpy()
    per_np = periods_s.detach().cpu().numpy()

    # ---------------- choose periods ----------------
    NT = len(per_np)
    if period_indices is None:
        # 默认选 4 个代表性周期，尽量覆盖浅-深敏感性变化
        target_periods = [5.0, 10.0, 20.0, 40.0]
        period_indices = [int(np.argmin(np.abs(per_np - tp))) for tp in target_periods]

        # 去重，避免周期数组不含这些值时重复命中
        seen = set()
        uniq = []
        for i in period_indices:
            if i not in seen:
                uniq.append(i)
                seen.add(i)
        period_indices = uniq

    period_indices = [max(0, min(int(i), NT - 1)) for i in period_indices]
    nrow = len(period_indices)

    # ---------------- depth truncation ----------------
    if max_depth_km is not None:
        keep = (depth_np >= 0.0) & (depth_np <= float(max_depth_km))
    else:
        keep = slice(None)

    x = depth_np[keep]

    # ---------------- figure size ----------------
    # 每行高度约 2.1 in，2列宽图适合 GRL
    fig_h = max(4.2, 2.15 * nrow)
    fig, axes = plt.subplots(
        nrow, 2,
        figsize=(11.5, fig_h),
        sharex=True,
        sharey=False,
        squeeze=False
    )

    # ---------------- labels / styles ----------------
    xlab = "Depth (km)"
    ylab = "Normalized Sensitivity"
    name1 = "Neural Surrogate Gradient"
    name2 = "Theoretical Sensitivity Kernel"

    c_nn = "0.15"   # dark gray / near black
    c_th = "0.50"   # medium gray
    lw_main = 2.0
    lw_ref = 2.0

    # panel labels: (a), (b), (c), ...
    letters = [chr(ord("a") + i) for i in range(nrow * 2)]

    # ---------------- plot each row ----------------
    global_ymin = np.inf
    global_ymax = -np.inf

    # 先统计统一 y 范围，保证对比整洁
    for tidx in period_indices:
        KR = K_Rn[tidx].detach().cpu().numpy()[keep]
        DR = D_Rn[tidx].detach().cpu().numpy()[keep]
        KL = K_Ln[tidx].detach().cpu().numpy()[keep]
        DL = D_Ln[tidx].detach().cpu().numpy()[keep]

        ymin = min(KR.min(), DR.min(), KL.min(), DL.min())
        ymax = max(KR.max(), DR.max(), KL.max(), DL.max())
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)

    # 给一点 padding
    yr = global_ymax - global_ymin
    if yr <= 0:
        yr = 1.0
    global_ymin -= 0.06 * yr
    global_ymax += 0.06 * yr

    for i, tidx in enumerate(period_indices):
        T0 = float(per_np[tidx])

        KR = K_Rn[tidx].detach().cpu().numpy()[keep]
        DR = D_Rn[tidx].detach().cpu().numpy()[keep]
        KL = K_Ln[tidx].detach().cpu().numpy()[keep]
        DL = D_Ln[tidx].detach().cpu().numpy()[keep]

        # ----- left: Rayleigh -----
        # ----- left: Rayleigh -----
        ax = axes[i, 0]
        ax.plot(x, KR, color=c_nn, lw=lw_main, label=name1)
        ax.plot(x, DR, color=c_th, lw=lw_ref, ls="--", label=name2)
        ax.axhline(0.0, color="0.65", lw=0.9)
        ax.grid(alpha=0.18, linewidth=0.8)

        # independent y-limits
        ymin = min(KR.min(), DR.min(), 0.0)
        ymax = max(KR.max(), DR.max(), 0.0)
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

        ax.set_title(f"Rayleigh wave (T = {T0:.0f} s)")
        ax.text(
            0.02, 0.95, f"({letters[2*i]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylabel(ylab)

        # ----- right: Love -----
        # ----- right: Love -----
        ax = axes[i, 1]
        ax.plot(x, KL, color=c_nn, lw=lw_main, label=name1)
        ax.plot(x, DL, color=c_th, lw=lw_ref, ls="--", label=name2)
        ax.axhline(0.0, color="0.65", lw=0.9)
        ax.grid(alpha=0.18, linewidth=0.8)

        # independent y-limits
        ymin = min(KL.min(), DL.min(), 0.0)
        ymax = max(KL.max(), DL.max(), 0.0)
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

        ax.set_title(f"Love wave (T = {T0:.0f} s)")
        ax.text(
            0.02, 0.95, f"({letters[2*i+1]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold"
        )

    # ---------------- shared x settings ----------------
    if max_depth_km is not None:
        for ax in axes.ravel():
            ax.set_xlim(0.0, float(max_depth_km))

    # 只在最后一行放 x label
    axes[-1, 0].set_xlabel(xlab)
    axes[-1, 1].set_xlabel(xlab)

    # ---------------- shared legend ----------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.01),
    )

    # ---------------- layout ----------------
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.96,
        bottom=0.10,
        wspace=0.08,
        hspace=0.28,
    )

    # ---------------- save ----------------
    tlist = "_".join([f"{float(per_np[i]):.0f}s" for i in period_indices])
    out_path = f"{out_prefix}_kernel_multiperoid_compare_{tlist}.{seed}.v1.1.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path



def plot_kernels_with_disba_multiband(
    depth_km: torch.Tensor,          # [H]
    periods_s: torch.Tensor,         # [NT]
    K_log_dz: torch.Tensor,          # [2,NT,H]
    K_disba_vs: torch.Tensor,        # [2,NT,H]
    out_prefix: str,
    smooth_win: int = 9,
    norm_mode: str = "l1abs",
    period_bands: list[tuple[float, float, str]] | None = None,
    max_depth_km: float = 150.0,
    dpi: int = 400,
    #seed: int = 7,
):
    """
    按周期区间绘制平均敏感核：
      - 每一行一个 period band
      - 左列 Rayleigh, 右列 Love
      - band 内对各周期的 normalized kernel 求平均
      - 深度显示 0–150 km
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    if period_bands is None:
        period_bands = [
            (2.0, 5.0, "2–5 s"),
            (10.0, 15.0, "10–15 s"),
            (20.0, 30.0, "20–30 s"),
            (30.0, 40.0, "30–40 s"),
            (50.0, 60.0, "50–60 s"),
        ]

    # ---------------- style ----------------
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["legend.fontsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["axes.linewidth"] = 0.8

    # ---------------- smooth + normalize ----------------
    K_R = smooth_1d_along_depth(K_log_dz[0], win=smooth_win)
    K_L = smooth_1d_along_depth(K_log_dz[1], win=smooth_win)
    K_Rn = normalize_each_period(K_R, mode=norm_mode)
    K_Ln = normalize_each_period(K_L, mode=norm_mode)

    D_R = smooth_1d_along_depth(K_disba_vs[0], win=smooth_win)
    D_L = smooth_1d_along_depth(K_disba_vs[1], win=smooth_win)
    D_Rn = normalize_each_period(D_R, mode=norm_mode)
    D_Ln = normalize_each_period(D_L, mode=norm_mode)

    depth_np = depth_km.detach().cpu().numpy()
    per_np = periods_s.detach().cpu().numpy()

    # ---------------- depth truncation ----------------
    if max_depth_km is not None:
        keep_depth = (depth_np >= 0.0) & (depth_np <= float(max_depth_km))
    else:
        keep_depth = slice(None)

    x = depth_np[keep_depth]

    # ---------------- build band averages ----------------
    band_data = []
    for lo, hi, label in period_bands:
        # 最后一个区间右端点包含
        if hi == period_bands[-1][1]:
            pmask = (per_np >= lo) & (per_np <= hi)
        else:
            pmask = (per_np >= lo) & (per_np < hi)

        idx = np.where(pmask)[0]
        if len(idx) == 0:
            continue

        KR_band = K_Rn[idx][:, keep_depth].detach().cpu().numpy()
        KL_band = K_Ln[idx][:, keep_depth].detach().cpu().numpy()
        DR_band = D_Rn[idx][:, keep_depth].detach().cpu().numpy()
        DL_band = D_Ln[idx][:, keep_depth].detach().cpu().numpy()

        # 若未来某些周期存在 NaN，这里也能稳住
        KR_mean = np.nanmean(KR_band, axis=0)
        KL_mean = np.nanmean(KL_band, axis=0)
        DR_mean = np.nanmean(DR_band, axis=0)
        DL_mean = np.nanmean(DL_band, axis=0)

        band_data.append({
            "label": label,
            "lo": lo,
            "hi": hi,
            "n_periods": len(idx),
            "KR": KR_mean,
            "KL": KL_mean,
            "DR": DR_mean,
            "DL": DL_mean,
        })

    nrow = len(band_data)
    if nrow == 0:
        raise ValueError("No valid period bands found in periods_s.")

    # ---------------- figure ----------------
    fig_h = max(5.5, 1.95 * nrow)
    fig, axes = plt.subplots(
        nrow, 2,
        figsize=(11.5, fig_h),
        sharex=True,
        sharey=False,
        squeeze=False
    )

    xlab = "Depth (km)"
    ylab = "Normalized Sensitivity"
    name1 = "Neural Surrogate Gradient"
    name2 = "Theoretical Sensitivity Kernel"

    c_nn = "0.15"
    c_th = "0.50"
    lw_main = 2.0
    lw_ref = 2.0

    letters = [chr(ord("a") + i) for i in range(nrow * 2)]

    for i, bd in enumerate(band_data):
        KR = bd["KR"]
        KL = bd["KL"]
        DR = bd["DR"]
        DL = bd["DL"]
        band_label = bd["label"]

        # ----- Rayleigh -----
        ax = axes[i, 0]
        ax.plot(x, KR, color=c_nn, lw=lw_main, label=name1)
        ax.plot(x, DR, color=c_th, lw=lw_ref, ls="--", label=name2)
        ax.axhline(0.0, color="0.65", lw=0.9)
        ax.grid(alpha=0.18, linewidth=0.8)

        ymin = np.nanmin([KR.min(), DR.min(), 0.0])
        ymax = np.nanmax([KR.max(), DR.max(), 0.0])
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

        ax.set_title(f"Rayleigh wave ({band_label})")
        ax.set_ylabel(ylab)
        ax.text(
            0.02, 0.95, f"({letters[2*i]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold"
        )

        # ----- Love -----
        ax = axes[i, 1]
        ax.plot(x, KL, color=c_nn, lw=lw_main, label=name1)
        ax.plot(x, DL, color=c_th, lw=lw_ref, ls="--", label=name2)
        ax.axhline(0.0, color="0.65", lw=0.9)
        ax.grid(alpha=0.18, linewidth=0.8)

        ymin = np.nanmin([KL.min(), DL.min(), 0.0])
        ymax = np.nanmax([KL.max(), DL.max(), 0.0])
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

        ax.set_title(f"Love wave ({band_label})")
        ax.text(
            0.02, 0.95, f"({letters[2*i+1]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold"
        )

    # x range
    if max_depth_km is not None:
        for ax in axes.ravel():
            ax.set_xlim(0.0, float(max_depth_km))

    # bottom x labels
    axes[-1, 0].set_xlabel(xlab)
    axes[-1, 1].set_xlabel(xlab)

    # legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.97,
        bottom=0.08,
        wspace=0.08,
        hspace=0.30,
    )
    ts = int(time.time())
    out_path = f"{out_prefix}_kernel_multiband_compare_0to{int(max_depth_km)}km_seed_{ts}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path

def run_one_batch(
    loader,
    ckpt_path="ckpt/struct2disp_transformer.pt",
    fig_dir="tfig",
    device=None,
    smooth_win=9,
    norm_mode="l1abs",
    pick_t_index=10,
    assume_units_kms_gcc=True,
    #seed=12345, 
):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # device
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    print("Device:", device)

    # peek one batch
    model_batch, disp_batch, mask_batch = next(iter(loader))
    B, C_model, H = model_batch.shape   # [B,4,H]
    _, C_disp, NT = disp_batch.shape    # [B,3,NT]
    print(f"H={H}, NT={NT}")
    assert NT == 59, "This script assumes T=59"

    # periods min/max for query embedding（用于没传 periods 的 fallback；我们这里会显式传）
    pmin = float(disp_batch[:, 0, :].min().item())
    pmax = float(disp_batch[:, 0, :].max().item())

    # build Transformer model
    model = Struct2DispTransformer(
        H=H,
        T=NT,
        C_in=4,
        d_model=512,
        nhead=8,
        num_enc_layers=8,
        num_dec_layers=8,
        dim_ff=1024,
        dropout=0.1,
        use_period_values=True,
        period_minmax=(pmin, pmax),
    ).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded ckpt:", ckpt_path)
    else:
        print("ckpt not exists:", ckpt_path)

    model.eval()
    _disable_inplace(model)

    # 只处理一个 batch（建议 batch_size=1）
    for model_batch, disp_batch, mask_batch in loader:
        x0 = model_batch.to(device).detach()   # [N,4,H]
        N = x0.shape[0]

        # 深度坐标（km），来自输入的第0通道
        depth_km = x0[0, 0, :].detach().cpu()  # [H]
        #print("深度", depth_km)
        # 周期：disp_batch[:,0,:]
        periods = disp_batch[:, 0, :].to(device).detach()   # [N,NT]
        periods_s = periods[0].detach().cpu()

        # NN 梯度核：dC/dVs（C 是 mu）
        K, y, vs = compute_dcdvs_full_jacobian(model, x0, periods=periods)

        # 转成 Fréchet-style: ∂ln c / ∂ln Vs = (Vs/c) * ∂c/∂Vs
        eps = 1e-12
        c = y.clamp_min(eps)  # [N,2,NT]
        K_log = (K * vs[:, None, None, :]) / c[:, :, :, None]  # [N,2,NT,H]

        # 乘 dz（离散积分权重）
        dz_km = float(depth_km[1] - depth_km[0]) if H > 1 else 1.0
        K_log_dz = K_log * dz_km  # [N,2,NT,H]

        # ---------------- disba kernel ----------------
        depth_np = depth_km.numpy()
        vp_np = x0[0, 1, :].detach().cpu().numpy()
        vs_np = x0[0, 2, :].detach().cpu().numpy()
        rho_np = x0[0, 3, :].detach().cpu().numpy()
        periods_np = periods_s.numpy()  # [NT]

        Kd_R = disba_vs_phase_sensitivity(
            depth_np, vp_np, vs_np, rho_np, periods_np,
            wave="rayleigh", mode=0, assume_units_kms_gcc=assume_units_kms_gcc
        )  # [NT,H]
        Kd_L = disba_vs_phase_sensitivity(
            depth_np, vp_np, vs_np, rho_np, periods_np,
            wave="love", mode=0, assume_units_kms_gcc=assume_units_kms_gcc
        )  # [NT,H]

        K_disba_vs = torch.stack([
            torch.tensor(Kd_R, dtype=torch.float32),
            torch.tensor(Kd_L, dtype=torch.float32),
        ], dim=0)  # [2,NT,H]

        # ---------------- plot ----------------
        #out_prefix = os.path.join(fig_dir, "frechet_kernel_sample0")
        #plot_kernels_with_disba(
        #    depth_km=depth_km,
        #    periods_s=periods_s,
        #    K_log_dz=K_log_dz[0].detach().cpu(),      # [2,NT,H]
        #    K_disba_vs=K_disba_vs.detach().cpu(),     # [2,NT,H]
        #    out_prefix=out_prefix,
        #    smooth_win=smooth_win,
        #    norm_mode=norm_mode,
        #    pick_t_index=pick_t_index,
        #)

        #plot_kernels_with_disba_multiperoid(
        #    depth_km=depth_km,
        #    periods_s=periods_s,
        #    K_log_dz=K_log_dz[0].detach().cpu(),      # [2,NT,H]
        #    K_disba_vs=K_disba_vs.detach().cpu(),     # [2,NT,H]
        #    out_prefix=os.path.join(fig_dir, "frechet_kernel_multiperoid"),
        #    smooth_win=smooth_win,
        #    norm_mode=norm_mode,
        #    period_indices=[3, 8, 18, 28],  # 代表性周期索引
        #    max_depth_km=80.0,
        #    dpi=400,
        #    seed=seed,
        #)

        plot_kernels_with_disba_multiband(
            depth_km=depth_km,
            periods_s=periods_s,
            K_log_dz=K_log_dz[0].detach().cpu(),      # [2,NT,H]
            K_disba_vs=K_disba_vs.detach().cpu(),     # [2,NT,H]
            out_prefix=os.path.join(fig_dir, "frechet_kernel_multiband"),
            smooth_win=smooth_win,
            norm_mode=norm_mode,
            period_bands=[
                (2.0, 5.0,  "2–5 s"),
                (10.0, 15.0, "10–15 s"),
                (20.0, 30.0, "20–30 s"),
                (30.0, 40.0, "30–40 s"),
                (50.0, 60.0, "50–60 s"),
            ],
            max_depth_km=110.0,
            dpi=400,
            #seed=seed,
        )

        ## 原始 dC/dVs（帮助排查）
        #t0 = max(0, min(int(pick_t_index), NT - 1))
        #t0s = float(periods_np[t0])
        #plt.figure(figsize=(7, 4))
        #plt.plot(depth_km.numpy(), K[0, 0, t0, :].detach().cpu().numpy(), label=f"Rayleigh #dC/dVs @T={t0s:.3f}s")
        #plt.plot(depth_km.numpy(), K[0, 1, t0, :].detach().cpu().numpy(), label=f"Love dC/#dVs @T={t0s:.3f}s")
        #plt.axhline(0.0, linewidth=0.8)
        #plt.xlabel("Depth (km)")
        #plt.ylabel("dC/dVs (raw)")
        #plt.legend()
        #plt.tight_layout()
        #plt.savefig(os.path.join(fig_dir, f"raw_dcdvs_T{t0s:.3f}s.png"), dpi=200)
        #plt.close()

        print("Saved figures to:", fig_dir)
        break


if __name__ == "__main__":
    SEED = 2026
    seed_everything(SEED)
    from torch.utils.data import DataLoader
    rng = torch.Generator()
    rng.manual_seed(SEED)

    for k in range(100):
        
        
        ds = SurfaceWaveDataset(
            n_samples=100000,
            z_max_km=150.0,
            z_max_num=256,
            dz_km=0.5,
            seed=SEED,
        )

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, generator=rng)

        run_one_batch(
            loader,
            ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",  # <-- 这里换成 Transformer 的 ckpt
            fig_dir="tfig/multi2",
            device=None,
            smooth_win=9,
            norm_mode="l1abs",
            pick_t_index=10,
            assume_units_kms_gcc=True,
            #seed=SEED, 
        )
