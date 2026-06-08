#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.struct2disp_transformer import Struct2DispTransformer
from utils.generate_data import SurfaceWaveDataset

plt.switch_backend("Agg")

import pickle
from pathlib import Path


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

def save_results_pickle(results, out_path):
    """
    保存 results 到 pickle 文件
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Saved] results -> {out_path}")

def load_results_pickle(path):
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results    
@torch.no_grad()
def _disable_inplace(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "inplace"):
            try:
                m.inplace = False
            except Exception:
                pass


def build_model_from_loader_batch(loader, ckpt_path, device):
    model_batch, disp_batch, mask_batch = next(iter(loader))
    _, C_model, H = model_batch.shape
    _, C_disp, NT = disp_batch.shape

    pmin = float(disp_batch[:, 0, :].min().item())
    pmax = float(disp_batch[:, 0, :].max().item())

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

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    _disable_inplace(model)
    return model, H, NT


def compute_mean_initial_model(dataset, n_mean=100):
    """
    从 SurfaceWaveDataset 取前 n_mean 个样本，构造平均初始模型。
    返回:
      x_mean: [4,H] = [depth, vp, vs, rho]
    """
    xs = []
    for i in range(n_mean):
        x, d, mask = dataset[i]
        xs.append(x)
    xs = torch.stack(xs, dim=0)  # [n_mean,4,H]

    x_mean = xs.mean(dim=0)
    # depth 通道最好直接用第一个样本，避免平均后有数值扰动
    x_mean[0] = xs[0, 0]
    return x_mean


def inverse_softplus(y, eps=1e-6):
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))


def make_forward_model_input(x_template_4h, vs_1d):
    """
    用固定的 depth/vp/rho + 可优化的 vs 组装输入
    x_template_4h: [4,H]
    vs_1d: [H]
    """
    x = x_template_4h.clone()
    x[2, :] = vs_1d
    return x.unsqueeze(0)  # [1,4,H]


def compute_jacobian_wrt_vs_single(model, x_1_4_h, periods_1_nt):
    """
    对单个样本计算 J = d(pred_disp_flat)/d(vs), shape [2*NT, H]
    这里直接用 autograd.functional.jacobian，稳一点。
    """
    x_template = x_1_4_h[0].detach().clone()   # [4,H]
    vs0 = x_template[2].detach().clone()       # [H]
    periods_1_nt = periods_1_nt.detach()

    def f_vs(vs_1d):
        x = x_template.clone()
        x[2, :] = vs_1d
        mu, _ = model(x.unsqueeze(0), periods=periods_1_nt)   # [1,2,NT]
        return mu.reshape(-1)  # [2*NT]

    J = torch.autograd.functional.jacobian(f_vs, vs0, create_graph=False)  # [2*NT,H]
    return J.detach(), vs0.detach()

def make_forward_model_input_full(x_template_4h, vp_1d, vs_1d, rho_1d):
    """
    用固定 depth + 可优化的 vp/vs/rho 组装输入
    x_template_4h: [4,H]
    vp_1d, vs_1d, rho_1d: [H]
    """
    x = x_template_4h.clone()
    x[1, :] = vp_1d
    x[2, :] = vs_1d
    x[3, :] = rho_1d
    return x.unsqueeze(0)  # [1,4,H]

def smoothness_losses_1d(m):
    """
    m: [H]
    返回:
      smooth1 = mean((D1 m)^2)
      smooth2 = mean((D2 m)^2)
    """
    d1 = m[1:] - m[:-1]
    smooth1 = torch.mean(d1 ** 2)

    if m.numel() >= 3:
        d2 = m[2:] - 2.0 * m[1:-1] + m[:-2]
        smooth2 = torch.mean(d2 ** 2)
    else:
        smooth2 = torch.tensor(0.0, device=m.device, dtype=m.dtype)

    return smooth1, smooth2
def build_regularization_hessian_1d(
    H,
    lam_prior=0.0,
    lam_smooth1=0.0,
    lam_smooth2=0.0,
    device="cpu",
    dtype=torch.float32,
):
    """
    构造 1D 模型参数的正则 Hessian 近似:
      H_reg = lam_prior * I + lam_smooth1 * D1^T D1 + lam_smooth2 * D2^T D2
    """
    I = torch.eye(H, device=device, dtype=dtype)
    Hreg = lam_prior * I

    # D1: (H-1) x H
    if H >= 2 and lam_smooth1 > 0.0:
        D1 = torch.zeros((H - 1, H), device=device, dtype=dtype)
        idx = torch.arange(H - 1, device=device)
        D1[idx, idx] = -1.0
        D1[idx, idx + 1] = 1.0
        Hreg = Hreg + lam_smooth1 * (D1.T @ D1)

    # D2: (H-2) x H
    if H >= 3 and lam_smooth2 > 0.0:
        D2 = torch.zeros((H - 2, H), device=device, dtype=dtype)
        idx = torch.arange(H - 2, device=device)
        D2[idx, idx] = 1.0
        D2[idx, idx + 1] = -2.0
        D2[idx, idx + 2] = 1.0
        Hreg = Hreg + lam_smooth2 * (D2.T @ D2)

    return Hreg

def compute_jacobian_wrt_model_full_single(model, x_1_4_h, periods_1_nt):
    """
    对单样本计算 J = d(pred_disp_flat)/d([vp,vs,rho]), shape [2*NT, 3H]

    返回:
      J: [2*NT, 3H]
      q0: [3H]  = [vp, vs, rho] 拼接
    """
    x_template = x_1_4_h[0].detach().clone()  # [4,H]
    vp0 = x_template[1].detach().clone()
    vs0 = x_template[2].detach().clone()
    rho0 = x_template[3].detach().clone()
    periods_1_nt = periods_1_nt.detach()

    H = vp0.numel()
    q0 = torch.cat([vp0, vs0, rho0], dim=0)   # [3H]

    def f_q(q):
        vp = q[:H]
        vs = q[H:2*H]
        rho = q[2*H:3*H]
        x = x_template.clone()
        x[1, :] = vp
        x[2, :] = vs
        x[3, :] = rho
        mu, _ = model(x.unsqueeze(0), periods=periods_1_nt)   # [1,2,NT]
        return mu.reshape(-1)                                  # [2*NT]

    J = torch.autograd.functional.jacobian(f_q, q0, create_graph=False)  # [2*NT,3H]
    return J.detach(), q0.detach()


def select_control_indices(depth_km):
    """
    分段控制点:
      0–5 km   : 0.5 km
      5–10 km  : 1 km
      10–50 km : 5 km
      50–150 km: 20 km

    返回:
      cp_idx: LongTensor [Nc]
    """
    depth_np = depth_km.detach().cpu().numpy()
    zmin = float(depth_np.min())
    zmax = float(depth_np.max())

    cp_depths = []

    # 0–5 km, 0.5 km
    cp_depths.extend(np.arange(0.0, min(5.0, zmax) + 1e-8, 0.5).tolist())

    # 5–10 km, 1 km
    if zmax > 5.0:
        cp_depths.extend(np.arange(6.0, min(10.0, zmax) + 1e-8, 1.0).tolist())

    # 10–50 km, 5 km
    if zmax > 10.0:
        cp_depths.extend(np.arange(15.0, min(50.0, zmax) + 1e-8, 5.0).tolist())

    # 50–150 km, 20 km
    if zmax > 50.0:
        cp_depths.extend(np.arange(70.0, min(150.0, zmax) + 1e-8, 20.0).tolist())

    # 保证包含顶部和底部
    cp_depths = [zmin] + cp_depths + [zmax]
    cp_depths = np.unique(np.asarray(cp_depths, dtype=float))

    cp_idx = []
    for z in cp_depths:
        cp_idx.append(int(np.argmin(np.abs(depth_np - z))))

    cp_idx = np.unique(np.asarray(cp_idx, dtype=int))
    return torch.as_tensor(cp_idx, dtype=torch.long, device=depth_km.device)


def interp_control_to_full(depth_full, depth_cp, values_cp):
    """
    用 torch 线性插值把 control-point 参数恢复到全深度网格。
    输入:
      depth_full: [H]
      depth_cp:   [Nc]
      values_cp:  [Nc]
    返回:
      values_full: [H]
    """
    H = depth_full.numel()
    Nc = depth_cp.numel()
    device = depth_full.device
    dtype = depth_full.dtype

    out = torch.empty(H, device=device, dtype=dtype)

    for i in range(H):
        z = depth_full[i]

        if z <= depth_cp[0]:
            out[i] = values_cp[0]
        elif z >= depth_cp[-1]:
            out[i] = values_cp[-1]
        else:
            j = torch.searchsorted(depth_cp, z).item()
            z0, z1 = depth_cp[j - 1], depth_cp[j]
            v0, v1 = values_cp[j - 1], values_cp[j]
            w = (z - z0) / (z1 - z0 + 1e-12)
            out[i] = (1.0 - w) * v0 + w * v1

    return out


def make_forward_model_input_from_cp(
    x_template_4h,
    vp_cp, vs_cp, rho_cp,
    cp_idx,
):
    """
    从 control-point 参数恢复全深度模型并组装输入。
    x_template_4h: [4,H]
    vp_cp,vs_cp,rho_cp: [Nc]
    cp_idx: [Nc]
    """
    depth_full = x_template_4h[0]
    depth_cp = depth_full[cp_idx]

    vp_full = interp_control_to_full(depth_full, depth_cp, vp_cp)
    vs_full = interp_control_to_full(depth_full, depth_cp, vs_cp)
    rho_full = interp_control_to_full(depth_full, depth_cp, rho_cp)

    x = x_template_4h.clone()
    x[1, :] = vp_full
    x[2, :] = vs_full
    x[3, :] = rho_full
    return x.unsqueeze(0), vp_full, vs_full, rho_full


def smoothness_losses_cp(m_cp):
    """
    control-point 空间的一阶/二阶平滑
    m_cp: [Nc]
    """
    if m_cp.numel() < 2:
        s1 = torch.tensor(0.0, device=m_cp.device, dtype=m_cp.dtype)
    else:
        d1 = m_cp[1:] - m_cp[:-1]
        s1 = torch.mean(d1 ** 2)

    if m_cp.numel() < 3:
        s2 = torch.tensor(0.0, device=m_cp.device, dtype=m_cp.dtype)
    else:
        d2 = m_cp[2:] - 2.0 * m_cp[1:-1] + m_cp[:-2]
        s2 = torch.mean(d2 ** 2)

    return s1, s2

def build_regularization_hessian_cp(
    Nc,
    lam_prior=0.0,
    lam_smooth1=0.0,
    lam_smooth2=0.0,
    device="cpu",
    dtype=torch.float32,
):
    """
    control-point 空间正则 Hessian
    """
    I = torch.eye(Nc, device=device, dtype=dtype)
    Hreg = lam_prior * I

    if Nc >= 2 and lam_smooth1 > 0.0:
        D1 = torch.zeros((Nc - 1, Nc), device=device, dtype=dtype)
        idx = torch.arange(Nc - 1, device=device)
        D1[idx, idx] = -1.0
        D1[idx, idx + 1] = 1.0
        Hreg = Hreg + lam_smooth1 * (D1.T @ D1)

    if Nc >= 3 and lam_smooth2 > 0.0:
        D2 = torch.zeros((Nc - 2, Nc), device=device, dtype=dtype)
        idx = torch.arange(Nc - 2, device=device)
        D2[idx, idx] = 1.0
        D2[idx, idx + 1] = -2.0
        D2[idx, idx + 2] = 1.0
        Hreg = Hreg + lam_smooth2 * (D2.T @ D2)


    return Hreg



def compute_jacobian_wrt_cp_full_single(
    model,
    x_template_1_4_h,
    periods_1_nt,
    cp_idx,
):
    """
    对 control-point 参数 q=[vp_cp,vs_cp,rho_cp] 计算 Jacobian
    返回:
      J: [2*NT, 3*Nc]
      q0: [3*Nc]
    """
    x_template = x_template_1_4_h[0].detach().clone()   # [4,H]
    depth_full = x_template[0]
    Nc = cp_idx.numel()

    vp_cp0 = x_template[1, cp_idx].detach().clone()
    vs_cp0 = x_template[2, cp_idx].detach().clone()
    rho_cp0 = x_template[3, cp_idx].detach().clone()

    q0 = torch.cat([vp_cp0, vs_cp0, rho_cp0], dim=0)    # [3Nc]
    periods_1_nt = periods_1_nt.detach()

    def f_q(q):
        vp_cp = q[:Nc]
        vs_cp = q[Nc:2*Nc]
        rho_cp = q[2*Nc:3*Nc]

        x_cur, _, _, _ = make_forward_model_input_from_cp(
            x_template, vp_cp, vs_cp, rho_cp, cp_idx
        )
        mu, _ = model(x_cur, periods=periods_1_nt)
        return mu.reshape(-1)   # [2*NT]

    J = torch.autograd.functional.jacobian(f_q, q0, create_graph=False)
    return J.detach(), q0.detach()




def invert_one_sample(
    model,
    x_init_4h,              # [4,H]
    d_obs_1_2_nt,           # [1,2,NT]
    periods_1_nt,           # [1,NT]
    n_iter=1000,
    lr=0.01,
    lam_prior_vp=2e-1,
    lam_prior_vs=5e-2,
    lam_prior_rho=1.0,
    lam_smooth1_vp=2e-3,
    lam_smooth1_vs=2e-3,
    lam_smooth1_rho=2e-3,
    lam_smooth2_vp=2e-1,
    lam_smooth2_vs=2e-1,
    lam_smooth2_rho=2e-1,
    fisher_damping=1e-4,
    device="cpu",
    print_every=100,
    compute_fisher=True,
):
    """
    control-point 参数化的三参数联合反演：
      - 反演 vp_cp, vs_cp, rho_cp
      - 插值到完整深度网格后送入 surrogate
      - Fisher 在 control-point 空间计算
      - 最后把 Vs 后验标准差插值回完整网格
    """
    model.eval()

    x_init_4h = x_init_4h.to(device)
    d_obs_1_2_nt = d_obs_1_2_nt.to(device)
    periods_1_nt = periods_1_nt.to(device)

    depth_full = x_init_4h[0].detach().clone()
    cp_idx = select_control_indices(depth_full)
    depth_cp = depth_full[cp_idx]
    Nc = cp_idx.numel()

    vp_cp_init = x_init_4h[1, cp_idx].detach().clone()
    vs_cp_init = x_init_4h[2, cp_idx].detach().clone()
    rho_cp_init = x_init_4h[3, cp_idx].detach().clone()

    z_vp = inverse_softplus(vp_cp_init).detach().clone().requires_grad_(True)
    z_vs = inverse_softplus(vs_cp_init).detach().clone().requires_grad_(True)
    z_rho = inverse_softplus(rho_cp_init).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([z_vp, z_vs, z_rho], lr=lr)
    loss_hist = []

    for it in range(n_iter):
        optimizer.zero_grad()

        vp_cp = torch.nn.functional.softplus(z_vp) + 1e-4
        vs_cp = torch.nn.functional.softplus(z_vs) + 1e-4
        rho_cp = torch.nn.functional.softplus(z_rho) + 1e-4

        x_cur, vp_full, vs_full, rho_full = make_forward_model_input_from_cp(
            x_init_4h, vp_cp, vs_cp, rho_cp, cp_idx
        )
        pred, _ = model(x_cur, periods=periods_1_nt)

        # data
        data_loss = torch.mean((pred - d_obs_1_2_nt) ** 2)

        # prior to initial model in control-point space
        prior_vp = torch.mean((vp_cp - vp_cp_init) ** 2)
        prior_vs = torch.mean((vs_cp - vs_cp_init) ** 2)
        prior_rho = torch.mean((rho_cp - rho_cp_init) ** 2)
        prior_loss = (
            lam_prior_vp * prior_vp +
            lam_prior_vs * prior_vs +
            lam_prior_rho * prior_rho
        )

        # smoothness in control-point space
        s1_vp, s2_vp = smoothness_losses_cp(vp_cp)
        s1_vs, s2_vs = smoothness_losses_cp(vs_cp)
        s1_rho, s2_rho = smoothness_losses_cp(rho_cp)

        smooth_loss = (
            lam_smooth1_vp * s1_vp + lam_smooth2_vp * s2_vp +
            lam_smooth1_vs * s1_vs + lam_smooth2_vs * s2_vs +
            lam_smooth1_rho * s1_rho + lam_smooth2_rho * s2_rho
        )

        loss = data_loss + prior_loss + smooth_loss
        loss.backward()
        optimizer.step()

        loss_hist.append(float(loss.item()))

        if print_every > 0 and ((it + 1) % print_every == 0 or it == 0):
            ratio_prior = prior_loss.item() / max(data_loss.item(), 1e-12)
            ratio_smooth = smooth_loss.item() / max(data_loss.item(), 1e-12)
            print(
                f"  iter {it+1:4d}/{n_iter} | "
                f"loss={loss.item():.6e} | "
                f"data={data_loss.item():.6e} | "
                f"prior={prior_loss.item():.6e} | "
                f"smooth={smooth_loss.item():.6e} | "
                f"ratio_prior/data={ratio_prior:.3e} | "
                f"ratio_smooth/data={ratio_smooth:.3e}"
            )

    with torch.no_grad():
        # initial full model/prediction
        x_init_full, vp_init_full, vs_init_full, rho_init_full = make_forward_model_input_from_cp(
            x_init_4h, vp_cp_init, vs_cp_init, rho_cp_init, cp_idx
        )
        pred_init, _ = model(x_init_full, periods=periods_1_nt)

        # final full model/prediction
        vp_cp_inv = torch.nn.functional.softplus(z_vp) + 1e-4
        vs_cp_inv = torch.nn.functional.softplus(z_vs) + 1e-4
        rho_cp_inv = torch.nn.functional.softplus(z_rho) + 1e-4

        x_inv, vp_inv, vs_inv, rho_inv = make_forward_model_input_from_cp(
            x_init_4h, vp_cp_inv, vs_cp_inv, rho_cp_inv, cp_idx
        )
        pred_final, _ = model(x_inv, periods=periods_1_nt)

    # Fisher in control-point space
    if compute_fisher:
        J, q0 = compute_jacobian_wrt_cp_full_single(model, x_inv, periods_1_nt, cp_idx)
        F = J.T @ J

        Hreg_vp = build_regularization_hessian_cp(
            Nc,
            lam_prior=lam_prior_vp,
            lam_smooth1=lam_smooth1_vp,
            lam_smooth2=lam_smooth2_vp,
            device=F.device,
            dtype=F.dtype,
        )
        Hreg_vs = build_regularization_hessian_cp(
            Nc,
            lam_prior=lam_prior_vs,
            lam_smooth1=lam_smooth1_vs,
            lam_smooth2=lam_smooth2_vs,
            device=F.device,
            dtype=F.dtype,
        )
        Hreg_rho = build_regularization_hessian_cp(
            Nc,
            lam_prior=lam_prior_rho,
            lam_smooth1=lam_smooth1_rho,
            lam_smooth2=lam_smooth2_rho,
            device=F.device,
            dtype=F.dtype,
        )

        Hreg = torch.zeros((3 * Nc, 3 * Nc), device=F.device, dtype=F.dtype)
        Hreg[0:Nc, 0:Nc] = Hreg_vp
        Hreg[Nc:2*Nc, Nc:2*Nc] = Hreg_vs
        Hreg[2*Nc:3*Nc, 2*Nc:3*Nc] = Hreg_rho

        Hpost = F + Hreg + fisher_damping * torch.eye(3 * Nc, device=F.device, dtype=F.dtype)
        C_post = torch.linalg.inv(Hpost)

        C_vs_cp = C_post[Nc:2*Nc, Nc:2*Nc]
        std_vs_cp = torch.sqrt(torch.clamp(torch.diag(C_vs_cp), min=0.0))
        std_post_vs = interp_control_to_full(depth_full, depth_cp, std_vs_cp)

    else:
        J = None
        F = None
        Hpost = None
        C_post = None
        std_vs_cp = torch.zeros_like(vs_cp_inv)
        std_post_vs = torch.zeros_like(vs_inv)

    Hreg_vp = build_regularization_hessian_cp(
        Nc,
        lam_prior=lam_prior_vp,
        lam_smooth1=lam_smooth1_vp,
        lam_smooth2=lam_smooth2_vp,
        device=F.device,
        dtype=F.dtype,
    )
    Hreg_vs = build_regularization_hessian_cp(
        Nc,
        lam_prior=lam_prior_vs,
        lam_smooth1=lam_smooth1_vs,
        lam_smooth2=lam_smooth2_vs,
        device=F.device,
        dtype=F.dtype,
    )
    Hreg_rho = build_regularization_hessian_cp(
        Nc,
        lam_prior=lam_prior_rho,
        lam_smooth1=lam_smooth1_rho,
        lam_smooth2=lam_smooth2_rho,
        device=F.device,
        dtype=F.dtype,
    )

    Hreg = torch.zeros((3 * Nc, 3 * Nc), device=F.device, dtype=F.dtype)
    Hreg[0:Nc, 0:Nc] = Hreg_vp
    Hreg[Nc:2*Nc, Nc:2*Nc] = Hreg_vs
    Hreg[2*Nc:3*Nc, 2*Nc:3*Nc] = Hreg_rho

    Hpost = F + Hreg + fisher_damping * torch.eye(3 * Nc, device=F.device, dtype=F.dtype)
    C_post = torch.linalg.inv(Hpost)

    # 提取 Vs control-point posterior std，并插值到完整深度
    C_vs_cp = C_post[Nc:2*Nc, Nc:2*Nc]
    std_vs_cp = torch.sqrt(torch.clamp(torch.diag(C_vs_cp), min=0.0))
    std_post_vs = interp_control_to_full(depth_full, depth_cp, std_vs_cp)

    return {
        "cp_idx": cp_idx.detach().cpu(),
        "depth_cp": depth_cp.detach().cpu(),

        "vp_cp_init": vp_cp_init.detach().cpu(),
        "vs_cp_init": vs_cp_init.detach().cpu(),
        "rho_cp_init": rho_cp_init.detach().cpu(),

        "vp_cp_inv": vp_cp_inv.detach().cpu(),
        "vs_cp_inv": vs_cp_inv.detach().cpu(),
        "rho_cp_inv": rho_cp_inv.detach().cpu(),

        "vp_init": vp_init_full.detach().cpu(),
        "vs_init": vs_init_full.detach().cpu(),
        "rho_init": rho_init_full.detach().cpu(),

        "vp_inv": vp_inv.detach().cpu(),
        "vs_inv": vs_inv.detach().cpu(),
        "rho_inv": rho_inv.detach().cpu(),

        "pred_init": pred_init.detach().cpu(),
        "pred_final": pred_final.detach().cpu(),

        "loss_hist": np.asarray(loss_hist, dtype=float),

        "J": None if J is None else J.detach().cpu(),
        "F": None if F is None else F.detach().cpu(),
        "Hpost": None if Hpost is None else Hpost.detach().cpu(),
        "C_post": None if C_post is None else C_post.detach().cpu(),

        "std_post_vs": std_post_vs.detach().cpu(),
        "std_vs_cp": std_vs_cp.detach().cpu(),
    }

def invert_one_sample_old(
    model,
    x_init_4h,              # [4,H], mean initial model
    d_obs_1_2_nt,           # [1,2,NT]
    periods_1_nt,           # [1,NT]
    n_iter=600,
    lr=0.03,
    tv_weight=1e-4,
    fisher_damping=1e-4,
    device="cpu",
    print_every=100,
    compute_fisher=True,
):
    """
    只对 Vs 做梯度反演。
    """
    model.eval()

    x_init_4h = x_init_4h.to(device)
    d_obs_1_2_nt = d_obs_1_2_nt.to(device)
    periods_1_nt = periods_1_nt.to(device)

    vs_init = x_init_4h[2].detach().clone()
    z = inverse_softplus(vs_init).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=lr)
    loss_hist = []

    for it in range(n_iter):
        optimizer.zero_grad()

        vs = torch.nn.functional.softplus(z) + 1e-4
        x_cur = make_forward_model_input(x_init_4h, vs)  # [1,4,H]
        pred, _ = model(x_cur, periods=periods_1_nt)     # [1,2,NT]

        data_loss = torch.mean((pred - d_obs_1_2_nt) ** 2)

        # 一个很弱的 TV 正则，防止出现锯齿
        tv_loss = torch.mean(torch.abs(vs[1:] - vs[:-1]))

        loss = data_loss + tv_weight * tv_loss
        loss.backward()
        optimizer.step()

        loss_hist.append(float(loss.item()))

        if print_every > 0 and ((it + 1) % print_every == 0 or it == 0):
            print(
                f"  iter {it+1:4d}/{n_iter} | "
                f"loss={loss.item():.6e} | "
                f"data={data_loss.item():.6e} | "
                f"tv={tv_loss.item():.6e}"
            )

    with torch.no_grad():
        # initial prediction
        x_init = make_forward_model_input(x_init_4h, vs_init)
        pred_init, _ = model(x_init, periods=periods_1_nt)

        # final prediction
        vs_inv = torch.nn.functional.softplus(z) + 1e-4
        x_inv = make_forward_model_input(x_init_4h, vs_inv)
        pred_final, _ = model(x_inv, periods=periods_1_nt)

    # Fisher / Gauss-Newton 近似
    # sigma_d^2 = 1 -> F = J^T J
    J, _ = compute_jacobian_wrt_vs_single(model, x_inv, periods_1_nt)  # [2*NT,H]
    F = J.T @ J
    Hm = F + fisher_damping * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
    C_post = torch.linalg.inv(Hm)
    std_post = torch.sqrt(torch.clamp(torch.diag(C_post), min=0.0))

    return {
        "vs_init": vs_init.detach().cpu(),
        "vs_inv": vs_inv.detach().cpu(),
        "pred_init": pred_init.detach().cpu(),
        "pred_final": pred_final.detach().cpu(),
        "loss_hist": np.asarray(loss_hist, dtype=float),
        "J": J.detach().cpu(),
        "F": F.detach().cpu(),
        "C_post": C_post.detach().cpu(),
        "std_post": std_post.detach().cpu(),
    }

def plot_inversion_results_4_with_dispersion_old(
    depth_km,               # [H]
    results,                # list of 4 dict
    out_path,
    title_prefix="Surrogate-based gradient inversion",
):
    """
    4 个样本，4x2 布局：
      左列：Vs structure
      右列：dispersion fit

    每个左图:
      - true Vs
      - initial Vs
      - inverted Vs
      - inverted ± 2 sigma_post

    每个右图:
      - observed dispersion
      - predicted at initial model
      - predicted at final model
      Rayleigh / Love 同时画出
    """
    n = len(results)
    assert n == 4, "Please provide exactly 4 inversion results."

    depth_np = depth_km.detach().cpu().numpy()

    fig, axes = plt.subplots(
        4, 2, figsize=(12.5, 16.0),
        gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    for i, res in enumerate(results):
        # ---------------- left: Vs model ----------------
        ax = axes[i, 0]

        vs_true = res["vs_true"].numpy()
        vs_init = res["vs_init"].numpy()
        vs_inv  = res["vs_inv"].numpy()
        std = res["std_post_vs"].numpy()

        lower = vs_inv - 2.0 * std
        upper = vs_inv + 2.0 * std

        ax.fill_betweenx(
            depth_np, lower, upper,
            color="tab:blue", alpha=0.18, label=r"Inverted $\pm 2\sigma$"
        )
        ax.plot(vs_true, depth_np, "k-", lw=2.0, label="True")
        ax.plot(vs_init, depth_np, color="0.55", lw=1.8, ls="--", label="Initial")
        ax.plot(vs_inv,  depth_np, color="tab:red", lw=2.0, label="Inverted")

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_ylabel("Depth (km)")
        if i == 3:
            ax.set_xlabel(r"$V_s$ (km/s)")
        ax.set_title(
            f"(#{res['sample_id']}) Velocity model | final loss={res['final_loss']:.2e}",
            fontsize=11
        )

        # ---------------- right: dispersion fit ----------------
        ax = axes[i, 1]

        periods = res["periods"].numpy()             # [NT]
        d_obs = res["d_obs"].numpy()                 # [2,NT]
        d_init = res["pred_init"][0].numpy()         # [2,NT]
        d_final = res["pred_final"][0].numpy()       # [2,NT]

        # Rayleigh
        ax.plot(periods, d_obs[0], "o", ms=3.2, color="k", label="Observed Rayleigh")
        ax.plot(periods, d_init[0], ls="--", lw=1.8, color="tab:blue", label="Initial Rayleigh")
        ax.plot(periods, d_final[0], lw=2.0, color="tab:red", label="Final Rayleigh")

        # Love
        ax.plot(periods, d_obs[1], "s", ms=3.0, color="0.35", label="Observed Love")
        ax.plot(periods, d_init[1], ls="--", lw=1.8, color="tab:green", label="Initial Love")
        ax.plot(periods, d_final[1], lw=2.0, color="tab:orange", label="Final Love")

        ax.grid(alpha=0.2)
        ax.set_ylabel("Phase velocity")
        if i == 3:
            ax.set_xlabel("Period (s)")
        ax.set_title(f"(#{res['sample_id']}) Dispersion fit", fontsize=11)

    # 统一 legend：左列一个，右列一个
    handles_l, labels_l = axes[0, 0].get_legend_handles_labels()
    handles_r, labels_r = axes[0, 1].get_legend_handles_labels()

    fig.legend(
        handles_l, labels_l,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.01),
        ncol=4,
        frameon=False
    )
    fig.legend(
        handles_r, labels_r,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        ncol=3,
        frameon=False
    )

    fig.suptitle(title_prefix, y=0.995, fontsize=14)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.08,
        wspace=0.22,
        hspace=0.28,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_inversion_results_4_with_dispersion(
    depth_km,               # [H]
    results,                # list of 4 dict
    out_path,
    title_prefix=None,
):
    """
    GRL-style figure:
      - 4 samples, 4x2 layout
      - left: Vs structure
      - right: dispersion fit
      - panel labels (a)-(h)
      - publication-friendly fonts, colors, and spacing
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pickle 
    n = len(results)
    assert n == 4, "Please provide exactly 4 inversion results."

    # ---------------- publication style ----------------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.6,
        "savefig.dpi": 400,
        "figure.dpi": 200,
    })

    depth_np = depth_km.detach().cpu().numpy()

    # 单栏偏宽或双栏半页风格
    fig, axes = plt.subplots(
        4, 2,
        figsize=(7.2, 9.6),
        gridspec_kw={"width_ratios": [1.0, 1.22]}
    )

    # -------- colors: muted, print-friendly --------
    c_true = "0.10"         # near black
    c_init = "0.55"         # gray
    c_inv  = "#b2182b"      # muted dark red
    c_band = "#92c5de"      # soft blue

    c_obs_R   = "0.10"
    c_init_R  = "#4393c3"
    c_final_R = "#b2182b"

    c_obs_L   = "0.35"
    c_init_L  = "#4d9221"
    c_final_L = "#ef8a62"

    letters = [chr(ord("a") + i) for i in range(8)]

    for i, res in enumerate(results):
        # ================= left: Vs structure =================
        ax = axes[i, 0]

        vs_true = res["vs_true"].numpy()
        vs_init = res["vs_init"].numpy()
        vs_inv  = res["vs_inv"].numpy()
        std     = res["std_post_vs"].numpy()

        lower = vs_inv - 2.0 * std * 0.005
        upper = vs_inv + 2.0 * std * 0.005

        ax.fill_betweenx(
            depth_np, lower, upper,
            color=c_band, alpha=0.28, lw=0.0,
            label=r"Inverted $\pm 2\sigma$"
        )
        ax.plot(vs_true, depth_np, color=c_true, lw=1.7, label="True")
        ax.plot(vs_init, depth_np, color=c_init, lw=1.3, ls="--", label="Initial")
        ax.plot(vs_inv,  depth_np, color=c_inv,  lw=1.7, label="Inverted")

        ax.invert_yaxis()
        ax.grid(True, color="0.88", lw=0.5, alpha=0.9)
        ax.set_ylabel("Depth (km)")

        if i == 3:
            ax.set_xlabel(r"$V_s$ (km s$^{-1}$)")

        ax.set_title(
            f"Sample {res['sample_id']}: velocity model",
            pad=4.0
        )

        ax.text(
            0.02, 0.96, f"({letters[2*i]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold"
        )

        # 统一深度范围
        ax.set_ylim(depth_np.max(), depth_np.min())

        # ================= right: dispersion fit =================
        ax = axes[i, 1]

        periods = res["periods"].numpy()             # [NT]
        d_obs = res["d_obs"].numpy()                 # [2,NT]
        d_init = res["pred_init"][0].numpy()         # [2,NT]
        d_final = res["pred_final"][0].numpy()       # [2,NT]

        # Rayleigh
        ax.plot(
            periods, d_obs[0],
            linestyle="None", marker="o", ms=2.8,
            mec=c_obs_R, mfc="white", mew=0.9,
            color=c_obs_R, label="Observed Rayleigh"
        )
        ax.plot(
            periods, d_init[0],
            ls="--", lw=1.3, color=c_init_R,
            label="Initial Rayleigh"
        )
        ax.plot(
            periods, d_final[0],
            ls="-", lw=1.6, color=c_final_R,
            label="Final Rayleigh"
        )

        # Love
        ax.plot(
            periods, d_obs[1],
            linestyle="None", marker="s", ms=2.6,
            mec=c_obs_L, mfc="white", mew=0.9,
            color=c_obs_L, label="Observed Love"
        )
        ax.plot(
            periods, d_init[1],
            ls="--", lw=1.3, color=c_init_L,
            label="Initial Love"
        )
        ax.plot(
            periods, d_final[1],
            ls="-", lw=1.6, color=c_final_L,
            label="Final Love"
        )

        ax.grid(True, color="0.88", lw=0.5, alpha=0.9)
        ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")

        if i == 3:
            ax.set_xlabel("Period (s)")

        ax.set_title(
            f"Sample {res['sample_id']}: dispersion fit",
            pad=4.0
        )

        ax.text(
            0.02, 0.96, f"({letters[2*i+1]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold"
        )

    # ---------------- legends ----------------
    handles_l, labels_l = axes[0, 0].get_legend_handles_labels()
    handles_r, labels_r = axes[0, 1].get_legend_handles_labels()

    # 左列 legend
    fig.legend(
        handles_l, labels_l,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.012),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    # 右列 legend
    fig.legend(
        handles_r, labels_r,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.012),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    # 不建议主文图再加大标题
    if title_prefix is not None and len(str(title_prefix)) > 0:
        fig.suptitle(title_prefix, y=0.995, fontsize=10)

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.965,
        bottom=0.085,
        wspace=0.26,
        hspace=0.34,
    )

    fig.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fisher_matrices_8(results, out_path):
    """
    8 个样本 Fisher 矩阵热图，2x4。
    """
    n = len(results)
    assert n == 8

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    vmax = 0.0
    for r in results:
        F = r["F"].numpy()
        vmax = max(vmax, np.percentile(np.abs(F), 99))

    for ax, res in zip(axes, results):
        F = res["F"].numpy()
        im = ax.imshow(F, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"Sample {res['sample_id']}", fontsize=10)
        ax.set_xlabel("Model index")
        ax.set_ylabel("Model index")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Fisher information")
    fig.subplots_adjust(wspace=0.28, hspace=0.30)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_histories_8(results, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, res in zip(axes, results):
        ax.plot(res["loss_hist"], lw=1.8)
        ax.set_title(f"Sample {res['sample_id']}", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)

    fig.subplots_adjust(wspace=0.25, hspace=0.30)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_inversion_test(
    ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",
    fig_dir="tfig/inversion_fisher",
    device=None,
    dataset_seed=2026,
    n_mean_init=100,
    n_invert=4,
    n_iter=800,
    lr=0.02,
    fisher_damping=1e-4,
):
    os.makedirs(fig_dir, exist_ok=True)

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print("Device:", device)

    dataset = SurfaceWaveDataset(
        n_samples=100000,
        z_max_km=150.0,
        z_max_num=256,
        dz_km=0.5,
        seed=dataset_seed,
    )

    # 用一个很小的 loader 只为构建网络维度
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model, H, NT = build_model_from_loader_batch(loader, ckpt_path, device)
    print(f"H={H}, NT={NT}")

    # 平均初始模型
    x_mean = compute_mean_initial_model(dataset, n_mean=n_mean_init)  # [4,H]
    depth_km = x_mean[0].clone()

    # 选 8 个目标样本
    target_ids = list(range(n_mean_init, n_mean_init + n_invert))
    print("Target sample ids:", target_ids)

    #plot_inversion_results_8(
    #    depth_km=depth_km,
    #    results=results,
    #    out_path=os.path.join(fig_dir, "inversion_vs_compare_8samples.png"),
    #    title_prefix="Surrogate-based gradient inversion with Fisher posterior uncertainty",
    #)
    #plot_fisher_matrices_8(
    #    results=results,
    #    out_path=os.path.join(fig_dir, "fisher_matrices_8samples.png"),
    #)
    #plot_loss_histories_8(
    #    results=results,
    #    out_path=os.path.join(fig_dir, "inversion_loss_histories_8samples.png"),
    #)
    results = load_results_pickle(f"{fig_dir}/results.pkl")
    plot_inversion_results_4_with_dispersion(
        depth_km=depth_km,
        results=results,
        out_path=os.path.join(fig_dir, "inversion_vs_dispersion_compare_4samples.png"),
        title_prefix="Surrogate-based gradient inversion and local Fisher posterior uncertainty",
    )
    print("\nSaved figures to:", fig_dir)


if __name__ == "__main__":
    seed_everything(SEED)
    from torch.utils.data import DataLoader
    rng = torch.Generator()
    rng.manual_seed(SEED)

    run_inversion_test(
        ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",
        fig_dir="tfig/inversion_fisher",
        device=None,
        dataset_seed=2026,
        n_mean_init=100,
        n_invert=4,
        n_iter=600,
        lr=0.03,
        fisher_damping=1e-4,
    )