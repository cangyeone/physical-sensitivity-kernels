import csv
from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 替换：用 Transformer
from models.struct2disp_transformer import Struct2DispTransformer
from utils.generate_data_weak_prior import SurfaceWaveDataset
import random 
plt.switch_backend("Agg")



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
    verbose: bool = False,
):
    """
    用 disba 计算 phase velocity 对 Vs 的敏感核 (parameter="velocity_s")。
    若某个周期失败，则该周期整行填 NaN，不报错。

    返回:
      K_disba: [NT,H]
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

    NT = len(periods_s)
    K_disba = np.full((NT, H), np.nan, dtype=float)

    for i, T in enumerate(periods_s):
        try:
            out = ps(float(T), mode=mode, wave=wave, parameter="velocity_s")
            depth_out = np.asarray(out.depth, dtype=float)
            k = np.asarray(out.kernel, dtype=float)
            K_disba[i] = _interp1d_to_depth(depth_out, k, depth_km)
        except Exception as e:
            if verbose:
                print(f"[skip] disba failed: wave={wave}, T={float(T):.3f}, err={e}")
            continue

    return K_disba

def get_period_bin_labels_and_masks(periods_1d: np.ndarray):
    """
    periods_1d: [NT]
    返回:
      bin_defs: [(label, mask), ...]
    """
    edges = [2, 5, 10, 20, 30, 45, 60]
    bin_defs = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (periods_1d >= lo) & (periods_1d < hi)
            label = f"[{lo},{hi})"
        else:
            mask = (periods_1d >= lo) & (periods_1d <= hi)
            label = f"[{lo},{hi}]"
        bin_defs.append((label, mask))

    return bin_defs


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    sx = np.sqrt((x * x).sum())
    sy = np.sqrt((y * y).sum())
    if sx < eps or sy < eps:
        return np.nan
    return float((x * y).sum() / (sx * sy))


def safe_cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = np.sqrt((x * x).sum())
    ny = np.sqrt((y * y).sum())
    if nx < eps or ny < eps:
        return np.nan
    return float((x * y).sum() / (nx * ny))


def kernel_metrics_1period(k_nn: np.ndarray, k_th: np.ndarray, eps: float = 1e-12):
    """
    单个周期、单个波型的一条深度核比较
    输入 shape: [H]
    若存在 NaN/Inf 或有效点太少，则返回 None
    """
    k_nn = np.asarray(k_nn, dtype=float)
    k_th = np.asarray(k_th, dtype=float)

    valid = np.isfinite(k_nn) & np.isfinite(k_th)
    if valid.sum() < 2:
        return None

    x = k_nn[valid]
    y = k_th[valid]

    diff = x - y
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    corr = safe_corrcoef(x, y, eps=eps)
    cosine = safe_cosine_similarity(x, y, eps=eps)

    if not (np.isfinite(rmse) and np.isfinite(mae)):
        return None

    return {
        "cosine": cosine,
        "corr": corr,
        "rmse": rmse,
        "mae": mae,
    }


def summarize_metric_list(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "count": 0,
        }
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p16": float(np.percentile(vals, 16)),
        "p84": float(np.percentile(vals, 84)),
        "count": int(vals.size),
    }


def evaluate_200_samples_stats(
    loader,
    ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",
    out_dir="tfig/eval200",
    device=None,
    n_eval=200,
    smooth_win=9,
    norm_mode="l1abs",
    assume_units_kms_gcc=True,
    max_depth_km=150.0,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print("Device:", device)

    # 先取一个 batch 读尺寸
    model_batch, disp_batch, mask_batch = next(iter(loader))
    B, C_model, H = model_batch.shape
    _, C_disp, NT = disp_batch.shape
    print(f"H={H}, NT={NT}")

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

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded ckpt:", ckpt_path)
    else:
        raise FileNotFoundError(f"ckpt not exists: {ckpt_path}")

    model.eval()
    _disable_inplace(model)

    # 按波型和周期区间累计
    disp_stats = {
        "rayleigh": defaultdict(list),
        "love": defaultdict(list),
    }
    kernel_stats = {
        "rayleigh": {
            "cosine": defaultdict(list),
            "corr": defaultdict(list),
            "rmse": defaultdict(list),
            "mae": defaultdict(list),
        },
        "love": {
            "cosine": defaultdict(list),
            "corr": defaultdict(list),
            "rmse": defaultdict(list),
            "mae": defaultdict(list),
        },
    }

    n_done = 0

    for model_batch, disp_batch, mask_batch in loader:
        if n_done >= n_eval:
            break

        x0 = model_batch.to(device).detach()      # [1,4,H]
        disp_batch = disp_batch.to(device).detach()

        # 深度与周期
        depth_km = x0[0, 0, :].detach().cpu()     # [H]
        periods = disp_batch[:, 0, :].to(device)  # [1,NT]
        periods_s = periods[0].detach().cpu()
        periods_np = periods_s.numpy()

        # 深度截取
        depth_np = depth_km.numpy()
        depth_keep = (depth_np >= 0.0) & (depth_np <= float(max_depth_km))

        # ---------- 1) 频散预测 ----------
        with torch.no_grad():
            mu, _ = model(x0, periods=periods)    # [1,2,NT]

        pred_R = mu[0, 0, :].detach().cpu().numpy()
        pred_L = mu[0, 1, :].detach().cpu().numpy()

        true_R = disp_batch[0, 1, :].detach().cpu().numpy()
        true_L = disp_batch[0, 2, :].detach().cpu().numpy()

        abs_err_R = np.abs(pred_R - true_R)
        abs_err_L = np.abs(pred_L - true_L)

        rel_err_R = abs_err_R / np.clip(np.abs(true_R), 1e-8, None)
        rel_err_L = abs_err_L / np.clip(np.abs(true_L), 1e-8, None)

        # ---------- 2) NN 核 ----------
        K, y, vs = compute_dcdvs_full_jacobian(model, x0, periods=periods)

        eps = 1e-12
        c = y.clamp_min(eps)
        K_log = (K * vs[:, None, None, :]) / c[:, :, :, None]   # [1,2,NT,H]

        dz_km = float(depth_km[1] - depth_km[0]) if H > 1 else 1.0
        K_log_dz = K_log * dz_km                                # [1,2,NT,H]
        K_log_dz_0 = K_log_dz[0].detach().cpu()                 # [2,NT,H]

        # 只截 0-150 km
        K_log_dz_0 = K_log_dz_0[:, :, depth_keep]

        # 平滑 + 归一化
        nn_R = normalize_each_period(
            smooth_1d_along_depth(K_log_dz_0[0], win=smooth_win),
            mode=norm_mode
        ).numpy()   # [NT,Hsub]
        nn_L = normalize_each_period(
            smooth_1d_along_depth(K_log_dz_0[1], win=smooth_win),
            mode=norm_mode
        ).numpy()

        # ---------- 3) disba 理论核 ----------
        vp_np = x0[0, 1, :].detach().cpu().numpy()
        vs_np = x0[0, 2, :].detach().cpu().numpy()
        rho_np = x0[0, 3, :].detach().cpu().numpy()


        Kd_R = disba_vs_phase_sensitivity(
            depth_np, vp_np, vs_np, rho_np, periods_np,
            wave="rayleigh", mode=0,
            assume_units_kms_gcc=assume_units_kms_gcc,
            verbose=False,
        )   # [NT,H]

        Kd_L = disba_vs_phase_sensitivity(
            depth_np, vp_np, vs_np, rho_np, periods_np,
            wave="love", mode=0,
            assume_units_kms_gcc=assume_units_kms_gcc,
            verbose=False,
        )   # [NT,H]

        Kd_R = Kd_R[:, depth_keep]
        Kd_L = Kd_L[:, depth_keep]

        # 这里先正常做；后面统计时若某周期有 NaN，就直接跳过
        th_R = normalize_each_period(
            smooth_1d_along_depth(torch.tensor(Kd_R, dtype=torch.float32), win=smooth_win),
            mode=norm_mode
        ).numpy()

        th_L = normalize_each_period(
            smooth_1d_along_depth(torch.tensor(Kd_L, dtype=torch.float32), win=smooth_win),
            mode=norm_mode
        ).numpy()

        # ---------- 4) 按周期区间累计 ----------
        bin_defs = get_period_bin_labels_and_masks(periods_np)

        for label, pmask in bin_defs:
            idxs = np.where(pmask)[0]
            if len(idxs) == 0:
                continue

            # 频散误差：把该区间所有周期都累计进去
            for v in abs_err_R[idxs]:
                if np.isfinite(v):
                    disp_stats["rayleigh"][f"{label}_mae"].append(float(v))
            for v in (100.0 * rel_err_R[idxs]):
                if np.isfinite(v):
                    disp_stats["rayleigh"][f"{label}_mape"].append(float(v))

            for v in abs_err_L[idxs]:
                if np.isfinite(v):
                    disp_stats["love"][f"{label}_mae"].append(float(v))
            for v in (100.0 * rel_err_L[idxs]):
                if np.isfinite(v):
                    disp_stats["love"][f"{label}_mape"].append(float(v))

            # 核误差：逐周期统计，再累积
            for j in idxs:
                mR = kernel_metrics_1period(nn_R[j], th_R[j])
                if mR is not None:
                    for k, v in mR.items():
                        if np.isfinite(v):
                            kernel_stats["rayleigh"][k][label].append(v)

                mL = kernel_metrics_1period(nn_L[j], th_L[j])
                if mL is not None:
                    for k, v in mL.items():
                        if np.isfinite(v):
                            kernel_stats["love"][k][label].append(v)

        n_done += 1
        if n_done % 10 == 0:
            print(f"Processed {n_done}/{n_eval}")

    # ---------- 5) 汇总成表 ----------
    disp_rows = []
    kernel_rows = []

    bin_defs = get_period_bin_labels_and_masks(periods_np)
    labels = [x[0] for x in bin_defs]

    for wave in ["rayleigh", "love"]:
        for label in labels:
            row = {
                "wave": wave,
                "period_bin": label,
            }

            s1 = summarize_metric_list(disp_stats[wave][f"{label}_mae"])
            s2 = summarize_metric_list(disp_stats[wave][f"{label}_mape"])

            row.update({
                "disp_mae_mean": s1["mean"],
                "disp_mae_std": s1["std"],
                "disp_mae_median": s1["median"],
                "disp_mape_mean_percent": s2["mean"],
                "disp_mape_std_percent": s2["std"],
                "disp_count": s1["count"],
            })
            disp_rows.append(row)

    for wave in ["rayleigh", "love"]:
        for label in labels:
            row = {
                "wave": wave,
                "period_bin": label,
            }

            for mname in ["cosine", "corr", "rmse", "mae"]:
                s = summarize_metric_list(kernel_stats[wave][mname][label])
                row[f"{mname}_mean"] = s["mean"]
                row[f"{mname}_std"] = s["std"]
                row[f"{mname}_median"] = s["median"]
                row[f"{mname}_count"] = s["count"]

            kernel_rows.append(row)

    # ---------- 6) 保存 CSV ----------
    disp_csv = os.path.join(out_dir, "dispersion_accuracy_200samples.csv")
    kern_csv = os.path.join(out_dir, "kernel_accuracy_200samples.csv")

    if len(disp_rows) > 0:
        with open(disp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(disp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(disp_rows)

    if len(kernel_rows) > 0:
        with open(kern_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(kernel_rows[0].keys()))
            writer.writeheader()
            writer.writerows(kernel_rows)

    # ---------- 7) 打印简表 ----------
    print("\n=== Dispersion accuracy summary ===")
    for row in disp_rows:
        print(
            f"{row['wave']:8s}  {row['period_bin']:8s}  "
            f"MAE={row['disp_mae_mean']:.5f}  "
            f"MAPE={row['disp_mape_mean_percent']:.3f}%  "
            f"n={row['disp_count']}"
        )

    print("\n=== Kernel accuracy summary ===")
    for row in kernel_rows:
        print(
            f"{row['wave']:8s}  {row['period_bin']:8s}  "
            f"cos={row['cosine_mean']:.4f}  "
            f"corr={row['corr_mean']:.4f}  "
            f"rmse={row['rmse_mean']:.4f}  "
            f"mae={row['mae_mean']:.4f}  "
            f"n={row['cosine_count']}"
        )

    print("\nSaved:")
    print(disp_csv)
    print(kern_csv)

    return disp_rows, kernel_rows


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    SEED = 2026
    seed_everything(SEED)
    from torch.utils.data import DataLoader
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ds = SurfaceWaveDataset(
        n_samples=100000,
        z_max_km=150.0,
        z_max_num=256,
        dz_km=0.5,
        seed=2026,
    )

    loader = DataLoader(
        ds,
        batch_size=1,      # 这里建议保持 1，最稳
        shuffle=False,
        num_workers=0,
        generator=rng,    # 只有 shuffle=True 时才用这个 generator 来固定随机数
    )

    evaluate_200_samples_stats(
        loader,
        ckpt_path="ckpt/struct2disp_transformer.v1.1a.pt",
        out_dir="tfig/eval200",
        device=None,
        n_eval=1000,
        smooth_win=9,
        norm_mode="l1abs",
        assume_units_kms_gcc=True,
        max_depth_km=110.0,
    )