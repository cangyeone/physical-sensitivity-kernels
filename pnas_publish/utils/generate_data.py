import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 经验密度公式：Brocher (2005)
# -----------------------------
def brocher_rho_from_vp(vp_kms: np.ndarray) -> np.ndarray:
    """
    Brocher (2005, BSSA) 经验关系：
    ρ(g/cm^3) = 1.6612 Vp - 0.4721 Vp^2 + 0.0671 Vp^3 - 0.0043 Vp^4 + 1.06e-4 Vp^5
    Vp 单位：km/s
    """
    v = vp_kms
    rho = (
        1.6612 * v
        - 0.4721 * v**2
        + 0.0671 * v**3
        - 0.0043 * v**4
        + 0.000106 * v**5
    )
    return rho


# -----------------------------
# tectonic-type 统计先验配置
# （数值基于 CRUST1.0/ECM1 + 文献典型值略微扩展）
# -----------------------------
TECTONIC_PRIORS = {
    # 洋壳：薄壳 + 高速基性地壳，上地幔相对较热
    "oceanic": {
        "crust_thickness_km": (5.0, 15.0),   # 壳厚 ~5–10 km，略放宽
        "sed_thickness_km":   (0.0, 3.0),
        "uc_fraction":        (0.5, 0.8),    # 上地壳占洋壳厚度的比例（简化处理）
        "vs_ranges": {
            "sediment":    (0.2, 2.5),
            "upper_crust": (3.2, 3.9),      # 洋壳基性上地壳，Vs 偏高
            "lower_crust": (3.8, 4.4),      # 下洋壳更高
            "upper_mantle": (4.2, 4.9),     # 0–100 km 上地幔
        },
        "vpvs_means": {  # 平均 Vp/Vs
            "sediment":    1.80,
            "upper_crust": 1.80,
            "lower_crust": 1.82,
            "upper_mantle": 1.85,
        },
        "vpvs_sigma": 0.05,
        # LVZ 配置：洋壳上地幔通常有明显 LVZ
        "lvz": {
            "prob": 0.7,
            "depth_center_km": (60.0, 120.0),
            "thickness_km":    (30.0, 60.0),
            "dvs_fraction":    (0.05, 0.12),  # 低速幅度（相对 Vs）
        },
    },

    # 稳定克拉通盾区：厚壳 + 冷上地幔，Vs 和壳厚都偏高
    "shield": {
        "crust_thickness_km": (35.0, 55.0),
        "sed_thickness_km":   (0.0, 3.0),
        "uc_fraction":        (0.3, 0.5),
        "vs_ranges": {
            "sediment":    (0.3, 2.5),
            "upper_crust": (3.0, 3.7),
            "lower_crust": (3.6, 4.2),
            "upper_mantle": (4.5, 5.2),     # 冷硬岩石圈 Vs 偏高
        },
        "vpvs_means": {
            "sediment":    1.75,
            "upper_crust": 1.73,
            "lower_crust": 1.76,
            "upper_mantle": 1.80,
        },
        "vpvs_sigma": 0.04,
        "lvz": {
            "prob": 0.3,   # 许多克拉通 LVZ 不明显
            "depth_center_km": (80.0, 140.0),
            "thickness_km":    (40.0, 70.0),
            "dvs_fraction":    (0.03, 0.08),
        },
    },

    # 平台 / 普通大陆地区
    "platform": {
        "crust_thickness_km": (28.0, 45.0),
        "sed_thickness_km":   (0.5, 8.0),
        "uc_fraction":        (0.3, 0.6),
        "vs_ranges": {
            "sediment":    (0.3, 2.8),
            "upper_crust": (2.8, 3.8),
            "lower_crust": (3.4, 4.2),
            "upper_mantle": (4.2, 5.0),
        },
        "vpvs_means": {
            "sediment":    1.78,
            "upper_crust": 1.75,
            "lower_crust": 1.78,
            "upper_mantle": 1.82,
        },
        "vpvs_sigma": 0.05,
        "lvz": {
            "prob": 0.5,
            "depth_center_km": (70.0, 130.0),
            "thickness_km":    (30.0, 60.0),
            "dvs_fraction":    (0.05, 0.10),
        },
    },

    # 造山带：厚壳 + 热的上地幔，壳内低速层和 LVZ 都更常见
    "orogen": {
        "crust_thickness_km": (40.0, 70.0),
        "sed_thickness_km":   (1.0, 10.0),
        "uc_fraction":        (0.3, 0.5),
        "vs_ranges": {
            "sediment":    (0.3, 2.8),
            "upper_crust": (2.7, 3.7),
            "lower_crust": (3.2, 4.0),
            "upper_mantle": (4.0, 4.8),     # 整体偏低速
        },
        "vpvs_means": {
            "sediment":    1.80,
            "upper_crust": 1.78,
            "lower_crust": 1.80,
            "upper_mantle": 1.86,
        },
        "vpvs_sigma": 0.06,
        "lvz": {
            "prob": 0.8,
            "depth_center_km": (60.0, 120.0),
            "thickness_km":    (40.0, 80.0),
            "dvs_fraction":    (0.07, 0.15),
        },
    },

    # 裂谷 / 弧后盆地：壳偏薄，上地幔更热，LVZ 很常见
    "rift": {
        "crust_thickness_km": (20.0, 40.0),
        "sed_thickness_km":   (1.0, 12.0),
        "uc_fraction":        (0.3, 0.6),
        "vs_ranges": {
            "sediment":    (0.3, 3.0),
            "upper_crust": (2.6, 3.6),
            "lower_crust": (3.2, 4.0),
            "upper_mantle": (4.0, 4.7),
        },
        "vpvs_means": {
            "sediment":    1.82,
            "upper_crust": 1.80,
            "lower_crust": 1.82,
            "upper_mantle": 1.88,
        },
        "vpvs_sigma": 0.06,
        "lvz": {
            "prob": 0.85,
            "depth_center_km": (50.0, 110.0),
            "thickness_km":    (30.0, 70.0),
            "dvs_fraction":    (0.07, 0.15),
        },
    },
}


def _sample_normal_trunc(rng, mean, sigma, lo, hi):
    """简单截断高斯采样，用在 Vp/Vs 上。"""
    while True:
        x = rng.normal(mean, sigma)
        if lo <= x <= hi:
            return x


def sample_global_1d_model(
    z_max_km: float = 100.0,
    dz_km: float = 0.5,
    tectonic_type: str | None = None,
    rng: np.random.Generator | None = None,
):
    """
    生成一个“全球先验”下的 1D Vs/Vp/ρ 模型（0–z_max_km）。
    - 区分 tectonic-type（oceanic / shield / platform / orogen / rift）
    - 按类型指定壳厚、沉积厚度、Vs 范围、Vp/Vs 分布
    - 上地幔可随机生成 LVZ（低速层）

    返回：
      depth_km: (N,)
      vs:       (N,)  km/s
      vp:       (N,)  km/s
      rho:      (N,)  g/cm^3
      meta:     dict  包含 tectonic_type, 各层厚度, LVZ 参数等
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) 随机选择 tectonic-type（若未指定）
    if tectonic_type is None:
        # 训练上可以偏向大陆：shield/platform/orogen/rift 多一点
        types = ["oceanic", "shield", "platform", "orogen", "rift"]
        weights = np.array([0.15, 0.25, 0.25, 0.2, 0.15])
        tectonic_type = rng.choice(types, p=weights)

    if tectonic_type not in TECTONIC_PRIORS:
        raise ValueError(f"Unknown tectonic_type: {tectonic_type}")

    cfg = TECTONIC_PRIORS[tectonic_type]
    vs_ranges = cfg["vs_ranges"]

    # 2) 壳厚 / 沉积层厚度 / 上地壳比例
    crust_thick = rng.uniform(*cfg["crust_thickness_km"])
    sed_thick = rng.uniform(*cfg["sed_thickness_km"])
    sed_thick = min(sed_thick, crust_thick * 0.7)  # 沉积厚度不能超过壳厚大部分

    uc_frac = rng.uniform(*cfg["uc_fraction"])
    uc_thick = uc_frac * (crust_thick - sed_thick)
    lc_thick = max(crust_thick - sed_thick - uc_thick, 2.0)

    z1 = sed_thick
    z2 = z1 + uc_thick
    moho = z1 + uc_thick + lc_thick  # 理论上≈ crust_thick（但可能略有调整）

    # 深度网格
    z_max_km = max(z_max_km, moho + 1.0)
    depth_km = np.arange(0.0, z_max_km + dz_km, dz_km)

    # 3) 生成 Vs 剖面：每层线性梯度 + 噪声 + clip + 平滑
    vs = np.zeros_like(depth_km)

    # 每层顶点 Vs
    sed_vs0 = rng.uniform(*vs_ranges["sediment"])
    uc_vs0  = rng.uniform(*vs_ranges["upper_crust"])
    lc_vs0  = rng.uniform(*vs_ranges["lower_crust"])
    um_vs0  = rng.uniform(*vs_ranges["upper_mantle"])

    # 梯度（km^-1），小到中等
    grad_sed = rng.uniform(0.00, 0.20)
    grad_uc  = rng.uniform(0.00, 0.06)
    grad_lc  = rng.uniform(0.00, 0.05)
    grad_um  = rng.uniform(0.00, 0.02)

    # 确保“正常情况”下：下地壳平均 Vs > 上地壳平均 Vs
    normal_case = rng.random() < 0.85
    if normal_case and lc_vs0 <= uc_vs0:
        lc_vs0 = min(uc_vs0 + rng.uniform(0.1, 0.3),
                     vs_ranges["lower_crust"][1] - 0.05)

    for i, z in enumerate(depth_km):
        if z <= z1:                    # 沉积层
            vs[i] = sed_vs0 + grad_sed * z
        elif z1 < z <= z2:             # 上地壳
            z_rel = z - z1
            vs[i] = uc_vs0 + grad_uc * z_rel
        elif z2 < z <= moho:           # 下地壳
            z_rel = z - z2
            vs[i] = lc_vs0 + grad_lc * z_rel
        else:                          # 上地幔
            z_rel = z - moho
            vs[i] = um_vs0 + grad_um * z_rel

    # 噪声
    noise = rng.normal(0.0, 0.05, size=vs.shape)
    vs += noise

    # 分层 clip 到各自的 Vs 范围
    mask_sed = depth_km <= z1
    mask_uc  = (depth_km > z1) & (depth_km <= z2)
    mask_lc  = (depth_km > z2) & (depth_km <= moho)
    mask_um  = depth_km > moho

    vs[mask_sed] = np.clip(vs[mask_sed], *vs_ranges["sediment"])
    vs[mask_uc]  = np.clip(vs[mask_uc],  *vs_ranges["upper_crust"])
    vs[mask_lc]  = np.clip(vs[mask_lc],  *vs_ranges["lower_crust"])
    vs[mask_um]  = np.clip(vs[mask_um],  *vs_ranges["upper_mantle"])

    # 简单滑动平均平滑
    window = 5
    kernel = np.ones(window) / window
    vs = np.convolve(vs, kernel, mode="same")

    # 允许少量小反转（0.1 km/s）
    for i in range(1, len(vs)):
        if vs[i] < vs[i-1] - 0.10:
            vs[i] = vs[i-1] - 0.10

    # 再用层平均修一次，保证 normal_case 下 mean_LC > mean_UC
    mean_uc = vs[mask_uc].mean()
    mean_lc = vs[mask_lc].mean()
    if normal_case and mean_lc <= mean_uc:
        delta = (mean_uc - mean_lc) + 0.10
        vs[mask_lc] += delta
        vs[mask_lc] = np.clip(vs[mask_lc], *vs_ranges["lower_crust"])

    # 4) 上地幔 LVZ：在 Vs 基础上叠加一个 Gauss 型低速层
    lvz_cfg = cfg["lvz"]
    has_lvz = rng.random() < lvz_cfg["prob"]
    lvz_params = None
    if has_lvz:
        zc = rng.uniform(*lvz_cfg["depth_center_km"])
        th = rng.uniform(*lvz_cfg["thickness_km"])
        dvs_frac = rng.uniform(*lvz_cfg["dvs_fraction"])
        sigma = th / 2.5

        mask_m = depth_km > moho
        z_m = depth_km[mask_m]
        gauss = np.exp(-0.5 * ((z_m - zc) / sigma) ** 2)

        dvs = dvs_frac * vs[mask_m] * gauss
        vs[mask_m] = vs[mask_m] - dvs

        # clip 回上地幔范围
        vs[mask_m] = np.clip(vs[mask_m], *vs_ranges["upper_mantle"])

        lvz_params = {
            "center_km": float(zc),
            "thickness_km": float(th),
            "dvs_fraction": float(dvs_frac),
        }

    # 5) Vp/Vs：按层给不同均值 + 截断高斯随机
    vp = np.zeros_like(vs)
    vpvs_means = cfg["vpvs_means"]
    sig = cfg["vpvs_sigma"]

    k_sed = _sample_normal_trunc(rng, vpvs_means["sediment"], sig, 1.6, 2.0)
    k_uc  = _sample_normal_trunc(rng, vpvs_means["upper_crust"], sig, 1.6, 2.0)
    k_lc  = _sample_normal_trunc(rng, vpvs_means["lower_crust"], sig, 1.6, 2.0)
    k_um  = _sample_normal_trunc(rng, vpvs_means["upper_mantle"], sig, 1.7, 2.0)

    vp[mask_sed] = vs[mask_sed] * k_sed
    vp[mask_uc]  = vs[mask_uc]  * k_uc
    vp[mask_lc]  = vs[mask_lc]  * k_lc
    vp[mask_um]  = vs[mask_um]  * k_um

    # 6) 密度
    rho = brocher_rho_from_vp(vp)

    meta = {
        "tectonic_type": tectonic_type,
        "crust_thickness_km": float(crust_thick),
        "sed_thickness_km": float(sed_thick),
        "z1_sediment_base": float(z1),
        "z2_upper_crust_base": float(z2),
        "moho": float(moho),
        "dz_km": float(dz_km),
        "z_max_km": float(z_max_km),
        "normal_case": bool(normal_case),
        "mean_vs_uc": float(mean_uc),
        "mean_vs_lc": float(mean_lc),
        "has_lvz": bool(has_lvz),
        "lvz_params": lvz_params,
        "vpvs_layers": {
            "sediment": float(k_sed),
            "upper_crust": float(k_uc),
            "lower_crust": float(k_lc),
            "upper_mantle": float(k_um),
        },
    }

    return depth_km, vs, vp, rho, meta


def plot_1d_model(depth_km, vs, vp, rho, meta=None,
                  show=True, save_path=None):
    """简单的 Vs/Vp/ρ 剖面图 + Moho + LVZ 标注。"""
    fig, ax1 = plt.subplots(figsize=(5, 8))

    ax1.plot(vs, depth_km, label="Vs", linewidth=2)
    ax1.plot(vp, depth_km, label="Vp", linestyle="--", linewidth=2)
    ax1.set_xlabel("Velocity (km/s)")
    ax1.set_ylabel("Depth (km)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twiny()
    ax2.plot(rho, depth_km, label="Density", alpha=0.7)
    ax2.set_xlabel("Density (g/cm³)")

    if meta is not None:
        z1 = meta.get("z1_sediment_base", None)
        z2 = meta.get("z2_upper_crust_base", None)
        moho = meta.get("moho", None)
        xmax = max(vs.max(), vp.max())

        for z, name, color in [
            (z1, "Sediment base", "gray"),
            (z2, "Upper crust base", "blue"),
            (moho, "Moho", "red"),
        ]:
            if z is not None:
                ax1.axhline(z, color=color, linestyle=":", alpha=0.7)
                ax1.text(
                    xmax * 0.98, z,
                    f" {name} ~ {z:.1f} km",
                    va="center", ha="right",
                    color=color, fontsize=9,
                )

        # tectonic-type & LVZ 标记
        text = f"{meta.get('tectonic_type','?')}"
        if meta.get("has_lvz", False):
            lp = meta.get("lvz_params", {})
            text += f" | LVZ @ ~{lp.get('center_km', 0):.0f} km"
        ax1.text(
            0.01, 0.02, text,
            transform=ax1.transAxes,
            ha="left", va="bottom",
            fontsize=9, color="k",
        )

    fig.suptitle("Global Prior 1D Velocity Model", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
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
        depth_km, vs, vp, rho, meta = sample_global_1d_model(
            z_max_km=self.z_max_km,
            dz_km=self.dz_km,
            tectonic_type=self.tectonic_type,
            rng=rng,
        )
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
