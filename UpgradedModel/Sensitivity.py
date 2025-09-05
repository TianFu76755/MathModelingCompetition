# -*- coding: utf-8 -*-
"""
灵敏度分析与可视化
- 四个扰动包（物理/窗口/预处理/算法）
- Tornado 图（Δd 排序）
- Bootstrap（抽样 + 直方图/小提琴）
- 轮廓代价 Profile J(d)
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from UpgradedModel.Model import run_dual_angle_fit, NModel, JointTwoBeamFitter, cos_theta_t, PreprocessConfig

# -*- coding: utf-8 -*-
"""
ReliabilityAnalysis.py
针对两束干涉厚度估计的可靠性分析：
- 四类扰动包（物理、窗口、预处理、算法）
- Tornado 图（按 |Δd| 排序）
- Bootstrap（返回样本、画直方/小提琴）
- Profile 轮廓曲线 J(d)：固定 d 网格，最小化其余参数

需要：UpgradedTwoBeam.py 中的 run_dual_angle_fit 与其返回的 fitter 对象。
不使用 seaborn，仅 matplotlib。
"""


def _extract_window_from_fitter(fitter) -> Tuple[Optional[float], Optional[float]]:
    """
    兼容两种写法：
      - fitter.fit_cfg.nu_min / nu_max
      - fitter.fit_cfg.fit_range_cm1 == (nu_min, nu_max)
    """
    cfg = getattr(fitter, "fit_cfg", None)
    if cfg is None:
        return None, None

    # 优先取 nu_min / nu_max
    nu_min = getattr(cfg, "nu_min", None)
    nu_max = getattr(cfg, "nu_max", None)
    if (nu_min is not None) or (nu_max is not None):
        return nu_min, nu_max

    # 兼容 fit_range_cm1
    fr = getattr(cfg, "fit_range_cm1", None)
    if fr is not None and isinstance(fr, (tuple, list)) and len(fr) == 2:
        return fr[0], fr[1]

    return None, None

# ================
# 基础跑一次
# ================
def fit_base(
    df1: pd.DataFrame, df2: pd.DataFrame,
    theta1=10.0, theta2=15.0,
    n_kind="constant", n0=2.60, B0=0.0,
    nu_window=(1200.0, 3800.0),
    fit_signal="detrended",
    loss="cauchy", lock_frac=0.15,
    do_bootstrap=False, n_boot=200
) -> Dict[str, Any]:
    """按照推荐配置跑一次，返回 run_dual_angle_fit 的结果字典"""
    return run_dual_angle_fit(
        df1, df2,
        theta1_deg=theta1, theta2_deg=theta2,
        n_kind=n_kind, n0_init=n0, B_init=B0,
        pre_cfg=PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2),
        fit_signal=fit_signal,
        loss=loss,
        do_bootstrap=do_bootstrap, n_boot=n_boot
    )

# =========================================================
# 一、四类扰动包（返回每个扰动下的厚度与 Δd）
# =========================================================
def run_sensitivity_packages(
    df1: pd.DataFrame, df2: pd.DataFrame,
    base_res: Dict[str, Any],
    base_cfg: Dict[str, Any]
) -> Dict[str, List[Tuple[str, float, float]]]:
    """
    base_res: fit_base 的结果（包含 d_um、fitter 等）
    base_cfg: 记录基线配置（theta1, theta2, n0, nu_window, fit_signal, loss, lock_frac）
    返回一个 dict：
      {
        "physics":   [(name, d_um, delta), ...],
        "window":    [(name, d_um, delta), ...],
        "preproc":   [(name, d_um, delta), ...],
        "algorithm": [(name, d_um, delta), ...],
      }
    delta = d_um - d_base
    """
    d_base = float(base_res["d_um"])

    theta1, theta2 = base_cfg["theta1"], base_cfg["theta2"]
    n0 = base_cfg["n0"]
    nu_window = base_cfg["nu_window"]
    fit_signal = base_cfg["fit_signal"]
    loss = base_cfg["loss"]
    lock_frac = base_cfg["lock_frac"]

    out = dict(physics=[], window=[], preproc=[], algorithm=[])

    # ---------- 1) 物理扰动包 ----------
    # 折射率：±0.02 / ±0.01；入射角：±0.1°
    for n0_p in [n0 - 0.02, n0 - 0.01, n0 + 0.01, n0 + 0.02]:
        r = fit_base(df1, df2, theta1, theta2, "constant", n0_p, 0.0, nu_window, fit_signal, loss, lock_frac)
        out["physics"].append((f"n0={n0_p:.2f}", float(r["d_um"]), float(r["d_um"]) - d_base))
    for dth in [-0.1, +0.1]:
        r = fit_base(df1, df2, theta1 + dth, theta2 + dth, "constant", n0, 0.0, nu_window, fit_signal, loss, lock_frac)
        out["physics"].append((f"θ±{dth:+.1f}°", float(r["d_um"]), float(r["d_um"]) - d_base))
    # 波数轴缩放：±0.05% 模拟标定误差（通过窗口等比例放缩近似，应谨慎）
    if nu_window is not None:
        a, b = nu_window
        for scale in [0.9995, 1.0005]:
            r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, (a*scale, b*scale), fit_signal, loss, lock_frac)
            out["physics"].append((f"wavenum×{scale:.5f}", float(r["d_um"]), float(r["d_um"]) - d_base))

    # ---------- 2) 窗口扰动包 ----------
    if nu_window is not None:
        a, b = nu_window
        for shift in [-200.0, +200.0]:
            r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, (a+shift, b+shift), fit_signal, loss, lock_frac)
            out["window"].append((f"window shift {shift:+.0f}", float(r["d_um"]), float(r["d_um"]) - d_base))
        for widen in [-200.0, +200.0]:
            r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, (a-widen, b+widen), fit_signal, loss, lock_frac)
            out["window"].append((f"window widen {widen:+.0f}", float(r["d_um"]), float(r["d_um"]) - d_base))

    # ---------- 3) 预处理扰动包 ----------
    # 这里通过 run_dual_angle_fit 的参数只影响拟合信号选择；SG 细节在 UpgradedTwoBeam 内固定，
    # 若要更细调，可在 UpgradedTwoBeam 暴露接口。先给常用组合：
    for sig in ["detrended", "grid", "proc"]:
        r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, nu_window, sig, loss, lock_frac)
        out["preproc"].append((f"signal={sig}", float(r["d_um"]), float(r["d_um"]) - d_base))

    # ---------- 4) 算法扰动包 ----------
    for lf in [0.10, 0.20]:
        r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, nu_window, fit_signal, loss, lock_frac=lf)
        out["algorithm"].append((f"lock_frac={lf:.2f}", float(r["d_um"]), float(r["d_um"]) - d_base))
    for loss_name in ["linear", "cauchy"]:
        r = fit_base(df1, df2, theta1, theta2, "constant", n0, 0.0, nu_window, fit_signal, loss_name, lock_frac)
        out["algorithm"].append((f"loss={loss_name}", float(r["d_um"]), float(r["d_um"]) - d_base))

    return out

# =========================================================
# 二、Tornado 图
# =========================================================
def plot_tornado(sens_dict: Dict[str, List[Tuple[str, float, float]]], title="Tornado 灵敏度（|Δd|排序）", savepath: Optional[str]=None):
    """
    sens_dict: run_sensitivity_packages 的返回
    """
    # 汇总为 (label, delta_um)
    rows = []
    for cat, items in sens_dict.items():
        for (name, d_val, delta) in items:
            rows.append((f"[{cat}] {name}", float(delta)))
    if not rows:
        raise ValueError("没有可绘制的灵敏度数据。")

    labels = [r[0] for r in rows]
    deltas = np.array([r[1] for r in rows], dtype=float)
    order = np.argsort(np.abs(deltas))[::-1]
    labels = [labels[i] for i in order]
    deltas = deltas[order]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35*len(labels))))
    ax.barh(y, deltas, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δd  (μm)")
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # 最大的在上方
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig

# =========================================================
# 三、Bootstrap（返回样本 + 画直方/小提琴）
# =========================================================
def bootstrap_samples(
    df1: pd.DataFrame, df2: pd.DataFrame,
    base_res: Dict[str, Any],
    keep_ratio: float = 0.7,
    n_boot: int = 400,
    random_state: int = 0
) -> List[float]:
    """
    生成 bootstrap 样本（d_um 列表），与 UpgradedTwoBeam.bootstrap_ci 思路一致，
    但返回全部样本以便画图。
    """
    rng = np.random.default_rng(random_state)
    fitter: JointTwoBeamFitter = base_res["fitter"]

    # 取与当前 fitter 一致的窗口/信号设置
    nu_min, nu_max = _extract_window_from_fitter(fitter)

    d_vals: List[float] = []
    for b in range(n_boot):
        dfs_sub = []
        # 对每个角独立采样连续窗口
        for df in [df1, df2]:
            col_nu = "波数 (cm-1)"; col_R = "反射率 (%)"
            if col_nu not in df.columns or col_R not in df.columns:
                col_nu, col_R = df.columns[:2]
            nu = df[col_nu].to_numpy(float)
            R = df[col_R].to_numpy(float)

            # 先按与 fitter 一致的窗口裁剪
            if nu_min is not None:
                m = (nu >= nu_min)
                nu, R = nu[m], R[m]
            if nu_max is not None:
                m = (nu <= nu_max)
                nu, R = nu[m], R[m]

            n = len(nu)
            mlen = max(8, int(round(n * keep_ratio)))
            start = int(rng.integers(low=0, high=max(n - mlen, 1)))
            idx = slice(start, start + mlen)

            df_sub = pd.DataFrame({col_nu: nu[idx], col_R: R[idx]})
            dfs_sub.append(df_sub)

        try:
            res_sub = fitter.fit(dfs_sub)  # 直接调用现有拟合
            d_vals.append(float(res_sub["d_um"]))
        except Exception:
            continue

    return d_vals

def plot_bootstrap_distribution(d_samples: List[float], d_base: float, savepath: Optional[str]=None):
    """
    画直方图 + 小提琴（单图两轴并排）
    """
    if len(d_samples) == 0:
        raise ValueError("bootstrap 样本为空。")
    arr = np.array(d_samples, dtype=float)
    mean, std = float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0
    q05, q95 = float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.hist(arr, bins=30)
    ax.axvline(d_base, linestyle='--')
    ax.set_title("Bootstrap 直方图")
    ax.set_xlabel("d (μm)")
    ax.set_ylabel("频数")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.violinplot([arr], showmeans=True, showextrema=True)
    ax2.set_title("Bootstrap 小提琴")
    ax2.set_xticks([1]); ax2.set_xticklabels(["d"])
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Bootstrap：mean={mean:.4f}, std={std:.4f}, 5%-95%=[{q05:.4f},{q95:.4f}] μm", y=1.02, fontsize=10)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig, dict(mean=mean, std=std, q05=q05, q95=q95)

# =========================================================
# 四、Profile J(d)：固定 d 网格，最小化其余参数
# =========================================================
def profile_cost_curve(
    df1: pd.DataFrame, df2: pd.DataFrame,
    base_res: Dict[str, Any],
    d_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (d_grid, J(d))；J(d) = 最小二乘残差平方和 / N（或RMSE^2）
    注意：这里只做“常数 n 模型”的 Profile，且沿用 base_res 的 fitter 配置。
    """
    fitter: JointTwoBeamFitter = base_res["fitter"]

    # 准备数据（与 fitter 一致的预处理 + 掩膜）
    nu_list, y_list, _ = fitter._prep_many([df1, df2])  # 复用内部工具

    # 组装“去掉 d”的参数初值与边界
    # 先跑一次得到完整参数作为初值
    res0 = fitter.fit([df1, df2])
    params0 = []

    # const-n: [d, n0] + per-angle [phi,a0,a1]...
    # 我们要把 d 固定掉，优化剩余部分
    n0 = float(res0["n0"])
    K = len(fitter.theta_list_rad)
    for k in range(K):
        pa = res0["per_angle"][f"angle_{k+1}"]
        params0.extend([pa["phi"], pa["A"], pa["B"]])
    p0 = np.array([n0] + params0, dtype=float)

    # 边界
    n0_lo, n0_hi = 2.2, 3.2
    lo = [n0_lo]
    hi = [n0_hi]
    for y in y_list:
        y_min, y_max = float(np.min(y)), float(np.max(y))
        span = max(y_max - y_min, 1e-3)
        lo += [-2.0*math.pi, y_min - 2.0*span, 0.0]
        hi += [ 2.0*math.pi, y_max + 2.0*span, 3.0*span]
    lo = np.array(lo, dtype=float); hi = np.array(hi, dtype=float)

    # 残差（对“除 d 之外的参数”）
    def resid_without_d(p, fixed_d_um):
        n0 = float(p[0])
        off = 1
        per_angle = []
        for k in range(K):
            phi, a0, a1 = p[off], p[off+1], p[off+2]
            per_angle.append((phi, a0, a1))
            off += 3

        nm = NModel(kind="constant", params=np.array([n0]))
        res_all = []
        for k in range(K):
            nu = nu_list[k]; y = y_list[k]
            n_nu = nm.n_of(nu)
            phi, a0, a1 = per_angle[k]
            y_hat = a0 + a1*np.cos(
                4.0*np.pi*n_nu*(fixed_d_um*1e-4)*cos_theta_t(n_nu, fitter.theta_list_rad[k]) * nu + phi
            )
            res_all.append(y_hat - y)
        return np.concatenate(res_all)

    from scipy.optimize import least_squares as lsq
    J = []
    for d_um in d_grid:
        r = lsq(resid_without_d, p0, args=(d_um,), bounds=(lo, hi), max_nfev=15000)
        resvec = resid_without_d(r.x, d_um)
        J.append(float(np.mean(resvec**2)))  # RMSE^2
    return d_grid, np.array(J, dtype=float)

def plot_profile(d_grid: np.ndarray, J: np.ndarray, d_base: float, savepath: Optional[str]=None):
    """画轮廓曲线 J(d)，标注最小点与基线 d"""
    jmin_idx = int(np.argmin(J))
    d_star = float(d_grid[jmin_idx]); Jmin = float(J[jmin_idx])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(d_grid, J)
    ax.axvline(d_base, linestyle='--', label=f"base d={d_base:.4f} μm")
    ax.axvline(d_star, linestyle=':', label=f"argmin d={d_star:.4f} μm")
    ax.set_xlabel("d (μm)")
    ax.set_ylabel("J(d) = RMSE^2")
    ax.set_title("Profile 轮廓曲线")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig, dict(d_star=d_star, Jmin=Jmin)

# =========================================================
# 五、示例入口（按需运行）
# =========================================================
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # --- 基线 ---
    base = fit_base(df1, df2, n_kind="constant", n0=2.60, nu_window=(1200, 3800))
    print("BASE:", base["summary"])
    base_cfg = dict(theta1=10.0, theta2=15.0, n0=2.60, nu_window=(1200, 3800),
                    fit_signal="detrended", loss="cauchy", lock_frac=0.15)

    # --- 四类扰动包 & Tornado ---
    sens = run_sensitivity_packages(df1, df2, base, base_cfg)
    fig_tornado = plot_tornado(sens, savepath=None)

    # --- Bootstrap ---
    samples = bootstrap_samples(df1, df2, base, keep_ratio=0.7, n_boot=400)
    fig_boot, stats_boot = plot_bootstrap_distribution(samples, d_base=float(base["d_um"]), savepath=None)
    print("Bootstrap stats:", stats_boot)

    # --- Profile ---
    d0 = float(base["d_um"])
    d_grid = np.linspace(max(0.7*d0, d0-2.0), d0+2.0, 60)
    D, J = profile_cost_curve(df1, df2, base, d_grid)
    fig_prof, info_prof = plot_profile(D, J, d_base=d0, savepath=None)
    print("Profile argmin:", info_prof)

    plt.show()

