from pprint import pprint
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from UpgradedModel.Model import run_dual_angle_fit, NModel, JointTwoBeamFitter, cos_theta_t, PreprocessConfig, \
    model_reflectance_two_beam, robust_mad, phase_two_beam


# ======= 工具：统一根据当前结果计算每角 y_hat / 残差 =======
def _predict_one_angle(self, nu: np.ndarray, y: np.ndarray, k: int, fit_result: Dict[str, Any]):
    """返回 yhat, resid, (phi,A,B), n(ν) """
    kind = self.n_model.kind
    if kind == "cauchy":
        n_params = np.array([fit_result["n0"], fit_result["B"]], dtype=float)
    else:
        n_params = np.array([fit_result["n0"]], dtype=float)
    pa = fit_result["per_angle"][f"angle_{k+1}"]
    phi, A, B = float(pa["phi"]), float(pa["A"]), float(pa["B"])
    n_nu = NModel(kind=kind, params=n_params).n_of(nu)
    yhat = model_reflectance_two_beam(nu, n_nu, fit_result["d_um"], self.theta_list_rad[k], A, B, phi)
    resid = yhat - y
    return yhat, resid, (phi, A, B), n_nu

# ======= 残差随波数（两角分面） =======
def plot_residuals(self, dfs: List[pd.DataFrame], fit_result: Dict[str, Any], savepath: Optional[str] = None):
    nu_list, y_list, _ = self._prep_many(dfs)
    K = len(self.theta_list_rad)
    fig, axes = plt.subplots(K, 1, figsize=(9, 2.8*K), sharex=True)
    if K == 1: axes = [axes]
    for k in range(K):
        nu, y = nu_list[k], y_list[k]
        yhat, resid, _, _ = self._predict_one_angle(nu, y, k, fit_result)
        sigma = float(robust_mad(resid))
        ax = axes[k]
        ax.plot(nu, resid, lw=1.0, label="残差")
        ax.axhline(0, color="k", lw=0.8, linestyle="--")
        ax.fill_between(nu, -sigma, sigma, alpha=0.12, label=f"±1σ (MAD)={sigma:.3g}")
        ax.set_ylabel("残差")
        if self.fit_cfg.fit_range_cm1 is not None:
            lo, hi = self.fit_cfg.fit_range_cm1
            ax.axvspan(lo, hi, alpha=0.08)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("波数 (cm$^{-1}$)")
    fig.suptitle("两束模型：残差 vs 波数", y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig

# ======= 残差统计（直方图 + QQ 图） =======
def plot_residual_stats(self, dfs: List[pd.DataFrame], fit_result: Dict[str, Any], savepath: Optional[str] = None):
    from scipy import stats
    nu_list, y_list, _ = self._prep_many(dfs)
    resid_all = []
    for k in range(len(self.theta_list_rad)):
        _, resid, _, _ = self._predict_one_angle(nu_list[k], y_list[k], k, fit_result)
        resid_all.append(resid)
    r = np.concatenate(resid_all)
    fig = plt.figure(figsize=(9, 3.2))
    # 直方图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(r, bins=40, density=True, alpha=0.8)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    ax1.set_title(f"残差直方图 (μ={mu:.3g}, σ={sd:.3g})")
    ax1.set_xlabel("残差"); ax1.set_ylabel("密度")
    ax1.grid(alpha=0.3)
    # QQ 图
    ax2 = fig.add_subplot(1, 2, 2)
    stats.probplot(r, dist="norm", plot=ax2)
    ax2.set_title("残差 QQ 图 (Normal)")
    ax2.grid(alpha=0.3)
    fig.suptitle("残差统计性检验", y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig

# ======= 跨角一致性（单角拟合 d 的对比） =======
def plot_cross_angle_consistency(self, dfs: List[pd.DataFrame], fit_result: Dict[str, Any], savepath: Optional[str] = None):
    # 如果外部没有 diagnostics，就内部跑一次
    if "diagnostics" not in fit_result:
        _ = self._single_angle_diagnostics(*self._prep_many(dfs)[:2], fit_result)
    diag = fit_result["diagnostics"]
    d_list = diag["single_angle_d_um"]
    rel_spread = diag["cross_angle_rel_spread"]
    K = len(d_list)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
    xs = np.arange(1, K+1)
    ax.bar(xs, d_list, width=0.55)
    for i, d in enumerate(d_list, 1):
        ax.text(i, d, f"{d:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(np.degrees(t))}°" for t in self.theta_list_rad])
    ax.set_ylabel("厚度 d (μm)")
    ax.set_title(f"跨角一致性：相对散布 = {rel_spread*100:.2f}%")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig

# ======= 相位线性检验 =======
def plot_phase_linearity(self, dfs: List[pd.DataFrame], fit_result: Dict[str, Any], savepath: Optional[str] = None):
    """用拟合得到的 d、n(ν)、θ_i 计算 Φ(ν)，对 ν 做线性回归并画相位残差。"""
    from numpy.linalg import lstsq
    nu_list, y_list, _ = self._prep_many(dfs)
    kind = self.n_model.kind
    if kind == "cauchy":
        n_params = np.array([fit_result["n0"], fit_result["B"]], dtype=float)
    else:
        n_params = np.array([fit_result["n0"]], dtype=float)

    K = len(self.theta_list_rad)
    fig, axes = plt.subplots(K, 2, figsize=(11, 3.2*K), sharex=False)
    if K == 1: axes = np.array([axes])

    for k in range(K):
        nu, y = nu_list[k], y_list[k]
        pa = fit_result["per_angle"][f"angle_{k+1}"]
        phi, A, B = float(pa["phi"]), float(pa["A"]), float(pa["B"])
        n_nu = NModel(kind=kind, params=n_params).n_of(nu)
        phi_theory = phase_two_beam(nu, n_nu, fit_result["d_um"], self.theta_list_rad[k], phi)

        # 对 φ(ν) 做线性回归：phi ≈ a*ν + b
        X = np.vstack([nu, np.ones_like(nu)]).T
        a, b = lstsq(X, phi_theory, rcond=None)[0]
        phi_fit = a * nu + b
        phi_resid = phi_theory - phi_fit

        ax1, ax2 = axes[k, 0], axes[k, 1]
        ax1.plot(nu, phi_theory, lw=1.0, label="相位 Φ(ν)")
        ax1.plot(nu, phi_fit, lw=1.0, linestyle="--", label=f"线性回归 (斜率={a:.3g})")
        ax1.set_ylabel("相位 Φ(ν)")
        ax1.grid(alpha=0.3); ax1.legend(loc="upper left")
        if self.fit_cfg.fit_range_cm1 is not None:
            lo, hi = self.fit_cfg.fit_range_cm1
            ax1.axvspan(lo, hi, alpha=0.08)

        ax2.plot(nu, phi_resid, lw=1.0, label="相位残差")
        ax2.axhline(0, color="k", lw=0.8, linestyle="--")
        ax2.set_ylabel("相位残差")
        ax2.grid(alpha=0.3); ax2.legend(loc="upper right")
        axes[-1, 0].set_xlabel("波数 (cm$^{-1}$)")
        axes[-1, 1].set_xlabel("波数 (cm$^{-1}$)")

    fig.suptitle("相位线性检验", y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig

# ======= Bootstrap：d 的分布直方图 =======
def plot_bootstrap_hist(self, dfs: List[pd.DataFrame], n_boot: int = 200, keep_ratio: float = 0.7,
                        random_state: int = 0, savepath: Optional[str] = None):
    """内部做一次 bootstrap，画 d 的分布（直方图 + 竖线: 5/50/95%）"""
    # 复用 bootstrap 采样逻辑，但直接在此记录样本
    rng = np.random.default_rng(random_state)
    nu_list, y_list, aux = self._prep_many(dfs)
    prepped = aux["prepped"]

    d_vals = []
    for b in range(n_boot):
        dfs_sub = []
        for (nu, prep) in zip(nu_list, prepped):
            n = len(nu)
            m = max(8, int(round(n * keep_ratio)))
            start = rng.integers(low=0, high=max(n - m, 1))
            idx = slice(start, start + m)
            df_sub = pd.DataFrame({"波数 (cm-1)": prep["nu_grid"][idx], "反射率 (%)": (prep["R_grid"][idx] * 100.0)})
            dfs_sub.append(df_sub)
        try:
            res_sub = self.fit(dfs_sub)
            d_vals.append(float(res_sub["d_um"]))
        except Exception:
            continue

    if len(d_vals) == 0:
        raise RuntimeError("bootstrap 无有效样本，无法绘图。")

    d_arr = np.array(d_vals, dtype=float)
    q05, q50, q95 = np.quantile(d_arr, [0.05, 0.50, 0.95])
    mu, sd = float(np.mean(d_arr)), float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.6))
    ax.hist(d_arr, bins=30, alpha=0.85, density=True)
    for q, lab in zip([q05, q50, q95], ["5%", "50%", "95%"]):
        ax.axvline(q, lw=1.2, linestyle="--", label=f"{lab}={q:.4f}")
    ax.set_xlabel("厚度 d (μm)"); ax.set_ylabel("概率密度")
    ax.set_title(f"Bootstrap d 分布 (n={len(d_arr)}, μ={mu:.4f}, σ={sd:.4f})")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig

if __name__=="__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # —— 一键双角拟合（推荐） ——
    res = run_dual_angle_fit(
        df1, df2,
        theta1_deg=10.0, theta2_deg=15.0,
        n_kind="cauchy",
        n0_init=2.60, B_init=0.0,
        pre_cfg=PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2, normalize=True),
        fit_signal="detrended",
        loss="cauchy",
    )

    # 先把 n 固定住，确保 d 与 FFT 周期一致；需要时再放开
    fitter: JointTwoBeamFitter = res["fitter"]
    fitter.fit_cfg.fix_n0 = 2.60
    fitter.fit_cfg.fix_B = 0.0
    res = fitter.fit([df1, df2], n0_init=2.60, B_init=0.0)

    print("=== 结果摘要 ===")
    pprint(res)

    # 作图（可选）
    fitter.plot_fft([df1, df2], savepath=None)
    fitter.plot_spectrum_and_fit([df1, df2], res, savepath=None)
    plt.show()

    # 1) 残差 vs 波数
    fitter.plot_residuals([df1, df2], res, savepath="fig_residuals.png")

    # 2) 残差统计（直方图 + QQ）
    fitter.plot_residual_stats([df1, df2], res, savepath="fig_residual_stats.png")

    # 3) 跨角一致性（单角 d）
    fitter.plot_cross_angle_consistency([df1, df2], res, savepath="fig_cross_angle_consistency.png")

    # 4) 相位线性检验
    fitter.plot_phase_linearity([df1, df2], res, savepath="fig_phase_linearity.png")

    # 5) Bootstrap d 分布（内部会独立跑一轮bootstrap）
    fitter.plot_bootstrap_hist([df1, df2], n_boot=300, keep_ratio=0.72, random_state=42,
                               savepath="fig_bootstrap_d_hist.png")
