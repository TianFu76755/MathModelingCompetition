from pprint import pprint
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from UpgradedModel.Model import run_dual_angle_fit, NModel, JointTwoBeamFitter, cos_theta_t, PreprocessConfig, \
    model_reflectance_two_beam, robust_mad, phase_two_beam, JointFitConfig


# ======= 折射率敏感性扫描：扫 n0 或扫 B =======
def plot_sensitivity_scan(self, dfs: List[pd.DataFrame],
                          fit_result: Dict[str, Any],
                          sweep_which: str = "n0",
                          sweep_values: Optional[np.ndarray] = None,
                          fix_other_at_fit: bool = True,
                          savepath: Optional[str] = None):
    """
    sweep_which: 'n0' 或 'B'
    sweep_values: 要扫描的取值数组。None 时给一组默认值。
    fix_other_at_fit: True 时将“未扫描的那一项”固定为当前拟合结果（建议 True）
    """
    assert sweep_which in ("n0", "B"), "sweep_which 必须是 'n0' 或 'B'"

    # 默认扫描范围
    if sweep_values is None:
        if sweep_which == "n0":
            sweep_values = np.linspace(2.55, 2.65, 9)
        else:
            sweep_values = np.linspace(-3e-3, 3e-3, 13)
    sweep_values = np.asarray(sweep_values, dtype=float)

    # 备份当前设置
    old_fix_n0 = self.fit_cfg.fix_n0
    old_fix_B  = self.fit_cfg.fix_B
    old_range  = self.fit_cfg.fit_range_cm1

    d_list, rmse_list = [], []

    try:
        for val in sweep_values:
            # 固定被扫描的参数
            if sweep_which == "n0":
                self.fit_cfg.fix_n0 = float(val)
                # 另一项是否固定
                if self.n_model.kind == "cauchy":
                    self.fit_cfg.fix_B = float(fit_result.get("B", 0.0)) if fix_other_at_fit else None
                n0_init = float(val)
                B_init  = float(fit_result.get("B", 0.0))
            else:  # sweep B
                # 若当前模型不是 cauchy，则临时以 cauchy 方式解释（B=0）
                if self.n_model.kind != "cauchy":
                    # 允许扫 B 仅在 cauchy 下有意义
                    raise ValueError("当前 n_model 不是 'cauchy'，无法扫描 B。")
                self.fit_cfg.fix_B = float(val)
                self.fit_cfg.fix_n0 = float(fit_result["n0"]) if fix_other_at_fit else None
                n0_init = float(fit_result["n0"])
                B_init  = float(val)

            res_i = self.fit(dfs, n0_init=n0_init, B_init=B_init)

            d_list.append(float(res_i["d_um"]))
            rmse_list.append(float(res_i["rmse"]))
    finally:
        # 还原设置
        self.fit_cfg.fix_n0 = old_fix_n0
        self.fit_cfg.fix_B  = old_fix_B
        self.fit_cfg.fit_range_cm1 = old_range

    # 画图
    fig, ax1 = plt.subplots(1, 1, figsize=(8.2, 3.6))
    ax1.plot(sweep_values, d_list, marker="o", lw=1.2, label="d(μm)")
    ax1.set_xlabel("n0" if sweep_which=="n0" else "B")
    ax1.set_ylabel("厚度 d (μm)")
    ax1.grid(alpha=0.3)
    # 次轴 RMSE
    ax2 = ax1.twinx()
    ax2.plot(sweep_values, rmse_list, marker="s", lw=1.0, linestyle="--", label="RMSE")
    ax2.set_ylabel("RMSE")
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="best")
    title = "敏感性扫描：d 随 n0" if sweep_which=="n0" else "敏感性扫描：d 随 B"
    ax1.set_title(title)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig


# ======= 频段鲁棒性：滑动窗口 / 不同跨度 =======
def plot_band_robustness(self, dfs: List[pd.DataFrame],
                         fit_result: Dict[str, Any],
                         window_widths: Optional[List[float]] = None,
                         step_cm1: float = 150.0,
                         savepath: Optional[str] = None):
    """
    window_widths: 拟合窗口宽度列表（cm^-1），None 时默认 [1800, 2000, 2200]
    step_cm1: 滑动步长（cm^-1）
    """
    # 用全域的联合网格估计范围
    nu_list, _, aux = self._prep_many(dfs)
    lo_all = min([float(nu.min()) for nu in nu_list])
    hi_all = max([float(nu.max()) for nu in nu_list])

    if window_widths is None:
        window_widths = [1800.0, 2000.0, 2200.0]

    # 备份设置
    old_range = self.fit_cfg.fit_range_cm1
    old_fix_n0 = self.fit_cfg.fix_n0
    old_fix_B  = self.fit_cfg.fix_B

    results = []  # (width, centers[], d_list[], rmse_list[])
    try:
        for W in window_widths:
            centers, dvals, rvals = [], [], []
            start = lo_all + W/2
            stop  = hi_all - W/2
            if stop <= start:
                continue
            centers_grid = np.arange(start, stop+1e-6, step_cm1)
            for c in centers_grid:
                lo, hi = c - W/2, c + W/2
                self.fit_cfg.fit_range_cm1 = (lo, hi)
                # 固定折射率为当前结果，专注看 d 随窗口变化
                self.fit_cfg.fix_n0 = float(fit_result["n0"])
                if self.n_model.kind == "cauchy":
                    self.fit_cfg.fix_B = float(fit_result["B"])
                res_i = self.fit(dfs, n0_init=fit_result["n0"], B_init=fit_result.get("B", 0.0))
                centers.append(c); dvals.append(float(res_i["d_um"])); rvals.append(float(res_i["rmse"]))
            if centers:
                results.append((W, np.array(centers), np.array(dvals), np.array(rvals)))
    finally:
        self.fit_cfg.fit_range_cm1 = old_range
        self.fit_cfg.fix_n0 = old_fix_n0
        self.fit_cfg.fix_B  = old_fix_B

    # 绘图：上 d, 下 RMSE
    H = max(3.0, 2.5*len(results))
    fig, axes = plt.subplots(len(results), 1, figsize=(9, H), sharex=True)
    if len(results) == 1: axes = [axes]
    for ax, (W, C, D, R) in zip(axes, results):
        ax.plot(C, D, marker="o", lw=1.0, label=f"d(μm), 窗口宽度 {W:.0f} cm$^{{-1}}$")
        ax2 = ax.twinx()
        ax2.plot(C, R, marker="s", lw=1.0, linestyle="--", label="RMSE")
        ax.set_ylabel("d (μm)")
        ax2.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(lines+l2, labels+lb2, loc="best")
    axes[-1].set_xlabel("拟合窗口中心 (cm$^{-1}$)")
    fig.suptitle("频段鲁棒性：滑动窗口对 d 的影响", y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=160)
    return fig


# ======= 留一段验证：训练/验证区间分离 =======
def plot_holdout_validation(self, dfs: List[pd.DataFrame],
                            train_range: Optional[Tuple[float, float]] = None,
                            val_range:   Optional[Tuple[float, float]] = None,
                            savepath: Optional[str] = None):
    """
    train_range / val_range: (lo,hi)。若任一为 None，则自动按全域中点二分。
    输出：上行每角验证段实测/预测叠加，下行显示验证 RMSE 栏注。
    """
    # 先得到全域范围
    nu_list_full, y_list_full, aux_full = self._prep_many(dfs)
    lo_all = min([float(nu.min()) for nu in nu_list_full])
    hi_all = max([float(nu.max()) for nu in nu_list_full])
    mid = 0.5*(lo_all + hi_all)

    if train_range is None or val_range is None:
        train_range = (lo_all, mid)
        val_range   = (mid, hi_all)

    # 备份
    old_range = self.fit_cfg.fit_range_cm1
    old_fix_n0 = self.fit_cfg.fix_n0
    old_fix_B  = self.fit_cfg.fix_B

    try:
        # 仅用训练段拟合（固定 n 参数可选——这里不固定，完全复现流程）
        self.fit_cfg.fit_range_cm1 = train_range
        res_train = self.fit(dfs)
        # 用得到的 res_train 去验证段上做预测残差
        # 准备验证段数据（只用于画图/算RMSE，不改变 fit 设置）
        K = len(self.theta_list_rad)
        fig, axes = plt.subplots(K, 1, figsize=(9, 2.8*K), sharex=True)
        if K == 1: axes = [axes]

        # 计算验证 RMSE
        resid_all = []
        for k in range(K):
            nu = nu_list_full[k]
            y  = y_list_full[k]
            m_val = (nu >= val_range[0]) & (nu <= val_range[1])
            nu_v, y_v = nu[m_val], y[m_val]
            # 预测
            yhat_v, resid_v, _, _ = self._predict_one_angle(nu_v, y_v, k, res_train)
            resid_all.append(resid_v)
            ax = axes[k]
            ax.plot(nu_v, y_v, lw=1.0, label="验证段：实测")
            ax.plot(nu_v, yhat_v, lw=1.0, linestyle="--", label="验证段：预测")
            ax.set_ylabel("信号幅值"); ax.grid(alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("波数 (cm$^{-1}$)")

        # 验证 RMSE 汇总
        resid_all = np.concatenate(resid_all) if resid_all else np.array([])
        rmse_val = float(np.sqrt(np.mean(resid_all**2))) if resid_all.size else float("nan")
        fig.suptitle(f"留一段验证：训练{train_range} → 验证{val_range}；验证RMSE={rmse_val:.4g}", y=0.98)
        fig.tight_layout()
        if savepath: fig.savefig(savepath, dpi=160)
    finally:
        self.fit_cfg.fit_range_cm1 = old_range
        self.fit_cfg.fix_n0 = old_fix_n0
        self.fit_cfg.fix_B  = old_fix_B
    return fig


# ======= 模型对比：constant-n vs cauchy-n =======
def plot_model_comparison(self, dfs: List[pd.DataFrame],
                          n0_init: float = 2.60, B_init: float = 0.0,
                          savepath: Optional[str] = None):
    """
    基于与当前 fitter 一致的预处理/拟合设置，对比 constant-n 与 cauchy-n。
    输出：条形图对比 RMSE 与 AIC (文本注记显示 AIC/BIC)。
    """
    # 统一预处理
    nu_list, y_list, _ = self._prep_many(dfs)
    N = int(sum([len(nu) for nu in nu_list]))  # 总样本数
    Kangles = len(self.theta_list_rad)

    # 构造两个 fitter（复用同一 pre_cfg / fit_cfg 复制件）
    pre_cfg_copy = self.pre.cfg
    fit_cfg_copy = JointFitConfig(**vars(self.fit_cfg))

    # constant-n
    fitter_const = JointTwoBeamFitter(NModel(kind="constant", params=np.array([n0_init])),
                                      [np.degrees(t) for t in self.theta_list_rad],
                                      pre_cfg=pre_cfg_copy, fit_cfg=fit_cfg_copy)
    res_c = fitter_const.fit(dfs, n0_init=n0_init)
    # cauchy-n
    fitter_cau = JointTwoBeamFitter(NModel(kind="cauchy", params=np.array([n0_init, B_init])),
                                    [np.degrees(t) for t in self.theta_list_rad],
                                    pre_cfg=pre_cfg_copy, fit_cfg=fit_cfg_copy)
    res_ca = fitter_cau.fit(dfs, n0_init=n0_init, B_init=B_init)

    # 计算 RSS 以便 AIC/BIC：RSS = sum(resid^2)
    def rss_from(fitter, res, nu_list, y_list):
        resid = fitter._residual_joint(
            np.concatenate(([res["d_um"], res["n0"]] + (([res["B"]] if "B" in res else [])) +
                            [v for k in range(Kangles)
                             for v in (res["per_angle"][f"angle_{k+1}"]["phi"],
                                       res["per_angle"][f"angle_{k+1}"]["A"],
                                       res["per_angle"][f"angle_{k+1}"]["B"])])),
            nu_list, y_list
        )
        return float(np.sum(resid**2)), resid.size

    RSS_c, _ = rss_from(fitter_const, res_c, nu_list, y_list)
    RSS_ca,_ = rss_from(fitter_cau,   res_ca, nu_list, y_list)

    # 参数个数：d + n0 + (B?) + (phi,A,B)*K
    k_const = 1 + 1 + 3*Kangles
    k_cau   = 1 + 2 + 3*Kangles

    def aic_bic(RSS, k, N):
        # 高斯噪声的最小二乘近似：AIC = N*ln(RSS/N) + 2k
        # BIC = N*ln(RSS/N) + k*ln(N)
        RSS = max(RSS, 1e-30)
        aic = N*np.log(RSS/N) + 2*k
        bic = N*np.log(RSS/N) + k*np.log(N)
        return aic, bic

    AIC_c, BIC_c   = aic_bic(RSS_c,  k_const, N)
    AIC_ca, BIC_ca = aic_bic(RSS_ca, k_cau,   N)

    # 绘制对比条形图（RMSE）
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.4))
    labs = ["constant-n", "cauchy-n"]
    rmse_vals = [res_c["rmse"], res_ca["rmse"]]
    ax.bar(labs, rmse_vals, width=0.55)
    for i, v in enumerate(rmse_vals):
        ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("联合 RMSE")
    ax.set_title("模型对比：constant-n vs cauchy-n")
    ax.grid(axis="y", alpha=0.3)

    # 注记 AIC/BIC
    txt = (f"constant-n:  AIC={AIC_c:.1f}, BIC={BIC_c:.1f}\n"
           f"cauchy-n:    AIC={AIC_ca:.1f}, BIC={BIC_ca:.1f}")
    ax.annotate(txt, xy=(0.5, 0.02), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888"))
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

    # 假设已有：
    # fitter: JointTwoBeamFitter = res["fitter"]

    # 1) 敏感性扫描：扫 n0
    fitter.plot_sensitivity_scan([df1, df2], res, sweep_which="n0",
                                 sweep_values=np.linspace(2.56, 2.64, 9),
                                 savepath="fig_sensitivity_n0.png")

    #   扫 B（cauchy 模型时）
    fitter.plot_sensitivity_scan([df1, df2], res, sweep_which="B",
                                 sweep_values=np.linspace(-3e-3, 3e-3, 13),
                                 savepath="fig_sensitivity_B.png")

    # 2) 频段鲁棒性
    fitter.plot_band_robustness([df1, df2], res,
                                window_widths=[1800, 2000, 2200],
                                step_cm1=150,
                                savepath="fig_band_robustness.png")

    # 3) 留一段验证（自动二分）
    fitter.plot_holdout_validation([df1, df2], savepath="fig_holdout.png")
    #   或自定义训练/验证区间：
    # fitter.plot_holdout_validation([df1, df2], train_range=(1200, 2600), val_range=(2600, 3800),
    #                                savepath="fig_holdout_custom.png")

    # 4) 模型对比（constant-n vs cauchy-n）
    fitter.plot_model_comparison([df1, df2], n0_init=res["n0"], B_init=res.get("B", 0.0),
                                 savepath="fig_model_compare.png")
