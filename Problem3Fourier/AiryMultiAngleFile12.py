from typing import Tuple, List

import numpy as np

from Problem3Fourier.AiryMultiAngle import fit_single_angle, fit_multi_angle, plot_multiangle_fit
from Problem3Fourier.yuchuli_fft_viz import preprocess_and_plot_compare


def n_SiC_dispersion(wn_cm_inv):
    """
    硅的实折射率色散（无吸收项）：
      n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814 * λ^2
    其中 λ 单位为 μm；输入为 wn(cm^-1)。
    返回 complex（实数部分为 n，虚部=0）。
    """
    wn = np.asarray(wn_cm_inv, float)
    lam_um = 1e4 / wn        # cm^-1 -> μm
    l2 = lam_um**2
    n2 = 11.6858 + 0.939816/(l2 - 0.0086024) + 0.00089814*l2
    # 数值健壮性处理（极端波段防负数/NaN）：
    n2 = np.maximum(n2, 1e-12)
    return np.sqrt(n2) + 0j   # 若有吸收，可改成 n + 1j*k(λ)


if __name__ == "__main__":
    # 中文显示（按你工程）
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()

    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)


    # 1) 预处理（把这里替换为你的等间距波数 & 去基线/可用于拟合的反射信号）
    # 10°
    df = df1
    include_range: Tuple[float, float] = (1800, 2500)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
    )
    nu_10 = out["nu_uniform"]                 # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]        # 对应反射率/信号（已去均值或基线）

    # 15°
    df = df2
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
    )
    nu_15 = out["nu_uniform"]
    R15_meas = out["y_uniform_demean"]

    # 2) FFT 主峰得到的厚度初值（μm）——请替换为你的估计
    d0_um = 8.12

    # 3) 单角拟合（与多角保持相同的基线阶数！此处统一用二次）
    out10 = fit_single_angle(
        nu_10, R10_meas, d0_um,
        n1=n_SiC_dispersion, n2=2.60, theta_deg=10.0,
        poly_deg_baseline=1, verbose=True
    )
    out15 = fit_single_angle(
        nu_15, R15_meas, d0_um,
        n1=n_SiC_dispersion, n2=2.60, theta_deg=15.0,
        poly_deg_baseline=1, verbose=True
    )

    # 4) 多角联合拟合（与单角一致的二次基线；可选 sample_weighting="size" 按样本数加权）
    out_joint = fit_multi_angle(
        [(nu_10, R10_meas, 10.0), (nu_15, R15_meas, 15.0)],
        d0_um,
        n1=n_SiC_dispersion, n2=2.60,
        poly_deg_each_angle=1,
        force_positive_thickness=True,
        verbose=True,
        sample_weighting="mean",   # 或 "size"
    )

    # 5) 中文论文图（自动带上联合厚度）
    plot_multiangle_fit(out_joint, out10, out15, title_prefix="单层 Airy 多角度联合拟合")
