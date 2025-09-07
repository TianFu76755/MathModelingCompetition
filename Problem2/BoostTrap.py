# 5. 执行Bootstrap重抽样
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from Problem2.ModelReconstruct import perform_dual_angle_fit
from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


def bootstrap_resampling(nu1: np.ndarray, y1: np.ndarray, nu2: np.ndarray, y2: np.ndarray, theta1_deg: float, theta2_deg: float, n_bootstrap: int = 200) -> Dict[str, Any]:
    """进行Bootstrap重抽样并返回拟合结果的统计数据"""
    d_vals = []
    n0_vals = []
    for _ in range(n_bootstrap):
        # 重抽样（带放回采样）
        indices1 = np.random.choice(len(nu1), len(nu1), replace=True)
        indices2 = np.random.choice(len(nu2), len(nu2), replace=True)

        # 从原数据中提取重抽样数据
        nu1_resampled, y1_resampled = nu1[indices1], y1[indices1]
        nu2_resampled, y2_resampled = nu2[indices2], y2[indices2]

        # 执行双角度联合拟合
        fit_result = perform_dual_angle_fit(nu1_resampled, y1_resampled, nu2_resampled, y2_resampled, theta1_deg, theta2_deg)

        # 记录拟合结果
        d_vals.append(fit_result["params"][0])  # 厚度 d
        n0_vals.append(fit_result["params"][1])  # 折射率 n0

    # 计算结果的均值、标准差、分位数
    d_vals = np.array(d_vals)
    n0_vals = np.array(n0_vals)

    print(f"d_means: {np.mean(d_vals):.4f} μm, d_std: {np.std(d_vals):.4f} μm")
    print(f"d_5%: {np.percentile(d_vals, 5):.4f} μm, d_95%: {np.percentile(d_vals, 95):.4f} μm")
    # 返回包含参数均值、标准差及分位数的结果
    return {
        "d_vals": d_vals,  # 返回厚度的所有重抽样结果
        "n0_vals": n0_vals,  # 返回折射率的所有重抽样结果
        "d_mean": np.mean(d_vals),
        "d_std": np.std(d_vals),
        "d_q05": np.percentile(d_vals, 5),
        "d_q95": np.percentile(d_vals, 95),
        "n0_mean": np.mean(n0_vals),
        "n0_std": np.std(n0_vals),
        "n0_q05": np.percentile(n0_vals, 5),
        "n0_q95": np.percentile(n0_vals, 95),
    }


def clean_outliers(data: np.ndarray, rule="iqr", center="median", k=1.5, hard_cap=None):
    """
    对数据进行清洗，去除异常值（基于 IQR 或者标准差等方法）
    :param data: 需要清洗的数组
    :param rule: 清洗规则 ("iqr" 使用四分位距, "std" 使用标准差)
    :param center: 中心值（"mean" 或 "median"）
    :param k: IQR 或标准差的倍数，用于判断异常值
    :param hard_cap: 极限阈值，用于直接裁剪数据
    :return: 清洗后的数据
    """
    # 计算中心值和偏差
    if center == "median":
        center_val = np.median(data)
        deviation = np.median(np.abs(data - center_val))
    elif center == "mean":
        center_val = np.mean(data)
        deviation = np.std(data)

    # 基于规则进行清洗
    if rule == "iqr":
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - k * iqr
        upper_bound = q75 + k * iqr
        clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
        bounds = (lower_bound, upper_bound)
    elif rule == "std":
        clean_data = data[(data >= center_val - k * deviation) & (data <= center_val + k * deviation)]
        bounds = (center_val - k * deviation, center_val + k * deviation)
    else:
        raise ValueError("Unknown cleaning rule. Use 'iqr' or 'std'.")

    if hard_cap:
        clean_data = np.clip(clean_data, hard_cap[0], hard_cap[1])
        bounds = hard_cap

    return {"clean": clean_data, "bounds": bounds, "rule_used": rule}


def plot_bootstrap_results(bootstrap_results: Dict[str, Any]):
    """绘制Bootstrap结果的分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 清洗数据
    cleaned = clean_outliers(bootstrap_results["d_vals"], rule="iqr", center="median", k=1.5)
    xc = cleaned["clean"]
    lo, hi = cleaned["bounds"]

    # Zoom-in 直方图（只看主体）
    zmin, zmax = lo, hi
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
        zmin, zmax = np.min(bootstrap_results["d_vals"]), np.max(bootstrap_results["d_vals"])

    ax = axes[0]
    # 只对 zoom 范围内的数据画直方图
    x_zoom = bootstrap_results["d_vals"][(bootstrap_results["d_vals"] >= zmin) & (bootstrap_results["d_vals"] <= zmax)]
    ax.hist(x_zoom, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(bootstrap_results["d_mean"], linestyle='--', label=f"base={bootstrap_results['d_mean']:.3f}")
    ax.axvline(bootstrap_results["d_q05"], linestyle='-.', lw=1.0, label=f"q05={bootstrap_results['d_q05']:.3f}")
    ax.axvline(bootstrap_results["d_q95"], linestyle='-.', lw=1.0, label=f"q95={bootstrap_results['d_q95']:.3f}")
    ax.set_xlim(zmin, zmax)
    ax.set_xlabel("Thickness (μm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution (Zoom-in)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 绘制清洗后数据的小提琴图（去除异常值的圆圈）
    ax2 = axes[1]
    ax2.violinplot([xc], showmeans=True, showextrema=True, showmedians=True)
    ax2.boxplot([xc], widths=0.2, positions=[1.0])
    ax2.set_xticks([1]); ax2.set_xticklabels(["Cleaned d"])
    ax2.set_title("Bootstrap Distribution (Violin + Boxplot)")
    ax2.grid(alpha=0.3)

    supt = (f"Raw: mean={bootstrap_results['d_mean']:.4f}, std={bootstrap_results['d_std']:.4f}, "
            f"[5%,95%]=[{bootstrap_results['d_q05']:.4f},{bootstrap_results['d_q95']:.4f}] μm | "
            f"Cleaned (IQR method): mean={np.mean(xc):.4f}, std={np.std(xc, ddof=1):.4f}, "
            f"[5%,95%]=[{np.percentile(xc, 5):.4f},{np.percentile(xc, 95):.4f}] μm")
    fig.suptitle(supt, y=1.02, fontsize=10)
    fig.tight_layout()
    plt.show()

# 简化后的主程序
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # ==============两个单角度拟合==============
    # 预处理数据
    result1 = preprocess_and_plot_compare(df1, include_range=(2060, 2280), is_plot=True)  # TODO
    # 获取处理后的数据
    nu1_uniform = result1["nu_uniform"]
    y1_uniform_demean = result1["y_uniform_demean"]

    # 预处理数据
    result2 = preprocess_and_plot_compare(df2, include_range=(2060, 2280), is_plot=True)  # TODO
    # 获取处理后的数据
    nu2_uniform = result2["nu_uniform"]
    y2_uniform_demean = result2["y_uniform_demean"]

    # 执行Bootstrap重抽样
    bootstrap_results = bootstrap_resampling(nu1_uniform, y1_uniform_demean, nu2_uniform, y2_uniform_demean, 10, 15,
                                             n_bootstrap=200)
    # 绘制Bootstrap结果
    plot_bootstrap_results(bootstrap_results)
