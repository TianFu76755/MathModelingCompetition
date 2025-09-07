from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from Problem3Fourier.AiryMultiAngle import fit_multi_angle, n_Si_dispersion_with_k
from Problem3Fourier.AiryMultiAngleFile12 import n_SiC_dispersion
from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


def add_normal_noise(R_meas, noise_level):
    """
    使用正态分布（高斯噪声）对数据进行加扰动
    """
    noise = np.random.normal(0, noise_level, size=R_meas.shape)
    return R_meas + noise


def add_uniform_noise(R_meas, noise_level):
    """
    使用均匀分布对数据进行加扰动
    """
    noise = np.random.uniform(-noise_level, noise_level, size=R_meas.shape)
    return R_meas + noise


def monte_carlo_simulation(nu_10, R10_meas, nu_15, R15_meas, d0_um, n_SiC_dispersion, noise_levels, distributions,
                           num_simulations=10):
    results = {dist: {level: [] for level in noise_levels} for dist in distributions}

    for dist in distributions:
        for noise_level in noise_levels:
            for _ in range(num_simulations):
                # 根据噪声类型添加扰动
                if dist == 'normal':
                    R10_noisy = add_normal_noise(R10_meas, noise_level)
                    R15_noisy = add_normal_noise(R15_meas, noise_level)
                elif dist == 'uniform':
                    R10_noisy = add_uniform_noise(R10_meas, noise_level)
                    R15_noisy = add_uniform_noise(R15_meas, noise_level)

                # 进行联合拟合
                out_joint_noisy = fit_multi_angle(
                    [(nu_10, R10_noisy, 10.0), (nu_15, R15_noisy, 15.0)],
                    d0_um,
                    n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55,
                    poly_deg_each_angle=1,
                    force_positive_thickness=True,
                    verbose=True,
                    sample_weighting="mean",  # 或 "size"
                )

                # 记录扰动后的结果
                results[dist][noise_level].append(out_joint_noisy["d_um"])

    return results


def analyze_results(results):
    """分析结果，计算均值、标准差和最大变化率"""
    analysis = {}

    for dist in results:
        analysis[dist] = {}
        for noise_level in results[dist]:
            data = np.array(results[dist][noise_level])

            # 计算均值、标准差和最大变化率
            mean_value = np.mean(data)
            std_value = np.std(data)
            max_change_rate = np.max(np.abs(data - mean_value) / mean_value)

            analysis[dist][noise_level] = {
                "mean": mean_value,
                "std": std_value,
                "max_change_rate": max_change_rate
            }

    return analysis


def plot_boxplots(results, noise_levels, distributions):
    """绘制箱形图"""
    # 创建 2 行 3 列的网格
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 使用 flatten() 来简化对 axes 的索引
    axes = axes.flatten()  # 将 2x3 的网格转换为一个 1 维数组

    index = 0  # 初始化索引

    for i, dist in enumerate(distributions):
        for j, noise_level in enumerate(noise_levels):
            ax = axes[index]
            data = np.array(results[dist][noise_level])

            ax.boxplot(data)
            ax.set_title(f'{dist.capitalize()} 分布 - 噪声级别 {noise_level}', fontsize=16)
            ax.set_xlabel('厚度 (um)', fontsize=12)
            ax.set_ylabel('频率', fontsize=12)

            index += 1  # 更新索引

    # 设置中文标题
    # plt.suptitle('敏感性分析结果箱形图', fontsize=16)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.92)  # 调整标题位置
    plt.show()



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
    df = df3
    include_range: Tuple[float, float] = (2032, 2721)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = []  # 强吸收段（可多段）
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
        is_plot=True,
    )
    nu_10 = out["nu_uniform"]                 # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]        # 对应反射率/信号（已去均值或基线）

    # 15°
    df = df4
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
        is_plot=True,
    )
    nu_15 = out["nu_uniform"]
    R15_meas = out["y_uniform_demean"]

    # 2) FFT 主峰得到的厚度初值（μm）——请替换为你的估计
    d0_um = 3.39

    # 执行蒙特卡洛模拟，噪声级别为[0.01, 0.05, 0.1]，扰动类型为['normal', 'uniform']
    noise_levels = [0.01, 0.05, 0.1]
    distributions = ['normal', 'uniform']
    results = monte_carlo_simulation(
        nu_10, R10_meas, nu_15, R15_meas, d0_um, n_SiC_dispersion, noise_levels, distributions,
        num_simulations=100
    )

    # 分析结果
    analysis = analyze_results(results)
    # 输出分析结果
    for dist in distributions:
        for noise_level in noise_levels:
            print(f"扰动类型: {dist}, 噪声级别: {noise_level}")
            print(f"均值: {analysis[dist][noise_level]['mean']}")
            print(f"标准差: {analysis[dist][noise_level]['std']}")
            print(f"最大变化率: {analysis[dist][noise_level]['max_change_rate']}")
            print("-" * 40)

    # 绘制箱形图
    plot_boxplots(results, noise_levels, distributions)
