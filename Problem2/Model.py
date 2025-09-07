from typing import Tuple, List

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


# 给定的折射率模型
def refractive_index(nu, n0=2.5610, B=3.4e4):
    """
    计算波数nu对应的折射率n(nu)
    n0: 折射率模型的常数
    B: 折射率模型的常数
    nu: 波数（cm^-1）
    """
    return n0 + B / (nu ** 2)


# 计算理论反射率模型
def model_reflectance(nu, d, n0, B, theta_deg):
    """
    计算理论反射率模型R_model
    nu: 波数（cm^-1）
    d: 外延层厚度（cm）
    n0, B: 折射率模型的参数
    theta_deg: 入射角（度）
    """
    # 计算折射率
    n = refractive_index(nu, n0, B)

    # 将入射角转换为弧度
    theta = np.radians(theta_deg)

    # 根据干涉公式计算路径差
    path_diff = 2 * n * d * np.cos(theta)

    # 使用干涉公式计算理论反射率（假设 m = 1）
    R_model = np.abs(np.cos(np.pi * path_diff / nu)) ** 2
    return R_model


# 残差函数：计算实测反射率和理论反射率之间的差异
def residuals(params, nu, R_meas, theta_deg):
    """
    残差函数，用于最小化实测反射率和理论反射率之间的差异
    params: 优化参数 [d, n0, B]
    nu: 波数（cm^-1）
    R_meas: 实测反射率
    theta_deg: 入射角（度）
    """
    d, n0, B = params
    R_model = model_reflectance(nu, d, n0, B, theta_deg)
    return R_meas - R_model


# 使用最小二乘法进行拟合
def optimize_thickness(nu, R_meas, theta_deg, d0, n0_init=2.5610, B_init=3.4e4):
    """
    使用最小二乘法优化外延层厚度
    nu: 波数（cm^-1）
    R_meas: 实测反射率
    theta_deg: 入射角（度）
    d0: FFT计算得到的初始厚度（cm）
    n0_init, B_init: 折射率模型的初始值
    """
    # 初始参数 [d, n0, B]
    initial_params = [d0, n0_init, B_init]

    # 使用最小二乘法拟合
    result = least_squares(residuals, initial_params, args=(nu, R_meas, theta_deg))

    # 优化后的参数
    d_opt, n0_opt, B_opt = result.x
    return d_opt, n0_opt, B_opt


# 主程序：输入实测数据并计算外延层厚度
def main(nu, R_meas, theta_deg, d0):
    """
    主程序，输入波数、实测反射率、入射角和初始厚度，输出优化后的外延层厚度
    """
    # 调用优化函数
    d_opt, n0_opt, B_opt = optimize_thickness(nu, R_meas, theta_deg, d0)

    # 输出优化后的结果
    print(f"Optimized Thickness: {d_opt * 1e6:.4f} µm")
    print(f"Optimized n0: {n0_opt:.4f}")
    print(f"Optimized B: {B_opt:.4f}")

    return d_opt, n0_opt, B_opt


# 绘制实测反射率与拟合反射率的对比图
def plot_results(nu, R_meas, theta_deg, d_opt, n0_opt, B_opt):
    """
    绘制实测反射率和理论反射率的对比图
    """
    R_model = model_reflectance(nu, d_opt, n0_opt, B_opt, theta_deg)

    plt.figure(figsize=(8, 6))
    plt.plot(nu, R_meas, label='Measured Reflectance', color='black')
    plt.plot(nu, R_model, label='Fitted Reflectance', color='red', linestyle='--')
    plt.xlabel('Wave number (cm$^{-1}$)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.title('Measured vs Fitted Reflectance')
    plt.grid(True)
    plt.show()


if __name__=="__main__":
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
    include_range: Tuple[float, float] = (1800, 2400)  # 条纹最明显波段
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
    nu_10 = out["nu_uniform"]  # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]  # 对应反射率/信号（已去均值或基线）
    theta_deg = 10.0  # 入射角（假设为 10 度）
    d0 = 6.76686e-6  # FFT 初始厚度估计（单位：cm）

    # 计算并输出优化后的外延层厚度
    d_opt, n0_opt, B_opt = main(nu_10, R10_meas, theta_deg, d0)

    # 绘制结果
    plot_results(nu_10, R10_meas, theta_deg, d_opt, n0_opt, B_opt)
