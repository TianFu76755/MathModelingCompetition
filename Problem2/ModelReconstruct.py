import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Dict, Any, Optional, Tuple

from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


# 2. 两束干涉反射率模型
def phase_two_beam(nu: np.ndarray, n_nu: np.ndarray, d_um: float, theta_i_rad: float, phi0: float) -> np.ndarray:
    """
    计算反射率相位：Φ(ν) = 4π n(ν) d cm cos(θ_t(ν)) ν + φ0
    """
    d_cm = d_um * 1e-4
    ct = np.cos(theta_i_rad)
    return 4.0 * np.pi * n_nu * d_cm * ct * nu + phi0

def model_reflectance_two_beam(nu: np.ndarray, n_nu: np.ndarray, d_um: float, theta_i_rad: float, A: float, B: float, phi0: float) -> np.ndarray:
    """
    使用两束干涉模型预测反射率： R(ν) = A + B cos(Φ(ν))
    """
    ph = phase_two_beam(nu, n_nu, d_um, theta_i_rad, phi0)
    return A + B * np.cos(ph)

# 3. 执行最小二乘拟合
def perform_fit(nu: np.ndarray, y: np.ndarray, theta_i_rad: float) -> Dict[str, Any]:
    """
    执行最小二乘拟合并返回结果。
    """
    # 初始参数（假设初始猜测值）
    initial_params = [5.0, 2.6, 0.0, np.mean(y), np.std(y)]  # d_um, n0, phi, A, B

    # 执行最小二乘优化
    result = least_squares(fit_model, initial_params, args=(nu, y, theta_i_rad))

    # 返回拟合结果
    return {
        "params": result.x,  # 拟合得到的参数
        "residuals": result.fun,  # 残差
        "success": result.success,  # 拟合成功标志
        "message": result.message,  # 拟合信息
    }

def fit_model(params: np.ndarray, nu: np.ndarray, R_meas: np.ndarray, theta_i_rad: float) -> np.ndarray:
    """
    基于模型的反射率计算： R(ν) = A + B * cos(Φ(ν))
    """
    d_um, n0, phi, A, B = params
    n_nu = np.full_like(nu, n0)  # 假设折射率为常数 n0
    R_model = model_reflectance_two_beam(nu, n_nu, d_um, theta_i_rad, A, B, phi)
    return R_model - R_meas

# 4. 绘制拟合结果
def plot_fit_results(nu: np.ndarray, y: np.ndarray, fit_result: Dict[str, Any]):
    """
    绘制拟合结果与实际数据对比图
    """
    params = fit_result["params"]
    d_um, n0, phi, A, B = params
    R_fit = model_reflectance_two_beam(nu, np.full_like(nu, n0), d_um, np.radians(10), A, B, phi)  # 使用拟合结果

    plt.figure(figsize=(10, 4))
    plt.plot(nu, y, label="原始数据", color="black", linewidth=1.5)
    plt.plot(nu, R_fit, label="拟合曲线", linestyle="--", color="red", linewidth=2.0)
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("信号 (a.u.)")
    plt.title(f"拟合结果 (d={d_um:.4f} μm, n0={n0:.4f})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# =========================双角度联合拟合======================
# 目标函数：最小化两个角度的残差
def fit_model_dual(params: np.ndarray, nu1: np.ndarray, y1: np.ndarray, nu2: np.ndarray, y2: np.ndarray, theta1_rad: float,
                   theta2_rad: float) -> np.ndarray:
    """联合拟合模型：最小化残差"""
    d_um, n0, phi1, A1, B1, phi2, A2, B2 = params
    n_nu = np.full_like(nu1, n0)  # 假设折射率为常数 n0

    # 预测反射率
    R1_pred = model_reflectance_two_beam(nu1, n_nu, d_um, theta1_rad, A1, B1, phi1)
    R2_pred = model_reflectance_two_beam(nu2, n_nu, d_um, theta2_rad, A2, B2, phi2)

    # 计算残差
    residuals = np.concatenate([R1_pred - y1, R2_pred - y2])
    return residuals


# 进行联合拟合
def perform_dual_angle_fit(nu1: np.ndarray, y1: np.ndarray, nu2: np.ndarray, y2: np.ndarray, theta1_deg: float,
                           theta2_deg: float) -> Dict[str, Any]:
    """双角度联合拟合"""
    # 初始猜测参数
    initial_params = [5.0, 2.6, 0.0, np.mean(y1), np.std(y1), 0.0, np.mean(y2),
                      np.std(y2)]  # d_um, n0, phi1, A1, B1, phi2, A2, B2

    # 执行最小二乘法拟合
    result = least_squares(fit_model_dual, initial_params,
                           args=(nu1, y1, nu2, y2, np.radians(theta1_deg), np.radians(theta2_deg)))

    # 返回拟合结果
    return {
        "params": result.x,  # 拟合得到的参数
        "residuals": result.fun,  # 残差
        "success": result.success,  # 拟合成功标志
        "message": result.message,  # 拟合信息
    }


# 绘制拟合结果
def plot_dual_angle_fit(nu1: np.ndarray, y1: np.ndarray, nu2: np.ndarray, y2: np.ndarray, fit_result: Dict[str, Any]):
    """绘制双角度联合拟合结果"""
    params = fit_result["params"]
    d_um, n0, phi1, A1, B1, phi2, A2, B2 = params

    # 预测反射率
    R1_fit = model_reflectance_two_beam(nu1, np.full_like(nu1, n0), d_um, np.radians(10), A1, B1, phi1)
    R2_fit = model_reflectance_two_beam(nu2, np.full_like(nu2, n0), d_um, np.radians(15), A2, B2, phi2)

    plt.figure(figsize=(10, 6))
    # 绘制第一个角度的拟合结果
    plt.subplot(211)
    plt.plot(nu1, y1, label="原始数据 (10°)", color="black")
    plt.plot(nu1, R1_fit, label="拟合曲线", linestyle="--", color="red")
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("信号 (a.u.)")
    plt.title(f"拟合结果 10° (d={d_um:.4f} μm, n0={n0:.4f})")
    plt.legend(loc="best")

    # 绘制第二个角度的拟合结果
    plt.subplot(212)
    plt.plot(nu2, y2, label="原始数据 (15°)", color="black")
    plt.plot(nu2, R2_fit, label="拟合曲线", linestyle="--", color="red")
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("信号 (a.u.)")
    plt.title(f"拟合结果 15° (d={d_um:.4f} μm, n0={n0:.4f})")
    plt.legend(loc="best")

    plt.tight_layout()
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
    result1 = preprocess_and_plot_compare(df1, include_range=(2060, 2280), is_plot=True)
    # 获取处理后的数据
    nu1_uniform = result1["nu_uniform"]
    y1_uniform_demean = result1["y_uniform_demean"]
    # 执行最小二乘拟合，假设入射角为 10°
    fit_result1 = perform_fit(nu1_uniform, y1_uniform_demean, np.radians(10))
    # 打印拟合结果
    print("拟合结果：", fit_result1)
    # 绘制拟合图
    plot_fit_results(nu1_uniform, y1_uniform_demean, fit_result1)

    # 预处理数据
    result2 = preprocess_and_plot_compare(df2, include_range=(2060, 2280), is_plot=True)
    # 获取处理后的数据
    nu2_uniform = result2["nu_uniform"]
    y2_uniform_demean = result2["y_uniform_demean"]
    # 执行最小二乘拟合，假设入射角为 10°
    fit_result2 = perform_fit(nu2_uniform, y2_uniform_demean, np.radians(10))
    # 打印拟合结果
    print("拟合结果：", fit_result2)
    # 绘制拟合图
    plot_fit_results(nu2_uniform, y2_uniform_demean, fit_result2)


    # ==============执行双角度联合拟合==============
    fit_result_dual = perform_dual_angle_fit(nu1_uniform, y1_uniform_demean, nu2_uniform, y2_uniform_demean, 10, 15)
    # 打印拟合结果
    print("拟合结果：", fit_result_dual)
    # 绘制拟合图
    plot_dual_angle_fit(nu1_uniform, y1_uniform_demean, nu2_uniform, y2_uniform_demean, fit_result_dual)
