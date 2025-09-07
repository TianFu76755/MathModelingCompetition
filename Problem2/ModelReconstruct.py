import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Dict, Any, Optional, Tuple

from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


# 目标函数（用于最小二乘拟合）
def fit_model(params: np.ndarray, nu: np.ndarray, R_meas: np.ndarray) -> np.ndarray:
    """
    基于模型的反射率计算： R(ν) = A + B * cos(Φ(ν))
    """
    d_um, n0, phi, A, B = params
    phase = 4 * np.pi * n0 * d_um * nu  # 示例计算，简化版模型
    R_model = A + B * np.cos(phase + phi)  # 简单的余弦模型
    return R_model - R_meas

# 进行拟合
def perform_fit(nu: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    执行最小二乘拟合并返回结果。
    """
    # 初始参数（假设初始猜测值）
    initial_params = [5.0, 2.6, 0.0, np.mean(y), np.std(y)]  # d_um, n0, phi, A, B

    # 执行最小二乘优化
    result = least_squares(fit_model, initial_params, args=(nu, y))

    # 返回拟合结果
    return {
        "params": result.x,  # 拟合得到的参数
        "residuals": result.fun,  # 残差
        "success": result.success,  # 拟合成功标志
        "message": result.message,  # 拟合信息
    }

# 进行绘图
def plot_fit_results(nu: np.ndarray, y: np.ndarray, fit_result: Dict[str, Any]):
    """
    绘制拟合结果与实际数据对比图
    """
    params = fit_result["params"]
    d_um, n0, phi, A, B = params
    R_fit = A + B * np.cos(4 * np.pi * n0 * d_um * nu + phi)  # 计算拟合结果

    plt.figure(figsize=(10, 4))
    plt.plot(nu, y, label="原始数据", color="black", linewidth=1.5)
    plt.plot(nu, R_fit, label="拟合曲线", linestyle="--", color="red", linewidth=2.0)
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("信号 (a.u.)")
    plt.title(f"拟合结果 (d={d_um:.4f} μm, n0={n0:.4f})")
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

    # 预处理数据
    result1 = preprocess_and_plot_compare(df1, include_range=(1850, 2200), is_plot=True)

    # 获取处理后的数据
    nu1_uniform = result1["nu_uniform"]
    y1_uniform_demean = result1["y_uniform_demean"]

    # 执行最小二乘拟合
    fit_result = perform_fit(nu1_uniform, y1_uniform_demean)

    # 打印拟合结果
    print("拟合结果：", fit_result)

    # 绘制拟合图
    plot_fit_results(nu1_uniform, y1_uniform_demean, fit_result)
