import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class SiliconThicknessCalculator:
    def __init__(self, n_constant=3.5, n_substrate=3.5):
        """
        初始化硅外延层厚度计算器

        参数:
        n_constant: 假设的常数折射率
        n_substrate: 衬底折射率
        """
        self.n_constant = n_constant
        self.n_substrate = n_substrate
        self.n_air = 1.0  # 空气折射率
        self.results = {}

    def load_data(self, df10, df15):
        """
        加载数据

        参数:
        file_path_10deg: 10度入射角数据文件路径
        file_path_15deg: 15度入射角数据文件路径
        """
        self.df_10deg = df10
        self.df_15deg = df15

        # 将波数转换为波长(μm)
        self.df_10deg['wavelength'] = 10000 / self.df_10deg.iloc[:, 0]
        self.df_15deg['wavelength'] = 10000 / self.df_15deg.iloc[:, 0]

        # 反射率数据
        self.reflectivity_10deg = self.df_10deg.iloc[:, 1].values
        self.reflectivity_15deg = self.df_15deg.iloc[:, 1].values

        # 波长数据
        self.wavelengths_10deg = self.df_10deg['wavelength'].values
        self.wavelengths_15deg = self.df_15deg['wavelength'].values

    def multi_beam_reflectivity(self, d, theta, wavelengths, n_epi):
        """
        计算多光束干涉的理论反射率

        参数:
        d: 外延层厚度
        theta: 入射角(度)
        wavelengths: 波长数组
        n_epi: 外延层折射率

        返回:
        理论反射率数组
        """
        # 转换为弧度
        theta_rad = np.deg2rad(theta)

        # 计算折射角
        theta_1 = np.arcsin(np.sin(theta_rad) / n_epi)

        # 计算界面反射率
        R1 = ((self.n_air - n_epi) / (self.n_air + n_epi)) ** 2
        R2 = ((n_epi - self.n_substrate) / (n_epi + self.n_substrate)) ** 2

        # 计算相位差
        delta = 4 * np.pi * n_epi * d * np.cos(theta_1) / wavelengths

        # 计算多光束干涉反射率
        numerator = R1 + R2 + 2 * np.sqrt(R1 * R2) * np.cos(delta)
        denominator = 1 + R1 * R2 + 2 * np.sqrt(R1 * R2) * np.cos(delta)

        return numerator / denominator

    def objective_function(self, d, theta, wavelengths, measured_reflectivity):
        """
        目标函数：计算理论反射率与测量反射率之间的差异

        参数:
        d: 外延层厚度
        theta: 入射角(度)
        wavelengths: 波长数组
        measured_reflectivity: 测量的反射率数组

        返回:
        残差平方和
        """
        theoretical_reflectivity = self.multi_beam_reflectivity(d, theta, wavelengths, self.n_constant)
        return np.sum((theoretical_reflectivity - measured_reflectivity / 100) ** 2)

    def calculate_thickness(self, initial_guess=10.0, bounds=(0.1, 100.0)):
        """
        计算外延层厚度

        参数:
        initial_guess: 初始厚度猜测值(μm)
        bounds: 厚度搜索范围

        返回:
        优化结果
        """
        # 为10度入射角数据计算厚度
        res_10deg = minimize(
            self.objective_function,
            initial_guess,
            args=(10, self.wavelengths_10deg, self.reflectivity_10deg),
            bounds=[bounds]
        )

        # 为15度入射角数据计算厚度
        res_15deg = minimize(
            self.objective_function,
            initial_guess,
            args=(15, self.wavelengths_15deg, self.reflectivity_15deg),
            bounds=[bounds]
        )

        self.results = {
            '10_deg': {
                'thickness': res_10deg.x[0],
                'success': res_10deg.success,
                'message': res_10deg.message,
                'fun': res_10deg.fun
            },
            '15_deg': {
                'thickness': res_15deg.x[0],
                'success': res_15deg.success,
                'message': res_15deg.message,
                'fun': res_15deg.fun
            }
        }

        return self.results

    def plot_results(self, theta_deg):
        """
        绘制理论反射率与测量反射率的对比图

        参数:
        theta_deg: 入射角(10或15度)
        """
        if theta_deg == 10:
            wavelengths = self.wavelengths_10deg
            measured_reflectivity = self.reflectivity_10deg
            d = self.results['10_deg']['thickness']
            title = "10° Incident Angle"
        else:
            wavelengths = self.wavelengths_15deg
            measured_reflectivity = self.reflectivity_15deg
            d = self.results['15_deg']['thickness']
            title = "15° Incident Angle"

        theoretical_reflectivity = self.multi_beam_reflectivity(
            d, theta_deg, wavelengths, self.n_constant
        )

        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, measured_reflectivity / 100, 'b-', label='Measured Reflectivity')
        plt.plot(wavelengths, theoretical_reflectivity, 'r--', label='Theoretical Reflectivity')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectivity')
        plt.title(f'{title} - Thickness: {d:.2f} μm')
        plt.legend()
        plt.grid(True)
        plt.show()


# 使用示例
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 创建计算器实例，假设折射率为3.5
    calculator = SiliconThicknessCalculator(n_constant=3.5, n_substrate=3.5)

    # 加载数据（请替换为实际文件路径）
    calculator.load_data(df3, df4)

    # 计算厚度
    results = calculator.calculate_thickness(initial_guess=3.0, bounds=(0.1, 100.0))

    # 打印结果
    print("10度入射角计算结果:")
    print(f"厚度: {results['10_deg']['thickness']:.4f} μm")
    print(f"优化是否成功: {results['10_deg']['success']}")
    print(f"残差: {results['10_deg']['fun']:.6f}")

    print("\n15度入射角计算结果:")
    print(f"厚度: {results['15_deg']['thickness']:.4f} μm")
    print(f"优化是否成功: {results['15_deg']['success']}")
    print(f"残差: {results['15_deg']['fun']:.6f}")

    # 绘制结果（需要先加载数据）
    calculator.plot_results(10)
    calculator.plot_results(15)
