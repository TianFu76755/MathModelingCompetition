import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress

from Model.Physics.PhysicsModel import PeakSpacingFFT, SpectrumPreprocessor, PhysicalParams
from Model.Physics.TwoBeamPhysics import TwoBeamPhysics


class DeltaNuEstimator:
    """
    条纹间隔 Δν 的两种估计方法：
      - 极值点回归法
      - FFT 主频法（已在 PeakSpacingFFT 实现）
    """

    @staticmethod
    def from_peaks(nu_grid: np.ndarray, R_signal: np.ndarray, prominence: float = 0.001) -> dict:
        """
        极值点法：找峰位/谷位，做线性回归，得到 Δν。
        :param nu_grid: 波数 (cm^-1)，等间隔
        :param R_signal: 处理后的反射率信号
        :param prominence: 峰的显著性阈值
        """
        # 找峰
        peaks, _ = find_peaks(R_signal, prominence=prominence)
        valleys, _ = find_peaks(-R_signal, prominence=prominence)
        idx_all = np.sort(np.concatenate([peaks, valleys]))
        if len(idx_all) < 3:
            raise ValueError("极值点太少，无法估计 Δν。")

        nu_extrema = nu_grid[idx_all]
        order = np.arange(len(nu_extrema))

        # 线性回归（更稳健地估计条纹间距）
        slope, intercept, r_value, p_value, std_err = linregress(order, nu_extrema)
        delta_nu = slope

        return dict(
            method="peak_regression",
            delta_nu_cm1=delta_nu,
            n_extrema=len(nu_extrema),
            r2=r_value**2
        )

    @staticmethod
    def from_fft(nu_grid: np.ndarray, R_signal: np.ndarray, phys_params) -> dict:
        """
        调用之前的 PeakSpacingFFT，得到 Δν
        """
        fft_estimator = PeakSpacingFFT(phys_params)
        return fft_estimator.estimate(nu_grid, R_signal)


def calculate_and_print_res(df: pd.DataFrame, phys: PhysicalParams, name: str = "默认附件") -> None:
    """打包计算过程"""
    # 预处理（统一网格+去趋势+标准化）
    pre = SpectrumPreprocessor()
    prep1 = pre.run(df)

    # 方法A：极值点回归
    res1_peaks = DeltaNuEstimator.from_peaks(prep1["nu_grid"], prep1["R_detrended"])

    # 方法B：FFT 主频
    res1_fft = DeltaNuEstimator.from_fft(prep1["nu_grid"], prep1["R_proc"], phys)

    print(name, "Δν:(极值点回归法)", res1_peaks)
    thickness1 = TwoBeamPhysics.compute_thickness(
        theta_i_deg=phys.theta_i_deg,
        n=phys.n,
        delta_nu_cm1=res1_peaks["delta_nu_cm1"],
        n0=1.0)
    print(name, "厚度:(极值点回归法)", thickness1)

    print(name, "Δν:(FFT 主频方法)", res1_fft)
    thickness2 = TwoBeamPhysics.compute_thickness(
        theta_i_deg=phys.theta_i_deg,
        n=phys.n,
        delta_nu_cm1=res1_fft["delta_nu_cm1"],
        n0=1.0)
    print(name, "厚度:(FFT 主频方法)", thickness2)


if __name__ == "__main__":
    from Data.DataManager import DM
    # 计算附件1的厚度
    df1 = DM.get_data(1)
    phys1 = PhysicalParams(n=2.59, theta_i_deg=10.0)  # 附件1 对应 10°
    calculate_and_print_res(df1, phys1, "附件1")
    # 计算附件2的厚度
    df2 = DM.get_data(2)
    phys2 = PhysicalParams(n=2.59, theta_i_deg=15.0)  # 附件1 对应 15°
    calculate_and_print_res(df2, phys2, "附件2")
