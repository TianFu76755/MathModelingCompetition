# -*- coding: utf-8 -*-
"""
quick_fit_const_n.py
最简可跑版（常数折射率 n、无吸收）：单层薄膜多光束反射，两角联合拟合厚度 d。
- 频域：波数 ν (cm^-1)，数据输入与附件一致：第二列为反射率(%)。
- 模型：未偏振 R = (Rs + Rp)/2；Airy/TMM 单层等效反射振幅 r_eff = (r01 + r12 e^{2iβ})/(1 + r01 r12 e^{2iβ}),
        β = 2π n1 d cosθ1 / λ ，而 λ = 1/ν（单位：cm），因此 β = 2π n1 d cosθ1 * ν。
- 初值：对每个角做 FFT，主频 f ≈ 2 n1 d cosθ1  ⇒ d ≈ f / (2 n1 cosθ1)，两角平均为全局初值。
- 拟合：最小二乘，变量 [d_um, α1, β1, α2, β2]，d 用 μm 表达（展示友好），内部换算 cm。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ------------- 1) Fresnel & 单层膜反射（常数 n） ----------------

def _snell_cos_theta(n_from: float, n_to: float, theta_from_rad: float) -> float:
    """由斯涅尔定律得到目的介质的 cosθ。返回实值（假设无全反射情形）。"""
    sin_from = np.sin(theta_from_rad)
    sin_to = (n_from / n_to) * sin_from
    sin_to = np.clip(sin_to, -1.0, 1.0)
    cos_to = np.sqrt(np.maximum(0.0, 1.0 - sin_to**2))
    return cos_to

def _fresnel_rs(n1: float, n2: float, cos1: float, cos2: float) -> float:
    return (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)

def _fresnel_rp(n1: float, n2: float, cos1: float, cos2: float) -> float:
    # 注意 p-偏振公式的对称形式
    return (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)

@dataclass
class SingleLayerConstN:
    n0: float = 1.0      # 入射介质（空气）
    n1: float = 3.42     # 外延层（常数）
    n2: float = 3.42     # 衬底（常数）
    # 无吸收版本：κ=0

    def reflectance_unpolarized(self, nu_cm1: np.ndarray, d_cm: float, theta_i_deg: float) -> np.ndarray:
        """
        计算未偏振反射率 R(ν)。使用 Airy/TMM 的等效反射振幅公式。
        - nu_cm1: 波数数组 [cm^-1]
        - d_cm  : 厚度（cm）
        - theta_i_deg: 入射角（空气中）
        """
        th_i = np.deg2rad(theta_i_deg)
        # 各介质内的 cosθ
        cos0 = np.cos(th_i)
        cos1 = _snell_cos_theta(self.n0, self.n1, th_i)
        cos2 = _snell_cos_theta(self.n1, self.n2, np.arccos(cos1))

        # 两界面的 Fresnel 振幅反射（s/p）
        r01_s = _fresnel_rs(self.n0, self.n1, cos0, cos1)
        r01_p = _fresnel_rp(self.n0, self.n1, cos0, cos1)
        r12_s = _fresnel_rs(self.n1, self.n2, cos1, cos2)
        r12_p = _fresnel_rp(self.n1, self.n2, cos1, cos2)

        # 相位： β = 2π n1 d cosθ1 * ν   （注意 ν=1/λ，单位一致：d[cm]、ν[cm^-1]）
        beta = 2.0 * np.pi * self.n1 * d_cm * cos1 * nu_cm1
        e2ib = np.exp(2j * beta)

        # 等效反射振幅 reff = (r01 + r12 e^{2iβ}) / (1 + r01 r12 e^{2iβ})
        r_eff_s = (r01_s + r12_s * e2ib) / (1.0 + r01_s * r12_s * e2ib)
        r_eff_p = (r01_p + r12_p * e2ib) / (1.0 + r01_p * r12_p * e2ib)
        R_s = np.abs(r_eff_s)**2
        R_p = np.abs(r_eff_p)**2
        return 0.5 * (R_s + R_p)

# ------------- 2) 工具：FFT 初值（波数域） ----------------

def estimate_d_um_from_fft(nu_cm1: np.ndarray, y: np.ndarray, n1: float, theta_i_deg: float) -> Tuple[float, float]:
    """
    在波数域对 y(ν) 做 FFT，估主频 f_peak（单位：cycle / (cm^-1)），
    利用 f ≈ 2 n1 d cosθ1  =>  d ≈ f / (2 n1 cosθ1) 得到厚度初值（μm）。
    返回 (d_um, f_peak)
    """
    # 统一等间隔（如果不是），用线性插值到等距网格（便于 FFT）
    nu = nu_cm1
    if not np.allclose(np.diff(nu), np.diff(nu)[0], rtol=1e-3, atol=1e-6):
        nu_uniform = np.linspace(nu.min(), nu.max(), len(nu))
        y = np.interp(nu_uniform, nu, y)
        nu = nu_uniform

    y_detrend = y - np.mean(y)
    window = np.hanning(len(y_detrend))
    yw = y_detrend * window

    # 频率轴（cycles per cm^-1）
    dnu = nu[1] - nu[0]
    freqs = np.fft.rfftfreq(len(yw), d=dnu)  # cycles per (cm^-1)
    Y = np.fft.rfft(yw)
    mag = np.abs(Y)

    # 跳过直流附近，找主峰
    idx_min = 3
    peak_idx = np.argmax(mag[idx_min:]) + idx_min
    f_peak = freqs[peak_idx]

    th_i = np.deg2rad(theta_i_deg)
    cos1 = _snell_cos_theta(1.0, n1, th_i)
    d_cm = f_peak / (2.0 * n1 * cos1)  # d ≈ f / (2 n cosθ1)
    d_um = d_cm * 1e4
    return d_um, f_peak

# ------------- 3) 两角联合拟合器（常数 n） ----------------

@dataclass
class TwoAngleFitterConstN:
    nu1: np.ndarray
    y1: np.ndarray   # 0~1 之间
    theta1_deg: float

    nu2: np.ndarray
    y2: np.ndarray   # 0~1 之间
    theta2_deg: float

    n1: float = 3.42
    n2: float = 3.42
    n0: float = 1.0

    def _common_nu_and_interp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求两角公共波数轴，并把 y1,y2 都插值到这个公共轴，避免不同采样点导致的拟合抖动。
        """
        nu_min = max(self.nu1.min(), self.nu2.min())
        nu_max = min(self.nu1.max(), self.nu2.max())
        m = min(len(self.nu1), len(self.nu2))
        nu_common = np.linspace(nu_min, nu_max, m)
        y1i = np.interp(nu_common, self.nu1, self.y1)
        y2i = np.interp(nu_common, self.nu2, self.y2)
        return nu_common, y1i, y2i

    def _pack_params(self, d_um, a1, b1, a2, b2) -> np.ndarray:
        return np.array([d_um, a1, b1, a2, b2], dtype=float)

    def _unpack_params(self, x: np.ndarray) -> Tuple[float, float, float, float, float]:
        return float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])

    def fit(self) -> Dict[str, Any]:
        nu, y1, y2 = self._common_nu_and_interp()

        # 初值：FFT 估计每个角的厚度，再取平均
        d1_um, f1 = estimate_d_um_from_fft(nu, y1, self.n1, self.theta1_deg)
        d2_um, f2 = estimate_d_um_from_fft(nu, y2, self.n1, self.theta2_deg)
        d0_um = 0.5 * (d1_um + d2_um)

        model = SingleLayerConstN(n0=self.n0, n1=self.n1, n2=self.n2)

        def residuals(x: np.ndarray) -> np.ndarray:
            d_um, a1, b1, a2, b2 = self._unpack_params(x)
            d_cm = d_um * 1e-4
            R1 = model.reflectance_unpolarized(nu, d_cm, self.theta1_deg)
            R2 = model.reflectance_unpolarized(nu, d_cm, self.theta2_deg)
            y1_hat = a1 + b1 * R1
            y2_hat = a2 + b2 * R2
            return np.concatenate([y1 - y1_hat, y2 - y2_hat], axis=0)

        x0 = self._pack_params(d0_um, a1=np.median(y1), b1=1.0, a2=np.median(y2), b2=1.0)
        # 合理边界：厚度正、β>0（b1,b2>0），α 允许一定范围
        lb = [0.01, -1.0, 0.0, -1.0, 0.0]   # d_um >= 0.01 μm；β>=0
        ub = [2000.,  1.0,  5.0,  1.0,  5.0]

        sol = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf", verbose=0, max_nfev=200)

        d_um, a1, b1, a2, b2 = self._unpack_params(sol.x)
        res = residuals(sol.x)
        n = len(nu)
        rmse1 = float(np.sqrt(np.mean(res[:n]**2)))
        rmse2 = float(np.sqrt(np.mean(res[n:]**2)))
        out = {
            "d_um": float(d_um),
            "per_angle": {
                "angle_1": {"theta_deg": self.theta1_deg, "alpha": float(a1), "beta": float(b1), "rmse": rmse1},
                "angle_2": {"theta_deg": self.theta2_deg, "alpha": float(a2), "beta": float(b2), "rmse": rmse2},
            },
            "fft_init": {"d1_um": float(d1_um), "d2_um": float(d2_um), "f1": float(f1), "f2": float(f2)},
            "message": sol.message,
            "success": bool(sol.success),
            "nfev": int(sol.nfev),
            "nu_common": nu,
        }
        return out

# ------------- 4) 便捷绘图（可选） ----------------

def plot_measured_vs_fit(nu: np.ndarray, y: np.ndarray, yhat: np.ndarray, title: str):
    plt.figure(figsize=(8,4))
    plt.plot(nu, y, lw=1.0, label="Measured")
    plt.plot(nu, yhat, lw=1.2, label="Model fit")
    plt.gca().invert_xaxis()  # 波数常见从大到小
    plt.xlabel("Wavenumber ν (cm$^{-1}$)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_residuals(nu: np.ndarray, r: np.ndarray, title: str):
    plt.figure(figsize=(8,3.5))
    plt.plot(nu, r, lw=0.8)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber ν (cm$^{-1}$)")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()

# ------------- 5) 示例主程（把 df3/df4 填进来就能跑） ----------------

def run_quick_demo_with_dataframes(df3: pd.DataFrame, df4: pd.DataFrame,
                                   theta1_deg: float = 10.0, theta2_deg: float = 15.0,
                                   n1: float = 3.42, n2: float = 3.42) -> Dict[str, Any]:
    """
    df3/df4: 必须包含列名 ["波数 (cm-1)","反射率 (%)"] 或 ["波数","反射率"] 之一。
    返回：拟合结果字典，并画出三张图（两角“实测 vs 拟合”、各自残差）。
    """
    # 列名适配
    def pick_cols(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cols = df.columns.tolist()
        # 容错匹配
        key_nu = [c for c in cols if "波数" in c][0]
        key_R  = [c for c in cols if "反射率" in c][0]
        nu = df[key_nu].to_numpy(dtype=float)
        y  = df[key_R].to_numpy(dtype=float) / 100.0  # % -> 0~1
        return nu, y

    nu1, y1 = pick_cols(df3)
    nu2, y2 = pick_cols(df4)

    fitter = TwoAngleFitterConstN(nu1=nu1, y1=y1, theta1_deg=theta1_deg,
                                  nu2=nu2, y2=y2, theta2_deg=theta2_deg,
                                  n1=n1, n2=n2, n0=1.0)
    res = fitter.fit()

    # 用拟合参数生成拟合曲线（在公共波数轴上）
    nu = res["nu_common"]
    model = SingleLayerConstN(n0=1.0, n1=n1, n2=n2)
    d_cm = res["d_um"] * 1e-4

    a1 = res["per_angle"]["angle_1"]["alpha"]; b1 = res["per_angle"]["angle_1"]["beta"]
    a2 = res["per_angle"]["angle_2"]["alpha"]; b2 = res["per_angle"]["angle_2"]["beta"]

    # 把原 y1,y2 也插值到公共轴，便于可视化
    y1i = np.interp(nu, nu1, y1)
    y2i = np.interp(nu, nu2, y2)

    R1 = model.reflectance_unpolarized(nu, d_cm, theta1_deg)
    R2 = model.reflectance_unpolarized(nu, d_cm, theta2_deg)
    y1_hat = a1 + b1 * R1
    y2_hat = a2 + b2 * R2

    # 绘图（可注释掉）
    plot_measured_vs_fit(nu, y1i, y1_hat,
        title=f"Angle {theta1_deg}°: Measured vs Fit (d = {res['d_um']:.4f} μm, RMSE={res['per_angle']['angle_1']['rmse']:.4f})")
    plot_measured_vs_fit(nu, y2i, y2_hat,
        title=f"Angle {theta2_deg}°: Measured vs Fit (d = {res['d_um']:.4f} μm, RMSE={res['per_angle']['angle_2']['rmse']:.4f})")

    plot_residuals(nu, y1i - y1_hat, title=f"Residuals vs ν @ {theta1_deg}°")
    plot_residuals(nu, y2i - y2_hat, title=f"Residuals vs ν @ {theta2_deg}°")
    plt.show()

    return res

# ------------- 6) 从 Excel 读取的极简示例（可选） ----------------
if __name__ == "__main__":
    from Data.DataManager import DM

    df3 = DM.get_data(3)  # 硅 @ 10°
    df4 = DM.get_data(4)  # 硅 @ 15°（同片）


    result = run_quick_demo_with_dataframes(df3, df4, theta1_deg=10.0, theta2_deg=15.0, n1=3.42, n2=3.42)
    print("=== 拟合结果（常数 n） ===")
    print({k: v for k, v in result.items() if k != "nu_common"})
