# -*- coding: utf-8 -*-
"""
Two-beam（两束）联合拟合：两角度共享厚度 d
- 输入：附件1（10°）与附件2（15°）的 DataFrame（两列：波数 (cm-1), 反射率 (%)）
- 预处理：等间隔波数栅格 + 轻度去趋势（拟合推荐用“去趋势后的 R_detrended”）
- 模型：R_j(nu) = A_j + B_j * cos( 4π n d cosθ_tj * nu + φ0_j )
- 拟合参数：d（共享） + (A1,B1,φ1) + (A2,B2,φ2)
- 输出：d（cm/μm）、各角拟合参数、残差RMS等
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from Problem3.IndicatorsCalculator import PreprocessConfig, FFTMetricsConfig, SpectrumPreprocessor, FFTMetrics


# ---------------------- 物理与模型 ----------------------
def snell_theta_t(theta_i_deg: float, n: float, n0: float = 1.0) -> float:
    """斯涅尔定律：theta_t（弧度）"""
    ti = math.radians(theta_i_deg)
    s = (n0 / n) * math.sin(ti)
    s = min(1.0, max(-1.0, s))
    return math.asin(s)


def two_beam_model(nu_cm1: np.ndarray, n: float, d_cm: float, theta_t_rad: float,
                   A: float, B: float, phi0: float) -> np.ndarray:
    """R(nu) = A + B * cos( 4π n d cosθ_t * nu + φ0 )"""
    arg = 4.0 * math.pi * n * d_cm * math.cos(theta_t_rad) * nu_cm1 + phi0
    return A + B * np.cos(arg)


# ---------------------- 拟合器主体 ----------------------
@dataclass
class JointFitConfig:
    n: float = 2.59               # 4H-SiC 工程近似
    theta1_deg: float = 10.0      # 附件1角度
    theta2_deg: float = 15.0      # 附件2角度
    use_signal: str = "detrended" # 'grid' / 'detrended' / 'proc'
    # 预处理参数（轻度去趋势；对FFT主频初猜用 proc 更稳）
    pre: PreprocessConfig = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=True)
    # FFT 主频获取 Δν 的配置（用于 d 的初猜）
    fft: FFTMetricsConfig = FFTMetricsConfig(min_periods=3.0, harmonic_max_order=4, band_frac=0.08)
    # 拟合边界
    d_bounds_cm: Tuple[float, float] = (1e-6, 1.0)  # [1e-6 cm, 1 cm] ⇒ [0.01 μm, 1e4 μm]
    B_scale_limit: float = 5.0     # B 的上限= B0 估计值 * 该倍数（避免过大）
    phi_bounds: Tuple[float, float] = (-2.0*math.pi, 2.0*math.pi)


class JointTwoBeamFitter:
    """
    两角度联合拟合（共享 d）的小工具
    """
    def __init__(self, cfg: Optional[JointFitConfig] = None):
        self.cfg = cfg or JointFitConfig()
        self.pre = SpectrumPreprocessor(self.cfg.pre)
        self.fftm = FFTMetrics(self.cfg.fft)

        # 角度 → 膜内角
        self.theta_t1 = snell_theta_t(self.cfg.theta1_deg, self.cfg.n, n0=1.0)
        self.theta_t2 = snell_theta_t(self.cfg.theta2_deg, self.cfg.n, n0=1.0)

    def _prepare(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """预处理两条谱，返回用于拟合的信号与用于FFT初猜的 Δν"""
        prep1 = self.pre.run(df1)
        prep2 = self.pre.run(df2)

        # 选拟合信号
        sig_key = {
            "grid": "R_grid",
            "detrended": "R_detrended",
            "proc": "R_proc"
        }.get(self.cfg.use_signal, "R_detrended")

        nu1, y1 = prep1["nu_grid"], prep1[sig_key]
        nu2, y2 = prep2["nu_grid"], prep2[sig_key]

        # 用 FFT 的 Δν 给 d 的初猜（两条各自估计，再取平均）
        fft1 = self.fftm.compute(prep1["nu_grid"], prep1["R_proc"])
        fft2 = self.fftm.compute(prep2["nu_grid"], prep2["R_proc"])
        d0_1 = self._d_from_delta_nu(fft1.get("delta_nu_cm1", np.nan), self.theta_t1)
        d0_2 = self._d_from_delta_nu(fft2.get("delta_nu_cm1", np.nan), self.theta_t2)

        # 初猜 d：两者可用“按 cosθ_t”折算到统一量纲后平均（直接平均也行）
        d0s = [d for d in (d0_1, d0_2) if (d is not None and np.isfinite(d) and d > 0)]
        d0 = float(np.median(d0s)) if d0s else 5e-4  # 50 μm 的保守初猜

        # A/B/φ 的初猜
        A1_0, B1_0, phi1_0 = float(np.mean(y1)), 0.5*(float(np.max(y1))-float(np.min(y1))), 0.0
        A2_0, B2_0, phi2_0 = float(np.mean(y2)), 0.5*(float(np.max(y2))-float(np.min(y2))), 0.0

        # B 的边界（避免飙大/飙负）
        span1 = max(np.max(y1) - np.min(y1), 1e-3)
        span2 = max(np.max(y2) - np.min(y2), 1e-3)
        B1_bound = (0.0, self.cfg.B_scale_limit * span1)
        B2_bound = (0.0, self.cfg.B_scale_limit * span2)

        # A 的宽松边界（围绕均值）
        A1_bound = (np.min(y1)-span1, np.max(y1)+span1)
        A2_bound = (np.min(y2)-span2, np.max(y2)+span2)

        return dict(
            nu1=nu1, y1=y1,
            nu2=nu2, y2=y2,
            d0=d0,
            A1_0=A1_0, B1_0=B1_0, phi1_0=phi1_0,
            A2_0=A2_0, B2_0=B2_0, phi2_0=phi2_0,
            bounds=dict(
                d=self.cfg.d_bounds_cm,
                A1=A1_bound, B1=B1_bound, phi1=self.cfg.phi_bounds,
                A2=A2_bound, B2=B2_bound, phi2=self.cfg.phi_bounds
            )
        )

    def _d_from_delta_nu(self, delta_nu: float, theta_t_rad: float) -> Optional[float]:
        """两束公式：d = 1 / (2 n cosθ_t Δν)；返回 cm"""
        if not (delta_nu and delta_nu > 0):
            return None
        denom = 2.0 * self.cfg.n * math.cos(theta_t_rad) * delta_nu
        if denom <= 0:
            return None
        return 1.0 / denom

    def fit_pair(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        data = self._prepare(df1, df2)

        nu1, y1 = data["nu1"], data["y1"]
        nu2, y2 = data["nu2"], data["y2"]

        # 参数向量：p = [d, A1, B1, phi1, A2, B2, phi2]
        p0 = np.array([data["d0"], data["A1_0"], data["B1_0"], data["phi1_0"],
                       data["A2_0"], data["B2_0"], data["phi2_0"]], dtype=float)

        lb = np.array([
            data["bounds"]["d"][0],
            data["bounds"]["A1"][0], data["bounds"]["B1"][0], data["bounds"]["phi1"][0],
            data["bounds"]["A2"][0], data["bounds"]["B2"][0], data["bounds"]["phi2"][0],
        ], dtype=float)
        ub = np.array([
            data["bounds"]["d"][1],
            data["bounds"]["A1"][1], data["bounds"]["B1"][1], data["bounds"]["phi1"][1],
            data["bounds"]["A2"][1], data["bounds"]["B2"][1], data["bounds"]["phi2"][1],
        ], dtype=float)

        # 残差函数：拼接两条谱的残差
        def residuals(p: np.ndarray) -> np.ndarray:
            d_cm, A1, B1, phi1, A2, B2, phi2 = p.tolist()
            y1_hat = two_beam_model(nu1, self.cfg.n, d_cm, self.theta_t1, A1, B1, phi1)
            y2_hat = two_beam_model(nu2, self.cfg.n, d_cm, self.theta_t2, A2, B2, phi2)
            # 可选加权：这里使用同权；如需按各自RMS或点数加权，可在此处修改
            return np.concatenate([y1_hat - y1, y2_hat - y2], axis=0)

        sol = least_squares(
            residuals, p0, bounds=(lb, ub), method="trf", max_nfev=30000, verbose=0
        )
        popt = sol.x
        d_cm, A1, B1, phi1, A2, B2, phi2 = popt.tolist()

        # 残差评估
        y1_hat = two_beam_model(nu1, self.cfg.n, d_cm, self.theta_t1, A1, B1, phi1)
        y2_hat = two_beam_model(nu2, self.cfg.n, d_cm, self.theta_t2, A2, B2, phi2)
        resid1 = (y1_hat - y1)
        resid2 = (y2_hat - y2)
        rms1 = float(np.sqrt(np.mean(resid1**2)))
        rms2 = float(np.sqrt(np.mean(resid2**2)))
        rms_all = float(np.sqrt(np.mean(np.concatenate([resid1, resid2])**2)))

        return dict(
            d_cm=float(d_cm),
            d_um=float(d_cm * 1.0e4),
            params=dict(
                angle1_deg=self.cfg.theta1_deg, A1=float(A1), B1=float(B1), phi1=float(phi1),
                angle2_deg=self.cfg.theta2_deg, A2=float(A2), B2=float(B2), phi2=float(phi2),
                n=self.cfg.n,
                theta_t1_deg=float(math.degrees(self.theta_t1)),
                theta_t2_deg=float(math.degrees(self.theta_t2)),
            ),
            residual_rms=dict(rms_angle1=rms1, rms_angle2=rms2, rms_joint=rms_all),
            nit=int(sol.nfev),
            success=bool(sol.success),
            message=str(sol.message),
            # 可选：回传拟合曲线，便于你后续画对比图
            fitted_curves=dict(
                nu1=nu1, y1_fit=y1_hat, y1_data=y1,
                nu2=nu2, y2_fit=y2_hat, y2_data=y2
            )
        )


# ---------------------- 例：跑附件1/2 ----------------------
if __name__ == "__main__":
    # 你已有 DM.get_data(idx)
    from Data.DataManager import DM
    df1 = DM.get_data(1)   # SiC @ 10°
    df2 = DM.get_data(2)   # SiC @ 15°

    # cfg 可按需调整（比如改用 use_signal="grid" 试试对比）
    cfg = JointFitConfig(
        n=2.59,
        theta1_deg=10.0, theta2_deg=15.0,
        use_signal="detrended",
        pre=PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=True),
        fft=FFTMetricsConfig(min_periods=3.0, harmonic_max_order=4, band_frac=0.08),
        d_bounds_cm=(1e-6, 1.0),  # 约束个合理范围即可
        B_scale_limit=5.0,
        phi_bounds=(-2.0*math.pi, 2.0*math.pi)
    )

    fitter = JointTwoBeamFitter(cfg)
    result = fitter.fit_pair(df1, df2)
    from pprint import pprint
    pprint(result)
