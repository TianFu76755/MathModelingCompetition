# -*- coding: utf-8 -*-
"""
Q3 指标计算程序（不做自动判定）：
- 预处理：等间隔波数栅格、轻度去趋势（可开关）
- 峰距 & FWHM & 精细度：FSR、FWHM、Finesse = FSR/FWHM、对比度 V
- FFT 谐波指标：主频/Δν、E2-4/E1
- Fabry–Pérot增益指标：|G| = |r01*r12|*a（给定 n0,n1,n2, θ0, 可选 k1）
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq


# -----------------------------
# 基础：预处理（等间隔 & 去趋势）
# -----------------------------
@dataclass
class PreprocessConfig:
    detrend: bool = True
    sg_window_frac: float = 0.12  # 相对长度（0~1），自动取最近奇数
    sg_polyorder: int = 2  # 低阶，多用于去基线
    normalize_proc: bool = True  # 是否对处理信号做零均值单位RMS（仅供FFT/找峰稳定）


class SpectrumPreprocessor:
    """把 (ν, R%) → 等间隔 ν & 可选去趋势；保留原始幅度版本以算对比度等。"""

    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg = cfg or PreprocessConfig()

    @staticmethod
    def _uniform_grid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        dx = np.mean(np.diff(x))
        if not np.isfinite(dx) or dx <= 0:
            return x, y
        xg = np.arange(x[0], x[-1] + 0.5 * dx, dx)
        yg = np.interp(xg, x, y)
        return xg, yg

    def _detrend(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.detrend:
            return y - np.mean(y)
        n = len(y)
        if n < 11:
            return y - np.mean(y)
        w = max(5, int(round(self.cfg.sg_window_frac * n)))
        if w % 2 == 0:
            w += 1
        # 保证 window > polyorder 且 < n
        w = min(max(w, self.cfg.sg_polyorder + 3), n - (1 if (n - 1) % 2 == 1 else 2))
        baseline = savgol_filter(y, window_length=w, polyorder=self.cfg.sg_polyorder, mode='interp')
        return y - baseline

    @staticmethod
    def _norm_unit_rms(y: np.ndarray) -> np.ndarray:
        y0 = y - np.mean(y)
        rms = np.sqrt(np.mean(y0 ** 2))
        return y0 / rms if rms > 0 else y0

    def run(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        col_nu = "波数 (cm-1)"
        col_Rp = "反射率 (%)"
        if col_nu not in df.columns or col_Rp not in df.columns:
            col_nu, col_Rp = df.columns[:2]
        nu = df[col_nu].to_numpy(dtype=float)
        R = (df[col_Rp].to_numpy(dtype=float)) / 100.0  # 0~1

        nu_g, R_g = self._uniform_grid(nu, R)
        R_det = self._detrend(R_g)
        R_proc = self._norm_unit_rms(R_det) if self.cfg.normalize_proc else R_det

        return dict(
            nu_grid=nu_g,  # cm^-1
            R_grid=R_g,  # 0~1
            R_detrended=R_det,  # 去趋势
            R_proc=R_proc  # 标准化（FFT/找峰）
        )


# -----------------------------
# 指标 A：峰距/FSR & FWHM & 精细度
# -----------------------------
@dataclass
class PeakMetricsConfig:
    prominence: float = 0.25  # 峰显著性（标准化信号上）
    width_rel_height: float = 0.5  # FWHM 使用的相对高度 0.5
    min_peaks: int = 5  # 至少若干峰以保证稳健
    use_valleys: bool = False  # 也可切换用“谷值”


class PeakMetrics:
    """
    在去趋势/标准化信号上找峰，估计：
      - FSR (cm^-1)：相邻峰位置回归的平均间距
      - FWHM (cm^-1)：峰半高宽平均
      - Finesse：FSR / FWHM
      - 对比度 V：在原始 R_grid 上计算 (Rmax - Rmin) / (Rmax + Rmin)
    """

    def __init__(self, cfg: Optional[PeakMetricsConfig] = None):
        self.cfg = cfg or PeakMetricsConfig()

    @staticmethod
    def _contrast(R_grid: np.ndarray, q_clip: float = 0.005) -> float:
        # 用分位数抑制孤立噪点
        lo = np.quantile(R_grid, q_clip)
        hi = np.quantile(R_grid, 1 - q_clip)
        return (hi - lo) / (hi + lo + 1e-12)

    def compute(self, nu_grid: np.ndarray, R_proc: np.ndarray, R_grid_for_contrast: np.ndarray) -> Dict[str, Any]:
        y = R_proc
        # 根据 use_valleys 选择对 -y 找峰
        y_find = (-y) if self.cfg.use_valleys else y
        peaks, props = find_peaks(y_find, prominence=self.cfg.prominence)

        if len(peaks) < self.cfg.min_peaks:
            return dict(FSR_cm1=np.nan, FWHM_cm1=np.nan, Finesse=np.nan,
                        contrast_V=self._contrast(R_grid_for_contrast),
                        n_peaks=int(len(peaks)))

        # 峰位置
        nu_pk = nu_grid[peaks]
        nu_pk.sort()

        # FSR：对 (k, nu_k) 做线性回归斜率
        k = np.arange(len(nu_pk))
        A = np.vstack([k, np.ones_like(k)]).T
        slope, intercept = np.linalg.lstsq(A, nu_pk, rcond=None)[0]
        fsr = float(slope)  # cm^-1

        # FWHM：用 peak_widths 在相对高度求宽度（单位：样本点），再乘 dν
        widths, width_heights, left_ips, right_ips = peak_widths(
            y_find, peaks, rel_height=1 - self.cfg.width_rel_height
        )
        dnu = float(np.mean(np.diff(nu_grid)))
        fwhm = float(np.nanmean(widths) * abs(dnu))

        finesse = fsr / fwhm if (np.isfinite(fsr) and np.isfinite(fwhm) and fwhm > 0) else np.nan
        V = self._contrast(R_grid_for_contrast)

        return dict(
            FSR_cm1=fsr,
            FWHM_cm1=fwhm,
            Finesse=finesse,
            contrast_V=V,
            n_peaks=int(len(peaks)),
        )


# -----------------------------
# 指标 B：FFT 主频与谐波占比
# -----------------------------
@dataclass
class FFTMetricsConfig:
    min_periods: float = 3.0  # 至少包含若干个条纹周期
    harmonic_max_order: int = 4  # 统计到 4 次谐波
    band_frac: float = 0.08  # 主频/谐波能量统计的带宽（相对频率）


class FFTMetrics:
    """
    在等间隔波数栅格上做 FFT，给出：
      - 主频 f1（cycles / (cm^-1)），Δν ≈ 1 / f1
      - 谐波能量比 ρ = sum(E2..E_N) / E1
    """

    def __init__(self, cfg: Optional[FFTMetricsConfig] = None):
        self.cfg = cfg or FFTMetricsConfig()

    @staticmethod
    def _band_energy(freqs: np.ndarray, amp2: np.ndarray, f0: float, bw: float) -> float:
        mask = (freqs >= max(0.0, f0 - bw)) & (freqs <= (f0 + bw))
        return float(np.sum(amp2[mask]))

    def compute(self, nu_grid: np.ndarray, R_proc: np.ndarray) -> Dict[str, Any]:
        n = len(nu_grid)
        if n < 16:
            return dict(delta_nu_cm1=np.nan, f1=np.nan, harmonic_ratio=np.nan)

        dnu = float(np.mean(np.diff(nu_grid)))
        nu_span = float(nu_grid[-1] - nu_grid[0])
        if nu_span <= 0:
            return dict(delta_nu_cm1=np.nan, f1=np.nan, harmonic_ratio=np.nan)

        # FFT（仅幅度即可；能量用 |Y|^2）
        Y = rfft(R_proc)
        freqs = rfftfreq(n, d=dnu)
        amp = np.abs(Y)
        amp2 = amp ** 2

        # 跳过极低频（避免基线残留），至少保证 min_periods 个条纹
        low_cut = self.cfg.min_periods / (nu_span + 1e-12)
        valid = freqs > low_cut
        if not np.any(valid):
            return dict(delta_nu_cm1=np.nan, f1=np.nan, harmonic_ratio=np.nan)

        fpos = freqs[valid]
        ap = amp[valid]
        a2 = amp2[valid]

        i1 = int(np.argmax(ap))
        f1 = float(fpos[i1])
        delta_nu = 1.0 / f1 if f1 > 0 else np.nan

        # 主频能量 E1 与谐波能量和 E2-4
        bw = self.cfg.band_frac * f1
        E1 = self._band_energy(freqs, amp2, f1, bw)
        Eh = 0.0
        for k in range(2, self.cfg.harmonic_max_order + 1):
            Eh += self._band_energy(freqs, amp2, k * f1, bw * (1.0 if k <= 2 else 1.2))
        rho = (Eh / E1) if (E1 > 0 and np.isfinite(E1)) else np.nan

        return dict(delta_nu_cm1=delta_nu, f1=f1, harmonic_ratio=rho)


# -----------------------------
# 指标 C：Fabry–Pérot 往返增益 |G|
# -----------------------------
@dataclass
class FPParams:
    n0: float = 1.0  # 入射介质（空气）
    n1: float = 3.48  # 薄膜（默认 Si 近红外示意；SiC 可改 2.59）
    n2: float = 3.48  # 衬底（同材可设相同，或按需填写）
    theta0_deg: float = 10.0
    pol: str = "unpolarized"  # "s"/"p"/"unpolarized"
    k1: float = 0.0  # 吸收系数（n1 - i k1）
    use_absorption: bool = False  # 若 True，则按 ν 中位数估算 a<1；否则 a=1


class FabryPerotGainMetric:
    """
    估计多光束级数的“往返增益模值”：|G| = |r01 * r12| * a
    - r_ij 取 Fresnel 振幅反射系数
    - a = exp(-4π k1 d / (λ cosθ1))；由于未知 d，这里仅给“单位往返”的吸收
      估计：取 ν 的中位数对应 λ，d 用『一个条纹半程量纲消除』的工程近似：
      用 Δν 的估计把『2 n1 d cosθ1』≈ 1/Δν 代入，则 a ≈ exp( - 2π k1 / (ν_med) * 2π ?? )
      ——为避免过拟合/误导，默认不启用吸收 (use_absorption=False)。
    """

    def __init__(self, params: Optional[FPParams] = None):
        self.p = params or FPParams()

    @staticmethod
    def _snell_theta(n_from: float, n_to: float, theta_from: float) -> float:
        s = (n_from / n_to) * math.sin(theta_from)
        s = max(-1.0, min(1.0, s))
        return math.asin(s)

    @staticmethod
    def _r_s(n_i: float, n_j: float, ci: float, cj: float) -> float:
        return (n_i * ci - n_j * cj) / (n_i * ci + n_j * cj)

    @staticmethod
    def _r_p(n_i: float, n_j: float, ci: float, cj: float) -> float:
        return (n_j * ci - n_i * cj) / (n_j * ci + n_i * cj)

    def _r01_r12(self) -> Tuple[complex, complex]:
        th0 = math.radians(self.p.theta0_deg)
        th1 = self._snell_theta(self.p.n0, self.p.n1, th0)
        th2 = self._snell_theta(self.p.n1, self.p.n2, th1)

        c0, c1, c2 = math.cos(th0), math.cos(th1), math.cos(th2)

        rs01 = self._r_s(self.p.n0, self.p.n1, c0, c1)
        rs12 = self._r_s(self.p.n1, self.p.n2, c1, c2)
        rp01 = self._r_p(self.p.n0, self.p.n1, c0, c1)
        rp12 = self._r_p(self.p.n1, self.p.n2, c1, c2)

        pol = self.p.pol.lower()
        if pol == "s":
            return complex(rs01), complex(rs12)
        elif pol == "p":
            return complex(rp01), complex(rp12)
        else:
            # 非偏振：幅度平均并不严格，这里用反射率平均再取有效幅度的工程近似
            R01 = 0.5 * (abs(rs01) ** 2 + abs(rp01) ** 2)
            R12 = 0.5 * (abs(rs12) ** 2 + abs(rp12) ** 2)
            r01_eff = np.sign((rs01 + rp01) / 2.0) * math.sqrt(R01)
            r12_eff = np.sign((rs12 + rp12) / 2.0) * math.sqrt(R12)
            return complex(r01_eff), complex(r12_eff)

    def compute(self, nu_grid: np.ndarray, delta_nu_cm1: Optional[float] = None) -> Dict[str, Any]:
        r01, r12 = self._r01_r12()

        # 吸收项（默认关闭）。若开启，则估算 a≈exp(-4π k1 d / (λ cosθ1))。
        a = 1.0
        if self.p.use_absorption and self.p.k1 > 0:
            # 工程近似：λ 取 ν 中位数对应；d 若未知，可用 Δν 关系 2 n1 d cosθ1 ≈ 1/Δν
            nu_med = float(np.median(nu_grid))
            if nu_med > 0 and delta_nu_cm1 and delta_nu_cm1 > 0:
                th0 = math.radians(self.p.theta0_deg)
                th1 = self._snell_theta(self.p.n0, self.p.n1, th0)
                cos1 = math.cos(th1)
                # 2 n1 d cosθ1 ≈ 1/Δν -> d ≈ 1/(2 n1 cosθ1 Δν)
                d_est = 1.0 / (2.0 * self.p.n1 * cos1 * delta_nu_cm1)  # cm
                lam_cm = 1.0 / nu_med  # cm
                a = math.exp(-4.0 * math.pi * self.p.k1 * d_est / (lam_cm * cos1))

        G = abs(r01 * r12) * a
        return dict(G=G, R01=abs(r01) ** 2, R12=abs(r12) ** 2, a=a)


# -----------------------------
# 总打包：一键算出所有指标
# -----------------------------
@dataclass
class IndicatorsConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    peak: PeakMetricsConfig = PeakMetricsConfig()
    fft: FFTMetricsConfig = FFTMetricsConfig()
    fp_params: FPParams = FPParams()  # 根据样品/角度自行填写（Si/SiC 等）


class IndicatorsRunner:
    """
    用法：
        df = DM.get_data(3)  # 或 4 / 1 / 2
        cfg = IndicatorsConfig(fp_params=FPParams(n0=1.0, n1=3.42, n2=3.42, theta0_deg=10, pol="unpolarized"))
        result = IndicatorsRunner(cfg).run(df)
        print(result)
    """

    def __init__(self, cfg: Optional[IndicatorsConfig] = None):
        self.cfg = cfg or IndicatorsConfig()
        self.pre = SpectrumPreprocessor(self.cfg.preprocess)
        self.pk = PeakMetrics(self.cfg.peak)
        self.fft = FFTMetrics(self.cfg.fft)
        self.fp = FabryPerotGainMetric(self.cfg.fp_params)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        prep = self.pre.run(df)
        nu = prep["nu_grid"]
        R_grid = prep["R_grid"]
        R_proc = prep["R_proc"]

        # 峰相关指标
        peak_res = self.pk.compute(nu, R_proc, R_grid)

        # FFT 指标
        fft_res = self.fft.compute(nu, R_proc)

        # Fabry–Pérot 往返增益 |G|
        fp_res = self.fp.compute(nu, delta_nu_cm1=fft_res.get("delta_nu_cm1"))

        # 汇总（不做任何判断，仅回传指标）
        return dict(
            peak_metrics=peak_res,  # FSR_cm1, FWHM_cm1, Finesse, contrast_V, n_peaks
            fft_metrics=fft_res,  # delta_nu_cm1, f1, harmonic_ratio
            fabry_perot_gain=fp_res,  # G, R01, R12, a
            notes="仅指标计算，不做自动判定。"
        )


if __name__ == "__main__":
    from Data.DataManager import DM
    # 例：硅片（附件3，10°）
    df3 = DM.get_data(3)
    cfg3 = IndicatorsConfig(
        fp_params=FPParams(n0=1.0, n1=3.42, n2=3.42 + 0.005, theta0_deg=10, pol="unpolarized")
    )
    res3 = IndicatorsRunner(cfg3).run(df3)
    pprint(f"res3: {res3}")

    # 例：硅片（附件4，15°）
    df4 = DM.get_data(4)
    cfg4 = IndicatorsConfig(
        fp_params=FPParams(n0=1.0, n1=3.42, n2=3.42 + 0.005, theta0_deg=15, pol="unpolarized")
    )
    res4 = IndicatorsRunner(cfg4).run(df4)
    pprint(f"res4: {res4}")

    from Model.Physics.TwoBeamPhysics import TwoBeamPhysics
    pprint(TwoBeamPhysics.compute_thickness(theta_i_deg=10, n=3.42, delta_nu_cm1=360.0929416671038))