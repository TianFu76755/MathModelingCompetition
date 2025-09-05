# -*- coding: utf-8 -*-
"""
R & Airy-F (多光束) 参数估计 + 条纹对比（严谨版，n(λ) 由 Callable 提供）

功能：
- 接收 DataFrame（两列：'波数 (cm-1)', '反射率 (%)'）
- 预处理：等间距、Savitzky–Golay 去趋势、可选波数窗口
- 由外部提供的 n(λ[μm]) 计算 R(λ)、F(λ)=4R/(1-R)^2（逐波长）
- 统计输出：R/F 的 mean/median/p5/p95；FFT 主频、FSR、主峰清晰度等
- 可视化：实测（去趋势） vs. Airy(反射) 与 两束 Cosine 的对比（频率取 FFT 主峰）

依赖：numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares


# =============================
# 公共小工具
# =============================

EPS = 1e-12

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    """λ[μm] = 1e4 / ν[cm^-1]"""
    return 1e4 / np.maximum(nu_cm1, EPS)

def fresnel_R_from_n(n: np.ndarray, n0: complex | float = 1.0) -> np.ndarray:
    """
    正入射 Fresnel 反射率（允许 n 为复数）：
    R = |(n - n0) / (n + n0)|^2
    """
    r = (n - n0) / (n + n0 + 0j)
    return np.abs(r) ** 2

def airy_F_from_R(R: np.ndarray) -> np.ndarray:
    """Airy 系数 F = 4R / (1 - R)^2（逐波长）"""
    return 4.0 * R / (np.maximum(1.0 - R, 1e-9) ** 2)

def stats_dict(x: np.ndarray) -> Dict[str, float]:
    """返回均值/中位数/分位数的简表"""
    return dict(
        mean=float(np.nanmean(x)),
        median=float(np.nanmedian(x)),
        p05=float(np.nanpercentile(x, 5)),
        p95=float(np.nanpercentile(x, 95)),
        min=float(np.nanmin(x)),
        max=float(np.nanmax(x)),
    )


# =============================
# 预处理
# =============================

@dataclass
class PreprocessCfg:
    detrend: bool = True
    sg_window_frac: float = 0.15
    sg_polyorder: int = 2
    normalize_for_fft: bool = True
    nu_min: Optional[float] = None  # 波数窗口（cm^-1）
    nu_max: Optional[float] = None

class Preprocessor:
    def __init__(self, cfg: Optional[PreprocessCfg] = None):
        self.cfg = cfg or PreprocessCfg()

    def _uniform_grid(self, nu: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(nu)
        nu, R = nu[idx], R[idx]
        dnu = np.mean(np.diff(nu))
        if not np.isfinite(dnu) or dnu <= 0:
            return nu, R
        nu_g = np.arange(nu[0], nu[-1] + 0.5*dnu, dnu)
        R_g = np.interp(nu_g, nu, R)
        return nu_g, R_g

    def _window(self, nu: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = -np.inf if self.cfg.nu_min is None else self.cfg.nu_min
        hi =  np.inf if self.cfg.nu_max is None else self.cfg.nu_max
        m = (nu >= lo) & (nu <= hi)
        return nu[m], y[m]

    def _detrend(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.detrend or len(y) < 11:
            return y - float(np.nanmean(y))
        n = len(y)
        w = max(5, int(round(self.cfg.sg_window_frac * n)))
        if w % 2 == 0: w += 1
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        w = max(w, self.cfg.sg_polyorder + 3)
        base = savgol_filter(y, window_length=w, polyorder=self.cfg.sg_polyorder, mode='interp')
        return y - base

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_for_fft:
            return y
        z = y - float(np.mean(y))
        rms = math.sqrt(float(np.mean(z**2))) + EPS
        return z / rms

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        col_nu = "波数 (cm-1)"
        col_R  = "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            col_nu, col_R = df.columns[:2]
        nu = df[col_nu].to_numpy(dtype=float)
        R  = (df[col_R].to_numpy(dtype=float)) / 100.0

        nu_g, R_g = self._uniform_grid(nu, R)
        nu_w, R_w = self._window(nu_g, R_g)
        y_det = self._detrend(R_w)
        y_fft = self._normalize(y_det)

        return dict(
            nu_full=nu_g, R_full=R_g,
            nu=nu_w, R=R_w,
            y_det=y_det, y_fft=y_fft
        )


# =============================
# 频域与条纹形状指标
# =============================

def fft_primary(nu: np.ndarray, y_fft: np.ndarray, min_cycles: float = 3.0) -> Dict[str, float]:
    if len(nu) < 8:
        raise ValueError("样本过少，无法 FFT。")
    dnu = float(np.mean(np.diff(nu)))
    Y = rfft(y_fft - float(np.mean(y_fft)))
    freqs = rfftfreq(len(y_fft), d=dnu)
    span = float(nu[-1] - nu[0] + EPS)
    low_cut = (1.0 / span) * min_cycles
    m = freqs > low_cut
    amp = np.abs(Y)[m]; fpos = freqs[m]
    k = int(np.argmax(amp))
    f_peak = float(fpos[k])
    a_sorted = np.sort(amp)
    sec = float(a_sorted[-2]) if len(a_sorted) >= 2 else 0.0
    clarity = float(amp[k]) / (sec + EPS)
    return dict(f_peak=f_peak, delta_nu=1.0/f_peak, clarity=clarity)

def peaks_shape_metrics(nu: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n >= 21:
        y_s = savgol_filter(y, window_length=(max(11, (n//120)*2+1)), polyorder=2, mode='interp')
    else:
        y_s = y
    pk, prop = find_peaks(y_s, prominence=np.percentile(np.abs(y_s), 75) * 0.2)
    if len(pk) >= 3:
        periods = np.diff(nu[pk])
        pw = peak_widths(y_s, pk, rel_height=0.5)
        fwhm = pw[0] * np.mean(np.diff(nu))
        mean_period = float(np.mean(periods))
        mean_fwhm   = float(np.mean(fwhm)) if np.isfinite(np.mean(fwhm)) else np.nan
        sharp = float(mean_period / (mean_fwhm + EPS)) if mean_fwhm>0 else np.nan
    else:
        mean_period = np.nan; mean_fwhm = np.nan; sharp = np.nan
    vis = float(np.percentile(y, 95) - np.percentile(y, 5)) / 2.0
    return dict(mean_period_cm1=mean_period, mean_fwhm_cm1=mean_fwhm, sharp_ratio=sharp, visibility=vis)


# =============================
# 模型函数（仅用于对比可视化）
# =============================

def airy_reflection(delta: np.ndarray, F_eff: float) -> np.ndarray:
    s2 = np.sin(delta/2.0)**2
    return (F_eff * s2) / (1.0 + F_eff * s2 + EPS)

def cosine_base(delta: np.ndarray) -> np.ndarray:
    return np.cos(delta)

def fit_A_B_phi(nu: np.ndarray, y: np.ndarray, f_peak: float, base: str, F_eff: Optional[float] = None) -> Tuple[float, float, float]:
    """
    拟合 A+B*base(delta) 的 A/B/phi；delta = 2π f_peak (nu - nu0) + phi
    base: 'airy' or 'cos'
    """
    nu0 = float(0.5*(nu[0]+nu[-1]))
    def residual(p):
        A, B, phi = p
        delta = 2.0*np.pi*f_peak*(nu - nu0) + phi
        if base == 'airy':
            s = airy_reflection(delta, F_eff if F_eff is not None else 0.0)
        else:
            s = cosine_base(delta)
        return A + B*s - y

    A0 = float(np.mean(y))
    B0 = float(0.5*(np.max(y)-np.min(y)))
    p0 = np.array([A0, B0, 0.0], dtype=float)
    r = least_squares(residual, p0, loss='cauchy', f_scale=1.5*robust_mad(y), max_nfev=15000)
    return float(r.x[0]), float(r.x[1]), float(r.x[2])


# =============================
# 数据集配置 & 主分析器
# =============================

@dataclass
class DatasetCfg:
    name: str
    n_of_lambda_um: Callable[[np.ndarray], np.ndarray]  # 折射率函数：λ[μm] -> n(λ)（可为复数）
    n0_env: float = 1.0                                  # 环境折射率
    theta_i_deg: float = 0.0                             # 仅用于报告
    nu_min: Optional[float] = None                       # 波数窗口（cm^-1）
    nu_max: Optional[float] = None

class RandFAnalyzer:
    def __init__(self, pre_cfg: Optional[PreprocessCfg] = None):
        self.pre_cfg = pre_cfg or PreprocessCfg()

    def analyze_one(self, df: pd.DataFrame, cfg: DatasetCfg, make_plot: bool = True) -> Dict[str, Any]:
        # 预处理（应用每个数据集自己的窗口）
        local_pre = PreprocessCfg(
            detrend=self.pre_cfg.detrend,
            sg_window_frac=self.pre_cfg.sg_window_frac,
            sg_polyorder=self.pre_cfg.sg_polyorder,
            normalize_for_fft=self.pre_cfg.normalize_for_fft,
            nu_min=cfg.nu_min, nu_max=cfg.nu_max
        )
        pre = Preprocessor(local_pre).run(df)
        nu, y_det, y_fft = pre["nu"], pre["y_det"], pre["y_fft"]

        # 频域主峰（与窗口一致）
        fft = fft_primary(nu, y_fft, min_cycles=3.0)
        fpk, fsr = fft["f_peak"], fft["delta_nu"]

        # 由 n(λ) 计算 R(λ)、F(λ)
        lam_um = wavelength_um_from_wavenumber_cm1(nu)
        n_lambda = np.asarray(cfg.n_of_lambda_um(lam_um))
        R_lambda = fresnel_R_from_n(n_lambda, n0=cfg.n0_env)
        F_lambda = airy_F_from_R(R_lambda)
        R_stats  = stats_dict(R_lambda)
        F_stats  = stats_dict(F_lambda)

        # 条纹形状指标
        pk_metrics = peaks_shape_metrics(nu, y_det)

        # 用 F_eff（中位数）做 Airy 对比拟合；同时给两束 cos 对比
        F_eff = float(np.nanmedian(F_lambda))
        A1,B1,phi1 = fit_A_B_phi(nu, y_det, fpk, base='airy', F_eff=F_eff)
        A2,B2,phi2 = fit_A_B_phi(nu, y_det, fpk, base='cos',  F_eff=None)

        fig = None
        if make_plot:
            nu0 = float(0.5*(nu[0]+nu[-1]))
            delta_air = 2.0*np.pi*fpk*(nu - nu0) + phi1
            delta_cos = 2.0*np.pi*fpk*(nu - nu0) + phi2
            y_air = A1 + B1*airy_reflection(delta_air, F_eff)
            y_cos = A2 + B2*cosine_base(delta_cos)

            fig, ax = plt.subplots(1,1, figsize=(9,3.6))
            ax.plot(nu, y_det, lw=1.0, label="实测(去趋势)")
            ax.plot(nu, y_air, lw=1.0, ls="--", label=f"Airy(反射)  F_eff≈{F_eff:.2f}")
            ax.plot(nu, y_cos, lw=1.0, ls=":",  label="两束 Cosine")
            ax.set_xlabel("波数 (cm$^{-1}$)")
            ax.set_ylabel("相对幅值")
            title = f"{cfg.name} 条纹对比（窗口：{cfg.nu_min or 'min'}~{cfg.nu_max or 'max'} cm$^{{-1}}$）"
            ax.set_title(title)
            ax.grid(alpha=0.3); ax.legend(loc="best")
            fig.tight_layout()

        out = dict(
            name=cfg.name,
            n0_env=cfg.n0_env,
            theta_i_deg=cfg.theta_i_deg,
            f_peak=fpk, FSR_cm1=fsr, clarity_fft=fft["clarity"],
            R_stats=R_stats,              # {'mean','median','p05','p95','min','max'}
            F_stats=F_stats,              # 同上
            visibility=pk_metrics["visibility"],
            mean_period_cm1=pk_metrics["mean_period_cm1"],
            mean_fwhm_cm1=pk_metrics["mean_fwhm_cm1"],
            sharp_ratio=pk_metrics["sharp_ratio"],
            F_eff=F_eff,
            fit_params=dict(
                Airy=dict(A=A1,B=B1,phi=phi1),
                Cosine=dict(A=A2,B=B2,phi=phi2)
            ),
            arrays=dict(nu=nu, y_det=y_det, R_lambda=R_lambda, F_lambda=F_lambda),  # 如需另存
            figure=fig
        )
        return out

    def analyze_many(self, data_map: Dict[str, Tuple[pd.DataFrame, DatasetCfg]], make_plot: bool=True) -> Dict[str, Any]:
        results = {}
        for key, (df, cfg) in data_map.items():
            results[key] = self.analyze_one(df, cfg, make_plot=make_plot)
        return results


# =============================
# 折射率函数示例
# =============================

def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    """
    硅的色散： n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814*λ^2
    参考式要求 λ[μm]；为安全起见对分母做极小量保护。
    """
    lam2 = np.asarray(lam_um, dtype=float)**2
    denom = np.maximum(lam2 - 0.0086024, 1e-9)
    n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
    n = np.sqrt(np.maximum(n2, 1e-12))
    return n

def n_const_um(lam_um: np.ndarray, n0: float = 2.60) -> np.ndarray:
    """常数折射率示例（可用于 SiC 的简化近似或占位）"""
    return np.full_like(lam_um, float(n0), dtype=float)

# 你已有 SiC 的 n(λ) 函数时，可以写成：
def n_sic_um(lam_um: np.ndarray) -> np.ndarray:
    """
    4H-SiC 折射率公式（普通方向近似）:
    n(λ[nm]) = 2.5610 + 3.4e4 / λ^2
    输入 lam_um: λ[μm]
    """
    lam_nm = np.asarray(lam_um, dtype=float) * 1000.0
    n_val = 2.5610 + 3.4e4 / (np.maximum(lam_nm**2, 1e-9))
    return n_val



# =============================
# 使用示例（把 df3/df4 传入）
# =============================

if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 配置：SiC (附件1/2)，Si (附件3/4)
    cfg1 = DatasetCfg(name="SiC 样品-附件1 (10°)", n_of_lambda_um=n_sic_um, n0_env=1.0, theta_i_deg=10.0, nu_min=1200,
                      nu_max=3800)
    cfg2 = DatasetCfg(name="SiC 样品-附件2 (15°)", n_of_lambda_um=n_sic_um, n0_env=1.0, theta_i_deg=15.0, nu_min=1200,
                      nu_max=3800)
    cfg3 = DatasetCfg(name="Si 样品-附件3", n_of_lambda_um=n_si_um, n0_env=1.0, theta_i_deg=0.0, nu_min=1200,
                      nu_max=3800)
    cfg4 = DatasetCfg(name="Si 样品-附件4", n_of_lambda_um=n_si_um, n0_env=1.0, theta_i_deg=0.0, nu_min=1200,
                      nu_max=3800)

    analyzer = RandFAnalyzer(PreprocessCfg(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_for_fft=True))
    results = analyzer.analyze_many({
        "df1": (df1, cfg1),
        "df2": (df2, cfg2),
        "df3": (df3, cfg3),
        "df4": (df4, cfg4),
    }, make_plot=True)

    for k, r in results.items():
        print(f"\n=== {r['name']} ===")
        print(f"  R(λ) 统计: {r['R_stats']}")
        print(f"  F(λ) 统计: {r['F_stats']}")
        print(f"  F_eff(中位) = {r['F_eff']:.2f}")
        print(f"  FFT f_peak = {r['f_peak']:.6f} → FSR = {r['FSR_cm1']:.3f} cm^-1；主峰清晰度 = {r['clarity_fft']:.2f}")
        print(
            f"  可见度 = {r['visibility']:.4f}；平均峰距 = {r['mean_period_cm1']:.3f}；平均FWHM = {r['mean_fwhm_cm1']:.3f}；尖锐比 = {r['sharp_ratio']:.2f}")

    plt.show()
