# -*- coding: utf-8 -*-
"""
多光束（Fabry–Pérot / TMM）模型 —— 两角度联合拟合硅外延层厚度（n(λ) 可调用版本）
- 物理模型：单层薄膜的传输矩阵法（s/p & 未偏振）
- 折射率：通过 Callable 传入 N1(λ)、N2(λ)，λ 单位 μm，返回复数 n+iκ，逐波长计算
- 观测模型：Y_j(ν) ≈ α_j + β_j * R_j(ν) ，两角共享 d，(α_j,β_j) 各角独立
- 拟合：两角联合非线性最小二乘（scipy.optimize.least_squares）
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from Problem3.IndicatorCalculator import PreprocessConfig, SpectrumPreprocessor


# 复用你在问题3里已有的预处理器



# ================== TMM 基元：单层反射率（支持 s/p） ==================

def _eta(pol: str, n: complex | np.ndarray, cos_theta: float | np.ndarray) -> np.ndarray:
    """相对波阻抗：s: η=n cosθ；p: η=n / cosθ"""
    cos_theta = np.asarray(cos_theta)
    if pol == "s":
        return np.asarray(n) * cos_theta
    else:  # p
        safe_cos = np.where(cos_theta > 1e-12, cos_theta, 1e-12)
        return np.asarray(n) / safe_cos

def _tmm_one_layer_R(
    nu_cm1: np.ndarray,
    N0: complex | float,
    N1_lambda: np.ndarray,   # 逐点 N1(λ)（复数）
    N2_lambda: np.ndarray,   # 逐点 N2(λ)（复数或实数）
    d_cm: float,
    theta0_deg: float,
    pol: str
) -> np.ndarray:
    """
    单层薄膜反射率 R(ν) —— 传输矩阵法
    - 允许 N1, N2 随 λ 变化（复数），内部逐频点计算
    - 入参 ν 为等间隔波数（cm^-1），λ[μm] = 1e4/ν
    """
    nu = np.asarray(nu_cm1, dtype=float)               # (N,)
    lam_um = (1.0 / np.maximum(nu, 1e-12)) * 1.0e4     # μm
    N1 = np.asarray(N1_lambda)                         # (N,)
    N2 = np.asarray(N2_lambda)                         # (N,) 或标量广播

    # 入射角
    t0 = math.radians(theta0_deg)
    cos0 = math.cos(t0)
    sin0 = math.sin(t0)

    # 逐点膜内角（用 Re(N) 做几何；虚部进入相位与振幅）
    n1_re = np.real(N1)
    n2_re = np.real(N2)
    s1 = (np.real(N0) / np.maximum(n1_re, 1e-9)) * sin0
    s1 = np.clip(s1, -1.0, 1.0)
    theta1 = np.arcsin(s1)
    cos1 = np.cos(theta1)

    s2 = (np.real(N0) / np.maximum(n2_re, 1e-9)) * sin0
    s2 = np.clip(s2, -1.0, 1.0)
    theta2 = np.arcsin(s2)
    cos2 = np.cos(theta2)

    # 阻抗
    eta0 = _eta(pol, N0 + 0j, cos0)       # 标量
    eta1 = _eta(pol, N1,    cos1)         # (N,)
    eta2 = _eta(pol, N2,    cos2)         # (N,)

    # 相位厚度 δ = 2π * N1 * d * cosθ1 * ν
    delta = 2.0 * math.pi * N1 * d_cm * cos1 * nu   # (N,), 复数
    cosd = np.cos(delta)
    sind = np.sin(delta)

    # 单层特征矩阵 M
    M11 = cosd
    M12 = 1j * (1.0 / eta1) * sind
    M21 = 1j * eta1 * sind
    M22 = cosd

    # r = ((M11+M12*η2)η0 - (M21+M22*η2)) / ((M11+M12*η2)η0 + (M21+M22*η2))
    num = (M11 + M12 * eta2) * eta0 - (M21 + M22 * eta2)
    den = (M11 + M12 * eta2) * eta0 + (M21 + M22 * eta2)
    r = num / den
    R = np.abs(r) ** 2
    return np.real(R)


# ================== 配置对象：把 n(λ) 作为 Callable 传入 ==================

@dataclass
class DispersionCallable:
    """
    以 Callable 形式提供色散：
      N1_func(λ_um) -> N1(λ) 复数；  N2_func(λ_um) -> N2(λ) 复数/实数
    你可以在外部用硅的公式/SiC 公式自行构造这两个函数。
    """
    N1_func: Callable[[np.ndarray], np.ndarray]
    N2_func: Callable[[np.ndarray], np.ndarray]

@dataclass
class TMMFitConfig:
    # 介质与色散
    N0: float = 1.0
    dispersion: DispersionCallable = None  # 必填：N1_func/N2_func
    # 角度
    theta_deg_1: float = 10.0
    theta_deg_2: float = 15.0
    # 预处理
    preprocess: PreprocessConfig = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=False)
    fit_signal: str = "grid"   # "grid" / "detrended" / "proc"
    # 厚度边界（cm）
    d_bounds_cm: Tuple[float, float] = (1e-6, 2e-3)  # 0.01 μm ~ 2000 μm（按需收紧）
    # 观测外包络参数边界
    alpha_bounds: Tuple[float, float] = (-0.5, 1.5)
    beta_bounds:  Tuple[float, float] = (0.0, 5.0)
    # 偏振
    polarization: str = "unpolarized"


# ================== 主类：两角联合拟合（n(λ) 动态） ==================

class MultiBeamThicknessFitterCallableN:
    """
    与你原来的类完全等价的“方法学”：
      - 物理：TMM 单层多光束
      - 观测：Y = α + β R（每角独立 α/β）
      - 估计：两角联合 least_squares（带边界）
    只是把 n1 常数/多项式，改成了“外部传入 n(λ) 的 Callable”，逐频点计算。
    """
    def __init__(self, cfg: Optional[TMMFitConfig] = None):
        if cfg is None or cfg.dispersion is None:
            raise ValueError("请在 TMMFitConfig.dispersion 里提供 N1_func/N2_func。")
        self.cfg = cfg
        self.pre = SpectrumPreprocessor(self.cfg.preprocess)

    # ---------- 观测模型 ----------
    @staticmethod
    def _obs_map(R: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha + beta * R

    # ---------- 预处理 ----------
    def _prepare(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        prep1 = self.pre.run(df1); prep2 = self.pre.run(df2)
        key = {"grid": "R_grid", "detrended": "R_detrended", "proc": "R_proc"}.get(self.cfg.fit_signal, "R_grid")
        nu1, y1 = prep1["nu_grid"], prep1[key]
        nu2, y2 = prep2["nu_grid"], prep2[key]
        lam1_um = (1.0 / np.maximum(nu1, 1e-12)) * 1.0e4
        lam2_um = (1.0 / np.maximum(nu2, 1e-12)) * 1.0e4
        return dict(nu1=nu1, y1=y1, lam1_um=lam1_um, nu2=nu2, y2=y2, lam2_um=lam2_um)

    # ---------- 初值与边界 ----------
    def _init_and_bounds(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        names: List[str] = []
        p0: List[float] = []
        lb: List[float] = []
        ub: List[float] = []

        # 1) 厚度初值（用 FSR 粗估）
        def _delta_nu_fft(nu: np.ndarray, y: np.ndarray) -> Optional[float]:
            if nu.size < 16: return None
            dnu = float(np.mean(np.diff(nu)))
            if not np.isfinite(dnu) or dnu <= 0: return None
            y0 = y - np.mean(y)
            Y = np.fft.rfft(y0); freqs = np.fft.rfftfreq(nu.size, d=dnu)
            span = float(nu[-1]-nu[0]); low_cut = 3.0 / (span + 1e-12)
            m = freqs > low_cut
            if not np.any(m): return None
            ap = np.abs(Y)[m]; f1 = float(freqs[m][np.argmax(ap)])
            return 1.0/f1 if f1>0 else None

        dnu1 = _delta_nu_fft(data["nu1"], data["y1"])
        dnu2 = _delta_nu_fft(data["nu2"], data["y2"])
        # 估一个平均 n1 与 cosθ1（取谱段中值）
        lam_all = np.concatenate([data["lam1_um"], data["lam2_um"]], axis=0)
        n1_mid = float(np.median(np.real(self.cfg.dispersion.N1_func(lam_all))))
        def _theta_t(theta_i_deg: float, n_in: float, n0: float = 1.0):
            s = (n0/n_in) * math.sin(math.radians(theta_i_deg))
            s = max(-1.0, min(1.0, s)); return math.asin(s)
        th1 = _theta_t(self.cfg.theta_deg_1, n1_mid)
        th2 = _theta_t(self.cfg.theta_deg_2, n1_mid)

        def _d_from_delta(delta_nu: Optional[float], cos_th: float) -> Optional[float]:
            if delta_nu is None or delta_nu <= 0: return None
            denom = 2.0 * n1_mid * cos_th * delta_nu
            return 1.0/denom if denom>0 else None

        d0_list = []
        for dn, th in ((dnu1, math.cos(th1)), (dnu2, math.cos(th2))):
            d_est = _d_from_delta(dn, th)
            if d_est is not None and np.isfinite(d_est):
                d0_list.append(d_est)
        d0 = float(np.median(d0_list)) if d0_list else 5e-4  # 默认 5 μm

        p0.append(d0); lb.append(self.cfg.d_bounds_cm[0]); ub.append(self.cfg.d_bounds_cm[1]); names.append("d_cm")

        # 2) 每角 α/β
        if self.cfg.fit_signal == "grid":
            alpha_bounds, beta_bounds = (-0.5, 1.5), (0.0, 5.0)
        else:
            alpha_bounds, beta_bounds = (-0.3, 0.3), (0.1, 3.0)

        def _ab_init(y: np.ndarray) -> Tuple[float,float]:
            a0 = float(np.median(y))
            iqr = float(np.quantile(y, 0.75)-np.quantile(y, 0.25))
            b0 = iqr if iqr>1e-6 else 1.0
            a0 = min(max(a0, alpha_bounds[0]+1e-6), alpha_bounds[1]-1e-6)
            b0 = min(max(b0, beta_bounds[0]+1e-6), beta_bounds[1]-1e-6)
            return a0,b0

        for y in (data["y1"], data["y2"]):
            a0,b0=_ab_init(y); p0.extend([a0,b0])
            lb.extend([alpha_bounds[0], beta_bounds[0]])
            ub.extend([alpha_bounds[1], beta_bounds[1]])
            names.extend([f"alpha{1 if y is data['y1'] else 2}", f"beta{1 if y is data['y1'] else 2}"])

        p0, lb, ub = np.asarray(p0,float), np.asarray(lb,float), np.asarray(ub,float)
        eps=1e-12; p0=np.minimum(np.maximum(p0,lb+eps),ub-eps)
        return p0, lb, ub, names

    # ---------- 残差 ----------
    def _residuals(self, p: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        pol = self.cfg.polarization.lower()
        N0  = self.cfg.N0
        d_cm, alpha1, beta1, alpha2, beta2 = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])

        nu1, y1, lam1 = data["nu1"], data["y1"], data["lam1_um"]
        nu2, y2, lam2 = data["nu2"], data["y2"], data["lam2_um"]

        N1_1 = self.cfg.dispersion.N1_func(lam1)  # 逐点复数
        N2_1 = self.cfg.dispersion.N2_func(lam1)
        N1_2 = self.cfg.dispersion.N1_func(lam2)
        N2_2 = self.cfg.dispersion.N2_func(lam2)

        def R_for(nu, N1, N2, theta_deg):
            if pol == "s":
                return _tmm_one_layer_R(nu, N0, N1, N2, d_cm, theta_deg, "s")
            elif pol == "p":
                return _tmm_one_layer_R(nu, N0, N1, N2, d_cm, theta_deg, "p")
            else:
                Rs = _tmm_one_layer_R(nu, N0, N1, N2, d_cm, theta_deg, "s")
                Rp = _tmm_one_layer_R(nu, N0, N1, N2, d_cm, theta_deg, "p")
                return 0.5*(Rs+Rp)

        R1 = R_for(nu1, N1_1, N2_1, self.cfg.theta_deg_1)
        R2 = R_for(nu2, N1_2, N2_2, self.cfg.theta_deg_2)

        y1_hat = self._obs_map(R1, alpha1, beta1)
        y2_hat = self._obs_map(R2, alpha2, beta2)
        return np.concatenate([y1_hat - y1, y2_hat - y2], axis=0)

    # ---------- 拟合入口 ----------
    def fit_pair(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        data = self._prepare(df1, df2)
        p0, lb, ub, names = self._init_and_bounds(data)
        sol = least_squares(self._residuals, p0, bounds=(lb,ub), args=(data,), method="trf", max_nfev=60000)
        p_opt = sol.x

        res = self._residuals(p_opt, data)
        n1 = len(data["y1"]); n2 = len(data["y2"])
        rms_joint = float(np.sqrt(np.mean(res**2)))
        rms_1 = float(np.sqrt(np.mean(res[:n1]**2)))
        rms_2 = float(np.sqrt(np.mean(res[n1:]**2)))

        out = dict(
            success=bool(sol.success), message=str(sol.message), nfev=int(sol.nfev),
            param_names=names, param_values={names[i]: float(p_opt[i]) for i in range(len(names))},
            d_cm=float(p_opt[0]), d_um=float(p_opt[0]*1e4),
            residual_rms=dict(rms_joint=rms_joint, rms_angle1=rms_1, rms_angle2=rms_2)
        )
        return out


# ================== 示例：传入硅的 n(λ)（可选加入 Δn） ==================

def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    """
    硅（Si）的折射率： n^2 = 11.6858 + 0.939816/(λ^2-0.0086024) + 0.00089814*λ^2 ，λ[μm]
    返回实数 n（无吸收），如需吸收可改为 n + 1j*k(λ)
    """
    lam2 = np.asarray(lam_um, float)**2
    denom = np.maximum(lam2 - 0.0086024, 1e-9)
    n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
    n = np.sqrt(np.maximum(n2, 1e-12))
    return n + 0j

def make_N1_N2_funcs(delta_n: float = 0.0) -> DispersionCallable:
    """
    例子：外延层与衬底使用同一色散，但外延层相对衬底有一个小的 Δn（掺杂差导致）。
    你也可以把 N2_func 单独写成另一条文献曲线。
    """
    def N2_func(lam_um: np.ndarray) -> np.ndarray:
        return n_si_um(lam_um)            # 衬底
    def N1_func(lam_um: np.ndarray) -> np.ndarray:
        return (np.real(n_si_um(lam_um)) + delta_n) + 0j
    return DispersionCallable(N1_func=N1_func, N2_func=N2_func)


# ================== main 示例（两角联合） ==================

if __name__ == "__main__":
    # 你自己的数据读取部分
    from Data.DataManager import DM
    df3 = DM.get_data(3)   # Si @ 10°
    df4 = DM.get_data(4)   # Si @ 15°

    # 构造 n(λ) 的 Callable（这里演示 Δn 可设为 0 或小量）
    disp = make_N1_N2_funcs(delta_n=0.0)

    cfg = TMMFitConfig(
        N0=1.0,
        dispersion=disp,
        theta_deg_1=10.0, theta_deg_2=15.0,
        preprocess=PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=False),
        fit_signal="grid",
        d_bounds_cm=(1e-4, 3e-2),      # 可按 FFT 初值收紧
        alpha_bounds=(-1.5, 1.5),
        beta_bounds=(0.0, 5.0),
        polarization="unpolarized"
    )

    fitter = MultiBeamThicknessFitterCallableN(cfg)
    result = fitter.fit_pair(df3, df4)

    from pprint import pprint
    pprint(result)
    print(f"Thickness d = {result['d_um']:.4f} μm; RMS(joint) = {result['residual_rms']['rms_joint']:.4g}")
