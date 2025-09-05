# -*- coding: utf-8 -*-
"""
UpgradedTwoBeam.py
两束干涉（仅一次反/透射）升级版：
- 允许弱色散 n(ν)：Cauchy 到 1/λ²
- 支持双角联合拟合（共享 d 和 n(·) 参数；各角独立 A,B,φ）
- FFT 用于初值与质控；相位拟合给最终厚度
- 可靠性：RMSE、主峰清晰度、跨角一致性、自举置信区间（可选）
- 可视化函数（matplotlib）

依赖：
  numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# =========================================================
# 0) 通用工具 & 物理函数
# =========================================================

EPS = 1e-12

def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    """λ[μm] = 1e4 / ν[cm^-1]"""
    return 1e4 / np.maximum(nu_cm1, EPS)

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

def cos_theta_t(n: np.ndarray | float, theta_i_rad: float) -> np.ndarray:
    """cosθ_t = sqrt(1 - (sinθ_i / n)^2)；自动广播 n 的形状"""
    s = math.sin(theta_i_rad)
    n = np.asarray(n, dtype=float)
    val = 1.0 - (s / np.maximum(n, EPS)) ** 2
    return np.sqrt(np.clip(val, 0.0, 1.0))

def phase_two_beam(nu_cm1: np.ndarray, n_nu: np.ndarray, d_um: float,
                   theta_i_rad: float, phi0: float) -> np.ndarray:
    """
    Φ(ν) = 4π n(ν) d_cm cosθ_t(ν) ν + φ0
    d_cm = d_um * 1e-4
    """
    d_cm = d_um * 1e-4
    ct = cos_theta_t(n_nu, theta_i_rad)
    return 4.0 * np.pi * n_nu * d_cm * ct * nu_cm1 + phi0

def model_reflectance_two_beam(nu_cm1: np.ndarray, n_nu: np.ndarray, d_um: float,
                               theta_i_rad: float, A: float, B: float, phi0: float) -> np.ndarray:
    """R(ν) = A + B cos(Φ(ν))"""
    ph = phase_two_beam(nu_cm1, n_nu, d_um, theta_i_rad, phi0)
    return A + B * np.cos(ph)


# =========================================================
# 1) 预处理（等间距 + 温和去趋势 + 可选归一化）
# =========================================================

@dataclass
class PreprocessConfig:
    detrend: bool = True
    sg_window_frac: float = 0.15  # 相对样本数；(0,1)
    sg_polyorder: int = 2
    normalize: bool = True        # 仅用于 FFT；拟合建议 False（或用 detrended）

class SpectrumPreprocessor:
    """
    将 (ν[cm^-1], R[%]) → 等间隔 ν 栅格，并进行温和去趋势。
    """
    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg: PreprocessConfig = cfg or PreprocessConfig()

    def _to_uniform_grid(self, nu: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(nu)
        nu = nu[idx]
        R = R[idx]

        dnu = np.mean(np.diff(nu))
        if not np.isfinite(dnu) or dnu <= 0:
            return nu, R

        nu_grid = np.arange(nu[0], nu[-1] + 0.5 * dnu, dnu)
        R_grid = np.interp(nu_grid, nu, R)
        return nu_grid, R_grid

    def _detrend(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.detrend:
            return y.copy()
        n = len(y)
        if n < 11:
            return y - float(np.nanmean(y))
        w = max(5, int(round(self.cfg.sg_window_frac * n)))
        if w % 2 == 0:
            w += 1
        # 保证 window_length 合法且 > polyorder
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        w = max(w, self.cfg.sg_polyorder + 3)
        baseline = savgol_filter(y, window_length=w, polyorder=self.cfg.sg_polyorder, mode='interp')
        return y - baseline

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return y
        y0 = y - float(np.mean(y))
        rms = math.sqrt(float(np.mean(y0 ** 2))) if len(y0) > 0 else 0.0
        return y0 / (rms + EPS) if rms > 0 else y0

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 列名容错
        col_nu = "波数 (cm-1)"
        col_R = "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            col_nu, col_R = df.columns[:2]

        nu = df[col_nu].to_numpy(dtype=float)
        R_percent = df[col_R].to_numpy(dtype=float)
        R = R_percent / 100.0

        nu_grid, R_grid = self._to_uniform_grid(nu, R)
        R_detrended = self._detrend(R_grid)
        R_proc = self._normalize(R_detrended)

        return dict(
            nu_grid=nu_grid,
            R_grid=R_grid,
            R_detrended=R_detrended,
            R_proc=R_proc,
        )


# =========================================================
# 2) 折射率模型（弱色散）
# =========================================================

@dataclass
class NModel:
    """
    折射率模型：
      - constant: n(ν) = n0
      - cauchy  : n(ν) = n0 + B / λ(ν)^2    (λ in μm)
    """
    kind: str = "cauchy"
    params: np.ndarray = np.array([2.65, 0.0], dtype=float)  # [n0, B] for cauchy; [n0] for constant

    def n_of(self, nu_cm1: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            n0 = float(self.params[0])
            return np.full_like(nu_cm1, n0, dtype=float)
        elif self.kind == "cauchy":
            n0, B = [float(x) for x in self.params]
            lam_um = wavelength_um_from_wavenumber_cm1(nu_cm1)
            return n0 + B / (lam_um ** 2 + EPS)
        else:
            raise ValueError(f"Unknown n-model kind: {self.kind}")

    def param_names(self) -> List[str]:
        return ["n0"] if self.kind == "constant" else ["n0", "B"]


# =========================================================
# 3) FFT 初值与质控
# =========================================================

class FFTPrimaryFreq:
    """
    估计波数域主频 f_peak（cycles per cm^-1），并给出“主峰清晰度”。
    """
    def __init__(self, low_cycle_count_threshold: float = 3.0):
        self.low_cycle_count_threshold = low_cycle_count_threshold

    def estimate(self, nu_grid: np.ndarray, y_proc: np.ndarray) -> Dict[str, float]:
        if len(nu_grid) < 8:
            raise ValueError("样本点过少，无法FFT。")
        dnu = float(np.mean(np.diff(nu_grid)))
        if dnu <= 0:
            raise ValueError("波数轴应为严格升序且等间距。")

        Y = rfft(y_proc - float(np.mean(y_proc)))
        freqs = rfftfreq(len(y_proc), d=dnu)  # cycles per (cm^-1)

        # 低频截止：至少包含 ~N 个条纹
        span = float(nu_grid[-1] - nu_grid[0] + EPS)
        low_cut = (1.0 / span) * self.low_cycle_count_threshold
        mask = freqs > low_cut
        if not np.any(mask):
            raise ValueError("频率范围不足以分辨条纹。")

        amp = np.abs(Y)[mask]
        fpos = freqs[mask]

        # 主峰
        k = int(np.argmax(amp))
        f_peak = float(fpos[k])
        main_amp = float(amp[k])
        # 主峰清晰度：主峰幅度 / (次峰幅度 + eps)
        amp_sorted = np.sort(amp)
        second = float(amp_sorted[-2]) if len(amp_sorted) >= 2 else 0.0
        clarity = main_amp / (second + EPS)

        return dict(f_peak=f_peak, delta_nu=1.0 / f_peak, clarity=clarity)


# =========================================================
# 4) 双角联合拟合（共享 d 与 n 参数；各角独立 A,B,φ）
# =========================================================

@dataclass
class JointFitConfig:
    # 拟合用信号 & 频段
    use_signal: str = "detrended"
    fit_range_cm1: Optional[Tuple[float, float]] = (1200.0, 3800.0)

    # 最小二乘
    max_nfev: int = 25000
    loss: str = "cauchy"
    f_scale: Optional[float] = "auto"

    # ---- 折射率参数的物理约束 / 先验 ----
    # n0 的硬边界（物理可接受范围）
    n0_bounds: Tuple[float, float] = (2.45, 2.80)
    # n0 的高斯先验中心与强度（惩罚项系数；0=不用）
    n0_prior: float = 2.60
    n0_prior_strength: float = 0.5   # ~0.1-1.0 合理

    # Cauchy 弱色散 B 的边界与先验（一般很小）
    B_bounds: Tuple[float, float] = (-5e-3, 5e-3)
    B_prior: float = 0.0
    B_prior_strength: float = 0.2

    # 是否直接固定（强行设定）某些参数
    fix_n0: Optional[float] = None   # 例如 2.60；None 表示不固定
    fix_B: Optional[float] = None    # 例如 0.0；None 表示不固定


class JointTwoBeamFitter:
    def __init__(self,
                 n_model: NModel,
                 theta_list_deg: List[float],
                 pre_cfg: Optional[PreprocessConfig] = None,
                 fit_cfg: Optional[JointFitConfig] = None):
        """
        n_model       : 折射率模型（建议 'cauchy'）
        theta_list_deg: [10.0, 15.0] 等入射角列表（>=1 个）
        pre_cfg       : 预处理参数
        fit_cfg       : 拟合参数
        """
        assert len(theta_list_deg) >= 1
        self.n_model = n_model
        self.theta_list_rad = [math.radians(x) for x in theta_list_deg]
        self.pre = SpectrumPreprocessor(pre_cfg or PreprocessConfig())
        self.fit_cfg = fit_cfg or JointFitConfig()
        self.fft = FFTPrimaryFreq()

    # ---------- 预处理多个角 ----------
    def _prep_many(self, dfs: List[pd.DataFrame]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        预处理每个角的数据，并按照 fit_range_cm1 做波数区间掩膜。
        返回：
          nu_list    : 每角用于拟合的 ν（掩膜后）
          y_list     : 每角用于拟合的信号（掩膜后）
          aux        : 额外信息（含原始预处理结果、用于 FFT 的标准化信号等）
        """
        assert len(dfs) == len(self.theta_list_rad)
        prepped = [self.pre.run(df) for df in dfs]

        # 选择用于拟合的信号
        sel = self.fit_cfg.use_signal
        if sel == "grid":
            y_all = [d["R_grid"] for d in prepped]
        elif sel == "detrended":
            y_all = [d["R_detrended"] for d in prepped]
        elif sel == "proc":
            y_all = [d["R_proc"] for d in prepped]
        else:
            raise ValueError("use_signal 应为 'grid'/'detrended'/'proc'")

        # 应用频段掩膜
        nu_list, y_list, fft_list = [], [], []
        for d in prepped:
            nu0 = d["nu_grid"]
            # 用于 FFT 的标准化信号（与拟合无关，但也做同样掩膜）
            y_fft0 = d["R_proc"]

            if self.fit_cfg.fit_range_cm1 is not None:
                lo, hi = self.fit_cfg.fit_range_cm1
                m = (nu0 >= lo) & (nu0 <= hi)
            else:
                m = np.ones_like(nu0, dtype=bool)

            nu_list.append(nu0[m])
            fft_list.append(y_fft0[m])

        # 与 nu_list 对齐提取 y
        for y0, nu0, d in zip(y_all, [p["nu_grid"] for p in prepped], prepped):
            if self.fit_cfg.fit_range_cm1 is not None:
                lo, hi = self.fit_cfg.fit_range_cm1
                m = (nu0 >= lo) & (nu0 <= hi)
            else:
                m = np.ones_like(nu0, dtype=bool)
            y_list.append(y0[m])

        return nu_list, y_list, dict(prepped=prepped, fft_signals=fft_list)

    # ---------- FFT 初值（平均） ----------
    def _initial_d_um_from_fft(self, nu_list: List[np.ndarray], prepped: List[Dict[str, Any]],
                               n_eff: float, fft_signals: List[np.ndarray]) -> Tuple[float, List[float]]:
        """
        依据掩膜后的数据估计 FFT 主频，给出 d 的初值。
        """
        fpeaks = []
        for nu, y_fft in zip(nu_list, fft_signals):
            est = self.fft.estimate(nu, y_fft)
            fpeaks.append(est["f_peak"])
        fpk_mean = float(np.mean(fpeaks))
        ct_mean = float(np.mean([cos_theta_t(n_eff, th) for th in self.theta_list_rad]))
        d0_um = fpk_mean / (2.0 * n_eff * max(ct_mean, EPS)) * 1e4
        return d0_um, fpeaks

    # ---------- 残差（联合） ----------
    def _residual_joint(self, params: np.ndarray, nu_list: List[np.ndarray], y_list: List[np.ndarray]) -> np.ndarray:
        """
        参数布局：
          - constant: [d_um, n0, phi1,a01,a11, phi2,a02,a12, ...]
          - cauchy  : [d_um, n0, B, phi1,a01,a11, phi2,a02,a12, ...]
        其中 a0k,a1k 是第 k 个角的 A、B（幅度）
        """
        kind = self.n_model.kind
        K = len(self.theta_list_rad)
        offset = 0

        d_um = float(params[offset]); offset += 1
        n0 = float(params[offset]); offset += 1

        if kind == "cauchy":
            B = float(params[offset]); offset += 1
            n_params = np.array([n0, B], dtype=float)
        else:
            n_params = np.array([n0], dtype=float)
            B = 0.0  # 为先验打印兼容

        res_all = []
        for k in range(K):
            phi = float(params[offset]); a0 = float(params[offset+1]); a1 = float(params[offset+2])
            offset += 3
            nu = nu_list[k]
            y = y_list[k]
            # 计算 n(ν)
            nm = NModel(kind=kind, params=n_params)
            n_nu = nm.n_of(nu)
            # 预测
            y_hat = model_reflectance_two_beam(nu, n_nu, d_um, self.theta_list_rad[k], a0, a1, phi)
            res_all.append(y_hat - y)

        # ---- 先验惩罚（正则） ----
        pri = []
        # n0 先验
        if self.n_model.kind in ("constant", "cauchy"):
            if self.fit_cfg.n0_prior_strength and self.fit_cfg.n0_prior_strength > 0:
                pri.append(math.sqrt(self.fit_cfg.n0_prior_strength) * (n0 - self.fit_cfg.n0_prior))
        # B 先验
        if self.n_model.kind == "cauchy":
            if self.fit_cfg.B_prior_strength and self.fit_cfg.B_prior_strength > 0:
                pri.append(math.sqrt(self.fit_cfg.B_prior_strength) * (B - self.fit_cfg.B_prior))

        if pri:
            res_all.append(np.array(pri, dtype=float))

        return np.concatenate(res_all)

    # ---------- 拟合主流程 ----------
    def fit(self, dfs: List[pd.DataFrame], n0_init: float = 2.65, B_init: float = 0.0) -> Dict[str, Any]:
        """
        返回字典：
          - d_um, n0, (B)
          - per_angle: {k: {phi, A, B}}
          - rmse, fpeaks, clarity_list
          - diagnostics: {单角 d, ...}
        """
        assert len(dfs) == len(self.theta_list_rad)
        # 预处理
        nu_list, y_list, aux = self._prep_many(dfs)
        prepped = aux["prepped"]
        fft_signals = aux["fft_signals"]

        # FFT 初值
        d0_um, fpeaks = self._initial_d_um_from_fft(nu_list, prepped, n_eff=n0_init, fft_signals=fft_signals)

        clarity_list = []
        for item in prepped:
            est = self.fft.estimate(item["nu_grid"], item["R_proc"])
            clarity_list.append(float(est["clarity"]))

        # ---------- A,B,phi 初值 per-angle（按当前 y_list） ----------
        K = len(self.theta_list_rad)
        per_angle_inits = []
        for y in y_list:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            span = max(y_max - y_min, 1e-3)
            A0 = float(np.mean(y))
            B0 = 0.5 * span
            phi0 = 0.0
            per_angle_inits.append((phi0, A0, B0))

        # ---------- 统一构造 p0 与 bounds ----------
        p_elems: List[float] = []
        lo_elems: List[float] = []
        hi_elems: List[float] = []

        # d 参数
        d_lo, d_hi = 1e-6, 1e6
        p_elems.append(d0_um)
        lo_elems.append(d_lo)
        hi_elems.append(d_hi)

        # n0 参数（可固定或给边界）
        if self.fit_cfg.fix_n0 is not None:
            n0_init = float(self.fit_cfg.fix_n0)
            n0_lo = n0_hi = n0_init
        else:
            n0_lo, n0_hi = self.fit_cfg.n0_bounds

        p_elems.append(float(n0_init))
        lo_elems.append(float(n0_lo))
        hi_elems.append(float(n0_hi))

        EPS_FIX = 1e-9  # 用于“固定参数”的极小宽度，确保 lb < ub

        # B 参数（仅 cauchy；可固定或给边界）
        if self.n_model.kind == "cauchy":
            if self.fit_cfg.fix_B is not None:
                B_init = float(self.fit_cfg.fix_B)
                # 用一个极小宽度“夹住”以满足 least_squares: lb < ub
                B_lo = B_init - EPS_FIX
                B_hi = B_init + EPS_FIX
            else:
                B_lo, B_hi = self.fit_cfg.B_bounds
                # 兜底，防止外部给成相等或反序
                if not (B_lo < B_hi):
                    B_hi = B_lo + 1e-9
            p_elems.append(float(B_init))
            lo_elems.append(float(B_lo))
            hi_elems.append(float(B_hi))

        # 每个角的 (phi, A, B) 及其边界
        for (nu, y, (phi0, A0, B0)) in zip(nu_list, y_list, per_angle_inits):
            y_min, y_max = float(np.min(y)), float(np.max(y))
            span = max(y_max - y_min, 1e-3)

            # phi
            p_elems.append(phi0)
            lo_elems.append(-2.0 * math.pi)
            hi_elems.append( 2.0 * math.pi)

            # A
            p_elems.append(A0)
            lo_elems.append(y_min - 2.0 * span)
            hi_elems.append(y_max + 2.0 * span)

            # B(幅度)
            p_elems.append(B0)
            lo_elems.append(0.0)
            hi_elems.append(3.0 * span)

        # 转为 numpy 并做一致性检查 + 拉平
        p0 = np.asarray(p_elems, dtype=np.float64).ravel()
        lb = np.asarray(lo_elems, dtype=np.float64).ravel()
        ub = np.asarray(hi_elems, dtype=np.float64).ravel()

        # ---- 强制修正：保证所有 lb < ub ----
        for i in range(len(p0)):
            if not (lb[i] < ub[i]):
                center = float(p0[i])
                lb[i] = center - 1e-9
                ub[i] = center + 1e-9

        assert lb.shape == p0.shape and ub.shape == p0.shape, \
            f"bounds shape {lb.shape}/{ub.shape} != p0 {p0.shape}"
        bounds = (lb, ub)

        # —— 鲁棒尺度：'auto' 则用 MAD 自适应 ——
        if self.fit_cfg.f_scale == "auto":
            mads = [max(robust_mad(y), 1e-6) for y in y_list]
            f_scale_to_use = float(np.median(mads)) if len(mads) else 1.0
        else:
            f_scale_to_use = float(self.fit_cfg.f_scale) if self.fit_cfg.f_scale is not None else 1.0
            f_scale_to_use = max(f_scale_to_use, 1e-6)

        # ---------- 最小二乘 ----------
        res = least_squares(
            self._residual_joint,
            p0,
            args=(nu_list, y_list),
            bounds=bounds,
            max_nfev=self.fit_cfg.max_nfev,
            loss=self.fit_cfg.loss,
            f_scale=f_scale_to_use,
        )

        # 解析结果
        params = res.x.copy()
        offset = 0
        d_um = float(params[offset]); offset += 1
        n0 = float(params[offset]);    offset += 1
        out = dict(d_um=d_um, n0=n0)
        if self.n_model.kind == "cauchy":
            B = float(params[offset]); offset += 1
            out["B"] = B

        per_angle = {}
        for k in range(len(self.theta_list_rad)):
            phi = float(params[offset]); a0 = float(params[offset+1]); a1 = float(params[offset+2])
            offset += 3
            per_angle[f"angle_{k+1}"] = dict(phi=phi, A=a0, B=a1)

        # RMSE
        resid = self._residual_joint(res.x, nu_list, y_list)
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        out.update(dict(
            per_angle=per_angle,
            rmse=rmse,
            fpeaks=[float(x) for x in fpeaks],
            clarity_list=clarity_list,
            success=bool(res.success),
            message=str(res.message),
            nfev=int(res.nfev),
        ))

        # 单角“一致性”诊断：固定 n 参数，逐角只拟合 (d,A,B,φ)
        diag = self._single_angle_diagnostics(nu_list, y_list, out)
        out["diagnostics"] = diag
        return out

    # ---------- 单角一致性：固定 n(·)，每角独立拟合 d,A,B,φ ----------
    def _single_angle_diagnostics(self, nu_list: List[np.ndarray], y_list: List[np.ndarray], result: Dict[str, Any]) -> Dict[str, Any]:
        kind = self.n_model.kind
        if kind == "cauchy":
            n_params = np.array([result["n0"], result["B"]], dtype=float)
        else:
            n_params = np.array([result["n0"]], dtype=float)

        d_list = []
        rmse_list = []
        for k, (nu, y) in enumerate(zip(nu_list, y_list), start=1):
            # 初值：用联合的 d + 当角的 A,B,φ
            p_angle0 = result["per_angle"][f"angle_{k}"]
            d0 = float(result["d_um"])
            phi0, A0, B0 = float(p_angle0["phi"]), float(p_angle0["A"]), float(p_angle0["B"])

            def residual_single(p):
                d_um, phi, a0, a1 = p
                n_nu = NModel(kind=kind, params=n_params).n_of(nu)
                y_hat = model_reflectance_two_beam(nu, n_nu, d_um, self.theta_list_rad[k-1], a0, a1, phi)
                return y_hat - y

            # 边界
            y_min, y_max = float(np.min(y)), float(np.max(y))
            span = max(y_max - y_min, 1e-3)
            lo = np.array([1e-6, -2.0*math.pi, y_min - 2.0*span, 0.0])
            hi = np.array([1e6,  2.0*math.pi, y_max + 2.0*span, 3.0*span])
            p0 = np.array([d0, phi0, A0, B0], dtype=float)

            r = least_squares(residual_single, p0, bounds=(lo, hi), max_nfev=15000)
            d_hat = float(r.x[0])
            rms = float(np.sqrt(np.mean(residual_single(r.x) ** 2)))
            d_list.append(d_hat)
            rmse_list.append(rms)

        # 跨角一致性
        if len(d_list) >= 2:
            d_mean = float(np.mean(d_list))
            spread = float(np.max(np.abs(np.array(d_list) - d_mean)))
            rel_spread = spread / (d_mean + EPS)
        else:
            d_mean = float(d_list[0])
            rel_spread = 0.0

        return dict(
            single_angle_d_um=[float(x) for x in d_list],
            single_angle_rmse=[float(x) for x in rmse_list],
            cross_angle_rel_spread=rel_spread
        )

    # ---------- 自举置信区间（可选） ----------
    def bootstrap_ci(self, dfs: List[pd.DataFrame], n_boot: int = 200, keep_ratio: float = 0.7, random_state: int = 0) -> Dict[str, Any]:
        """
        对波数轴做“连续窗口采样”的 bootstrap（block-bootstrap 近似）
        返回 d 的均值/标准差/分位数
        """
        rng = np.random.default_rng(random_state)
        nu_list, y_list, aux = self._prep_many(dfs)
        prepped = aux["prepped"]

        d_vals = []
        for b in range(n_boot):
            dfs_sub = []
            for (nu, prep) in zip(nu_list, prepped):
                # 连续窗口
                n = len(nu)
                m = max(8, int(round(n * keep_ratio)))
                start = rng.integers(low=0, high=max(n - m, 1))
                idx = slice(start, start + m)

                df_sub = pd.DataFrame({
                    "波数 (cm-1)": prep["nu_grid"][idx],
                    "反射率 (%)": (prep["R_grid"][idx] * 100.0)
                })
                dfs_sub.append(df_sub)

            try:
                res_sub = self.fit(dfs_sub)  # 递归调用
                d_vals.append(float(res_sub["d_um"]))
            except Exception:
                continue

        if len(d_vals) == 0:
            return dict(success=False, note="bootstrap 无有效样本")
        d_arr = np.array(d_vals, dtype=float)
        return dict(
            success=True,
            n=int(len(d_arr)),
            mean=float(np.mean(d_arr)),
            std=float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else 0.0,
            q05=float(np.quantile(d_arr, 0.05)),
            q50=float(np.quantile(d_arr, 0.50)),
            q95=float(np.quantile(d_arr, 0.95)),
        )

    # ---------- 可视化 ----------
    def plot_spectrum_and_fit(self, dfs: List[pd.DataFrame], fit_result: Dict[str, Any], savepath: Optional[str] = None):
        """
        绘制每个入射角的（选择的拟合信号）与拟合曲线对比。
        """
        nu_list, y_list, aux = self._prep_many(dfs)
        kind = self.n_model.kind
        if kind == "cauchy":
            n_params = np.array([fit_result["n0"], fit_result["B"]], dtype=float)
        else:
            n_params = np.array([fit_result["n0"]], dtype=float)

        K = len(self.theta_list_rad)
        fig, axes = plt.subplots(K, 1, figsize=(9, 3.2*K), sharex=True)
        if K == 1:
            axes = [axes]

        for k in range(K):
            nu = nu_list[k]; y = y_list[k]
            pa = fit_result["per_angle"][f"angle_{k+1}"]
            phi, A, B = float(pa["phi"]), float(pa["A"]), float(pa["B"])
            n_nu = NModel(kind=kind, params=n_params).n_of(nu)
            yhat = model_reflectance_two_beam(nu, n_nu, fit_result["d_um"], self.theta_list_rad[k], A, B, phi)

            ax = axes[k]
            ax.plot(nu, y, lw=1.0, label=f"实测（信号={self.fit_cfg.use_signal}）")
            ax.plot(nu, yhat, lw=1.0, linestyle="--", label="拟合")
            ax.set_ylabel("信号幅值")

            if self.fit_cfg.fit_range_cm1 is not None:
                lo, hi = self.fit_cfg.fit_range_cm1
                ax.axvspan(lo, hi, alpha=0.08)

            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("波数 (cm$^{-1}$)")
        fig.suptitle("两束模型：实测 vs 拟合", y=0.98)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=160)
        return fig

    def plot_fft(self, dfs: List[pd.DataFrame], savepath: Optional[str] = None):
        """
        绘制 FFT 幅度谱并标注主峰
        """
        prepped = [self.pre.run(df) for df in dfs]
        K = len(dfs)
        fig, axes = plt.subplots(K, 1, figsize=(9, 2.8*K), sharex=False)
        if K == 1:
            axes = [axes]

        for k in range(K):
            nu = prepped[k]["nu_grid"]
            y = prepped[k]["R_proc"]
            est = self.fft.estimate(nu, y)
            dnu = float(np.mean(np.diff(nu)))
            Y = rfft(y - float(np.mean(y)))
            freqs = rfftfreq(len(y), d=dnu)

            ax = axes[k]
            ax.plot(freqs, np.abs(Y), lw=1.0)
            ax.axvline(est["f_peak"], color="r", linestyle="--", lw=1.0, label=f"f_peak={est['f_peak']:.4g}")
            ax.set_xlim(left=0.0)
            ax.set_xlabel("频率  (cycles per cm$^{-1}$)")
            ax.set_ylabel("|FFT|")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
        fig.suptitle("FFT 幅度谱与主峰", y=0.98)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=160)
        return fig


# =========================================================
# 5) 快速入口封装（单角/双角）
# =========================================================

def run_dual_angle_fit(df1: pd.DataFrame, df2: pd.DataFrame,
                       theta1_deg: float = 10.0, theta2_deg: float = 15.0,
                       n_kind: str = "cauchy",
                       n0_init: float = 2.65, B_init: float = 0.0,
                       pre_cfg: Optional[PreprocessConfig] = None,
                       fit_signal: str = "detrended",
                       loss: str = "linear",
                       do_bootstrap: bool = False, n_boot: int = 200) -> Dict[str, Any]:
    """
    双角（推荐）的一键流程。返回结果 dict，附带可读的 summary。
    """
    pre_cfg = pre_cfg or PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2, normalize=True)
    fit_cfg = JointFitConfig(use_signal=fit_signal, loss=loss)
    fitter = JointTwoBeamFitter(NModel(kind=n_kind, params=np.array([n0_init, B_init] if n_kind=="cauchy" else [n0_init])),
                                [theta1_deg, theta2_deg], pre_cfg, fit_cfg)

    res = fitter.fit([df1, df2], n0_init=n0_init, B_init=B_init)

    # 可靠性辅助：自举
    boot = None
    if do_bootstrap:
        boot = fitter.bootstrap_ci([df1, df2], n_boot=n_boot)
        res["bootstrap"] = boot

    # 可读化摘要
    s = []
    s.append(f"厚度 d = {res['d_um']:.4f} μm")
    if n_kind == "cauchy":
        s.append(f"n(λ) = n0 + B/λ²，拟合 n0 = {res['n0']:.5f}，B = {res['B']:.5g}")
    else:
        s.append(f"常数折射率 n0 = {res['n0']:.5f}")
    s.append(f"联合拟合 RMSE = {res['rmse']:.4g}；主峰清晰度/角 = {['%.2f'%x for x in res['clarity_list']]}")
    if "diagnostics" in res:
        d_single = res["diagnostics"]["single_angle_d_um"]
        spread = res["diagnostics"]["cross_angle_rel_spread"]
        s.append(f"单角厚度 d(10°)={d_single[0]:.4f} μm, d(15°)={d_single[1]:.4f} μm；跨角相对散布={spread*100:.2f}%")
    if boot and boot.get("success"):
        s.append(f"Bootstrap({boot['n']}次)：均值={boot['mean']:.4f}, std={boot['std']:.4f}, 5%-95%区间=[{boot['q05']:.4f}, {boot['q95']:.4f}] μm")

    res["summary"] = "；".join(s)
    res["fitter"] = fitter  # 方便外部直接调用作图
    return res


# =========================================================
# 6) 示例（假设你已把附件1/2读取为 df1/df2）
# =========================================================

if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # —— 一键双角拟合（推荐） ——
    res = run_dual_angle_fit(
        df1, df2,
        theta1_deg=10.0, theta2_deg=15.0,
        n_kind="cauchy",
        n0_init=2.60, B_init=0.0,
        pre_cfg=PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2, normalize=True),
        fit_signal="detrended",
        loss="cauchy",
    )

    # 先把 n 固定住，确保 d 与 FFT 周期一致；需要时再放开
    fitter: JointTwoBeamFitter = res["fitter"]
    fitter.fit_cfg.fix_n0 = 2.60
    fitter.fit_cfg.fix_B  = 0.0
    res = fitter.fit([df1, df2], n0_init=2.60, B_init=0.0)

    print("=== 结果摘要 ===")
    pprint(res)

    # 作图（可选）
    fitter.plot_fft([df1, df2], savepath=None)
    fitter.plot_spectrum_and_fit([df1, df2], res, savepath=None)
    plt.show()
