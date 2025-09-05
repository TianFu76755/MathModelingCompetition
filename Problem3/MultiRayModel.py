# -*- coding: utf-8 -*-
"""
Si epitaxial thickness estimation via multiple-beam (Airy/TMM) model
Python 3.8

- 输入：多角反射谱（例如 df3, df4），列名 ['波数 (cm-1)', '反射率 (%)']
- 物理：空气(0)/外延层(1)/硅衬底(2) 单层膜，多光束干涉
- 模型：r_tot = (r01 + r12*exp(2iβ)) / (1 + r01*r12*exp(2iβ)), R = 0.5(|r_s|^2 + |r_p|^2)
- 参数：厚度 d（μm）；每角独立的线性标定 A_k、B_k（基线/增益）
- 初值：FFT 得 FSR → d0 ≈ 1/(2 * n̄1 * cosθ̄1 * FSR)
- 拟合：多角联合最小二乘（Cauchy 鲁棒损失），可选 Bootstrap CI
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares

import matplotlib.pyplot as plt


# =========================
# 工具函数（单位/数值）
# =========================
EPS = 1e-12

def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    """λ[μm] = 1e4 / ν[cm^-1]"""
    return 1e4 / np.maximum(nu_cm1, EPS)

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

def snell_cos_theta_t(n: np.ndarray, theta_i_rad: float, n0: float = 1.0) -> np.ndarray:
    """cosθ_t = sqrt(1 - (n0*sinθ_i/n)^2)，允许 n 为复数时用实部幅值近似"""
    s = math.sin(theta_i_rad)
    n_abs = np.maximum(np.abs(n), EPS)
    val = 1.0 - (s * n0 / n_abs) ** 2
    return np.sqrt(np.clip(val, 0.0, 1.0))

# =========================
# 折射率函数（Required）
# =========================
def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    """
    硅的色散（λ 单位 μm）：
    n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814 λ^2
    """
    lam_um = np.asarray(lam_um, dtype=float)
    lam2 = lam_um**2
    denom = np.maximum(lam2 - 0.0086024, 1e-12)
    n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
    return np.sqrt(np.maximum(n2, 1e-12))

# 若需要考虑极弱吸收，可把 n 替换为 n + 1j*kappa(λ)，本代码会自动适配（Fresnel 支持复数）


# =========================
# 预处理
# =========================
@dataclass
class PreprocessCfg:
    detrend: bool = True
    sg_window_frac: float = 0.12
    sg_polyorder: int = 2
    normalize_for_fft: bool = True
    nu_min: Optional[float] = 1200.0
    nu_max: Optional[float] = 3800.0

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

    def run(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        col_nu, col_R = "波数 (cm-1)", "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            col_nu, col_R = df.columns[:2]
        nu = df[col_nu].to_numpy(dtype=float)
        R  = (df[col_R].to_numpy(dtype=float)) / 100.0

        nu_g, R_g = self._uniform_grid(nu, R)
        nu_w, R_w = self._window(nu_g, R_g)
        y_det = self._detrend(R_w)
        y_fft = self._normalize(y_det)
        return dict(nu=nu_w, R=R_w, y_det=y_det, y_fft=y_fft)


# =========================
# FFT 初值（FSR→d0）
# =========================
class FFTInit:
    def __init__(self, min_cycles: float = 3.0):
        self.min_cycles = min_cycles

    def estimate_fsr(self, nu: np.ndarray, y_fft: np.ndarray) -> Dict[str, float]:
        dnu = float(np.mean(np.diff(nu)))
        Y = rfft(y_fft - float(np.mean(y_fft)))
        freqs = rfftfreq(len(y_fft), d=dnu)
        span = float(nu[-1] - nu[0] + EPS)
        low_cut = (1.0/span) * self.min_cycles
        m = freqs > low_cut
        amp = np.abs(Y)[m]; fpos = freqs[m]
        k = int(np.argmax(amp))
        f_peak = float(fpos[k])
        a_sorted = np.sort(amp)
        sec = float(a_sorted[-2]) if len(a_sorted) >= 2 else 0.0
        clarity = float(amp[k])/(sec+EPS)
        return dict(f_peak=f_peak, fsr=1.0/f_peak, clarity=clarity)

    def d0_from_fsr(self, nu: np.ndarray, n_of_lambda_um: Callable[[np.ndarray], np.ndarray],
                    theta_i_deg: float, fsr_cm1: float, n0_env: float = 1.0) -> float:
        lam = wavelength_um_from_wavenumber_cm1(nu)
        n1 = np.asarray(n_of_lambda_um(lam))
        ct = snell_cos_theta_t(n1, math.radians(theta_i_deg), n0=n0_env)
        nbar = float(np.median(np.real(n1)))
        ctbar = float(np.median(ct))
        d0 = 1.0/(2.0 * nbar * max(ctbar, 1e-6) * fsr_cm1)  # μm (因为 1/cm * 1e4? 注意：FSR 在波数，d 公式直接 μm OK)
        # 推导：Δν̃ = 1/(2 n d cosθ)；d = 1/(2 n cosθ Δν̃)，单位一致（d[cm]）；我们要 μm -> 乘 1e4
        # 这里 Δν̃ 单位 cm^-1，d 的公式给出的是 cm，需要 ×1e4 变 μm：
        d0 *= 1e4
        return d0


# =========================
# 薄膜光学（Airy/TMM）
# =========================
def fresnel_r_s(n_i, n_j, ct_i, ct_j):
    return (n_i*ct_i - n_j*ct_j) / (n_i*ct_i + n_j*ct_j + 0j)

def fresnel_r_p(n_i, n_j, ct_i, ct_j):
    return (n_j*ct_i - n_i*ct_j) / (n_j*ct_i + n_i*ct_j + 0j)

class AirySingleLayer:
    """
    空气(0)/外延(1)/衬底(2) 单层膜反射率
    支持复折射率；未偏振：R = 0.5(|r_s|^2 + |r_p|^2)
    """
    def __init__(self, n0_env: float = 1.0):
        self.n0 = n0_env

    def reflectance(self, lam_um: np.ndarray, theta_i_rad: float,
                    n1_of_lam: Callable[[np.ndarray], np.ndarray],
                    n2_of_lam: Callable[[np.ndarray], np.ndarray],
                    d_um: float) -> np.ndarray:
        lam = np.asarray(lam_um, dtype=float)
        k0 = 2.0 * np.pi / np.maximum(lam, EPS)  # 单位 μm^-1
        n0 = self.n0 + 0j
        n1 = np.asarray(n1_of_lam(lam)) + 0j
        n2 = np.asarray(n2_of_lam(lam)) + 0j

        # 角度
        s0 = math.sin(theta_i_rad)
        # 对复数折射率，Snell 用 n 的模长近似决定 cosθ_t（工程上常用）
        ct1 = snell_cos_theta_t(n1, theta_i_rad, n0=self.n0)
        ct2 = snell_cos_theta_t(n2, theta_i_rad, n0=self.n0)
        ct0 = math.cos(theta_i_rad)

        # 相位厚度 β = k0 * n1 * d * cosθ1
        beta = k0 * n1 * d_um * ct1

        # Fresnel 振幅系数
        r01_s = fresnel_r_s(n0, n1, ct0, ct1)
        r12_s = fresnel_r_s(n1, n2, ct1, ct2)
        r01_p = fresnel_r_p(n0, n1, ct0, ct1)
        r12_p = fresnel_r_p(n1, n2, ct1, ct2)

        e2iβ = np.exp(2j * beta)
        # Airy 总反射振幅：r = (r01 + r12 e^{2iβ}) / (1 + r01 r12 e^{2iβ})
        r_s = (r01_s + r12_s * e2iβ) / (1.0 + r01_s * r12_s * e2iβ + 0j)
        r_p = (r01_p + r12_p * e2iβ) / (1.0 + r01_p * r12_p * e2iβ + 0j)

        R = 0.5 * (np.abs(r_s)**2 + np.abs(r_p)**2)
        return R.real.clip(0.0, 1.0)


# =========================
# 数据集 & 拟合配置
# =========================
@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    theta_deg: float = 0.0

@dataclass
class FitConfig:
    n1_of_lam: Callable[[np.ndarray], np.ndarray]  # 外延层 n(λ)
    n2_of_lam: Callable[[np.ndarray], np.ndarray]  # 衬底 n(λ)
    n0_env: float = 1.0
    pre: PreprocessCfg = PreprocessCfg()
    loss: str = "cauchy"       # 'linear' / 'cauchy' / 'huber' ...
    f_scale: Optional[float] = None  # 若 None 自动用 MAD
    max_nfev: int = 30000
    # 初值/边界控制
    tighten_init_bounds: bool = True
    init_bound_frac: float = 0.15  # 先在 d0±15% 内拟合一轮，再放宽
    d_bounds_um: Tuple[float, float] = (1e-3, 1e6)  # μm


# =========================
# 厚度拟合器（多角联合）
# =========================
class ThicknessFitter:
    def __init__(self, cfg: FitConfig):
        self.cfg = cfg
        self.pre = Preprocessor(cfg.pre)
        self.fft = FFTInit()
        self.model = AirySingleLayer(cfg.n0_env)

    def _prep_all(self, datasets: List[Dataset]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        nu_list, ydet_list, yfft_list = [], [], []
        for ds in datasets:
            p = self.pre.run(ds.df)
            nu_list.append(p["nu"]); ydet_list.append(p["y_det"]); yfft_list.append(p["y_fft"])
        return nu_list, ydet_list, yfft_list

    def _initial_d(self, nu_list: List[np.ndarray], yfft_list: List[np.ndarray], theta_list: List[float]) -> Dict[str, Any]:
        fsr_list, fpk_list, clarity_list = [], [], []
        d0_list = []
        for nu, yfft, th_deg in zip(nu_list, yfft_list, theta_list):
            out = self.fft.estimate_fsr(nu, yfft)
            fsr, fpk, clarity = out["fsr"], out["f_peak"], out["clarity"]
            fsr_list.append(fsr); fpk_list.append(fpk); clarity_list.append(clarity)
            d0 = self.fft.d0_from_fsr(nu, self.cfg.n1_of_lam, th_deg, fsr, n0_env=self.cfg.n0_env)
            d0_list.append(d0)
        return dict(
            d0=float(np.median(d0_list)),
            fpk=fpk_list, fsr=fsr_list, clarity=clarity_list, d0_list=d0_list
        )

    def _residual_joint(self, params: np.ndarray, datasets: List[Dataset], nu_list, ydet_list) -> np.ndarray:
        """
        参数向量：
          p = [ d_um,  A1, B1,  A2, B2,  ...  AK, BK ]
        其中 Ak + Bk*(R_model - mean) 做每角的线性标定
        """
        d_um = float(params[0])
        offs = 1
        res_all = []
        for k, ds in enumerate(datasets):
            A = float(params[offs]); B = float(params[offs+1]); offs += 2
            nu = nu_list[k]; y = ydet_list[k]
            lam = wavelength_um_from_wavenumber_cm1(nu)
            Rmod = self.model.reflectance(lam, math.radians(ds.theta_deg),
                                          self.cfg.n1_of_lam, self.cfg.n2_of_lam, d_um)
            # 线性标定（中心化提高数值稳定）
            Rm = float(np.mean(Rmod))
            yhat = A + B*(Rmod - Rm)
            res_all.append(yhat - y)
        return np.concatenate(res_all)

    def fit(self, datasets: List[Dataset]) -> Dict[str, Any]:
        nu_list, ydet_list, yfft_list = self._prep_all(datasets)
        theta_list = [ds.theta_deg for ds in datasets]

        # 初值
        init = self._initial_d(nu_list, yfft_list, theta_list)
        d0 = init["d0"]

        # 初轮边界（收紧）
        if self.cfg.tighten_init_bounds:
            lo = max(self.cfg.d_bounds_um[0], (1.0 - self.cfg.init_bound_frac) * d0)
            hi = min(self.cfg.d_bounds_um[1], (1.0 + self.cfg.init_bound_frac) * d0)
        else:
            lo, hi = self.cfg.d_bounds_um

        # 参数向量初值
        p0 = [d0]
        bounds_lo = [lo]; bounds_hi = [hi]
        for _ in datasets:
            # A,B 初值与边界（相对量，给宽松范围）
            p0 += [0.0, 1.0]
            bounds_lo += [-1.0, 0.0]
            bounds_hi += [ 1.0, 5.0]
        p0 = np.array(p0, dtype=float)
        bounds = (np.array(bounds_lo, dtype=float), np.array(bounds_hi, dtype=float))

        f_scale = self.cfg.f_scale
        if f_scale is None:
            mad_vals = [robust_mad(y) for y in ydet_list]
            f_scale = float(np.median(mad_vals))

        # 第一次拟合（收紧边界）
        r1 = least_squares(self._residual_joint, p0, args=(datasets, nu_list, ydet_list),
                           bounds=bounds, loss=self.cfg.loss, f_scale=f_scale,
                           max_nfev=self.cfg.max_nfev)

        # 第二次拟合（放宽 d 边界）
        p1 = r1.x.copy()
        bounds2 = (np.array([self.cfg.d_bounds_um[0]] + [-1.0, 0.0]*len(datasets), dtype=float),
                   np.array([self.cfg.d_bounds_um[1]] + [ 1.0, 5.0]*len(datasets), dtype=float))
        r2 = least_squares(self._residual_joint, p1, args=(datasets, nu_list, ydet_list),
                           bounds=bounds2, loss=self.cfg.loss, f_scale=f_scale,
                           max_nfev=self.cfg.max_nfev)

        # 解析结果
        params = r2.x
        d_um = float(params[0])
        per_angle = {}
        offs = 1
        for k, ds in enumerate(datasets):
            A, B = float(params[offs]), float(params[offs+1]); offs += 2
            per_angle[ds.name] = dict(A=A, B=B)

        # 残差与 RMSE
        resid = self._residual_joint(params, datasets, nu_list, ydet_list)
        rmse = float(np.sqrt(np.mean(resid**2)))

        # 单角一致性（固定 d，只拟 A,B）——可作为诊断
        d_consistency = []
        for k, ds in enumerate(datasets):
            nu = nu_list[k]; y = ydet_list[k]
            lam = wavelength_um_from_wavenumber_cm1(nu)
            Rmod = self.model.reflectance(lam, math.radians(ds.theta_deg),
                                          self.cfg.n1_of_lam, self.cfg.n2_of_lam, d_um)
            Rm = float(np.mean(Rmod))
            # 拟合 A,B
            def res_ab(p):
                A, B = p
                return (A + B*(Rmod - Rm)) - y
            r_ab = least_squares(res_ab, x0=np.array([0.0, 1.0]),
                                 bounds=([-1.0, 0.0],[1.0, 5.0]),
                                 loss=self.cfg.loss, f_scale=f_scale, max_nfev=20000)
            d_consistency.append(dict(name=ds.name, rmse=float(np.sqrt(np.mean(res_ab(r_ab.x)**2))),
                                      A=float(r_ab.x[0]), B=float(r_ab.x[1])))

        out = dict(
            success=bool(r2.success),
            message=str(r2.message),
            d_um=d_um,
            rmse=rmse,
            init=init,
            per_angle=per_angle,
            d_consistency=d_consistency,
            nfev=int(r1.nfev + r2.nfev),
        )
        return out

    def bootstrap_ci(self, datasets: List[Dataset], n_boot: int = 200, keep_ratio: float = 0.7, seed: int = 0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        nu_list, ydet_list, _ = self._prep_all(datasets)

        d_vals = []
        for _ in range(n_boot):
            # 连续窗口子采样（block bootstrap）
            sub_sets = []
            for ds, nu, y in zip(datasets, nu_list, ydet_list):
                n = len(nu)
                m = max(16, int(round(n*keep_ratio)))
                start = rng.integers(low=0, high=max(n-m, 1))
                idx = slice(start, start+m)
                sub_df = pd.DataFrame({"波数 (cm-1)": nu[idx], "反射率 (%)": (y[idx]+np.mean(y))*100.0})
                sub_sets.append(Dataset(name=ds.name, df=sub_df, theta_deg=ds.theta_deg))
            try:
                res = self.fit(sub_sets)
                if res["success"]:
                    d_vals.append(float(res["d_um"]))
            except Exception:
                continue

        if len(d_vals)==0:
            return dict(success=False, note="no valid bootstrap samples")
        arr = np.array(d_vals, dtype=float)
        return dict(success=True, n=len(arr),
                    mean=float(np.mean(arr)), std=float(np.std(arr, ddof=1)),
                    q05=float(np.quantile(arr, 0.05)),
                    q50=float(np.quantile(arr, 0.50)),
                    q95=float(np.quantile(arr, 0.95)))


# =========================
# 可视化（可选）
# =========================
def plot_fit(datasets: List[Dataset], fitter: ThicknessFitter, fit_res: Dict[str, Any]):
    pre = fitter.pre
    d_um = fit_res["d_um"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(9, 3.2*len(datasets)), sharex=False)
    if len(datasets)==1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        p = pre.run(ds.df)
        nu, y = p["nu"], p["y_det"]
        lam = wavelength_um_from_wavenumber_cm1(nu)
        Rmod = fitter.model.reflectance(lam, math.radians(ds.theta_deg),
                                        fitter.cfg.n1_of_lam, fitter.cfg.n2_of_lam, d_um)
        Rm = float(np.mean(Rmod))
        A = fit_res["per_angle"][ds.name]["A"]; B = fit_res["per_angle"][ds.name]["B"]
        yhat = A + B*(Rmod - Rm)

        ax.plot(nu, y, lw=1.0, label="实测(去趋势)")
        ax.plot(nu, yhat, lw=1.0, ls="--", label=f"Airy 拟合(d={d_um:.3f} μm)")
        ax.set_ylabel("相对幅值")
        ax.set_title(f"{ds.name} （θ={ds.theta_deg:.1f}°）")
        ax.grid(alpha=0.3); ax.legend(loc="best")
    axes[-1].set_xlabel("波数 (cm$^{-1}$)")
    fig.tight_layout()
    return fig


# =========================
# —— 使用示例 ——
# 假设 df3/df4 已经加载为 DataFrame
# =========================
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 角度：如果试验没有记录，就设 0°；若有多角，按实测填写
    datasets = [
        Dataset(name="Si 附件3", df=df3, theta_deg=0.0),
        Dataset(name="Si 附件4", df=df4, theta_deg=0.0),
    ]

    # 配置：外延与衬底都用 Si 的色散（若认为外延与衬底有微小差异，可把 n1_of_lam 改成 lambda lam: n_si_um(lam)+Δn）
    fit_cfg = FitConfig(
        n1_of_lam=n_si_um,
        n2_of_lam=n_si_um,
        n0_env=1.0,
        pre=PreprocessCfg(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_for_fft=True,
                          nu_min=1200.0, nu_max=3800.0),
        loss="cauchy",
        f_scale=None,         # None 表示用 MAD 自适应
        max_nfev=40000,
        tighten_init_bounds=True,
        init_bound_frac=0.15,
        d_bounds_um=(1e-3, 1e6)
    )

    fitter = ThicknessFitter(fit_cfg)
    result = fitter.fit(datasets)

    print("\n=== 多角联合拟合结果 ===")
    print(f"success={result['success']}, message={result['message']}")
    print(f"d = {result['d_um']:.4f} μm,  RMSE = {result['rmse']:.6f},  nfev={result['nfev']}")
    print("FFT/初值诊断：")
    print("  fpk(Hz/cm^-1) =", ["%.6f" % x for x in result['init']['fpk']])
    print("  FSR(cm^-1)    =", ["%.3f"   % x for x in result['init']['fsr']])
    print("  d0_list(μm)   =", ["%.4f"   % x for x in result['init']['d0_list']])
    print("每角标定参数与单角一致性：")
    for k, (name, ab) in enumerate(result["per_angle"].items()):
        print(f"  {name}:  A={ab['A']:.4f}, B={ab['B']:.4f}")
    for item in result["d_consistency"]:
        print(f"  [单角固定d拟合] {item['name']}: rmse={item['rmse']:.6f}, A={item['A']:.4f}, B={item['B']:.4f}")

    # 可选：Bootstrap 置信区间（耗时可控，示例 200 次）
    ci = fitter.bootstrap_ci(datasets, n_boot=200, keep_ratio=0.7, seed=0)
    print("\nBootstrap CI:", ci)

    fig = plot_fit(datasets, fitter, result)
    if fig: plt.show()
