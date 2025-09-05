# -*- coding: utf-8 -*-
"""
硅外延层厚度反演（多光束/TMM，多角联合 + 鲁棒非线性最小二乘）
- 输入：df3, df4（两列：波数 (cm-1), 反射率 (%))
- 输出：厚度 d（μm），每角(a_k,b_k)线性标定，RMSE/诊断
- 特色：FFT→FSR自动初值；鲁棒损失；可选小Δn共拟合（默认关闭）

依赖：numpy, pandas, scipy (optimize, signal, fft), matplotlib
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares


# =========================
# 工具函数
# =========================
EPS = 1e-12

def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    """λ[μm] = 1e4 / ν[cm^-1]"""
    return 1e4 / np.maximum(nu_cm1, EPS)

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def complex_cos_theta(N: np.ndarray, N0: float, theta0_rad: float) -> np.ndarray:
    """
    由 Snell 定律计算各介质中的 cosθ：cosθ_j = sqrt(1 - (N0^2/N_j^2) sin^2 θ0)
    支持 N 为复数；取主值分支。
    """
    s0 = np.sin(theta0_rad)
    val = 1.0 - (N0**2) * (s0**2) / (N**2 + 0j)
    # 数值稳健的小正实部
    return np.sqrt(val + 0j)

def fresnel_r_s_p(Ni: np.ndarray, Nj: np.ndarray, cos_ti: np.ndarray, cos_tj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """界面 i|j 的 s/p 振幅反射系数（Ni, Nj, cosθ_i, cosθ_j 可为复数/数组）"""
    # s 偏振
    rs = (Ni*cos_ti - Nj*cos_tj) / (Ni*cos_ti + Nj*cos_tj + 0j)
    # p 偏振
    rp = (Nj*cos_ti - Ni*cos_tj) / (Nj*cos_ti + Ni*cos_tj + 0j)
    return rs, rp

def airy_single_layer_r(N0: float, N1: np.ndarray, N2: np.ndarray,
                        theta0_rad: float, lam_um: np.ndarray, d_um: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    单层（介质1）在半无限衬底（介质2）上的总振幅反射系数 r_s, r_p
    Airy/TMM 等价公式：
      r = (r01 + r12 * e^{2iβ}) / (1 + r01 * r12 * e^{2iβ}),  β = 2π N1 d cosθ1 / λ
    """
    # 各介质内 cosθ
    cos0 = math.cos(theta0_rad) + 0j  # 空气近似实数
    cos1 = complex_cos_theta(N1, N0=N0, theta0_rad=theta0_rad)
    cos2 = complex_cos_theta(N2, N0=N0, theta0_rad=theta0_rad)

    # Fresnel 振幅 r01, r12
    r01_s, r01_p = fresnel_r_s_p(N0+0j, N1, cos0+0j, cos1)
    r12_s, r12_p = fresnel_r_s_p(N1, N2, cos1, cos2)

    # 相位厚度 β
    beta = 2.0*np.pi * (N1 * d_um * cos1) / (lam_um + 0j)

    e2ib = np.exp(2j * beta)

    # Airy 总反射
    r_s = (r01_s + r12_s * e2ib) / (1.0 + r01_s * r12_s * e2ib + 0j)
    r_p = (r01_p + r12_p * e2ib) / (1.0 + r01_p * r12_p * e2ib + 0j)
    return r_s, r_p

def unpolarized_R_from_r(r_s: np.ndarray, r_p: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(r_s)**2 + np.abs(r_p)**2)


# =========================
# 折射率模型（传入 Callable）
# =========================
def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    """
    题设的硅色散（λ[μm]）：
      n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814 λ^2
    """
    lam2 = np.asarray(lam_um, dtype=float)**2
    denom = np.maximum(lam2 - 0.0086024, 1e-12)
    n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
    n = np.sqrt(np.maximum(n2, 1e-12))
    return n  # 实折射率；如需吸收，可返回 n + 1j*k

def n_sic_um(lam_um: np.ndarray) -> np.ndarray:
    """
    一个常用的 4H-SiC 近似（可换成你的更精确拟合）：
      n(λ[nm]) ≈ 2.5610 + 3.4e4 / λ^2
    这里输入 λ[μm]，先转 nm。
    """
    lam_nm = np.asarray(lam_um, dtype=float) * 1000.0
    n = 2.5610 + 3.4e4 / np.maximum(lam_nm**2, 1e-9)
    return n


# =========================
# 预处理与 FFT 初值
# =========================
@dataclass
class PreprocessCfg:
    detrend: bool = True
    sg_window_frac: float = 0.12
    sg_polyorder: int = 2
    nu_min: Optional[float] = 1200.0
    nu_max: Optional[float] = 3800.0
    normalize_for_fft: bool = True

def preprocess_df(df: pd.DataFrame, cfg: PreprocessCfg) -> Dict[str, np.ndarray]:
    col_nu = "波数 (cm-1)"
    col_R  = "反射率 (%)"
    if col_nu not in df.columns or col_R not in df.columns:
        col_nu, col_R = df.columns[:2]

    nu = df[col_nu].to_numpy(dtype=float)
    R  = df[col_R].to_numpy(dtype=float) / 100.0

    # 升序 + 等间隔
    idx = np.argsort(nu)
    nu, R = nu[idx], R[idx]
    dnu = np.mean(np.diff(nu))
    if not np.isfinite(dnu) or dnu <= 0:
        nu_g, R_g = nu, R
    else:
        nu_g = np.arange(nu[0], nu[-1] + 0.5*dnu, dnu)
        R_g  = np.interp(nu_g, nu, R)

    # 窗口
    lo = -np.inf if cfg.nu_min is None else cfg.nu_min
    hi =  np.inf if cfg.nu_max is None else cfg.nu_max
    m  = (nu_g >= lo) & (nu_g <= hi)
    nu_w, R_w = nu_g[m], R_g[m]

    # 去趋势
    if cfg.detrend and len(R_w) >= 11:
        n = len(R_w)
        w = max(11, int(round(cfg.sg_window_frac * n)))
        if w % 2 == 0: w += 1
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        w = max(w, cfg.sg_polyorder + 3)
        base = savgol_filter(R_w, window_length=w, polyorder=cfg.sg_polyorder, mode="interp")
        y_det = R_w - base
    else:
        y_det = R_w - float(np.mean(R_w))

    # FFT 用单位 RMS
    if cfg.normalize_for_fft:
        z = y_det - float(np.mean(y_det))
        rmsv = math.sqrt(float(np.mean(z**2))) + EPS
        y_fft = z / rmsv
    else:
        y_fft = y_det

    return dict(nu=nu_w, R=R_w, y_det=y_det, y_fft=y_fft)

def fft_fsr(nu: np.ndarray, y_fft: np.ndarray, min_cycles: float = 3.0) -> Tuple[float, float, float]:
    """返回 f_peak, Δν(=1/f_peak), 主峰清晰度"""
    dnu = float(np.mean(np.diff(nu)))
    Y = rfft(y_fft - float(np.mean(y_fft)))
    freqs = rfftfreq(len(y_fft), d=dnu)
    span = float(nu[-1] - nu[0] + EPS)
    low = (1.0 / span) * min_cycles
    m   = freqs > low
    amp = np.abs(Y)[m]; fpos = freqs[m]
    k   = int(np.argmax(amp))
    fpk = float(fpos[k])
    a_sorted = np.sort(amp)
    sec = float(a_sorted[-2]) if len(a_sorted) >= 2 else 0.0
    clarity = float(amp[k]) / (sec + EPS)
    return fpk, 1.0/fpk, clarity

def thickness_init_from_fsr(Delta_nu_cm1: float, nbar: float, theta0_deg: float) -> float:
    """
    d_μm ≈ 1e4 / (2 n cosθ_t Δν)；θ_t 用 Snell：sinθ_t = sinθ0 / n
    近似用 nbar 代表膜内折射率的中值
    """
    theta0 = math.radians(theta0_deg)
    s0 = math.sin(theta0)
    # 膜内角（近似：n0=1, N1≈nbar）
    sin_t = s0 / max(nbar, 1e-6)
    sin_t = min(max(sin_t, -0.999999), 0.999999)
    cos_t = math.sqrt(1.0 - sin_t**2)
    d_cm  = 1.0 / (2.0 * nbar * cos_t * Delta_nu_cm1 + EPS)
    return d_cm * 1.0e4  # μm


# =========================
# 拟合配置与残差构造
# =========================
@dataclass
class FitCfg:
    n_of_lambda_um: Callable[[np.ndarray], np.ndarray]   # Si 的折射率函数
    n_substrate_of_lambda_um: Optional[Callable[[np.ndarray], np.ndarray]] = None  # 默认为同一函数
    n0_env: float = 1.0
    theta_list_deg: Tuple[float, ...] = (10.0, 15.0)
    allow_delta_n: bool = False         # 是否拟合外延层相对基底的 Δn（常数）
    delta_n_bounds: Tuple[float, float] = (-0.01, 0.01)
    use_robust_loss: bool = True
    f_scale: Optional[float] = None     # 若 None 用 1.5 * MAD
    # 初值与边界
    d_bounds_um: Tuple[float, float] = (0.1, 200.0)  # μm，广域边界；会在运行时先收紧再放开

def build_residuals(params: np.ndarray,
                    packed_data: Dict[str, Any],
                    cfg: FitCfg) -> np.ndarray:
    """
    params: [d_um, (可选) delta_n, (a_k,b_k)*K]
    """
    K = len(packed_data["angles"])
    i = 0
    d_um = float(params[i]); i += 1
    if cfg.allow_delta_n:
        delta_n = float(params[i]); i += 1
    else:
        delta_n = 0.0
    a_list = []; b_list = []
    for _ in range(K):
        a_list.append(float(params[i])); b_list.append(float(params[i+1])); i += 2

    res_all = []
    for k,(theta_deg, nu, R_meas, y_det) in enumerate(zip(packed_data["angles"], packed_data["nu_list"], packed_data["R_list"], packed_data["y_list"])):
        lam = wavelength_um_from_wavenumber_cm1(nu)

        # 折射率（外延/衬底）
        n_base = cfg.n_of_lambda_um(lam)
        n1 = n_base + delta_n          # 外延
        n2 = cfg.n_substrate_of_lambda_um(lam) if cfg.n_substrate_of_lambda_um else n_base  # 衬底
        N1 = n1 + 0j; N2 = n2 + 0j

        # 反射谱（未偏振）
        r_s, r_p = airy_single_layer_r(cfg.n0_env, N1, N2, math.radians(theta_deg), lam, d_um)
        R_model = unpolarized_R_from_r(r_s, r_p).real  # 安全起见取实部

        # 线性标定
        R_center = float(np.median(R_model))
        R_cal = a_list[k] + b_list[k] * (R_model - R_center)

        # 残差：用“去趋势的实测 y_det”和“去中心的模型”对齐，避免基线主导
        # 让模型残差与 y_det 处于同一尺度（单位一致）
        # 这里把 y_det 作为目标，R_cal 也先去中心
        y_model = R_cal - float(np.median(R_cal))
        res = y_model - y_det
        res_all.append(res)

    res_all = np.concatenate(res_all)
    if cfg.use_robust_loss:
        fscale = cfg.f_scale if cfg.f_scale is not None else 1.5 * robust_mad(res_all)
        # Cauchy ρ 的等价“伪残差”（用于 least_squares 的 loss='linear' 情况）：
        # 这里我们直接返回真实残差，并在 least_squares 中使用 loss='cauchy', f_scale=fscale
        packed_data["_f_scale"] = fscale
    return res_all


# =========================
# 主求解器
# =========================
def fit_thickness_joint(dfs: List[pd.DataFrame],
                        pre_cfg: PreprocessCfg,
                        fit_cfg: FitCfg,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    多角联合拟合厚度 d（μm）
    dfs 的顺序应与 fit_cfg.theta_list_deg 对应
    """
    assert len(dfs) == len(fit_cfg.theta_list_deg), "dfs 与 theta_list_deg 长度需一致"

    # 预处理 & FFT 初值
    nu_list, R_list, y_list, angles = [], [], [], []
    d0_list = []
    for df, theta_deg in zip(dfs, fit_cfg.theta_list_deg):
        pre = preprocess_df(df, pre_cfg)
        nu, R, y_det, y_fft = pre["nu"], pre["R"], pre["y_det"], pre["y_fft"]
        nu_list.append(nu); R_list.append(R); y_list.append(y_det); angles.append(theta_deg)

        fpk, Delta_nu, clarity = fft_fsr(nu, y_fft)
        lam = wavelength_um_from_wavenumber_cm1(nu)
        nbar = float(np.median(fit_cfg.n_of_lambda_um(lam)))
        d0 = thickness_init_from_fsr(Delta_nu, nbar, theta_deg)
        d0_list.append(d0)

        if verbose:
            print(f"[初值] θ={theta_deg:.1f}°: f_peak={fpk:.6f}, FSR={Delta_nu:.3f} cm^-1 → d0≈{d0:.3f} μm, clarity={clarity:.2f}")

    d0_joint = float(np.median(d0_list))
    if verbose:
        print(f"[初值] 联合 d0 = {d0_joint:.3f} μm")

    # 打包拟合数据
    packed = dict(nu_list=nu_list, R_list=R_list, y_list=y_list, angles=angles)

    # 参数顺序：[d_um, (可选)delta_n, (a1,b1), (a2,b2), ...]
    K = len(angles)
    p0 = [d0_joint]
    if fit_cfg.allow_delta_n: p0 += [0.0]
    for _ in range(K):
        p0 += [0.0, 1.0]   # a_k=0, b_k=1 初值
    p0 = np.array(p0, dtype=float)

    # 边界：先在 d0 的±20% 收紧一轮，再放宽
    bounds_narrow = []
    d_low = max(fit_cfg.d_bounds_um[0], 0.8*d0_joint)
    d_high= min(fit_cfg.d_bounds_um[1], 1.2*d0_joint)
    bounds_lower = [d_low]
    bounds_upper = [d_high]
    if fit_cfg.allow_delta_n:
        bounds_lower += [fit_cfg.delta_n_bounds[0]]
        bounds_upper += [fit_cfg.delta_n_bounds[1]]
    for _ in range(K):
        bounds_lower += [-0.2, 0.5]   # a_k ∈ [-0.2,0.2]；b_k ∈ [0.5, 1.5]
        bounds_upper += [ 0.2, 1.5]
    bounds_narrow = (np.array(bounds_lower, float), np.array(bounds_upper, float))

    # 第一轮（收紧）
    r1 = least_squares(build_residuals, p0,
                       args=(packed, fit_cfg),
                       bounds=bounds_narrow,
                       loss='cauchy' if fit_cfg.use_robust_loss else 'linear',
                       f_scale=1.0,  # 实际 f_scale 在 residual 内设置并由 least_squares 使用
                       max_nfev=20000, verbose=0)

    # 第二轮（放宽到全局边界）
    bounds_lower2 = [fit_cfg.d_bounds_um[0]]
    bounds_upper2 = [fit_cfg.d_bounds_um[1]]
    if fit_cfg.allow_delta_n:
        bounds_lower2 += [fit_cfg.delta_n_bounds[0]]
        bounds_upper2 += [fit_cfg.delta_n_bounds[1]]
    for _ in range(K):
        bounds_lower2 += [-0.5, 0.2]
        bounds_upper2 += [ 0.5, 2.0]
    bounds_wide = (np.array(bounds_lower2, float), np.array(bounds_upper2, float))

    p_init2 = r1.x
    r2 = least_squares(build_residuals, p_init2,
                       args=(packed, fit_cfg),
                       bounds=bounds_wide,
                       loss='cauchy' if fit_cfg.use_robust_loss else 'linear',
                       f_scale=packed.get("_f_scale", 1.0),
                       max_nfev=40000, verbose=0)

    # 结果解析
    sol = r2.x
    i = 0
    d_um = float(sol[i]); i += 1
    if fit_cfg.allow_delta_n:
        delta_n = float(sol[i]); i += 1
    else:
        delta_n = 0.0
    akbk = []
    for _ in range(K):
        akbk.append((float(sol[i]), float(sol[i+1]))); i += 2

    # 计算拟合残差、RMSE、绘图数据
    res = build_residuals(r2.x, packed, fit_cfg)
    rmse_all = rms(res)
    rmse_per = []
    curves = []
    idx = 0
    for k,(theta_deg, nu, R_meas, y_det) in enumerate(zip(angles, packed["nu_list"], packed["R_list"], packed["y_list"])):
        npts = len(nu)
        lam = wavelength_um_from_wavenumber_cm1(nu)
        n_base = fit_cfg.n_of_lambda_um(lam)
        n1 = n_base + delta_n
        n2 = fit_cfg.n_substrate_of_lambda_um(lam) if fit_cfg.n_substrate_of_lambda_um else n_base
        r_s, r_p = airy_single_layer_r(fit_cfg.n0_env, n1+0j, n2+0j, math.radians(theta_deg), lam, d_um)
        R_model = unpolarized_R_from_r(r_s, r_p).real
        a_k, b_k = akbk[k]
        R_center = float(np.median(R_model))
        R_cal = a_k + b_k * (R_model - R_center)
        y_model = R_cal - float(np.median(R_cal))
        curves.append((nu, y_det, y_model))
        res_k = res[idx: idx+npts]; idx += npts
        rmse_per.append(rms(res_k))

    out = dict(
        success=r2.success, message=r2.message, nfev=r2.nfev,
        d_um=d_um, delta_n=delta_n, akbk=akbk,
        rmse_all=rmse_all, rmse_per=rmse_per,
        curves=curves,  # [(nu, y_meas_detrended, y_model_centered), ...]
        pre_cfg=pre_cfg, fit_cfg=fit_cfg
    )
    return out


# =========================
# 可视化
# =========================
def plot_joint_fit(result: Dict[str, Any], title_prefix: str = "Si 外延层 多角联合拟合"):
    K = len(result["curves"])
    fig, axes = plt.subplots(K, 1, figsize=(10, 3.2*K), sharex=True)
    if K == 1: axes = [axes]
    for ax_idx,(ax, (nu, y_meas, y_model)) in enumerate(zip(axes, result["curves"])):
        ax.plot(nu, y_meas, lw=1.0, label="实测(去趋势)")
        ax.plot(nu, y_model, lw=1.0, ls="--", label="模型(去中心)")
        ax.set_ylabel("相对幅值")
        ax.grid(alpha=0.3); ax.legend(loc="best")
    axes[-1].set_xlabel("波数 (cm$^{-1}$)")
    axes[0].set_title(f"{title_prefix}：d={result['d_um']:.3f} μm；RMSE={result['rmse_all']:.4f}")
    fig.tight_layout()
    return fig


# =========================
# main：把 df3/df4 代入
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

    # 预处理（窗口可按你的判据调节）
    pre_cfg = PreprocessCfg(detrend=True, sg_window_frac=0.12, sg_polyorder=2,
                            nu_min=1200.0, nu_max=3800.0, normalize_for_fft=True)

    # 拟合配置（Si 外延层/衬底都用相同 n(λ)，如需考虑外延与衬底微差可设 allow_delta_n=True）
    fit_cfg = FitCfg(
        n_of_lambda_um=n_si_um,
        n_substrate_of_lambda_um=n_si_um,
        n0_env=1.0,
        theta_list_deg=(10.0, 15.0),
        allow_delta_n=False,           # 打开则会同时估计 Δn
        delta_n_bounds=(-0.01, 0.01),
        use_robust_loss=True,
        f_scale=None,
        d_bounds_um=(0.1, 200.0)
    )

    # 运行：多角联合
    result = fit_thickness_joint([df3, df4], pre_cfg, fit_cfg, verbose=True)

    print("\n=== 拟合结果 ===")
    print(f"success={result['success']}, message={result['message']}, nfev={result['nfev']}")
    print(f"d = {result['d_um']:.4f} μm")
    if fit_cfg.allow_delta_n:
        print(f"Δn = {result['delta_n']:.6f}")
    for k,(a,b) in enumerate(result['akbk'], start=1):
        print(f"(a_{k}, b_{k}) = ({a:+.5f}, {b:.5f})")
    print(f"RMSE(all) = {result['rmse_all']:.6f}")
    for k,rm in enumerate(result['rmse_per'], start=1):
        print(f"  RMSE(angle {k}) = {rm:.6f}")

    # 图
    fig = plot_joint_fit(result, title_prefix="Si 外延层 多角联合拟合")
    plt.show()
