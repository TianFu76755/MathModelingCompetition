# -*- coding: utf-8 -*-
"""
Si epitaxial thickness estimation (Extended)
- Multi-beam Airy/TMM model
- Joint fit of thickness d and layer/substrate optical contrast: Δn, Δκ
- Multi-angle, robust loss, two-stage bounds, per-angle linear calibration (A,B)

Python 3.8+
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

# =========================
# 工具 & 折射率
# =========================
EPS = 1e-12


def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    return 1e4 / np.maximum(nu_cm1, EPS)


def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS


def snell_cos_theta_t(n: np.ndarray, theta_i_rad: float, n0: float = 1.0) -> np.ndarray:
    s = math.sin(theta_i_rad)
    n_abs = np.maximum(np.abs(n), EPS)
    val = 1.0 - (s * n0 / n_abs) ** 2
    return np.sqrt(np.clip(val, 0.0, 1.0))


def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    lam_um = np.asarray(lam_um, dtype=float)
    lam2 = lam_um ** 2
    denom = np.maximum(lam2 - 0.0086024, 1e-12)
    n2 = 11.6858 + 0.939816 / denom + 0.00089814 * lam2
    return np.sqrt(np.maximum(n2, 1e-12))


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
        nu_g = np.arange(nu[0], nu[-1] + 0.5 * dnu, dnu)
        R_g = np.interp(nu_g, nu, R)
        return nu_g, R_g

    def _window(self, nu: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = -np.inf if self.cfg.nu_min is None else self.cfg.nu_min
        hi = np.inf if self.cfg.nu_max is None else self.cfg.nu_max
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
        rms = math.sqrt(float(np.mean(z ** 2))) + EPS
        return z / rms

    def run(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        col_nu, col_R = "波数 (cm-1)", "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            col_nu, col_R = df.columns[:2]
        nu = df[col_nu].to_numpy(dtype=float)
        R = (df[col_R].to_numpy(dtype=float)) / 100.0

        nu_g, R_g = self._uniform_grid(nu, R)
        nu_w, R_w = self._window(nu_g, R_g)
        y_det = self._detrend(R_w)
        y_fft = self._normalize(y_det)
        return dict(nu=nu_w, R=R_w, y_det=y_det, y_fft=y_fft)


# =========================
# FFT 初值
# =========================
class FFTInit:
    def __init__(self, min_cycles: float = 3.0):
        self.min_cycles = min_cycles

    def estimate_fsr(self, nu: np.ndarray, y_fft: np.ndarray) -> Dict[str, float]:
        dnu = float(np.mean(np.diff(nu)))
        Y = rfft(y_fft - float(np.mean(y_fft)))
        freqs = rfftfreq(len(y_fft), d=dnu)
        span = float(nu[-1] - nu[0] + EPS)
        low_cut = (1.0 / span) * self.min_cycles
        m = freqs > low_cut
        amp = np.abs(Y)[m];
        fpos = freqs[m]
        k = int(np.argmax(amp))
        f_peak = float(fpos[k])
        a_sorted = np.sort(amp)
        sec = float(a_sorted[-2]) if len(a_sorted) >= 2 else 0.0
        clarity = float(amp[k]) / (sec + EPS)
        return dict(f_peak=f_peak, fsr=1.0 / f_peak, clarity=clarity)

    def d0_from_fsr(self, nu: np.ndarray, n_of_lambda_um: Callable[[np.ndarray], np.ndarray],
                    theta_i_deg: float, fsr_cm1: float, n0_env: float = 1.0) -> float:
        lam = wavelength_um_from_wavenumber_cm1(nu)
        n1 = np.asarray(n_of_lambda_um(lam))
        ct = snell_cos_theta_t(n1, math.radians(theta_i_deg), n0=n0_env)
        nbar = float(np.median(np.real(n1)))
        ctbar = float(np.median(ct))
        d0_cm = 1.0 / (2.0 * nbar * max(ctbar, 1e-6) * fsr_cm1)  # in cm
        return d0_cm * 1e4  # -> μm


# =========================
# Airy/TMM 单层反射
# =========================
def fresnel_r_s(n_i, n_j, ct_i, ct_j):
    return (n_i * ct_i - n_j * ct_j) / (n_i * ct_i + n_j * ct_j + 0j)


def fresnel_r_p(n_i, n_j, ct_i, ct_j):
    return (n_j * ct_i - n_i * ct_j) / (n_j * ct_i + n_i * ct_j + 0j)


class AirySingleLayer:
    def __init__(self, n0_env: float = 1.0):
        self.n0 = n0_env

    def reflectance(self, lam_um: np.ndarray, theta_i_rad: float,
                    n1_of_lam: Callable[[np.ndarray], np.ndarray],
                    n2_of_lam: Callable[[np.ndarray], np.ndarray],
                    d_um: float) -> np.ndarray:
        lam = np.asarray(lam_um, dtype=float)
        k0 = 2.0 * np.pi / np.maximum(lam, EPS)  # μm^-1
        n0 = self.n0 + 0j
        n1 = np.asarray(n1_of_lam(lam)) + 0j
        n2 = np.asarray(n2_of_lam(lam)) + 0j

        ct0 = math.cos(theta_i_rad)
        ct1 = snell_cos_theta_t(n1, theta_i_rad, n0=self.n0)
        ct2 = snell_cos_theta_t(n2, theta_i_rad, n0=self.n0)

        beta = k0 * n1 * d_um * ct1

        r01_s = fresnel_r_s(n0, n1, ct0, ct1)
        r12_s = fresnel_r_s(n1, n2, ct1, ct2)
        r01_p = fresnel_r_p(n0, n1, ct0, ct1)
        r12_p = fresnel_r_p(n1, n2, ct1, ct2)

        e2iβ = np.exp(2j * beta)
        r_s = (r01_s + r12_s * e2iβ) / (1.0 + r01_s * r12_s * e2iβ + 0j)
        r_p = (r01_p + r12_p * e2iβ) / (1.0 + r01_p * r12_p * e2iβ + 0j)

        R = 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)
        return R.real.clip(0.0, 1.0)


# =========================
# 数据与配置
# =========================
@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    theta_deg: float = 0.0


@dataclass
class FitConfig:
    # 基础
    n_base_of_lam: Callable[[np.ndarray], np.ndarray]  # 基准 Si 折射率
    n_sub_of_lam: Callable[[np.ndarray], np.ndarray]  # 衬底折射率（一般同 n_base）
    n0_env: float = 1.0
    pre: PreprocessCfg = PreprocessCfg()
    loss: str = "cauchy"
    f_scale: Optional[float] = None
    max_nfev: int = 40000

    # 是否拟合光学对比
    fit_delta_n: bool = True
    fit_delta_k: bool = True

    # 边界与初值
    tighten_init_bounds: bool = True
    init_bound_frac: float = 0.15
    d_bounds_um: Tuple[float, float] = (1e-3, 1e6)
    delta_n_bounds: Tuple[float, float] = (-0.1, 0.1)  # 可按样品收紧
    delta_k_bounds: Tuple[float, float] = (0.0, 0.05)  # 吸收非负，按需要也可允许微负
    delta_n0: float = 0.0
    delta_k0: float = 0.0


# =========================
# 厚度 + Δn/Δκ 拟合器
# =========================
class ThicknessFitterExtended:
    def __init__(self, cfg: FitConfig):
        self.cfg = cfg
        self.pre = Preprocessor(cfg.pre)
        self.fft = FFTInit()
        self.model = AirySingleLayer(cfg.n0_env)

    # ---------- 数据准备与初值 ----------
    def _prep_all(self, datasets: List[Dataset]):
        nu_list, ydet_list, yfft_list = [], [], []
        for ds in datasets:
            p = self.pre.run(ds.df)
            nu_list.append(p["nu"]);
            ydet_list.append(p["y_det"]);
            yfft_list.append(p["y_fft"])
        return nu_list, ydet_list, yfft_list

    def _initials(self, nu_list, yfft_list, theta_list):
        fsr_list, fpk_list, clarity_list, d0_list = [], [], [], []
        for nu, yfft, th in zip(nu_list, yfft_list, theta_list):
            out = self.fft.estimate_fsr(nu, yfft)
            fsr_list.append(out["fsr"]);
            fpk_list.append(out["f_peak"]);
            clarity_list.append(out["clarity"])
            d0_list.append(self.fft.d0_from_fsr(nu, self.cfg.n_base_of_lam, th, out["fsr"], n0_env=self.cfg.n0_env))
        return dict(fsr=fsr_list, fpk=fpk_list, clarity=clarity_list,
                    d0_list=d0_list, d0=float(np.median(d0_list)),
                    delta_n0=self.cfg.delta_n0, delta_k0=self.cfg.delta_k0)

    # ---------- 参数打包/解包 ----------
    def _pack_p0_bounds(self, d0: float, n_datasets: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        p = []
        lo, hi = [], []

        # d
        if self.cfg.tighten_init_bounds:
            dlo = max(self.cfg.d_bounds_um[0], (1.0 - self.cfg.init_bound_frac) * d0)
            dhi = min(self.cfg.d_bounds_um[1], (1.0 + self.cfg.init_bound_frac) * d0)
        else:
            dlo, dhi = self.cfg.d_bounds_um
        p += [d0];
        lo += [dlo];
        hi += [dhi]

        # delta_n
        if self.cfg.fit_delta_n:
            p += [self.cfg.delta_n0]
            lo += [self.cfg.delta_n_bounds[0]]
            hi += [self.cfg.delta_n_bounds[1]]

        # delta_k
        if self.cfg.fit_delta_k:
            p += [self.cfg.delta_k0]
            lo += [self.cfg.delta_k_bounds[0]]
            hi += [self.cfg.delta_k_bounds[1]]

        # per-angle A,B
        for _ in range(n_datasets):
            p += [0.0, 1.0]
            lo += [-1.0, 0.0]
            hi += [1.0, 5.0]

        return np.array(p, dtype=float), (np.array(lo, dtype=float), np.array(hi, dtype=float))

    def _unpack_params(self, params: np.ndarray, n_datasets: int):
        idx = 0
        d_um = float(params[idx]);
        idx += 1
        delta_n = float(params[idx]) if self.cfg.fit_delta_n else 0.0
        if self.cfg.fit_delta_n: idx += 1
        delta_k = float(params[idx]) if self.cfg.fit_delta_k else 0.0
        if self.cfg.fit_delta_k: idx += 1

        AB = []
        for _ in range(n_datasets):
            A = float(params[idx]);
            B = float(params[idx + 1]);
            idx += 2
            AB.append((A, B))
        return d_um, delta_n, delta_k, AB

    # ---------- 残差 ----------
    def _residual_joint(self, params: np.ndarray, datasets: List[Dataset], nu_list, ydet_list) -> np.ndarray:
        d_um, delta_n, delta_k, AB = self._unpack_params(params, len(datasets))

        def n1_eff(lam):
            return self.cfg.n_base_of_lam(lam) + (delta_n + 1j * delta_k)

        res_all = []
        for (ds, (A, B), nu, y) in zip(datasets, AB, nu_list, ydet_list):
            lam = wavelength_um_from_wavenumber_cm1(nu)
            Rmod = self.model.reflectance(lam, math.radians(ds.theta_deg), n1_eff, self.cfg.n_sub_of_lam, d_um)
            Rm = float(np.mean(Rmod))
            yhat = A + B * (Rmod - Rm)
            res_all.append(yhat - y)
        return np.concatenate(res_all)

    # ---------- 拟合主过程 ----------
    def fit(self, datasets: List[Dataset]) -> Dict[str, Any]:
        nu_list, ydet_list, yfft_list = self._prep_all(datasets)
        theta_list = [ds.theta_deg for ds in datasets]
        init = self._initials(nu_list, yfft_list, theta_list)

        p0, bounds = self._pack_p0_bounds(init["d0"], len(datasets))
        f_scale = self.cfg.f_scale or float(np.median([robust_mad(y) for y in ydet_list]))

        # Round 1: 收紧 d 边界
        r1 = least_squares(self._residual_joint, p0, args=(datasets, nu_list, ydet_list),
                           bounds=bounds, loss=self.cfg.loss, f_scale=f_scale, max_nfev=self.cfg.max_nfev)

        # Round 2: 放宽 d 边界
        p1 = r1.x.copy()
        lo2, hi2 = list(bounds[0]), list(bounds[1])
        lo2[0] = self.cfg.d_bounds_um[0];
        hi2[0] = self.cfg.d_bounds_um[1]
        r2 = least_squares(self._residual_joint, p1, args=(datasets, nu_list, ydet_list),
                           bounds=(np.array(lo2), np.array(hi2)), loss=self.cfg.loss,
                           f_scale=f_scale, max_nfev=self.cfg.max_nfev)

        d_um, delta_n, delta_k, AB = self._unpack_params(r2.x, len(datasets))
        resid = self._residual_joint(r2.x, datasets, nu_list, ydet_list)
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        per_angle = {}
        for ds, (A, B) in zip(datasets, AB):
            per_angle[ds.name] = dict(A=A, B=B)

        # 单角固定 d, Δn, Δk 复查
        d_consistency = []
        for ds, nu, y in zip(datasets, nu_list, ydet_list):
            lam = wavelength_um_from_wavenumber_cm1(nu)

            def n1_eff(lam_in): return self.cfg.n_base_of_lam(lam_in) + (delta_n + 1j * delta_k)

            Rmod = self.model.reflectance(lam, math.radians(ds.theta_deg), n1_eff, self.cfg.n_sub_of_lam, d_um)
            Rm = float(np.mean(Rmod))

            def res_ab(p):
                A, B = p
                return (A + B * (Rmod - Rm)) - y

            r_ab = least_squares(res_ab, x0=np.array([0.0, 1.0]),
                                 bounds=([-1.0, 0.0], [1.0, 5.0]),
                                 loss=self.cfg.loss, f_scale=f_scale, max_nfev=20000)
            d_consistency.append(dict(name=ds.name, rmse=float(np.sqrt(np.mean(res_ab(r_ab.x) ** 2))),
                                      A=float(r_ab.x[0]), B=float(r_ab.x[1])))

        return dict(
            success=bool(r2.success),
            message=str(r2.message),
            d_um=d_um,
            delta_n=delta_n,
            delta_k=delta_k,
            rmse=rmse,
            init=init,
            per_angle=per_angle,
            d_consistency=d_consistency,
            nfev=int(r1.nfev + r2.nfev),
        )

    # ---------- Bootstrap 置信区间 ----------
    def bootstrap_ci(self, datasets: List[Dataset], n_boot: int = 200, keep_ratio: float = 0.7, seed: int = 0) -> Dict[
        str, Any]:
        rng = np.random.default_rng(seed)
        nu_list, ydet_list, _ = self._prep_all(datasets)
        d_vals, dn_vals, dk_vals = [], [], []
        for _ in range(n_boot):
            sub_sets = []
            for ds, nu, y in zip(datasets, nu_list, ydet_list):
                n = len(nu);
                m = max(16, int(round(n * keep_ratio)))
                start = rng.integers(low=0, high=max(n - m, 1))
                idx = slice(start, start + m)
                sub_df = pd.DataFrame({"波数 (cm-1)": nu[idx], "反射率 (%)": (y[idx] + np.mean(y)) * 100.0})
                sub_sets.append(Dataset(name=ds.name, df=sub_df, theta_deg=ds.theta_deg))
            try:
                res = self.fit(sub_sets)
                if res["success"]:
                    d_vals.append(res["d_um"]);
                    dn_vals.append(res["delta_n"]);
                    dk_vals.append(res["delta_k"])
            except Exception:
                pass

        def stats(a):
            a = np.asarray(a, dtype=float)
            return dict(mean=float(np.mean(a)), std=float(np.std(a, ddof=1)),
                        q05=float(np.quantile(a, 0.05)), q50=float(np.quantile(a, 0.50)),
                        q95=float(np.quantile(a, 0.95)))

        if len(d_vals) == 0:
            return dict(success=False, note="no valid bootstrap samples")
        return dict(success=True, n=len(d_vals),
                    d=stats(d_vals), delta_n=stats(dn_vals), delta_k=stats(dk_vals))


# =========================
# 可视化（拟合曲线）
# =========================
def plot_fit(datasets: List[Dataset], fitter: ThicknessFitterExtended, fit_res: Dict[str, Any]):
    pre = fitter.pre
    d, dn, dk = fit_res["d_um"], fit_res["delta_n"], fit_res["delta_k"]

    def n1_eff(lam):
        return fitter.cfg.n_base_of_lam(lam) + (dn + 1j * dk)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(9, 3.2 * len(datasets)), sharex=False)
    if len(datasets) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        p = pre.run(ds.df)
        nu, y = p["nu"], p["y_det"]
        lam = wavelength_um_from_wavenumber_cm1(nu)
        Rmod = fitter.model.reflectance(lam, math.radians(ds.theta_deg), n1_eff, fitter.cfg.n_sub_of_lam, d)
        Rm = float(np.mean(Rmod))
        A = fit_res["per_angle"][ds.name]["A"];
        B = fit_res["per_angle"][ds.name]["B"]
        yhat = A + B * (Rmod - Rm)
        ax.plot(nu, y, lw=1.0, label="实测(去趋势)")
        ax.plot(nu, yhat, lw=1.0, ls="--", label=f"Airy 拟合  d={d:.3f} μm, Δn={dn:.4f}, Δκ={dk:.4f}")
        ax.set_ylabel("相对幅值");
        ax.grid(alpha=0.3);
        ax.legend(loc="best")
        ax.set_title(f"{ds.name}（θ={ds.theta_deg:.1f}°）")
    axes[-1].set_xlabel("波数 (cm$^{-1}$)")
    fig.tight_layout();
    return fig


# =========================
# —— 使用示例 ——
# df3/df4 若已在外部加载，下面会直接使用；否则会生成演示数据。
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

    datasets = [
        Dataset(name="Si 附件3", df=df3, theta_deg=0.0),
        Dataset(name="Si 附件4", df=df4, theta_deg=0.0),
    ]

    fit_cfg = FitConfig(
        n_base_of_lam=n_si_um,  # 外延层基准折射率
        n_sub_of_lam=n_si_um,  # 衬底折射率
        n0_env=1.0,
        pre=PreprocessCfg(detrend=True, sg_window_frac=0.12, sg_polyorder=2,
                          normalize_for_fft=True, nu_min=1200.0, nu_max=3800.0),
        loss="cauchy", f_scale=None, max_nfev=60000,
        fit_delta_n=True, fit_delta_k=True,
        tighten_init_bounds=True, init_bound_frac=0.15,
        d_bounds_um=(1e-3, 1e6),
        delta_n_bounds=(-0.05, 0.05),  # 可按样品再收紧
        delta_k_bounds=(0.0, 0.05),
        delta_n0=0.0, delta_k0=0.0
    )

    fitter = ThicknessFitterExtended(fit_cfg)
    res = fitter.fit(datasets)

    print("\n=== 多角联合拟合（扩展） ===")
    print(f"success={res['success']}, message={res['message']}")
    print(
        f"d = {res['d_um']:.4f} μm,  Δn = {res['delta_n']:.5f},  Δκ = {res['delta_k']:.5f},  RMSE = {res['rmse']:.6f},  nfev={res['nfev']}")
    print("初值诊断：")
    print("  FSR(cm^-1) =", ["%.3f" % x for x in res['init']['fsr']])
    print("  d0_list(μm)=", ["%.4f" % x for x in res['init']['d0_list']])
    print("每角标定参数：")
    for name, ab in res["per_angle"].items():
        print(f"  {name}: A={ab['A']:.4f}, B={ab['B']:.4f}")
    for item in res["d_consistency"]:
        print(f"  [单角固定d,Δn,Δκ] {item['name']}: rmse={item['rmse']:.6f}, A={item['A']:.4f}, B={item['B']:.4f}")

    fig = plot_fit(datasets, fitter, res)
    if fig: plt.show()

    # 先只拟合 df3
    datasets = [Dataset(name="Si 附件3", df=df3, theta_deg=0.0)]
    res3 = ThicknessFitterExtended(fit_cfg).fit(datasets)
    print("\n--- 附件3 单谱 ---")
    print(f"d3={res3['d_um']:.4f} μm, Δn={res3['delta_n']:.5f}, Δκ={res3['delta_k']:.5f}, RMSE={res3['rmse']:.6f}")

    # 只拟合 df4
    datasets = [Dataset(name="Si 附件4", df=df4, theta_deg=0.0)]
    res4 = ThicknessFitterExtended(fit_cfg).fit(datasets)
    print("\n--- 附件4 单谱 ---")
    print(f"d4={res4['d_um']:.4f} μm, Δn={res4['delta_n']:.5f}, Δκ={res4['delta_k']:.5f}, RMSE={res4['rmse']:.6f}")
