# -*- coding: utf-8 -*-
"""
问题3：硅外延层厚度的计算（多光束/Airy/TMM + 多角联合拟合，Python 3.8）

依赖：numpy, pandas, scipy, matplotlib(仅可选画图)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares


# ========================
# 0) 通用小工具
# ========================

EPS = 1e-12

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

def wavelength_um_from_wavenumber_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    """λ[μm] = 1e4 / ν[cm^-1]"""
    return 1e4 / np.maximum(nu_cm1, EPS)

def complex_sqrt(z: np.ndarray) -> np.ndarray:
    # numpy 的 sqrt 对复数是主值；这里封装以便需要时替换
    return np.sqrt(z + 0j)

# ========================
# 1) 预处理
# ========================

@dataclass
class PreprocessConfig:
    detrend: bool = True
    sg_window_frac: float = 0.12
    sg_polyorder: int = 2
    normalize_for_fft: bool = True
    nu_min: Optional[float] = 1200.0
    nu_max: Optional[float] = 3800.0

class Preprocessor:
    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg = cfg or PreprocessConfig()

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
        hi =  np.inf if self.cfg.nu_max is None else self.cfg.nu_max
        m = (nu >= lo) & (nu <= hi)
        return nu[m], y[m]

    def _detrend(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.detrend or len(y) < 11:
            return y - float(np.mean(y))
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
        return dict(nu=nu_w, R=R_w, y_det=y_det, y_fft=y_fft, nu_full=nu_g, R_full=R_g)


# ========================
# 2) 折射率（硅）
# ========================

class SiDispersion:
    """
    硅的色散： n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814*λ^2
    输入 λ[μm]，返回 n(λ)（实数）。如需考虑吸收，可扩展为复折射率。
    """
    @staticmethod
    def n_of_lambda_um(lam_um: np.ndarray) -> np.ndarray:
        lam_um = np.asarray(lam_um, dtype=float)
        lam2 = lam_um**2
        denom = np.maximum(lam2 - 0.0086024, 1e-9)
        n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
        return np.sqrt(np.maximum(n2, 1e-12))


# ========================
# 3) 光学（Fresnel / Airy）
# ========================

class Optics:
    @staticmethod
    def snell_cos_theta(N0: complex, N: np.ndarray, theta0_rad: float) -> np.ndarray:
        s0 = math.sin(theta0_rad)
        # cosθ = sqrt(1 - (N0/N sinθ0)^2)
        u = 1.0 - (N0 * s0 / (N + 0j))**2
        return complex_sqrt(u)

    @staticmethod
    def r_s(Ni: np.ndarray, Nj: np.ndarray, cos_i: np.ndarray, cos_j: np.ndarray) -> np.ndarray:
        return (Ni * cos_i - Nj * cos_j) / (Ni * cos_i + Nj * cos_j + 0j)

    @staticmethod
    def r_p(Ni: np.ndarray, Nj: np.ndarray, cos_i: np.ndarray, cos_j: np.ndarray) -> np.ndarray:
        return (Nj * cos_i - Ni * cos_j) / (Nj * cos_i + Ni * cos_j + 0j)

    @staticmethod
    def airy_reflectance_single_layer(
        lam_um: np.ndarray, theta0_rad: float,
        N0: complex, N1: np.ndarray, N2: np.ndarray, d_um: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 (R_s, R_p, R_unpolarized) for a single film on a substrate:
        r_tot = (r01 + r12 e^{2iβ}) / (1 + r01 r12 e^{2iβ})
        β = 2π/λ * N1 d cosθ1
        """
        lam = lam_um
        cos1 = Optics.snell_cos_theta(N0, N1, theta0_rad)
        cos0 = math.cos(theta0_rad) * np.ones_like(cos1)
        cos2 = Optics.snell_cos_theta(N0, N2, theta0_rad)

        r01_s = Optics.r_s(N0, N1, cos0, cos1)
        r12_s = Optics.r_s(N1, N2, cos1, cos2)
        r01_p = Optics.r_p(N0, N1, cos0, cos1)
        r12_p = Optics.r_p(N1, N2, cos1, cos2)

        beta = 2.0 * np.pi * (N1 * d_um * 1e-4) * cos1 / (lam + 0j)  # 注意 d(cm)=d_um*1e-4

        e2ib = np.exp(2j * beta)
        r_tot_s = (r01_s + r12_s * e2ib) / (1.0 + r01_s * r12_s * e2ib + 0j)
        r_tot_p = (r01_p + r12_p * e2ib) / (1.0 + r01_p * r12_p * e2ib + 0j)

        R_s = np.abs(r_tot_s) ** 2
        R_p = np.abs(r_tot_p) ** 2
        R_u = 0.5 * (R_s + R_p)
        return R_s.real, R_p.real, R_u.real


# ========================
# 4) 系统与模型
# ========================

@dataclass
class FilmSystem:
    # 折射率函数（λ[μm]→n）
    n_layer_of_lambda: callable
    n_sub_of_lambda: callable
    n_env: float = 1.0      # 空气
    # 可选：外延层与衬底的小差异（常数偏移，用于检验）
    delta_n: float = 0.0

@dataclass
class AngleDataset:
    name: str
    df: pd.DataFrame
    theta_deg: float

class MultiAngleReflectanceModel:
    """
    负责：给定参数（d_um, delta_n, per-angle a,b），产生每个角的前向 R_model
    并支持“相对标定”：R_cal = a_k + b_k*(R_model - mean(R_model))
    """
    def __init__(self, system: FilmSystem, preprocess: PreprocessConfig):
        self.sys = system
        self.pre = Preprocessor(preprocess)

    def prepare(self, datasets: List[AngleDataset]) -> Dict[str, Any]:
        prep = []
        for ds in datasets:
            p = self.pre.run(ds.df)
            lam = wavelength_um_from_wavenumber_cm1(p["nu"])
            prep.append(dict(
                name=ds.name, theta_rad=math.radians(ds.theta_deg),
                nu=p["nu"], lam=lam, y_det=p["y_det"], y_fft=p["y_fft"], R_raw=p["R"]
            ))
        return dict(items=prep)

    def forward_one(self, lam_um: np.ndarray, theta_rad: float, d_um: float, delta_n: float) -> np.ndarray:
        n1 = self.sys.n_layer_of_lambda(lam_um) + float(delta_n)
        n2 = self.sys.n_sub_of_lambda(lam_um)
        _, _, Ru = Optics.airy_reflectance_single_layer(
            lam_um, theta_rad, self.sys.n_env, n1, n2, d_um
        )
        return Ru

    def forward_many(self, prep_pack: Dict[str, Any], d_um: float, delta_n: float,
                     calib_params: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        calib_params: [(a1,b1), (a2,b2), ...] 与 items 顺序一致
        """
        out = []
        for i, it in enumerate(prep_pack["items"]):
            Ru = self.forward_one(it["lam"], it["theta_rad"], d_um, delta_n)
            mRu = float(np.mean(Ru))
            a, b = calib_params[i]
            Rcal = a + b * (Ru - mRu)
            out.append(Rcal)
        return out


# ========================
# 5) 拟合器
# ========================

@dataclass
class ThicknessFitConfig:
    use_delta_n: bool = False          # 是否拟合外延-衬底的小折射率差
    loss: str = "cauchy"               # 'linear' | 'cauchy' | 'soft_l1' | 'huber'
    f_scale: Optional[float] = None    # 若 None 自动用 1.5*MAD
    max_nfev: int = 40000
    # 初始化：用 FFT 的 FSR 转换为 d0
    init_from_fft: bool = True
    tighten_factor: float = 0.15       # 第一阶段收紧边界比例 ±15%
    do_bootstrap: bool = False
    n_boot: int = 200
    keep_ratio: float = 0.75           # bootstrap 采样比例（连续窗口）

class ThicknessFitter:
    def __init__(self, model: MultiAngleReflectanceModel, cfg: ThicknessFitConfig):
        self.model = model
        self.cfg = cfg

    # ---------- FFT 初值 ----------
    def _init_d_from_fft(self, prep_pack: Dict[str, Any]) -> float:
        fpeaks = []
        for it in prep_pack["items"]:
            nu = it["nu"]; y = it["y_fft"]
            dnu = float(np.mean(np.diff(nu)))
            Y = rfft(y - float(np.mean(y)))
            freqs = rfftfreq(len(y), d=dnu)
            span = float(nu[-1] - nu[0] + EPS)
            low_cut = (1.0 / span) * 3.0
            m = freqs > low_cut
            amp = np.abs(Y)[m]; fpos = freqs[m]
            if len(amp) == 0:
                continue
            fpeaks.append(float(fpos[int(np.argmax(amp))]))
        if len(fpeaks) == 0:
            # 兜底初值
            return 5.0
        fpk = float(np.mean(fpeaks))
        # Δν = 1/fpk； d ≈ 1/(2 n cosθ1 Δν)
        # 取谱段中位 n 与角度中位 θ
        # 用外延层 n(λ) 中位
        all_lam = np.concatenate([it["lam"] for it in prep_pack["items"]])
        n_med = float(np.median(self.model.sys.n_layer_of_lambda(all_lam)))
        th_med = float(np.median([it["theta_rad"] for it in prep_pack["items"]]))
        # cosθ1 用 Snell 近似（实数）：cosθ1≈sqrt(1-(sinθ0/n)^2)
        ct = math.sqrt(max(0.0, 1.0 - (math.sin(th_med) / max(n_med, 1e-6))**2))
        d0_um = (fpk / (2.0 * n_med * max(ct, 1e-6))) * 1e4
        return d0_um

    # ---------- 残差 ----------
    def _residual(self, x: np.ndarray, prep_pack: Dict[str, Any], n_angles: int) -> np.ndarray:
        """
        x 布局：
          use_delta_n=False: [d, a1,b1, a2,b2, ...]
          use_delta_n=True : [d, delta_n, a1,b1, a2,b2, ...]
        """
        off = 0
        d_um = float(x[off]); off += 1
        if self.cfg.use_delta_n:
            delta_n = float(x[off]); off += 1
        else:
            delta_n = 0.0
        calib = []
        for k in range(n_angles):
            a, b = float(x[off]), float(x[off+1]); off += 2
            calib.append((a, b))

        y_hat_list = self.model.forward_many(prep_pack, d_um, delta_n, calib)

        res_all = []
        for (it, y_hat) in zip(prep_pack["items"], y_hat_list):
            y = it["y_det"]
            res = y_hat - y
            res_all.append(res)
        return np.concatenate(res_all)

    # ---------- 拟合主流程 ----------
    def fit(self, datasets: List[AngleDataset]) -> Dict[str, Any]:
        prep = self.model.prepare(datasets)
        n_angles = len(prep["items"])

        # 初值
        d0 = self._init_d_from_fft(prep) if self.cfg.init_from_fft else 5.0
        x0 = [d0]
        if self.cfg.use_delta_n:
            x0.append(0.0)
        for _ in range(n_angles):
            # a,b 初值：把去趋势后的均值/幅度大致吸收
            x0 += [0.0, 1.0]
        x0 = np.array(x0, dtype=float)

        # 边界（两阶段）
        # d 的第一阶段 ± tighten_factor
        d_lo_t = (1.0 - self.cfg.tighten_factor) * d0
        d_hi_t = (1.0 + self.cfg.tighten_factor) * d0
        # 宽边界
        d_lo_w, d_hi_w = 1e-3, 1e6

        def run_lsq(bounds_pair):
            lo, hi = bounds_pair
            f_scale = self.cfg.f_scale
            if f_scale is None:
                # 取所有角的 MAD 均值作为尺度
                mad_list = [robust_mad(it["y_det"]) for it in prep["items"]]
                f_scale = 1.5 * float(np.mean(mad_list))
            return least_squares(
                self._residual, x0,
                args=(prep, n_angles),
                bounds=(lo, hi),
                loss=self.cfg.loss, f_scale=f_scale,
                max_nfev=self.cfg.max_nfev
            )

        def make_bounds(d_lo, d_hi):
            lo = [d_lo]
            hi = [d_hi]
            if self.cfg.use_delta_n:
                lo += [-0.01]
                hi += [0.01]  # Δn 极小范围（可按需放宽）
            for _ in range(n_angles):
                lo += [-0.2, 0.2]  # a∈[-0.2,0.2]（去趋势尺度）
                hi += [ 0.2, 5.0]  # b∈[0.2,5.0]（允许幅度缩放）
            return np.array(lo, float), np.array(hi, float)

        # 第一阶段（收紧）
        res1 = run_lsq(make_bounds(d_lo_t, d_hi_t))
        x1 = res1.x.copy()

        # 第二阶段（放宽）
        x0 = x1
        res2 = least_squares(
            self._residual, x0,
            args=(prep, n_angles),
            bounds=make_bounds(d_lo_w, d_hi_w),
            loss=self.cfg.loss,
            f_scale=(self.cfg.f_scale or 1.5*float(np.mean([robust_mad(it["y_det"]) for it in prep["items"]]))),
            max_nfev=self.cfg.max_nfev
        )

        # 结果解析
        x = res2.x
        off = 0
        d_hat = float(x[off]); off += 1
        if self.cfg.use_delta_n:
            delta_n_hat = float(x[off]); off += 1
        else:
            delta_n_hat = 0.0
        calib = []
        for k in range(n_angles):
            a, b = float(x[off]), float(x[off+1]); off += 2
            calib.append((a, b))

        # 拟合后残差/每角单独 d（诊断）
        y_hat_list = self.model.forward_many(prep, d_hat, delta_n_hat, calib)
        resid = []
        for (it, y_hat) in zip(prep["items"], y_hat_list):
            resid.append(y_hat - it["y_det"])
        resid_all = np.concatenate(resid)
        rmse = float(np.sqrt(np.mean(resid_all**2)))

        # 单角再拟合（固定 n, 不做 calib，拟 d,a,b）——用于一致性检查
        d_single = []
        for (it) in prep["items"]:
            def res_single(q):
                d_u, a_s, b_s = q
                Ru = self.model.forward_one(it["lam"], it["theta_rad"], d_u, delta_n_hat)
                Rcal = a_s + b_s * (Ru - float(np.mean(Ru)))
                return Rcal - it["y_det"]
            q0 = np.array([d_hat, 0.0, 1.0], float)
            r = least_squares(res_single, q0, bounds=([1e-3,-0.2,0.2],[1e6,0.2,5.0]),
                              loss=self.cfg.loss, f_scale=robust_mad(it["y_det"]), max_nfev=20000)
            d_single.append(float(r.x[0]))
        d_single = np.array(d_single)
        d_spread_rel = float(np.max(np.abs(d_single - np.mean(d_single))) / (np.mean(d_single)+EPS))

        out = dict(
            d_um=d_hat,
            delta_n=delta_n_hat,
            calib_params=calib,
            rmse=rmse,
            d_single=list(d_single),
            cross_angle_rel_spread=d_spread_rel,
            success=bool(res2.success),
            message=str(res2.message),
            nfev=int(res2.nfev),
            prep=prep
        )

        # Bootstrap（可选）
        if self.cfg.do_bootstrap:
            rng = np.random.default_rng(0)
            d_vals = []
            for _ in range(self.cfg.n_boot):
                items_sub = []
                for it in prep["items"]:
                    n = len(it["nu"])
                    m = max(16, int(round(n * self.cfg.keep_ratio)))
                    start = int(rng.integers(0, max(n-m, 1)))
                    sl = slice(start, start + m)
                    # 构造子df
                    df_sub = pd.DataFrame({
                        "波数 (cm-1)": it["nu"][sl],
                        "反射率 (%)": 100.0 * (it["R_raw"][sl])
                    })
                    items_sub.append(AngleDataset("sub", df_sub, math.degrees(it["theta_rad"])))
                try:
                    sub_res = self.fit(items_sub)  # 递归（注意：这会很慢；可改为简化版）
                    if sub_res.get("success"):
                        d_vals.append(sub_res["d_um"])
                except Exception:
                    pass
            if len(d_vals) > 2:
                d_arr = np.array(d_vals, float)
                out["bootstrap"] = dict(
                    n=len(d_arr),
                    mean=float(np.mean(d_arr)),
                    std=float(np.std(d_arr, ddof=1)),
                    q05=float(np.quantile(d_arr, 0.05)),
                    q50=float(np.quantile(d_arr, 0.50)),
                    q95=float(np.quantile(d_arr, 0.95)),
                )
        return out


# ========================
# 7) 绘图工具
# ========================
import matplotlib.pyplot as plt


class FitPlotter:
    def __init__(self, model: MultiAngleReflectanceModel):
        self.model = model

    def _rebuild_model_series(self, result: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        """用拟合得到的参数重建每个角的模型曲线（去趋势尺度下的“标定后模型”）。"""
        prep = result["prep"]
        d_um = result["d_um"]
        delta_n = result["delta_n"]
        calib = result["calib_params"]
        y_hat_list = self.model.forward_many(prep, d_um, delta_n, calib)
        series = []
        for it, yhat in zip(prep["items"], y_hat_list):
            series.append(dict(
                name=it["name"],
                nu=it["nu"],
                lam=it["lam"],
                y_det=it["y_det"],   # 实测（去趋势）
                y_hat=yhat,          # 模型（标定后，对齐去趋势尺度）
                theta_rad=it["theta_rad"]
            ))
        return series

    @staticmethod
    def _fft_mag(nu: np.ndarray, y: np.ndarray, min_cycles: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
        dnu = float(np.mean(np.diff(nu)))
        Y = np.fft.rfft(y - float(np.mean(y)))
        freqs = np.fft.rfftfreq(len(y), d=dnu)
        # 去掉极低频
        span = float(nu[-1] - nu[0] + 1e-9)
        low_cut = (1.0 / span) * min_cycles
        m = freqs > low_cut
        freqs = freqs[m]; mag = np.abs(Y)[m]
        fpeak = float(freqs[int(np.argmax(mag))]) if len(freqs) else np.nan
        return freqs, mag, fpeak

    def plot_spectra(self, result: Dict[str, Any], figsize: Tuple[int, int]=(11, 3)):
        """每个角：左图 实测(去趋势) vs 模型；右图 残差。"""
        series = self._rebuild_model_series(result)
        nrow = len(series)
        fig, axes = plt.subplots(nrow, 2, figsize=(figsize[0], figsize[1]*nrow), squeeze=False)
        for i, s in enumerate(series):
            axL, axR = axes[i, 0], axes[i, 1]
            # 左：拟合对比
            axL.plot(s["nu"], s["y_det"], lw=1.2, label="实测（去趋势）")
            axL.plot(s["nu"], s["y_hat"], lw=1.0, ls="--", label="模型（标定后）")
            axL.set_title(f"{s['name']}  实测 vs 模型")
            axL.set_xlabel("波数 (cm$^{-1}$)"); axL.set_ylabel("相对幅值")
            axL.grid(alpha=0.3); axL.legend(loc="best")
            # 右：残差
            res = s["y_hat"] - s["y_det"]
            axR.plot(s["nu"], res, lw=1.0, color="tab:gray")
            axR.set_title(f"{s['name']} 残差")
            axR.set_xlabel("波数 (cm$^{-1}$)"); axR.set_ylabel("模型 - 实测")
            axR.grid(alpha=0.3)
        fig.suptitle("多角联合拟合：光谱域对比与残差", y=0.99, fontsize=12)
        fig.tight_layout()
        return fig

    def plot_fft(self, result: Dict[str, Any], figsize: Tuple[int, int]=(10, 2.8)):
        """每个角的 FFT 幅度谱：对比实测与模型，标出各自主峰。"""
        series = self._rebuild_model_series(result)
        nrow = len(series)
        fig, axes = plt.subplots(nrow, 1, figsize=(figsize[0], figsize[1]*nrow), squeeze=False)
        for i, s in enumerate(series):
            ax = axes[i, 0]
            f_m, M_m, fp_m = self._fft_mag(s["nu"], s["y_det"])
            f_h, M_h, fp_h = self._fft_mag(s["nu"], s["y_hat"])
            ax.plot(f_m, M_m, lw=1.0, label=f"实测 FFT（f*={fp_m:.6f}）")
            ax.plot(f_h, M_h, lw=1.0, ls="--", label=f"模型 FFT（f*={fp_h:.6f}）")
            if np.isfinite(fp_m):
                ax.axvline(fp_m, color="tab:blue", ls=":", lw=1.0)
            if np.isfinite(fp_h):
                ax.axvline(fp_h, color="tab:orange", ls=":", lw=1.0)
            ax.set_xlabel("频率  (cycles per cm$^{-1}$)")
            ax.set_ylabel("|FFT|")
            ax.set_title(f"{s['name']} ：FFT 主峰对比")
            ax.grid(alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def plot_overlay_raw(self, result: Dict[str, Any], figsize: Tuple[int, int]=(11, 3)):
        """
        原始反射率（未去趋势） vs 标定后模型（在原尺度上）。
        作用：检查 a,b 标定是否把基线/幅度对齐。
        """
        prep = result["prep"]
        d_um = result["d_um"]; delta_n = result["delta_n"]; calib = result["calib_params"]
        # 生成“原尺度”模型：对 raw R 使用同样 a,b 标定
        y_hat_raw_list = []
        for i, it in enumerate(prep["items"]):
            Ru = self.model.forward_one(it["lam"], it["theta_rad"], d_um, delta_n)
            a, b = calib[i]
            mRu = float(np.mean(Ru))
            y_hat_raw = a + b * (Ru - mRu)
            y_hat_raw_list.append(y_hat_raw)

        nrow = len(prep["items"])
        fig, axes = plt.subplots(nrow, 1, figsize=(figsize[0], figsize[1]*nrow), squeeze=False)
        for i, (it, yhat) in enumerate(zip(prep["items"], y_hat_raw_list)):
            ax = axes[i, 0]
            ax.plot(it["nu"], it["R_raw"], lw=1.0, label="原始反射率（未去趋势）")
            ax.plot(it["nu"], yhat + np.mean(it["R_raw"]) - np.mean(yhat), lw=1.0, ls="--", label="模型（标定后）")
            ax.set_xlabel("波数 (cm$^{-1}$)"); ax.set_ylabel("反射率")
            ax.set_title(f"{it['name']} ：原始尺度的模型对齐检查")
            ax.grid(alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()
        return fig


# ========================
# 6) main：对 df3/df4 进行联合拟合
# ========================
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # —— 系统：空气 / Si外延 / Si衬底 ——（可选 Δn）
    system = FilmSystem(
        n_layer_of_lambda=SiDispersion.n_of_lambda_um,
        n_sub_of_lambda=SiDispersion.n_of_lambda_um,
        n_env=1.0,
        delta_n=0.0
    )

    # —— 多角数据（角度你可以按实际填；这里示例用 0° / 0°）——
    ds3 = AngleDataset(name="Si 附件3", df=df3, theta_deg=0.0)
    ds4 = AngleDataset(name="Si 附件4", df=df4, theta_deg=0.0)
    datasets = [ds3, ds4]

    # —— 模型 + 拟合器 ——
    pre_cfg = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2,
                               normalize_for_fft=True, nu_min=1200.0, nu_max=3800.0)
    model = MultiAngleReflectanceModel(system, pre_cfg)

    fit_cfg = ThicknessFitConfig(
        use_delta_n=True,  # 开启 Δn 拟合！（关键）
        loss="cauchy", f_scale=None,
        max_nfev=40000,
        init_from_fft=True, tighten_factor=0.15,
        do_bootstrap=False, n_boot=200, keep_ratio=0.75
    )

    fitter = ThicknessFitter(model, fit_cfg)
    result = fitter.fit(datasets)

    # —— 结果输出 ——
    print("\n=== 多角联合拟合（Airy/TMM）结果 ===")
    print(f" 厚度 d = {result['d_um']:.4f} μm")
    if fit_cfg.use_delta_n:
        print(f" Δn (外延-衬底) = {result['delta_n']:.5f}")
    print(f" RMSE = {result['rmse']:.5g}, 迭代次数 nfev = {result['nfev']}, success = {result['success']}")
    print(f" 单角厚度 = {[f'{x:.4f}' for x in result['d_single']]} μm, 跨角相对散布 = {100*result['cross_angle_rel_spread']:.2f}%")
    if 'bootstrap' in result:
        b = result['bootstrap']
        print(f" Bootstrap(n={b['n']})：mean={b['mean']:.4f}, std={b['std']:.4f}, 5%-95%=[{b['q05']:.4f}, {b['q95']:.4f}] μm")

    # —— 绘图 —— #
    plotter = FitPlotter(model)
    fig1 = plotter.plot_spectra(result, figsize=(11, 3.2))  # 实测 vs 模型 + 残差
    fig2 = plotter.plot_fft(result, figsize=(10, 2.6))  # FFT 主峰一致性
    fig3 = plotter.plot_overlay_raw(result, figsize=(11, 3.0))  # 原尺度对齐检查

    plt.show()