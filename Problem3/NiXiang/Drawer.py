# Forward model for Si/SiC epitaxial layer: build a clean, reusable module
# - Dataclass settings for dataset & preprocessing
# - Physics class (single-layer on substrate, Airy/TMM-equivalent)
# - Forward comparator: given thickness d, compute model curve and plot Measured vs Fit
#
# Notes:
# * Units: input DataFrame columns: ["波数 (cm-1)", "反射率 (%)"]
# * Refractive index callable n(λ): λ in micrometers, returns real or complex array
# * Angle in degrees (external, in medium 0). Unpolarized assumed (avg s/p).
#
# This cell creates no figures until you call `demo()` at the end.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
EPS = 1e-12

def lam_um_from_nu_cm1(nu_cm1: np.ndarray) -> np.ndarray:
    return 1e4 / np.maximum(nu_cm1, EPS)

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + EPS

# -----------------------------
# Refractive index models (examples)
# -----------------------------
def n_si_um(lam_um: np.ndarray) -> np.ndarray:
    """
    Silicon dispersion provided by user:
    n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814*λ^2, λ in μm.
    Returns real n (no absorption term here). You can extend to complex if needed.
    """
    lam2 = np.asarray(lam_um, dtype=float)**2
    denom = np.maximum(lam2 - 0.0086024, 1e-9)
    n2 = 11.6858 + 0.939816/denom + 0.00089814*lam2
    n = np.sqrt(np.maximum(n2, 0.0))
    return n

def n_sic_um(lam_um: np.ndarray) -> np.ndarray:
    """
    Example 4H-SiC ordinary-ray like simple dispersion (placeholder).
    If you already have a more accurate formula, swap it in.
    Here we use a gentle Cauchy-like form for demonstration.
    """
    lam = np.asarray(lam_um, dtype=float)
    # Simple smooth model around ~2.6 in mid-IR, weak dispersion
    n = 2.55 + 0.02/(np.maximum(lam, 0.2)**2)
    return n

# -----------------------------
# Dataclass settings
# -----------------------------
@dataclass
class PreprocessCfg:
    detrend: bool = True
    sg_window_frac: float = 0.12
    sg_polyorder: int = 2
    normalize_for_compare: bool = True   # z-score both signals before comparing/plotting (prevents collapse to y=k)

@dataclass
class DatasetCfg:
    name: str
    n_layer: Callable[[np.ndarray], np.ndarray]     # N1(λ) for epitaxial layer
    n_sub: Callable[[np.ndarray], np.ndarray]       # N2(λ) for substrate
    n_env: complex | float = 1.0                    # N0
    theta_deg: float = 0.0                          # external angle in degrees
    nu_min: Optional[float] = None                  # wavenumber window [cm^-1]
    nu_max: Optional[float] = None

# -----------------------------
# Preprocessor
# -----------------------------
class Preprocessor:
    def __init__(self, cfg: PreprocessCfg):
        self.cfg = cfg

    def _uniform_grid(self, nu: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(nu)
        nu, R = nu[idx], R[idx]
        dnu = np.mean(np.diff(nu))
        if not np.isfinite(dnu) or dnu <= 0:
            return nu, R
        nu_g = np.arange(nu[0], nu[-1] + 0.5*dnu, dnu)
        R_g = np.interp(nu_g, nu, R)
        return nu_g, R_g

    def _window(self, nu: np.ndarray, y: np.ndarray, nu_min: Optional[float], nu_max: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        lo = -np.inf if nu_min is None else nu_min
        hi =  np.inf if nu_max is None else nu_max
        m = (nu >= lo) & (nu <= hi)
        return nu[m], y[m]

    def _savgol_baseline(self, y: np.ndarray, frac: float, poly: int) -> np.ndarray:
        n = len(y)
        if n < 11:
            return np.full_like(y, np.mean(y))
        # choose odd window
        w = max(5, int(round(frac * n)))
        if w % 2 == 0:
            w += 1
        w = min(w, n-1 if (n-1)%2==1 else n-2)
        w = max(w, poly+3)
        # implement simple Savitzky-Golay via numpy polyfit on sliding windows (fallback)
        # For robustness without scipy, do a light moving-average baseline instead:
        # We'll use a convolution as a smooth baseline here to avoid extra deps.
        k = w
        kernel = np.ones(k)/k
        # pad reflectively
        pad = k//2
        ypad = np.pad(y, (pad, pad), mode='edge')
        base = np.convolve(ypad, kernel, mode='valid')
        return base

    def run(self, df: pd.DataFrame, ds_cfg: DatasetCfg) -> Dict[str, np.ndarray]:
        col_nu = "波数 (cm-1)"
        col_R  = "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            col_nu, col_R = df.columns[:2]
        nu = df[col_nu].to_numpy(float)
        R  = (df[col_R].to_numpy(float)) / 100.0

        nu_g, R_g = self._uniform_grid(nu, R)
        nu_w, R_w = self._window(nu_g, R_g, ds_cfg.nu_min, ds_cfg.nu_max)

        if self.cfg.detrend:
            base = self._savgol_baseline(R_w, self.cfg.sg_window_frac, self.cfg.sg_polyorder)
            y_meas = R_w - base
        else:
            y_meas = R_w - np.mean(R_w)

        # Normalize for comparison (z-score)
        if self.cfg.normalize_for_compare:
            z = y_meas - np.mean(y_meas)
            std = np.std(z) + EPS
            y_meas = z / std

        return dict(nu=nu_w, R_raw=R_w, y_meas=y_meas)

# -----------------------------
# Physics: single-layer on substrate (Airy/TMM-equivalent)
# -----------------------------
class SingleLayerReflectance:
    """
    Forward model: unpolarized reflectance of air / layer(N1,d) / substrate(N2)
    """
    def __init__(self, n_layer: Callable[[np.ndarray], np.ndarray],
                 n_sub: Callable[[np.ndarray], np.ndarray],
                 n_env: complex | float = 1.0):
        self.n_layer = n_layer
        self.n_sub = n_sub
        self.n_env = n_env

    @staticmethod
    def _cos_theta_inside(N_in: complex | float, N_out: np.ndarray, cos_theta_in: float) -> np.ndarray:
        """
        Given incident from medium with index N_in (can be complex) at angle theta_in with cos=cos_theta_in,
        compute cos(theta_out) in medium with index N_out using generalized Snell: N_in*sinθ_in = N_out*sinθ_out.
        cosθ_out = sqrt(1 - (N_in/N_out)^2 * (1 - cos^2θ_in)).
        """
        ratio2 = (N_in / N_out)**2
        sin2_in = (1.0 - cos_theta_in**2)
        val = 1.0 - ratio2 * sin2_in
        # Complex-safe sqrt
        return np.sqrt(val + 0j)

    @staticmethod
    def _fresnel_rs(Ni, Nj, cos_ti, cos_tj):
        return (Ni*cos_ti - Nj*cos_tj) / (Ni*cos_ti + Nj*cos_tj + 0j)

    @staticmethod
    def _fresnel_rp(Ni, Nj, cos_ti, cos_tj):
        return (Nj*cos_ti - Ni*cos_tj) / (Nj*cos_ti + Ni*cos_tj + 0j)

    def reflectance_unpolarized(self, lam_um: np.ndarray, d_um: float, theta0_deg: float) -> np.ndarray:
        lam = np.asarray(lam_um, dtype=float)
        N0 = complex(self.n_env)
        N1 = np.asarray(self.n_layer(lam), dtype=complex)
        N2 = np.asarray(self.n_sub(lam), dtype=complex)

        theta0 = math.radians(theta0_deg)
        cos0 = math.cos(theta0)

        # inside cosines
        cos1 = self._cos_theta_inside(N0, N1, cos0)
        cos2 = self._cos_theta_inside(N0, N2, cos0)

        # Fresnel at interfaces 0|1 and 1|2 for s and p
        r01_s = self._fresnel_rs(N0, N1, cos0, cos1)
        r12_s = self._fresnel_rs(N1, N2, cos1, cos2)
        r01_p = self._fresnel_rp(N0, N1, cos0, cos1)
        r12_p = self._fresnel_rp(N1, N2, cos1, cos2)

        # phase thickness
        d_cm = d_um * 1e-4
        beta = 2.0 * np.pi * (N1 * d_cm * cos1) / (lam * 1e-4)  # lam[μm] -> cm

        e2i = np.exp(2j * beta)

        # total reflection coefficient for s and p
        rs = (r01_s + r12_s * e2i) / (1.0 + r01_s * r12_s * e2i + 0j)
        rp = (r01_p + r12_p * e2i) / (1.0 + r01_p * r12_p * e2i + 0j)

        R = 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)
        return np.real_if_close(R)

# -----------------------------
# Forward comparator (Measured vs Fit)
# -----------------------------
class ForwardComparator:
    def __init__(self, ds_cfg: DatasetCfg, pre_cfg: PreprocessCfg):
        self.ds_cfg = ds_cfg
        self.pre = Preprocessor(pre_cfg)
        self.model = SingleLayerReflectance(ds_cfg.n_layer, ds_cfg.n_sub, ds_cfg.n_env)

    def compare(self, df: pd.DataFrame, d_um: float, show_residual: bool = True) -> Dict[str, Any]:
        # preprocess measured
        prep = self.pre.run(df, self.ds_cfg)
        nu = prep["nu"]
        lam = lam_um_from_nu_cm1(nu)

        # forward model on same grid
        R_model = self.model.reflectance_unpolarized(lam, d_um, self.ds_cfg.theta_deg)

        # apply same preprocessing to model (detrend + z-score if requested)
        # Reuse preprocessor methods in a minimal way
        # Here re-create baseline and normalization to mirror measured pipeline:
        # We'll re-run simple moving-average baseline with same parameters.
        # (Using internal method for consistency)
        base_mod = self.pre._savgol_baseline(R_model, self.pre.cfg.sg_window_frac, self.pre.cfg.sg_polyorder) if self.pre.cfg.detrend else np.mean(R_model)
        y_model = R_model - base_mod if self.pre.cfg.detrend else R_model - np.mean(R_model)
        if self.pre.cfg.normalize_for_compare:
            z = y_model - np.mean(y_model)
            std = np.std(z) + EPS
            y_model = z / std

        y_meas = prep["y_meas"]

        # Optional linear calibration to overlay (solve a,b in least squares on the already z-scored or detrended data)
        A = np.vstack([np.ones_like(y_model), y_model]).T
        sol, *_ = np.linalg.lstsq(A, y_meas, rcond=None)
        a_cal, b_cal = sol
        y_fit = a_cal + b_cal * y_model

        # Correlation score (shape similarity)
        num = float(np.dot(y_meas - np.mean(y_meas), y_model - np.mean(y_model)))
        den = float(np.linalg.norm(y_meas - np.mean(y_meas)) * np.linalg.norm(y_model - np.mean(y_model)) + EPS)
        ncc = num / den

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.6))
        ax.plot(nu, y_meas, lw=1.0, label="Measured (preprocessed)")
        ax.plot(nu, y_fit, lw=1.0, ls="--", label=f"Fit from model @ d={d_um:.3f} μm")
        ax.set_xlabel("波数 (cm$^{-1}$)")
        ax.set_ylabel("相对幅值")
        ax.set_title(f"{self.ds_cfg.name} — Measured vs Fit (θ={self.ds_cfg.theta_deg:.1f}°), NCC={ncc:.3f}")
        ax.grid(alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()

        fig_res = None
        if show_residual:
            fig_res, axr = plt.subplots(1, 1, figsize=(9, 2.4))
            axr.plot(nu, y_meas - y_fit, lw=1.0)
            axr.set_xlabel("波数 (cm$^{-1}$)")
            axr.set_ylabel("残差")
            axr.set_title("Residual (Measured - Fit)")
            axr.grid(alpha=0.3)
            fig_res.tight_layout()

        return dict(
            nu=nu, lam_um=lam,
            R_model=R_model,
            y_meas=y_meas, y_model=y_model, y_fit=y_fit,
            ncc=ncc, a=a_cal, b=b_cal,
            fig=fig, fig_residual=fig_res
        )


if __name__=="__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 假设你已有 df3, df4（Si），df1, df2（SiC）
    ds_si = DatasetCfg(name="Si 附件3", n_layer=n_si_um, n_sub=n_si_um, theta_deg=0.0, nu_min=1200, nu_max=3800)
    pre   = PreprocessCfg(detrend=True, normalize_for_compare=True)
    cmptr = ForwardComparator(ds_si, pre)

    out = cmptr.compare(df3, d_um=15.0, show_residual=True)  # 这里的 8.0 μm 是你要测试的厚度
    plt.show()
    # out['fig'], out['fig_residual'] 即为图；out['ncc'] 是形状相似度评分
