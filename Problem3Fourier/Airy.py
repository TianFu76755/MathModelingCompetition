# Airy amplitude-model fitter for a single film on a substrate.
# It plugs into your existing FFT pipeline by accepting (nu_u, R_meas)
# and an FFT-derived initial guess d0. It fits thickness d plus a linear
# scale/offset (a, b) and returns figures: modeled spectrum and residuals.
#
# Assumptions:
# - nu_u is uniformly spaced wavenumber in cm^-1
# - R_meas is the (baseline-corrected) reflectance-like signal on the same grid
# - n1, n2 can be constants (float) or callables n1(wn), n2(wn) returning complex n
# - Unpolarized light: average of s and p
#
# No external deps beyond numpy/matplotlib.
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from Problem3Fourier.UseLess.yuchuli_fft_viz import preprocess_and_plot_compare


def _as_func(n):
    """Allow n to be float/complex or callable n(wn)->complex."""
    if callable(n):
        return n
    else:
        return lambda wn: np.asarray(n, dtype=complex) + 0*wn

def fresnel_r_s(n_i, th_i, n_j, th_j):
    """Fresnel reflection coefficient (s-pol) for complex indices/angles."""
    return (n_i*np.cos(th_i) - n_j*np.cos(th_j)) / (n_i*np.cos(th_i) + n_j*np.cos(th_j))

def fresnel_r_p(n_i, th_i, n_j, th_j):
    """Fresnel reflection coefficient (p-pol) for complex indices/angles."""
    return (n_j*np.cos(th_i) - n_i*np.cos(th_j)) / (n_j*np.cos(th_i) + n_i*np.cos(th_j))

def snell_theta(n_i, th_i, n_j):
    """Return transmitted angle in medium j (complex allowed)."""
    # n_i sin th_i = n_j sin th_j  -> sin th_j = (n_i/n_j) sin th_i
    s = (n_i/ n_j) * np.sin(th_i)
    # handle complex arcsin robustly
    return np.arcsin(s)

def airy_single_layer_reflectance(
    wn,                 # wavenumber array (cm^-1)
    d_cm,               # thickness in cm
    n0=1.0,             # incident medium (e.g., air)
    n1=3.50,            # film index (can be callable -> complex index vs wn)
    n2=3.42,            # substrate index (could be callable)
    theta0_deg=10.0,    # external incidence (deg)
):
    """
    Return modeled reflectance R(wn) using Airy amplitude formula for a single layer on substrate.
    Works for s/p and averages for unpolarized light.
    """
    wn = np.asarray(wn, float)
    lam_cm = 1.0/wn  # wavelength in cm (since wn in cm^-1)
    k0 = 2*np.pi/lam_cm

    n0f = _as_func(n0)
    n1f = _as_func(n1)
    n2f = _as_func(n2)

    th0 = np.deg2rad(theta0_deg) + 0j
    n0c = n0f(wn)
    n1c = n1f(wn)
    n2c = n2f(wn)

    # angles in each medium (complex allowed)
    th1 = snell_theta(n0c, th0, n1c)
    th2 = snell_theta(n1c, th1, n2c)

    # Fresnel at interfaces
    r01s = fresnel_r_s(n0c, th0, n1c, th1)
    r12s = fresnel_r_s(n1c, th1, n2c, th2)
    r01p = fresnel_r_p(n0c, th0, n1c, th1)
    r12p = fresnel_r_p(n1c, th1, n2c, th2)

    # phase thickness (two passes inside the film)
    beta = k0 * n1c * np.cos(th1) * (2*d_cm)  # 2*n1*d*cos(th1) * (2π/λ)

    # Airy amplitude reflection for s/p
    exp2ib = np.exp(1j*beta)
    rs = (r01s + r12s*exp2ib) / (1.0 + r01s*r12s*exp2ib)
    rp = (r01p + r12p*exp2ib) / (1.0 + r01p*r12p*exp2ib)

    R = 0.5*(np.abs(rs)**2 + np.abs(rp)**2)  # unpolarized
    return R.real

def _solve_linear_ab(y, m):
    """Given data y and model m, find a,b minimizing || y - (a*m + b) ||^2."""
    M = np.vstack([m, np.ones_like(m)]).T
    # normal equation solution
    x, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = x[0], x[1]
    return float(a), float(b)

def fit_airy_single_layer(
    nu_u, R_meas, d0_um,
    n0=1.0, n1=3.50, n2=3.42, theta0_deg=10.0,
    search_span_um=40.0,  # +/- span around d0 for initial coarse search (in μm)
    coarse_N=1201,        # coarse grid points for 1D search
    refine_iters=3        # number of zoom-in refinements
):
    """
    Fit thickness d (and linear scale/offset a,b) to measured spectrum using Airy amplitude model.
    Returns dict with best-fit d, a, b, model, residual, and figures.
    """
    nu_u = np.asarray(nu_u, float)
    R_meas = np.asarray(R_meas, float)

    # convert μm -> cm
    d0_cm = d0_um * 1e-4
    span_cm = search_span_um * 1e-4

    # 1D golden-section-like zoom search on d; for each d, solve a,b in closed form
    def objective(d_cm):
        Rm = airy_single_layer_reflectance(nu_u, d_cm, n0, n1, n2, theta0_deg)
        a,b = _solve_linear_ab(R_meas, Rm)
        resid = R_meas - (a*Rm + b)
        return float(np.mean(resid**2)), a, b

    # coarse grid
    left = d0_cm - span_cm
    right = d0_cm + span_cm
    for _ in range(refine_iters):
        ds = np.linspace(left, right, coarse_N)
        vals = np.array([objective(d)[0] for d in ds])
        k = int(np.argmin(vals))
        # zoom-in window around best index
        k0 = max(0, k-5); k1 = min(coarse_N-1, k+5)
        left, right = ds[k0], ds[k1]
        # shrink grid for next iteration
        coarse_N = max(401, coarse_N//3)

    # final evaluation at best d
    d_best = 0.5*(left+right)
    chi2, a_best, b_best = objective(d_best)

    R_model = airy_single_layer_reflectance(nu_u, d_best, n0, n1, n2, theta0_deg)
    y_fit = a_best*R_model + b_best
    resid = R_meas - y_fit

    out = {
        "d_um": d_best*1e4,
        "a": a_best, "b": b_best, "chi2": chi2,
        "nu_u": nu_u, "R_meas": R_meas,
        "R_airymodel": R_model, "R_fit": y_fit, "residual": resid
    }
    return out

def plot_fit_and_residual(out, title="Airy fit (single layer on substrate)"):
    nu = out["nu_u"]; Rm = out["R_meas"]; yfit = out["R_fit"]
    resid = out["residual"]; d_um = out["d_um"]
    fig, ax = plt.subplots(2, 1, figsize=(9,6), sharex=True,
                           gridspec_kw={"height_ratios":[3,1]})
    ax[0].plot(nu, Rm, lw=1.5, label="预处理数据")
    ax[0].plot(nu, yfit, lw=1.6, label=f"Airy拟合 (d = {-d_um:.4g} μm)")
    ax[0].set_ylabel("信号 / 反射率 (a.u.)")
    # ax[0].set_title(title)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(nu, resid, lw=1.0, label="Residual")
    ax[1].axhline(0, color="k", lw=0.8, alpha=0.5)
    ax[1].set_xlabel("波数 (cm$^{-1}$)")
    ax[1].set_ylabel("残差")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    return fig

# ---- Example usage (commented) ----
# Suppose you already have nu_u and a preprocessed reflective-like signal y_uniform (before window) or your original R after baseline correction.
# Use the FFT thickness as d0_um (μm). Then call:
#
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    df = df3
    n = 3.50
    theta_deg = 10.0
    include_range: Tuple[float, float] = (1800, 2500)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）

    # ============预处理阶段===========
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,  # 条纹最明显波段
        exclude_ranges=exclude_ranges,  # 强吸收段（可多段）
        tukey_alpha=0.5,  # Tukey 窗参数；设 0 关闭
        baseline_poly_deg=3,  # 基线多项式阶数
        uniform_points=None,  # 等间距采样点数（默认跟随数据）
        window_name="tukey",  # "tukey" / "hann" / "rect"
        show_windowed=True,  # 是否同时画“乘窗后”的曲线
    )
    nu_u = out["nu_uniform"]
    # y_u = out["y_uniform_demean"]
    y_w = out["y_windowed"]

    d0_um = 3.5   # from your FFT main peak, for example
    out = fit_airy_single_layer(nu_u, y_w, d0_um, n0=1.0, n1=n, n2=n+0.05, theta0_deg=10.0)
    _ = plot_fit_and_residual(out, title="Airy Fit (n1=3.50 on n2=2.55, θ=10°)")


