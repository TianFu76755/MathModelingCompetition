# ir_thickness.py
# 两束干涉测厚（包含色散与入射角自洽）
# 依赖：numpy, pandas (仅作 DataFrame 列读取), math

from typing import Callable, Tuple, Dict
import numpy as np
import pandas as pd
import math


# -----------------------------
# 公共物理类
# -----------------------------
class OpticalModel:
    """
    kappa(ν) = n(λ(ν)) * cos(theta_t(ν)) * ν
    其中 theta_t(ν) 由斯涅尔定律：n0*sin(theta_i)=n(λ)*sin(theta_t)
    """
    def __init__(self, n_of_lambda: Callable[[np.ndarray], np.ndarray], n0: float = 1.0):
        self.n_of_lambda = n_of_lambda
        self.n0 = n0

    @staticmethod
    def lambda_um_from_nu_cm1(nu_cm1: np.ndarray) -> np.ndarray:
        # 波数(cm^-1) -> 波长(μm)
        return 1e4 / np.asarray(nu_cm1, dtype=float)

    def theta_t(self, theta_i_deg: float, n_lambda: np.ndarray) -> np.ndarray:
        theta_i = np.deg2rad(theta_i_deg)
        s = (self.n0 / n_lambda) * np.sin(theta_i)
        s = np.clip(s, -1.0, 1.0)   # 数值安全
        return np.arcsin(s)

    def kappa(self, nu_cm1: np.ndarray, theta_i_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam_um = self.lambda_um_from_nu_cm1(nu_cm1)
        n_lam = self.n_of_lambda(lam_um)
        theta_t_rad = self.theta_t(theta_i_deg, n_lam)
        cos_t = np.cos(theta_t_rad)
        kap = n_lam * cos_t * nu_cm1
        return kap, n_lam, cos_t


# -----------------------------
# 折射率：按你给的色散式（λ 用 μm）
# n(λ) = 2.5610 + (3.4×10^4)/λ^2
# -----------------------------
def n_user_dispersion(lambda_um: np.ndarray) -> np.ndarray:
    lam2 = np.maximum(lambda_um**2, 1e-12)
    return 2.5610 + 3.4e-2 / lam2


# -----------------------------
# 路线 B：轴变换 + FFT 主频 -> d ≈ f/2
# -----------------------------
def estimate_d_via_kappa_axis(
    df: pd.DataFrame,
    theta_i_deg: float,
    model: OpticalModel,
    col_nu: str = None,
    col_R: str = None
) -> Dict[str, float]:
    # 自动列名
    if col_nu is None:
        col_nu = [c for c in df.columns if "cm" in c or "Wavenumber" in c][0]
    if col_R is None:
        col_R = [c for c in df.columns if "%" in c or "Reflect" in c][0]

    nu = df[col_nu].to_numpy(dtype=float)
    R = df[col_R].to_numpy(dtype=float)

    # 升序
    order = np.argsort(nu)
    nu = nu[order]
    R = R[order]

    # x = kappa(ν)
    x, _, _ = model.kappa(nu, theta_i_deg)

    # 等间隔重采样
    N = len(x)
    x_uniform = np.linspace(x.min(), x.max(), N)
    R_uniform = np.interp(x_uniform, x, R)

    # 去均值 + 汉宁窗
    y = R_uniform - np.mean(R_uniform)
    y *= np.hanning(N)

    # FFT
    dx = (x_uniform[-1] - x_uniform[0]) / (N - 1)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dx)
    mag = np.abs(Y)
    if len(freqs) > 1:
        mag[0] = 0.0
    f_cyc = float(freqs[int(np.argmax(mag))])  # cycles per unit x

    d_est = f_cyc / 2.0  # 因 cos(4π d x) = cos(2π*(2d) x)
    return {"d_est": d_est, "f_cyc": f_cyc}


# -----------------------------
# 路线 A：两束模型拟合  R(ν)=A+B*cos(4π d kappa(ν)+φ0)
# 一维搜索 d，固定 ω=4πd 时对 A,c,s 线性最小二乘
# -----------------------------
def _fit_ACS_for_fixed_omega(x: np.ndarray, R: np.ndarray, omega: float):
    cosx = np.cos(omega * x)
    sinx = np.sin(omega * x)
    X = np.vstack([np.ones_like(x), cosx, sinx]).T
    beta, *_ = np.linalg.lstsq(X, R, rcond=None)  # A, c, s
    A, c, s = beta
    resid = R - (A + c * cosx + s * sinx)
    sse = float(np.dot(resid, resid))
    return A, c, s, sse

def fit_d_two_beam_dispersion(
    df: pd.DataFrame,
    theta_i_deg: float,
    model: OpticalModel,
    d_init: float = None,
    search_half_width_ratio: float = 0.3,
    refine_steps: int = 3,
    col_nu: str = None,
    col_R: str = None
) -> Dict[str, float]:
    # 自动列名
    if col_nu is None:
        col_nu = [c for c in df.columns if "cm" in c or "Wavenumber" in c][0]
    if col_R is None:
        col_R = [c for c in df.columns if "%" in c or "Reflect" in c][0]

    nu = df[col_nu].to_numpy(dtype=float)
    R = df[col_R].to_numpy(dtype=float)

    # 升序
    order = np.argsort(nu)
    nu = nu[order]
    R = R[order]

    # x = kappa(ν)
    x, _, _ = model.kappa(nu, theta_i_deg)

    # 初猜（若未给）
    if d_init is None:
        d_init = estimate_d_via_kappa_axis(df, theta_i_deg, model, col_nu, col_R)["d_est"]
        if not np.isfinite(d_init) or d_init <= 0:
            d_init = 0.5 * 1.0 / max(1.0, (x.max() - x.min()) / 50.0)

    # 逐步细化的一维搜索
    d_center = d_init
    best = None
    for _ in range(refine_steps):
        half_width = max(1e-12, search_half_width_ratio * d_center)
        d_grid = np.linspace(d_center - half_width, d_center + half_width, 81)

        sse_list, params = [], []
        for d in d_grid:
            omega = 4.0 * math.pi * d
            A, c, s, sse = _fit_ACS_for_fixed_omega(x, R, omega)
            sse_list.append(sse)
            params.append((A, c, s))

        idx = int(np.argmin(sse_list))
        d_center = float(d_grid[idx])
        A_best, c_best, s_best = params[idx]
        best = {"d_fit": d_center, "A": A_best, "c": c_best, "s": s_best, "SSE": float(sse_list[idx])}
        search_half_width_ratio *= 0.5

    # 从 (c, s) 得 B, φ0
    B = math.hypot(best["c"], best["s"])
    phi0 = math.atan2(-best["s"], best["c"])
    return {"d_fit": best["d_fit"], "A": best["A"], "B": B, "phi0": phi0, "SSE": best["SSE"]}


# -----------------------------
# 便捷封装：对一个 DataFrame 同时跑 A/B 两路线
# -----------------------------
def calc_thickness_for_df(
    df: pd.DataFrame,
    theta_i_deg: float,
    n_of_lambda: Callable[[np.ndarray], np.ndarray] = n_user_dispersion
) -> Dict[str, float]:
    model = OpticalModel(n_of_lambda)
    outB = estimate_d_via_kappa_axis(df, theta_i_deg, model)
    outA = fit_d_two_beam_dispersion(df, theta_i_deg, model, d_init=outB["d_est"])
    return {
        "theta_i_deg": theta_i_deg,
        "routeB_d_est": outB["d_est"],
        "routeA_d_fit": outA["d_fit"],
        "routeA_SSE": outA["SSE"]
    }

if __name__=="__main__":
    from Data.DataManager import DM
    df1 = DM.get_data(1)  # 附件1 (10°)
    df2 = DM.get_data(2)  # 附件2 (15°)

    # df1: 入射角 10°
    res1 = calc_thickness_for_df(df1, theta_i_deg=10.0, n_of_lambda=n_user_dispersion)

    # df2: 入射角 15°
    res2 = calc_thickness_for_df(df2, theta_i_deg=15.0, n_of_lambda=n_user_dispersion)

    print("df1 (10°):", res1)
    print("df2 (15°):", res2)