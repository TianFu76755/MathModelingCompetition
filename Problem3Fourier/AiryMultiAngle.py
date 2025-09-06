# -*- coding: utf-8 -*-
"""
多角度 Airy 拟合（单层薄膜 + 衬底），中文标注出图（统一求解逻辑版）

变更要点：
1) 统一线性子问题：给定 d，构造设计矩阵 M=[R_Airy(d;nu), 1, x, x^2, ...]，单角/多角共用
2) 多项式基线使用标准化后的 x=(nu-mean)/std，数值更稳健
3) 单角与多角的目标函数一致，避免隐藏差异
4) 演示部分：单角与多角的基线阶数统一（默认二次）；你也可以都设为 0（仅常数）
"""

from typing import Tuple, List, Sequence, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# 你的工程里的预处理依赖
from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


# ========================= 基础物理函数 =========================

def _as_func(n):
    """若 n 为常数，转成常量函数；若为可调用，则原样返回。"""
    if callable(n):
        return n
    else:
        return lambda wn: np.asarray(n, dtype=complex) + 0*wn

def fresnel_r_s(n_i, th_i, n_j, th_j):
    return (n_i*np.cos(th_i) - n_j*np.cos(th_j)) / (n_i*np.cos(th_i) + n_j*np.cos(th_j))

def fresnel_r_p(n_i, th_i, n_j, th_j):
    return (n_j*np.cos(th_i) - n_i*np.cos(th_j)) / (n_j*np.cos(th_i) + n_i*np.cos(th_j))

def snell_theta(n_i, th_i, n_j):
    s = (n_i/n_j) * np.sin(th_i)
    return np.arcsin(s)

def airy_single_layer_reflectance(
    wn,                 # 波数数组 cm^-1（等间距更好）
    d_cm,               # 厚度（cm）
    n0=1.0,             # 入射介质（空气）
    n1=3.50,            # 薄膜折射率：可为常数或函数 n1(wn)->complex
    n2=2.55,            # 衬底折射率：可为常数或函数
    theta0_deg=10.0,    # 入射角（度）
):
    """Airy 单层 + 衬底，非偏振反射率"""
    wn = np.asarray(wn, float)
    lam_cm = 1.0/wn  # 波长 cm
    k0 = 2*np.pi/lam_cm

    n0f = _as_func(n0); n1f = _as_func(n1); n2f = _as_func(n2)
    th0 = np.deg2rad(theta0_deg) + 0j
    n0c = n0f(wn); n1c = n1f(wn); n2c = n2f(wn)

    th1 = snell_theta(n0c, th0, n1c)
    th2 = snell_theta(n1c, th1, n2c)

    r01s = fresnel_r_s(n0c, th0, n1c, th1)
    r12s = fresnel_r_s(n1c, th1, n2c, th2)
    r01p = fresnel_r_p(n0c, th0, n1c, th1)
    r12p = fresnel_r_p(n1c, th1, n2c, th2)

    beta = k0 * n1c * np.cos(th1) * (2*d_cm)
    exp2ib = np.exp(1j*beta)
    rs = (r01s + r12s*exp2ib) / (1.0 + r01s*r12s*exp2ib)
    rp = (r01p + r12p*exp2ib) / (1.0 + r01p*r12p*exp2ib)

    R = 0.5*(np.abs(rs)**2 + np.abs(rp)**2)  # 非偏振
    return R.real


# ========================= 数值稳健性与统一线性子问题 =========================

def _standardize_x(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    标准化工具：返回 x_std, 均值 mu, 标准差 sig
    若 sig==0，则用 1.0 防止除零。
    """
    x = np.asarray(x, float)
    mu = float(x.mean())
    sig = float(x.std())
    if sig <= 0:
        sig = 1.0
    return (x - mu) / sig, mu, sig

def _build_design_matrix(R_model: np.ndarray, nu: np.ndarray, poly_deg: int) -> np.ndarray:
    """
    设计矩阵：
    - 首列是 R_model
    - 若 poly_deg>0，则附加 [1, x, x^2, ..., x^poly_deg]，其中 x 是标准化后的波数
      注意包含常数项（k=0）
    - 若 poly_deg<=0，则只加常数项（与旧版兼容）
    """
    R_model = np.asarray(R_model, float)
    nu = np.asarray(nu, float)

    if poly_deg <= 0:
        M = np.vstack([R_model, np.ones_like(R_model)]).T
    else:
        x_std, _, _ = _standardize_x(nu)
        cols = [R_model]
        for k in range(poly_deg + 1):
            cols.append(x_std**k)
        M = np.vstack(cols).T
    return M

def _solve_linear_least_squares(y: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    闭式最小二乘解：min ||M c - y||^2
    返回：coef（含 a 与基线系数）、拟合值 y_fit、均方误差 chi2
    其中 coef[0] = a，coef[1:] = baseline 多项式系数（若存在）
    """
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    y_fit = M @ coef
    resid = y - y_fit
    chi2 = float(np.mean(resid**2))
    return coef, y_fit, chi2

def _solve_per_angle_given_d(
    nu: np.ndarray,
    R_meas: np.ndarray,
    R_model: np.ndarray,
    poly_deg: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    统一线性子问题（单角/多角共用）：
    给定该角度的 Airy 模型值 R_model(=R_Airy(d; nu))，
    在线性模型 y ≈ a*R_model + poly(x_std) 下求闭式最小二乘。
    返回: a（尺度）, bvec（基线系数）, y_fit, residual, chi2
    """
    M = _build_design_matrix(R_model, nu, poly_deg)
    coef, y_fit, chi2 = _solve_linear_least_squares(R_meas, M)
    a = float(coef[0])
    bvec = coef[1:].astype(float) if len(coef) > 1 else np.array([0.0])
    residual = R_meas - y_fit
    return a, bvec, y_fit, residual, chi2


# ========================= 单角与多角拟合（统一逻辑） =========================

def fit_single_angle(
    nu: Sequence[float],
    R_meas: Sequence[float],
    d0_um: float,
    n0: Any = 1.0,
    n1: Any = 3.50,
    n2: Any = 2.55,
    theta_deg: float = 10.0,
    search_span_um: float = 40.0,
    coarse_N: int = 1201,
    refine_iters: int = 3,
    poly_deg_baseline: int = 0,
    force_positive_thickness: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    单角度 Airy 拟合（统一线性子问题；可选多项式基线，默认仅常数）
    返回：包含 d_um, a, baseline_coeffs, chi2, rmse, 各曲线与残差。
    """
    nu = np.asarray(nu, float)
    R_meas = np.asarray(R_meas, float)
    d0_cm = d0_um * 1e-4
    span_cm = search_span_um * 1e-4

    def objective(d_cm: float):
        Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, theta_deg)
        a, bvec, y_fit, resid, chi2 = _solve_per_angle_given_d(nu, R_meas, Rm, poly_deg_baseline)
        return chi2, a, bvec, y_fit, resid, Rm

    left  = max(1e-9, d0_cm - span_cm) if force_positive_thickness else (d0_cm - span_cm)
    right = max(1e-9, d0_cm + span_cm)

    N = coarse_N
    best = None
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals, cache = [], []
        for d in ds:
            chi2, a, bvec, y_fit, resid, Rm = objective(d)
            vals.append(chi2)
            cache.append((d, chi2, a, bvec, y_fit, resid, Rm))
        k = int(np.argmin(vals))
        d_star, chi2_star, a_star, bvec_star, yfit_star, resid_star, Rm_star = cache[k]
        # 缩窗：以最优附近 5 点为窗
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]
        N = max(401, N//3)
        best = (d_star, chi2_star, a_star, bvec_star, yfit_star, resid_star, Rm_star)

    d_best_cm, chi2, a_best, bvec_best, yfit, resid, Rm = best
    rmse = float(np.sqrt(np.mean(resid**2)))

    if verbose:
        print(f"[单角] θ={theta_deg:.0f}°  d = {d_best_cm*1e4:.6f} μm  χ² = {chi2:.3e}  RMSE = {rmse:.4f}")
        print(f"       a = {a_best:.3f},  baseline deg = {poly_deg_baseline},  coeffs = {bvec_best}")

    return {
        "theta_deg": theta_deg,
        "d_um": d_best_cm*1e4,
        "a": a_best,
        "baseline_coeffs": bvec_best,
        "chi2": chi2,
        "rmse": rmse,
        "nu": nu,
        "R_meas": R_meas,
        "R_model": Rm,
        "R_fit": yfit,
        "residual": resid
    }

def fit_multi_angle(
    data_list: Sequence[Tuple[Sequence[float], Sequence[float], float]],
    d0_um: float,
    n0: Any = 1.0,
    n1: Any = 3.50,
    n2: Any = 2.55,
    search_span_um: float = 40.0,
    coarse_N: int = 1201,
    refine_iters: int = 3,
    poly_deg_each_angle: int = 0,
    force_positive_thickness: bool = True,
    verbose: bool = True,
    sample_weighting: str = "mean"
) -> Dict[str, Any]:
    """
    多角联合拟合（与单角共享完全一致的线性子问题）
    参数：
      - poly_deg_each_angle: 每个角度的基线多项式阶数（0=仅常数）
      - sample_weighting: "mean"（各角 MSE 取算术平均）或 "size"（按样本数加权平均）
    返回：
      - d_um, chi2_joint, angles[ {theta_deg, a, bvec, chi2, rmse, nu, R_meas, R_model, R_fit, residual} ]
    """
    d0_cm = d0_um*1e-4
    span_cm = search_span_um*1e-4
    D = [(np.asarray(nu,float), np.asarray(R,float), float(th)) for (nu,R,th) in data_list]

    def aggregate_chi2(chi2_list: List[float], sizes: List[int]) -> float:
        if sample_weighting == "size":
            w = np.asarray(sizes, float)
            w = w / w.sum()
            return float(np.sum(w * np.asarray(chi2_list, float)))
        else:
            # 默认算术平均
            return float(np.mean(chi2_list))

    def objective(d_cm: float):
        chi2_list, sizes = [], []
        per_angle = []
        for (nu, R, th) in D:
            Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, th)
            a, bvec, y_fit, resid, chi2 = _solve_per_angle_given_d(nu, R, Rm, poly_deg_each_angle)
            chi2_list.append(chi2)
            sizes.append(len(nu))
            per_angle.append((a, bvec, Rm, resid, chi2, th, nu, R, y_fit))
        chi2_joint = aggregate_chi2(chi2_list, sizes)
        return chi2_joint, per_angle

    left  = max(1e-9, d0_cm - span_cm) if force_positive_thickness else (d0_cm - span_cm)
    right = max(1e-9, d0_cm + span_cm)

    N = coarse_N
    best = None
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals, ints = [], []
        for d in ds:
            chi2_joint, per = objective(d)
            vals.append(chi2_joint)
            ints.append((d, per))
        k = int(np.argmin(vals))
        d_star, per_star = ints[k]
        # 缩窗
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]
        N = max(401, N//3)
        best = (d_star, per_star, vals[k])

    d_best_cm, per_best, chi2_joint = best
    out = {"d_um": d_best_cm*1e4, "chi2_joint": chi2_joint, "angles": []}
    if verbose:
        print(f"[联合拟合] d = {out['d_um']:.6f} μm,  χ²_joint = {chi2_joint:.3e}  (weight={sample_weighting})")

    for (a,bvec,Rm,resid,chi2,th,nu,R,y_fit) in per_best:
        rmse = float(np.sqrt(np.mean(resid**2)))
        if verbose:
            print(f"  θ={th:.0f}°: a={a:.3f}, 基线项数={len(bvec)}, χ²={chi2:.3e}, RMSE={rmse:.4f}")
        out["angles"].append({
            "theta_deg": th, "a": float(a), "bvec": bvec, "chi2": float(chi2), "rmse": rmse,
            "nu": nu, "R_meas": R, "R_model": Rm, "R_fit": y_fit, "residual": resid
        })
    return out


# ========================= 出图（中文标注） =========================

def plot_multiangle_fit(
    out_joint: Dict[str, Any],
    out_10: Dict[str, Any] = None,
    out_15: Dict[str, Any] = None,
    title_prefix: str = "单层 Airy 多角度联合拟合"
):
    # 图1：实测 vs 拟合（两角）
    plt.figure(figsize=(9,4.8))
    for ang in out_joint["angles"]:
        nu = ang["nu"]; Rm = ang["R_meas"]; yfit = ang["R_fit"]
        th = ang["theta_deg"]; chi2 = ang["chi2"]
        plt.plot(nu, Rm, lw=1.2, label=f"实测  θ={th:.0f}°")
        plt.plot(nu, yfit, lw=1.6, label=f"拟合 θ={th:.0f}°（$\chi^2$={chi2:.2e}）")
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("反射率 / 信号 (a.u.)")
    plt.title(f"{title_prefix}：实测 vs 拟合（各角度）\n联合厚度 d = {out_joint['d_um']:.4g} μm")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show()

    # 图 1.2 Separate plotting for each angle
    for angle_data, angle_name in zip([out_10, out_15], ['10', '15']):
        plt.figure(figsize=(9, 4.8))

        if angle_data:
            nu = angle_data["nu"]
            Rm = angle_data["R_meas"]
            yfit = angle_data["R_fit"]
            th = angle_data["theta_deg"]
            chi2 = angle_data["chi2"]

            # Plot measured vs fitted data for the specific angle
            plt.plot(nu, Rm, lw=1.2, label=f"实测  θ={th:.0f}°")
            plt.plot(nu, yfit, lw=1.6, label=f"拟合 θ={th:.0f}°（$\chi^2$={chi2:.2e}）")

            plt.xlabel("波数 (cm$^{-1}$)")
            plt.ylabel("反射率 / 信号 (a.u.)")
            plt.title(f"{title_prefix}：实测 vs 拟合（角度 θ={th:.0f}°）\n联合厚度 d = {out_joint['d_um']:.4g} μm")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 图2：残差（两角）
    plt.figure(figsize=(9,4.2))
    for ang in out_joint["angles"]:
        nu = ang["nu"]; resid = ang["residual"]; th = ang["theta_deg"]
        plt.plot(nu, resid, lw=1.0, label=f"残差  θ={th:.0f}°")
    plt.axhline(0, color="k", lw=0.8, alpha=0.5)
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("残差")
    plt.title("残差随波数分布（应围绕0上下波动、无系统形状）")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show()

    # 图3：跨角一致性图（单角 vs 联合）
    if (out_10 is not None) and (out_15 is not None):
        angles = [10, 15]
        d_single = [out_10["d_um"], out_15["d_um"]]
        plt.figure(figsize=(6.4,4.2))
        plt.scatter(angles, d_single, s=60, label="单角拟合厚度", zorder=3)
        # 联合拟合水平线
        a_min, a_max = min(angles)-2, max(angles)+2
        plt.hlines(out_joint["d_um"], a_min, a_max, linestyles="--", label="多角联合拟合厚度")
        for x,y in zip(angles, d_single):
            plt.text(x+0.3, y, f"{y:.3g} μm", fontsize=10)
        plt.xlabel("入射角 (°)")
        plt.ylabel("厚度 (μm)")
        plt.title("跨角一致性图：单角 vs 多角联合")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.xlim(a_min, a_max)
        plt.tight_layout()
        plt.show()

    def plot_consistency_dumbbell_v2(out_joint, singles: dict):
        """
        singles: {angle_deg: d_um, ...} 例如 {10: out_10["d_um"], 15: out_15["d_um"]}
        """
        import matplotlib.pyplot as plt
        angles = sorted(singles.keys())
        y = [0.25, 0.75]  # 给 10° 和 15° 分配的 y 位置
        d_joint = out_joint["d_um"]
        d_single = [singles[a] for a in angles]

        plt.figure(figsize=(7.0, 3.8))

        # 连接线（基准 -> 单角）
        for yi, ds in zip(y, d_single):
            plt.plot([min(d_joint, ds), max(d_joint, ds)], [yi, yi], lw=2.2, alpha=0.8, zorder=1)

        # 基准点与单角点（横向）
        plt.scatter([d_joint] * len(y), y, s=120, label="联合厚度", zorder=3)
        plt.scatter(d_single, y, s=140, edgecolor="k", linewidths=0.6, label="单角厚度", zorder=4)

        # 标注 Δ
        for yi, ds in zip(y, d_single):
            delta = ds - d_joint
            plt.text(ds - 0.004, yi + 0.055, f"Δ={delta:+.3g} μm", fontsize=13, va="center")

        # 设置 x 轴范围紧缩
        x_all = d_single + [d_joint]
        x_pad = max(1e-3, 0.003 * max(x_all))
        plt.xlim(min(x_all) - x_pad, max(x_all) + x_pad)

        # 去掉 y 轴，使用自定义位置
        plt.yticks(y, [f"{a}°" for a in angles])

        # 强制设置 y 轴的上下限，确保点不会出现在最上面或最下面
        plt.ylim(min(y) - 0.2, max(y) + 0.2)  # 稍微加一些空间

        plt.xticks(fontsize=10)  # 让 x 轴的字体大小更清晰

        plt.xlabel("厚度 (μm)")
        plt.title("跨角一致性（哑铃图）：相对联合基准的偏差")
        plt.grid(True, axis="x", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_consistency_dumbbell_v2(out_joint, {10: out_10["d_um"], 15: out_15["d_um"]})


def n_Si_dispersion(wn_cm_inv):
    """
    硅的实折射率色散（无吸收）：
      n^2 = 11.6858 + 0.939816/(λ^2 - 0.0086024) + 0.00089814 * λ^2
    λ 用 μm；输入 wn 为 cm^-1。
    """
    wn = np.asarray(wn_cm_inv, float)
    lam_um = 1e4 / wn
    l2 = lam_um**2
    n2 = 11.6858 + 0.939816/(l2 - 0.0086024) + 0.00089814*l2
    n2 = np.maximum(n2, 1e-12)
    return np.sqrt(n2) + 0j


def n_Si_dispersion_with_k(wn_cm_inv, k0=1e-3, kind="const", ref_um=4.0, p=0.0):
    """
    硅的复折射率色散：n(λ) + i*k(λ)
    - 实部 n(λ) 用上面的经验公式；
    - 虚部 k(λ) 提供两种简便模型：
        kind="const"  : k(λ) = k0                  （默认）
        kind="power"  : k(λ) = k0 * (λ/ref_um)^p   （幂律，可模拟远红外增长/衰减）
    参  数：
      wn_cm_inv : 波数(cm^-1)
      k0        : 常数吸收基准（建议先在 1e-5 ~ 1e-2 扫）
      kind      : "const" 或 "power"
      ref_um    : 幂律参考波长（μm），仅当 kind="power" 时使用
      p         : 幂指数，>0 表示随波长增大而增大
    返回：
      n_complex(wn) : 复数折射率（n + 1j*k）
    """
    wn = np.asarray(wn_cm_inv, float)
    lam_um = 1e4 / wn

    # n(λ) 实部
    l2 = lam_um**2
    n2 = 11.6858 + 0.939816/(l2 - 0.0086024) + 0.00089814*l2
    n2 = np.maximum(n2, 1e-12)
    n_real = np.sqrt(n2)

    # k(λ) 虚部
    if kind == "power":
        k = k0 * (lam_um / float(ref_um))**float(p)
    else:  # "const"
        k = np.full_like(lam_um, float(k0))

    # 数值安全：非负、小范围保护
    k = np.clip(k, 0.0, 1.0)

    return n_real + 1j*k


# ========================= 演示（用合成/实测数据替换） =========================

if __name__ == "__main__":
    # 中文显示（按你工程）
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()

    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 1) 预处理（把这里替换为你的等间距波数 & 去基线/可用于拟合的反射信号）
    # 10°
    df = df3
    include_range: Tuple[float, float] = (2000, 2700)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = []  # 强吸收段（可多段）
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
        is_plot=True,
    )
    nu_10 = out["nu_uniform"]                 # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]        # 对应反射率/信号（已去均值或基线）

    # 15°
    df = df4
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        tukey_alpha=0.5,
        baseline_poly_deg=3,
        uniform_points=None,
        window_name="tukey",
        show_windowed=True,
        is_plot=True,
    )
    nu_15 = out["nu_uniform"]
    R15_meas = out["y_uniform_demean"]

    # 2) FFT 主峰得到的厚度初值（μm）——请替换为你的估计
    d0_um = 3.39

    # 3) 单角拟合（与多角保持相同的基线阶数！此处统一用二次）
    out10 = fit_single_angle(
        nu_10, R10_meas, d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55, theta_deg=10.0,
        poly_deg_baseline=1, verbose=True
    )
    out15 = fit_single_angle(
        nu_15, R15_meas, d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55, theta_deg=15.0,
        poly_deg_baseline=1, verbose=True
    )

    # 4) 多角联合拟合（与单角一致的二次基线；可选 sample_weighting="size" 按样本数加权）
    out_joint = fit_multi_angle(
        [(nu_10, R10_meas, 10.0), (nu_15, R15_meas, 15.0)],
        d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55,
        poly_deg_each_angle=1,
        force_positive_thickness=True,
        verbose=True,
        sample_weighting="mean",   # 或 "size"
    )

    # 5) 中文论文图（自动带上联合厚度）
    plot_multiangle_fit(out_joint, out10, out15, title_prefix="单层 Airy 多角度联合拟合")
