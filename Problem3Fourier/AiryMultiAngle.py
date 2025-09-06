# 多角度 Airy 拟合（单层薄膜 + 衬底），含中文标注出图
# 说明：
# - 提供“单角度拟合”和“多角度联合拟合”两套接口；
# - 支持常数折射率 n1_const；（若你要 Cauchy 色散，可在 n1_func 里自定义 n(λ) 返回复数）
# - 每个角度都有各自的线性尺度/偏置 (a_j, b_j)，厚度 d 在各角共享；
# - 无 SciPy，仅用 numpy：对厚度 d 做 1D 缩放网格搜索；每次用线性最小二乘闭式解 a_j, b_j；
# - 下面演示部分用“合成数据”演示流程与图形；把合成数据替换为你的实测 (nu_10, R_10), (nu_15, R_15) 即可。
#
# 生成的图：
# 1) “实测 vs 拟合”（两角各一条）
# 2) “残差对比”（两角）
# 3) “跨角一致性图”——10°与15°的单角解散点 + 多角联合解的水平线
#
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from Problem3Fourier.yuchuli_fft_viz import preprocess_and_plot_compare


# ---------------- 基础物理函数（与单角版一致） ----------------

def _as_func(n):
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

def _solve_linear_ab_poly(y, m, x=None, poly_deg=0):
    """
    用闭式最小二乘解线性尺度/基线：
    y ≈ a*m + (b0 + b1*x + b2*x^2 + ...), 其中 x=nu 或 lambda
    poly_deg=0 表示只有常数项（即原来的 a,b）
    """
    if x is None or poly_deg == 0:
        M = np.vstack([m, np.ones_like(m)]).T
    else:
        cols = [m]
        for k in range(poly_deg+1):
            cols.append(x**k)
        M = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a = float(coef[0])
    b = coef[1:].astype(float) if len(coef) > 1 else np.array([0.0])
    return a, b, M @ coef  # 同时返回拟合后的线性部分 a*m + baseline

# ---------------- 单角度拟合 & 多角联合拟合 ----------------

def fit_single_angle(
    nu, R_meas, d0_um,
    n0=1.0, n1=3.50, n2=2.55, theta_deg=10.0,
    search_span_um=40.0, coarse_N=1201, refine_iters=3,
    poly_deg_baseline=0,               # 新增：每角可选多项式基线（0=仅常数）
    force_positive_thickness=True,     # 新增：厚度强制为正
    verbose=False
):
    """
    单角度 Airy 拟合（避免负厚度；可选多项式基线）
    返回字典包含 d_um(正值)、a、基线系数、χ²、RMSE、拟合曲线与残差。
    """
    nu = np.asarray(nu, float); R_meas = np.asarray(R_meas, float)
    d0_cm = d0_um * 1e-4
    span_cm = search_span_um * 1e-4

    def objective(d_cm):
        Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, theta_deg)
        a, c_vec, y_lin = _solve_linear_ab_poly(R_meas, Rm, x=nu, poly_deg=poly_deg_baseline)
        resid = R_meas - y_lin
        return float(np.mean(resid**2)), a, c_vec, y_lin, resid, Rm

    # 厚度搜索区间：若强制为正，从一个极小正数起
    left  = max(1e-9, d0_cm - span_cm) if force_positive_thickness else (d0_cm - span_cm)
    right = max(1e-9, d0_cm + span_cm)

    N = coarse_N
    best = None
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals = []
        cache = []
        for d in ds:
            chi2, a, c_vec, y_lin, resid, Rm = objective(d)
            vals.append(chi2)
            cache.append((d, chi2, a, c_vec, y_lin, resid, Rm))
        k = int(np.argmin(vals))
        d_star, chi2_star, a_star, c_star, ylin_star, resid_star, Rm_star = cache[k]
        # 缩窗
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]
        N = max(401, N//3)
        best = (d_star, chi2_star, a_star, c_star, ylin_star, resid_star, Rm_star)

    d_best_cm, chi2, a_best, c_vec_best, yfit, resid, Rm = best
    rmse = float(np.sqrt(np.mean(resid**2)))

    if verbose:
        print(f"[单角] θ={theta_deg:.0f}°  d = {d_best_cm*1e4:.6f} μm  χ² = {chi2:.3e}  RMSE = {rmse:.4f}")
        print(f"       a = {a_best:.3f},  baseline degree = {poly_deg_baseline},  coeffs = {c_vec_best}")

    return {
        "theta_deg": theta_deg,
        "d_um": d_best_cm*1e4,           # 已保证为正
        "a": a_best,
        "baseline_coeffs": c_vec_best,   # 基线多项式系数（c0,c1,...）
        "chi2": chi2,
        "rmse": rmse,
        "nu": nu,
        "R_meas": R_meas,
        "R_model": Rm,
        "R_fit": yfit,
        "residual": resid
    }

def fit_multi_angle(data_list, d0_um, n0=1.0, n1=3.50, n2=2.55,
                    search_span_um=40.0, coarse_N=1201, refine_iters=3,
                    poly_deg_each_angle=0,  # ← 新增：每角的基线多项式阶数（0=仅常数）
                    force_positive_thickness=True, verbose=True):
    d0_cm = d0_um*1e-4; span_cm = search_span_um*1e-4
    D = [(np.asarray(nu,float), np.asarray(R,float), float(th)) for (nu,R,th) in data_list]

    def objective(d_cm):
        chi2_sum = 0.0; per_angle = []
        for (nu, R, th) in D:
            Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, th)
            # 这里用 nu 当作 x 来拟合二阶基线（如果需要）
            a, bvec, y_lin = _solve_linear_ab_poly(R, Rm, x=nu, poly_deg=poly_deg_each_angle)
            resid = R - y_lin
            chi2 = float(np.mean(resid**2))
            chi2_sum += chi2
            per_angle.append((a, bvec, Rm, resid, chi2, th, nu, R, y_lin))
        return chi2_sum/len(D), per_angle

    # 厚度搜索区间（强制为正）
    left = max(1e-9, d0_cm - span_cm) if force_positive_thickness else (d0_cm - span_cm)
    right = max(1e-9, d0_cm + span_cm)
    N = coarse_N; best = None
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals = []; ints = []
        for d in ds:
            chi2, per = objective(d); vals.append(chi2); ints.append((d, per))
        k = int(np.argmin(vals)); d_star, per_star = ints[k]
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]; N = max(401, N//3)
        best = (d_star, per_star, vals[k])

    d_best_cm, per_best, chi2_joint = best
    out = {"d_um": d_best_cm*1e4, "chi2_joint": chi2_joint, "angles": []}
    if verbose:
        print(f"[联合拟合] d = {out['d_um']:.6f} μm,  χ²_joint = {chi2_joint:.3e}")
    for (a,bvec,Rm,resid,chi2,th,nu,R,y_lin) in per_best:
        rmse = float(np.sqrt(np.mean(resid**2)))
        if verbose:
            bl = " + ".join([f"c{k}" for k in range(len(bvec))])
            print(f"  θ={th:.0f}°: a={a:.3f}, 基线项数={len(bvec)}, χ²={chi2:.3e}, RMSE={rmse:.4f}")
        out["angles"].append({
            "theta_deg": th, "a": float(a), "bvec": bvec, "chi2": float(chi2), "rmse": rmse,
            "nu": nu, "R_meas": R, "R_model": Rm, "R_fit": y_lin, "residual": resid
        })
    return out

# ---------------- 跨角一致性图 + 拟合结果出图（中文标注） ----------------

def plot_multiangle_fit(out_joint, out_10=None, out_15=None, title_prefix="单层Airy 多角度联合拟合"):
    # 图1：实测 vs 拟合（两角）
    plt.figure(figsize=(9,4.8))
    for ang in out_joint["angles"]:
        nu = ang["nu"]; Rm = ang["R_meas"]; yfit = ang["R_fit"]
        th = ang["theta_deg"]; chi2 = ang["chi2"]
        plt.plot(nu, Rm, lw=1.2, label=f"实测  θ={th:.0f}°")
        plt.plot(nu, yfit, lw=1.6,
                 label=f"拟合 θ={th:.0f}°（$\chi^2$={chi2:.2e}）")
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("反射率 / 信号 (a.u.)")
    plt.title(f"{title_prefix}：实测 vs 拟合（各角度）\n联合厚度 d = {out_joint['d_um']:.4g} μm")
    plt.grid(True, alpha=0.3); plt.legend()
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

# ---------------- 演示：用“合成数据”跑一遍（请替换为你的实测） ----------------


if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 1) 把这里替换为你的等间距波数 & 去基线/可用于拟合的反射信号
    df = df3
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
    nu_10 = out["nu_uniform"]  # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]  # 对应反射率/信号

    df = df4
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
    nu_15 = out["nu_uniform"]
    R15_meas = out["y_uniform_demean"]

    # 2) FFT 主峰得到的厚度初值（μm）
    d0_um = 3.36  # 例如 8.1

    # 3) 单角拟合（便于画“跨角一致性”）
    out10 = fit_single_angle(nu_10, R10_meas, d0_um, n1=3.50, n2=2.55, theta_deg=10.0)
    out15 = fit_single_angle(nu_15, R15_meas, d0_um, n1=3.50, n2=2.55, theta_deg=15.0)

    # 4) 多角联合拟合（核心结论）
    out_joint = fit_multi_angle([(nu_10, R10_meas, 10.0),
                                 (nu_15, R15_meas, 15.0)],
                                d0_um,
                                n1=3.50, n2=2.55,
                                poly_deg_each_angle=2,  # ← 每个角度加一个二次基线
                                force_positive_thickness=True,
                                verbose=True)

    # 5) 中文论文图（自动带上联合厚度）
    plot_multiangle_fit(out_joint, out10, out15,
                        title_prefix="单层 Airy 多角度联合拟合")
