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

def _solve_linear_ab(y, m):
    M = np.vstack([m, np.ones_like(m)]).T
    x, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = float(x[0]), float(x[1])
    return a, b

# ---------------- 单角度拟合 & 多角联合拟合 ----------------

def fit_single_angle(nu, R_meas, d0_um, n0=1.0, n1=3.50, n2=2.55, theta_deg=10.0,
                     search_span_um=40.0, coarse_N=1201, refine_iters=3):
    nu = np.asarray(nu, float); R_meas = np.asarray(R_meas, float)
    d0_cm = d0_um*1e-4; span_cm = search_span_um*1e-4
    def objective(d_cm):
        Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, theta_deg)
        a,b = _solve_linear_ab(R_meas, Rm)
        resid = R_meas - (a*Rm + b)
        return float(np.mean(resid**2)), a, b

    left = d0_cm - span_cm; right = d0_cm + span_cm
    N = coarse_N
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals = np.array([objective(d)[0] for d in ds])
        k = int(np.argmin(vals))
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]
        N = max(401, N//3)

    d_best = 0.5*(left+right)
    chi2, a_best, b_best = objective(d_best)
    Rm = airy_single_layer_reflectance(nu, d_best, n0, n1, n2, theta_deg)
    yfit = a_best*Rm + b_best
    resid = R_meas - yfit
    return {
        "theta_deg": theta_deg,
        "d_um": d_best*1e4, "a": a_best, "b": b_best, "chi2": chi2,
        "nu": nu, "R_meas": R_meas, "R_model": Rm, "R_fit": yfit, "residual": resid
    }

def fit_multi_angle(data_list, d0_um, n0=1.0, n1=3.50, n2=2.55,
                    search_span_um=40.0, coarse_N=1201, refine_iters=3):
    """
    data_list: 列表 [ (nu_1, R_1, theta_1), (nu_2, R_2, theta_2), ... ]
               每条曲线的 nu 可不同，但建议都为等间距波数
    返回：联合拟合后的 d 以及每个角度的 a,b、拟合与残差
    """
    # 准备插值到公共网格（可选）：这里直接各自用各自的 nu，目标函数相加
    d0_cm = d0_um*1e-4; span_cm = search_span_um*1e-4

    # 预先转换为 numpy
    D = []
    for (nu, R, th) in data_list:
        D.append((np.asarray(nu, float), np.asarray(R, float), float(th)))

    def objective(d_cm):
        chi2_sum = 0.0
        per_angle = []
        for (nu, R, th) in D:
            Rm = airy_single_layer_reflectance(nu, d_cm, n0, n1, n2, th)
            a,b = _solve_linear_ab(R, Rm)
            resid = R - (a*Rm + b)
            chi2 = float(np.mean(resid**2))
            chi2_sum += chi2
            per_angle.append((a,b, Rm, resid, chi2, th, nu, R))
        return chi2_sum/len(D), per_angle

    left = d0_cm - span_cm; right = d0_cm + span_cm
    N = coarse_N
    best = None
    for _ in range(refine_iters):
        ds = np.linspace(left, right, N)
        vals = []; intermediates = []
        for d in ds:
            chi2, per = objective(d)
            vals.append(chi2); intermediates.append((d, per))
        k = int(np.argmin(vals))
        d_star, per_star = intermediates[k]
        # 缩窗
        k0 = max(0, k-5); k1 = min(N-1, k+5)
        left, right = ds[k0], ds[k1]
        N = max(401, N//3)
        best = (d_star, per_star, vals[k])

    # 整理输出
    d_best_cm, per_best, _ = best
    out = {
        "d_um": d_best_cm*1e4,
        "angles": [],
    }
    for (a,b,Rm,resid,chi2,th,nu,R) in per_best:
        out["angles"].append({
            "theta_deg": th, "a": float(a), "b": float(b), "chi2": float(chi2),
            "nu": nu, "R_meas": R, "R_model": Rm, "R_fit": a*Rm + b, "residual": resid
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
        angles = [out_10["theta_deg"], out_15["theta_deg"]]
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
    R10_meas = out["y_windowed"]  # 对应反射率/信号

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
    R15_meas = out["y_windowed"]

    # 2) FFT 主峰得到的厚度初值（μm）
    d0_um = 3.36  # 例如 8.1

    # 3) 单角拟合（便于画“跨角一致性”）
    out10 = fit_single_angle(nu_10, R10_meas, d0_um, n1=3.50, n2=2.55, theta_deg=10.0)
    out15 = fit_single_angle(nu_15, R15_meas, d0_um, n1=3.50, n2=2.55, theta_deg=15.0)

    # 4) 多角联合拟合（核心结论）
    out_joint = fit_multi_angle([(nu_10, R10_meas, 10.0),
                                 (nu_15, R15_meas, 15.0)],
                                d0_um, n1=3.50, n2=2.55)

    # 5) 中文论文图（自动带上联合厚度）
    plot_multiangle_fit(out_joint, out10, out15,
                        title_prefix="单层 Airy 多角度联合拟合")
