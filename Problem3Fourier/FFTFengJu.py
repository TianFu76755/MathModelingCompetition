from typing import Tuple, List

from matplotlib import pyplot as plt

from Problem3Fourier.UseLess.yuchuli_fft_viz import preprocess_and_plot_compare


import numpy as np


def fft_peaks_and_thickness(
    nu_u,                  # 等间距波数数组 (cm^-1)
    y_w,                   # 预处理后的信号（去均值+乘窗）
    n=3.50,                # 折射率（可改成你的材料）
    theta_deg=10.0,        # 入射角（度）
    peak_count=3,          # 返回前几个最显著峰
    min_prominence=0.02,   # 峰显著性阈值（相对最大值）
):
    nu_u = np.asarray(nu_u, float)
    y_w  = np.asarray(y_w,  float)
    assert nu_u.ndim==1 and y_w.ndim==1 and len(nu_u)==len(y_w)

    # 1) 频谱：幅度 vs “光学厚度”
    dnu = float(nu_u[1] - nu_u[0])           # 等间距步长 (cm^-1)
    N   = len(nu_u)

    # 只取正频（不含0），避免直流分量
    Y = np.fft.rfft(y_w)                     # N//2+1 点，含0与Nyquist
    freqs = np.fft.rfftfreq(N, d=dnu)        # 单位：cycles per (cm^-1)

    # 去掉直流点，提高鲁棒性
    freqs = freqs[1:]
    mag   = np.abs(Y[1:])

    # 2) 将频率轴“标尺化”为光学厚度T（单位cm）：
    # 余弦项写成 cos(2π * f * nu)，且 f_peak = 2 n d cosθ
    T_axis = freqs.copy()    # 数值等同，但语义换成 "optical thickness (cm)"

    # 3) 粗糙平滑（防毛刺，可选）
    # 用三点均值滤波
    if len(mag) >= 3:
        mag_s = np.convolve(mag, np.ones(3)/3.0, mode='same')
    else:
        mag_s = mag

    # 4) 简易峰检：找局部极大 + 显著性筛选
    idx = np.arange(1, len(mag_s)-1)
    is_peak = (mag_s[idx] > mag_s[idx-1]) & (mag_s[idx] > mag_s[idx+1])
    cand = idx[is_peak]
    if len(cand)==0:
        return {"peaks":[],"T_axis":T_axis,"magnitude":mag_s}

    # prominence 相对阈值
    rel = mag_s[cand] / (mag_s.max() + 1e-15)
    cand = cand[rel >= min_prominence]
    if len(cand)==0:
        return {"peaks":[],"T_axis":T_axis,"magnitude":mag_s}

    # 按幅度排序取前 peak_count 个
    order = np.argsort(mag_s[cand])[::-1]
    sel   = cand[order[:peak_count]]

    cos_th = np.cos(np.deg2rad(theta_deg))
    results = []
    for k in sel:
        T_cm = T_axis[k]                # 光学厚度 2 n d cosθ (cm)
        d_cm = T_cm / (2.0 * n * cos_th)
        results.append({
            "T_cm": T_cm,
            "thickness_cm": d_cm,
            "thickness_um": d_cm * 1e4,   # 1 cm = 1e4 μm
            "thickness_nm": d_cm * 1e7,   # 1 cm = 1e7 nm
            "rel_strength": float(mag_s[k]/(mag_s.max()+1e-15)),
            "peak_index": int(k)
        })

    # 按厚度从大到小（或强度）排序，便于阅读
    results.sort(key=lambda x: x["rel_strength"], reverse=True)

    return {
        "peaks": results,       # 列表：每个峰的光学厚度与几何厚度
        "T_axis": T_axis,       # 用于可视化：横轴就是 “optical thickness (cm)”
        "magnitude": mag_s      # FFT 幅度（可画谱）
    }

def fft_analyze_and_plot(
    nu_u, y_w,
    n=3.50, theta_deg=10.0,
    peak_count=5, min_prominence=0.02,
    figsize=(8,4), title="FFT on Interference (geometric thickness domain)",
    xlim_um=(0, 10)   # 横坐标范围，单位 μm，默认 0~10
):
    """
    计算 FFT 并绘图：
      - 横轴：几何厚度 d [μm]
      - 纵轴：FFT 幅度（归一化）
      - 自动标注主峰/多峰及其换算厚度
    """
    out = fft_peaks_and_thickness(
        nu_u, y_w, n=n, theta_deg=theta_deg,
        peak_count=peak_count, min_prominence=min_prominence
    )
    T = out["T_axis"]
    M = out["magnitude"].astype(float)
    Mn = M / (M.max() + 1e-15)

    cos_th = np.cos(np.deg2rad(theta_deg))
    d_axis_um = (T / (2*n*cos_th)) * 1e4   # cm → μm

    plt.figure(figsize=figsize)
    plt.plot(d_axis_um, Mn, lw=1.6, label="FFT幅度（标准化）")
    plt.xlabel("几何厚度 d (μm)")
    plt.ylabel("幅度 (a.u.)")
    # plt.title(title)
    plt.grid(True, alpha=0.3)

    if xlim_um is not None:
        plt.xlim(*xlim_um)   # 设置横坐标范围

    # 标注峰
    if out["peaks"]:
        for p in out["peaks"]:
            d_um = p["thickness_um"]
            rs   = p["rel_strength"]
            idx  = p["peak_index"]
            plt.scatter([d_um], [Mn[idx]], s=40)
            plt.annotate(
                f"d={d_um:.3g} μm\n rel={rs:.2f}",
                xy=(d_um, Mn[idx]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.3", alpha=0.85),
                arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.6)
            )
    else:
        plt.annotate("No clear peaks found\n(check band/dispersion/window)",
                     xy=(0.02, 0.85), xycoords="axes fraction", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.3", alpha=0.85))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return out




if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    df = df4
    n = 3.50
    theta_deg = 15.0
    include_range: Tuple[float, float] = (550, 3000)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）

    # ============预处理阶段===========
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,     # 条纹最明显波段
        exclude_ranges=exclude_ranges,   # 强吸收段（可多段）
        tukey_alpha=0.5,                # Tukey 窗参数；设 0 关闭
        baseline_poly_deg=3,            # 基线多项式阶数
        uniform_points=None,            # 等间距采样点数（默认跟随数据）
        window_name="tukey",            # "tukey" / "hann" / "rect"
        show_windowed=True,             # 是否同时画“乘窗后”的曲线
    )
    nu_u = out["nu_uniform"]
    y_w = out["y_windowed"]

    # ============FFT 分析阶段===========
    out = fft_analyze_and_plot(
        nu_u, y_w,
        n=n, theta_deg=theta_deg,
        peak_count=5,  # 最多找5个峰
        min_prominence=0.05,  # 只标注强度≥5%的峰
        figsize=(9, 4.5),
        title=f"FFT (n={n}, theta={theta_deg}°)",
        xlim_um=(3, 25)
    )

    # 结果结构
    for i, p in enumerate(out["peaks"], 1):
        print(f"[{i}] T = {p['T_cm']:.6g} cm  |  d = {p['thickness_um']:.6g} μm  "
              f"| rel_strength={p['rel_strength']:.3f}")


"""
附件1：7.96μm
    df = df1
    n = 2.55
    theta_deg = 10.0
    include_range: Tuple[float, float] = (2000, 2500)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）

附件2：8.12μm
    df = df2
    n = 2.55
    theta_deg = 15.0
    include_range: Tuple[float, float] = (2000, 2500)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）
    
附件3：
[1] T = 0.00244886 cm  |  d = 3.55234 μm  | rel_strength=1.000
[2] T = 0.00530587 cm  |  d = 7.69674 μm  | rel_strength=0.216
    df = df3
    n = 3.50
    theta_deg = 10.0
    include_range: Tuple[float, float] = (550, 3000)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）

附件4：
[1] T = 0.00244886 cm  |  d = 3.62178 μm  | rel_strength=1.000
[2] T = 0.00530587 cm  |  d = 7.8472 μm  | rel_strength=0.216
    df = df4
    n = 3.50
    theta_deg = 15.0
    include_range: Tuple[float, float] = (550, 3000)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = [(3000, 4000)]  # 强吸收段（可多段）

"""

