from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def preprocess_and_plot_compare(
    df: pd.DataFrame,
    include_range: Optional[Tuple[float, float]] = None,
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    tukey_alpha: float = 0.5,
    baseline_poly_deg: int = 3,
    uniform_points: Optional[int] = None,
    window_name: str = "tukey",
    show_windowed: bool = True,
):
    """
    Explain and visualize preprocessing steps by plotting:
      - Original raw signal R(nu) (black)
      - Preprocessed signal on uniform grid before window y_u (blue)
      - (Optional) Windowed signal y_w (thin, gray)

    Returns a dict with nu, R, nu_uniform, y_uniform, y_windowed, baseline, mask, etc.
    """
    # pick first two numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError("DataFrame must have two numeric columns.")
    nu_raw = df[num_cols[0]].to_numpy(float)
    R_raw  = df[num_cols[1]].to_numpy(float)

    # 1) baseline removal
    x = (nu_raw - nu_raw.mean()) / (nu_raw.std() + 1e-12)
    coeff = np.polyfit(x, R_raw, deg=baseline_poly_deg)
    baseline = np.polyval(coeff, x)
    R_detr = R_raw - baseline

    # 2) band selection mask
    def build_mask(nu):
        m = np.ones_like(nu, dtype=bool)
        if include_range is not None:
            lo, hi = include_range
            m &= (nu >= lo) & (nu <= hi)
        if exclude_ranges:
            for lo, hi in exclude_ranges:
                m &= ~((nu >= lo) & (nu <= hi))
        return m

    mask = build_mask(nu_raw)
    if not np.any(mask):
        raise ValueError("Mask removes all points. Check include/exclude ranges.")
    nu_m, y_m = nu_raw[mask], R_detr[mask]

    # 3) keep longest contiguous segment to avoid interpolation across gaps
    dnu_raw = np.diff(nu_m)
    median_step = np.median(np.abs(dnu_raw)) if len(dnu_raw)>0 else 0.0
    gaps = np.where(dnu_raw > 3*median_step)[0] if median_step>0 else np.array([])
    seg_starts = np.r_[0, gaps+1]; seg_ends = np.r_[gaps, len(nu_m)-1]
    lengths = seg_ends - seg_starts + 1
    k = int(np.argmax(lengths))
    i0, i1 = int(seg_starts[k]), int(seg_ends[k])
    nu_seg, y_seg = nu_m[i0:i1+1], y_m[i0:i1+1]

    # 4) resample to uniform grid
    if uniform_points is None:
        uniform_points = len(nu_seg)
    nu_u = np.linspace(nu_seg.min(), nu_seg.max(), uniform_points)
    y_u  = np.interp(nu_u, nu_seg, y_seg)

    # 5) mean detrend + window
    y_u_dm = y_u - y_u.mean()
    if window_name.lower() == "tukey":
        # local Tukey
        N = len(nu_u); a = float(tukey_alpha)
        n = np.arange(N); w = np.ones(N)
        if a > 0:
            edge = int(np.floor(a*(N-1)/2.0))
            ramp = 0.5*(1 + np.cos(np.pi*(2*n/(a*(N-1)) - 1)))
            w[:edge+1] = ramp[:edge+1]
            w[-edge-1:] = ramp[:edge+1][::-1]
    elif window_name.lower() == "hann":
        w = np.hanning(len(nu_u))
    else:
        w = np.ones(len(nu_u))
    y_w = y_u_dm * w

    # 6) plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(nu_raw, R_raw, label="原始 (原坐标)", linewidth=1.2)
    # map preprocessed back to raw axis for visual comparison (optional)
    # Here we just plot on its own uniform axis
    plt.plot(nu_u, y_u_dm, label="预处理后（去基线+等间距+去均值）", linewidth=1.5)
    if show_windowed:
        plt.plot(nu_u, y_w, label="预处理后（再乘窗）", linewidth=1.0)
    plt.xlabel("波数 (cm$^{-1}$)")
    plt.ylabel("信号 (a.u.)")
    plt.title("原始信号 vs 预处理结果")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return {
        "nu_raw": nu_raw,
        "R_raw": R_raw,
        "baseline": baseline,
        "mask_raw": mask,
        "nu_uniform": nu_u,
        "y_uniform_demean": y_u_dm,
        "y_windowed": y_w,
    }

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

    dfs = [df1, df2, df3, df4]
    include_range: Tuple[float, float] = (400, 4000)  # 条纹最明显波段
    exclude_ranges: List[Tuple[float, float]] = []  # 强吸收段（可多段）
    for df in dfs:
        out = preprocess_and_plot_compare(
            df,
            include_range=include_range,
            exclude_ranges=exclude_ranges,
            tukey_alpha=0.5,
            baseline_poly_deg=3,
            uniform_points=None,
            window_name="tukey",
            show_windowed=True,
        )
