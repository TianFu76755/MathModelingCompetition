
"""
Epi FFT Visualizer (Route A)
----------------------------
- Preprocess (baseline removal, band selection, Tukey window, mean detrend)
- FFT with amplitude normalization
- Peak picking with automatic tau_min ≈ 1/Δν and noise threshold
- Sparse reconstruction from selected τ via linear least squares (stable)
- Figures:
  (1) original preprocessed signal vs. each selected component wave;
  (2) original preprocessed signal vs. sum of selected components.

Usage example (pseudocode):
    import pandas as pd
    df = ... # two numeric columns: [wavenumber_cm-1, signal]
    from epi_fft_viz import visualize_fft_decomposition
    visualize_fft_decomposition(df,
        include_range=(1800, 3300),
        exclude_ranges=[(900, 1100)],
        k_components=10,
        tukey_alpha=0.5,
        zero_pad_factor=8,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- Core helpers (Route A) -----------------------

@dataclass
class SpectrumPreprocessor:
    baseline_poly_deg: int = 3
    window: str = "tukey"
    tukey_alpha: float = 0.5
    include_range: Optional[Tuple[float, float]] = None
    exclude_ranges: Optional[List[Tuple[float, float]]] = None
    detrend_mean: bool = True

    def _mask_ranges(self, nu: np.ndarray) -> np.ndarray:
        m = np.ones_like(nu, dtype=bool)
        if self.include_range is not None:
            lo, hi = self.include_range
            m &= (nu >= lo) & (nu <= hi)
        if self.exclude_ranges:
            for lo, hi in self.exclude_ranges:
                m &= ~((nu >= lo) & (nu <= hi))
        return m

    def _tukey(self, N: int, alpha: float) -> np.ndarray:
        if alpha <= 0:
            return np.ones(N)
        if alpha >= 1:
            return np.hanning(N)
        n = np.arange(N)
        w = np.ones(N)
        edge = int(np.floor(alpha*(N-1)/2.0))
        ramp = 0.5*(1 + np.cos(np.pi*(2*n/(alpha*(N-1)) - 1)))
        w[:edge+1] = ramp[:edge+1]
        w[-edge-1:] = ramp[:edge+1][::-1]
        return w

    def run(self, nu: np.ndarray, R: np.ndarray, uniform_points: Optional[int] = None):
        nu = np.asarray(nu, dtype=float)
        R  = np.asarray(R,  dtype=float)

        # 1) Baseline removal
        x = (nu - nu.mean()) / (nu.std() + 1e-12)
        coeff = np.polyfit(x, R, deg=self.baseline_poly_deg)
        baseline = np.polyval(coeff, x)
        y = R - baseline

        # 2) Band selection
        m = self._mask_ranges(nu)
        if not np.any(m):
            raise ValueError("Mask removes all points. Check include/exclude ranges.")
        nu_m, y_m = nu[m], y[m]

        # handle gaps → keep the longest contiguous segment
        dnu_raw = np.diff(nu_m)
        if len(dnu_raw) == 0:
            raise ValueError("Not enough points after masking.")
        median_step = np.median(np.abs(dnu_raw))
        gaps = np.where(dnu_raw > 3*median_step)[0]
        seg_starts = np.r_[0, gaps+1]; seg_ends = np.r_[gaps, len(nu_m)-1]
        lengths = seg_ends - seg_starts + 1
        k = int(np.argmax(lengths))
        i0, i1 = int(seg_starts[k]), int(seg_ends[k])
        nu_m, y_m = nu_m[i0:i1+1], y_m[i0:i1+1]

        # 3) Uniform resampling
        if uniform_points is None:
            uniform_points = len(nu_m)
        nu_u = np.linspace(nu_m.min(), nu_m.max(), uniform_points)
        y_u  = np.interp(nu_u, nu_m, y_m)

        # 4) Window & mean detrend
        if self.detrend_mean:
            y_u = y_u - y_u.mean()
        if self.window.lower() == "tukey":
            w = self._tukey(uniform_points, self.tukey_alpha)
        elif self.window.lower() == "hann":
            w = np.hanning(uniform_points)
        else:
            w = np.ones(uniform_points)
        y_w = y_u * w

        dnu = (nu_u[-1] - nu_u[0]) / (uniform_points - 1)
        delta_nu = nu_u.max() - nu_u.min()
        return nu_u, y_u, y_w, w, dnu, delta_nu

@dataclass
class FourierAnalyzer:
    zero_pad_factor: int = 8
    pow2_pad: bool = True

    def run(self, signal_uniform: np.ndarray, dnu: float, window: np.ndarray):
        y = np.asarray(signal_uniform, dtype=float)
        N = len(y)
        Npad = int(max(1, self.zero_pad_factor) * N)
        if self.pow2_pad:
            Npad = 1 << (int(np.ceil(np.log2(max(2, Npad)))))
        # amplitude normalization: * dnu / RMS(window)
        S = np.fft.rfft(y, n=Npad) * dnu / np.sqrt((window**2).mean())
        tau = np.fft.rfftfreq(Npad, d=dnu)
        return tau, S, Npad

@dataclass
class PeakPicker:
    k_max: int = 10
    tau_min: Optional[float] = None
    prominence: float = 0.0
    amp_threshold_factor: float = 5.0
    exclude_first_bins: int = 3

    def run(self, tau: np.ndarray, S: np.ndarray, delta_nu: Optional[float] = None) -> pd.DataFrame:
        amp = np.abs(S); phase = np.angle(S)

        if self.tau_min is None and (delta_nu is not None and delta_nu > 0):
            tmin = 1.0 / float(delta_nu)
        else:
            tmin = float(self.tau_min or 0.0)

        start = max(self.exclude_first_bins, int(np.searchsorted(tau, tmin, side="left")))

        noise_floor = np.median(amp[start:]) * 1.4826
        thr = max(self.prominence, self.amp_threshold_factor * noise_floor)

        cand = []
        for i in range(start+1, len(amp)-1):
            if amp[i] > amp[i-1] and amp[i] > amp[i+1] and amp[i] >= thr:
                cand.append(i)
        cand = sorted(cand, key=lambda i: amp[i], reverse=True)[:self.k_max]

        rows = [{"bin": int(i), "tau_cm": float(tau[i]),
                 "amplitude": float(amp[i]), "phase_rad": float(phase[i])} for i in cand]
        return pd.DataFrame(rows).sort_values("tau_cm").reset_index(drop=True)


# ----------------------- Sparse reconstruction -----------------------

def reconstruct_from_taus(nu: np.ndarray, y_target: np.ndarray,
                          taus: np.ndarray, ridge_alpha: float = 0.0):
    """
    Given fixed taus, fit y ≈ sum_i (a_i cos(2π τ_i ν) + b_i sin(2π τ_i ν))
    by linear least squares (optionally ridge). Returns per-component waves and sum.
    """
    nu = np.asarray(nu, float); y = np.asarray(y_target, float)
    taus = np.asarray(taus, float).ravel()
    if len(taus) == 0:
        return np.zeros((0, len(nu))), np.zeros_like(y), np.zeros((0,2))

    C = np.cos(2*np.pi*taus[:,None]*nu[None,:])  # [K, N]
    S = np.sin(2*np.pi*taus[:,None]*nu[None,:])

    # Design matrix: [cos blocks, sin blocks]
    X = np.concatenate([C, S], axis=0).T  # [N, 2K]

    if ridge_alpha > 0:
        # Ridge: solve (X^T X + αI)β = X^T y
        XtX = X.T @ X
        XtX.flat[::XtX.shape[0]+1] += ridge_alpha  # add α to diagonal
        beta = np.linalg.solve(XtX, X.T @ y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    a = beta[:len(taus)]
    b = beta[len(taus):]

    components = (a[:,None]*C + b[:,None]*S)  # [K, N]
    y_sum = components.sum(axis=0)

    coeffs = np.stack([a, b], axis=1)  # for optional inspection
    return components, y_sum, coeffs


# ----------------------- Main viz function -----------------------

def visualize_fft_decomposition(
    df: pd.DataFrame,
    include_range: Optional[Tuple[float, float]] = None,
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    k_components: int = 10,
    tukey_alpha: float = 0.5,
    zero_pad_factor: int = 8,
    tau_min: Optional[float] = None,
    amp_threshold_factor: float = 5.0,
    ridge_alpha: float = 0.0,
    uniform_points: Optional[int] = None,
    show_max_components_in_fig1: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run the Route-A pipeline and draw two figures:

    Fig 1: preprocessed (windowed) signal vs each selected component wave.
    Fig 2: preprocessed (windowed) signal vs sum of selected components.

    Returns a dict with all intermediates.
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError("DataFrame must have two numeric columns.")
    nu = df[num_cols[0]].to_numpy(float)
    R  = df[num_cols[1]].to_numpy(float)

    pre = SpectrumPreprocessor(
        window="tukey",
        tukey_alpha=tukey_alpha,
        include_range=include_range,
        exclude_ranges=exclude_ranges,
        baseline_poly_deg=3,
        detrend_mean=True,
    )
    nu_u, y_u, y_w, w, dnu, delta_nu = pre.run(nu, R, uniform_points=uniform_points)

    fft = FourierAnalyzer(zero_pad_factor=zero_pad_factor, pow2_pad=True)
    tau_axis, S, Npad = fft.run(y_w, dnu, w)

    picker = PeakPicker(
        k_max=k_components,
        tau_min=tau_min,
        amp_threshold_factor=amp_threshold_factor,
        exclude_first_bins=3,
    )
    peaks = picker.run(tau_axis, S, delta_nu=delta_nu)

    # sparse reconstruction on the *uniform* grid (length = len(nu_u))
    taus = peaks["tau_cm"].to_numpy() if not peaks.empty else np.array([])
    components, y_sum, coeffs = reconstruct_from_taus(nu_u, y_w, taus, ridge_alpha=ridge_alpha)

    # ----------------------- Plotting -----------------------
    # Figure 1: each component vs preprocessed
    plt.figure(figsize=(10, 4))
    plt.plot(nu_u, y_w, label="预处理后 (加窗)", linewidth=1.5)
    K = components.shape[0]
    max_show = K if show_max_components_in_fig1 is None else min(show_max_components_in_fig1, K)
    for i in range(max_show):
        plt.plot(nu_u, components[i], linewidth=0.8, label=f"分波 {i+1} (τ≈{taus[i]:.3g})")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Signal (a.u.)")
    plt.title("分波与预处理后信号")
    plt.legend(loc="best")
    plt.tight_layout()

    # Figure 2: sum of components vs preprocessed
    plt.figure(figsize=(10, 4))
    plt.plot(nu_u, y_w, label="预处理后 (加窗)", linewidth=1.5)
    if K > 0:
        plt.plot(nu_u, y_sum, label=f"前{K}个分波之和", linewidth=1.2)
        plt.title(f"分波之和 与 预处理后信号（K={K}）")
    else:
        plt.title("未选出任何分波（阈值过高或波段不合适）")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Signal (a.u.)")
    plt.legend(loc="best")
    plt.tight_layout()

    out = {
        "nu_uniform_cm-1": nu_u,
        "y_uniform": y_u,
        "y_windowed": y_w,
        "window": w,
        "dnu": dnu,
        "delta_nu": delta_nu,
        "tau_axis": tau_axis,
        "S_tau": S,
        "peaks": peaks,
        "taus_selected": taus,
        "components": components,  # shape [K, N]
        "sum_components": y_sum,
        "coeffs_cos_sin": coeffs,
    }
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

    visualize_fft_decomposition(df3, include_range=(1800,3300),
                                exclude_ranges=[(2200,2300)],
                                k_components=5, show_max_components_in_fig1=5)
    plt.show()
