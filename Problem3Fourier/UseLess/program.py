"""
SiC/Silicon Epitaxy FFT Toolkit
===============================
Classes:
- DispersionModel, Geometry, SpectrumPreprocessor, FourierAnalyzer,
  PeakPicker, ThicknessEstimator, EpiFFTWorkflow
See docstrings for details.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

@dataclass
class DispersionModel:
    a0: float = 2.5610
    a2: float = 3.4e-10
    def n(self, nu: np.ndarray) -> np.ndarray:
        nu = np.asarray(nu, dtype=float)
        return self.a0 + self.a2 * nu**2
    def ng(self, nu: np.ndarray) -> np.ndarray:
        nu = np.asarray(nu, dtype=float)
        return self.a0 + 3.0 * self.a2 * nu**2

@dataclass
class Geometry:
    angle_inc_deg: float = 10.0
    n_incident: float = 1.0
    def theta2(self, n2: float) -> float:
        theta1 = np.deg2rad(self.angle_inc_deg)
        s = np.sin(theta1) * self.n_incident / n2
        s = np.clip(s, -1.0, 1.0)
        return float(np.arcsin(s))
    def cos_theta2(self, n2: float) -> float:
        return float(np.cos(self.theta2(n2)))

@dataclass
class SpectrumPreprocessor:
    baseline_poly_deg: int = 3
    window: str = "tukey"              # 改：默认 tukey
    tukey_alpha: float = 0.5           # Tukey 窗参数，0~1
    include_range: Optional[Tuple[float, float]] = None  # 只在该范围做 FFT
    exclude_ranges: Optional[List[Tuple[float, float]]] = None  # 要屏蔽的强吸收段
    detrend_mean: bool = True          # 加窗前再减一次均值（防残余 DC）

    def _mask_ranges(self, nu: np.ndarray) -> np.ndarray:
        m = np.ones_like(nu, dtype=bool)
        if self.include_range is not None:
            lo, hi = self.include_range
            m &= (nu >= lo) & (nu <= hi)
        if self.exclude_ranges:
            for lo, hi in self.exclude_ranges:
                m &= ~((nu >= lo) & (nu <= hi))
        return m

    def run(self, nu: np.ndarray, R: np.ndarray, uniform_points: Optional[int] = None):
        nu = np.asarray(nu, dtype=float)
        R  = np.asarray(R,  dtype=float)

        # 1) 基线去除（保留条纹）
        x = (nu - nu.mean()) / (nu.std() + 1e-12)
        coeff = np.polyfit(x, R, deg=self.baseline_poly_deg)
        baseline = np.polyval(coeff, x)
        y = R - baseline

        # 2) 选择“干净条纹”波段；若出现不连续，取最长的连续段
        m = self._mask_ranges(nu)
        if not np.any(m):
            raise ValueError("Mask removes all points. Check include/exclude ranges.")
        nu_m, y_m = nu[m], y[m]
        # 若掩蔽造成断裂，自动取最长连续段
        gaps = np.where(np.diff(nu_m) > 3*np.median(np.diff(nu_m)))[0]  # 粗略找断点
        seg_starts = np.r_[0, gaps+1]; seg_ends = np.r_[gaps, len(nu_m)-1]
        lengths = seg_ends - seg_starts + 1
        k = int(np.argmax(lengths))
        i0, i1 = int(seg_starts[k]), int(seg_ends[k])
        nu_m, y_m = nu_m[i0:i1+1], y_m[i0:i1+1]

        # 3) 等间隔重采样（只在选中的段上）
        if uniform_points is None:
            uniform_points = len(nu_m)
        nu_u = np.linspace(nu_m.min(), nu_m.max(), uniform_points)
        y_u  = np.interp(nu_u, nu_m, y_m)

        # 4) 加窗 + 去均值（强建议）
        if self.detrend_mean:
            y_u = y_u - y_u.mean()
        wname = self.window.lower()
        if wname == "tukey":
            # numpy 没自带 tukey，手写一个简单版
            N = uniform_points
            a = float(self.tukey_alpha)
            n = np.arange(N)
            w = np.ones(N)
            if a > 0:
                edge = int(np.floor(a*(N-1)/2.0))
                ramp = 0.5*(1 + np.cos(np.pi*(2*n/(a*(N-1)) - 1)))
                w[:edge+1] = ramp[:edge+1]
                w[-edge-1:] = ramp[:edge+1][::-1]
        elif wname == "hann":
            w = np.hanning(uniform_points)
        else:
            w = np.ones(uniform_points)
        y_w = y_u * w

        dnu = (nu_u[-1] - nu_u[0]) / (uniform_points - 1)
        # 同时返回“有效带宽”，供 tau_min 自动化使用
        delta_nu = nu_u.max() - nu_u.min()
        return nu_u, y_w, w, dnu, delta_nu

@dataclass
class FourierAnalyzer:
    zero_pad_factor: int = 8
    pow2_pad: bool = True  # 可选：pad 到 2 的幂，速度更快

    def run(self, signal_uniform: np.ndarray, dnu: float, window: np.ndarray):
        y = np.asarray(signal_uniform, dtype=float)
        N = len(y)
        Npad = int(max(1, self.zero_pad_factor) * N)
        if self.pow2_pad:
            # pad 到最近的 2 的幂
            Npad = 1 << (int(np.ceil(np.log2(max(2, Npad)))))

        # 关键：乘以 dnu 并按窗 RMS 做幅值归一，便于阈值与跨样本比较
        S = np.fft.rfft(y, n=Npad) * dnu / np.sqrt((window**2).mean())
        tau = np.fft.rfftfreq(Npad, d=dnu)  # τ 轴（cm）
        return tau, S


@dataclass
class PeakPicker:
    k_max: int = 5
    tau_min: Optional[float] = None     # 改：允许自动
    prominence: float = 0.0
    amp_threshold_factor: float = 5.0   # ≥ 噪声地板的倍数
    exclude_first_bins: int = 3         # 丢掉最靠近 DC 的若干 bin

    def run(self, tau: np.ndarray, S: np.ndarray, delta_nu: Optional[float] = None) -> pd.DataFrame:
        amp = np.abs(S); phase = np.angle(S)

        # 自动 tau_min：1/Δν（经验值），再排除前几个 DC 邻近 bin
        if self.tau_min is None and (delta_nu is not None and delta_nu > 0):
            tau_min = 1.0 / float(delta_nu)
        else:
            tau_min = float(self.tau_min or 0.0)

        start = max(self.exclude_first_bins, int(np.searchsorted(tau, tau_min, side="left")))

        # 粗略噪声地板：用中位数绝对偏差估计
        noise_floor = np.median(amp[start:]) * 1.4826  # ~MAD→σ 的系数
        thr = max(self.prominence, self.amp_threshold_factor * noise_floor)

        cand = []
        for i in range(start+1, len(amp)-1):
            if amp[i] > amp[i-1] and amp[i] > amp[i+1] and amp[i] >= thr:
                cand.append(i)

        cand = sorted(cand, key=lambda i: amp[i], reverse=True)[:self.k_max]
        rows = [{"bin": int(i), "tau_cm": float(tau[i]),
                 "amplitude": float(amp[i]), "phase_rad": float(phase[i])} for i in cand]
        return pd.DataFrame(rows).sort_values("tau_cm").reset_index(drop=True)


@dataclass
class ThicknessEstimator:
    dispersion: DispersionModel
    geometry: Geometry
    def d_from_tau(self, tau_cm: float, nu_center: float) -> float:
        ng = float(self.dispersion.ng(np.array([nu_center]))[0])
        n2 = float(self.dispersion.n(np.array([nu_center]))[0])
        cos_t2 = self.geometry.cos_theta2(n2)
        return tau_cm / (2.0 * ng * cos_t2)

@dataclass
class EpiFFTWorkflow:
    dispersion: DispersionModel = field(default_factory=DispersionModel)
    geometry: Geometry = field(default_factory=Geometry)
    pre: SpectrumPreprocessor = field(default_factory=SpectrumPreprocessor)
    fft: FourierAnalyzer = field(default_factory=FourierAnalyzer)
    picker: PeakPicker = field(default_factory=PeakPicker)
    estimator: ThicknessEstimator = None

    def __post_init__(self):
        if self.estimator is None:
            self.estimator = ThicknessEstimator(self.dispersion, self.geometry)

    def run(self, df: pd.DataFrame, uniform_points: Optional[int] = None):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            raise ValueError("DataFrame must have two numeric columns.")
        nu = df[num_cols[0]].to_numpy(float)
        R  = df[num_cols[1]].to_numpy(float)

        nu_u, y_w, w, dnu, delta_nu = self.pre.run(nu, R, uniform_points=uniform_points)
        tau, S = self.fft.run(y_w, dnu, w)
        peaks = self.picker.run(tau, S, delta_nu=delta_nu)

        return {"nu_uniform_cm-1": nu_u, "signal_windowed": y_w, "window": w,
                "dnu": dnu, "tau_cm": tau, "S_tau": S, "peaks": peaks}
    def thickness_from_first_peak(self, peaks: pd.DataFrame, nu_uniform: np.ndarray) -> float:
        if peaks.empty:
            raise ValueError("No peaks found.")
        tau1 = float(peaks.iloc[0]["tau_cm"])
        nu_center = float(0.5 * (nu_uniform.min() + nu_uniform.max()))
        return self.estimator.d_from_tau(tau1, nu_center)


if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    flow = EpiFFTWorkflow(
        pre=SpectrumPreprocessor(
            window="tukey", tukey_alpha=0.5,
            # 只在条纹明显的波段做 FFT（示例，按你数据改）
            include_range=(1800, 3300),
            # 强吸收段一律剔除（示例：900–1100 cm⁻1）
            exclude_ranges=[(900, 1100)],
            baseline_poly_deg=3,
            detrend_mean=True,
        ),
        fft=FourierAnalyzer(zero_pad_factor=8, pow2_pad=True),
        picker=PeakPicker(
            k_max=5,
            tau_min=None,  # 自动=1/Δν
            amp_threshold_factor=5.0,  # ≥ 噪声 5 倍才记为峰
            exclude_first_bins=3,
        ),
        geometry=Geometry(angle_inc_deg=10, n_incident=2.55),
    )
    out = flow.run(df1)
    peaks = out["peaks"]
    d_cm = flow.thickness_from_first_peak(peaks, out["nu_uniform_cm-1"])
    print("Thickness ≈ %.3f μm" % (d_cm * 1e4))

