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
    window: str = "hann"
    def run(self, nu: np.ndarray, R: np.ndarray, uniform_points: Optional[int] = None):
        nu = np.asarray(nu, dtype=float)
        R = np.asarray(R, dtype=float)
        x = (nu - nu.mean()) / (nu.std() + 1e-12)
        coeff = np.polyfit(x, R, deg=self.baseline_poly_deg)
        baseline = np.polyval(coeff, x)
        y = R - baseline
        if uniform_points is None:
            uniform_points = len(nu)
        nu_u = np.linspace(nu.min(), nu.max(), uniform_points)
        y_u = np.interp(nu_u, nu, y)
        if self.window.lower() == "hann":
            w = np.hanning(uniform_points)
        else:
            w = np.ones(uniform_points)
        y_w = y_u * w
        dnu = (nu_u[-1] - nu_u[0]) / (uniform_points - 1)
        return nu_u, y_w, w, dnu

@dataclass
class FourierAnalyzer:
    zero_pad_factor: int = 8
    def run(self, signal_uniform: np.ndarray, dnu: float):
        y = np.asarray(signal_uniform, dtype=float)
        N = len(y)
        Npad = int(max(1, self.zero_pad_factor) * N)
        S = np.fft.rfft(y, n=Npad)
        tau = np.fft.rfftfreq(Npad, d=dnu)
        return tau, S

@dataclass
class PeakPicker:
    k_max: int = 5
    tau_min: float = 0.0
    prominence: float = 0.0
    def run(self, tau: np.ndarray, S: np.ndarray) -> pd.DataFrame:
        amp = np.abs(S); phase = np.angle(S)
        start = max(1, int(np.searchsorted(tau, self.tau_min, side="left")))
        cand = []
        for i in range(start+1, len(amp)-1):
            if amp[i] > amp[i-1] and amp[i] > amp[i+1]:
                if self.prominence <= 0 or (amp[i] - 0.5*(amp[i-1]+amp[i+1])) >= self.prominence:
                    cand.append(i)
        cand = sorted(cand, key=lambda i: amp[i], reverse=True)[:self.k_max]
        rows = []
        for i in cand:
            rows.append({"bin": int(i), "tau_cm": float(tau[i]),
                         "amplitude": float(amp[i]), "phase_rad": float(phase[i])})
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
        nu_u, y_w, w, dnu = self.pre.run(nu, R, uniform_points=uniform_points)
        tau, S = self.fft.run(y_w, dnu)
        peaks = self.picker.run(tau, S)
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
        geometry=Geometry(angle_inc_deg=10, n_incident=2.55)
    )
    out = flow.run(df1)  # 执行整个 FFT 流程
    peaks = out["peaks"]  # τ-域峰表
    d_cm = flow.thickness_from_first_peak(peaks, out["nu_uniform_cm-1"])
    d_um = d_cm * 1e4
    print("Thickness ≈ %.3f μm" % d_um)
