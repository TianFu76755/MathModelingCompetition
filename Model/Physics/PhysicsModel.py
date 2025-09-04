from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

from Model.Physics.TwoBeamPhysics import TwoBeamPhysics


# ---------------------------------
# 1) 预处理：统一网格 + 去趋势（轻度）
# ---------------------------------
@dataclass
class PreprocessConfig:
    detrend: bool = True
    sg_window_frac: float = 0.15   # Savitzky–Golay 窗口相对长度（相对数据点数，需>0且<1）
    sg_polyorder: int = 2          # SG 多项式阶数
    normalize: bool = True         # 是否把反射率映射到零均值单位幅度，便于FFT/拟合的数值稳定


class SpectrumPreprocessor:
    """
    将 (ν[cm^-1], R[%]) → 等间隔 ν 网格，并进行温和去趋势。
    """

    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg = cfg or PreprocessConfig()

    def _to_uniform_grid(self, nu: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """按原始平均间距构造等间隔波数栅格并线性插值"""
        # 升序
        idx = np.argsort(nu)
        nu = nu[idx]
        R = R[idx]

        # 构造等间隔网格
        dnu = np.mean(np.diff(nu))
        if not np.isfinite(dnu) or dnu <= 0:
            # 退化情况：仅返回原数据
            return nu, R

        nu_grid = np.arange(nu[0], nu[-1] + 0.5 * dnu, dnu)
        R_grid = np.interp(nu_grid, nu, R)
        return nu_grid, R_grid

    def _detrend(self, y: np.ndarray) -> np.ndarray:
        """Savitzky–Golay 低频趋势估计并相减"""
        if not self.cfg.detrend:
            return y.copy()

        n = len(y)
        if n < 11:
            return y - np.nanmean(y)

        # 窗口长度：基于样本数的奇数窗口
        w = max(5, int(round(self.cfg.sg_window_frac * n)))
        if w % 2 == 0:
            w += 1
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        w = max(w, self.cfg.sg_polyorder + 3)  # polyorder < window_length

        baseline = savgol_filter(y, window_length=w, polyorder=self.cfg.sg_polyorder, mode='interp')
        return y - baseline

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return y
        # 将信号做零均值+单位RMS的标准化，避免数值尺度影响 FFT/拟合
        y0 = y - np.mean(y)
        rms = np.sqrt(np.mean(y0 ** 2))
        return y0 / rms if rms > 0 else y0

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        输入：包含两列 [波数 (cm-1)], [反射率 (%)] 的 DataFrame
        输出字典：
          nu_grid: 等间隔 ν
          R_grid: 线性插值后的反射率(0~1, 非百分比)
          R_detrended: 去趋势后的信号（随后可能再归一化）
          R_proc: 归一化后的信号（用于 FFT / 拟合）
        """
        # 容错读取
        col_nu = "波数 (cm-1)"
        col_R = "反射率 (%)"
        if col_nu not in df.columns or col_R not in df.columns:
            # 退而求其次：取前两列
            col_nu, col_R = df.columns[:2]

        nu = df[col_nu].to_numpy(dtype=float)
        R_percent = df[col_R].to_numpy(dtype=float)
        R = R_percent / 100.0  # 变为 0~1

        nu_grid, R_grid = self._to_uniform_grid(nu, R)
        R_detrended = self._detrend(R_grid)
        R_proc = self._normalize(R_detrended)

        return dict(
            nu_grid=nu_grid,
            R_grid=R_grid,
            R_detrended=R_detrended,
            R_proc=R_proc,
        )


# -----------------------------------
# 2) 方法层：峰距(FFT)法 与 余弦拟合法
# -----------------------------------
@dataclass
class PhysicalParams:
    n: float = 2.59           # 4H-SiC 工程近似
    theta_i_deg: float = 0.0  # 入射角（空气侧），度


class BaseThicknessEstimator:
    """方法层基类：仅输出物理量，不做统计置信度。"""

    def __init__(self, phys: PhysicalParams):
        self.phys = phys
        # 预计算膜内角
        self.theta_t = TwoBeamPhysics.snell_theta_t(
            math.radians(self.phys.theta_i_deg), self.phys.n, n0=1.0
        )

    def estimate(self, nu_grid: np.ndarray, R_signal: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class PeakSpacingFFT(BaseThicknessEstimator):
    """
    峰距法（物理信号处理版）：用 FFT 在波数域估计条纹主频 f_ν，
    Δν ≈ 1 / f_ν -> d = 1/(2 n cosθ_t Δν)
    """

    def __init__(self, phys: PhysicalParams, min_vis_ratio: float = 0.01):
        super().__init__(phys)
        self.min_vis_ratio = min_vis_ratio

    def estimate(self, nu_grid: np.ndarray, R_signal: np.ndarray) -> Dict[str, Any]:
        # 采样间隔
        if len(nu_grid) < 8:
            raise ValueError("样本点过少，无法FFT。")
        dnu = np.mean(np.diff(nu_grid))
        if dnu <= 0:
            raise ValueError("波数轴异常（非升序或重复）。")

        # FFT 幅度谱
        Y = rfft(R_signal)
        freqs = rfftfreq(len(R_signal), d=dnu)  # 单位：cycles per cm^-1

        # 去掉 DC 与极低频（趋势残留），从一个小阈值开始搜索主峰
        low_cut = 1.0 / (nu_grid[-1] - nu_grid[0] + 1e-9) * 3.0  # 至少容纳 ~3 个条纹
        mask = freqs > low_cut
        if not np.any(mask):
            raise ValueError("频率范围不足以分辨条纹。")

        amp = np.abs(Y)[mask]
        fpos = freqs[mask]

        # 主峰频率
        k = np.argmax(amp)
        f_peak = fpos[k]  # cycles / (cm^-1)

        # 可选：检查可见度（FFT主峰相对能量占比）
        vis_ratio = amp[k] / (np.sum(amp) + 1e-12)
        if vis_ratio < self.min_vis_ratio:
            # 仅给出警告性质的标志位（统计层再统一处理）
            pass

        # Δν 与厚度
        delta_nu = 1.0 / f_peak
        d_cm = TwoBeamPhysics.thickness_from_delta_nu(delta_nu, self.phys.n, self.theta_t)

        return dict(
            method="fft_peakspacing",
            delta_nu_cm1=delta_nu,
            d_cm=d_cm,
            d_um=d_cm * 1.0e4,
            f_peak=f_peak,
            fft_visibility_ratio=vis_ratio,
            notes="FFT 主频估计 Δν；未包含统计置信度。",
        )


class CosineFitTwoBeam(BaseThicknessEstimator):
    """
    余弦全谱拟合法：R(ν) = A + B cos(4π n d cosθ_t * ν + φ0)
    仅拟合物理参数，不输出统计区间（留待后续）。
    """

    def __init__(self, phys: PhysicalParams, init_from_fft: bool = True):
        super().__init__(phys)
        self.init_from_fft = init_from_fft
        self._fft_helper = PeakSpacingFFT(phys)

    def _initial_guess(self, nu: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """
        生成 (A, B, d_cm, phi0) 初值：
          - A: 信号均值（使用去趋势前的 R_grid 更合理，但这里用 y 的均值亦可）
          - B: ~ 半峰谷幅
          - d: 来自 FFT 的初猜
          - φ0: 0
        """
        A0 = float(np.mean(y))
        B0 = 0.5 * float(np.max(y) - np.min(y))
        B0 = max(min(B0, abs(A0) + 1.0), 1e-3)  # 数值安全

        # d0：可使用 FFT 辅助
        d0_cm = 5e-4  # 50 μm 的保守初猜
        if self.init_from_fft:
            try:
                est = self._fft_helper.estimate(nu, y - np.mean(y))
                d0_cm = est["d_cm"]
            except Exception:
                pass

        phi0_0 = 0.0
        return A0, B0, d0_cm, phi0_0

    def estimate(self, nu_grid: np.ndarray, R_signal: np.ndarray) -> Dict[str, Any]:
        # 注意：拟合最好用“未归一化的去趋势前信号”或“轻度归一化信号”。
        # 这里假设输入 R_signal 已是“预处理返回的 R_grid 或 R_detrended 的合适选择”
        A0, B0, d0_cm, phi0_0 = self._initial_guess(nu_grid, R_signal)

        def model(nu, A, B, d_cm, phi0):
            return TwoBeamPhysics.two_beam_reflectance_cos(
                nu_cm1=nu,
                n=self.phys.n,
                d_cm=d_cm,
                theta_t_rad=self.theta_t,
                A=A,
                B=B,
                phi0=phi0
            )

        # 参数边界：A ∈ [min-幅度, max+幅度]；B ≥ 0；d ∈ (0, +inf)；φ0 ∈ [-π, π] 或自由
        y = R_signal.astype(float)
        ymin, ymax = float(np.min(y)), float(np.max(y))
        span = max(ymax - ymin, 1e-3)

        bounds_lower = [ymin - span, 0.0, 1e-6, -2.0 * math.pi]
        bounds_upper = [ymax + span, 5.0 * span, 1.0,  2.0 * math.pi]
        p0 = [A0, B0, d0_cm, phi0_0]

        popt, _pcov = curve_fit(model, nu_grid, y, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=20000)
        A_hat, B_hat, d_hat_cm, phi0_hat = [float(v) for v in popt]

        return dict(
            method="cosine_fit",
            A=A_hat,
            B=B_hat,
            phi0=phi0_hat,
            d_cm=d_hat_cm,
            d_um=d_hat_cm * 1.0e4,
            notes="两束余弦模型全谱拟合；未包含统计置信度。",
        )


# -------------------------------
# 3) 流水线：读取表格 → 预处理 → 估计
# -------------------------------
@dataclass
class PipelineConfig:
    method: str = "fft"  # "fft" or "fit"
    preprocess: PreprocessConfig = PreprocessConfig()
    phys: PhysicalParams = PhysicalParams()
    # 选择拟合时使用的信号：'grid'（R_grid）或 'detrended'（R_detrended）或 'proc'（R_proc）
    fit_signal: str = "detrended"


class ThicknessPipeline:
    """
    统一入口：
      - 输入：任一 DataFrame（两列：波数cm-1 & 反射率%）
      - 预处理：统一等间隔、去趋势/归一化
      - 方法：FFT 峰距 或 余弦全谱拟合
      - 输出：仅物理量（厚度等），统计层日后增配
    """
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.pre = SpectrumPreprocessor(self.cfg.preprocess)
        if self.cfg.method == "fft":
            self.estimator = PeakSpacingFFT(self.cfg.phys)
        elif self.cfg.method == "fit":
            self.estimator = CosineFitTwoBeam(self.cfg.phys)
        else:
            raise ValueError("未知方法，请使用 'fft' 或 'fit'。")

    def run_on_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        prep = self.pre.run(df)

        # 选择用于方法估计的信号
        if self.cfg.method == "fft":
            # FFT 推荐用 R_proc（零均值单位幅度）
            signal = prep["R_proc"]
        else:
            # 拟合更适合用“去趋势前的 R_grid”或“轻度去趋势后的 R_detrended”
            sel = self.cfg.fit_signal
            if sel == "grid":
                signal = prep["R_grid"]
            elif sel == "detrended":
                signal = prep["R_detrended"]
            elif sel == "proc":
                signal = prep["R_proc"]
            else:
                raise ValueError("fit_signal 取值应为 'grid'/'detrended'/'proc'。")

        est = self.estimator.estimate(prep["nu_grid"], signal)

        # 汇总（方便上层统一记录参数）
        result = dict(
            method=est.get("method"),
            d_cm=est.get("d_cm"),
            d_um=est.get("d_um"),
            meta=dict(
                n=self.cfg.phys.n,
                theta_i_deg=self.cfg.phys.theta_i_deg,
                theta_t_deg=math.degrees(TwoBeamPhysics.snell_theta_t(
                    math.radians(self.cfg.phys.theta_i_deg), self.cfg.phys.n, 1.0)),
                notes="仅物理方法；统计与可靠性评估后续补充。",
            ),
            details=dict(
                preprocess=prep,
                estimator_raw=est
            )
        )
        return result


# -------------------------------
# 4) 使用示例
# -------------------------------
if __name__ == "__main__":
    from Data.DataManager import DM
    # 假设你已有 DM.get_data(idx) 返回 DataFrame
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # 1) FFT 峰距法
    pipe_fft = ThicknessPipeline(PipelineConfig(
        method="fft",
        preprocess=PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2, normalize=True),
        phys=PhysicalParams(n=2.59, theta_i_deg=0.0)
    ))
    res1 = pipe_fft.run_on_df(df1)
    print(res1["d_um"], res1["meta"])

    # 2) 两束余弦拟合法（建议 fit_signal="detrended" 或 "grid" 试对比）
    pipe_fit = ThicknessPipeline(PipelineConfig(
        method="fit",
        preprocess=PreprocessConfig(detrend=True, sg_window_frac=0.15, sg_polyorder=2, normalize=False),
        phys=PhysicalParams(n=2.59, theta_i_deg=0.0),
        fit_signal="detrended"
    ))
    res2 = pipe_fit.run_on_df(df1)
    print(res2["d_um"], res2["meta"])
