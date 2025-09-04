# -*- coding: utf-8 -*-
"""
多光束（Fabry–Pérot / TMM）模型 —— 两角度联合拟合硅外延层厚度
- 物理模型：单层薄膜的传输矩阵法（支持 s/p & 非偏振；可选常数 n1 或 Cauchy 色散；可选常数吸收 k1）
- 观测模型：Y_j(ν) ≈ α_j + β_j * R_j(ν; d, n1(λ), k1) ，两角度共享 d，(α_j,β_j) 各角独立
- 拟合：最小二乘联合拟合；保留边界、合理论域
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from Problem3.IndicatorsCalculator import PreprocessConfig, SpectrumPreprocessor


# ============== 1) 光学与 TMM 工具 ==============

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def snell_theta_t(theta0_deg: float, n0: float, n1_real: float) -> float:
    """斯涅尔：入射角 theta0、折射率 n0 -> 层内角（弧度）。用 n1 的实部做折射角。"""
    t0 = math.radians(theta0_deg)
    s = (n0 / n1_real) * math.sin(t0)
    s = _clamp(s, -1.0, 1.0)
    return math.asin(s)

def _eta(pol: str, n: complex | np.ndarray, cos_theta: float | np.ndarray) -> np.ndarray:
    """
    光学波阻抗（相对阻抗），支持标量或数组。
    s:  η = n * cosθ
    p:  η = n / cosθ
    """
    cos_theta = np.asarray(cos_theta)
    if pol == "s":
        return np.asarray(n) * cos_theta
    else:  # "p"
        safe_cos = np.where(cos_theta > 1e-12, cos_theta, 1e-12)
        return np.asarray(n) / safe_cos


def _tmm_one_layer_R(
    nu_cm1: np.ndarray,
    n0: float,
    n1_complex: complex | np.ndarray,
    n2: float,
    d_cm: float,
    theta0_deg: float,
    pol: str
) -> np.ndarray:
    """
    单层薄膜的 TMM 反射率 R(ν)：s 或 p 偏振。
    - 允许 n1_complex 为标量或与 nu_cm1 同形状的数组（频率相关）
    """
    nu = np.asarray(nu_cm1)  # (N,)

    # 角度（入射与衬底角度用实数几何；膜内角对每个频点用 Re(n1) 计算）
    t0 = math.radians(theta0_deg)
    cos0 = math.cos(t0)

    # 衬底角度（常数 n2）
    s2 = (n0 / n2) * math.sin(t0)
    s2 = np.clip(s2, -1.0, 1.0)
    theta2 = math.asin(s2)
    cos2 = math.cos(theta2)

    # 外延层：n1 可以是标量或数组
    n1c = np.asarray(n1_complex)          # (,) 或 (N,)
    n1_re = np.real(n1c)

    # 逐点膜内角（用 Re(n1) 做几何）
    s1 = (n0 / n1_re) * math.sin(t0)      # (N,) 或标量
    s1 = np.clip(s1, -1.0, 1.0)
    theta1 = np.arcsin(s1)
    cos1 = np.cos(theta1)

    # 阻抗（支持数组）
    eta0 = _eta(pol, n0 + 0j, cos0)       # 标量
    eta1 = _eta(pol, n1c,    cos1)        # (N,)
    eta2 = _eta(pol, n2 + 0j, cos2)       # 标量

    # 相位：δ = 2π * n1 * d * cosθ1 * ν   （全部逐点广播）
    delta = 2.0 * math.pi * n1c * d_cm * cos1 * nu  # (N,), 复数

    cosd = np.cos(delta)
    sind = np.sin(delta)

    # 单层特征矩阵元素（逐点）
    M11 = cosd
    M12 = 1j * (1.0 / eta1) * sind
    M21 = 1j * eta1 * sind
    M22 = cosd

    # r = ((M11+M12*η2)η0 - (M21+M22*η2)) / ((M11+M12*η2)η0 + (M21+M22*η2))
    num = (M11 + M12 * eta2) * eta0 - (M21 + M22 * eta2)
    den = (M11 + M12 * eta2) * eta0 + (M21 + M22 * eta2)
    r = num / den
    R = np.abs(r) ** 2
    return np.real(R)



# ============== 2) 模型配置 ==============

@dataclass
class DispersionConfig:
    """
    色散模型选择：
    - mode = "const": n1 = const（默认）
    - mode = "cauchy": n1(λ_um) = A + B/λ^2 + C/λ^4 （常用于 Si 的近红外）
    - fit_k1: 是否拟合常数吸收 k1（否则固定为 0）
    """
    mode: str = "const"           # "const" 或 "cauchy"
    fit_k1: bool = False
    # 初值（const 模式）
    n1_init: float = 3.42         # Si 近红外的常见量级
    # 初值（cauchy）
    A_init: float = 3.42
    B_init: float = 0.0
    C_init: float = 0.0
    # 边界
    n1_bounds: Tuple[float, float] = (3.0, 4.5)
    A_bounds: Tuple[float, float] = (3.0, 4.5)
    B_bounds: Tuple[float, float] = (-5.0, 5.0)
    C_bounds: Tuple[float, float] = (-10.0, 10.0)
    k1_bounds: Tuple[float, float] = (0.0, 0.1)   # 常数吸收的简单上界


@dataclass
class TMMFitConfig:
    """
    拟合的总体配置（角度、介质、预处理、参数边界）
    """
    # 介质
    n0: float = 1.0
    n2: float = 3.42            # 衬底 Si 折射率（也可用文献曲线；这里给常数近似）
    # 角度
    theta_deg_1: float = 10.0   # 附件3
    theta_deg_2: float = 15.0   # 附件4
    # 预处理
    preprocess: PreprocessConfig = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=False)
    # 选择用于拟合的信号
    fit_signal: str = "grid"    # "grid"（原始R_grid）/ "detrended" / "proc"
    # 厚度边界
    d_bounds_cm: Tuple[float, float] = (1e-6, 1e-2)  # [0.01 μm, 100 μm]
    # 观测线性外包络参数边界（α、β）
    alpha_bounds: Tuple[float, float] = (-1.0, 1.0)  # R 在 [0,1] 范围，去趋势后 α 可能小
    beta_bounds: Tuple[float, float] = (0.0, 5.0)
    # 偏振
    polarization: str = "unpolarized"               # "s"/"p"/"unpolarized"
    # 色散配置
    dispersion: DispersionConfig = DispersionConfig()


# ============== 3) 多光束模型 + 两角联合拟合 ==============

class MultiBeamSiThicknessFitter:
    """
    多光束（TMM）模型 + 两角度联合拟合硅外延层厚度：
      - 共享 d（目标参数）
      - 可选：n1（常数）或 Cauchy(A,B,C)，以及常数吸收 k1
      - 每个角度有自己的一对 (α_j, β_j) 以吸收基线/增益差异
    """
    def __init__(self, cfg: Optional[TMMFitConfig] = None):
        self.cfg = cfg or TMMFitConfig()
        self.pre = SpectrumPreprocessor(self.cfg.preprocess)

    # ---------- n1(λ) 的工厂 ----------
    def _n1_complex_factory(self, dispersion: DispersionConfig) -> Callable[[np.ndarray, np.ndarray], complex]:
        """
        返回一个函数 f(nu_cm1, lam_um, params) -> n1_complex
        - const: params = (n1, k1?)
        - cauchy: params = (A, B, C, k1?)
        """
        if dispersion.mode.lower() == "const":
            def nfun(nu_cm1, lam_um, p):
                n1 = p[0]
                k1 = p[1] if dispersion.fit_k1 else 0.0
                return n1 - 1j * k1
            return nfun
        elif dispersion.mode.lower() == "cauchy":
            def nfun(nu_cm1, lam_um, p):
                A, B, C = p[0], p[1], p[2]
                k1 = p[3] if dispersion.fit_k1 else 0.0
                n1 = A + B / (lam_um**2 + 1e-30) + C / (lam_um**4 + 1e-30)
                return n1 - 1j * k1
            return nfun
        else:
            raise ValueError("dispersion.mode 必须为 'const' 或 'cauchy'。")

    # ---------- 观测模型 ----------
    @staticmethod
    def _obs_map(R: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        观测模型：Y_hat = α + β * R
        说明：若拟合信号选择 'detrended' 或 'proc'，α 的拟合值会接近 0，β 会吸收幅度差异
        """
        return alpha + beta * R

    # ---------- 预处理 ----------
    def _prepare(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """
        返回：两条谱的 ν 网格、拟合用信号 y、以及用于构造 n1(λ) 的 λ（μm）
        """
        prep1 = self.pre.run(df1)
        prep2 = self.pre.run(df2)

        key = {"grid": "R_grid", "detrended": "R_detrended", "proc": "R_proc"}.get(self.cfg.fit_signal, "R_grid")
        nu1, y1 = prep1["nu_grid"], prep1[key]
        nu2, y2 = prep2["nu_grid"], prep2[key]

        # λ = 1/ν（cm），换算到 μm：λ_um = (1/ν_cm^-1) * 1e4
        lam1_um = (1.0 / np.maximum(nu1, 1e-12)) * 1.0e4
        lam2_um = (1.0 / np.maximum(nu2, 1e-12)) * 1.0e4

        # 为了稳妥，保证两条 ν 轴的范围相近；若不一致也可直接分别计算（模型已按各自 ν）
        return dict(
            nu1=nu1, y1=y1, lam1_um=lam1_um,
            nu2=nu2, y2=y2, lam2_um=lam2_um
        )

    # ---------- 初值/边界 ----------
    def _init_and_bounds(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        构造参数向量 p、下上界 lb/ub、以及每个参数的名字（便于读结果）
        参数顺序：
            p = [ d,    # 共享厚度（cm）
                  (n1 或 A,B,C), (k1 可选),
                  α1, β1, α2, β2 ]
        """
        disp = self.cfg.dispersion
        names: List[str] = []
        p0: List[float] = []
        lb: List[float] = []
        ub: List[float] = []

        # 共享厚度 d
        p0.append(5e-4)                          # 50 μm 的保守初值；你也可换成 FFT 初值
        lb.append(self.cfg.d_bounds_cm[0])
        ub.append(self.cfg.d_bounds_cm[1])
        names.append("d_cm")

        # 折射率参数
        if disp.mode == "const":
            p0.append(disp.n1_init)
            lb.append(disp.n1_bounds[0]); ub.append(disp.n1_bounds[1]); names.append("n1")
            if disp.fit_k1:
                p0.append(0.0); lb.append(disp.k1_bounds[0]); ub.append(disp.k1_bounds[1]); names.append("k1")
        else:  # "cauchy"
            p0.extend([disp.A_init, disp.B_init, disp.C_init])
            lb.extend([disp.A_bounds[0], disp.B_bounds[0], disp.C_bounds[0]])
            ub.extend([disp.A_bounds[1], disp.B_bounds[1], disp.C_bounds[1]])
            names.extend(["A", "B", "C"])
            if disp.fit_k1:
                p0.append(0.0); lb.append(disp.k1_bounds[0]); ub.append(disp.k1_bounds[1]); names.append("k1")

        # 观测侧 α、β（两角分别）
        # α 初值取数据中位数（若是去趋势信号则接近 0），β 初值取 IQR 或 1.0
        for j, key_y in enumerate(["y1", "y2"], start=1):
            y = data[key_y]
            alpha0 = float(np.median(y))
            iqr = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
            beta0 = iqr if iqr > 1e-3 else 1.0
            p0.extend([alpha0, beta0])
            lb.extend([self.cfg.alpha_bounds[0], self.cfg.beta_bounds[0]])
            ub.extend([self.cfg.alpha_bounds[1], self.cfg.beta_bounds[1]])
            names.extend([f"alpha{j}", f"beta{j}"])

        return np.array(p0, dtype=float), np.array(lb, dtype=float), np.array(ub, dtype=float), names

    # ---------- 残差 ----------
    def _residuals(self, p: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """
        计算两条谱的残差向量（串联）
        """
        disp = self.cfg.dispersion
        n0, n2 = self.cfg.n0, self.cfg.n2
        pol = self.cfg.polarization.lower()

        # 解析参数向量
        idx = 0
        d_cm = float(p[idx]); idx += 1

        if disp.mode == "const":
            n1 = float(p[idx]); idx += 1
            k1 = float(p[idx]); idx += 1 if disp.fit_k1 else 0
            def n1_complex_fun(nu, lam):  # 常数
                return (n1 - 1j * k1) + 0.0 * nu
            par_names = ["n1"] + (["k1"] if disp.fit_k1 else [])
        else:
            A = float(p[idx]); B = float(p[idx+1]); C = float(p[idx+2]); idx += 3
            k1 = float(p[idx]); idx += 1 if disp.fit_k1 else 0
            def n1_complex_fun(nu, lam):  # Cauchy
                n = A + B / (lam**2 + 1e-30) + C / (lam**4 + 1e-30)
                return n - 1j * (k1 if disp.fit_k1 else 0.0)
            par_names = ["A", "B", "C"] + (["k1"] if disp.fit_k1 else [])

        alpha1, beta1 = float(p[idx]), float(p[idx+1]); idx += 2
        alpha2, beta2 = float(p[idx]), float(p[idx+1]); idx += 2

        # 数据
        nu1, y1, lam1_um = data["nu1"], data["y1"], data["lam1_um"]
        nu2, y2, lam2_um = data["nu2"], data["y2"], data["lam2_um"]

        # 计算 R_s、R_p 并做非偏振平均或取单偏振
        def R_model_for(nu, lam_um, theta_deg):
            n1c = n1_complex_fun(nu, lam_um)
            if pol == "s":
                Rs = _tmm_one_layer_R(nu, n0, n1c, n2, d_cm, theta_deg, "s")
                return Rs
            elif pol == "p":
                Rp = _tmm_one_layer_R(nu, n0, n1c, n2, d_cm, theta_deg, "p")
                return Rp
            else:
                Rs = _tmm_one_layer_R(nu, n0, n1c, n2, d_cm, theta_deg, "s")
                Rp = _tmm_one_layer_R(nu, n0, n1c, n2, d_cm, theta_deg, "p")
                return 0.5 * (Rs + Rp)

        R1 = R_model_for(nu1, lam1_um, self.cfg.theta_deg_1)
        R2 = R_model_for(nu2, lam2_um, self.cfg.theta_deg_2)

        y1_hat = self._obs_map(R1, alpha1, beta1)
        y2_hat = self._obs_map(R2, alpha2, beta2)

        resid = np.concatenate([y1_hat - y1, y2_hat - y2], axis=0)
        return resid

    # ---------- 对外主入口 ----------
    def fit_pair(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """
        输入：附件3/4的 DataFrame
        输出：厚度 d（cm/μm）、拟合的（n1 或 Cauchy 参数、k1）、每角 α/β、残差 RMS 等
        """
        data = self._prepare(df1, df2)
        p0, lb, ub, names = self._init_and_bounds(data)

        sol = least_squares(self._residuals, p0, bounds=(lb, ub),
                            args=(data,), method="trf", max_nfev=50000, verbose=0)

        p_opt = sol.x
        resvec = self._residuals(p_opt, data)
        n_tot = resvec.size
        rms_all = float(np.sqrt(np.mean(resvec**2)))

        # 拆回 y1/y2 的 RMS
        n1 = len(data["y1"]); n2 = len(data["y2"])
        rms_1 = float(np.sqrt(np.mean(resvec[:n1]**2)))
        rms_2 = float(np.sqrt(np.mean(resvec[n1:]**2)))

        # 组装输出字典
        out = {
            "success": bool(sol.success),
            "message": str(sol.message),
            "nfev": int(sol.nfev),
            "param_names": names,
            "param_values": {names[i]: float(p_opt[i]) for i in range(len(names))},
            "d_cm": float(p_opt[0]),
            "d_um": float(p_opt[0] * 1.0e4),
            "residual_rms": {"rms_joint": rms_all, "rms_angle1": rms_1, "rms_angle2": rms_2},
        }
        # 回传用于画图的拟合曲线（以便你可视化）
        # 构造一次预测
        disp = self.cfg.dispersion
        def predict_curves():
            # 解析参数（复用 _residuals 的解析逻辑较复杂，这里简单重建）
            idx = 0
            d_cm = float(p_opt[idx]); idx += 1
            if disp.mode == "const":
                n1 = float(p_opt[idx]); idx += 1
                k1 = float(p_opt[idx]); idx += 1 if disp.fit_k1 else 0
                def n1_complex_fun(nu, lam):  # 常数
                    return (n1 - 1j * k1) + 0.0 * nu
            else:
                A = float(p_opt[idx]); B = float(p_opt[idx+1]); C = float(p_opt[idx+2]); idx += 3
                k1 = float(p_opt[idx]); idx += 1 if disp.fit_k1 else 0
                def n1_complex_fun(nu, lam):  # Cauchy
                    n = A + B / (lam**2 + 1e-30) + C / (lam**4 + 1e-30)
                    return n - 1j * (k1 if disp.fit_k1 else 0.0)
            alpha1, beta1 = float(p_opt[idx]), float(p_opt[idx+1]); idx += 2
            alpha2, beta2 = float(p_opt[idx]), float(p_opt[idx+1]); idx += 2

            nu1, y1, lam1_um = data["nu1"], data["y1"], data["lam1_um"]
            nu2, y2, lam2_um = data["nu2"], data["y2"], data["lam2_um"]

            def R_model_for(nu, lam_um, theta_deg):
                n1c = n1_complex_fun(nu, lam_um)
                if self.cfg.polarization == "s":
                    Rs = _tmm_one_layer_R(nu, self.cfg.n0, n1c, self.cfg.n2, d_cm, theta_deg, "s")
                    return Rs
                elif self.cfg.polarization == "p":
                    Rp = _tmm_one_layer_R(nu, self.cfg.n0, n1c, self.cfg.n2, d_cm, theta_deg, "p")
                    return Rp
                else:
                    Rs = _tmm_one_layer_R(nu, self.cfg.n0, n1c, self.cfg.n2, d_cm, theta_deg, "s")
                    Rp = _tmm_one_layer_R(nu, self.cfg.n0, n1c, self.cfg.n2, d_cm, theta_deg, "p")
                    return 0.5 * (Rs + Rp)

            R1 = R_model_for(nu1, lam1_um, self.cfg.theta_deg_1)
            R2 = R_model_for(nu2, lam2_um, self.cfg.theta_deg_2)

            y1_hat = self._obs_map(R1, alpha1, beta1)
            y2_hat = self._obs_map(R2, alpha2, beta2)

            return dict(
                nu1=nu1, y1_data=y1, y1_fit=y1_hat, R1_model=R1,
                nu2=nu2, y2_data=y2, y2_fit=y2_hat, R2_model=R2,
            )

        out["fitted_curves"] = predict_curves()
        return out


# ============== 4) 使用示例（跑附件3/4） ==============
if __name__ == "__main__":
    from Data.DataManager import DM
    df3 = DM.get_data(3)  # 硅 @ 10°
    df4 = DM.get_data(4)  # 硅 @ 15°（同片）
    # 下面给出两种典型配置：常数折射率 / Cauchy 色散

    # --- A) 常数折射率 + 无吸收（最简） ---
    cfg_const = TMMFitConfig(
        n0=1.0, n2=3.42,
        theta_deg_1=10.0, theta_deg_2=15.0,
        preprocess=PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=False),
        fit_signal="grid",                # 推荐先在原始 R_grid 上拟合
        d_bounds_cm=(1e-6, 2e-3),         # [0.01 μm, 20 μm] 按经验收紧更稳
        alpha_bounds=(-0.5, 1.5),
        beta_bounds=(0.0, 5.0),
        polarization="unpolarized",
        dispersion=DispersionConfig(mode="const", fit_k1=False, n1_init=3.42,
                                    n1_bounds=(3.2, 3.7))
    )
    fitter = MultiBeamSiThicknessFitter(cfg_const)
    res_const = fitter.fit_pair(df3, df4)
    from pprint import pprint
    pprint(res_const)
    print(f"[CONST] d = {res_const['d_um']:.4f} μm, RMS(joint)={res_const['residual_rms']['rms_joint']:.4g}")

    # --- B) Cauchy 色散（A,B 可自由；C 固定 0） ---
    cfg_cauchy = TMMFitConfig(
        n0=1.0, n2=3.42,
        theta_deg_1=10.0, theta_deg_2=15.0,
        preprocess=PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=False),
        fit_signal="grid",
        d_bounds_cm=(1e-6, 2e-3),
        alpha_bounds=(-0.2, 0.2),
        beta_bounds=(0.2, 2.0),
        polarization="unpolarized",
        dispersion=DispersionConfig(
            mode="cauchy", fit_k1=False,
            A_init=3.42, B_init=0.0, C_init=0.0,
            A_bounds=(3.2, 3.7), B_bounds=(-2.0, 2.0), C_bounds=(-0.5, 0.5)
        )
    )
    fitter2 = MultiBeamSiThicknessFitter(cfg_cauchy)
    res_cauchy = fitter2.fit_pair(df3, df4)
    pprint(res_cauchy)
    print(f"[CAUCHY] d = {res_cauchy['d_um']:.4f} μm, RMS(joint)={res_cauchy['residual_rms']['rms_joint']:.4g}")
