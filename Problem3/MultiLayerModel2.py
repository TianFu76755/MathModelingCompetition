"""
让gpt重新写的模型
"""
# -*- coding: utf-8 -*-
"""
多光束 Fabry–Pérot 模型厚度拟合（硅外延层）
- 输入：附件3（10°）、附件4（15°）的光谱 DataFrame（列：波数 (cm-1), 反射率 (%)）
- 模型：TMM（传输矩阵法）描述薄膜干涉，拟合共享厚度 d
- 输出：厚度 d (cm/μm)、每角度的拟合参数（alpha, beta）、残差 RMS
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ========== 1. 配置类 ==========
@dataclass
class FPFitConfig:
    n0: float = 1.0        # 入射介质（空气）
    n1: float = 3.42       # 外延层折射率（可作为常数或后续扩展拟合色散）
    n2: float = 3.50       # 衬底折射率
    theta1_deg: float = 10.0   # 附件3入射角
    theta2_deg: float = 15.0   # 附件4入射角
    pol: str = "unpolarized"   # 偏振类型："s" / "p" / "unpolarized"
    d_bounds_cm: Tuple[float, float] = (1e-6, 1e-2)  # 厚度边界 (cm) = [0.01 μm, 100 μm]
    alpha_bounds: Tuple[float, float] = (-1.0, 1.0)  # 基线偏置
    beta_bounds: Tuple[float, float] = (0.0, 2.0)    # 缩放因子
    verbose: bool = True


# ========== 2. Fabry–Pérot 模型工具 ==========
class FabryPerotModel:
    """Fabry–Pérot 干涉的 TMM 实现"""

    @staticmethod
    def snell_theta(n_from: float, n_to: float, theta_from_rad: float) -> float:
        """斯涅尔定律求折射角"""
        s = (n_from / n_to) * math.sin(theta_from_rad)
        s = max(-1.0, min(1.0, s))  # 数值安全
        return math.asin(s)

    @staticmethod
    def _impedance(n: float, theta: float, pol: str) -> complex:
        """光学阻抗（s/p 偏振不同）"""
        if pol == "s":
            return n * math.cos(theta)
        elif pol == "p":
            return n / math.cos(theta)
        else:
            raise ValueError("pol must be 's' or 'p'")

    @staticmethod
    def reflectance_tmm(n0: float, n1: float, n2: float,
                        d_cm: float, theta0_deg: float, nu_cm1: np.ndarray,
                        pol: str = "unpolarized") -> np.ndarray:
        """
        传输矩阵法计算反射率
        n0, n1, n2: 折射率
        d_cm: 薄膜厚度 (cm)
        theta0_deg: 入射角 (度)
        nu_cm1: 波数数组
        pol: "s", "p", "unpolarized"
        """
        theta0 = math.radians(theta0_deg)
        theta1 = FabryPerotModel.snell_theta(n0, n1, theta0)
        theta2 = FabryPerotModel.snell_theta(n1, n2, theta1)

        def _R_for_pol(pol_one: str) -> np.ndarray:
            # 光学阻抗
            eta0 = FabryPerotModel._impedance(n0, theta0, pol_one)
            eta1 = FabryPerotModel._impedance(n1, theta1, pol_one)
            eta2 = FabryPerotModel._impedance(n2, theta2, pol_one)

            # 相位延迟 δ
            delta = 2 * math.pi * n1 * d_cm * math.cos(theta1) * nu_cm1

            # 特征矩阵
            M11 = np.cos(delta)
            M12 = 1j * np.sin(delta) / eta1
            M21 = 1j * eta1 * np.sin(delta)
            M22 = np.cos(delta)

            # 总矩阵（单层就是 M1）
            M11, M12, M21, M22 = M11, M12, M21, M22

            # 反射系数
            num = (M11 + M12 * eta2) * eta0 - (M21 + M22 * eta2)
            den = (M11 + M12 * eta2) * eta0 + (M21 + M22 * eta2)
            r = num / den
            return np.abs(r) ** 2  # 反射率

        if pol == "s":
            return _R_for_pol("s")
        elif pol == "p":
            return _R_for_pol("p")
        else:  # 非偏振：取平均
            return 0.5 * (_R_for_pol("s") + _R_for_pol("p"))


# ========== 3. 拟合类 ==========
class MultiBeamThicknessFitter:
    """
    多光束 Fabry–Pérot 模型下的两角度联合拟合
    - 参数: 厚度 d (共享)，每角的 alpha/beta
    """

    def __init__(self, cfg: Optional[FPFitConfig] = None):
        self.cfg = cfg or FPFitConfig()

    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """把 DataFrame 转为 (nu_cm1, R_obs)"""
        col_nu, col_R = df.columns[:2]
        nu = df[col_nu].to_numpy(dtype=float)
        R = df[col_R].to_numpy(dtype=float) / 100.0  # 转换为 0~1
        return nu, R

    def fit(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """联合拟合附件3（10°）和附件4（15°）"""
        nu1, R1 = self._prepare(df1)
        nu2, R2 = self._prepare(df2)

        # 初值：用FFT周期法估个 d0，这里简单给一个中间值
        d0 = 6e-4  # 6 μm (cm单位)

        # 初值的 alpha/beta
        a1_0, b1_0 = 0.0, 1.0
        a2_0, b2_0 = 0.0, 1.0

        p0 = np.array([d0, a1_0, b1_0, a2_0, b2_0], dtype=float)

        # 参数边界
        lb = [self.cfg.d_bounds_cm[0], *([self.cfg.alpha_bounds[0], self.cfg.beta_bounds[0]] * 2)]
        ub = [self.cfg.d_bounds_cm[1], *([self.cfg.alpha_bounds[1], self.cfg.beta_bounds[1]] * 2)]

        def residuals(p: np.ndarray) -> np.ndarray:
            d, a1, b1, a2, b2 = p.tolist()
            R1_model = FabryPerotModel.reflectance_tmm(
                self.cfg.n0, self.cfg.n1, self.cfg.n2, d, self.cfg.theta1_deg, nu1, self.cfg.pol
            )
            R2_model = FabryPerotModel.reflectance_tmm(
                self.cfg.n0, self.cfg.n1, self.cfg.n2, d, self.cfg.theta2_deg, nu2, self.cfg.pol
            )
            y1_fit = a1 + b1 * R1_model
            y2_fit = a2 + b2 * R2_model
            return np.concatenate([y1_fit - R1, y2_fit - R2])

        sol = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=20000, verbose=2 if self.cfg.verbose else 0)

        d, a1, b1, a2, b2 = sol.x.tolist()

        # 结果整理
        R1_fit = FabryPerotModel.reflectance_tmm(
            self.cfg.n0, self.cfg.n1, self.cfg.n2, d, self.cfg.theta1_deg, nu1, self.cfg.pol
        )
        R2_fit = FabryPerotModel.reflectance_tmm(
            self.cfg.n0, self.cfg.n1, self.cfg.n2, d, self.cfg.theta2_deg, nu2, self.cfg.pol
        )

        return dict(
            d_cm=d,
            d_um=d * 1e4,
            params=dict(alpha1=a1, beta1=b1, alpha2=a2, beta2=b2),
            residual_rms=dict(
                rms1=float(np.sqrt(np.mean((a1 + b1 * R1_fit - R1) ** 2))),
                rms2=float(np.sqrt(np.mean((a2 + b2 * R2_fit - R2) ** 2))),
                rms_joint=float(np.sqrt(np.mean(np.concatenate([
                    (a1 + b1 * R1_fit - R1), (a2 + b2 * R2_fit - R2)
                ]) ** 2)))
            ),
            success=sol.success,
            message=sol.message,
            nit=sol.nfev,
            fitted_curves=dict(
                nu1=nu1, R1_obs=R1, R1_fit=a1 + b1 * R1_fit,
                nu2=nu2, R2_obs=R2, R2_fit=a2 + b2 * R2_fit
            )
        )


if __name__=="__main__":
    from Data.DataManager import DM
    df3 = DM.get_data(3)  # 附件3 (10°)
    df4 = DM.get_data(4)  # 附件4 (15°)

    fitter = MultiBeamThicknessFitter(FPFitConfig())
    res = fitter.fit(df3, df4)

    from pprint import pprint

    pprint(res)
