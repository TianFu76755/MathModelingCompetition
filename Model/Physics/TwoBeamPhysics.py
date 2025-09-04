import math

import numpy as np


# -----------------------------
# 0) 物理常量/工具：Two-beam 模型
# -----------------------------
class TwoBeamPhysics:
    """
    两束（单次反/透）干涉模型的物理工具：
      - 斯涅尔定律求膜内角 theta_t
      - 由 Δν 求厚度 d
      - 两束干涉的余弦谱模型 R(ν) = A + B cos(4π n d cosθ_t * ν + φ0)
    """

    @staticmethod
    def snell_theta_t(theta_i_rad: float, n: float, n0: float = 1.0) -> float:
        """斯涅尔定律：n0*sin(theta_i) = n*sin(theta_t) -> theta_t"""
        s = (n0 / n) * math.sin(theta_i_rad)
        # 数值安全钳制
        s = max(-1.0, min(1.0, s))
        return math.asin(s)

    @staticmethod
    def thickness_from_delta_nu(delta_nu_cm1: float, n: float, theta_t_rad: float) -> float:
        """
        d = 1 / (2 n cos(theta_t) Δν)
        返回 d 的单位：cm
        """
        if delta_nu_cm1 <= 0:
            raise ValueError("Δν 必须为正。")
        denom = 2.0 * n * math.cos(theta_t_rad) * delta_nu_cm1
        if denom <= 0:
            raise ValueError("分母无效，请检查 n、theta_t、Δν。")
        return 1.0 / denom

    @staticmethod
    def two_beam_reflectance_cos(nu_cm1: np.ndarray,
                                 n: float,
                                 d_cm: float,
                                 theta_t_rad: float,
                                 A: float,
                                 B: float,
                                 phi0: float) -> np.ndarray:
        """
        两束干涉近似反射谱：R(ν) = A + B cos( 4π n d cosθ_t * ν + φ0 )
        说明：此处 A、B 为工程拟合参数（非严格菲涅耳分解）
        """
        arg = 4.0 * math.pi * n * d_cm * math.cos(theta_t_rad) * nu_cm1 + phi0
        return A + B * np.cos(arg)

    # -------- 新增的便捷函数 --------
    @staticmethod
    def compute_thickness(theta_i_deg: float, n: float, delta_nu_cm1: float,
                          n0: float = 1.0) -> dict:
        """
        根据入射角(度)、折射率n和条纹间隔Δν(cm^-1)，计算折射角θ_t和厚度d。
        返回字典：theta_t_rad, theta_t_deg, d_cm, d_um
        """
        theta_i_rad = math.radians(theta_i_deg)
        theta_t_rad = TwoBeamPhysics.snell_theta_t(theta_i_rad, n, n0=n0)
        d_cm = TwoBeamPhysics.thickness_from_delta_nu(delta_nu_cm1, n, theta_t_rad)

        return dict(
            theta_t_rad=theta_t_rad,
            theta_t_deg=math.degrees(theta_t_rad),
            d_cm=d_cm,
            d_um=d_cm * 1.0e4
        )
