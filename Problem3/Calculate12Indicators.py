# -*- coding: utf-8 -*-
# run_sic_q3.py
import math
from pprint import pprint

from Data.DataManager import DM
from Problem3.IndicatorsCalculator import PreprocessConfig, PeakMetricsConfig, FFTMetricsConfig, IndicatorsConfig, \
    IndicatorsRunner, FPParams


# ---------- 小工具：由 Δν 估算厚度（两束公式，仅作一阶参考） ----------
def thickness_from_delta_nu(delta_nu_cm1: float, n: float, theta_i_deg: float) -> dict:
    """
    用两束公式估算厚度（单位：cm & μm）
        d = 1 / (2 n cos(theta_t) Δν),
        theta_t = asin(sin(theta_i)/n)
    """
    if not (delta_nu_cm1 and delta_nu_cm1 > 0):
        return {"d_cm": float("nan"), "d_um": float("nan")}
    theta_i = math.radians(theta_i_deg)
    theta_t = math.asin(min(1.0, max(-1.0, math.sin(theta_i) / n)))
    d_cm = 1.0 / (2.0 * n * math.cos(theta_t) * delta_nu_cm1)
    return {"d_cm": d_cm, "d_um": d_cm * 1.0e4}


# ---------- 公共配置（你也可以按需改动） ----------
# 预处理：轻度去趋势；FFT 用归一化信号更稳
pre_cfg = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=True)
# 峰参数：刚刚你用的参数
peak_cfg = PeakMetricsConfig(prominence=0.25, width_rel_height=0.5, min_peaks=5, use_valleys=False)
fft_cfg  = FFTMetricsConfig(min_periods=3.0, harmonic_max_order=4, band_frac=0.08)

# SiC 的 Fresnel 评估参数（仅用于 |G| 指标；不影响 FFT/峰相关指标）
# 外延层与衬底的折射率相近但非完全相等，这里给一个“略有失配”的示例
n_air = 1.0
n_sic_epi = 2.59   # 4H-SiC 外延层工程近似
n_sic_sub = 2.62   # 衬底略有不同（掺杂/应力/波段），你可按先验调整


def run_attachment(idx: int, angle_deg: float):
    """对某个附件（1或2）运行指标计算，并附带两束厚度的一阶参考值"""
    df = DM.get_data(idx)

    fp = FPParams(
        n0=n_air, n1=n_sic_epi, n2=n_sic_sub,
        theta0_deg=angle_deg,
        pol="unpolarized",
        use_absorption=False  # 如需考虑吸收，可设 True 并给 k1
    )

    cfg = IndicatorsConfig(
        preprocess=pre_cfg,
        peak=peak_cfg,
        fft=fft_cfg,
        fp_params=fp
    )

    runner = IndicatorsRunner(cfg)
    res = runner.run(df)

    # 顺带给一个两束厚度参考（来自 FFT 的 Δν）
    delta_nu = res["fft_metrics"].get("delta_nu_cm1", float("nan"))
    d_ref = thickness_from_delta_nu(delta_nu_cm1=delta_nu, n=n_sic_epi, theta_i_deg=angle_deg)

    # 打包输出
    out = {
        "angle_deg": angle_deg,
        "indicators": res,            # 三大指标：peaks / fft / fabry_perot_gain
        "thickness_ref_two_beam": d_ref  # 仅参考值（多光束时会有系统偏差）
    }
    return out


if __name__ == "__main__":
    # 附件1：SiC @ 10°
    res_att1 = run_attachment(idx=1, angle_deg=10.0)
    print("\n=== 附件1（SiC, 10°）指标与参考厚度 ===")
    pprint(res_att1)

    # 附件2：SiC @ 15°
    res_att2 = run_attachment(idx=2, angle_deg=15.0)
    print("\n=== 附件2（SiC, 15°）指标与参考厚度 ===")
    pprint(res_att2)

    # 也可以简单对比两个角度的 Δν 与 厚度参考
    d1 = res_att1["thickness_ref_two_beam"]["d_um"]
    d2 = res_att2["thickness_ref_two_beam"]["d_um"]
    if all(map(lambda x: isinstance(x, float) and math.isfinite(x), [d1, d2])):
        diff_pct = 100.0 * abs(d1 - d2) / ((d1 + d2) / 2.0)
        print(f"\n两角度厚度参考的相对差：{diff_pct:.3f}% （仅供参考）")
