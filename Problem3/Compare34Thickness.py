# compare_thickness.py
from pprint import pprint
import math

from Problem3.Calculate12Indicators import thickness_from_delta_nu
from Problem3.IndicatorsCalculator import PreprocessConfig, FFTMetricsConfig, IndicatorsRunner, FPParams, \
    IndicatorsConfig, PeakMetricsConfig
from Problem3.MultiLayerModel2 import FPFitConfig, MultiBeamThicknessFitter


# -----------------------------
# 对照计算器（硅片：附件3/4）
# -----------------------------
def compare_thickness_silicon(df3, df4,
                              n_for_two_beam: float = 3.42,
                              angle1_deg: float = 10.0,
                              angle2_deg: float = 15.0):
    """
    返回一个对照字典：
      - 指标法厚度（两束公式 + FFT Δν，分别对两角度 & 平均）
      - 多光束（TMM）联合拟合厚度（共享 d）
    参数：
      n_for_two_beam : 两束公式用的外延层折射率（硅常取 ~3.42）
      angle*_deg     : 两条谱的入射角
    """
    # 1) 指标法：各自跑一遍指标计算，拿 FFT Δν → d_est
    pre_cfg = PreprocessConfig(detrend=True, sg_window_frac=0.12, sg_polyorder=2, normalize_proc=True)
    peak_cfg = PeakMetricsConfig(prominence=0.25, width_rel_height=0.5, min_peaks=5, use_valleys=False)
    fft_cfg  = FFTMetricsConfig(min_periods=3.0, harmonic_max_order=4, band_frac=0.08)

    # 用 FPParams 只是为了 |G| 等指标，你不需要改它来影响厚度（厚度来自 Δν）
    ind3 = IndicatorsRunner(IndicatorsConfig(
        preprocess=pre_cfg, peak=peak_cfg, fft=fft_cfg,
        fp_params=FPParams(n0=1.0, n1=n_for_two_beam, n2=n_for_two_beam, theta0_deg=angle1_deg)
    )).run(df3)
    ind4 = IndicatorsRunner(IndicatorsConfig(
        preprocess=pre_cfg, peak=peak_cfg, fft=fft_cfg,
        fp_params=FPParams(n0=1.0, n1=n_for_two_beam, n2=n_for_two_beam, theta0_deg=angle2_deg)
    )).run(df4)

    d3 = thickness_from_delta_nu(ind3["fft_metrics"]["delta_nu_cm1"], n_for_two_beam, angle1_deg)["d_um"]
    d4 = thickness_from_delta_nu(ind4["fft_metrics"]["delta_nu_cm1"], n_for_two_beam, angle2_deg)["d_um"]
    d_fft_avg = (d3 + d4) / 2.0 if (math.isfinite(d3) and math.isfinite(d4)) else float("nan")

    # 2) 多光束 TMM：两角度共享 d 的联合拟合
    tmm_cfg = FPFitConfig(
        n0=1.0, n1=3.42, n2=3.50,            # 你之前用的硅 n 值；如有更准确的先验可替换
        theta1_deg=angle1_deg, theta2_deg=angle2_deg,
        pol="unpolarized",
        d_bounds_cm=(1e-6, 1e-2),            # [0.01 μm, 100 μm]
        alpha_bounds=(-2.0, 2.0),
        beta_bounds=(0.0, 10.0),
        verbose=False
    )
    tmm_res = MultiBeamThicknessFitter(tmm_cfg).fit(df3, df4)

    # 3) 打包汇总（便于直接打印或导出）
    out = {
        "two_beam_fft": {
            "delta_nu_cm1_angle10": ind3["fft_metrics"]["delta_nu_cm1"],
            "delta_nu_cm1_angle15": ind4["fft_metrics"]["delta_nu_cm1"],
            "d10_um": d3,
            "d15_um": d4,
            "d_avg_um": d_fft_avg,
            "notes": "两束公式 + FFT主频Δν；仅一阶参考"
        },
        "multibeam_TMM_joint": {
            "d_um": tmm_res["d_um"],
            "alpha_beta": tmm_res["params"],
            "residual_rms": tmm_res["residual_rms"],
            "fit_message": tmm_res["message"]
        }
    }
    return out


# -----------------------------
# 示例用法（你已有 DM）：
# -----------------------------
if __name__ == "__main__":
    from Data.DataManager import DM
    df3 = DM.get_data(3)  # 硅 @ 10°
    df4 = DM.get_data(4)  # 硅 @ 15°
    result = compare_thickness_silicon(df3, df4, n_for_two_beam=3.42, angle1_deg=10.0, angle2_deg=15.0)
    pprint(result)
