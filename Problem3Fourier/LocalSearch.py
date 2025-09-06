from typing import Tuple, List, Sequence, Dict, Any

from Problem3Fourier.AiryMultiAngle import fit_single_angle, n_Si_dispersion_with_k, fit_multi_angle
from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


def get_d_d10_d15(include_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    返回多角拟合的厚度 d & 两个单角拟合的厚度 d10 & d15
    """
    # 你工程里的读取方式
    from Data.DataManager import DM
    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    # 1) 预处理（把这里替换为你的等间距波数 & 去基线/可用于拟合的反射信号）
    # 10°
    df = df3
    exclude_ranges: List[Tuple[float, float]] = []  # 强吸收段（可多段）
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
    nu_10 = out["nu_uniform"]  # cm^-1，等间距
    R10_meas = out["y_uniform_demean"]  # 对应反射率/信号（已去均值或基线）

    # 15°
    df = df4
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
    nu_15 = out["nu_uniform"]
    R15_meas = out["y_uniform_demean"]

    # 2) FFT 主峰得到的厚度初值（μm）——请替换为你的估计
    d0_um = 3.39

    # 3) 单角拟合（与多角保持相同的基线阶数！此处统一用二次）
    out10 = fit_single_angle(
        nu_10, R10_meas, d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55, theta_deg=10.0,
        poly_deg_baseline=1, verbose=True
    )
    out15 = fit_single_angle(
        nu_15, R15_meas, d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55, theta_deg=15.0,
        poly_deg_baseline=1, verbose=True
    )

    # 4) 多角联合拟合（与单角一致的二次基线；可选 sample_weighting="size" 按样本数加权）
    out_joint = fit_multi_angle(
        [(nu_10, R10_meas, 10.0), (nu_15, R15_meas, 15.0)],
        d0_um,
        n1=lambda wn: n_Si_dispersion_with_k(wn, k0=1e-3, kind="const"), n2=3.55,
        poly_deg_each_angle=1,
        force_positive_thickness=True,
        verbose=True,
        sample_weighting="mean",  # 或 "size"
    )

    print("d: ", {out_joint["d_um"]}, "d10: ", {out10["d_um"]}, "d15: ", {out15["d_um"]})
    return out_joint["d_um"], out10["d_um"], out15["d_um"]


from typing import Tuple, List, Optional, Dict
import math
import random

# ===============================
# 稳定性评估与局部搜索
# ===============================

class StabilityConfig:
    def __init__(
        self,
        bounds: Tuple[float, float] = (400.0, 4000.0),
        min_width: float = 50.0,              # 区间最小宽度，避免太窄导致过拟合/不稳
        delta_probe: float = 1.0,             # 你案例里的“y->y+1”的Δ
        jitter_count: int = 2,                # 额外做几组±Δ的扰动（增强鲁棒评估）
        eps_abs_d: float = 2.0,               # d 的绝对跳变容忍（单位：μm，可按你的量级调）
        eps_rel_d: float = 0.2,               # d 的相对跳变容忍（如 0.2 表示 20%）
        eps_pair_abs: float = 1.0,            # |d10-d15| 的绝对容忍（μm）
        eps_pair_rel: float = 0.15,           # |d10-d15| 的相对容忍（相对 d 的比例）
        eps_d_vs_angles_rel: float = 0.15,    # |d-mean(d10,d15)| 相对 d 的容忍
        objective_width_weight: float = 1.0,  # 目标里对“更宽区间”的偏好权重
        objective_penalty_weight: float = 10.0, # 目标里对“不稳定”的惩罚权重
        seed: int = 42
    ):
        self.bounds = bounds
        self.min_width = min_width
        self.delta_probe = delta_probe
        self.jitter_count = jitter_count
        self.eps_abs_d = eps_abs_d
        self.eps_rel_d = eps_rel_d
        self.eps_pair_abs = eps_pair_abs
        self.eps_pair_rel = eps_pair_rel
        self.eps_d_vs_angles_rel = eps_d_vs_angles_rel
        self.objective_width_weight = objective_width_weight
        self.objective_penalty_weight = objective_penalty_weight
        random.seed(seed)


def clip_range(r: Tuple[float, float], bounds: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    a, b = r
    lo, hi = bounds
    a = max(a, lo)
    b = min(b, hi)
    if b - a <= 0:
        return None
    return (a, b)


def is_valid(r: Tuple[float, float], cfg: StabilityConfig) -> bool:
    if r is None:
        return False
    a, b = r
    if b <= a:
        return False
    if (b - a) < cfg.min_width:
        return False
    lo, hi = cfg.bounds
    return (a >= lo) and (b <= hi)


def _safe_get(r: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
    try:
        return get_d_d10_d15(r)
    except Exception:
        # 如果拟合失败，视为不稳定
        return None


def _rel_error(a: float, b: float) -> float:
    denom = max(1e-9, abs(a))
    return abs(a - b) / denom


def evaluate_stability(
    r: Tuple[float, float],
    cfg: StabilityConfig
) -> Dict:
    """
    返回对区间 r 的稳定性评估:
    {
        "ok": bool,                   # 是否通过稳定性门槛
        "d": float, "d10": float, "d15": float,
        "max_d_jump_abs": float,      # 在微扰下 d 的最大绝对跳变
        "max_d_jump_rel": float,      # 在微扰下 d 的最大相对跳变
        "pair_gap_abs": float,        # |d10 - d15|
        "pair_gap_rel": float,        # |d10 - d15| / d
        "d_vs_angles_rel": float,     # |d - mean(d10,d15)| / d
        "objective": float            # 用于局部搜索的目标值（越大越好）
    }
    """
    if not is_valid(r, cfg):
        return {"ok": False, "objective": -float("inf")}

    base = _safe_get(r)
    if base is None:
        return {"ok": False, "objective": -float("inf")}
    d, d10, d15 = base
    d = float(d); d10 = float(d10); d15 = float(d15)

    # --- 一致性：角度间差异 & 与 d 的一致性 ---
    pair_gap_abs = abs(d10 - d15)
    pair_gap_rel = pair_gap_abs / max(1e-9, abs(d))
    d_vs_angles_rel = abs(d - 0.5*(d10 + d15)) / max(1e-9, abs(d))

    # --- 灵敏度：对端点做微小扰动，观察 d 的跳变 ---
    cand_ranges: List[Tuple[float, float]] = []
    a, b = r
    Δ = cfg.delta_probe

    # 基本的两侧“+Δ”探测（对应你举的 [x, y] → [x, y+1] 的例子）
    cand_ranges.append(clip_range((a, b + Δ), cfg.bounds))
    cand_ranges.append(clip_range((a - Δ, b), cfg.bounds))

    # 轻微收缩（可选）：有时候放大和缩小都看看更稳妥
    cand_ranges.append(clip_range((a, b - Δ), cfg.bounds))
    cand_ranges.append(clip_range((a + Δ, b), cfg.bounds))

    # 随机抖动（增强鲁棒性）
    for _ in range(cfg.jitter_count):
        da = random.choice([-Δ, Δ])
        db = random.choice([-Δ, Δ])
        cand_ranges.append(clip_range((a + da, b + db), cfg.bounds))

    d_jumps_abs: List[float] = []
    d_jumps_rel: List[float] = []

    for rr in cand_ranges:
        if rr is None:
            continue
        out = _safe_get(rr)
        if out is None:
            # 抖动后拟合失败 => 视为大跳变
            d_jumps_abs.append(float("inf"))
            d_jumps_rel.append(float("inf"))
            continue
        d2, _, _ = out
        d_jumps_abs.append(abs(d2 - d))
        d_jumps_rel.append(_rel_error(d, d2))

    max_d_jump_abs = max(d_jumps_abs) if d_jumps_abs else float("inf")
    max_d_jump_rel = max(d_jumps_rel) if d_jumps_rel else float("inf")

    # --- 判定是否“稳定” ---
    ok = (
        (max_d_jump_abs <= cfg.eps_abs_d or max_d_jump_rel <= cfg.eps_rel_d)
        and (pair_gap_abs <= cfg.eps_pair_abs or pair_gap_rel <= cfg.eps_pair_rel)
        and (d_vs_angles_rel <= cfg.eps_d_vs_angles_rel)
    )

    # --- 目标函数（越大越好）---
    # 惩罚不稳定；在稳定的情况下，鼓励更大的区间宽度
    width = b - a
    instability = (
        (max_d_jump_rel if math.isfinite(max_d_jump_rel) else 1e6)
        + pair_gap_rel
        + d_vs_angles_rel
    )
    objective = (
        (cfg.objective_width_weight * width)
        - (cfg.objective_penalty_weight * instability)
        + (0.0 if ok else -1e6)  # 不稳定直接重罚
    )

    return {
        "ok": ok,
        "d": d, "d10": d10, "d15": d15,
        "max_d_jump_abs": max_d_jump_abs,
        "max_d_jump_rel": max_d_jump_rel,
        "pair_gap_abs": pair_gap_abs,
        "pair_gap_rel": pair_gap_rel,
        "d_vs_angles_rel": d_vs_angles_rel,
        "objective": objective
    }


def local_search_range(
    init_range: Tuple[float, float],
    cfg: Optional[StabilityConfig] = None,
    step_init: float = 20.0,     # 初始步长（波数单位）
    step_min: float = 1.0,       # 最小步长
    patience: int = 10,          # 若连续若干次无改进则减小步长
    max_iters: int = 200,        # 最大迭代次数
    verbose: bool = True,
) -> Dict:
    """
    以坐标式局部搜索为主，四个方向（左端-、左端+、右端-、右端+）微调；
    找不到改进则缩步长，直到步长 < step_min 或达到迭代上限。
    返回内容包含最佳区间、评估结果和历史。
    """
    if cfg is None:
        cfg = StabilityConfig()

    best_r = clip_range(init_range, cfg.bounds)
    if not is_valid(best_r, cfg):
        raise ValueError("init_range 无效或小于最小宽度，请检查参数。")

    best_eval = evaluate_stability(best_r, cfg)
    history = [(best_r, best_eval)]
    step = step_init
    no_improve = 0

    if verbose:
        print(f"[INIT] range={best_r}, ok={best_eval['ok']}, obj={best_eval['objective']:.3f}, "
              f"d={best_eval['d']:.3f}, d10={best_eval['d10']:.3f}, d15={best_eval['d15']:.3f}")

    for it in range(1, max_iters + 1):
        a, b = best_r
        candidates = [
            (a - step, b),   # 往左扩
            (a + step, b),   # 往右缩（左端右移）
            (a, b + step),   # 往右扩
            (a, b - step),   # 往左缩（右端左移）
            (a - step/2, b + step/2),  # 居中放大一点
            (a + step/2, b - step/2),  # 居中收缩一点
        ]

        improved = False
        local_best_eval = None
        local_best_r = None

        for cand in candidates:
            rr = clip_range(cand, cfg.bounds)
            if not is_valid(rr, cfg):
                continue
            ev = evaluate_stability(rr, cfg)
            history.append((rr, ev))

            if verbose:
                print(f"  [TRY] range={rr}, ok={ev['ok']}, obj={ev['objective']:.3f}")

            if ev["objective"] > best_eval["objective"]:
                improved = True
                if (local_best_eval is None) or (ev["objective"] > local_best_eval["objective"]):
                    local_best_eval = ev
                    local_best_r = rr

        if improved and local_best_eval is not None:
            best_r = local_best_r
            best_eval = local_best_eval
            no_improve = 0
            if verbose:
                print(f"[IMPROVE@{it}] range={best_r}, ok={best_eval['ok']}, obj={best_eval['objective']:.3f}, "
                      f"width={best_r[1]-best_r[0]:.1f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                step /= 2.0
                no_improve = 0
                if verbose:
                    print(f"[SHRINK STEP] step -> {step}")
                if step < step_min:
                    if verbose:
                        print("[STOP] step < step_min")
                    break

    return {
        "best_range": best_r,
        "best_eval": best_eval,
        "history": history,
        "step_final": step,
    }


if __name__=="__main__":
    # 1) 设置初始区间（你说的预测试较稳的区间）
    init_range = (1800.0, 2500.0)

    # 2) 可选：自定义稳定性阈值/偏好（如下只是示例，可按你的数据量级调整）
    cfg = StabilityConfig(
        bounds=(400.0, 4000.0),
        min_width=80.0,  # 如果你希望至少覆盖若干条纹，可适当增大
        delta_probe=1.0,  # 对应你举例的 y -> y+1
        jitter_count=2,  # 适度随机扰动，提升稳健性
        eps_abs_d=2.0,  # d 绝对跳变容忍（μm）
        eps_rel_d=0.2,  # d 相对跳变容忍（20%）
        eps_pair_abs=1.0,  # |d10-d15| 绝对容忍
        eps_pair_rel=0.15,  # |d10-d15| 相对 d 的容忍
        eps_d_vs_angles_rel=0.15,  # d 与两角一致性的容忍
        objective_width_weight=1.0,
        objective_penalty_weight=10.0,
    )

    # 3) 运行局部搜索
    result = local_search_range(
        init_range=init_range,
        cfg=cfg,
        step_init=20.0,  # 初始步长（以波数计），如果搜索较“僵”，可以稍微大一点
        step_min=1.0,  # 与 delta_probe 同量级即可
        patience=8,  # 若多次无改进就缩步长
        max_iters=200,
        verbose=True,  # 打印过程日志
    )

    # 4) 查看结果
    best_range = result["best_range"]
    best_eval = result["best_eval"]
    print("\n=== 最优区间与稳定性 ===")
    print("best_range:", best_range)
    print("ok:", best_eval["ok"])
    print("d, d10, d15:", best_eval["d"], best_eval["d10"], best_eval["d15"])
    print("max_d_jump_abs:", best_eval["max_d_jump_abs"])
    print("max_d_jump_rel:", best_eval["max_d_jump_rel"])
    print("pair_gap_abs:", best_eval["pair_gap_abs"])
    print("pair_gap_rel:", best_eval["pair_gap_rel"])
    print("d_vs_angles_rel:", best_eval["d_vs_angles_rel"])

