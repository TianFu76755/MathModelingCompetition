# -*- coding: utf-8 -*-
"""
Robust bootstrap summary:
- clean_outliers(): 基于 MAD 或 IQR 的离群点剔除
- robust_bootstrap_summary(): 计算稳健统计量 + 绘图（Zoom-in 直方图、箱线+小提琴）

依赖：numpy, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List

from UpgradedModel.Sensitivity import fit_base, bootstrap_samples


# ---------- 1) 清洗：MAD / IQR 规则 ----------
def clean_outliers(
    samples: List[float],
    rule: str = "mad3",           # "mad3" or "iqr"
    center: str = "median",       # "median" or "mean"
    k: float = 3.0,               # mad3: 3×MAD；iqr: [Q1-k*IQR, Q3+k*IQR]
    hard_cap: Optional[Tuple[float, float]] = None  # 如 (0, 20) μm 的绝对截断
) -> Dict[str, Any]:
    """
    输入：一组 bootstrap 样本（厚度 d 的集合）
    返回：
      dict(clean=..., outliers=..., keep_mask=..., bounds=(lo,hi), rule_used=...)
    """
    x = np.asarray(samples, dtype=float).copy()
    n0 = len(x)
    if n0 == 0:
        raise ValueError("samples 为空")

    # 绝对截断（可选）：确保明显不可能的值被剔除（例如 d<0 或 >50 μm）
    if hard_cap is not None:
        lo_cap, hi_cap = hard_cap
        mask_hard = (x >= lo_cap) & (x <= hi_cap)
        x_hard = x[mask_hard]
    else:
        mask_hard = np.ones_like(x, dtype=bool)
        x_hard = x

    if len(x_hard) == 0:
        return dict(clean=np.array([]), outliers=x, keep_mask=np.zeros_like(x, bool),
                    bounds=(None, None), rule_used=f"hard_cap={hard_cap}")

    if center == "median":
        c = np.median(x_hard)
    else:
        c = np.mean(x_hard)

    if rule.lower().startswith("mad"):
        # MAD 估计尺度
        mad = 1.4826 * np.median(np.abs(x_hard - c)) + 1e-12
        lo = c - k * mad
        hi = c + k * mad
        rule_used = f"MAD×{k:.1f} (center={center})"
    else:
        q1, q3 = np.quantile(x_hard, [0.25, 0.75])
        iqr = (q3 - q1) + 1e-12
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        rule_used = f"IQR×{k:.1f}"

    # 综合 hard_cap
    if hard_cap is not None:
        lo = max(lo, lo_cap)
        hi = min(hi, hi_cap)

    keep_mask_soft = (x >= lo) & (x <= hi)
    keep_mask = mask_hard & keep_mask_soft

    clean = x[keep_mask]
    outliers = x[~keep_mask]
    return dict(clean=clean, outliers=outliers, keep_mask=keep_mask,
                bounds=(float(lo), float(hi)), rule_used=rule_used)


# ---------- 2) 稳健统计 + Zoom-in 绘图 ----------
def robust_bootstrap_summary(
    d_samples: List[float],
    d_base: float,
    rule: str = "mad3",
    k: float = 3.0,
    hard_cap: Optional[Tuple[float, float]] = (0.0, 50.0),  # 绝对合理范围（按项目需要调整）
    zoom_by: str = "percentile",   # "percentile" or "bounds"
    zoom_percentiles: Tuple[float, float] = (0.02, 0.98),  # 2%~98% 作为缩放范围
    bins: int = 40,
    savepath: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    返回：(raw_stats, clean_stats)
      raw_stats  : {'mean','std','q05','q95','n'}
      clean_stats: {'mean','std','q05','q95','n'}
    并绘制两幅图：Zoom-in 直方图、箱线+小提琴
    """
    x = np.asarray(d_samples, dtype=float)
    if len(x) == 0:
        raise ValueError("bootstrap 样本为空。")

    # 原始统计
    raw_mean = float(np.mean(x))
    raw_std  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    raw_q05, raw_q95 = [float(q) for q in np.quantile(x, [0.05, 0.95])]

    # 清洗
    cleaned = clean_outliers(x, rule=rule, center="median", k=k, hard_cap=hard_cap)
    xc = cleaned["clean"]
    lo, hi = cleaned["bounds"]
    rule_used = cleaned["rule_used"]

    clean_mean = float(np.mean(xc)) if len(xc) else np.nan
    clean_std  = float(np.std(xc, ddof=1)) if len(xc) > 1 else 0.0
    clean_q05, clean_q95 = [float(q) for q in (np.quantile(xc, [0.05, 0.95]) if len(xc) else [np.nan, np.nan])]

    # ---- 绘图：Zoom-in 直方图（只看主体），并标注 base/均值/区间 ----
    if zoom_by == "percentile":
        zmin, zmax = np.quantile(xc if len(xc) else x, list(zoom_percentiles))
    else:
        zmin, zmax = lo, hi
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
        zmin, zmax = np.min(x), np.max(x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    # 只对 zoom 范围内的数据画直方
    x_zoom = x[(x >= zmin) & (x <= zmax)]
    ax.hist(x_zoom, bins=bins)
    ax.axvline(d_base, linestyle='--', label=f"base={d_base:.3f}")
    if len(xc):
        ax.axvline(clean_mean, linestyle=':', label=f"mean(clean)={clean_mean:.3f}")
        ax.axvline(clean_q05, linestyle='-.', lw=1.0, label=f"q05={clean_q05:.3f}")
        ax.axvline(clean_q95, linestyle='-.', lw=1.0, label=f"q95={clean_q95:.3f}")
    ax.set_xlim(zmin, zmax)
    ax.set_xlabel("d (μm)")
    ax.set_ylabel("频数")
    ax.set_title("Bootstrap 直方图（Zoom-in 主体）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 右图：箱线 + 小提琴（清洗后）
    ax2 = axes[1]
    if len(xc):
        ax2.violinplot([xc], showmeans=True, showextrema=True)
        ax2.boxplot([xc], widths=0.2, positions=[1.0])
        ax2.set_xticks([1]); ax2.set_xticklabels(["clean d"])
        ax2.set_title("清洗后分布（小提琴 + 箱线）")
    else:
        ax2.text(0.1, 0.5, "无清洗后样本", transform=ax2.transAxes)
    ax2.grid(alpha=0.3)

    supt = (f"原始: mean={raw_mean:.4f}, std={raw_std:.4f}, "
            f"[5%,95%]=[{raw_q05:.4f},{raw_q95:.4f}] μm | "
            f"清洗({rule_used}; cap={hard_cap}): "
            f"mean={clean_mean:.4f}, std={clean_std:.4f}, "
            f"[5%,95%]=[{clean_q05:.4f},{clean_q95:.4f}] μm")
    fig.suptitle(supt, y=1.02, fontsize=10)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)

    raw_stats = dict(mean=raw_mean, std=raw_std, q05=raw_q05, q95=raw_q95, n=int(len(x)))
    clean_stats = dict(mean=clean_mean, std=clean_std, q05=clean_q05, q95=clean_q95, n=int(len(xc)))
    return raw_stats, clean_stats


if __name__=="__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)

    # --- 基线 ---
    base = fit_base(df1, df2, n_kind="constant", n0=2.60, nu_window=(1200, 3800))
    samples = bootstrap_samples(df1, df2, base, keep_ratio=0.8, n_boot=600)

    d_base = float(base["d_um"])
    raw_stats, clean_stats = robust_bootstrap_summary(
        d_samples=samples,
        d_base=d_base,
        rule="mad3",  # MAD×3 规则
        k=3.0,
        hard_cap=(0.0, 50.0),  # 绝对合理范围（按你的工艺设定可调）
        zoom_by="percentile",  # 用2%~98%分位缩放显示主体
        zoom_percentiles=(0.02, 0.98),
        bins=40,
        savepath=None
    )
    plt.show()
    print("Raw:", raw_stats)
    print("Clean:", clean_stats)