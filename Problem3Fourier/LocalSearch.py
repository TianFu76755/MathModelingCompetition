from typing import Tuple, List

from Problem3Fourier.FFTFengJu import fft_analyze_and_plot
from Problem3Fourier.PreprocessAndPlotFunct import preprocess_and_plot_compare


def calculate_d(include_range):
    """
    根据 nu 和 y_w 计算得到的厚度 d（使用 FFT 峰聚法等方式）
    返回对应的 d 值
    """
    # 假设你有一个计算 d 的方法，这里直接调用
    # 使用 FFT 或其他方法分析波形并返回计算得到的厚度 d
    # 返回的 d 是一个浮动值
    exclude_ranges: List[Tuple[float, float]] = []  # 强吸收段

    # ============预处理阶段===========
    out = preprocess_and_plot_compare(
        df,
        include_range=include_range,  # 条纹最明显波段
        exclude_ranges=exclude_ranges,  # 强吸收段（可多段）
        tukey_alpha=0.5,  # Tukey 窗参数；设 0 关闭
        baseline_poly_deg=3,  # 基线多项式阶数
        uniform_points=None,  # 等间距采样点数（默认跟随数据）
        window_name="tukey",  # "tukey" / "hann" / "rect"
        show_windowed=True,  # 是否同时画“乘窗后”的曲线
    )
    nu_u = out["nu_uniform"]
    y_w = out["y_windowed"]

    # ============FFT 分析阶段===========
    out = fft_analyze_and_plot(
        nu_u, y_w,
        n=n, theta_deg=theta_deg,
        peak_count=5,  # 最多找5个峰
        min_prominence=0.05,  # 只标注强度≥5%的峰
        figsize=(9, 4.5),
        title=f"FFT (n={n}, theta={theta_deg}°)",
        xlim_um=(3, 25)
    )

    # 结果结构
    for i, p in enumerate(out["peaks"], 1):
        print(f"[{i}] T = {p['T_cm']:.6g} cm  |  d = {p['thickness_um']:.6g} μm  "
              f"| rel_strength={p['rel_strength']:.3f}")
    # 假设返回第一个峰的厚度作为 d 值
    if out["peaks"]:
        return out["peaks"][0]["thickness_um"]
    else:
        raise ValueError("未找到有效的 FFT 峰值，无法计算厚度 d。")


def stability_metric(d_new, d_old, stability_threshold=10):
    """
    判断扩展后的区间稳定性，如果d值变化大于稳定性阈值，则认为不稳定。
    stability_threshold：稳定性百分比阈值（例如10%）
    """
    stability_percentage = abs(d_new - d_old) / d_old * 100
    return stability_percentage < stability_threshold


def local_search_with_expansion(initial_range, large_step_size=100, small_step_size=20, stability_threshold=10):
    """
    使用局部搜索扩展初始区间，逐步增大区间直到找到稳定的边界。
    df: 数据
    initial_range: 初始区间 (lower, upper)
    large_step_size: 初始阶段扩展的步长
    small_step_size: 扩展不稳定时的精细步长
    stability_threshold: 稳定性阈值（百分比）
    """
    lower, upper = initial_range
    # 计算基准的 d 值
    d_base = calculate_d((lower, upper))  # 假设这里传入nu和y_w数据

    # 第一阶段：大步子扩展
    expanding = True
    step_size = large_step_size
    while expanding:
        # 向左扩展
        new_lower = lower - step_size
        d_left = calculate_d((lower, upper))  # 计算新的 d 值

        # 向右扩展
        new_upper = upper + step_size
        d_right = calculate_d((lower, upper))  # 计算新的 d 值

        # 检查稳定性
        left_stable = stability_metric(d_left, d_base, stability_threshold)
        right_stable = stability_metric(d_right, d_base, stability_threshold)

        if not left_stable or not right_stable:
            # 一旦发现不稳定，切换到小步子拓展
            step_size = small_step_size
            expanding = False
        else:
            # 如果扩展稳定，继续大步拓展
            lower = new_lower
            upper = new_upper

    # 第二阶段：小步子精细调整
    while step_size > 1:
        # 向左扩展
        new_lower = lower - step_size
        d_left = calculate_d((lower, upper))  # 计算新的 d 值

        # 向右扩展
        new_upper = upper + step_size
        d_right = calculate_d((lower, upper))  # 计算新的 d 值

        # 检查稳定性
        left_stable = stability_metric(d_left, d_base, stability_threshold)
        right_stable = stability_metric(d_right, d_base, stability_threshold)

        if not left_stable or not right_stable:
            break  # 如果不稳定，停止精细调整

        # 向左和向右扩展都保持稳定，继续调整
        lower = new_lower
        upper = new_upper

        # 精细调整步长
        step_size /= 2

    return lower, upper


# 主程序
if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()

    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)

    df = df1  # 选择要处理的数据
    n = 2.55
    theta_deg = 10.0

    # 初始区间 [2000, 2400]
    initial_range = (2000, 2400)
    # 大步子拓展步长
    large_step_size = 50
    # 小步子精细调整步长
    small_step_size = 5
    # 稳定性阈值（例如 10%）
    stability_threshold = 5

    # 执行局部搜索扩展
    best_lower, best_upper = local_search_with_expansion(
        initial_range, large_step_size, small_step_size, stability_threshold)

    print(f"最佳的扩展区间: [{best_lower}, {best_upper}]")
