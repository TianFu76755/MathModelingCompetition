import numpy as np
import matplotlib.pyplot as plt

from Problem3Fourier.UseLess.program import EpiFFTWorkflow, PeakPicker


def go_fft_decomposition(df, angle_deg=10.0, k_peaks=3):
    """
    执行 FFT 分解并绘制图像:
      1) 每个单独峰重构的波形 vs 原始信号
      2) 所有峰相加的重构波形 vs 原始信号
    """
    # 运行全流程
    flow = EpiFFTWorkflow()
    out = flow.run(df)
    nu_u = out["nu_uniform_cm-1"]
    y_u = out["signal_windowed"]  # 已去基线并加窗
    S = out["S_tau"]
    tau = out["tau_cm"]

    # 找主峰
    picker = PeakPicker(k_max=k_peaks, tau_min=0.0)
    peaks = picker.run(tau, S)

    fig, axes = plt.subplots(k_peaks + 1, 1, figsize=(8, 2.5 * (k_peaks + 1)), sharex=True)

    # 逐个峰重构
    y_components = []
    Npad = len(S) * 2 - 2  # rfft 的长度关系
    for idx, row in peaks.iterrows():
        b = int(row["bin"])
        # 保留该 bin, 其它设为 0
        S_sel = np.zeros_like(S, dtype=complex)
        S_sel[b] = S[b]
        y_rec = np.fft.irfft(S_sel, n=Npad)[:len(nu_u)]
        y_components.append(y_rec)

        axes[idx].plot(nu_u, y_u, 'k-', lw=1, label='原始(去基线)')
        axes[idx].plot(nu_u, y_rec, 'r-', lw=1.2, label=f'单峰重构 (τ={row["tau_cm"]:.2e} cm)')
        axes[idx].legend()

    # 所有峰相加
    y_sum = np.sum(y_components, axis=0)
    axes[-1].plot(nu_u, y_u, 'k-', lw=1, label='原始(去基线)')
    axes[-1].plot(nu_u, y_sum, 'b-', lw=1.2, label=f'前{k_peaks}峰合成')
    axes[-1].legend()

    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    for ax in axes:
        ax.set_ylabel("Signal (a.u.)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese
    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)
    go_fft_decomposition(df1, angle_deg=10.0, k_peaks=1000)
