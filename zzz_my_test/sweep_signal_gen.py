import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

# 定义扫频信号的参数
duration = 20  # 信号时长（秒）
sample_rate = 16000  # 采样率（每秒样本数）
start_freq = 50  # 起始频率 (Hz)
amplitude = 0.6
end_freq = sample_rate // 2  # 终止频率 (Hz)
out_wav_path = r"../data/output/sweep_signal.wav"
plot_signal = bool(0)

# 生成时间轴
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# 生成扫频信号
sweep_signal = amplitude * librosa.chirp(
    fmin=start_freq, fmax=end_freq, sr=sample_rate, duration=duration
)

# 绘制扫频信号
if plot_signal:
    plt.plot(t, sweep_signal)
    plt.title(f"Sweep Signal from {start_freq} Hz to {end_freq} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

soundfile.write(out_wav_path, sweep_signal, sample_rate)
