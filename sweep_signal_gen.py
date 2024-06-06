import numpy as np
import soundfile

# 定义扫频信号的参数
amplitude = 0.6
duration = 5  # 信号时长（秒）
sample_rate = 16000  # 采样率（每秒样本数）

# 生成时间轴
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# 生成扫频信号
start_freq = 100  # 起始频率 (Hz)
end_freq = 8000  # 终止频率 (Hz)
sweep_signal = amplitude * np.sin(
    2 * np.pi * np.logspace(np.log10(start_freq), np.log10(end_freq), len(t)) * t
)

soundfile.write(r"D:\Temp\sweep_signal.wav", sweep_signal, sample_rate)
