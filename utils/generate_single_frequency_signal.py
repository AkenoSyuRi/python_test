import numpy as np
import soundfile as sf


def get_freq_str(freq_list):
    return "+".join(map(str, freq_list)) + "Hz"


# 采样率设置
sample_rate = 16000
duration = 5  # 持续时间
frequencies = [1000]  # 1kHz
amplitude = 0.3 / len(frequencies)  # 将分贝转换为振幅

# 生成时间轴
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 生成1kHz正弦波信号
signal = np.sum([amplitude * np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)

# 写入音频文件
out_file = rf"D:\Temp\{get_freq_str(frequencies)}_{duration}s.wav"
sf.write(out_file, signal, sample_rate)
print(out_file)

# spec = np.fft.rfft(signal)
# mag = abs(spec)
# plt.plot(mag)
# plt.show()
