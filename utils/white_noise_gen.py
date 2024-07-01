import numpy as np
import soundfile

# 定义参数
fs = 16000  # 采样率
duration = 20  # 时长（秒）
amplitude = 0.6
samples = fs * duration  # 总样本数
out_wav_path = r"data/output/white_noise_normal.wav"

# 生成白噪声信号（均匀分布）
# white_noise = np.random.uniform(low=-1.0, high=1.0, size=samples)
# 如果需要高斯白噪声，可以使用以下代码
white_noise = np.random.normal(loc=0.0, scale=1.0, size=samples)

white_noise = amplitude * white_noise / np.max(np.abs(white_noise))

# 写入WAV文件
soundfile.write(out_wav_path, white_noise, fs)
