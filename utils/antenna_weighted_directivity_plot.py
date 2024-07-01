import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hamming, hann, blackman, chebwin, taylor

# 参数设置
eps = 1e-7
fs = 16000  # 采样频率
c = 343  # 声速，m/s
element_spacing = 0.0175 * (2**2)  # 元素间距
wavelength = 2 * element_spacing  # 波长
fc = c / wavelength  # 载波频率
num_elements = 5  # 阵列元素数目

# 创建ULA麦克风阵列位置
mic_positions = np.c_[np.arange(num_elements) * element_spacing]

# 定义加权函数
uniform_weights = np.ones(num_elements)
hamming_weights = hamming(num_elements)
hann_weights = hann(num_elements)
blackman_weights = blackman(num_elements)
chebyshev_weights = chebwin(num_elements, at=45)  # 旁瓣衰减45 dB
taylor_weights = taylor(num_elements, nbar=3, sll=45)  # 旁瓣衰减45 dB

# 定义波束方向
angles = np.linspace(-90, 90, 360)

# 计算波束图
steering_vector = np.exp(
    -1j * 2 * np.pi * fc / c * mic_positions * np.sin(np.deg2rad(angles[None]))
)
response_uniform = np.abs(np.einsum("i,ij->j", uniform_weights, steering_vector)) + eps
response_hamming = np.abs(np.einsum("i,ij->j", hamming_weights, steering_vector)) + eps
response_hann = np.abs(np.einsum("i,ij->j", hann_weights, steering_vector)) + eps
response_blackman = (
    np.abs(np.einsum("i,ij->j", blackman_weights, steering_vector)) + eps
)
response_chebyshev = (
    np.abs(np.einsum("i,ij->j", chebyshev_weights, steering_vector)) + eps
)
response_taylor = np.abs(np.einsum("i,ij->j", taylor_weights, steering_vector)) + eps

# 归一化并转换为dB
response_uniform = 20 * np.log10(response_uniform / np.max(response_uniform))
response_hamming = 20 * np.log10(response_hamming / np.max(response_hamming))
response_hann = 20 * np.log10(response_hann / np.max(response_hann))
response_blackman = 20 * np.log10(response_blackman / np.max(response_blackman))
response_chebyshev = 20 * np.log10(response_chebyshev / np.max(response_chebyshev))
response_taylor = 20 * np.log10(response_taylor / np.max(response_taylor))

# 绘制波束图
plt.figure()
plt.plot(angles, response_uniform, label="Uniform")
plt.plot(angles, response_hamming, label="Hamming")
plt.plot(angles, response_hann, label="Hann")
plt.plot(angles, response_blackman, label="Blackman")
plt.plot(angles, response_chebyshev, label="Chebyshev")
plt.plot(angles, response_taylor, label="Taylor")
plt.title(
    f"Beam Patterns with Different Weighting Functions: d={element_spacing:.4f}, fc={fc:.1f}"
)
plt.xlabel("Angle (degrees)")
plt.ylabel("Response (dB)")
plt.legend()
plt.grid()
plt.show()
