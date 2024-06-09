from collections import defaultdict

import numpy as np
import rich
from audio_utils import AudioUtils
from matplotlib import pyplot as plt


def time_domain_cross_correlation(signal1, signal2, plot=False):
    # 计算两个信号的长度
    N1 = len(signal1)
    N2 = len(signal2)
    assert N1 == N2

    # 计算交叉相关的长度
    cross_corr_length = N1 + N2 - 1

    # 计算交叉相关
    cross_corr = np.correlate(signal1, signal2, mode="full")
    assert len(cross_corr) == cross_corr_length

    idx = np.argmax(cross_corr)
    delay = idx - N1 + 1
    print("argmax:", idx, "delay:", delay)

    if plot:
        plt.plot(cross_corr)
        plt.show()

    return delay


if __name__ == "__main__":
    fs = 16000
    in_wav_path1 = r"D:\Temp\cma100_split_wav\pick_chn_00_file_deg_0_38.wav"
    in_wav_path2 = r"D:\Temp\cma100_split_wav\pick_chn_49_file_deg_0_40.wav"

    res = defaultdict(int)
    for data1, data2 in zip(
        AudioUtils.data_generator(in_wav_path1, 0.008, sr=fs),
        AudioUtils.data_generator(in_wav_path2, 0.008, sr=fs),
    ):
        idx = time_domain_cross_correlation(data1, data2, plot=bool(0))
        res[idx] += 1

    rich.print(res)

    for k in sorted(res.keys()):
        cos_theta = (340 * k) / (0.128 * fs)
        cos_theta = 1 if cos_theta > 1 else cos_theta
        print("offset:", k, "theta:", np.rad2deg(np.arccos(cos_theta)))
    time_domain_cross_correlation(data1, data2, plot=bool(1))
    ...
