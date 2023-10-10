import wave
from contextlib import ExitStack
from pprint import pprint

import numpy as np
from audio_utils import AudioUtils

k_filter_coeffs = np.array([
    [+0.00425496, +0.16547118, -0.00496888, -0.00047749, ],
    [0, 0, 0, 0],
    [-0.00154717, -0.01136076, +0.01387458, +0.00186353, ],
    [0, 0, 0, 0],
    [+0.00994113, +0.14989004, -0.01585778, -0.00173287, ],
    [+0.00607594, +0.04760441, -0.02587886, -0.00346946, ],
    [-0.00346946, -0.02587886, +0.04760441, +0.00607594, ],
    [-0.00173287, -0.01585778, +0.14989004, +0.00994113, ],
    [+0.01157993, +0.12154542, -0.02536082, -0.00304815, ],
    [+0.00186353, +0.01387458, -0.01136076, -0.00154717, ],
    [-0.00383509, -0.02982767, +0.08543175, +0.00983212, ],
    [-0.00047749, -0.00496888, +0.16547118, +0.00425496, ],
])

k_dct_modulation = np.array([
    [2., 2., 2.],
    [0, 0, 0],
    [-2., -2., -2.],
    [0, 0, 0],
    [1.73205077, 0., -1.73205077],
    [-1., 2., -1.],
    [-1.73205077, 0., 1.73205077],
    [1., -2., 1.],
    [1., -2., 1.],
    [-1.73205077, 0., 1.73205077],
    [-1., 2., -1.],
    [1.73205077, 0., -1.73205077]
])


class ThreeBandFilterBank:
    def __init__(self):
        self.k_num_bands = 3
        self.k_full_band_size = 768
        self.k_memory_size = 15
        self.k_filter_size = 4
        self.k_split_band_size = self.k_full_band_size // self.k_num_bands
        self.band_state = [np.zeros(self.k_memory_size) for _ in range(self.k_num_bands)]
        ...

    def analysis(self, input_data: np.ndarray):
        input_data = input_data.reshape(self.k_split_band_size, self.k_num_bands)

        out_sub_bands = [np.zeros(self.k_split_band_size) for _ in range(self.k_num_bands)]
        for i in range(self.k_num_bands):
            in_sub_sampled = input_data[:, self.k_num_bands - i - 1]
            sub_sampled_data = np.concatenate([self.band_state[i], in_sub_sampled])
            for x in range(self.k_filter_size):
                filters = k_filter_coeffs[x]
                out_sub_sampled = np.zeros(self.k_split_band_size)
                for j in range(self.k_split_band_size):
                    for m in range(self.k_filter_size):
                        out_sub_sampled[j] += filters[m] * sub_sampled_data[m * self.k_filter_size + j]
                for j in range(self.k_num_bands):
                    for m in range(self.k_split_band_size):
                        out_sub_bands[j][m] += k_dct_modulation[x][j] * out_sub_sampled[m]
            self.band_state[i][:] = in_sub_sampled[-self.k_memory_size:]
        return out_sub_bands


if __name__ == "__main__":
    in_wav_path = r"data/in_data/VOICEACTRESS100_054.wav"
    out_wav_paths = [
        r"data/out_data/tmp/filter_out_band_0.wav",
        r"data/out_data/tmp/filter_out_band_1.wav",
        r"data/out_data/tmp/filter_out_band_2.wav",
    ]
    frame_length, sr = 768, 48000

    filter_bank = ThreeBandFilterBank()
    with ExitStack() as stack:
        fp_list = []
        for wav_f in out_wav_paths:
            fp = stack.enter_context(wave.Wave_write(wav_f))
            fp.setframerate(16000)
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp_list.append(fp)
        for data in AudioUtils.data_generator(in_wav_path, frame_length / sr, sr=sr):
            data *= 32768
            bands_data = filter_bank.analysis(data)
            for i, fp in enumerate(fp_list):
                fp.writeframes(bands_data[i].astype("short").tobytes())
        pprint(out_wav_paths)
    ...
