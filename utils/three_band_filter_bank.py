import wave
from contextlib import ExitStack
from pprint import pprint

import librosa
import numpy as np


def data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
    data, fs = librosa.load(in_audio_path, sr=sr)
    frame_len = int(fs * frame_time)

    for i in range(0, len(data), frame_len):
        clip = data[i : i + frame_len]
        if len(clip) == frame_len:
            if ret_bytes:
                clip = (clip * 32768).astype(np.short)
                yield clip.tobytes()
            else:
                yield clip


class ThreeBandFilterBank:
    """
    resample a 48k signal to 16k (as well as the reverse transformation) while preserving mid and high band information.
    - An implementation of a 3-band FIR filter-bank with DCT modulation, similar to
    the proposed in "Multirate Signal Processing for Communication Systems" by
    Fredric J Harris.
    - translated from the webrtc three_band_filter_bank.cc
    """

    def __init__(self, full_band_size=768):
        self.k_num_bands = 3
        self.k_full_band_size = full_band_size
        self.k_memory_size = 15
        self.k_filter_size = 4
        self.k_stride = 4
        self.k_split_band_size = self.k_full_band_size // self.k_num_bands

        self.state_analysis = np.zeros([self.k_num_bands, self.k_memory_size])
        self.state_synthesis = np.zeros(
            [self.k_num_bands * self.k_stride, self.k_memory_size]
        )
        self.k_filter_coeffs = np.array(
            [
                [+0.00425496, +0.16547118, -0.00496888, -0.00047749],
                [0, 0, 0, 0],
                [-0.00154717, -0.01136076, +0.01387458, +0.00186353],
                [0, 0, 0, 0],
                [+0.00994113, +0.14989004, -0.01585778, -0.00173287],
                [+0.00607594, +0.04760441, -0.02587886, -0.00346946],
                [-0.00346946, -0.02587886, +0.04760441, +0.00607594],
                [-0.00173287, -0.01585778, +0.14989004, +0.00994113],
                [+0.01157993, +0.12154542, -0.02536082, -0.00304815],
                [+0.00186353, +0.01387458, -0.01136076, -0.00154717],
                [-0.00383509, -0.02982767, +0.08543175, +0.00983212],
                [-0.00047749, -0.00496888, +0.16547118, +0.00425496],
            ]
        ).reshape(self.k_num_bands, self.k_stride, self.k_filter_size)

        self.k_dct_modulation = np.array(
            [
                [2.0, 2.0, 2.0],
                [0, 0, 0],
                [-2.0, -2.0, -2.0],
                [0, 0, 0],
                [1.73205077, 0.0, -1.73205077],
                [-1.0, 2.0, -1.0],
                [-1.73205077, 0.0, 1.73205077],
                [1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0],
                [-1.73205077, 0.0, 1.73205077],
                [-1.0, 2.0, -1.0],
                [1.73205077, 0.0, -1.73205077],
            ]
        ).reshape(self.k_num_bands, self.k_stride, self.k_num_bands)

        ...

    def analysis(self, input_data: np.ndarray):
        assert len(input_data) == self.k_full_band_size
        input_data = input_data.reshape(self.k_split_band_size, self.k_num_bands)

        out_sub_bands = [
            np.zeros(self.k_split_band_size) for _ in range(self.k_num_bands)
        ]
        for i in range(self.k_num_bands):
            in_sub_sampled = input_data[:, self.k_num_bands - i - 1]
            sub_sampled_data = np.concatenate([self.state_analysis[i], in_sub_sampled])
            for in_shift in range(self.k_stride):
                offset = self.k_stride - in_shift - 1
                out_sub_sampled = np.zeros(self.k_split_band_size)
                for j in range(self.k_split_band_size):
                    indices = offset + np.array([0, 4, 8, 12]) + j
                    out_sub_sampled[j] = np.dot(
                        sub_sampled_data[indices], self.k_filter_coeffs[i][in_shift]
                    )
                for k in range(self.k_num_bands):
                    out_sub_bands[k] += (
                        self.k_dct_modulation[i][in_shift][k] * out_sub_sampled
                    )
                    ...
            self.state_analysis[i][:] = in_sub_sampled[-self.k_memory_size :]
            ...
        return out_sub_bands

    def synthesis(self, *input_bands):
        assert len(input_bands) == 3 and all(
            map(lambda x: len(x) == self.k_split_band_size, input_bands)
        )
        out_data = np.zeros(self.k_full_band_size)
        for i in range(self.k_num_bands):
            for in_shift in range(self.k_stride):
                in_sub_sampled = np.zeros(self.k_split_band_size)
                for k in range(self.k_num_bands):
                    in_sub_sampled += (
                        self.k_dct_modulation[i][in_shift][k] * input_bands[k]
                    )
                sub_sampled_data = np.concatenate(
                    [self.state_synthesis[i * self.k_stride + in_shift], in_sub_sampled]
                )
                out_sub_sampled = np.zeros(self.k_split_band_size)
                offset = self.k_stride - in_shift - 1
                for j in range(self.k_split_band_size):
                    indices = offset + np.array([0, 4, 8, 12]) + j
                    out_sub_sampled[j] = np.dot(
                        sub_sampled_data[indices], self.k_filter_coeffs[i][in_shift]
                    )
                for j in range(self.k_split_band_size):
                    out_data[i + self.k_num_bands * j] += (
                        self.k_num_bands * out_sub_sampled[j]
                    )
                self.state_synthesis[i * self.k_stride + in_shift] = in_sub_sampled[
                    -self.k_memory_size :
                ]
        return out_data


if __name__ == "__main__":
    in_wav_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\001760.wav"
    out_wav_paths = [
        r"D:\Temp\tmp1\filter_out_band_0_1536.wav",
        r"D:\Temp\tmp1\filter_out_band_1_1536.wav",
        r"D:\Temp\tmp1\filter_out_band_2_1536.wav",
        r"D:\Temp\tmp1\filter_syn_data_1536.wav",
    ]
    frame_length, sr = 768, 48000

    filter_bank = ThreeBandFilterBank(frame_length)
    with ExitStack() as stack:
        fp_list = []
        for i, wav_f in enumerate(out_wav_paths):
            fp = stack.enter_context(wave.Wave_write(wav_f))
            if i == len(out_wav_paths) - 1:
                fp.setframerate(sr)
            else:
                fp.setframerate(16000)
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp_list.append(fp)
        for data in data_generator(in_wav_path, frame_length / sr, sr=sr):
            data *= 32768
            bands_data = filter_bank.analysis(data)
            for i, fp in enumerate(fp_list[:-1]):
                fp.writeframes(bands_data[i].astype("short").tobytes())
            out_data = filter_bank.synthesis(*bands_data)
            fp_list[-1].writeframes(out_data.astype("short").tobytes())
        pprint(out_wav_paths)
    ...
