from functools import partial

import librosa
import numpy as np
from matplotlib import pyplot as plt
from time_utils import TimeUtils

np.set_printoptions(suppress=True)

class ArrayGainAnalyzer:
    def __init__(
        self,
        in_wav_path,
        freq_list=(153, 216, 300, 433, 613, 866, 1225, 1732, 2450, 3465, 4900, 6930),
        freq_duration=2,
        analyze_duration=1,
        angle_list=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90),
        angle_duration=32.2,
        win_len=512,
        win_inc=256,
    ):
        self.in_data, self.sr = librosa.load(in_wav_path, sr=None, mono=False)
        assert self.in_data.shape[0] == 2, "must be a 2 channels audio file"

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_bins = self.win_len // 2 + 1
        self.frame_time = self.win_inc / self.sr

        self.freq_res = self.sr / self.win_len
        self.freq_list = freq_list
        self.bid_list = np.ceil(np.array(self.freq_list) / self.freq_res).astype(int)
        self.freq_duration = freq_duration
        self.analyze_duration = analyze_duration

        self.angle_list = angle_list
        self.angle_duration = angle_duration
        ...

    def get_stft_func(self, win_len=None, win_inc=None):
        if win_len is None:
            win_len = self.win_len

        if win_inc is None:
            win_inc = self.win_inc

        stft_func = partial(
            librosa.stft,
            n_fft=win_len,
            hop_length=win_inc,
            win_length=win_len,
            window="hann",
            center=True,
        )
        return stft_func

    @staticmethod
    def get_valid_peak_freq_idx(mag_spec, tar_idx):
        peak_idx = np.argmax(mag_spec, axis=-2)
        if np.all(peak_idx == tar_idx):
            return tar_idx
        elif np.all(np.abs(peak_idx - tar_idx) <= 2):
            return peak_idx.reshape(-1)[0]
        else:
            return -1

    def analyze(self, start_time: str):
        sec = TimeUtils.hms2sec(start_time)
        start_offset = round(sec / self.frame_time)
        freq_offset = round(self.freq_duration / self.frame_time)
        angle_offset = round(self.angle_duration / self.frame_time)
        use_frame_cnt = round(self.analyze_duration / self.frame_time)

        num_angle = len(self.angle_list)
        num_freq = len(self.freq_list)

        spec_data = self.get_stft_func()(self.in_data)

        # res = defaultdict(list)
        res = np.zeros([num_freq, num_angle])
        for i in range(num_angle):
            offset = start_offset + i * angle_offset
            for j in range(num_freq):
                spec_data_cut = spec_data[..., offset : offset + use_frame_cnt]
                mag_spec = np.abs(spec_data_cut)
                peak_idx = self.get_valid_peak_freq_idx(mag_spec[0], self.bid_list[j])
                assert peak_idx >= 0, "peak index mismatch"

                pow_spec = 20 * np.log10(mag_spec[:, peak_idx])
                pow_spec = np.mean(pow_spec, axis=-1)
                gain = pow_spec[0] - pow_spec[1]
                print(
                    self.angle_list[i],
                    self.freq_list[j],
                    round(gain, 2),
                )
                res[j, i] = gain

                offset += freq_offset
                ...
        res = np.round(res, 2)
        print("=" * 30)
        print("[summary] (freq, angle)")
        print(f"[freq] {self.freq_list}")
        print(f"[angle] {self.angle_list}")
        print(res)

        return self.angle_list, self.freq_list, res

    def plot_beam_pattern(self, angles, freqs, array_gains):
        angles = np.concatenate([-np.array(angles)[1:][::-1], np.array(angles)])

        plt.figure(figsize=(15, 7))
        ax = plt.subplot(111, projection="polar")
        for i, freq in enumerate(freqs):
            gain = np.concatenate([array_gains[i][1:][::-1], array_gains[i]])
            ax.plot(angles, gain, label=str(int(freq)))
            ax.legend()
            break
        plt.show()
        ...


if __name__ == "__main__":
    # in_wav_path = r"D:\Temp\20240509_指向性测试\【指向性测试V3】【消音室】【单频】【幅值x2】0~90°.wav"
    # analyzer = ArrayGainAnalyzer(in_wav_path)
    # analyzer.analyze("0:05.327")

    in_wav_path = (
        r"D:\Temp\20240513_test_pra\【pra仿真67mic】【单频】【+3dB】0~90°距离阵列4m.wav"
    )
    analyzer = ArrayGainAnalyzer(in_wav_path, angle_duration=26.50)
    angles, freqs, gains = analyzer.analyze("0:01.468")
    # analyzer.plot_beam_pattern(angles, freqs, gains)
    ...
