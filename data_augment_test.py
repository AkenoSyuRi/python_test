import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile


class AudioAugment:
    def __init__(
        self,
        fs=32000,
        shift_range=(0.05, 0.41),
        speed_range=(0.7, 1.4),
        step_range=(-5, 6),
    ):
        self.fs = fs
        self.shift_list = (np.arange(*shift_range, 0.02) * fs).astype(int)
        self.speed_list = np.round(np.arange(*speed_range, 0.05), 2)
        self.step_list = np.setdiff1d(np.arange(*step_range, 1), [0])
        ...

    def time_shift(self, x, shift_index: int = None):
        shift = shift_index or random.choice(self.shift_list)
        return np.roll(x, shift)

    def time_stretch(self, x, speed: float = None):
        rate = speed or random.choice(self.speed_list)
        return librosa.effects.time_stretch(x, rate=rate)

    def pitch_shifting(self, x, step: float = None):
        n_step = step or random.choice(self.step_list)
        return librosa.effects.pitch_shift(
            x, sr=self.fs, n_steps=n_step, bins_per_octave=12
        )

    @staticmethod
    def freq_mask(
        x,
        num_mask=1,
        mask_percentage=0.015,
        win_len=1024,
        win_inc=512,
        window="hamming",
    ):
        x_stft = librosa.stft(
            x, n_fft=win_len, hop_length=win_inc, win_length=win_len, window=window
        )
        fft_bins = x_stft.shape[0]
        mask_width = int(mask_percentage * fft_bins)
        for i in range(num_mask):
            mask_start = np.random.randint(low=0, high=fft_bins - mask_width)
            x_stft[mask_start : mask_start + mask_width :] = 0
        x_masked = librosa.istft(
            x_stft, n_fft=win_len, hop_length=win_inc, win_length=win_len, window=window
        )
        return x_masked


def process_file(in_f, out_f, sr, func, *args):
    in_data, _ = librosa.load(in_f, sr=sr)
    out_data = func(in_data, *args)
    soundfile.write(out_f, out_data, sr)
    print(func.__name__, out_f.as_posix())
    ...


if __name__ == "__main__":
    in_wav_path_list = list(Path(r"D:\Temp\step2").glob("*.wav"))
    out_dir = Path(r"D:\Temp\step3")
    sr = 32000
    augment = AudioAugment(fs=sr)

    with ThreadPoolExecutor() as ex:
        for in_f in in_wav_path_list:
            # for speed in augment.speed_list:
            #     out_f = out_dir.joinpath(f"{in_f.stem}_speed{speed}.wav")
            #     ex.submit(process_file, in_f, out_f, sr, augment.time_stretch, speed)
            for i in range(10):
                out_f = out_dir.joinpath(f"{in_f.stem}_mask{i}.wav")
                ex.submit(process_file, in_f, out_f, sr, augment.freq_mask)

    ...
