import random
from pathlib import Path

import librosa
import numpy as np
import soundfile
from audio_utils import AudioUtils


def rir_filter(rir, end_range=(10, 60), fs=32000, x_zero=150, overlap=80):
    peak_idx = np.argmax(np.abs(rir))
    x = np.arange(len(rir))
    end = random.randint(*end_range) * fs // 1000
    left = (1 / (x_zero + overlap) * (x - (x_zero + overlap))) ** 2
    right = -((1 / ((x_zero - overlap) - end) * (x - end)) ** 2) + 1
    mask = np.ones_like(rir)

    mask[peak_idx : peak_idx + x_zero] = left[:x_zero]
    mask[peak_idx + x_zero : peak_idx + end] = right[x_zero:end]
    rir_filtered = rir * mask

    return rir_filtered


def rir_filter_v2(rir, end_range=(30, 80), fs=32000, c=0.1):
    """
    y=(a(x-b))^2+c, a=sqrt(1-c)/b
    """
    peak_idx = np.argmax(np.abs(rir))
    end = random.randint(*end_range) * fs // 1000
    b = end // 2
    x = np.arange(end)
    func = (np.sqrt(1 - c) / b * (x - b)) ** 2 + c
    mask = np.ones_like(rir)

    mask[peak_idx : peak_idx + end] = func
    rir_filtered = rir * mask

    return rir_filtered


if __name__ == "__main__":
    in_rir_dir = r"D:\Temp\real_rir_picked"
    out_rir_dir = r"D:\Temp\rir_filtered_v2"
    sr = 32000

    out_dir = Path(out_rir_dir)
    out_dir.mkdir(exist_ok=True)
    for in_f in Path(in_rir_dir).glob("*.wav"):
        in_rir_data, _ = librosa.load(in_f, sr=sr)
        out_rir_data = rir_filter_v2(in_rir_data)

        out_data = AudioUtils.merge_channels(in_rir_data, out_rir_data)

        out_f = out_dir.joinpath(in_f.name)
        soundfile.write(out_f, out_data, sr)
        print(out_f)
    ...
