from pathlib import Path

import librosa
import numpy as np
import soundfile
from scipy import signal


def get_rts_rir(
    rir,
    original_T60: float,
    target_T60: float = 0.15,
    sr: int = 32000,
    time_after_max: float = 0.1,
):
    assert rir.ndim == 1, "rir must be a 1D array."
    if original_T60 <= target_T60:
        return rir

    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)
    idx_max = int(np.argmax(np.abs(rir)))
    N1 = int(idx_max + time_after_max * sr)
    win = np.empty(shape=rir.shape, dtype=np.float32)
    win[:N1] = 1
    win[N1:] = 10 ** (-q * np.arange(rir.shape[0] - N1))
    rts_rir = rir * win

    scale = 0.7 / np.max(np.abs([rir, rts_rir]))
    rir *= scale
    rts_rir *= scale
    return rir, rts_rir


out_dir = r"F:\Projects\PycharmProjects\python_test\data\out_data\tmp"
clean_data, sr = librosa.load(
    r"F:\BaiduNetdiskDownload\cv-corpus-13.0-delta-2023-03-09\zh-CN\clips\common_voice_zh-CN_36533616.mp3",
    sr=None,
)

rir_data, _ = librosa.load(
    r"F:\Projects\PycharmProjects\python_test\data\in_data\tmp\gpu_rir_2_rt60_1.08s.wav",
    sr=None,
)

tar_rt60, tam = 0.05, 0.002
rir, rts_rir = get_rts_rir(rir_data, 1.08, target_T60=tar_rt60, time_after_max=tam)

reverb_data = signal.convolve(clean_data, rir)
label_data = signal.convolve(clean_data, rts_rir)

scale = 0.7 / np.max(np.abs([reverb_data, label_data]))
reverb_data *= scale
label_data *= scale

prefix = f"tar_rt60={tar_rt60};tam={tam};"
soundfile.write(Path(out_dir) / f"{prefix}rir.wav", rir, sr)
soundfile.write(Path(out_dir) / f"{prefix}rts_rir.wav", rts_rir, sr)
soundfile.write(Path(out_dir) / f"{prefix}reverb.wav", reverb_data, sr)
soundfile.write(Path(out_dir) / f"{prefix}label.wav", label_data, sr)
...
