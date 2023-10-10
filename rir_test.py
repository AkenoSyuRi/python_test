import re
from pathlib import Path

import librosa
import numpy as np
import soundfile
from scipy import signal


def scale_to_ref(tar_data, ref_data, eps=1e-7):
    ref_rms = np.sqrt(np.mean(ref_data ** 2))
    cur_rms = np.sqrt(np.mean(tar_data ** 2)) + eps
    return tar_data / cur_rms * ref_rms


def get_truncated_rir(rir, fs, early_ms=50):
    rir_early = np.zeros_like(rir)
    start_idx = np.argmax(np.abs(rir))
    end_idx = start_idx + int(fs / 1000 * early_ms)

    rir_early[:end_idx] = rir[:end_idx]

    scale = 0.7 / np.max(np.abs([rir, rir_early]))
    rir *= scale
    rir_early *= scale
    return rir, rir_early


def get_rts_rir(
        rir,
        original_T60: float,
        target_T60: float = 0.15,
        sr: int = 32000,
        time_after_max: float = 0.002,
):
    assert rir.ndim == 1, "rir must be a 1D array."
    if original_T60 <= target_T60:
        return rir, rir.copy()
        # target_T60 /= 2

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


out_dir = r"F:\Test\1.audio_test\3.out_data\tmp"
in_clean_path = r"F:\BaiduNetdiskDownload\cv-corpus-13.0-delta-2023-03-09\zh-CN\clips\common_voice_zh-CN_36533616.mp3"
in_rir_path = r"D:\Temp\rir_gen_0.1_to_1.5\rir_gen_90_rt60_1.43s_p1.wav"

clean_data, sr = librosa.load(
    in_clean_path,
    sr=None,
)

rir_data, _ = librosa.load(
    in_rir_path,
    sr=None,
)

if bool(1):
    tam, tar_rt60 = 0.070, 0.05
    prefix = f"{Path(in_rir_path).stem};tar_rt60={tar_rt60};tam={tam};"
    rir, rts_rir = get_rts_rir(
        rir_data,
        float(re.search(r"_rt60_(\d+\.\d+)s", in_rir_path).group(1)),
        target_T60=tar_rt60,
        time_after_max=tam,
    )
else:
    early = 75
    prefix = f"{Path(in_rir_path).stem};early={early};"
    rir, rts_rir = get_truncated_rir(rir_data, sr, early_ms=early)

reverb_data = signal.convolve(clean_data, rir)
label_data = signal.convolve(clean_data, rts_rir)

scale = 0.7 / np.max(np.abs([reverb_data, label_data]))
reverb_data *= scale
label_data *= scale

soundfile.write(Path(out_dir) / f"{prefix}label0.wav", label_data, sr)
label_data = scale_to_ref(label_data, reverb_data)

soundfile.write(Path(out_dir) / f"{prefix}rir.wav", rir, sr)
soundfile.write(Path(out_dir) / f"{prefix}rts_rir.wav", rts_rir, sr)
soundfile.write(Path(out_dir) / f"{prefix}reverb.wav", reverb_data, sr)
soundfile.write(Path(out_dir) / f"{prefix}label.wav", label_data, sr)
...
