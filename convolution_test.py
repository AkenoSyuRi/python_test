from pathlib import Path

import librosa
import numpy as np
import soundfile
from audio_utils import AudioUtils
from scipy import signal


def scale_to_ref(tar_data, ref_data, eps=1e-7):
    ref_rms = np.sqrt(np.mean(ref_data**2))
    cur_rms = np.sqrt(np.mean(tar_data**2)) + eps
    return tar_data / cur_rms * ref_rms


def get_truncated_rir(rir, fs, *, direct_range=(-0.006, 0.08)):
    rir_early = np.zeros_like(rir)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(fs * direct_range[0]))
    end_idx = peak_idx + int(fs * direct_range[1])

    rir_early[start_idx:end_idx] = rir[start_idx:end_idx]

    scale = 0.9 / np.max(np.abs([rir, rir_early]))
    rir *= scale
    rir_early *= scale
    return rir, rir_early


def get_rts_rir(
    rir,
    original_T60: float,
    direct_range=(-0.002, 0.08),
    target_T60=0.05,
    sr: int = 32000,
):
    assert rir.ndim == 1, "rir must be a 1D array."

    if original_T60 < target_T60:
        return rir.copy()

    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))
    end_idx = peak_idx + int(sr * direct_range[1])

    win = np.zeros_like(rir)
    win[start_idx:end_idx] = 1
    win[end_idx:] = 10 ** (-q * np.arange(rir.shape[0] - end_idx))
    rts_rir = rir * win

    scale = 0.9 / np.max(np.abs([rir, rts_rir]))
    rir *= scale
    rts_rir *= scale
    return rir, rts_rir


out_dir = Path(r"D:\Temp\out1")
in_clean_path = r"F:\Test\1.audio_test\1.in_data\anechoic_room_speech.wav"
# in_clean_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
in_rir_path = r"F:\Test\3.dataset\3.rir\wedo_rk_out_speedup_rir\large_meeting_room_rk_out_3_speed_1.0.wav"
# in_rir_path = r"D:\Temp\tmp1\gpu_rir_10026_rt60_0.82s.wav"

sr = 32000

clean_data, _ = librosa.load(
    in_clean_path,
    sr=sr,
)

rir_data, _ = librosa.load(
    in_rir_path,
    sr=sr,
)

if bool(1):
    tam, tar_rt60, ori_rt60 = 0.08, 0.15, 1.0
    prefix = f"{Path(in_clean_path).stem};{Path(in_rir_path).stem};tam={tam};tar_rt60={tar_rt60};ori_rt60={ori_rt60};test;"
    rir, rts_rir = get_rts_rir(
        rir_data,
        ori_rt60,
        direct_range=(-0.002, tam),
        target_T60=tar_rt60,
    )
else:
    direct_range = (-0.002, 0.08)
    prefix = f"{Path(in_clean_path).stem};{Path(in_rir_path).stem};early={direct_range};"
    rir, rts_rir = get_truncated_rir(rir_data, sr, direct_range=direct_range)

reverb_data = signal.convolve(clean_data, rir)
label_data = signal.convolve(clean_data, rts_rir)
label_data = scale_to_ref(label_data, reverb_data)

scale = 0.9 / np.max(np.abs([reverb_data, label_data]))
reverb_data *= scale
label_data *= scale

out_dir.mkdir(exist_ok=True)

soundfile.write(
    out_dir / f"{prefix}[speech]reverb_label.wav",
    AudioUtils.merge_channels(reverb_data, label_data),
    sr,
)
soundfile.write(
    out_dir / f"{prefix}[speech]reverb.wav",
    reverb_data,
    sr,
)
soundfile.write(
    out_dir / f"{prefix}[speech]label.wav",
    label_data,
    sr,
)
soundfile.write(
    out_dir / f"{prefix}[rir]original_target.wav",
    AudioUtils.merge_channels(rir, rts_rir),
    sr,
)
...
