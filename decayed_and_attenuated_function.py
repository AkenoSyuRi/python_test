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


def get_decayed_and_attenuated_rir(
    rir, fs, direct_range=(-0.001, 0.07), rd=0.2, t1=0.08, alpha=0.4
):
    # get decayed and attenuated function
    t = np.arange((len(rir)))
    t0 = int(fs * direct_range[1])
    t1 = int(fs * t1)
    rd = int(fs * rd)

    y1 = 10 ** (-3 * (t - t0) / rd)
    y1[:t0] = 1

    y2 = (1 + alpha) / 2 + (1 - alpha) / 2 * np.cos(np.pi * (t - t0) / (t1 - t0))
    y2[:t0] = 1
    y2[t1:] = alpha

    y = y1 * y2

    # apply function
    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(fs * direct_range[0]))

    rir[:start_idx] = 0

    target_rir = rir.copy()
    target_rir[peak_idx:] *= y[:-peak_idx]

    target_rir = scale_to_ref(target_rir, rir)
    return rir, target_rir


if __name__ == "__main__":
    out_dir = Path(r"D:\Temp\out1")
    in_clean_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
    in_rir_path = r"D:\Temp\gpu_rir\gpu_rir_10244_rt60_1.10s.wav"
    # in_rir_path = r"F:\Test\3.dataset\3.rir\wedo_rk_out_speedup_rir\large_meeting_room_rk_out_3_speed_1.0.wav"
    prefix = "daa2_"
    sr = 32000

    clean_data, _ = librosa.load(in_clean_path, sr=sr)
    rir_data, _ = librosa.load(in_rir_path, sr=sr)

    rir_data, rir_target = get_decayed_and_attenuated_rir(rir_data, sr)

    reverb = signal.convolve(clean_data, rir_data)
    target = signal.convolve(clean_data, rir_target)

    scale = 0.95 / np.max(np.abs([reverb, target]))
    reverb *= scale
    target *= scale

    out_dir.mkdir(exist_ok=True)

    soundfile.write(
        out_dir / f"{prefix}[speech]reverb_label.wav",
        AudioUtils.merge_channels(reverb, target),
        sr,
    )
    soundfile.write(
        out_dir / f"{prefix}[speech]reverb.wav",
        reverb,
        sr,
    )
    soundfile.write(
        out_dir / f"{prefix}[speech]label.wav",
        target,
        sr,
    )
    soundfile.write(
        out_dir / f"{prefix}[rir]original_target.wav",
        AudioUtils.merge_channels(rir_data, rir_target),
        sr,
    )
    ...
