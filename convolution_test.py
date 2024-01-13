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


def get_truncated_rir(rir, sr, *, direct_range=(-0.001, 0.08)):
    rir_early = np.zeros_like(rir)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))
    end_idx = peak_idx + int(sr * direct_range[1])

    rir[:start_idx] = 0

    rir_early[start_idx:end_idx] = rir[start_idx:end_idx]

    rir_early = scale_to_ref(rir_early, rir)
    return rir_early


def get_rts_rir(
    rir,
    sr,
    *,
    original_T60=1.0,
    target_T60=0.05,
    direct_range=(-0.001, 0.08),
):
    assert rir.ndim == 1, "rir must be a 1D array."

    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))
    end_idx = peak_idx + int(sr * direct_range[1])

    rir[:start_idx] = 0

    win = np.zeros_like(rir)
    win[start_idx:end_idx] = 1
    win[end_idx:] = 10 ** (-q * np.arange(rir.shape[0] - end_idx))
    rts_rir = rir * win

    rts_rir = scale_to_ref(rts_rir, rir)
    return rts_rir


def get_decayed_and_attenuated_rir(
    rir, sr, *, direct_range=(-0.001, 0.02), rd=0.2, t1=0.03, alpha=0.4
):
    # get decayed and attenuated function
    t = np.arange((len(rir)))
    t0 = int(sr * direct_range[1])
    t1 = int(sr * t1)
    rd = int(sr * rd)

    y1 = 10 ** (-3 * (t - t0) / rd)
    y1[:t0] = 1

    y2 = (1 + alpha) / 2 + (1 - alpha) / 2 * np.cos(np.pi * (t - t0) / (t1 - t0))
    y2[:t0] = 1
    y2[t1:] = alpha

    y = y1 * y2

    # apply function
    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))

    rir[:start_idx] = 0

    target_rir = rir.copy()
    target_rir[peak_idx:] *= y[:-peak_idx] if peak_idx else y

    target_rir = scale_to_ref(target_rir, rir)
    return target_rir


def process(in_clean_path, in_rir_path, out_dir, sr, use_func):
    clean_data, _ = librosa.load(in_clean_path, sr=sr)
    rir_data, _ = librosa.load(in_rir_path, sr=sr)

    direct_range = (-0.001, 0.001)
    target_type = {1: "early", 2: "rts", 3: "daa"}[use_func]
    prefix = f"[{target_type}]{Path(in_clean_path).stem};{Path(in_rir_path).stem};direct={direct_range};"
    if use_func == 1:
        rir_target = get_truncated_rir(rir_data, sr, direct_range=direct_range)
    elif use_func == 2:
        tar_rt60, ori_rt60 = 0.60, 1.0
        prefix += f"tar_rt60={tar_rt60};ori_rt60={ori_rt60};"
        rir_target = get_rts_rir(
            rir_data,
            sr,
            original_T60=ori_rt60,
            target_T60=tar_rt60,
            direct_range=direct_range,
        )
    elif use_func == 3:
        rd, t1, alpha = 0.2, direct_range[1] + 0.01, 0.4
        prefix += f"{rd=};{t1=};{alpha=};"
        rir_target = get_decayed_and_attenuated_rir(
            rir_data, sr, direct_range=direct_range, rd=rd, t1=t1, alpha=alpha
        )
    else:
        raise NotImplementedError

    reverb_data = signal.convolve(clean_data, rir_data)
    label_data = signal.convolve(clean_data, rir_target)

    scale = 0.95 / np.max(np.abs([reverb_data, label_data]))
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
        AudioUtils.merge_channels(rir_data, rir_target),
        sr,
    )
    print(prefix)
    ...


if __name__ == "__main__":
    out_dir = Path(r"D:\Temp\convolution_test_out_v7")
    in_clean_path = r"F:\Test\1.audio_test\1.in_data\anechoic_room_speech_lzf.wav"
    # in_rir_path = r"D:\Temp\rir_gen_simulated\rir_gen_1557_rt60_0.26s_p0.wav"
    # in_rir_path = r"D:\Temp\rir_gen_simulated\rir_gen_7704_rt60_0.52s_p0.wav"
    in_rir_path = r"D:\Temp\rir_gen_simulated\rir_gen_2098_rt60_1.10s_p0.wav"

    sr = 16000

    use_func = 2  # 1 for truncated_rir, 2 for rts_rir, 3 for daa_rir

    process(in_clean_path, in_rir_path, out_dir, sr, use_func)
    ...
