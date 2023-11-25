import random
from pathlib import Path
from typing import Iterable

import df
import librosa
import numpy as np
import soundfile
import torch
from acoustics import signal
from scipy import signal as ss


def preprocess_by_df(
    out_dir: Path,
    audio_paths: Iterable[Path],
    silent_path: Path,
    *,
    infer_sr=48000,
    save_sr=32000,
    scale_range=(0.3, 0.8),
):
    """
    1. resample to 48k
    2. 50Hz highpass filter
    3. change the amplitude by different scale
    4. enhance by network
    5. resample to 32k
    6. add silence data
    """
    need_resample = infer_sr != save_sr
    model, df_state, _ = df.init_df()
    silence_data, _ = librosa.load(silent_path, sr=save_sr)

    out_dir.mkdir(parents=True, exist_ok=True)
    for in_f in audio_paths:
        in_data, _ = librosa.load(in_f, sr=infer_sr)
        in_data = signal.highpass(in_data, 50, infer_sr)
        scale = random.uniform(*scale_range) / np.max(np.abs(in_data))
        in_data *= scale
        in_data = torch.FloatTensor(in_data.reshape(1, -1))

        out_data = df.enhance(model, df_state, in_data).squeeze().numpy()

        if need_resample:
            out_data = librosa.resample(out_data, orig_sr=infer_sr, target_sr=save_sr)
        out_data += silence_data

        out_f = out_dir.joinpath(in_f.name)
        soundfile.write(out_f, out_data, save_sr)
        print(out_f)


def get_scale_to_ref(tar_data, ref_data, eps=1e-7):
    ref_rms = np.sqrt(np.mean(ref_data**2))
    cur_rms = np.sqrt(np.mean(tar_data**2)) + eps
    return ref_rms / cur_rms


def get_rts_rir(
    rir,
    sr,
    original_T60: float = 1.0,
    direct_range=(-0.002, 0.08),
    target_T60=0.05,
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

    rts_rir = rts_rir * get_scale_to_ref(rts_rir, rir) * 0.95
    return rts_rir


def batch_add_reverb(clean_files, rir_files, out_reverb_dir, out_target_dir):
    out_reverb_dir.mkdir(parents=True, exist_ok=True)
    out_target_dir.mkdir(parents=True, exist_ok=True)
    for i, (cf, rf) in enumerate(zip(clean_files, rir_files), 1):
        clean_data, sr = librosa.load(cf, sr=None)
        rir_data, _ = librosa.load(rf, sr=None)
        rts_data = get_rts_rir(rir_data, sr)

        reverb_data = ss.convolve(clean_data, rir_data)
        target_data = ss.convolve(clean_data, rts_data)
        scale = get_scale_to_ref(reverb_data, clean_data)
        reverb_data *= scale
        target_data *= scale

        clean_name, rir_name = cf.name, rf.name
        out_reverb_path = out_reverb_dir.joinpath(rf"reverb;{clean_name};{rir_name}")
        out_target_path = out_target_dir.joinpath(rf"target;{clean_name};{rir_name}")

        soundfile.write(out_reverb_path, reverb_data, sr)
        soundfile.write(out_target_path, target_data, sr)
        print(i, out_reverb_path)
    ...


if __name__ == "__main__":
    clean_dir = Path(r"F:\Test\1.audio_test\5.evaluation\test_dataset\1.clean_original")
    cleaned_dir = Path(
        r"F:\Test\1.audio_test\5.evaluation\test_dataset\2.clean_denoised"
    )
    silent_path = Path(r"F:\Test\2.audio_recorded\small_stationay_noise_10s_32k.wav")

    # in_files = clean_dir.glob("*.wav")
    # preprocess_by_df(cleaned_dir, in_files, silent_path)

    rir_dir = Path(r"F:\Test\1.audio_test\5.evaluation\test_dataset\3.rir")
    reverb_dir = Path(r"F:\Test\1.audio_test\5.evaluation\test_dataset\4.reverb_speech")
    target_dir = Path(r"F:\Test\1.audio_test\5.evaluation\test_dataset\5.target_speech")

    cleaned_files = list(cleaned_dir.glob("*.wav"))
    rir_files = list(rir_dir.glob("*.wav"))
    random.shuffle(cleaned_files)
    random.shuffle(rir_files)
    batch_add_reverb(cleaned_files, rir_files, reverb_dir, target_dir)
    ...
