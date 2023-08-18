from pathlib import Path

import librosa
import numpy as np
import scipy.signal as ss
import soundfile
from audio_utils import AudioUtils
from file_utils import FileUtils


def get_truncated_rir(rir, fs, early_ms=100):
    rir_early = np.zeros_like(rir)
    start_idx = np.argmax(rir)
    end_idx = start_idx + int(fs / 1000 * early_ms)

    rir_early[:end_idx] = rir[:end_idx]
    return rir_early


def norm_amplitude(*data_list, max_samp_val=10000):
    scale = max_samp_val / np.max(np.abs(data_list))
    result = []
    for i in range(len(data_list)):
        result.append((data_list[i] * scale).astype(np.short))
    return result


if __name__ == '__main__':
    sr = 32000
    clean_path = r"F:\Test\0.audio_test\train_data\clean\999.wav"
    rir_files = FileUtils.iglob_files(r"F:\Test\0.audio_test\train_data\rir\*.wav")
    output_dir = r"F:\Test\0.audio_test\train_data\sim_out"

    clean_data, _ = librosa.load(clean_path, sr=sr)
    for rir_path in rir_files:
        rir_data, _ = librosa.load(rir_path, sr=sr)
        rir_label = get_truncated_rir(rir_data, sr)

        reverb_data = ss.convolve(clean_data, rir_data)
        label_data = ss.convolve(clean_data, rir_label)
        reverb_data, label_data = norm_amplitude(reverb_data, label_data, max_samp_val=10000)

        out_path = Path(output_dir) / f"{Path(clean_path).stem};{Path(rir_path).name}"
        out_data = AudioUtils.merge_channels(reverb_data, label_data)
        soundfile.write(out_path.as_posix(), out_data, sr)
        print(out_path)
    ...
