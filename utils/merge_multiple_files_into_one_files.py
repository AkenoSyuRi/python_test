from functools import partial

import librosa
import soundfile
from audio_utils import AudioUtils
from file_utils import FileUtils


def load_and_check(in_wav_path, sr):
    data, fs = librosa.load(in_wav_path, sr=None)
    assert fs == sr
    return data


if __name__ == "__main__":
    sample_rate = 16000
    out_wav_path = r"D:\Temp\cma100_mic22_28.wav"
    in_wav_path_list = [
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic28_c64.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic22_c06.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)

    load_and_check_partial = partial(load_and_check, sr=sample_rate)
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...
