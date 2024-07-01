from functools import partial

import librosa
import numpy as np
import soundfile
from audio_utils import AudioUtils
from file_utils import FileUtils


def load_and_check(in_wav_path, sr):
    data, fs = librosa.load(in_wav_path, sr=None)
    assert fs == sr
    return data


def multi_files_to_multi_channels(in_wav_path_list, out_wav_path):
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...


def multi_files_to_one_channels(in_wav_path_list, out_wav_path):
    in_wav_data_list = list(map(load_and_check_partial, in_wav_path_list))
    out_data = np.concatenate(in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...


if __name__ == "__main__":
    do_concat = bool(0)
    sample_rate = 16000
    out_wav_path = r"D:\Temp\mic0_notaper_withtaper.wav"
    in_wav_path_list = [
        r"D:\Temp\athena_test_out\单频\[sim]AG_sf_anechoic_snr20_v0_d0_n0_g0_1_inp.wav",
        r"D:\Temp\athena_test_out\单频\[sim]AG_sf_anechoic_snr20_v0_d0_n0_g0_2_out.wav",
        r"D:\Temp\athena_test_out\[real]test_v0_d0_n1_2_out.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)
    load_and_check_partial = partial(load_and_check, sr=sample_rate)

    if do_concat:
        multi_files_to_one_channels(in_wav_path_list, out_wav_path)
    else:
        multi_files_to_multi_channels(in_wav_path_list, out_wav_path)
    ...
