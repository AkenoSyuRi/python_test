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
    sample_rate = 32000
    out_wav_path = r"D:\Temp\11_mic_straight_line.wav"
    in_wav_path_list = [
        r"D:\Temp\cma100_out\pick_0_offset_1_chn41.wav",
        r"D:\Temp\cma100_out\pick_1_offset_1_chn42.wav",
        r"D:\Temp\cma100_out\pick_2_offset_1_chn35.wav",
        r"D:\Temp\cma100_out\pick_3_offset_1_chn36.wav",
        r"D:\Temp\cma100_out\pick_4_offset_1_chn30.wav",
        r"D:\Temp\cma100_out\pick_5_offset_1_chn39.wav",
        r"D:\Temp\cma100_out\pick_6_offset_1_chn8.wav",
        r"D:\Temp\cma100_out\pick_7_offset_1_chn12.wav",
        r"D:\Temp\cma100_out\pick_8_offset_1_chn11.wav",
        r"D:\Temp\cma100_out\pick_9_offset_1_chn54.wav",
        r"D:\Temp\cma100_out\pick_10_offset_1_chn53.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)

    load_and_check_partial = partial(load_and_check, sr=sample_rate)
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...
