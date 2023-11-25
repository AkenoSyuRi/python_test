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
    out_wav_path = r"D:\Temp\input;audition;finetune;before_finetune.wav"
    in_wav_path_list = [
        r"F:\Test\1.audio_test\1.in_data\input.wav",
        r"D:\Temp\audition.wav",
        r"F:\Test\1.audio_test\3.out_data\drb\input;DTLN_1127_wSDR_drb_triple_200u_Finetune_Decrease1dB_ep59;true.wav",
        r"F:\Test\1.audio_test\3.out_data\tmp\input;model_0030;true.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)

    load_and_check_partial = partial(load_and_check, sr=sample_rate)
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...
