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
    out_wav_path = r"D:\Temp\input;finetune_讲话时制造噪声.wav"
    in_wav_path_list = [
        r"F:\Test\2.audio_recorded\1.tb5w\imic_data_cap_for_denoising_and_dereverberation_test\imic_cap_out_1_channel.wav",
        r"F:\Test\2.audio_recorded\1.tb5w\imic_data_cap_for_denoising_and_dereverberation_test\imic_cap_out_1_channel;DTLN_1208_snr_dnsdrb_half_hs128_es1024_finetune_ep70;true.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)

    load_and_check_partial = partial(load_and_check, sr=sample_rate)
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...
