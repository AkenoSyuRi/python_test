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
    out_wav_path = r"D:\Temp\input_audition_1107_1108_1109.wav"
    in_wav_path_list = [
        r"F:\Test\1.audio_test\1.in_data\input.wav",
        r"F:\Test\1.audio_test\3.out_data\drb\input;autidion.wav",
        r"F:\Test\1.audio_test\3.out_data\drb\input;DTLN_1107_wSDR_drb_RealWedoRIR_newRTS_ep99;true.wav",
        r"F:\Test\1.audio_test\3.out_data\drb\input;DTLN_1108_wSDR_drb_RealWedoRIR_newRTS0.32_Scale2Ref_ep81;true.wav",
        r"F:\Test\1.audio_test\3.out_data\drb\input;DTLN_1109_wSDR_drb_winLen128_winInc32_ep90;true.wav",
    ]
    FileUtils.ensure_dir(out_wav_path, is_file=True)

    load_and_check_partial = partial(load_and_check, sr=sample_rate)
    in_wav_data_list = map(load_and_check_partial, in_wav_path_list)
    out_data = AudioUtils.merge_channels(*in_wav_data_list)
    soundfile.write(out_wav_path, out_data, sample_rate)
    print(out_wav_path)
    ...
