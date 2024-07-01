import librosa
import numpy as np
import soundfile
from acoustics import signal
from audio_utils import AudioUtils, AudioWriter
from scipy.signal import butter


def highpass_test1():
    in_wav_path = r"data/input/in72chn_sorted_c0_split_mic00_c38.wav"
    data, sr = librosa.load(in_wav_path, sr=None)
    data_new = signal.highpass(data, 100, sr,order=5)
    soundfile.write("data/output/hpf_100Hz_v1.wav", data_new, sr)


def highpass_test2():
    in_wav_path = r"F:\Test\4.play\test_noisy_record.wav"
    data, sr = librosa.load(in_wav_path, sr=None)

    data_new = np.zeros_like(data)
    sos = butter(4, 100 / (sr / 2.0), btype="high", output="sos")
    zi = np.zeros([2, 2])
    for n in range(data.size):
        x_cur = data[n]
        for s in range(2):
            x_new = sos[s, 0] * x_cur + zi[s, 0]
            zi[s, 0] = sos[s, 1] * x_cur - sos[s, 4] * x_new + zi[s, 1]
            zi[s, 1] = sos[s, 2] * x_cur - sos[s, 5] * x_new
            x_cur = x_new
        data_new[n] = x_cur

    soundfile.write("data/output/hpf_100Hz_v2.wav", data_new, sr)


def highpass_test3():
    from scipy.signal._sosfilt import _sosfilt

    in_wav_path = r"data/input/in72chn_sorted_c0_split_mic00_c38.wav"
    sr = 16000
    order = 4

    sos = butter(order, 100 / (sr / 2.0), btype="high", output="sos")
    zi = np.zeros([3, order // 2, 2])

    writer = AudioWriter("data/output", sr)
    for in_data in AudioUtils.data_generator(in_wav_path, 0.016, sr=sr):
        in_data = in_data[None].astype(float)
        in_data = np.tile(in_data, [3, 1])
        _sosfilt(sos, in_data, zi)
        writer.write_data_list("hpf", in_data)
    ...


if __name__ == "__main__":
    # highpass_test1()
    highpass_test2()
    # highpass_test3()
    ...
