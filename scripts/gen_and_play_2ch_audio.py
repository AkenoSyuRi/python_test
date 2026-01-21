import random
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
    return audio

def main():
    random.seed(666)
    check_sr = 16000
    in_wav_dir = Path(r"D:\Temp\data_aishell\wav")

    in_wav_path_list = list(in_wav_dir.rglob("*.wav"))
    assert len(in_wav_path_list) > 0

    try:
        print("start audio playing")
        while True:
            wav_l, wav_r = random.sample(in_wav_path_list, 2)
            print(wav_l.name, wav_r.name)
            data_l, sr_l = librosa.load(wav_l, sr=None)
            data_r, sr_r = librosa.load(wav_r, sr=None)
            assert sr_l == sr_r == check_sr
            data_l = normalize_audio(data_l)
            data_r = normalize_audio(data_r)

            len_l, len_r = len(data_l), len(data_r)
            max_len = max(len_l, len_r)
            half_len = max_len // 2
            if len_l < half_len:
                data_l = np.tile(data_l, 2)
            if len_r < half_len:
                data_r = np.tile(data_r, 2)
            data_l = np.pad(data_l, (0, max_len - len(data_l)), mode='constant')
            data_r = np.pad(data_r, (0, max_len - len(data_r)), mode='constant')

            data_lr = np.stack([data_l, data_r], axis=-1)
            sd.play(data_lr, check_sr, blocking=True)
    except KeyboardInterrupt:
        print("stop audio playing")
    ...


if __name__ == '__main__':
    main()
    ...
