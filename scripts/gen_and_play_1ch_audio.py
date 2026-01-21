import random
import time
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
    check_sr, pause_time = 16000, 0.1
    in_wav_dir = Path(r"D:\Temp\data_aishell\wav")

    in_wav_path_list = list(in_wav_dir.rglob("*.wav"))
    assert len(in_wav_path_list) > 0

    try:
        print("start audio playing")
        while True:
            wav_path = random.choice(in_wav_path_list)
            print(wav_path)
            data, sr = librosa.load(wav_path, sr=None)
            assert sr == check_sr, f"{wav_path} has wrong sample rate: {sr} != {check_sr}"
            data = normalize_audio(data)
            sd.play(data, check_sr, blocking=True)
            time.sleep(pause_time)
    except KeyboardInterrupt:
        print("stop audio playing")
    ...


if __name__ == "__main__":
    main()
    ...
