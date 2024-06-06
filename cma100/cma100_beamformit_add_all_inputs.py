from pathlib import Path

import librosa
import numpy as np
import soundfile


def scale_to_target_db(data, target_db=21.97):
    rms = np.sqrt(np.mean(np.square(data)))
    rms = max(rms, 1e-7)
    if target_db > 0:
        target_db = -target_db
    scale = 10 ** (target_db / 20)
    data = data / rms * scale
    return data


if __name__ == "__main__":
    sr = 16000
    in_wav_path = Path(r"D:\Temp\cma100_split_wav")
    in_wav_list = sorted(in_wav_path.glob("*.wav"))

    pick_mic_ids = list(range(61))
    # pick_mic_ids = [0] + list(range(25, 60, 2))

    in_wav_list = list(map(in_wav_list.__getitem__, pick_mic_ids))
    time_data_list = list(map(lambda x: librosa.load(x, sr=sr)[0], in_wav_list))

    # 1. sum up all signals in time domain
    data_mic0 = scale_to_target_db(time_data_list[0])
    soundfile.write("data_mic0.wav", data_mic0, sr)

    data_sum1 = scale_to_target_db(sum(time_data_list))
    soundfile.write("data_sum1.wav", data_sum1, sr)

    # 2. sum up all signals in frequency domain
    # freq_data_list = list(map(librosa.stft, time_data_list))
    # data_sum2 = sum(freq_data_list)
    # data_sum2 = scale_to_target_db(librosa.istft(data_sum2))
    # soundfile.write("data_sum2.wav", data_sum2, sr)

    print("done")
    ...
