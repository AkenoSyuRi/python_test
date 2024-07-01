from pathlib import Path

import librosa
import numpy as np
import soundfile

g_chn_list = [  # 67mic version
    1,  # 0
    14, 0, 3, 2, 13, 12,  # 1~6
    17, 25, 27, 4, 5, 53, 55, 41, 43, 65, 67, 16,  # 7~18
    19, 24, 26, 6, 7, 52, 54, 40, 42, 64, 66, 18,  # 19~30
    33, 32, 10, 11, 57, 56, 46, 47, 69, 68, 21, 20,  # 31~42
    34, 30, 28, 9, 58, 51, 49, 44, 70, 62, 61, 22,  # 43~54
    35, 31, 29, 8, 59, 50, 48, 45, 71, 63, 60, 23,  # 55~66
]


def get_data_indices(mic_id):
    index = g_chn_list[mic_id]
    a = index // 12
    b = index % 12
    # return a + 1, b + 1
    return a, b


def main():
    sr = 16000
    out_dir = Path(r"D:\Temp")
    # pick_mic_ids = [0, 2, 9, 21, 33, 45, 57][::-1]
    pick_mic_ids = [0, 5, 15, 27, 39, 51, 63]
    in_wav_path_list = [
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic01_c01.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic02_c02.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic03_c03.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic04_c04.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic05_c05.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_split_mic06_c06.wav",
    ]

    data_list = list(
        map(lambda x: librosa.load(x, sr=None, mono=False)[0], in_wav_path_list)
    )
    chn_idx_list = list(map(get_data_indices, pick_mic_ids))

    pick_data_list = [data_list[chn_idx[0]][chn_idx[1]] for chn_idx in chn_idx_list]
    tar_data = np.column_stack(pick_data_list)

    out_wav_path = out_dir / ("_".join(map(str, pick_mic_ids)) + ".wav")
    soundfile.write(out_wav_path, tar_data, sr)
    print(out_wav_path)
    ...


if __name__ == "__main__":
    main()
    ...
