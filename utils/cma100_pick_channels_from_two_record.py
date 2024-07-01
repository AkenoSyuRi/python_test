import librosa
import numpy as np
import soundfile


def main():
    sr = 16000
    out_wav_path = (
        r"D:\Temp\[mic31_v1,v2]_[mic32_v1,v2]_[mic43_v1,v2]_[mic55_v1,v2].wav"
    )
    in_wav_path_list = [
        r"D:\Temp\save_wav_out_正对mic43_无背板\in72chn_sorted_c0_split_mic03_c03.wav",
        r"D:\Temp\save_wav_out_正对mic43_有背板v2\in72chn_sorted_c0_正对mic43_有背板v2_split_mic03_c03.wav",
    ]
    chn_idx_list = (
        np.array(
            [
                [1, 10],
                [2, 10],
                [1, 9],
                [2, 9],
                [1, 11],
                [2, 11],
                [1, 12],
                [2, 12],
            ]
        )
        - 1
    )
    start_offset_list = [int(1.494 * sr), 0]

    data_list = list(
        map(lambda x: librosa.load(x, sr=None, mono=False)[0], in_wav_path_list)
    )

    pick_data_list = [
        data_list[chn_idx[0]][chn_idx[1]][start_offset_list[chn_idx[0]] :]
        for chn_idx in chn_idx_list
    ]
    min_len = min(map(len, pick_data_list))
    for i in range(len(pick_data_list)):
        pick_data_list[i] = pick_data_list[i][:min_len]

    tar_data = np.column_stack(pick_data_list)

    soundfile.write(out_wav_path, tar_data, sr)
    print(out_wav_path)
    ...


if __name__ == "__main__":
    main()
    ...
