from pathlib import Path

import numpy as np

file_idx = ...


def check_lower_bit_of_file(pcm_path, n_file=1, n_channel=12):
    with open(pcm_path, "rb") as fp:
        data = np.frombuffer(fp.read(), dtype=np.short)

    remainder = data.size % (n_file * n_channel)
    if remainder:
        data = data[:-remainder]
        print("remainder detected in the data")

    data = data.reshape(-1, n_file, n_channel)
    res1 = data[..., 0] & 1 == 1
    res2 = data[..., 1] & 1 == 1
    if res1.all() and res2.all():
        print(f"[info] check file {file_idx} passed")
        return
    print(f"[error] invalid sorted data in file: {file_idx}")
    ...


def check_process():
    global file_idx

    in_pcm_path_list = [Path(r"D:\Temp\tmp\20240705_data_cap\in72chn_sorted_c0.pcm")]
    # in_pcm_path_list = sorted(Path(r"D:\Temp\save_wav_out").glob("*.pcm"))
    num_file = len(in_pcm_path_list)

    if num_file == 1:
        file_idx = 1
        check_lower_bit_of_file(in_pcm_path_list[0], 6)
    elif num_file == 6:
        for file_idx, in_pcm_path in enumerate(in_pcm_path_list, 1):
            check_lower_bit_of_file(in_pcm_path, 1)
    else:
        raise NotImplementedError
    ...


if __name__ == "__main__":
    check_process()
    ...
