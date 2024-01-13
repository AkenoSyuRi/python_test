import shutil
from pathlib import Path

import numpy as np
import soundfile


def sort_groups(data_groups):
    new_groups = []
    for i, data in enumerate(data_groups):
        # data: (T,1,6,2)
        found = False
        for j in range(data.shape[2]):
            cut = data[0, 0, j]

            if cut[0] & 1 == cut[1] & 1 == 1:
                orig_shape = data.shape
                from_idx = j * data.shape[-1]
                to_idx = -(np.prod(orig_shape[2:]) - from_idx)
                data = data.reshape(-1)[from_idx:to_idx].reshape((-1, *orig_shape[1:]))
                new_groups.append(data)
                found = True
                break
        if not found:
            print("[error] not found the first group")
            return np.concatenate(data_groups, axis=1)

    new_data = np.concatenate(new_groups, axis=1)
    return new_data


def process_file():
    # 1.prepare directory
    if clean_dir:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 2.read data
    with open(in_pcm_path, "rb") as fp:
        raw_data = fp.read()
        data = np.frombuffer(raw_data, dtype=data_type)
    remainder = len(data) % n_channels
    data = data[:-remainder] if remainder else data

    # 3.resort data
    data = data.reshape(-1, 6, 6, 2)
    data = data.transpose([0, 2, 1, 3])
    data_groups = np.split(data, data.shape[1], axis=1)
    new_data = sort_groups(data_groups)
    # new_data = data

    # 4.set output form
    if save_72c:
        data = new_data.reshape(-1, n_channels)
    else:
        data = new_data.reshape(-1, 6, 12)

    # 5.save to files
    if pick_channels:
        for i, j in enumerate(pick_channels):
            if save_pcm:
                out_file = out_dir.joinpath(
                    f"pick_{i}_{in_pcm_path.stem}_chn{j+1:02d}.pcm"
                )
                with out_file.open("wb") as fp:
                    fp.write(data[:, j].tobytes())
            else:
                out_file = out_dir.joinpath(
                    f"pick_{i}_{in_pcm_path.stem}_chn{j+1:02d}.wav"
                )
                soundfile.write(out_file, data[:, j], sr)
            print(out_file)
    else:
        for i in range(data.shape[1]):
            if save_pcm:
                out_file = out_dir.joinpath(f"{in_pcm_path.stem}_chn{i+1:02d}.pcm")
                with out_file.open("wb") as fp:
                    fp.write(data[:, i].tobytes())
            else:
                out_file = out_dir.joinpath(f"{in_pcm_path.stem}_chn{i+1:02d}.wav")
                soundfile.write(out_file, data[:, i], sr)
            print(out_file)
    ...


if __name__ == "__main__":
    n_channels, sr, data_type = 72, 32000, np.short
    # in_pcm_path = Path(r"D:\Temp\tmp\file.pcm")
    in_pcm_path = Path(r"F:\Test\2.audio_recorded\2.cma100\offset_1.pcm")
    out_dir = Path(r"D:\Temp\cma100_out")

    save_72c = bool(1)
    save_pcm = bool(0)
    clean_dir = bool(1)

    # b1m1,b1m2,b0m34,b0m22,b0m10,b0m0,b0m4,b0m16,b0m28,b2m6.b2m5
    # pick_channels = [40, 41, 34, 35, 29, 38, 7, 11, 10, 53, 52]
    pick_channels = []

    process_file()
    ...
