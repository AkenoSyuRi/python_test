import shutil
import wave
from pathlib import Path

import numpy as np

save_72chns = bool(1)
save_pcm = bool(1)
padding = bool(0)
apply_gain = 1
in_pcm_path = Path(r"D:\Temp\tmp\224150556_exhibition_corrected\in72chn_sorted_c0.pcm")
out_wav_dir = Path(r"D:\Temp\save_wav_out")
pick_channels = [  # 67mic version
    1,  # 0
    14, 0, 3, 2, 13, 12,  # 1~6
    17, 25, 27, 4, 5, 53, 55, 41, 43, 65, 67, 16,  # 7~18
    19, 24, 26, 6, 7, 52, 54, 40, 42, 64, 66, 18,  # 19~30
    33, 32, 10, 11, 57, 56, 46, 47, 69, 68, 21, 20,  # 31~42
    34, 30, 28, 9, 58, 51, 49, 44, 70, 62, 61, 22,  # 43~54
    35, 31, 29, 8, 59, 50, 48, 45, 71, 63, 60, 23,  # 55~66
]
# pick_channels = [  # 55mic version
#     1,  # 0
#     14, 0, 3, 2, 13, 12,  # 1~6
#     17, 25, 27, 4, 5, 53, 55, 41, 43, 65, 67, 16,  # 7~18
#     19, 24, 26, 6, 7, 52, 54, 40, 42, 64, 66, 18,  # 19~30
#     20, 30, 28, 10, 8, 50, 48, 46, 44, 62, 60, 22,  # 31~42
#     21, 31, 29, 11, 9, 51, 49, 47, 45, 63, 61, 23,  # 43~54
# ]
shutil.rmtree(out_wav_dir, ignore_errors=True)
out_wav_dir.mkdir(exist_ok=True)

with open(in_pcm_path, "rb") as fp:
    b_data = fp.read()
    data = np.frombuffer(b_data, np.short)
    remainder = data.size % 72
    data = data[:-remainder] if remainder > 0 else data

    if save_72chns:
        data = data.reshape(-1, 72)
    else:
        data = data.reshape(-1, 6, 12)

    for chn in range(data.shape[1]):
        data_write = (data[:, chn] * apply_gain).astype(np.short)

        if save_72chns:
            if chn not in pick_channels:
                continue
            i = pick_channels.index(chn)
        else:
            chn += 1
            i = chn

        if padding:
            if i in [45, 46, 47, 48, 57, 58, 59, 60]:
                data_write = np.append([0] * 3, data_write[:-3]).astype(np.short)

        if save_pcm:
            save_path = out_wav_dir / f"{in_pcm_path.stem}_split_mic{i:02d}_c{chn:02d}.pcm"
            with open(save_path, "wb") as fp:
                fp.write(data_write.tobytes())
        else:
            save_path = out_wav_dir / f"{in_pcm_path.stem}_split_mic{i:02d}_c{chn:02d}.wav"
            with wave.Wave_write(save_path.as_posix()) as fp:
                fp.setframerate(16000)
                fp.setnchannels(1 if save_72chns else 12)
                fp.setsampwidth(2)
                fp.writeframes(data_write.tobytes())
        print(save_path)
        ...
