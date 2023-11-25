import random
from pathlib import Path

import librosa
import soundfile

if __name__ == "__main__":
    clean_path = Path(r"F:\BaiduNetdiskDownload\BZNSYP\Wave\001760.wav")
    clean_data, sr = librosa.load(clean_path, sr=32000)
    offset_ranges = list(range(1, 32 + 1))
    channels = 4

    offset = random.choice(offset_ranges)
    for i in range(1, channels):
        idx = i * offset
        clean_data[idx:] += clean_data[:-idx]
    out_path = Path(f"out_{offset=}.wav")
    soundfile.write(out_path, clean_data, sr)
    ...
