from pathlib import Path

import librosa
from audio_utils import AudioUtils

if __name__ == "__main__":
    in_wav_path_or_dir = Path(r"D:\Temp\input_autidion_1104_1107.wav")
    out_wav_dir = r"D:\Temp\out2"

    if in_wav_path_or_dir.is_dir():
        for in_f in in_wav_path_or_dir.glob("*original_target.wav"):
            data, sr = librosa.load(in_f, sr=None, mono=False)
            AudioUtils.save_to_mono(data, sr, out_wav_dir, in_f.name)
    elif in_wav_path_or_dir.is_file():
        in_f = in_wav_path_or_dir
        data, sr = librosa.load(in_f, sr=None, mono=False)
        AudioUtils.save_to_mono(data, sr, out_wav_dir, in_f.name)
    else:
        raise RuntimeError(f"unknown path: {in_wav_path_or_dir}")
    ...
