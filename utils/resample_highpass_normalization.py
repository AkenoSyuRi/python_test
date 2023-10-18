from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile
from acoustics import signal


def save_file(in_wav_file, in_dir, out_dir, sr=32000, eps=1e-7):
    ori_data, _ = librosa.load(in_wav_file, sr=sr, mono=False)
    assert ori_data.ndim == 1 or ori_data.ndim == 2

    def save_data(data, chn_idx):
        data = signal.highpass(data, 80, sr)
        data /= np.max(np.abs(data)) + eps

        filename = (
            in_wav_file.as_posix()[len(in_dir) :]
            .strip("/ ")
            .replace("/", "_")
            .replace(" ", "_")
        )
        out_wav_file = Path(
            out_dir, Path(filename).stem + f"_chn{chn_idx}" + in_wav_file.suffix
        )
        soundfile.write(out_wav_file, data, sr)
        print(out_wav_file)

    if ori_data.ndim == 2:
        n_channels = ori_data.shape[0]
        for i in range(n_channels):
            data = ori_data[i]
            save_data(data, i)
    else:
        save_data(ori_data, 0)


def rir_pre_process(in_dir, out_dir):
    wav_files = Path(in_dir).glob("**/*_cutrir.wav")
    Path(out_dir).mkdir(exist_ok=True)
    with ThreadPoolExecutor(1) as ex:
        for in_wav_file in wav_files:
            ex.submit(save_file, in_wav_file, in_dir, out_dir)


if __name__ == "__main__":
    in_dir = r"F:\Downloads\RIR_samples_2018_summer-autumn"
    out_dir = r"F:\Test\3.dataset\3.rir\DHSA_2018_RIR"
    rir_pre_process(in_dir, out_dir)
    ...
