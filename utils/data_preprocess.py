import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
from audio_utils import AudioUtils
from file_utils import FileUtils


def split_to_mono(f, sr, out_dir):
    try:
        data, _ = librosa.load(f, sr=sr, mono=False)
        AudioUtils.save_to_mono(data, sr, out_dir, FileUtils.name_ext(f)[0])
    except Exception as e:
        print(e)


def step1_split_to_mono(in_dir, out_dir, sr):
    files = FileUtils.glob_files(rf"{in_dir}\**\*.wav")
    FileUtils.ensure_dir(out_dir)

    with ThreadPoolExecutor(max_workers=use_cpu_count) as ex:
        for f in files:
            ex.submit(split_to_mono, f, sr, out_dir)
    ...


def split_to_segment(f, sr, duration, out_dir):
    try:
        data, __ = librosa.load(f, sr=None)
        assert __ == sr
        win_len = int(sr * duration)
        win_sft = win_len
        # win_sft = win_len // 4
        AudioUtils.save_to_segment(
            data, sr, win_len, win_sft, out_dir, FileUtils.name_ext(f)[0]
        )
    except Exception as e:
        print(e)


def step2_split_to_segment(in_dir, out_dir, sr, duration):
    files = FileUtils.glob_files(rf"{in_dir}\**\*.wav")
    FileUtils.ensure_dir(out_dir)

    with ThreadPoolExecutor(max_workers=use_cpu_count) as ex:
        for f in files:
            ex.submit(split_to_segment, f, sr, duration, out_dir)


def rename_and_save(in_dir, out_dir):
    FileUtils.ensure_dir(out_dir)
    for in_f in Path(in_dir).glob("**/*.wav"):
        out_f = Path(
            out_dir,
            in_f.as_posix()[len(in_dir) :]
            .strip("/ ")
            .replace("/", "_")
            .replace(" ", "_"),
        )
        with open(in_f, "rb") as fp1, wave.Wave_write(out_f.as_posix()) as fp2:
            data = fp1.read()
            fp2.setsampwidth(2)
            fp2.setnchannels(1)
            fp2.setframerate(44100)
            fp2.writeframes(data)
    ...


if __name__ == "__main__":
    use_cpu_count = os.cpu_count()
    in_dir = r"F:\Downloads\synthetic\audio\eval\soundbank"
    out_dir0 = r"D:\Temp\step0"
    out_dir1 = r"D:\Temp\step1"
    out_dir2 = r"D:\Temp\step2"
    sr = 32000

    rename_and_save(in_dir, out_dir0)

    step1_split_to_mono(out_dir0, out_dir1, sr)  # resample to sr, and save multiple channels into multiple files

    step2_split_to_segment(out_dir1, out_dir2, sr, 10)  # split every single file in to segments
    ...
