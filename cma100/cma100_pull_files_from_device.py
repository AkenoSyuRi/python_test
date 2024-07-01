import datetime
from pathlib import Path

import librosa
import soundfile
from adbutils import adb
from audio_utils import AudioUtils


def convert_pcm_to_wav(pcm_path: Path, *, sr=16000, channels=1, samp_width=2):
    wav_path = pcm_path.with_suffix(".wav")
    AudioUtils.pcm2wav(
        pcm_path.as_posix(), wav_path.as_posix(), sr, channels, samp_width
    )
    pcm_path.unlink()
    return wav_path


def pull_files_to(files_pattern, save_files_to_dir: Path):
    cmd = f"ls {files_pattern}"
    file_path_list = sorted(dev.shell(cmd).split())
    if not file_path_list:
        return

    if not save_files_to_dir.exists():
        save_files_to_dir.mkdir()

    for file_from_dev in file_path_list:
        filename = Path(file_from_dev).name
        save_pcm_path = save_files_to_dir / filename

        dev.sync.pull(file_from_dev, save_pcm_path)

        if save_as_wav:
            raise NotImplementedError
            # save_wav_path = convert_pcm_to_wav(save_pcm_path, sr=16000, channels=1)
            # print("pulled as:", save_wav_path)
            #
            # if apply_with_gain:
            #     apply_gain(save_wav_path)
        else:
            # print("pulled:", save_pcm_path)
            print(save_pcm_path)
    ...


def apply_gain(in_wav_path: Path, inc_db=30):
    data, sr = librosa.load(in_wav_path, sr=None)
    data = AudioUtils.apply_gain(data, inc_db)
    soundfile.write(in_wav_path, data, sr)
    ...


if __name__ == "__main__":
    target_files_pat = "/tmp/*.pcm"
    time_str = datetime.datetime.now().strftime("%Y%m%d")
    save_files_to_dir = Path(rf"D:\Temp\tmp\{time_str}_data_cap")

    save_as_wav = bool(0)
    apply_with_gain = bool(0)

    print(f"pull into dir: {save_files_to_dir}")
    dev = adb.device()
    pull_files_to(target_files_pat, save_files_to_dir)
    ...
