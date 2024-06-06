from pathlib import Path

from audio_utils import AudioUtils


def convert_between_pcm_and_wav(
    in_audio_path, out_dir, sample_rate=32000, n_channels=1, sample_width=2
):
    out_dir = in_audio_path.parent if out_dir is None else out_dir
    is_pcm = in_audio_path.suffix == ".pcm"
    if is_pcm:
        out_audio_path = out_dir.joinpath(in_audio_path.with_suffix(".wav").name)
        AudioUtils.pcm2wav(
            in_audio_path.as_posix(),
            out_audio_path.as_posix(),
            sample_rate,
            n_channels,
            sample_width,
        )
    else:
        out_audio_path = out_dir.joinpath(in_audio_path.with_suffix(".pcm").name)
        AudioUtils.wav2pcm(
            in_audio_path.as_posix(), out_audio_path.as_posix(), overwrite=False
        )
    print(out_audio_path)


def process(in_audio_or_dir: Path, out_dir: Path = None):
    if in_audio_or_dir.is_file():
        convert_between_pcm_and_wav(in_audio_or_dir, out_dir)
    elif in_audio_or_dir.is_dir():
        for in_f in in_audio_or_dir.glob("*.[wp][ac][vm]"):
            convert_between_pcm_and_wav(in_f, out_dir)
    else:
        raise NotImplementedError
    ...


if __name__ == "__main__":
    in_audio_path_or_dir = Path(r"D:\Temp\cma100_sim_data")
    process(in_audio_path_or_dir, None)
