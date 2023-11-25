from pathlib import Path

from audio_utils import AudioUtils


def convert_between_pcm_and_wav(
    in_audio_path, out_dir, sample_rate=32000, n_channels=1, sample_width=2
):
    is_pcm = in_audio_path.endswith(".pcm")
    if is_pcm:
        out_audio_path = Path(out_dir, Path(in_audio_path).with_suffix(".wav").name)
        AudioUtils.pcm2wav(
            in_audio_path,
            out_audio_path.as_posix(),
            sample_rate,
            n_channels,
            sample_width,
        )
    else:
        out_audio_path = Path(out_dir, Path(in_audio_path).with_suffix(".pcm").name)
        AudioUtils.wav2pcm(in_audio_path, out_audio_path.as_posix(), overwrite=False)
    print(out_audio_path)


if __name__ == "__main__":
    out_dir = r"F:\Projects\CLionProjects\DTLN_deploy\data\input"
    in_audio_path = r"F:\Test\1.audio_test\1.in_data\input.wav"
    convert_between_pcm_and_wav(
        in_audio_path,
        out_dir,
    )
