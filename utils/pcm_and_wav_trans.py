from pathlib import Path

from audio_utils import AudioUtils

in_wav_path = r"F:\Projects\PycharmProjects\python_test\data\in_data\TB5W_V1.50_RK_DRB_OFF.wav"
out_pcm_path = Path(r"F:\Projects\CLionProjects\GRUC_deploy_32k\data\input")
out_pcm_path /= Path(in_wav_path).with_suffix('.pcm').name

AudioUtils.wav2pcm(in_wav_path, out_pcm_path.as_posix(), overwrite=True)
