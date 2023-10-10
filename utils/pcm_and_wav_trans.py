from pathlib import Path

from audio_utils import AudioUtils

in_audio_path = r"F:\Projects\CLionProjects\DTLN_deploy\data\input\TB5W_V1.50_RK_DRB_OFF_16k.pcm"
out_audio_path = Path(r"F:\Projects\PycharmProjects\python_test\data\in_data",
                      Path(in_audio_path).with_suffix('.wav').name)

AudioUtils.pcm2wav(in_audio_path, out_audio_path.as_posix(), 16000)
