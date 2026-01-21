import time
from pathlib import Path

import edge_tts
import librosa
import numpy as np
import soundfile

TEXT_ENs = [
    "muted",
    "unmuted",
    "filter saved",
    "model 1",
    "model 2",
    "model 3",
    "left channel mode",
    "right channel mode",
    "stereo mode",
    "start recording",
    "end recording",
    "volume down",
    "volume up",
]
TEXT_CNs = [
    "开启静音",
    "关闭静音",
    "滤波器已保存",
    "模型1",
    "模型2",
    "模型3",
    "左声道模式",
    "右声道模式",
    "立体声模式",
    "开始录音",
    "结束录音",
    "减小音量",
    "增加音量",
]
VOICEs = ["en-US-JennyNeural", "zh-CN-XiaochenNeural"]
OUTPUT_DIR = Path("data/output")
Fs = 16000


def post_process(io_wav_path, unlink=bool(1)):
    io_wav_path = Path(io_wav_path)

    audio, _ = librosa.load(io_wav_path, sr=Fs)
    audio = (audio * 32768).astype(np.short)
    idx = np.nonzero(audio)[0]

    if len(idx) < 2:
        print("warning:", io_wav_path)
        return

    audio = audio[idx[0] - 1 : idx[-1] + 1]

    audio = np.column_stack([audio, audio])
    if unlink:
        io_wav_path.unlink()
    else:
        soundfile.write(io_wav_path, audio, Fs)

    out_16k_path = Path(io_wav_path).parent / f"{io_wav_path.stem}_16k.pcm"
    with open(out_16k_path, "wb") as fp:
        fp.write(audio.tobytes())

    out_48k_path = Path(io_wav_path).parent / f"{io_wav_path.stem}_48k.pcm"
    with open(out_48k_path, "wb") as fp:
        audio = librosa.resample(audio / 32768, orig_sr=16000, target_sr=48000, axis=0)
        fp.write((audio * 32768).astype(np.short).tobytes())

    print(io_wav_path)
    ...


def main() -> None:
    assert len(TEXT_ENs) == len(TEXT_CNs)

    Path(OUTPUT_DIR, "cn").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "en").mkdir(parents=True, exist_ok=True)
    for TEXTs in zip(TEXT_ENs, TEXT_CNs):
        assert len(TEXTs) == len(VOICEs)

        try:
            save_name = TEXTs[0].replace(" ", "_")
            save_path = OUTPUT_DIR / "en" / f"{save_name}.wav"

            communicate = edge_tts.Communicate(TEXTs[0], VOICEs[0], rate="+10%")
            communicate.save_sync(save_path.as_posix())
            post_process(save_path)
            time.sleep(0.1)

            communicate = edge_tts.Communicate(TEXTs[1], VOICEs[1], rate="+20%")
            save_path = OUTPUT_DIR / "cn" / f"{save_name}.wav"
            communicate.save_sync(save_path.as_posix())
            post_process(save_path)
            time.sleep(0.1)
        except Exception as e:
            print(TEXTs, VOICEs, e)
    ...


if __name__ == "__main__":
    main()
    ...
