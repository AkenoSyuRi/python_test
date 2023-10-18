import wave
from pathlib import Path

import numpy as np


def wav_data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
    assert in_audio_path.endswith(".wav"), "support wav format only"
    frame_len = int(sr * frame_time)

    with wave.Wave_read(in_audio_path) as fp:
        assert fp.getframerate() == sr
        assert fp.getnchannels() == 1
        assert fp.getsampwidth() == 2

        desired_buff_len = frame_len * 2
        buff = fp.readframes(frame_len)
        while len(buff) == desired_buff_len:
            clip = np.frombuffer(buff, dtype=np.short)
            if ret_bytes:
                yield clip.tobytes()
            else:
                yield clip / 32768
            buff = fp.readframes(frame_len)


if __name__ == "__main__":
    in_audio_path = r"F:\Test\3.dataset\0.original_backup\videoplayback_3h_32k.wav"
    out_dir = r"F:\Test\3.dataset\2.noise\youtube_air_conditioner_cut_200"
    for idx, clip in enumerate(
        wav_data_generator(in_audio_path, 10, sr=32000, ret_bytes=True)
    ):
        if idx == 200:
            break
        out_audio_path = Path(out_dir, f"air_conditioner_cut_fileid_{idx}.wav")
        with wave.Wave_write(out_audio_path.as_posix()) as fp:
            fp.setsampwidth(2)
            fp.setframerate(32000)
            fp.setnchannels(1)

            fp.writeframes(clip)
