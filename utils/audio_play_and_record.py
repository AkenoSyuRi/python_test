import time
import wave
from pathlib import Path

import pyaudio
import tqdm
from loguru import logger


def play_wav_file(in_wav_path):
    p = pyaudio.PyAudio()
    try:
        with wave.Wave_read(in_wav_path) as wf:

            def callback(in_data, frame_count, time_info, status):
                data = wf.readframes(frame_count)
                return data, pyaudio.paContinue

            sr = wf.getframerate()
            chunk_size = (sr // 1000) * 32
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=sr,
                output=True,
                frames_per_buffer=chunk_size,
                stream_callback=callback,
            )
            # Wait for stream to finish (4)
            while stream.is_active():
                time.sleep(0.1)

            # 停止数据流
            stream.close()
    finally:
        p.terminate()
    ...


def record_wav_file(out_wav_path: str, *, sr=32000, channels=1):
    p = pyaudio.PyAudio()
    chunk_size = (sr // 1000) * 32  # 32ms
    try:
        stream = p.open(
            sr, channels, pyaudio.paInt16, input=True, frames_per_buffer=chunk_size
        )
        with wave.Wave_write(out_wav_path) as wf:
            wf.setframerate(sr)
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            try:
                # for _ in range(11):  # skip the dirty data at the beginning, 1024*11
                #     stream.read(chunk_size)

                logger.info("star recording")
                pbar = tqdm.tqdm()
                while True:
                    buff = stream.read(chunk_size)
                    wf.writeframes(buff)
                    pbar.update(1)
            except KeyboardInterrupt:
                logger.info("stop recording")
        stream.close()
    finally:
        p.terminate()


def play_and_record_wav(
    in_wav_path, out_wav_path, /, out_sr=32000, out_channels=1, out_samp_with=2
):
    p = pyaudio.PyAudio()
    try:
        with (
            wave.Wave_read(in_wav_path) as wf_in,
            wave.Wave_write(out_wav_path) as wf_out,
        ):
            wf_out.setframerate(out_sr)
            wf_out.setnchannels(out_channels)
            wf_out.setsampwidth(out_samp_with)

            def play_callback(in_data, frame_count, time_info, status):
                data = wf_in.readframes(frame_count)
                return data, pyaudio.paContinue

            # def record_callback(in_data, frame_count, time_info, status):
            #     wf_out.writeframes(in_data)
            #     return in_data, pyaudio.paContinue

            in_sr = wf_in.getframerate()
            chunk_size = (in_sr // 1000) * 32  # 32ms chunk

            record_stream = p.open(
                format=p.get_format_from_width(out_samp_with),
                channels=out_channels,
                rate=out_sr,
                input=True,
                frames_per_buffer=chunk_size,
                # stream_callback=record_callback,
            )
            for _ in range(11):  # skip the dirty data at the beginning
                record_stream.read(chunk_size)

            play_stream = p.open(
                format=p.get_format_from_width(wf_in.getsampwidth()),
                channels=wf_in.getnchannels(),
                rate=in_sr,
                output=True,
                frames_per_buffer=chunk_size,
                stream_callback=play_callback,
            )
            # Wait for play_stream to finish (4)
            while play_stream.is_active():
                # time.sleep(0.5)
                buff = record_stream.read(chunk_size)
                wf_out.writeframes(buff)

            # 停止数据流
            play_stream.close()
            record_stream.close()
    finally:
        p.terminate()
    ...


#
# @pytest.mark.skip
# def test_play_audio():
#     in_wav_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
#     play_wav_file(in_wav_path)
#     ...
#
#
# @pytest.mark.skip
# def test_record_audio():
#     out_wav_path = r"D:\Temp\a_16k.wav"
#     record_wav_file(out_wav_path)
#     ...
#
#
# def test_play_and_record_audio():
#     in_wav_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
#     out_wav_path = r"D:\Temp\b.wav"
#     play_and_record_wav(in_wav_path, out_wav_path)
#     ...


def main():
    in_wav_dir = Path(r"F:\Test\3.dataset\1.clean\new_aishell3_1w")
    out_wav_dir = Path(r"E:\lizhifeng\tb5w_out_record")

    for i, in_f in enumerate(in_wav_dir.glob("*.wav")):
        out_f = out_wav_dir.joinpath(in_f.stem + "_tb5w_record.wav")
        print(i, in_f)
        play_and_record_wav(in_f.as_posix(), out_f.as_posix())
        # if i >= 5:
        #     break
    ...


if __name__ == "__main__":
    main()
    ...
