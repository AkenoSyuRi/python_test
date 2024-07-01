import librosa
import numpy as np
import webrtcvad
from audio_utils import AudioWriter


class BufferAdapter:
    def __init__(self, input_frame_len, output_frame_len):
        self.input_frame_len = input_frame_len
        self.output_frame_len = output_frame_len

        self.buf_len = np.lcm(input_frame_len, output_frame_len)
        self.buffer = np.zeros(self.buf_len)
        self.read_index = 0  # where to read the data for VAD
        self.write_index = 0  # where to write the data
        self.remain_size = 0  # the remaining data can be read

    def write(self, data_frame):
        assert len(data_frame) == self.input_frame_len
        self.buffer[
            self.write_index : self.write_index + self.input_frame_len
        ] = data_frame

        self.remain_size += self.input_frame_len
        self.write_index += self.input_frame_len

        if self.write_index >= self.buf_len:
            self.write_index = 0

    def read(self):
        assert self.readable(), "you need to check readable before you read"
        data = self.buffer[self.read_index : self.read_index + self.output_frame_len]

        self.remain_size -= self.output_frame_len
        self.read_index += self.output_frame_len

        if self.read_index >= self.buf_len:
            self.read_index = 0

        return data

    def readable(self):
        return self.remain_size >= self.output_frame_len


class Vad:
    def __init__(self, aggressive_mode=0, save_vad_file=bool(0)):
        self.sr, self.input_frame_len, self.output_frame_length = 16000, 256, 160
        self.mode = aggressive_mode
        self.vad = webrtcvad.Vad(self.mode)

        # 长帧转短帧（16ms -> 10ms）
        self.bfa = BufferAdapter(self.input_frame_len, self.output_frame_length)

        self.voice_active = False  # 当前帧是否为人声
        self.set2voice_n_frames = 1  # 只有连续n帧判断为人声，才将voice_active置为True
        self.set2noise_n_frames = 1  # 只有连续n帧判断为噪声，才将voice_active置为False
        self.consecutive_voice_cnt = 0  # 连续被vad识别为人声的帧数
        self.consecutive_noise_cnt = 0  # 连续被vad识别为噪声的帧数

        # TODO: debug
        self.save_vad_file = save_vad_file
        if save_vad_file:
            self.aw = AudioWriter("data/output", self.sr)
            self.one_vec = np.ones(self.input_frame_len) * 0.5
            self.zero_vec = np.zeros(self.input_frame_len)

    def is_speech(self, input_frame):
        """TODO:这里默认输入数据是浮点类型"""
        self.bfa.write(input_frame)

        while self.bfa.readable():
            data_frame = self.bfa.read()

            if not isinstance(data_frame.dtype, np.short):
                data_frame = (data_frame * 32768).astype(np.short)

            voice_active = bool(self.vad.is_speech(data_frame.tobytes(), self.sr))

            if voice_active:
                self.consecutive_voice_cnt += 1
                if self.consecutive_voice_cnt >= self.set2voice_n_frames:
                    self.voice_active = True
                    self.consecutive_noise_cnt = 0
            else:
                self.consecutive_noise_cnt += 1
                if self.consecutive_noise_cnt >= self.set2noise_n_frames:
                    self.voice_active = False
                    self.consecutive_voice_cnt = 0

        # TODO: debug
        if self.save_vad_file:
            vad_data = self.one_vec if self.voice_active else self.zero_vec
            self.aw.write_data_list(
                f"vad_mode{self.mode}_len{self.output_frame_length}",
                [input_frame, vad_data],
                onefile=True,
            )
        return self.voice_active


def data_generator(all_data, win_size):
    for i in range(0, len(all_data), win_size):
        if i + win_size > len(all_data):
            break
        data_clip = all_data[i : i + win_size]
        yield data_clip


def main():
    sr, mode = 16000, 0
    in_wav_path = r"D:\Temp\athena_test_out\[real]test_v0_d0_n1_1_inp.wav"
    vad = Vad(mode, save_vad_file=bool(1))

    all_data, _ = librosa.load(in_wav_path, sr=sr)
    for data in data_generator(all_data, vad.input_frame_len):
        vad.is_speech(data)
    ...


if __name__ == "__main__":
    main()
    ...
