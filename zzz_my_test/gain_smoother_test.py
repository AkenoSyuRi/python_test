import librosa
import numpy as np
from audio_utils import AudioWriter
from tqdm import trange


class GainSmoother:
    def __init__(self, frame_len, n_frames):
        self.frame_len = frame_len  # length of a frame of data
        self.n_frames = n_frames  # how many frames take for smoothing

        self.fade_in_gain, self.fade_out_gain = self._get_smoothing_gain()
        self.fade_in_idx = 0  # which part of fade in gain should be used
        self.fade_out_idx = n_frames  # suppress the beginning frame

    def _get_smoothing_gain(self):
        buff_len = self.n_frames * self.frame_len
        fade_out_gain = np.linspace(1, 0, buff_len, endpoint=True)
        fade_in_gain = fade_out_gain[::-1]

        fade_in_gain = fade_in_gain.reshape(self.n_frames, -1)
        fade_out_gain = fade_out_gain.reshape(self.n_frames, -1)
        return fade_in_gain, fade_out_gain

    def apply(self, in_data, suppress):
        if suppress:  # branch for fade out process
            if self.fade_out_idx >= self.n_frames:
                out_data = in_data * self.fade_out_gain[-1, -1]
            else:
                out_data = in_data * self.fade_out_gain[self.fade_out_idx]
                self.fade_out_idx += 1
                self.fade_in_idx = self.n_frames - self.fade_out_idx
        else:  # branch for fade in process
            if self.fade_in_idx >= self.n_frames:
                out_data = in_data * self.fade_in_gain[-1, -1]
            else:
                out_data = in_data * self.fade_in_gain[self.fade_in_idx]
                self.fade_in_idx += 1
                self.fade_out_idx = self.n_frames - self.fade_in_idx
        return out_data


def main():
    in_wav_path = r"D:\Temp\athena_test_out\[sim]dbg_v1_d1_n1_1_inp.wav"
    all_data, sr = librosa.load(in_wav_path, sr=None)
    frame_len = 256
    total_frames = len(all_data) // frame_len
    all_data = all_data[: frame_len * total_frames]

    states = np.ones(total_frames, dtype=bool)
    idx = np.random.randint(100, total_frames // 2)
    states[idx : total_frames // 2] = False

    smoother = GainSmoother(frame_len, 100)
    aw = AudioWriter("data/output", sr)

    for i in trange(total_frames):
        in_data = all_data[i * frame_len : (i + 1) * frame_len]
        out_data = smoother.apply(in_data, states[i])
        aw.write_data_list("out_smoothed", [out_data])
    ...


if __name__ == "__main__":
    main()
    ...
