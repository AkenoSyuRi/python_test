import numpy as np
from audio_utils import AudioWriter


class DynamicRangeController:
    def __init__(
        self,
        pregain=30,
        threshold=-30,
        knee_width=30,
        ratio=2,
        eps=1e-7,
    ):
        self.pregain = self.db2lin(pregain)
        self.threshold = threshold
        self.knee_width = knee_width
        self.ratio = ratio
        self.eps = eps
        ...

    @staticmethod
    def db2lin(db):
        return 10 ** (db / 20)

    @staticmethod
    def lin2db(data):
        return 20 * np.log10(np.maximum(np.abs(data), 1e-7))

    def process(self, input_frame):
        output_frame = input_frame * self.pregain
        output_pregain = output_frame.copy()

        output_db = self.lin2db(output_frame)
        idx = np.where(output_db > self.threshold)
        decay_db = (output_db[idx] - self.threshold) * (self.ratio - 1) / self.ratio
        out_gain = self.db2lin(-decay_db)
        output_frame[idx] *= out_gain

        return output_pregain, output_frame

    def plot_gain_curve(self):
        ...


if __name__ == "__main__":
    sr, frame_len = 16000, 256
    skip_nsamples = int(sr * 1)
    in_pcm_path = (
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_no_airconditioner_split_mic00_c01.pcm"
    )
    out_wav_path = r"data/output/out_drc.wav"
    drc = DynamicRangeController()
    writer = AudioWriter(r"data/output", sr)

    with open(in_pcm_path, "rb") as fp:
        raw_data = fp.read()
        data_all = np.frombuffer(raw_data, np.short)[skip_nsamples:]
    data_all = data_all / 32768

    for i in range(0, len(data_all), frame_len):
        input_frame = data_all[i : i + frame_len]

        output_frame = drc.process(input_frame)
        writer.write_data_list("drc_out", output_frame, convert2short=True)
    ...
