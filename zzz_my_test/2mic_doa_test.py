from functools import partial

import librosa
import numpy as np
import soundfile

FRAME_TIME = 0.016


class DoaEstimator:
    def __init__(
        self,
        num_mic=2,
        mic_spacing=0.07,
        plot=False,
        c=340,
        fs=16000,
        win_len=512,
        win_inc=256,
        doa_freq_range=(100, 8000),
    ):
        self.num_mic = num_mic
        self.mic_spacing = mic_spacing
        self.plot = plot
        self.c = c
        self.fs = fs
        self.win_len = win_len
        self.win_inc = win_inc

        self.fft_bins = win_len // 2 + 1
        self.freq_res = fs / win_len
        self.eps = 1e-7

        self.doa_bin_range = list(
            np.floor(np.array(doa_freq_range) / self.freq_res).astype(int)
        )
        self.doa_bin_range[-1] += 1

        self.pre_Sl = np.zeros(self.fft_bins)
        self.pre_Rl = np.zeros(self.fft_bins)
        self.pre_Gl = np.zeros(self.fft_bins)

        ...

    def envelope_tracking(
        self,
        in_spec,
        lambda_s=FRAME_TIME / 0.1,
        mu_r=FRAME_TIME / 0.02,
        lambda_r=FRAME_TIME / 0.5,
        mu_g=FRAME_TIME,
        lambda_g=FRAME_TIME / 0.1,
    ):
        """in_spec: (2, fft_bins)"""
        out_spec = np.zeros(in_spec.shape[-1], dtype=complex)
        for k in range(*self.doa_bin_range):
            El = np.log10(np.abs(in_spec[0, k] * in_spec[1, k].conj()))

            # update Sl
            if El >= self.pre_Sl[k]:
                Sl = El
            else:
                Sl = lambda_s * El + (1 - lambda_s) * self.pre_Sl[k]

            # update Rl
            if El >= self.pre_Rl[k]:
                Rl = mu_r * El + (1 - mu_r) * self.pre_Rl[k]
            else:
                Rl = lambda_r * El + (1 - lambda_r) * self.pre_Rl[k]

            # update Gl
            if El >= self.pre_Gl[k]:
                Gl = mu_g * El + (1 - mu_g) * self.pre_Gl[k]
            else:
                Gl = lambda_g * El + (1 - lambda_g) * self.pre_Gl[k]

            eta = 1 / k  # TODO: The higher the frequency, the lower η becomes
            if Sl >= Rl and Sl >= Gl + eta:
                out_spec[k] = in_spec[0, k]
            else:
                print("xixi")

            self.pre_Sl[k] = Sl
            self.pre_Rl[k] = Rl
            self.pre_Gl[k] = Gl
        return out_spec


def main():
    sr, win_len, win_inc = 16000, 512, 256
    # in_wav_dir = Path(r"D:\Temp\save_wav_out")
    # in_wav_list = sorted(in_wav_dir.glob("*.wav"))
    pick_mic_ids, mic_spacing = [15, 9], 0.07
    # pick_mic_ids, mic_spacing = [13, 7], 0.07

    # in_wav_list = list(map(in_wav_list.__getitem__, pick_mic_ids))
    in_wav_list = [
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_small_meeting_room_split_mic07_c17.wav",
        r"D:\Temp\save_wav_out\in72chn_sorted_c0_small_meeting_room_split_mic13_c55.wav",
    ]

    doa = DoaEstimator(num_mic=len(pick_mic_ids), mic_spacing=mic_spacing, plot=bool(1))

    stft_func = partial(
        librosa.stft,
        n_fft=win_len,
        hop_length=win_inc,
        win_length=win_len,
        window="hann",
        center=False,
    )
    istft_func = partial(
        librosa.istft,
        n_fft=win_len,
        hop_length=win_inc,
        win_length=win_len,
        window="hann",
        center=False,
    )

    time_data_list = list(map(lambda x: librosa.load(x, sr=sr)[0], in_wav_list))
    in_data = np.stack(time_data_list)
    # in_data = np.stack([time_data_list[0], time_data_list[0]])
    in_spec = stft_func(in_data)

    out_spec = np.zeros(in_spec.shape[1:], dtype=complex)
    for i in range(0, in_spec.shape[-1]):
        ana_spec = in_spec[..., i]
        syn_spec = doa.envelope_tracking(ana_spec)
        out_spec[..., i] = syn_spec

    out_data = istft_func(out_spec)
    soundfile.write("data/output/envelope_tracking.wav", out_data, sr)
    ...


if __name__ == "__main__":
    main()
    ...
