from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra
from audio_utils import AudioWriter
from scipy.signal import windows


class Stft:
    def __init__(self, win_size, hop_size, in_channels, out_channels):
        self.win_size = win_size
        self.hop_size = hop_size
        self.overlap = win_size - hop_size
        self.fft_bins = win_size // 2 + 1

        self.window = np.hanning(win_size + 1)[1:]
        self.win_sum = self.get_win_sum_of_1frame(self.window, win_size, hop_size)

        self.in_win_data = np.zeros([in_channels, win_size])
        self.out_ola_data = np.zeros([out_channels, win_size])
        ...

    @staticmethod
    def get_win_sum_of_1frame(window, win_len, win_inc):
        assert win_len % win_inc == 0, "win_len must be equally divided by win_inc"
        win_square = window**2
        overlap = win_len - win_inc
        win_tmp = np.zeros(overlap + win_len)

        loop_cnt = win_len // win_inc
        for i in range(loop_cnt):
            win_tmp[i * win_inc : i * win_inc + win_len] += win_square
        win_sum = win_tmp[overlap : overlap + win_inc]
        assert (
            np.min(win_sum) > 0
        ), "the nonzero overlap-add constraint is not satisfied"
        return win_sum

    def transform(self, input_data):
        self.in_win_data[:, : self.overlap] = self.in_win_data[:, self.hop_size :]
        self.in_win_data[:, self.overlap :] = input_data

        ana_data = self.in_win_data * self.window
        spec_data = np.fft.rfft(ana_data, axis=-1)
        return spec_data

    def inverse(self, input_spec):
        syn_data = np.fft.irfft(input_spec, axis=-1)
        syn_data *= self.window

        self.out_ola_data[:, : self.overlap] = self.out_ola_data[:, self.hop_size :]
        self.out_ola_data[:, self.overlap :] = 0

        self.out_ola_data += syn_data

        output_data = self.out_ola_data[:, : self.hop_size] / self.win_sum
        return output_data


def db2lin(db):
    return 10 ** (db / 20)


def get_mic_coords(
    coord_z=0,
    delta_theta=np.pi / 3,
    dis2mic0_per_circle=(0.0175, 0.0350, 0.0700, 0.1400, 0.2800, 0.5600),
):
    def keep_in_eps(num, eps=1e-7):
        if abs(num) < eps:
            return 0
        return num

    assert len(dis2mic0_per_circle) > 1

    num_mic = 1 + 6 + 12 * (len(dis2mic0_per_circle) - 1)
    mic_ids = list(range(num_mic))
    mic_coords = [[0, 0, coord_z] for _ in range(num_mic)]

    mic_idx = 1
    num_mic_per_circle = int(2 * np.pi // delta_theta)
    for circle, d in enumerate(dis2mic0_per_circle, 1):
        if circle == 1:
            idx_stride = 1
        else:
            idx_stride = 2

        mic_idx_i0 = -1
        theta = 0
        for i in range(num_mic_per_circle):
            x = keep_in_eps(d * np.cos(theta))
            y = keep_in_eps(d * np.sin(theta))
            mic_coords[mic_idx] = [x, y, coord_z]

            if circle > 1:
                if i > 0:
                    last2th_idx = mic_idx - 2
                    last1th_idx = mic_idx - 1
                    mic_coords[last1th_idx][0] = keep_in_eps(
                        (mic_coords[mic_idx][0] + mic_coords[last2th_idx][0]) / 2
                    )
                    mic_coords[last1th_idx][1] = keep_in_eps(
                        (mic_coords[mic_idx][1] + mic_coords[last2th_idx][1]) / 2
                    )

                    if i == num_mic_per_circle - 1:
                        next1th_idx = mic_idx + 1
                        mic_coords[next1th_idx][0] = keep_in_eps(
                            (mic_coords[mic_idx][0] + mic_coords[mic_idx_i0][0]) / 2
                        )
                        mic_coords[next1th_idx][1] = keep_in_eps(
                            (mic_coords[mic_idx][1] + mic_coords[mic_idx_i0][1]) / 2
                        )
                else:
                    mic_idx_i0 = mic_idx

            mic_idx += idx_stride
            theta += delta_theta
    return np.array(mic_coords), mic_ids


def convert_to_target_db(audio_data, target_db):
    def cal_rms(audio_data):
        rms = np.sqrt(np.mean(audio_data**2, axis=-1))
        return rms

    if target_db > 0:
        target_db = -target_db

    # 将语音信号能量转化到TargetDb
    rms = cal_rms(audio_data)
    scalar = 10 ** (target_db / 20) / (rms + 1e-7)
    audio_data *= scalar.reshape(-1, 1)
    return audio_data


def get_signal(audio_path, target_fs, fc=None, target_db=None):
    if fc is not None:
        t = np.linspace(0, 10, int(target_fs * 10))
        data = np.sin(2 * np.pi * fc * t)
    else:
        data, _ = librosa.load(audio_path, sr=target_fs)
    if target_db:
        data = convert_to_target_db(data, target_db)
    return data


def cal_source_direction(center_coord, src_pos):
    x = src_pos[0] - center_coord[0]
    y = src_pos[1] - center_coord[1]
    cos_az = x / np.sqrt(x**2 + y**2)
    azimuth = round(np.rad2deg(np.arccos(cos_az)), 3)
    if y < 0:
        azimuth = 360 - azimuth

    z_diff = abs(src_pos[2] - center_coord[2])
    dis = np.sqrt(np.sum(np.square(center_coord - src_pos)))
    cos_el = z_diff / dis
    elevation = round(np.rad2deg(np.arccos(cos_el)), 1)

    print(
        f"src_pos: {src_pos}, az/el: {azimuth:.1f}/{elevation:.1f}, dis2mic0: {dis:.1f}"
    )
    return azimuth, elevation


def doa_estimation(in_sig, mic_coords):
    methods = ["SRP", "NormMUSIC", "MUSIC"]
    pick_mic_ids = [0, 1, 2, 3, 4]
    # pick_mic_ids = [0, 7, 9, 11, 13, 15, 17]
    # pick_mic_ids = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    win_len, win_inc = 512, 256
    window = np.hanning(win_len + 1)[1:]
    X = pra.transform.stft.analysis(in_sig[pick_mic_ids].T, win_len, win_inc, window)
    X = X.transpose((2, 1, 0))

    L = mic_coords[:, pick_mic_ids]
    for method in methods:
        doa_func = getattr(pra.doa, method)
        doa_algo = doa_func(L, fs, win_len, dim=3)

        doa_algo.locate_sources(X, freq_range=(2000, 6000))
        az = np.rad2deg(doa_algo.azimuth_recon).item()
        el = np.rad2deg(doa_algo.colatitude_recon).item()
        print(f"[{method}] \t estimated doa(az,el): {az:.3f}, {el:.3f}")
    ...


def beamforming(in_sig, ssl=45, out_gain=5):
    win_len, win_inc, num_mic = 512, 256, len(in_sig)
    weight = windows.chebwin(num_mic, ssl) * complex(1, 0)

    stft1 = Stft(win_len, win_inc, num_mic, 1)
    stft2 = Stft(win_len, win_inc, num_mic, 1)

    for i in range(0, in_sig.shape[-1], win_inc):
        in_frame = in_sig[:, i : i + win_inc]
        if in_frame.shape[-1] != win_inc:
            break

        X = stft1.transform(in_frame)

        S1 = np.einsum("i,ij->j", weight, X) / num_mic
        S2 = X.mean(0)

        out_frame1 = stft1.inverse(S1) * db2lin(out_gain)
        out_frame2 = stft2.inverse(S2)
        if i == 0:
            continue

        writer.write_data_list(f"bf_out_ssl{ssl}", [out_frame1])
        writer.write_data_list(f"bf_out_sum", [out_frame2])
    ...


if __name__ == "__main__":
    save_pcm, do_doa, sim_anechoic, add_noise = bool(0), bool(0), bool(0), bool(1)
    fs, rt60_tgt, room_size, snr, out_db = 16000, 0.2, [8, 5, 3.5], 20, 30
    pick_mic_ids = [37, 25, 0, 19, 31]
    mic_coords = get_mic_coords()[0] + np.array(
        [room_size[0] / 2, room_size[1] / 2, 1.5]
    )  # mic0: 4,2.5,3.4
    mic_coords = mic_coords[pick_mic_ids]
    mic_positions = mic_coords.T
    # PlotUtils.plot_3d_coord(mic_coords)
    # exit(0)
    center_coord = np.mean(mic_coords, axis=0)
    print("array_center_coord:", center_coord)
    source_pos_list = [
        [room_size[0] / 2, 0.5, 1.5],
        [0.5, 2.4, 1.5],
    ]
    src_path_list = [
        r"F:\Test\4.play\007537_16k.wav",
        r"F:\Test\4.play\lzf_speech_cut9s_16k.wav",
    ]

    out_dir = Path(r"D:\Temp\pra_sim_out")

    if sim_anechoic:
        e_absorption, max_order = 1.0, 0
    else:
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_size)
    room = pra.ShoeBox(
        room_size, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

    msg = "test"
    if sim_anechoic:
        msg += "_anechoic"
    else:
        msg += f"_rt60_{rt60_tgt:.1f}"
    if add_noise:
        msg += f"_snr{snr}"

    for i in range(len(source_pos_list)):
        az, el = cal_source_direction(center_coord, source_pos_list[i])
        room.add_source(
            source_pos_list[i],
            signal=get_signal(src_path_list[i], fs, fc=[1000, 2000][i]),
        )

    if add_noise:
        np.random.seed(0)
        room.simulate(snr=snr)
    else:
        room.simulate()

    out_sig = convert_to_target_db(room.mic_array.signals, out_db)
    out_dir /= msg

    # shutil.rmtree(out_dir, ignore_errors=True)
    writer = AudioWriter(out_dir, fs, save_pcm)
    writer.write_data_list(f"pra", out_sig)

    if do_doa:
        doa_estimation(out_sig, mic_positions)

    beamforming(out_sig)
    ...
