import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra
from audio_utils import AudioWriter


# def get_mic_coords(
#     coord_z=0,
#     delta_theta=np.pi / 3,
#     dis2mic0_per_circle=(0.0175, 0.0350, 0.0700, 0.1400, 0.2800, 0.5600),
# ):
#     def keep_in_eps(num, eps=1e-7):
#         if abs(num) < eps:
#             return 0
#         return num
#
#     assert len(dis2mic0_per_circle) > 1
#
#     num_mic = 1 + 6 + 12 * (len(dis2mic0_per_circle) - 1)
#     mic_ids = list(range(num_mic))
#     mic_coords = [[0, 0, coord_z] for _ in range(num_mic)]
#
#     mic_idx = 1
#     num_mic_per_circle = int(2 * np.pi // delta_theta)
#     for circle, d in enumerate(dis2mic0_per_circle, 1):
#         if circle == 1:
#             idx_stride = 1
#         else:
#             idx_stride = 2
#
#         mic_idx_i0 = -1
#         theta = 0
#         for i in range(num_mic_per_circle):
#             x = keep_in_eps(d * np.cos(theta))
#             y = keep_in_eps(d * np.sin(theta))
#             mic_coords[mic_idx] = [x, y, coord_z]
#
#             if circle > 1:
#                 if i > 0:
#                     last2th_idx = mic_idx - 2
#                     last1th_idx = mic_idx - 1
#                     mic_coords[last1th_idx][0] = keep_in_eps(
#                         (mic_coords[mic_idx][0] + mic_coords[last2th_idx][0]) / 2
#                     )
#                     mic_coords[last1th_idx][1] = keep_in_eps(
#                         (mic_coords[mic_idx][1] + mic_coords[last2th_idx][1]) / 2
#                     )
#
#                     if i == num_mic_per_circle - 1:
#                         next1th_idx = mic_idx + 1
#                         mic_coords[next1th_idx][0] = keep_in_eps(
#                             (mic_coords[mic_idx][0] + mic_coords[mic_idx_i0][0]) / 2
#                         )
#                         mic_coords[next1th_idx][1] = keep_in_eps(
#                             (mic_coords[mic_idx][1] + mic_coords[mic_idx_i0][1]) / 2
#                         )
#                 else:
#                     mic_idx_i0 = mic_idx
#
#             mic_idx += idx_stride
#             theta += delta_theta
#     return np.array(mic_coords), mic_ids


def get_mic_coords(num_mic=4, mic_spacing=0.032):
    mic_ids = np.arange(num_mic)
    x = mic_ids * mic_spacing
    mic_coords = np.zeros([num_mic, 3])
    mic_coords[:, 0] = x
    return mic_coords, mic_ids


def convert_to_target_db(audio_data, target_db):
    if target_db > 0:
        target_db = -target_db

    # 将语音信号能量转化到TargetDb
    rms = np.sqrt(np.mean(audio_data**2, axis=-1))
    scalar = 10 ** (target_db / 20) / (rms + 1e-7)
    audio_data *= scalar.reshape(-1, 1)
    return audio_data


def get_signal(audio_path, target_fs, target_db=None):
    data, fs = librosa.load(audio_path, sr=target_fs)
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

    print(f"src_pos: {src_pos}, az/el: {azimuth:.1f}/{elevation:.1f}")
    return azimuth, elevation


@dataclass
class SigInfo:
    sig: np.ndarray
    pos: np.ndarray | list
    delay: float = 0


class RoomDataSimulator:
    def __init__(self, room_size, mic_pos, fs, snr=None, rt60=None):
        self.room_size = room_size
        self.mic_pos = mic_pos
        self.fs = fs
        self.snr = snr
        self.rt60 = rt60

        self.sim_anechoic = rt60 is None
        self.add_noise = snr is not None

        self.center_mic_coord = np.mean(mic_pos, -1)
        self.room = self._create_room()

        print(f"add_reverb={not self.sim_anechoic}, add_noise={self.add_noise}")
        ...

    def _create_room(self):
        if self.sim_anechoic:
            e_absorption, max_order = 1.0, 0
        else:
            e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_size)
        room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
        room.add_microphone_array(pra.MicrophoneArray(self.mic_pos, room.fs))
        return room

    def map2sig_infos(self, in_wav_list, src_pos_list, delay: float = 0):
        assert len(in_wav_list) > 0 and len(in_wav_list) == len(
            src_pos_list
        ), "invalid inputs"

        sig_infos = []
        for in_wav, src_pos in zip(in_wav_list, src_pos_list):
            if isinstance(in_wav, np.ndarray):
                data = in_wav
            else:
                data, _ = librosa.load(in_wav, sr=self.fs)
            cal_source_direction(self.center_mic_coord, src_pos)
            sig_infos.append(SigInfo(data, src_pos, delay=delay))

        return sig_infos

    def simulate(self, *sig_infos: SigInfo, random_seed=0):
        assert len(sig_infos) > 0, "no input signals"

        for sig_info in sig_infos:
            self.room.add_source(sig_info.pos, sig_info.sig, sig_info.delay)

        if self.add_noise:
            np.random.seed(random_seed)
            self.room.simulate(snr=self.snr)
        else:
            self.room.simulate()
        ...

    def save(
        self,
        out_dir: str | Path,
        out_name: str,
        out_db: float = None,
        mono=True,
        save_pcm=False,
    ):
        if self.sim_anechoic:
            out_name += "_anechoic"
        else:
            out_name += f"_rt60_{self.rt60:.1f}s"
        if self.add_noise:
            out_name += f"_snr{self.snr}"

        if out_db is not None:
            out_sig = convert_to_target_db(self.room.mic_array.signals, out_db)
        else:
            out_sig = self.room.mic_array.signals

        out_dir = Path(out_dir) / out_name
        shutil.rmtree(out_dir, ignore_errors=True)
        aw = AudioWriter(out_dir, self.fs, save_pcm)

        if mono:
            aw.write_data_list(out_name, out_sig)
        else:
            aw.write_data_list(out_name, out_sig, onefile=True)
        ...


def main():
    save_pcm, save_mono, out_name = bool(0), bool(0), "ula"
    fs, room_size, out_db, rt60_tgt, snr = 16000, [8, 5, 3], -50, 0.4, 15

    mic0_coord = [room_size[0] / 2, room_size[1] / 2, 1.5]
    mic_coords = get_mic_coords()[0] + np.array(mic0_coord)  # mic0: 4,2.5,3.4
    # PlotUtils.plot_3d_coord(mic_coords)  # TODO: debug

    src_pos_list = [
        [mic0_coord[0], 0.5, 1.7],
    ]
    src_path_list = [
        r"F:\Test\4.play\007537_16k.wav",
        # r"F:\Test\4.play\lzf_speech_cut9s_16k.wav",
    ]
    out_dir = Path(r"D:\Temp\pra_sim_out")
    # =======================================================================

    rds = RoomDataSimulator(room_size, mic_coords.T, fs, snr=snr, rt60=rt60_tgt)
    sig_infos = rds.map2sig_infos(src_path_list, src_pos_list)
    rds.simulate(*sig_infos)
    rds.save(out_dir, out_name, out_db=out_db, mono=save_mono, save_pcm=save_pcm)
    ...


if __name__ == "__main__":
    main()
    ...
