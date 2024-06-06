import shutil
import wave
from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra


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


def get_signal(audio_path, target_fs, target_db=None):
    data, fs = librosa.load(audio_path, sr=target_fs)
    if target_db:
        data = convert_to_target_db(data, target_db)
    return data


class AudioWriter:
    def __init__(self, out_wav_dir: Path, sr, write_pcm=False):
        self.out_wav_dir = out_wav_dir
        self.sr = sr
        self.files_map = dict()
        self.closed = False
        self.write_pcm = write_pcm
        self.format = ".pcm" if write_pcm else ".wav"

    def _get_or_open(self, name_without_ext: str):
        if name_without_ext in self.files_map:
            fp = self.files_map[name_without_ext]
        else:
            out_path = (self.out_wav_dir / name_without_ext).with_suffix(self.format)
            if self.write_pcm:
                fp = open(out_path, "wb")
            else:
                fp = wave.Wave_write(out_path.as_posix())
                fp.setsampwidth(2)
                fp.setnchannels(1)
                fp.setframerate(self.sr)
            self.files_map[name_without_ext] = fp
        return fp

    def write_data_list(self, prefix, data_list, convert2short=True):
        for i, data in enumerate(data_list):
            name = f"{prefix}_chn{i:02d}"
            fp = self._get_or_open(name)
            write_func = getattr(fp, "write" if self.write_pcm else "writeframes")
            if convert2short:
                write_func(self.to_short(data).tobytes())
            else:
                write_func(data.tobytes())

    def close(self):
        for fp in self.files_map.values():
            fp.close()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
        ...

    @staticmethod
    def to_short(data):
        data *= 32768
        np.clip(data, -32768, 32767, out=data)
        return data.astype(np.short)


def doa_estimation(in_sig, mic_coords):
    methods = ["SRP", "NormMUSIC", "MUSIC"]
    pick_mic_ids = [0, 7, 9, 11, 13, 15, 17]
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


if __name__ == "__main__":
    save_pcm, do_doa = bool(1), bool(0)
    fs, rt60_tgt, room_size, snr, out_db = 16000, 0.5, [7, 4, 3.5], 20, 50
    mic_coords = get_mic_coords()[0] + np.array(
        [room_size[0] / 2, room_size[1] / 2, room_size[2] - 0.1]
    )  # mic0: 3.5,2,3.4
    # PlotUtils.plot_3d_coord(mic_coords)
    mic_coords = mic_coords[:55]  # TODO: debug
    mic_positions = mic_coords.T
    source_pos_list = [[0.5, 2, 1.5]]  # (180, 60(or 50))

    src_path = r"F:\Test\4.play\007537.wav"
    out_dir = Path(r"D:\Temp\pra_sim_out")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(exist_ok=True)

    writer = AudioWriter(out_dir, fs, save_pcm)
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_size)
    for src_pos in source_pos_list:
        print(f"{src_pos=}")
        room = pra.ShoeBox(
            room_size, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )

        room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

        room.add_source(src_pos, signal=get_signal(src_path, fs))

        # room.simulate()
        room.simulate(snr=snr)

        out_sig = convert_to_target_db(room.mic_array.signals, out_db)
        writer.write_data_list(f"pra_snr{snr}", out_sig)

        if do_doa:
            doa_estimation(out_sig, mic_positions)
        ...
