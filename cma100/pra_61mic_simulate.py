import shutil
from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra
import soundfile
from acoustics.signal import highpass


def get_cma100_mics_coordinates(
    coord_z=3,
    delta_theta=np.pi / 6,
    dis2mic0_per_circle=(0.008, 0.016, 0.032, 0.064, 0.128),
):
    num_mic_per_circle = int(2 * np.pi // delta_theta)  # 12
    all_coords = [[0, 0, coord_z]]
    all_mics = [0] + list(range(1, len(dis2mic0_per_circle) * num_mic_per_circle + 1))

    theta = 0
    for d in dis2mic0_per_circle:
        for _ in range(num_mic_per_circle):
            x = round(d * np.cos(theta), 4)
            y = round(d * np.sin(theta), 4)
            coord = [x, y, coord_z]
            all_coords.append(coord)
            theta += delta_theta

    return np.array(all_coords), all_mics


def get_mic_coords(
    coord_z=0,
    delta_theta=np.pi / 3,
    dis2mic0_per_circle=(0.0175, 0.0350, 0.0700, 0.1400, 0.2800),
):
    def keep_in_eps(num, eps=1e-7):
        if abs(num) < eps:
            return 0
        return num

    num_mic = 55
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
        rms = np.sqrt(np.mean(audio_data**2))
        return rms

    # 将语音信号能量转化到TargetDb
    rms = cal_rms(audio_data)
    scalar = 10 ** (target_db / 20) / (rms + 1e-7)
    audio_data *= scalar
    return audio_data


def get_signal(audio_path, target_fs, target_db):
    data, fs = librosa.load(audio_path, sr=target_fs)
    data = convert_to_target_db(data, target_db)
    return data


if __name__ == "__main__":
    save_pcm = bool(1)
    fs = 16000
    rt60_tgt = 0.2
    room_dim = [10, 8, 5]
    mic_coords = get_mic_coords()[0] + np.array(
        [room_dim[0] / 2, room_dim[1] / 2, room_dim[2]]
    )  # mic0: 5,4,5
    mic_positions = mic_coords.T

    source1_position = [5, 4, 1]  # azimuth 90, elevation 60
    source2_position = [0, 8, 0]  # azimuth 141.34 (140, 145), elevation 25.115 (25, 30)
    noise1_position = source2_position
    source1_path = (
        r"F:\Test\4.play\指向性测试-单频2_16k.wav"
        # r"F:\Projects\PycharmProjects\cma100_beamforming\data\in_wav\001760.wav"
    )
    source2_path = (
        r"F:\Projects\PycharmProjects\cma100_beamforming\data\in_wav\007537.wav"
    )
    noise1_path = r"F:\Projects\PycharmProjects\cma100_beamforming\data\in_wav\air_conditioner_cut_fileid_1.wav"

    out_dir = Path(r"D:\Temp\pra_sim_out")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(exist_ok=True)

    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    # wall_material = pra.Material(energy_absorption=0.2, scattering=0.1)
    # room.set_wall_properties(wall_material)

    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

    room.add_source(source1_position, signal=get_signal(source1_path, fs, -30))
    # room.add_source(source2_position, signal=get_signal(source2_path, fs, -60))
    # room.add_source(noise1_position, signal=get_signal(noise1_path, fs, -60))

    room.simulate()

    for i, mic_signal in enumerate(room.mic_array.signals):
        # mic_signal = mic_signal[: 10 * fs]
        # mic_signal = highpass(mic_signal, 100, fs)
        if save_pcm:
            out_path = out_dir.joinpath(f"pra_chn{i:02d}.pcm")
            with open(out_path, "wb") as fp:
                fp.write((mic_signal * 32768).astype(np.int16).tobytes())
        else:
            out_path = out_dir.joinpath(f"pra_chn{i:02d}.wav")
            soundfile.write(out_path, mic_signal, fs)
        print(out_path)
