import numpy as np
import rir_generator
from scipy import stats


def measure_rt60(rir, fs):
    reg_start_list = [-5, -5, -5, -5]
    reg_end_list = [-15, -20, -25, -30]
    rir = np.array(rir)

    # The power of the impulse response in dB
    power = rir**2
    energy = np.sum(power) - np.cumsum(power)  # Integration according to Schroeder
    energy = energy / np.max(np.abs(energy))

    energy_db = 10 * np.log10(energy + 1e-10)
    # plt.plot(energy_db)
    # plt.show()

    sig_len = len(rir)
    t_indices = np.linspace(0, sig_len / sr, sig_len)
    result = []
    for reg_start, reg_end in zip(reg_start_list, reg_end_list):
        idx_start = np.where(reg_start >= energy_db)[0][0]
        idx_end = np.where(reg_end >= energy_db)[0][0]
        slope, intercept, _, _, _ = stats.linregress(
            t_indices[idx_start:idx_end], energy_db[idx_start:idx_end]
        )
        rt15 = -60 / slope
        result.append(rt15)
    return result


if __name__ == "__main__":
    # rir, sr = librosa.load("data/in_data/gpu_rir_0_rt60_1.15s.wav", sr=None)
    sr = 32000
    rir = rir_generator.generate(
        340, sr, [2.3, 9.5, 1.5], [2.5, 1.2, 1.7], [5, 10, 3], reverberation_time=0.785
    ).reshape(-1)

    print(measure_rt60(rir, sr))
    import pyroomacoustics as pra

    print(pra.measure_rt60(rir, sr))
