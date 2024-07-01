import numpy as np
from audio_utils import AudioWriter, AudioReader
from stft_utils import Stft
from tqdm import tqdm


def db2mag(db):
    return 10 ** (db / 20)


def process_subband():
    win_size, hop_size, sr, gain, in_channels = 512, 256, 16000, 30, 67
    use_channels = list(range(in_channels))
    chn_list_per_band = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [0, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        [0, 19, 21, 23, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        # [0, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        [0, 31, 33, 35, 37, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        [0, 43, 45, 47, 49, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
        [0, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
    ]

    bin_range_per_band = [
        [224, 256],
        [113, 223],
        [57, 112],
        # [0, 56],
        [29, 56],
        [15, 28],
        [0, 14],
    ]

    # ar = AudioReader(r"D:\Temp\save_wav_out", sr)
    ar = AudioReader(r"D:\Temp\pra_sim_out\AG_anechoic_snr30", sr)
    aw = AudioWriter("data/output", sr)
    stft = Stft(win_size, hop_size, in_channels, 1)
    out_prefix = f"subband_{in_channels}mic_v2"

    for i, data in tqdm(enumerate(ar.read_audio_data(hop_size))):
        in_data = data[use_channels] * db2mag(gain)
        out_data1 = np.mean(in_data, 0)

        in_spec = stft.transform(in_data)
        out_spec = np.zeros(in_spec.shape[1], dtype=complex)
        for chn_list, bin_range in zip(chn_list_per_band, bin_range_per_band):
            use_chns = chn_list
            use_bins = slice(bin_range[0], bin_range[1] + 1)
            out_spec[use_bins] = np.mean(in_spec[use_chns, use_bins], 0)
        out_data2 = stft.inverse(out_spec)

        aw.write_data_list(f"{out_prefix}_in", [in_data[0]])
        aw.write_data_list(f"{out_prefix}_out1", [out_data1])
        if i > 0:
            aw.write_data_list(f"{out_prefix}_out2", [out_data2])
    aw.write_data_list(f"{out_prefix}_out2", [np.zeros(hop_size)])
    ...


def process_multi():
    win_size, hop_size, sr, gain, in_channels = 512, 256, 16000, 30, 43
    use_channels = list(range(in_channels))

    # ar = AudioReader(r"D:\Temp\save_wav_out", sr)
    ar = AudioReader(r"D:\Temp\pra_sim_out\AG_anechoic_snr30", sr)
    aw = AudioWriter("data/output", sr)
    stft = Stft(win_size, hop_size, in_channels, 1)
    out_prefix = "sum_43mic"

    for i, data in tqdm(enumerate(ar.read_audio_data(hop_size))):
        in_data = data[use_channels] * db2mag(gain)
        out_data1 = np.mean(in_data, 0)

        in_spec = stft.transform(in_data)
        out_spec = np.mean(in_spec, 0)
        out_data2 = stft.inverse(out_spec)

        aw.write_data_list(f"{out_prefix}_in", [in_data[0]])
        aw.write_data_list(f"{out_prefix}_out1", [out_data1])
        if i > 0:
            aw.write_data_list(f"{out_prefix}_out2", [out_data2])
    aw.write_data_list(f"{out_prefix}_out2", [np.zeros(hop_size)])
    ...


def main():
    # process_multi()
    process_subband()
    ...


if __name__ == "__main__":
    main()
    ...
