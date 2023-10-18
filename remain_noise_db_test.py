import librosa
import numpy as np
import soundfile
from audio_utils import AudioUtils


def pad_or_cut(data, nsamples):
    data_len = len(data)
    if data_len < nsamples:
        data = np.pad(data, (0, nsamples - data_len), mode="wrap")
        return data

    return data[:nsamples]


def compute_scale_by_snr(ori_data, ref_data, snr, eps):
    # calculate the rms amplitude of ori and ref signal
    ori_rms = np.sqrt(np.mean(ori_data**2)) + eps
    ref_rms = np.sqrt(np.mean(ref_data**2)) + eps

    # Calculate the desired clean signal rms amplitude based on SNR
    tar_rms = ref_rms * (10 ** (snr / 20))

    # Scale the clean signal to the desired RMS amplitude
    scale = tar_rms / ori_rms
    return scale


if __name__ == "__main__":
    noise_path = (
        r"F:\Downloads\synthetic\audio\train\soundbank\foreground\Blender\245423_2.wav"
    )

    sr = 32000
    noise_data1 = np.random.normal(0, 1, 10 * sr)
    noise_data1 /= np.max(np.abs(noise_data1)) + 1e-7

    noise_data1 = AudioUtils.merge_channels(noise_data1, noise_data1 * 0.002)
    soundfile.write("gaussian_noise_35dB.wav", noise_data1, sr)

    noise_data2, _ = librosa.load(noise_path, sr=sr)
    noise_data2 /= np.max(np.abs(noise_data2)) + 1e-7

    noise_data2 = AudioUtils.merge_channels(noise_data2, noise_data2 * 0.002)
    soundfile.write("esc_noise_35dB.wav", noise_data2, sr)
    ...
