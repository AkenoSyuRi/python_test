import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio import functional as F
from torchaudio.utils import download_asset

SAMPLE_RATE = 16000
SAMPLE_CLEAN = download_asset("tutorial-assets/mvdr/clean_speech.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/mvdr/noise.wav")

print(SAMPLE_CLEAN)
print(SAMPLE_NOISE)


def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(
        spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto"
    )
    axis.set_title(title)
    plt.colorbar(img, ax=axis)


def plot_mask(mask, title="Mask"):
    mask = mask.numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(mask, cmap="viridis", origin="lower", aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)


def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()


def generate_mixture(waveform_clean, waveform_noise, target_snr):
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise


waveform_clean, sr = torchaudio.load(SAMPLE_CLEAN)
waveform_noise, sr2 = torchaudio.load(SAMPLE_NOISE)
assert sr == sr2 == SAMPLE_RATE
# The mixture waveform is a combination of clean and noise waveforms with a desired SNR.
target_snr = 3
waveform_mix = generate_mixture(waveform_clean, waveform_noise, target_snr)

waveform_mix = waveform_mix.to(torch.double)
waveform_clean = waveform_clean.to(torch.double)
waveform_noise = waveform_noise.to(torch.double)

torchaudio.save("data/output/mix.wav", waveform_mix, SAMPLE_RATE)

N_FFT = 1024
N_HOP = 256
stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)
istft = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP)

stft_mix = stft(waveform_mix)
stft_clean = stft(waveform_clean)
stft_noise = stft(waveform_noise)

plot_spectrogram(stft_mix[0], "Spectrogram of Mixture Speech (dB)")
plot_spectrogram(stft_clean[0], "Spectrogram of Clean Speech (dB)")
plot_spectrogram(stft_noise[0], "Spectrogram of Noise (dB)")

REFERENCE_CHANNEL = 0


def get_irms(stft_clean, stft_noise):
    mag_clean = stft_clean.abs() ** 2
    mag_noise = stft_noise.abs() ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[REFERENCE_CHANNEL], irm_noise[REFERENCE_CHANNEL]


irm_speech, irm_noise = get_irms(stft_clean, stft_noise)
plot_mask(irm_speech, "IRM of the Target Speech")
plot_mask(irm_noise, "IRM of the Noise")

psd_transform = torchaudio.transforms.PSD()

psd_speech = psd_transform(stft_mix, irm_speech)
psd_noise = psd_transform(stft_mix, irm_noise)

mvdr_transform = torchaudio.transforms.SoudenMVDR()
stft_souden = mvdr_transform(
    stft_mix, psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL
)
waveform_souden = istft(stft_souden, length=waveform_mix.shape[-1])
waveform_souden = waveform_souden.reshape(1, -1)
torchaudio.save("data/output/bf_souden.wav", waveform_souden, SAMPLE_RATE)

plot_spectrogram(stft_souden, "Enhanced Spectrogram by SoudenMVDR (dB)")

# =============== RTF_MVDR =================
rtf_evd = F.rtf_evd(psd_speech)
rtf_power = F.rtf_power(psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)

mvdr_transform = torchaudio.transforms.RTFMVDR()

# compute the enhanced speech based on F.rtf_evd
stft_rtf_evd = mvdr_transform(
    stft_mix, rtf_evd, psd_noise, reference_channel=REFERENCE_CHANNEL
)
waveform_rtf_evd = istft(stft_rtf_evd, length=waveform_mix.shape[-1])

# compute the enhanced speech based on F.rtf_power
stft_rtf_power = mvdr_transform(
    stft_mix, rtf_power, psd_noise, reference_channel=REFERENCE_CHANNEL
)
waveform_rtf_power = istft(stft_rtf_power, length=waveform_mix.shape[-1])

plot_spectrogram(stft_rtf_evd, "Enhanced Spectrogram by RTFMVDR and F.rtf_evd (dB)")
waveform_rtf_evd = waveform_rtf_evd.reshape(1, -1)
torchaudio.save("data/output/bf_rtf_evd.wav", waveform_rtf_evd, SAMPLE_RATE)

plot_spectrogram(stft_rtf_power, "Enhanced Spectrogram by RTFMVDR and F.rtf_power (dB)")
waveform_rtf_power = waveform_rtf_power.reshape(1, -1)
torchaudio.save("data/output/bf_rtf_power.wav", waveform_rtf_power, SAMPLE_RATE)
# plt.show()
