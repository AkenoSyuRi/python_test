import librosa
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft

# Load the WAV file
audio_data, sample_rate = librosa.load(
    r"F:\BaiduNetdiskDownload\cv-corpus-13.0-delta-2023-03-09\zh-CN\clips\common_voice_zh-CN_36533616.mp3",
    sr=None,
)
sample_rate = int(sample_rate)
audio_data = (audio_data * 32768).astype(np.int16)


# Apply FFT to the audio data
frequency_domain = fft(audio_data)

# Define frequency bands
freq_bands = [(0, 8000), (8000, 16000), (16000, 24000)]

# Initialize arrays to store separated signals
separated_signals = []

# Process each frequency band
for freq_range in freq_bands:
    start_freq, end_freq = freq_range
    start_bin = int(start_freq * len(frequency_domain) / sample_rate)
    end_bin = int(end_freq * len(frequency_domain) / sample_rate)

    # Create a mask for the desired frequency range
    mask = np.zeros(len(frequency_domain))
    mask[start_bin:end_bin] = 1

    # Apply the mask in frequency domain
    separated_signal_freq_domain = frequency_domain * mask

    # Transform back to time domain using IFFT
    separated_signal_time_domain = np.real(ifft(separated_signal_freq_domain))

    separated_signals.append(separated_signal_time_domain)

# Write separated signals to new WAV files
for i, signal in enumerate(separated_signals):
    output_filename = f"output_band_{i}.wav"
    wavfile.write(output_filename, sample_rate, signal.astype(np.int16))
    print(f"Saved {output_filename}")
