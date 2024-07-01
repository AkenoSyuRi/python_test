import librosa
import numpy as np
import soundfile


def main():
    """
                 /
               / θ
    mic1(ref) ------ mic2
    """
    win_size, hop_size = 512, 256
    fs, f, duration, d, c, theta = 16000, 2000, 5, 0.07, 343, 60
    t = np.linspace(0, duration, duration * fs, endpoint=False)

    mic1_sig = 0.6 * np.sin(2 * np.pi * f * t)
    mic1_stft = librosa.stft(
        mic1_sig, n_fft=win_size, hop_length=hop_size, win_length=win_size
    )

    tau = d * np.cos(np.deg2rad(theta)) / c
    print("the theoretical offset of samples:", round(tau * fs))
    # mic1为参考，1j表示mic2相对mic1，相位提前了这么多（如果是-1j，则表示mic2相对于mic1相位滞后了这么多）
    # 相位提前，在时域上体现的是波形整体前移
    phase = np.exp(1j * 2 * np.pi * f * tau)
    mic2_stft = mic1_stft * phase

    mic2_sig = librosa.istft(
        mic2_stft, n_fft=win_size, hop_length=hop_size, win_length=win_size
    )
    soundfile.write("data/output/mic1_sig.wav", mic1_sig, fs)
    soundfile.write("data/output/mic2_sig.wav", mic2_sig, fs)
    ...


if __name__ == "__main__":
    main()
    ...
