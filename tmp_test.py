from pathlib import Path

import librosa
import soundfile
from matplotlib import pyplot as plt

if __name__ == "__main__":
    audio_file_path = Path(r"F:\BaiduNetdiskDownload\BZNSYP\Wave\002944.wav")
    y, sr = librosa.load(audio_file_path, sr=32000)

    plt.figure(figsize=(15, 7))

    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # librosa.display.specshow(mfcc_features, x_axis="time")
    # plt.colorbar()
    # plt.title("MFCC Feature")
    # plt.show()

    y_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfcc_features)
    soundfile.write("a.wav", y_reconstructed, sr)
    ...
