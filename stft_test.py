import librosa
import numpy as np
import soundfile
from scipy import signal

from dtln.simple_stft import SimpleSTFT


def enframe(data, win_len, win_inc):
    assert data.ndim == 2, "data must has shape [B,T]"
    data_len = data.shape[-1]
    n_frames = (data_len - win_len) // win_inc + 1
    indices = (
        np.tile(np.arange(win_len), [n_frames, 1])
        + np.tile(np.arange(0, n_frames * win_inc, win_inc), [win_len, 1]).T
    )
    frames_data = data[:, indices]
    return frames_data


def overlap_add(data, window, win_len, win_inc):
    """data: (n_frames, win_len)"""
    n_frames = data.shape[0]
    result_len = n_frames * win_inc + (win_len - win_inc)
    result = np.zeros(result_len)
    w_sum = get_win_sum_of_1frame(window, win_len, win_inc)
    for i in range(n_frames):
        clip = result[i * win_inc : i * win_inc + win_len]
        clip += data[i] * window
        clip[:win_inc] /= w_sum
    return result


def get_win_sum_of_1frame(window, win_len, win_inc):
    assert win_len % win_inc == 0, "win_len must be equally divided by win_inc"
    win_square = window**2
    overlap = win_len - win_inc
    win_tmp = np.zeros(overlap + win_len)

    loop_cnt = win_len // win_inc
    for i in range(loop_cnt):
        win_tmp[i * win_inc : i * win_inc + win_len] += win_square
    win_sum = win_tmp[overlap : overlap + win_inc]
    assert np.min(win_sum) > 0, "the nonzero overlap-add constraint is not satisfied"
    return win_sum


if __name__ == "0__main__":
    from matplotlib import pyplot as plt

    win_len, win_inc, win_type = 1024, 512, "hann"
    window = signal.get_window(win_type, win_len)
    win_sum = get_win_sum_of_1frame(window, win_len, win_inc)

    plt.plot(win_sum)
    plt.show()
    ...

if __name__ == "__main__":
    win_len, win_inc, win_type = 1024, 512, "hann"
    window = signal.get_window(win_type, win_len)
    in_clean_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"

    stft = SimpleSTFT(win_len, win_inc, win_type)

    # load data
    data, sr = librosa.load(in_clean_path, sr=None)
    data = data.reshape(1, -1)

    # stft1
    frames_data = enframe(data, win_len, win_inc).squeeze()
    in_data = frames_data * window
    my_stft_results = np.fft.rfft(in_data).astype(np.complex64)

    # stft2
    # my_stft_results2 = stft.transform(
    #     torch.FloatTensor(data), return_complex=True
    # ).numpy()
    # my_stft_results2 = my_stft_results2.squeeze().transpose()

    # istft
    out_data = np.fft.irfft(my_stft_results)
    output = overlap_add(out_data, window, win_len, win_inc)
    soundfile.write("a.wav", output, sr)

    # istft
    # out_data = np.fft.irfft(my_stft_results2)
    # output = overlap_add(out_data, window, win_len, win_inc)
    # soundfile.write("b.wav", output, sr)
    ...
