import numpy as np
import torch

from gruc.conv_stft import ConvSTFT


def stft_frame_wise(data, win_len, win_inc, window):
    res = []
    for i in range(0, len(data), win_inc):
        clip = data[i:i + win_len]
        if len(clip) < win_len:
            break
        ana_data = clip * window
        out_rfft = np.fft.rfft(ana_data).astype(np.complex64)
        res.append(out_rfft)
    return np.column_stack(res)


def stft_torch(data, win_len, win_inc, window):
    data = torch.FloatTensor(data)
    window = torch.FloatTensor(window)
    out_rfft = torch.stft(data, win_len, win_inc, win_len, window, False, return_complex=True)
    return out_rfft.numpy()


def export_complex_to_txt(out_txt_path, real_data, imag_data=None):
    """
    real_data: F,T
    """
    real_data = np.array(real_data)
    if imag_data is not None:
        imag_data = np.array(imag_data)
    with open(out_txt_path, 'w', encoding='utf8') as fp:
        assert real_data.ndim == 2
        n_frames = real_data.shape[1]
        for i in range(n_frames):
            fp.write(",".join(real_data[:, i].astype(str)) + "\n")
            if imag_data is not None:
                fp.write(",".join(imag_data[:, i].astype(str)) + "\n\n")


if __name__ == '__main__':
    # np.random.seed(0)
    win_len, win_inc, fft_bins = 1024, 512, 513
    window = np.hanning(win_len + 1)[:win_len].astype(np.float32)
    data = np.random.randn(win_inc * 10).astype(np.float32)
    stft = ConvSTFT(win_len, win_inc, win_len, win_type='hann', feature_type='complex')

    out0 = stft(torch.FloatTensor(data.reshape([1, -1]))).squeeze()
    out1 = stft_frame_wise(data, win_len, win_inc, window)
    out2 = stft_torch(data, win_len, win_inc, window)

    export_complex_to_txt("out0.txt", out0[:fft_bins], out0[fft_bins:])
    export_complex_to_txt("out1.txt", out1.real, out1.imag)
    export_complex_to_txt("out2.txt", out2.real, out2.imag)
    # np.testing.assert_allclose(out1, out2, atol=1e-5)
    ...
