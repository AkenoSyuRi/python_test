import torch


class SimpleSTFT:
    def __init__(self, frame_len=1024, frame_hop=256, window=None, device=None):
        super(SimpleSTFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        if window == "hann_window":
            self.window = torch.hann_window(frame_len + 2, device=device)[1:-1]
        elif window != "none":
            self.window = getattr(torch, window)(frame_len, device=device)
            assert self.window is not None, "invalid window name for torch"
        else:
            self.window = None

    def transform(self, x):
        """
        in: [B,T']
        out: [B,F,T]
        """
        y = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            window=self.window,
            center=False,
            return_complex=True,
        )
        mag = torch.abs(y)
        phase = torch.angle(y)
        return mag, phase

    def inverse(self, x):
        """
        in: [B,F,T]
        out: [B,T']
        """
        y = torch.istft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            window=self.window,
            center=False,
            return_complex=False,
        )
        return y


if __name__ == "__main__":
    from icecream import ic
    import torchaudio

    win_len, win_inc, window = 768, 256, "hann_window"

    in_clean_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
    inputs, sr = torchaudio.load(in_clean_path)
    stft = SimpleSTFT(win_len, win_inc, window=window)
    ic(inputs.shape)
    mag, phase = stft.transform(inputs)
    ic(mag.shape)
    spec = mag * torch.exp(1j * phase)
    output = stft.inverse(spec)
    ic(output.shape)
    torchaudio.save("a.wav", output, sr)

    # inputs = torch.rand(2, 320000)
    # stft = SimpleSTFT(win_len, win_inc, window=window)
    # mag, phase = stft.transform(inputs)
    # ic(mag.shape)
    # spec = mag * torch.exp(1j * phase)
    # output = stft.inverse(spec)
    # ic(output.shape)
    ...
