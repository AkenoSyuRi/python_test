import math
from typing import Optional

import torch
from torch.nn import functional as F

from .stft import ConvSTFT


def _remove_mean(x: torch.Tensor, dim: Optional[int] = -1) -> torch.Tensor:
    return x - x.mean(dim=dim, keepdim=True)


class AsymmetricLoss(torch.nn.Module):
    def __init__(
        self,
        window_size: int = 1024,
        hop_size: Optional[int] = 512,
        fft_len: Optional[int] = 1024,
        power: float = 0.5,
        eps: float = 1.0e-12,
        zero_mean: bool = True,
        scale_asym: float = 1.0,
    ):
        super().__init__()

        if fft_len is None:
            fft_len = int(2 ** math.ceil(math.log2(window_size)))

        self.stft = ConvSTFT(
            win_len=window_size,
            win_inc=window_size // 2 if hop_size is None else hop_size,
            fft_len=fft_len,
            win_type="hamming",
            feature_type="complex",
            fix=True,
        )
        self.feat_dim = fft_len // 2 + 1
        self.power = power
        self.eps = eps
        self.zero_mean = zero_mean

        self.scale_asym = scale_asym
        ...

    def _asym_loss(self, est_spectrograms, ref_spectrograms):
        """
        The PHASEN loss comprises two parts: amplitude and phase-aware losses

        ref_spectrum: [B, F*2, T], the reference spectrograms
        est_spectrum: [B, F*2, T], the estimated spectrograms
        """

        def _amplitude(x):
            r = x[:, : self.feat_dim, :]
            i = x[:, self.feat_dim :, :]
            return torch.sqrt(r**2 + i**2 + self.eps)

        # step 1: amplitude loss
        est_amplitude = _amplitude(est_spectrograms)
        ref_amplitude = _amplitude(ref_spectrograms)

        # Hyper-parameter p is a spectral compression factor and is set to 0.5
        est_compression_amplitude = est_amplitude**self.power
        ref_compression_amplitude = ref_amplitude**self.power

        # To solve the speech over-suppression issue
        # Reference: TEA-PSE: Tencent-Ethereal-Audio-Lab Personalized Speech Enhancement System for ICASSP 2022 DNS Challenge
        delta = ref_compression_amplitude - est_compression_amplitude
        asym_loss = torch.square(F.relu(delta)).mean()
        loss = self.scale_asym * asym_loss

        return loss

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
        """phase-aware forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the asymmetric loss
        """
        assert ref.shape == est.shape, (ref.shape, est.shape)

        if self.zero_mean:
            ref = _remove_mean(ref, dim=-1)
            est = _remove_mean(est, dim=-1)

        return self._asym_loss(self.stft(ref), self.stft(est))
