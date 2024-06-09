from functools import partial
from pathlib import Path

import torch
import torchaudio
from icecream import ic


def sisdr_loss(preds, target, zero_mean: bool = False):
    """`Scale-invariant signal-to-distortion ratio` (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape
    """
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (
        torch.sum(noise**2, dim=-1) + eps
    )
    # return -torch.mean(val)
    return -10 * torch.mean(torch.log10(val))


def lsd_loss(y_est_t: torch.Tensor, y_true_t: torch.Tensor, n_fft=1024, eps=1e-7):
    """
    y_est_t: [B,T]
    y_true_t: [B,T]
    """
    stft = partial(
        torch.stft,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=y_est_t.device),
        center=False,
        return_complex=True,
    )
    yt_spec = stft(y_true_t)  # [B,F,T']
    ye_spec = stft(y_est_t)  # [B,F,T']

    yt_spec = torch.log10(torch.abs(yt_spec) ** 2 + eps)
    ye_spec = torch.log10(torch.abs(ye_spec) ** 2 + eps)
    loss = torch.mean(torch.sqrt(torch.mean((yt_spec - ye_spec) ** 2, dim=1)))
    return loss


def wSDRLoss(mixed, clean, clean_est, eps=1e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    # Batch preserving sum for convenience.
    def bsum(x):
        return torch.sum(x, dim=1)

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


if __name__ == "__main__":
    in_wav_dir = Path(r"D:\Temp\out_wav")

    file_indices = (0, 100)
    for idx in range(*file_indices):
        mixture_path = in_wav_dir.joinpath(f"{idx}_a_noisy.wav")
        clean_path = in_wav_dir.joinpath(f"{idx}_b_clean.wav")
        predict_path = in_wav_dir.joinpath(
            f"{idx}_a_noisy;DTLN_1130_snr_dnsdrb_triple_NewNoise_ep100;true.wav"
        )

        mixture, _ = torchaudio.load(mixture_path)
        target, _ = torchaudio.load(clean_path)
        predict, _ = torchaudio.load(predict_path)

        # loss = sisdr_loss(predict, target).item()
        # loss = wSDRLoss(mixture, target, predict).item()
        loss = lsd_loss(predict, target)
        ic(round(loss.item(), 5), idx)
    ...
