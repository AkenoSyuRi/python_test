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
    in_wav_dir = Path(r"D:\Temp\out_sisdr_dataset_wav_test")

    for idx in range(100):
        mixture_path = in_wav_dir.joinpath(f"{idx}_a_noisy.wav")
        clean_path = in_wav_dir.joinpath(f"{idx}_b_clean.wav")
        predict_path = in_wav_dir.joinpath(f"{idx}_a_noisy;model_0011;true.wav")

        mixture, _ = torchaudio.load(mixture_path)
        target, _ = torchaudio.load(clean_path)
        predict, _ = torchaudio.load(predict_path)

        loss = sisdr_loss(predict, target).item()
        # loss = wSDRLoss(mixture, target, predict).item()
        ic(round(loss, 5), idx)
    ...
