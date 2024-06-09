# -*- coding: utf-8 -*-
import os
from pathlib import Path

import librosa
import soundfile
import torch
import torch.nn as nn
from audio_utils import AudioUtils

from simple_stft import SimpleSTFT


class InstantLayerNorm(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
        Constructor
        """
        super(InstantLayerNorm, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class SeparationBlock(nn.Module):
    def __init__(
        self,
        input_size=513,
        hidden_size=128,
        hidden_layers=2,
        dropout=0.25,
    ):
        super(SeparationBlock, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0,
            bidirectional=False,
        )

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:  [N, T, input_size]
        :return:
        """
        x1, _ = self.rnn(x)

        mask = self.dense(x1)
        mask = self.sigmoid(mask)

        return mask


class DTLN_Network(nn.Module):
    def __init__(
        self,
        win_length=1024,
        hop_length=256,
        rnn_hidden_sizes=(128, 128),
        rnn_num_layers=(2, 2),
        rnn_dropouts=(0.25, 0.25),
        encoder_size=256,
        window="none",
    ):
        super(DTLN_Network, self).__init__()
        assert 2 == len(rnn_hidden_sizes) == len(rnn_num_layers) == len(rnn_dropouts)
        self.frame_len = win_length
        self.frame_hop = hop_length
        self.rnn_hidden_sizes = rnn_hidden_sizes
        self.rnn_num_layers = rnn_num_layers
        self.fft_bins = win_length // 2 + 1

        self.stft = SimpleSTFT(win_length, hop_length, window=window)

        self.sep1 = SeparationBlock(
            input_size=self.fft_bins,
            hidden_size=rnn_hidden_sizes[0],
            hidden_layers=rnn_num_layers[0],
            dropout=rnn_dropouts[0],
        )

        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(
            in_channels=win_length,
            out_channels=self.encoder_size,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = InstantLayerNorm(channels=self.encoder_size)

        self.sep2 = SeparationBlock(
            input_size=self.encoder_size,
            hidden_size=rnn_hidden_sizes[1],
            hidden_layers=rnn_num_layers[1],
            dropout=rnn_dropouts[1],
        )

        self.decoder_conv1 = nn.Conv1d(
            in_channels=self.encoder_size,
            out_channels=win_length,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        """
        :param x:  [N, T]
        :return:
        """
        mag, phase = self.stft.transform(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask1 = self.sep1(mag)
        estimated_mag = mask1 * mag

        s1_stft = estimated_mag * torch.exp(1j * phase)  # N,T,F
        y1 = torch.fft.irfft2(s1_stft, dim=-1)  # N,T,L
        y1 = y1.permute(0, 2, 1)  # N,L,T

        encoded_f = self.encoder_conv1(y1)  # B,Feat,T
        encoded_f = encoded_f.permute(0, 2, 1)  # B,T,Feat
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2 = self.sep2(encoded_f_norm)
        encoded_f = mask2 * encoded_f
        estimated = encoded_f.permute(0, 2, 1)  # B,Feat,T
        decoded_frame = self.decoder_conv1(estimated)  # B,L,T
        # decoded_frame *= self.stft.window.view(1, -1, 1)
        # overlap and add
        batch, n_frames = x.shape
        output = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        output = output.reshape(batch, -1)
        # output /= self.stft.win_sum
        return output


def pad_and_cut(data, fs, pad_duration_ms=16):
    assert data.ndim == 2
    pad_len = fs // 1000 * pad_duration_ms
    padded_data = torch.nn.functional.pad(data, [pad_len, 0])[..., :-pad_len]
    return (padded_data * 32768).type(torch.int16)


def process_file(model, in_wav_path, out_wav_path, sr, out_input=False):
    try:
        data, _ = librosa.load(in_wav_path, sr=sr)
        net_input = torch.FloatTensor(data).unsqueeze(0)

        net_output = model(net_input)

        if out_input:
            data0 = net_input.squeeze().numpy()
            data1 = net_output.squeeze().detach().numpy()
            out_data = AudioUtils.merge_channels(data0, data1)
            soundfile.write(out_wav_path, out_data, sr)
        else:
            data = net_output.squeeze().detach().numpy()
            # torchaudio.save(out_wav_path, pad_and_cut(net_output, sr), sr)
            soundfile.write(out_wav_path, data, sr)
        print(out_wav_path)
    except Exception as e:
        print(e)
    ...


def process(model, in_pt_path, in_wav_path_or_list, out_dir, *, sr, out_input):
    if isinstance(in_wav_path_or_list, (list, tuple)):
        for in_wav_path in in_wav_path_or_list:
            out_wav_basename = (
                f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};true.wav"
            )
            out_wav_path = os.path.join(out_dir, out_wav_basename)
            process_file(
                model,
                in_wav_path,
                out_wav_path,
                sr,
                out_input=out_input,
            )
    else:
        out_wav_basename = (
            f"{Path(in_wav_path_or_list).stem};{Path(in_pt_path).stem};true.wav"
        )
        out_wav_path = os.path.join(out_dir, out_wav_basename)
        process_file(
            model,
            in_wav_path_or_list,
            out_wav_path,
            sr,
            out_input=out_input,
        )
    ...


def _print_networks(models: list):
    print(
        f"This project contains {len(models)} models, the number of the parameters is: "
    )
    params_of_all_networks = 0
    for idx, model in enumerate(models, start=1):
        params_of_network = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            params_of_network += param.numel()
        print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network
        print(model)
    print(
        f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
    )


def main():
    # ============ config start ============ #
    (
        frame_len,
        frame_hop,
        rnn_hidden_sizes,
        rnn_num_layers,
        encoder_size,
        window,
        sr,
        out_input,
    ) = (512, 128, (128, 128), (3, 3), 256, "none", 16000, bool(1))
    # in_pt_path_list = Path(r"F:\Test\1.audio_test\2.in_models\tmp").glob("*.pth")
    in_pt_path_list = [
        r"F:\Test\1.audio_test\2.in_models\dnsdrb\DTLN_0108_wMSE_dnsdrb_quater_rts0.4_pre1ms_finetune_NoFactor_ep100.pth",
    ]
    in_wav_path_or_list = (
        # r"F:\Test\1.audio_test\1.in_data\input.wav",
        # r"F:\Test\1.audio_test\1.in_data\大会议室_男声_降噪去混响测试_RK降噪开启.wav",
        # r"F:\Test\1.audio_test\1.in_data\大会议室_男声_降噪去混响测试_RK降噪开启_mic1.wav",
        # r"F:\Test\1.audio_test\1.in_data\小会议室_女声_降噪去混响测试.wav",
        # r"F:\Test\1.audio_test\1.in_data\中会议室_女声_降噪去混响测试.wav",
        # r"F:\Projects\PycharmProjects\athena_signal_test\data\output\sim_c3_NS_AGC_BF1_DOA_6mic_z0_in.wav",
        # r"F:\Projects\PycharmProjects\athena_signal_test\data\output\sim_c3_NS_AGC_BF1_DOA_6mic_z0_out.wav",
        r"F:\Projects\PycharmProjects\ns_wrapper\data\input\ns_in_c0_16k.wav",
    )
    # out_dir = r"F:\Test\1.audio_test\3.out_data\tmp"
    # in_wav_path_or_list = list(Path(r"D:\Temp\out_wav").glob("*_a_noisy.wav"))
    out_dir = r"D:\Temp"
    # ============ config end ============ #

    torch.set_grad_enabled(False)
    Path(out_dir).mkdir(exist_ok=True)
    for in_pt_path in in_pt_path_list:
        model = DTLN_Network(
            win_length=frame_len,
            hop_length=frame_hop,
            rnn_hidden_sizes=rnn_hidden_sizes,
            rnn_num_layers=rnn_num_layers,
            encoder_size=encoder_size,
            window=window,
        )
        model.load_state_dict(torch.load(in_pt_path, "cpu"))
        model.eval()
        print(in_pt_path)

        process(
            model,
            in_pt_path,
            in_wav_path_or_list,
            out_dir,
            sr=sr,
            out_input=out_input,
        )
    ...


if __name__ == "__main__":
    main()

    # net = DTLN_Network(1024, 512, [128, 128], [2, 2], [0.25, 0.25], 512)
    # _print_networks([net])
    ...
