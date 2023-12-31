# -*- coding: utf-8 -*-
import os
from pathlib import Path

import librosa
import soundfile
import torch
import torch.nn as nn
import torchaudio
from audio_utils import AudioUtils


class Simple_STFT_Layer(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256, window=None, device=None):
        super(Simple_STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        if window == "hanning":
            self.window = torch.hann_window(frame_len, device=device)
        else:
            assert window == "none"
            self.window = None

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            return_complex=True,
            center=False,
            window=self.window,
        )
        r = y.real
        i = y.imag
        mag = torch.clamp(r ** 2 + i ** 2, self.eps) ** 0.5
        phase = torch.atan2(i + self.eps, r + self.eps)
        return mag, phase


class Pytorch_InstantLayerNormalization(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
        Constructor
        """
        super(Pytorch_InstantLayerNormalization, self).__init__()
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


class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.rnn2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_states):
        """
        :param x:  [N, T, input_size]
        :param in_states: [2, N, hidden_size]
        :return:
        """
        in_state1, in_state2 = torch.split(in_states, 1)

        x1, out_state1 = self.rnn1(x, in_state1)
        x1 = self.drop(x1)
        x2, out_state2 = self.rnn2(x1, in_state2)
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        out_states = torch.cat((out_state1, out_state2))
        return mask, out_states


class Pytorch_DTLN_stateful(nn.Module):
    def __init__(
            self,
            frameLength=1024,
            hopLength=256,
            hidden_size=128,
            encoder_size=256,
            window="hanning",
            device=None,
    ):
        super(Pytorch_DTLN_stateful, self).__init__()
        self.frame_len = frameLength
        self.frame_hop = hopLength
        self.stft = Simple_STFT_Layer(
            frameLength, hopLength, window=window, device=device
        )

        self.sep1 = SeperationBlock_Stateful(
            input_size=(frameLength // 2 + 1), hidden_size=hidden_size, dropout=0.25
        )

        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(
            in_channels=frameLength,
            out_channels=self.encoder_size,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = Pytorch_InstantLayerNormalization(
            channels=self.encoder_size
        )

        self.sep2 = SeperationBlock_Stateful(
            input_size=self.encoder_size, hidden_size=hidden_size, dropout=0.25
        )

        # TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(
            in_channels=self.encoder_size,
            out_channels=frameLength,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, x, in_state1, in_state2):
        """
        :param x:  [N, T]
        in_state: [2, N, hidden_size, 2]
        :return:
        """

        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        encoded_f = mask_2 * encoded_f
        estimated = encoded_f.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)
        # overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)
        return out, out_state1, out_state2


def pad_and_cut(data, fs, pad_duration_ms=16):
    assert data.ndim == 2
    pad_len = fs // 1000 * pad_duration_ms
    padded_data = torch.nn.functional.pad(data, [pad_len, 0])[..., :-pad_len]
    return (padded_data * 32768).type(torch.int16)


def process(model, in_wav_path, out_wav_path, hidden_size, sr, out_input=False):
    try:
        data, _ = librosa.load(in_wav_path, sr=sr)
        net_input = torch.FloatTensor(data).unsqueeze(0)

        in_state1 = torch.zeros(2, 1, hidden_size)
        in_state2 = torch.zeros(2, 1, hidden_size)
        net_output, _, _ = model(net_input, in_state1, in_state2)

        if out_input:
            data0 = net_input.squeeze().numpy()
            data1 = net_output.squeeze().detach().numpy()
            out_data = AudioUtils.merge_channels(data0, data1)
            soundfile.write(out_wav_path, out_data, sr)
        else:
            torchaudio.save(out_wav_path, pad_and_cut(net_output, sr), sr)
        print(out_wav_path)
    except Exception as e:
        print(e)
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
    frame_len, frame_hop, hidden_size, encoder_size, sr = 384, 128, 128, 256, 16000
    add_window, out_input = "none", bool(1)
    in_pt_path = "../data/models/drb_only/DTLN_0906_wSDR_drb_rts0.05_tam0.05_none_triple_endto1.0_16k_GRU_lr1e4_ep100.pth"
    in_wav_path_or_dir = r"../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_dir = r"../data/out_data/drb_only"
    # ============ config end ============ #

    torch.set_grad_enabled(False)
    model = Pytorch_DTLN_stateful(
        frameLength=frame_len,
        hopLength=frame_hop,
        hidden_size=hidden_size,
        encoder_size=encoder_size,
        window=add_window,
    )
    # print(model)
    # exit(0)
    model.load_state_dict(torch.load(in_pt_path, "cpu"))
    model.eval()

    out_wav_basename = (
        f"{Path(in_wav_path_or_dir).stem};{Path(in_pt_path).stem};true.wav"
    )
    out_wav_path = os.path.join(out_dir, out_wav_basename)
    process(
        model,
        in_wav_path_or_dir,
        out_wav_path,
        hidden_size,
        sr,
        out_input=out_input,
    )
    ...


if __name__ == "__main__":
    main()
    ...
