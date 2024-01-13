# -*- coding: utf-8 -*-
import wave
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from tqdm import tqdm

from simple_stft import SimpleSTFT

torch.set_grad_enabled(False)


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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
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

    def forward(self, x, in_states):
        """
        :param x:  [1, 1, input_size]
        :param in_states:  [1, 1, 2*hidden_layers*1*hidden_size]
        :return:
        """
        in_states = torch.reshape(
            in_states, [2, self.hidden_layers, 1, self.hidden_size]
        )
        h0, c0 = in_states[0], in_states[1]
        x1, (ht, ct) = self.rnn(x, (h0, c0))

        mask = self.dense(x1)
        mask = self.sigmoid(mask)

        out_states = torch.stack([ht, ct], 0)
        out_states = torch.reshape(out_states, [1, 1, -1])
        return mask, out_states


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

    def forward(self, mag, phase, in_states1, in_states2):
        """
        mag, phase: 1,1,fft_bins
        :return:
        """
        # N, T, hidden_size
        mask1, out_states1 = self.sep1(mag, in_states1)
        estimated_mag = mask1 * mag

        s1_stft = estimated_mag * torch.exp(1j * phase)  # N,T,F
        y1 = torch.fft.irfft2(s1_stft, dim=-1)  # N,T,L
        y1 = y1.permute(0, 2, 1)  # N,L,T

        encoded_f = self.encoder_conv1(y1)  # B,Feat,T
        encoded_f = encoded_f.permute(0, 2, 1)  # B,T,Feat
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2, out_states2 = self.sep2(encoded_f_norm, in_states2)
        encoded_f = mask2 * encoded_f
        estimated = encoded_f.permute(0, 2, 1)  # B,Feat,T
        decoded_frame = self.decoder_conv1(estimated)  # B,L,T
        return decoded_frame, out_states1, out_states2


def data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
    data, _ = librosa.load(in_audio_path, sr=sr)
    frame_len = int(sr * frame_time)

    for i in range(0, len(data), frame_len):
        clip = data[i : i + frame_len]
        if len(clip) == frame_len:
            if ret_bytes:
                clip = (clip * 32768).astype(np.short)
                yield clip.tobytes()
            else:
                yield clip


def infer():
    (
        frame_len,
        frame_hop,
        rnn_hidden_sizes,
        rnn_num_layers,
        encoder_size,
        sr,
        window,
        data_type,
    ) = (
        512,
        128,
        (128, 128),
        (3, 3),
        256,
        16000,
        "none",
        np.float32,
    )
    in_wav_path = r"F:\Test\1.audio_test\1.in_data\input.wav"
    # in_wav_path = r"F:\Test\1.audio_test\1.in_data\大会议室_男声_降噪去混响测试_RK降噪开启.wav"
    in_pt_path = r"F:\Test\1.audio_test\2.in_models\dnsdrb\DTLN_0108_wMSE_dnsdrb_quater_rts0.4_pre1ms_finetune_NoFactor_ep100.pth"
    out_wav_path = Path(
        r"D:\Temp\tmp2", f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};bbb.wav"
    ).as_posix()
    ############# splitter ############
    # net = DTLN_Network(
    #     win_length=frame_len,
    #     hop_length=frame_hop,
    #     rnn_hidden_sizes=rnn_hidden_sizes,
    #     rnn_num_layers=rnn_num_layers,
    #     encoder_size=encoder_size,
    #     window=window,
    # )
    # net.load_state_dict(torch.load(in_pt_path, "cpu"))
    net = torch.jit.load(r"D:\Temp\tmp2\DTLN_0108_wMSE_dnsdrb_quater_rts0.4_pre1ms_finetune_NoFactor_ep100.pt", "cpu")
    net.eval()

    in_state1 = torch.zeros([1, 1, 2 * rnn_num_layers[0] * 1 * rnn_hidden_sizes[0]])
    in_state2 = torch.zeros([1, 1, 2 * rnn_num_layers[1] * 1 * rnn_hidden_sizes[1]])

    ana_data = np.zeros(frame_len)
    if window != "none":
        window = signal.get_window(window, frame_len)
    else:
        window = np.ones(frame_len)
    output = np.zeros(frame_len)
    with wave.Wave_write(out_wav_path) as fp:
        fp.setsampwidth(2)
        fp.setnchannels(1)
        fp.setframerate(sr)

        gen = data_generator(in_wav_path, frame_hop / sr, sr=sr)
        for _ in range(frame_len // frame_hop - 1):
            ana_data[:-frame_hop] = ana_data[frame_hop:]
            ana_data[-frame_hop:] = next(gen)

        for idx, data in enumerate(tqdm(gen), 1):
            ana_data[:-frame_hop] = ana_data[frame_hop:]
            ana_data[-frame_hop:] = data

            data_rfft = np.fft.rfft(ana_data * window)
            mag = np.abs(data_rfft).reshape([1, 1, -1]).astype(data_type)
            phase = np.angle(data_rfft).reshape([1, 1, -1]).astype(data_type)

            mag, phase = torch.FloatTensor(mag), torch.FloatTensor(phase)
            out_frame, in_state1, in_state2 = net(mag, phase, in_state1, in_state2)
            out = out_frame.numpy().reshape(-1)

            output += out
            out = output[:frame_hop] * 32768
            out = np.clip(out, -32768, 32767).astype(np.short)
            fp.writeframes(out.tobytes())
            output[:-frame_hop] = output[frame_hop:]
            output[-frame_hop:] = 0
    ...


if __name__ == "__main__":
    infer()
    ...
