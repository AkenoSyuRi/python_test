# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn as nn
import torchaudio
from audio_utils import AudioUtils
from file_utils import FileUtils


class Simple_STFT_Layer(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256, window=None, device=None):
        super(Simple_STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        if window == 'hanning':
            self.window = torch.hann_window(frame_len, device=device)
        else:
            self.window = None

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, n_fft=self.frame_len, hop_length=self.frame_hop,
                       win_length=self.frame_len, return_complex=True, center=False, window=self.window)
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
        self.gamma = nn.Parameter(torch.ones(
            1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(
            1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(
            inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class SeperationBlock(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, dropout=0.25):
        super(SeperationBlock, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, (h, c) = self.rnn1(x)
        x1 = self.drop(x1)
        x2, _ = self.rnn2(x1)
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)
        return mask


class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_states):
        """

        :param x:  [N, T, input_size]
        :param in_states: [2, N, hidden_size, 2]
        :return:
        """
        h1_in, c1_in = in_states[:1, :, :, 0], in_states[:1, :, :, 1]
        h2_in, c2_in = in_states[1:, :, :, 0], in_states[1:, :, :, 1]

        h1_in = h1_in.contiguous()
        c1_in = c1_in.contiguous()
        h2_in = h2_in.contiguous()
        c2_in = c2_in.contiguous()

        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)
        out_states = torch.stack((h, c), dim=-1)
        return mask, out_states


class Pytorch_DTLN_stateful(nn.Module):
    def __init__(self, frameLength=1024, hopLength=256, hidden_size=128, encoder_size=256, window='hanning',
                 device=None):
        super(Pytorch_DTLN_stateful, self).__init__()
        self.frame_len = frameLength
        self.frame_hop = hopLength
        self.stft = Simple_STFT_Layer(frameLength, hopLength, window=window, device=device)

        self.sep1 = SeperationBlock_Stateful(input_size=(
                frameLength // 2 + 1), hidden_size=hidden_size, dropout=0.25)

        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(in_channels=frameLength, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = Pytorch_InstantLayerNormalization(
            channels=self.encoder_size)

        self.sep2 = SeperationBlock_Stateful(
            input_size=self.encoder_size, hidden_size=hidden_size, dropout=0.25)

        # TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frameLength,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x, in_state1, in_state2, clamp_min=None):
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
        mask, _ = self.sep1(mag, in_state1)
        if clamp_min is not None:
            # torch.clamp(mask, clamp_min, out=mask)  # FIXME: test
            mask = torch.clamp(mask, clamp_min)  # FIXME: test
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, _ = self.sep2(encoded_f_norm, in_state2)
        encoded_f = mask_2 * encoded_f
        estimated = encoded_f.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)  # B, encoder_size*2, T
        # overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)
        return out


def pad_and_cut(data, fs, pad_duration_ms=16):
    assert data.ndim == 2
    pad_len = fs // 1000 * pad_duration_ms
    padded_data = torch.nn.functional.pad(data, [pad_len, 0])[..., :-pad_len]
    return (padded_data * 32768).type(torch.int16)


def process(model, in_wav_path, out_wav_path, clamp_min=None, out_input=False):
    try:
        inputs, sr = torchaudio.load(in_wav_path)
        net_input = inputs[0, None]

        in_state1 = torch.zeros(2, 1, hidden_size, 2)
        in_state2 = torch.zeros(2, 1, hidden_size, 2)
        net_output = model(net_input, in_state1, in_state2, clamp_min=clamp_min)

        if out_input:
            n_channels = inputs.shape[0]
            data0_list = list(map(np.squeeze, np.split(inputs.numpy(), n_channels)))
            data1 = net_output.squeeze().detach().numpy()
            out_data = AudioUtils.merge_channels(*data0_list, data1)
            soundfile.write(out_wav_path, out_data, sr)
        else:
            torchaudio.save(out_wav_path, pad_and_cut(net_output, sr), sr)
        print(out_wav_path)
    except Exception as e:
        print(e)
    ...


def main():
    # ============ config start ============ #
    global hidden_size
    frame_len, frame_hop, hidden_size, encoder_size = 1024, 512, 128, 512
    add_window, out_input, clamp_min = True, False, None
    in_pt_path = "data/models/drb_only/DTLN_0809_snr_drb_only_hanning_pre100ms_rts_rir_ep17.pth"
    in_wav_path_or_dir = r"data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_dir = r"F:\Test\0.audio_test\model_predict_output"
    # out_dir = r"F:\Test\0.audio_test\train_data\model_out"
    # ============ config end ============ #

    torch.set_grad_enabled(False)
    model = Pytorch_DTLN_stateful(
        frameLength=frame_len,
        hopLength=frame_hop,
        hidden_size=hidden_size,
        encoder_size=encoder_size,
        window=('hanning' if add_window else 'none'),
    )
    model.load_state_dict(torch.load(in_pt_path, 'cpu'))
    model.eval()

    if os.path.isdir(in_wav_path_or_dir):
        files = FileUtils.glob_files(f'{in_wav_path_or_dir}/*.wav')
        assert len(files) > 0
        max_workers = len(files) if len(files) < os.cpu_count() else os.cpu_count()
        with ThreadPoolExecutor(max_workers=1) as ex:
            for in_wav_path in files:
                out_wav_basename = f"{Path(in_wav_path).stem};{Path(in_pt_path).stem}.wav"
                out_wav_path = os.path.join(out_dir, out_wav_basename)
                ex.submit(process, model, in_wav_path, out_wav_path, clamp_min=clamp_min, out_input=out_input)
    else:
        out_wav_basename = f"{Path(in_wav_path_or_dir).stem};{Path(in_pt_path).stem}.wav"
        out_wav_path = os.path.join(out_dir, out_wav_basename)
        process(model, in_wav_path_or_dir, out_wav_path, clamp_min=clamp_min, out_input=out_input)
    ...


if __name__ == "__main__":
    hidden_size = ...
    main()
    ...
