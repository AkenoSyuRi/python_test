import os
import wave
from pathlib import Path

import numpy as np
import torch
from audio_utils import AudioUtils
from scipy.signal import get_window
from torch import nn
from tqdm import tqdm

from gruc.fc_gru_impl import CustomGRU


class GRUC_Network(nn.Module):
    def __init__(self, win_len, win_inc, fft_len, hidden_layers, hidden_units, win_type='hann', dropout=0.2,
                 use_fcGRU=False):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.win_type = win_type
        self.dropout = dropout
        self.input_size = fft_len // 2 + 1
        self.use_fcGRU = use_fcGRU

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_units),
            nn.Tanh(),
        )
        if use_fcGRU:
            self.rnn_layer = CustomGRU(
                input_size=hidden_units,
                hidden_size=hidden_units,
                num_layers=hidden_layers,
            )
        else:
            self.rnn_layer = nn.GRU(
                input_size=hidden_units,
                hidden_size=hidden_units,
                num_layers=hidden_layers,
                batch_first=True,
                dropout=dropout if hidden_layers > 1 else 0,
            )
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_units - 2, self.input_size),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        ...

    def forward(self, mag, states):
        """
        :param mag: N,T,F
        :param states_list: [num_layers,N,hidden_units]
        """
        out = self.input_layer(mag)  # N,T,hidden_units

        rnn_out, h_states = self.rnn_layer(out, states)  # N,T,hidden_units
        rnn_out = torch.unsqueeze(rnn_out, 1)  # N,C,H,W

        last_rnn_out = states[-1, None, None]
        cnn_in = torch.cat([last_rnn_out, rnn_out], 2)

        cnn_out = self.cnn_layer(cnn_in)  # N,C',H,W'
        cnn_out = torch.permute(cnn_out, [0, 2, 3, 1])
        cnn_out = torch.flatten(cnn_out, 2)  # N,H,W*C

        mask = self.output_layer(cnn_out)  # N,T,F
        estimated_mag = mask * mag
        return estimated_mag, h_states


def get_gruc_network(in_pt_path, win_len, win_inc, fft_len, hidden_layers, hidden_units, use_fcGRU=False):
    net = GRUC_Network(win_len, win_inc, fft_len, hidden_layers, hidden_units, use_fcGRU=use_fcGRU)
    state_dict0 = net.state_dict()
    state_dict1 = torch.load(in_pt_path, "cpu")

    for v in state_dict0.values():
        torch.nn.init.zeros_(v)

    tar_keys = state_dict0.keys()
    for k, v in state_dict1.items():
        # if 'cnn_layer.1' in k:
        #     state_dict0[f'cnn_layer.0{k[11:]}'].copy_(v)
        # elif 'cnn_layer.2' in k:
        #     state_dict0[f'cnn_layer.1{k[11:]}'].copy_(v)
        # elif k in tar_keys:
        #     state_dict0[k].copy_(v)
        if k in tar_keys:
            state_dict0[k].copy_(v)

    if net.use_fcGRU:
        for i in range(net.hidden_layers):
            keys = [
                f"rnn_layer.weight_ih_l{i}",
                f"rnn_layer.weight_hh_l{i}",
                f"rnn_layer.bias_ih_l{i}",
                f"rnn_layer.bias_hh_l{i}"
            ]
            net.rnn_layer.gru[i].set_weights(*map(state_dict1.get, keys))

    net.eval()
    return net


def infer(in_pt_path: str, in_wav_path: str, out_dir: str, add_window=True):
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units, sr = 1, 1024, 512, 1024, 3, 300, 32000
    out_wav_basename = f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};fw.wav"
    out_wav_path = os.path.join(out_dir, out_wav_basename)

    print(f'inference: {in_pt_path}, {in_wav_path}')
    net = get_gruc_network(in_pt_path, win_len, win_inc, fft_len, hidden_layers, hidden_units)

    h_states = torch.zeros(hidden_layers, batch_size, hidden_units)

    ana_data = np.zeros(win_len)
    if add_window:
        window = get_window('hann', win_len) ** 0.5
    else:
        window = np.ones(win_len)
    output = np.zeros(win_inc)
    try:
        with wave.Wave_write(out_wav_path) as fp:
            fp.setsampwidth(2)
            fp.setnchannels(1)
            fp.setframerate(sr)
            for idx, data in enumerate(tqdm(AudioUtils.data_generator(in_wav_path, 0.016, sr=sr)), 1):
                ana_data[:win_inc] = ana_data[win_inc:]
                ana_data[win_inc:] = data

                data_rfft = np.fft.rfft(ana_data * window)
                mag = np.abs(data_rfft).reshape([1, 1, -1])
                phase = np.angle(data_rfft).reshape([1, 1, -1])

                estimated_mag, h_states = net(torch.FloatTensor(mag), h_states)
                estimated_mag = estimated_mag.detach().numpy()
                enhanced_fft = estimated_mag * np.exp(1j * phase)
                out = np.fft.irfft(enhanced_fft.reshape(-1)) * window

                output += out[:win_inc]
                output = (output * 32768).astype(np.short)
                if idx > 1:
                    fp.writeframes(output.tobytes())
                output = out[win_inc:]
    except Exception as e:
        print(e)
    ...


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    infer(
        in_pt_path="../data/models/GRUC/GRUC_0819_wSDR_drb_only_rts_0.25_sin_win_ep67.pth",
        in_wav_path="../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav",
        out_dir="../data/out_data/GRUC",
        add_window=True,
    )
    ...
