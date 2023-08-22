from pathlib import Path

import torch
import torchaudio
from torch import nn

from conv_stft import ConvSTFT, ConviSTFT


class GRUC_Network(nn.Module):
    def __init__(self, win_len, win_inc, fft_len, hidden_layers, hidden_units, win_type='hann', dropout=0.2):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.win_type = win_type
        self.dropout = dropout
        self.input_size = fft_len // 2 + 1

        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type=win_type, feature_type='real')
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type=win_type, feature_type='real')

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_units),
            nn.Tanh(),
        )
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

    def forward(self, inputs, states):
        """
        :param inputs: N,320000
        :param states: num_layers,N,hidden_units
        """
        mag, phase = self.stft(inputs)  # N,F,T

        mag_perm = torch.permute(mag, [0, 2, 1])
        out = self.input_layer(mag_perm)  # N,T,hidden_units

        rnn_out, hstates = self.rnn_layer(out, states)  # N,T,hidden_units

        rnn_out = torch.unsqueeze(rnn_out, 1)  # N,C,H,W
        cnn_in = torch.nn.functional.pad(rnn_out, (0, 0, 1, 0), "constant", 0)
        cnn_out = self.cnn_layer(cnn_in)  # N,C',H,W'
        cnn_out = torch.permute(cnn_out, [0, 2, 3, 1])
        cnn_out = torch.flatten(cnn_out, 2)  # N,H,W*C

        mask = self.output_layer(cnn_out)  # N,T,hidden_units-2

        est_mag = mask * mag_perm  # N,T,F
        est_mag_perm = torch.permute(est_mag, [0, 2, 1])
        outputs = self.istft(est_mag_perm, phase)  # N,1,320000
        outputs = torch.squeeze(outputs, 1)
        return outputs, mask, hstates


if __name__ == '0__main__':
    def _print_networks(models: list):
        print(f"This project contains {len(models)} models, the number of the parameters is: ")
        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                params_of_network += param.numel()
            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network
        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units = 4, 1024, 512, 1024, 3, 300
    net = GRUC_Network(win_len, win_inc, fft_len, hidden_layers, hidden_units)
    _print_networks([net])

    inputs = torch.randn(batch_size, 320000)
    states = torch.randn(hidden_layers, batch_size, hidden_units)

    outputs, mask = net(inputs, states)
    print(outputs.shape, mask.shape)
    ...

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    in_pt_path = r"../data/models/GRUC/GRUC_0819_wSDR_drb_only_rts_0.25_sin_win_ep67.pth"
    in_wav_path = "../data/in_data/TB5W-(软件版本V1.60-1)开关算法对比_L.wav"
    out_wav_path = f"../data/out_data/GRUC/{Path(in_wav_path).stem};{Path(in_pt_path).stem};true.wav"
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units = 1, 1024, 512, 1024, 3, 300

    net = GRUC_Network(win_len, win_inc, fft_len, hidden_layers, hidden_units)
    state_dict = torch.load(in_pt_path, "cpu")
    net.load_state_dict(state_dict)
    net.eval()

    inputs, sr = torchaudio.load(in_wav_path)
    states = torch.zeros(hidden_layers, 1, hidden_units)

    output, _, _ = net.forward(inputs, states)

    torchaudio.save(out_wav_path, output, sr)
    print(out_wav_path)
    ...