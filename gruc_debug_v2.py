import numpy as np
import torch
from torch import nn


class GRUC_Network_Original(nn.Module):
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
            nn.ZeroPad2d((0, 0, 1, 0)),
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
        :param states: num_layers,N,hidden_units
        """
        out = self.input_layer(mag)  # N,T,hidden_units

        rnn_out, hstates = self.rnn_layer(out, states)  # N,T,hidden_units

        rnn_out = torch.unsqueeze(rnn_out, 1)  # N,C,H,W
        cnn_out = self.cnn_layer(rnn_out)  # N,C',H,W'
        cnn_out = torch.permute(cnn_out, [0, 2, 3, 1])
        cnn_out = torch.flatten(cnn_out, 2)  # N,H,W*C

        mask = self.output_layer(cnn_out)  # N,T,F
        estimated_mag = mag * mask
        return estimated_mag, hstates


class GRUC_Network_Framewise(nn.Module):
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


def stft_frame_wise(data, win_len, win_inc, window):
    res = []
    for i in range(0, len(data), win_inc):
        clip = data[i:i + win_len]
        if len(clip) < win_len:
            break
        ana_data = clip * window
        out_rfft = np.fft.rfft(ana_data).astype(np.complex64)
        res.append(out_rfft)
    return np.column_stack(res)


def stft_torch(data, win_len, win_inc, window):
    data = torch.FloatTensor(data)
    window = torch.FloatTensor(window)
    out_rfft = torch.stft(data, win_len, win_inc, win_len, window, False, return_complex=True)
    return out_rfft.numpy()


def export_data_to_txt(out_txt_path, real_data, imag_data=None):
    """
    real_data: F,T
    """
    real_data = np.array(real_data)
    if imag_data is not None:
        imag_data = np.array(imag_data)
    with open(out_txt_path, 'w', encoding='utf8') as fp:
        assert real_data.ndim == 2
        n_frames = real_data.shape[1]
        for i in range(n_frames):
            fp.write(",".join(real_data[:, i].astype(str)) + "\n")
            if imag_data is not None:
                fp.write(",".join(imag_data[:, i].astype(str)) + "\n\n")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units, sr = 1, 1024, 512, 1024, 3, 300, 32000
    in_pt_path = r"F:\Projects\PycharmProjects\python_test\data\models\GRUC\GRUC_0817_wSDR_drb_only_pre80ms_ep69.pth"
    state_dict = torch.load(in_pt_path, "cpu")

    # load net frame wise
    net_fw = GRUC_Network_Framewise(win_len, win_inc, fft_len, hidden_layers, hidden_units)
    state_dict0 = net_fw.state_dict()
    for v in state_dict0.values():
        torch.nn.init.zeros_(v)
    tar_keys = state_dict0.keys()
    for k, v in state_dict.items():
        if 'cnn_layer.1' in k:
            state_dict0[f'cnn_layer.0{k[11:]}'].copy_(v)
        elif 'cnn_layer.2' in k:
            state_dict0[f'cnn_layer.1{k[11:]}'].copy_(v)
        elif k in tar_keys:
            state_dict0[k].copy_(v)
    net_fw.eval()

    # load net original
    net_ori = GRUC_Network_Original(win_len, win_inc, fft_len, hidden_layers, hidden_units)
    for v in net_ori.state_dict().values():
        torch.nn.init.zeros_(v)
    net_ori.load_state_dict(state_dict, strict=False)
    net_ori.eval()

    mag = torch.randn(1, 2, win_inc + 1)
    h_states = torch.zeros(hidden_layers, 1, hidden_units)

    out1 = []
    ht1 = h_states
    for i in range(mag.shape[1]):
        out, ht1 = net_fw(mag[:, i, None], ht1)
        out1.append(out)
    out1 = torch.cat(out1, 1)

    out2, ht2 = net_ori(mag, h_states)
    ...
