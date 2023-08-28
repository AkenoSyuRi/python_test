import os
import wave
from pathlib import Path

import numpy as np
import torch
from audio_utils import AudioUtils
from scipy.signal import get_window
from torch import nn
from tqdm import tqdm


class GRUC_Network_Deploy(nn.Module):
    def __init__(
        self,
        win_len,
        win_inc,
        fft_len,
        hidden_layers,
        hidden_units,
        win_type="hann",
        dropout=0.2,
    ):
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
            nn.Linear(((hidden_units - 3) // 2 + 1) * 2, self.input_size),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        ...

    def forward0(self, mag, *states_list):
        """
        :param mag: N,T,F
        :param states_list: [1,N,hidden_units]*num_layers
        """
        out = self.input_layer(mag)  # N,T,hidden_units

        states = torch.cat(states_list)
        rnn_out, h_states = self.rnn_layer(out, states)  # N,T,hidden_units
        rnn_out = torch.unsqueeze(rnn_out, 2)  # N,C,H,W

        last_rnn_out = torch.unsqueeze(states_list[-1], 2)
        cnn_in = torch.cat([last_rnn_out, rnn_out], 2)

        cnn_out = self.cnn_layer(cnn_in)  # N,C',H,W'
        cnn_out = torch.permute(cnn_out, [0, 2, 3, 1])
        cnn_out = torch.flatten(cnn_out, 2)  # N,H,W*C

        mask = self.output_layer(cnn_out)  # N,T,F
        estimated_mag = mask * mag
        return estimated_mag, h_states

    def forward1(self, mag, *states_list):
        """
        :param mag: 1,F
        :param states_list: [1,hidden_units]*num_layers
        """
        out = self.input_layer(mag)  # 1,hidden_units

        states = torch.cat(states_list)  # hidden_layers,hidden_units
        rnn_out, h_states = self.rnn_layer(out, states)  # hidden_layers,hidden_units

        cnn_in = torch.cat([states_list[-1], rnn_out])  # 2,hidden_units
        return cnn_in, h_states

    def forward2(self, mag, cnn_in):
        """
        :param mag: 1,1,513
        :param cnn_in: 1,1,2,400
        :return:
        """
        cnn_out = self.cnn_layer(cnn_in)  # 1,2,1,(hidden_units-2)//2
        cnn_out = torch.permute(cnn_out, [0, 2, 3, 1])  # 1,1,(hidden_units-2)//2,2
        cnn_out = torch.flatten(cnn_out, 2)  # 1,1,W*2

        mask = self.output_layer(cnn_out)  # 1,1,input_size
        estimated_mag = mask * mag
        return estimated_mag


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


def get_gruc_network(
    in_pt_path, win_len, win_inc, fft_len, hidden_layers, hidden_units
):
    net = GRUC_Network_Deploy(win_len, win_inc, fft_len, hidden_layers, hidden_units)
    state_dict0 = net.state_dict()
    state_dict1 = torch.load(in_pt_path, "cpu")

    for v in state_dict0.values():
        torch.nn.init.zeros_(v)

    tar_keys = state_dict0.keys()
    for k, v in state_dict1.items():
        if k in tar_keys:
            state_dict0[k].copy_(v)

    net.eval()
    return net


def export_ncnn_by_pnnx(out_pt_path, net, *inputs):
    # pnnx_exe = r"F:\Projects\GitProjects\ncnn\tools\pnnx\build\install\bin\pnnx.exe"
    pnnx_exe = r"F:\Tools\pnnx-20230816-windows\pnnx.exe"
    ncnn2mem = r"F:\Projects\CLionProjects\third_party\ncnn\build_vs2022_release\install\bin\ncnn2mem.exe"

    print("=========== exporting TorchScript model ===========")
    model1 = torch.jit.trace(net, inputs)
    model1.save(out_pt_path)

    def get_shapes(*inputs):
        return ",".join(map(lambda x: str(list(x.shape)), inputs)).replace(" ", "")

    print("=========== exporting ncnn model ===========")
    cmd = f"{pnnx_exe} {out_pt_path} inputshape={get_shapes(*inputs)}"
    print(cmd)
    os.system(cmd)

    # print("=========== exporting ncnn mem header ===========")
    # base_path = Path(out_pt_path).parent / Path(out_pt_path).stem
    # cmd = f"{ncnn2mem} {base_path}.ncnn.param {base_path}.ncnn.bin {base_path}.id.h {base_path}.mem.h"
    # os.system(cmd)
    ...


def export_ncnn():
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units = (
        1,
        1024,
        512,
        1024,
        3,
        301,
    )
    in_pt_path = "../data/models/GRUC/GRUC_0823_wSDR_drb_only_rts_0.05_tam_0.05_ep25.pth"
    out_ncnn_dir = "../data/export"

    torch.set_grad_enabled(False)
    Path(out_ncnn_dir).mkdir(exist_ok=True)

    net = get_gruc_network(
        in_pt_path, win_len, win_inc, fft_len, hidden_layers, hidden_units
    )

    # export part1
    mag = torch.randn(1, fft_len // 2 + 1)
    states_list = [torch.randn(1, hidden_units) for _ in range(hidden_layers)]
    out_pt_path = Path(out_ncnn_dir) / (Path(in_pt_path).stem + "_p1.pt")
    net.forward = net.forward1
    export_ncnn_by_pnnx(out_pt_path, net, mag, *states_list)

    # export part2
    mag = torch.randn(1, 1, fft_len // 2 + 1)
    cnn_in = torch.randn(1, 1, 2, hidden_units)
    out_pt_path = Path(out_ncnn_dir) / (Path(in_pt_path).stem + "_p2.pt")
    net.forward = net.forward2
    export_ncnn_by_pnnx(out_pt_path, net, mag, cnn_in)

    os.remove("debug.bin")
    os.remove("debug.param")
    os.remove("debug2.bin")
    os.remove("debug2.param")
    ...


def infer(in_pt_path: str, in_wav_path: str, out_dir: str, add_window=True):
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units, sr = (
        1,
        1024,
        512,
        1024,
        3,
        300,
        32000,
    )
    out_wav_basename = f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};exp.wav"
    out_wav_path = os.path.join(out_dir, out_wav_basename)

    print(f"inference: {in_pt_path}, {in_wav_path}")
    net = get_gruc_network(
        in_pt_path, win_len, win_inc, fft_len, hidden_layers, hidden_units
    )

    states_list = [torch.zeros(1, hidden_units)] * hidden_layers

    ana_data = np.zeros(win_len)
    if add_window:
        window = get_window("hann", win_len) ** 0.5
    else:
        window = np.ones(win_len)
    output = np.zeros(win_inc)
    try:
        with wave.Wave_write(out_wav_path) as fp:
            fp.setsampwidth(2)
            fp.setnchannels(1)
            fp.setframerate(sr)
            for idx, data in enumerate(
                tqdm(AudioUtils.data_generator(in_wav_path, 0.016, sr=sr)), 1
            ):
                ana_data[:win_inc] = ana_data[win_inc:]
                ana_data[win_inc:] = data

                data_rfft = np.fft.rfft(ana_data * window)
                mag = np.abs(data_rfft).reshape([1, 1, -1])
                phase = np.angle(data_rfft).reshape([1, 1, -1])

                mag = torch.FloatTensor(mag)
                cnn_in, h_states = net.forward1(mag.flatten(1), *states_list)
                states_list = torch.split(h_states, 1)
                cnn_in = cnn_in.reshape(1, 1, 2, hidden_units)
                estimated_mag = net.forward2(mag, cnn_in)

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


if __name__ == "__main__":
    if bool(1):
        export_ncnn()
    else:
        infer(
            in_pt_path="../data/models/GRUC/GRUC_0823_wSDR_drb_only_rts_0.05_tam_0.05_ep25.pth",
            in_wav_path="../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav",
            out_dir="../data/out_data/GRUC",
            add_window=True,
        )
    ...
