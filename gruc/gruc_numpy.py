import os
import wave
from pathlib import Path

import numpy as np
import torch
from audio_utils import AudioUtils
from scipy.signal import get_window
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FullyConnected:
    def __init__(self, weight, bias):
        """
        weight: (in_features, out_features)
        bias: (out_features,)
        """
        assert weight.ndim == 2 and bias.ndim == 1

        self.in_features = weight.shape[0]
        self.out_features = weight.shape[1]

        assert bias.shape[0] == self.out_features

        self.weight = weight
        self.bias = bias
        ...

    def __call__(self, inputs):
        """
        inputs: (..., in_features)
        """
        assert inputs.shape[-1] == self.in_features

        output = np.matmul(inputs, self.weight) + self.bias
        return output


class GruCell:
    def __init__(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
        weight_ih: (input_size, hidden_size * 3)
        weight_ih: (hidden_size, hidden_size * 3)
        bias: (hidden_size * 3)
        """
        assert weight_ih_l0.ndim == weight_hh_l0.ndim == 2
        assert bias_ih_l0.ndim == bias_hh_l0.ndim == 1

        self.input_size = weight_ih_l0.shape[0]
        self.hidden_size = weight_hh_l0.shape[0]

        assert (
            weight_ih_l0.shape[-1]
            == weight_hh_l0.shape[-1]
            == bias_ih_l0.shape[-1]
            == bias_hh_l0.shape[-1]
            == self.hidden_size * 3
        )

        self.w_ir, self.w_iz, self.w_in = np.split(weight_ih_l0, 3, axis=-1)
        self.w_hr, self.w_hz, self.w_hn = np.split(weight_hh_l0, 3, axis=-1)

        self.b_ir, self.b_iz, self.b_in = np.split(bias_ih_l0, 3, axis=-1)
        self.b_hr, self.b_hz, self.b_hn = np.split(bias_hh_l0, 3, axis=-1)
        ...

    def __call__(self, inputs, h0):
        """
        inputs: (1, 1, input_size)
        h0, c0: (1, 1, hidden_size)
        """
        rt = sigmoid(
            np.matmul(inputs, self.w_ir)
            + self.b_ir
            + np.matmul(h0, self.w_hr)
            + self.b_hr
        )
        zt = sigmoid(
            np.matmul(inputs, self.w_iz)
            + self.b_iz
            + np.matmul(h0, self.w_hz)
            + self.b_hz
        )
        nt = np.tanh(
            np.matmul(inputs, self.w_in)
            + self.b_in
            + rt * (np.matmul(h0, self.w_hn) + self.b_hn)
        )

        ht = (1 - zt) * nt + zt * h0

        return ht


class Conv2dBNRelu:
    def __init__(
        self,
        cnn_weight,
        cnn_bias,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        stride=(1, 2),
    ):
        assert cnn_weight.ndim == 4
        self.in_channels = cnn_weight.shape[1]
        self.out_channels = cnn_weight.shape[0]
        self.kernel_size = cnn_weight.shape[2:]
        self.stride = stride

        self.cnn_weight = cnn_weight
        self.cnn_bias = cnn_bias
        self.bn_weight = bn_weight.reshape(1, -1, 1, 1)
        self.bn_bias = bn_bias.reshape(1, -1, 1, 1)
        self.bn_running_mean = bn_running_mean.reshape(1, -1, 1, 1)
        self.bn_running_var = bn_running_var.reshape(1, -1, 1, 1)
        ...

    def bn2d(self, inputs):
        assert inputs.ndim == 4
        x_norm = (inputs - self.bn_running_mean) / np.sqrt(self.bn_running_var + 1e-5)
        y = x_norm * self.bn_weight + self.bn_bias
        return y

    def __call__(self, inputs):
        assert (
            inputs.ndim == 4
            and inputs.shape[1] == self.in_channels
            and inputs.shape[2] >= self.kernel_size[0]
            and inputs.shape[3] >= self.kernel_size[1]
        )

        out_w = (inputs.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        out = np.zeros([1, self.out_channels, 1, out_w])
        for i in range(self.out_channels):
            for j in range(out_w):
                idx = slice(
                    self.stride[1] * j, self.stride[1] * j + self.kernel_size[1]
                )
                out[:, i, :, j] = (
                    np.vdot(self.cnn_weight[i], inputs[0, :, :, idx]) + self.cnn_bias[i]
                )
        out = self.bn2d(out)
        out[out < 0] = 0
        return out


class GRUC_FrameWise:
    def __init__(self, state_dict, hidden_layers=3):
        self.hidden_layers = hidden_layers

        self.input_layer = FullyConnected(
            state_dict["input_layer.0.weight"].numpy().transpose(),
            state_dict["input_layer.0.bias"].numpy(),
        )

        self.rnn_layers = [
            GruCell(
                state_dict[f"rnn_layer.weight_ih_l{i}"].numpy().transpose(),
                state_dict[f"rnn_layer.weight_hh_l{i}"].numpy().transpose(),
                state_dict[f"rnn_layer.bias_ih_l{i}"].numpy(),
                state_dict[f"rnn_layer.bias_hh_l{i}"].numpy(),
            )
            for i in range(hidden_layers)
        ]

        self.cnn_layer = Conv2dBNRelu(
            state_dict[f"cnn_layer.0.weight"].numpy(),
            state_dict[f"cnn_layer.0.bias"].numpy(),
            state_dict[f"cnn_layer.1.weight"].numpy(),
            state_dict[f"cnn_layer.1.bias"].numpy(),
            state_dict[f"cnn_layer.1.running_mean"].numpy(),
            state_dict[f"cnn_layer.1.running_var"].numpy(),
        )

        self.output_layer = FullyConnected(
            state_dict["output_layer.0.weight"].numpy().transpose(),
            state_dict["output_layer.0.bias"].numpy(),
        )

    def __call__(self, mag, states):
        out = np.tanh(self.input_layer(mag))

        gru_out = out
        out_state_list = []
        state_list = np.split(states, self.hidden_layers)
        for i in range(self.hidden_layers):
            gru_out = self.rnn_layers[i](gru_out, state_list[i])
            out_state_list.append(gru_out)
        h_states = np.concatenate(out_state_list)

        cnn_in = np.concatenate([state_list[-1], gru_out], axis=1)
        cnn_in = np.expand_dims(cnn_in, axis=1)
        cnn_out = self.cnn_layer(cnn_in)

        cnn_out = np.transpose(cnn_out, (0, 2, 3, 1))
        cnn_out = cnn_out.reshape((1, 1, -1))

        mask = np.tanh(self.output_layer(cnn_out))
        estimated_mag = mask * mag
        return estimated_mag, h_states


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
    out_wav_basename = f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};np.wav"
    out_wav_path = os.path.join(out_dir, out_wav_basename)

    print(f"inference: {in_pt_path}, {in_wav_path}")
    state_dict = torch.load(in_pt_path, "cpu")
    net = GRUC_FrameWise(state_dict)

    h_states = np.zeros((hidden_layers, batch_size, hidden_units))

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

                estimated_mag, h_states = net(mag, h_states)
                # estimated_mag = estimated_mag.detach().numpy()
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
    torch.set_grad_enabled(False)
    infer(
        in_pt_path="../data/models/GRUC/GRUC_0819_wSDR_drb_only_rts_0.25_sin_win_ep67.pth",
        in_wav_path="../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav",
        out_dir="../data/out_data/tmp",
        add_window=True,
    )
    ...
