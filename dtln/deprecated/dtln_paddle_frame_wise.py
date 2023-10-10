import wave
from pathlib import Path

import numpy as np
import paddle
import torch
from audio_utils import AudioUtils
from paddle import nn
from scipy import signal
from tqdm import tqdm


class SeparationBlock(nn.Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

        self.dense = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_states):
        in_h1, in_c1, in_h2, in_c2 = paddle.split(in_states, 4)
        x1, (out_h1, out_c1) = self.rnn1(x, (in_h1, in_c1))
        x2, (out_h2, out_c2) = self.rnn2(x1, (in_h2, in_c2))

        mask = self.dense(x2)
        mask = self.sigmoid(mask)
        out_states = paddle.concat([out_h1, out_c1, out_h2, out_c2], 0)
        return mask, out_states


class InstantLayerNormalization(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.eps = 1e-7

        gamma = self.create_parameter([1, 1, channels])
        beta = self.create_parameter([1, 1, channels])
        self.add_parameter("gamma", gamma)
        self.add_parameter("beta", beta)

    def forward(self, inputs: paddle.Tensor):
        # calculate mean of each frame
        mean = paddle.mean(inputs, axis=-1, keepdim=True)

        # calculate variance of each frame
        sub = inputs - mean
        variance = paddle.mean(paddle.square(sub), axis=-1, keepdim=True)
        # calculate standard deviation
        std = paddle.sqrt(variance + self.eps)
        outputs = sub / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class DTLN_Paddle(nn.Layer):
    def __init__(self,
                 win_len,
                 hidden_size,
                 encoder_size,
                 ):
        super().__init__()

        self.sep1 = SeparationBlock(input_size=win_len // 2 + 1, hidden_size=hidden_size)
        self.encoder_conv1 = nn.Conv1D(in_channels=win_len, out_channels=encoder_size, kernel_size=1, stride=1,
                                       bias_attr=False)
        self.encoder_norm1 = InstantLayerNormalization(encoder_size)
        self.sep2 = SeparationBlock(input_size=encoder_size, hidden_size=hidden_size)
        self.decoder_conv1 = nn.Conv1D(in_channels=encoder_size, out_channels=win_len, kernel_size=1, stride=1,
                                       bias_attr=False)

    def forward(self, mag, phase, in_states1, in_states2):
        mask1, out_states1 = self.sep1(mag, in_states1)
        est_mag = mag * mask1

        s1_stft = est_mag * (paddle.cos(phase) + 1j * paddle.sin(phase))
        y1 = paddle.fft.irfft(s1_stft)
        y1 = paddle.transpose(y1, [0, 2, 1])

        encoded_f = self.encoder_conv1(y1)
        encoded_f = paddle.transpose(encoded_f, [0, 2, 1])
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2, out_states2 = self.sep2(encoded_f_norm, in_states2)
        encoded_f = mask2 * encoded_f
        estimated = paddle.transpose(encoded_f, [0, 2, 1])
        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states1, out_states2

    def forward_part1(self, mag, in_states1):
        mask1, out_states1 = self.sep1(mag, in_states1)
        est_mag = mag * mask1
        return est_mag, out_states1

    def forward_part2(self, y1, in_states2):
        encoded_f = self.encoder_conv1(y1)
        encoded_f = paddle.transpose(encoded_f, [0, 2, 1])
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2, out_states2 = self.sep2(encoded_f_norm, in_states2)
        encoded_f = mask2 * encoded_f
        estimated = paddle.transpose(encoded_f, [0, 2, 1])
        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states2


def load_weights_for_torch(net, in_pt_path):
    paddle_weights = net.state_dict()
    torch_weights = torch.load(in_pt_path, "cpu")

    for v in paddle_weights.values():
        v.set_value(paddle.zeros_like(v))

    for k, v in torch_weights.items():
        new_v = paddle.to_tensor(v.numpy())
        if "dense.weight" in k:
            new_v = new_v.transpose([1, 0])
        paddle_weights[k].set_value(new_v)
        if 'rnn' in k:
            name_parts: list = k.split(".")
            name_parts.insert(2, "0.cell")
            cell_name = ".".join(name_parts).rstrip("_l0")
            paddle_weights[cell_name].set_value(new_v)

    net.eval()
    ...


def infer():
    paddle.set_grad_enabled(False)
    win_len, win_inc, hidden_size, encoder_size, sr, window = 768, 256, 128, 512, 32000, "none"
    in_pt_path = "data/models/drb_only/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50.pth"
    in_wav_path = "data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_dir = "data/out_data/drb_only"
    out_wav_path = Path(out_dir, f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};pd.wav").as_posix()

    print(f"inference: {in_pt_path}, {in_wav_path}")

    net = DTLN_Paddle(win_len, hidden_size, encoder_size)
    load_weights_for_torch(net, in_pt_path)

    in_state1 = paddle.zeros([4, 1, hidden_size])
    in_state2 = paddle.zeros([4, 1, hidden_size])

    ana_data = np.zeros(win_len)
    if window != "none":
        window = signal.get_window(window, win_len)
    else:
        window = np.ones(win_len)
    output = np.zeros(win_len)
    with wave.Wave_write(out_wav_path) as fp:
        fp.setsampwidth(2)
        fp.setnchannels(1)
        fp.setframerate(sr)
        for idx, data in enumerate(tqdm(AudioUtils.data_generator(in_wav_path, win_inc / sr, sr=sr)), 1):
            ana_data[:-win_inc] = ana_data[win_inc:]
            ana_data[-win_inc:] = data

            data_rfft = np.fft.rfft(ana_data * window)
            mag = np.abs(data_rfft).reshape([1, 1, -1])
            phase = np.angle(data_rfft).reshape([1, 1, -1])

            out, in_state1, in_state2 = net(
                paddle.to_tensor(mag, dtype=paddle.float32),
                paddle.to_tensor(phase, dtype=paddle.float32),
                in_state1,
                in_state2,
            )
            out = out.numpy().reshape(-1)

            output += out
            out = (output[:win_inc] * 32768).astype(np.short)
            fp.writeframes(out.tobytes())
            output[:-win_inc] = output[win_inc:]
            output[-win_inc:] = 0
    ...


def export():
    win_len, hidden_size, encoder_size = 768, 128, 512
    save_prefix = "data/export/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50"
    in_pt_path = "data/models/drb_only/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50.pth"

    net = DTLN_Paddle(win_len, hidden_size, encoder_size)
    load_weights_for_torch(net, in_pt_path)

    mag = paddle.rand([1, 1, win_len // 2 + 1])
    in_states1 = paddle.rand([4, 1, hidden_size])
    net.forward = net.forward_part1
    net_static = paddle.jit.to_static(net)
    paddle.jit.save(net_static, save_prefix + "_p1", input_spec=(mag, in_states1))

    enhanced_data = paddle.rand([1, win_len, 1])
    in_states2 = paddle.rand([4, 1, hidden_size])
    net.forward = net.forward_part2
    net_static = paddle.jit.to_static(net)
    paddle.jit.save(net_static, save_prefix + "_p2", input_spec=(enhanced_data, in_states2))
    ...


if __name__ == '__main__':
    infer()
    # export()
    ...
