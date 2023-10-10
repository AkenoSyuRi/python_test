import os
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from audio_utils import AudioUtils
from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
from onnxruntime import quantization
from onnxruntime.quantization import QuantType
from scipy import signal
from tqdm import tqdm


class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.rnn2 = nn.LSTM(
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
        :param in_states: [1, 1, 128, 4]
        :return:
        """
        h1_in, c1_in, h2_in, c2_in = torch.split(in_states, in_states.shape[0] // 4)
        # NCNN not support Gather
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        out_states = torch.cat((h1, c1, h2, c2), dim=0)
        return mask, out_states
        # return mask, (h1, c1, h2, c2)


class Pytorch_DTLN_P1_stateful(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256, hidden_size=128):
        super(Pytorch_DTLN_P1_stateful, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        self.sep1 = SeperationBlock_Stateful(
            input_size=(frame_len // 2 + 1), hidden_size=hidden_size, dropout=0.25
        )

    def forward(self, mag, in_states):
        """

        :param mag:  [1, 1, 257]
        :param in_state1: [1, 1, 128, 4]
        :return:
        """
        # assert in_state1.shape[0] == 1
        # assert in_state1.shape[-1] == 4
        # N, T, hidden_size
        mask, out_states = self.sep1(mag, in_states)
        estimated_mag = mask * mag

        return estimated_mag, out_states


class Pytorch_InstantLayerNormalization_NCNN_Compat(nn.Module):
    def __init__(self, channels):
        """
        Constructor
        """
        super(Pytorch_InstantLayerNormalization_NCNN_Compat, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs)
        sub = inputs - mean
        # calculate variance of each frame
        variance = torch.mean(torch.square(sub))
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = sub / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class Pytorch_DTLN_P2_stateful(nn.Module):
    def __init__(self, frame_len=512, hidden_size=128, encoder_size=256):
        super(Pytorch_DTLN_P2_stateful, self).__init__()
        self.frame_len = frame_len
        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(
            in_channels=frame_len,
            out_channels=self.encoder_size,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = Pytorch_InstantLayerNormalization_NCNN_Compat(
            channels=self.encoder_size
        )

        self.sep2 = SeperationBlock_Stateful(
            input_size=self.encoder_size, hidden_size=hidden_size, dropout=0.25
        )

        ## TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(
            in_channels=self.encoder_size,
            out_channels=frame_len,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, y1, in_states):
        """
        :param y1: [1, framelen, 1]
        :param in_state2:  [1, 1, 128, 4]
        :return:
        """
        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2, out_states = self.sep2(encoded_f_norm, in_states)

        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states


def onnx2ort(in_onnx_path):
    cmd = f"{sys.executable} -m onnxruntime.tools.convert_onnx_models_to_ort {in_onnx_path}"
    os.system(cmd)
    ...


def int8_quant(in_onnx_path, nodes_to_quantize, weight_type=QuantType.QInt8):
    out_onnx_path = os.path.splitext(in_onnx_path)[0] + "_int8.onnx"
    quantization.quantize_dynamic(
        Path(in_onnx_path),
        Path(out_onnx_path),
        op_types_to_quantize=[],
        nodes_to_quantize=nodes_to_quantize,
        weight_type=weight_type
    )
    onnx2ort(out_onnx_path)
    ...


def fp16_convert(in_onnx_path):
    onnx_model = load_model(in_onnx_path)
    trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    out_fp16_path = os.path.splitext(in_onnx_path)[0] + "_fp16.onnx"
    save_model(trans_model, out_fp16_path)

    onnx2ort(out_fp16_path)
    ...


def infer():
    win_len, win_inc, hidden_size, encoder_size, sr, window, data_type = 384, 128, 128, 256, 16000, "none", np.float32
    in_onnx1_path = "../data/export/DTLN_0906_wSDR_drb_rts_0.05_tam_0.05_none_triple_endto1.0_16k_lr1e4_ep87_p1_sim_fp16.onnx"
    in_onnx2_path = "../data/export/DTLN_0906_wSDR_drb_rts_0.05_tam_0.05_none_triple_endto1.0_16k_lr1e4_ep87_p2_sim_fp16.onnx"
    in_wav_path = "../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_dir = "../data/out_data/drb_only"
    out_wav_path = Path(out_dir, f"{Path(in_wav_path).stem};{Path(in_onnx1_path).stem};ort;fp16.wav").as_posix()

    sess1 = ort.InferenceSession(in_onnx1_path)
    sess2 = ort.InferenceSession(in_onnx2_path)

    in_state1 = np.zeros([4, 1, hidden_size], dtype=data_type)
    in_state2 = np.zeros([4, 1, hidden_size], dtype=data_type)

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
            mag = np.abs(data_rfft).reshape([1, 1, -1]).astype(data_type)
            phase = np.angle(data_rfft).reshape([1, 1, -1]).astype(data_type)

            est_mag, in_state1 = sess1.run(None, {"mag": mag, "in_states": in_state1})

            s1_stft = est_mag * np.exp(1j * phase)
            y1 = np.fft.irfft(s1_stft)
            y1 = y1.transpose(0, 2, 1).astype(data_type)

            out, in_state2 = sess2.run(None, {"input": y1, "in_states": in_state2})
            out = out.reshape(-1)

            output += out
            out = (output[:win_inc] * 32768).astype(np.short)
            fp.writeframes(out.tobytes())
            output[:-win_inc] = output[win_inc:]
            output[-win_inc:] = 0
    ...


def export():
    frame_len, frame_hop, hidden_size, encoder_size, load_weights = 768, 256, 128, 512, bool(1)
    in_pt_path = r"F:\Test\1.audio_test\2.in_models\drb\DTLN_0828_wSDR_drb_only_rts_0.05_tam_0.07_none_triple_ep62.pth"
    out_onnx_dir = Path(r"F:\Test\1.audio_test\4.out_models\onnx", Path(in_pt_path).stem)

    torch.set_grad_enabled(False)
    Path(out_onnx_dir).mkdir(parents=True, exist_ok=True)

    model_list = [
        Pytorch_DTLN_P1_stateful(
            frame_len=frame_len, frame_hop=frame_hop, hidden_size=hidden_size
        ),
        Pytorch_DTLN_P2_stateful(
            frame_len=frame_len, hidden_size=hidden_size, encoder_size=encoder_size
        ),
    ]

    net_input_datas = [
        (
            torch.randn(1, 1, frame_len // 2 + 1),
            torch.randn(4, 1, hidden_size),
        ),
        (
            torch.randn(1, frame_len, 1),
            torch.randn(4, 1, hidden_size),
        ),
    ]
    net_input_names = [
        ("mag", "in_states"),
        ("input", "in_states"),
    ]
    net_output_names = [
        ("est_mag", "out_states"),
        ("output", "out_states"),
    ]

    print("=========== exporting onnx model ===========")
    for idx in range(2):
        if load_weights:
            model_list[idx].load_state_dict(torch.load(in_pt_path, "cpu"), strict=False)
        model_list[idx].eval()

        out_onnx_path = (
                Path(out_onnx_dir) / (Path(in_pt_path).stem + f"_p{idx + 1}.onnx")
        ).as_posix()

        torch.onnx.export(
            model_list[idx],
            net_input_datas[idx],
            out_onnx_path,
            input_names=net_input_names[idx],
            output_names=net_output_names[idx],
            opset_version=12,
        )

        out_onnx_sim_path = (
                Path(out_onnx_path).parent / (Path(out_onnx_path).stem + "_sim.onnx")
        ).as_posix()
        # shutil.copyfile(out_onnx_path, out_onnx_sim_path)
        os.system(f"{sys.executable} -m onnxsim {out_onnx_path} {out_onnx_sim_path}")

        fp16_convert(out_onnx_sim_path)
        int8_quant(out_onnx_sim_path, (
            [
                # "/sep1/rnn1/LSTM",
                # "/sep1/rnn2/LSTM",
                "/sep1/dense/MatMul",
            ], [
                # "/encoder_conv1/Conv",
                # "/sep2/rnn1/LSTM",
                # "/sep2/rnn2/LSTM",
                "/sep2/dense/MatMul",
                # "/decoder_conv1/Conv",
            ]
        )[idx], QuantType.QInt8)


if __name__ == "__main__":
    export()
    # infer()
    ...
