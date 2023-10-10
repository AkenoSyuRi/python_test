import os
import sys
from pathlib import Path

import onnx
import tensorflow as tf
import torch
import torch.nn as nn
from onnx_tf.backend import prepare


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
        :param in_states: [1, 4, 128]
        :return:
        """
        h1_in, c1_in, h2_in, c2_in = torch.split(in_states, in_states.shape[1] // 4, 1)
        # NCNN not support Gather
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        out_states = torch.cat((h1, c1, h2, c2), dim=1)
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        ret = os.system(f"{sys.executable} {__file__} 0")
        ret += os.system(f"{sys.executable} {__file__} 1")
        exit(0)
    elif len(sys.argv) == 2:
        pass
    else:
        raise ValueError(f"invalid argument counts: {len(sys.argv)}, valid value is 2")

    frame_len, frame_hop, hidden_size, encoder_size, load_weights = (384, 128, 128, 256, bool(1))
    in_pt_path = "../data/models/drb_only/DTLN_0906_wSDR_drb_rts_0.05_tam_0.05_none_triple_endto1.0_16k_lr1e4_ep87.pth"
    out_onnx_dir = "../data/export"

    torch.set_grad_enabled(False)
    Path(out_onnx_dir).mkdir(exist_ok=True)

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
            torch.randn(1, 4, hidden_size),
        ),
        (
            torch.randn(1, frame_len, 1),
            torch.randn(1, 4, hidden_size),
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
    idx = int(sys.argv[1])
    assert idx == 0 or idx == 1
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

    out_tf_path = (
            Path(out_onnx_dir) / (Path(in_pt_path).stem + f"_p{idx + 1}")
    ).as_posix()
    onnx_model = onnx.load(out_onnx_sim_path)
    tf_rep = prepare(onnx_model, auto_cast=True)
    tf_rep.export_graph(out_tf_path)

    out_tflite_path = (
            Path(out_onnx_dir) / (Path(in_pt_path).stem + f"_p{idx + 1}.tflite")
    ).as_posix()
    converter = tf.lite.TFLiteConverter.from_saved_model(out_tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8, tf.float16]
    tf_lite_model = converter.convert()
    with open(out_tflite_path, "wb") as f:
        f.write(tf_lite_model)
        print(out_tflite_path)
    ...
