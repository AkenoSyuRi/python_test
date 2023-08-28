import os
from pathlib import Path

import torch
import torch.nn as nn
from file_utils import FileUtils


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

    def forward(self, x, h1_in, c1_in, h2_in, c2_in):
        """

        :param x:  [N, T, input_size]
        :param in_states: [1, 1, 128, 4]
        :return:
        """
        # h1_in, c1_in = in_states[:, :, :, 0], in_states[:, :, :, 1]
        # h2_in, c2_in = in_states[:, :, :, 2], in_states[:, :, :, 3]

        # NCNN not support Gather
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        out_states = torch.cat((h1, c1, h2, c2), dim=0)
        return mask, out_states


class Pytorch_DTLN_P1_stateful(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256, hidden_size=128):
        super(Pytorch_DTLN_P1_stateful, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        self.sep1 = SeperationBlock_Stateful(
            input_size=(frame_len // 2 + 1), hidden_size=hidden_size, dropout=0.25
        )

        self.fix_ncnn_err = True

    def forward(self, mag, h1_in, c1_in, h2_in, c2_in):
        """

        :param mag:  [1, 1, 257]
        :param in_state1: [1, 1, 128, 4]
        :return:
        """
        # assert in_state1.shape[0] == 1
        # assert in_state1.shape[-1] == 4
        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, h1_in, c1_in, h2_in, c2_in)

        if self.fix_ncnn_err:
            mask = mask.reshape(-1)
            mag = mag.reshape(-1)

            estimated_mag = mask * mag
            estimated_mag = estimated_mag.view(1, 1, -1)
        else:
            # NCNN BinaryOP result err
            estimated_mag = mask * mag

        return estimated_mag, out_state1


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

        self.fix_ncnn_err = True

    def forward(self, y1, h1_in, c1_in, h2_in, c2_in):
        """
        :param y1: [1, framelen, 1]
        :param in_state2:  [1, 1, 128, 4]
        :return:
        """
        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2, out_state2 = self.sep2(encoded_f_norm, h1_in, c1_in, h2_in, c2_in)

        if self.fix_ncnn_err:
            mask_2 = mask_2.reshape(-1)
            encoded_f = encoded_f.reshape(-1)

            estimated = mask_2 * encoded_f
            estimated = estimated.view(1, 1, self.encoder_size)
        else:
            # NCNN BinaryOP result err
            estimated = mask_2 * encoded_f

        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_state2


if __name__ == "__main__":
    pnnx_exe = r"F:\Tools\pnnx-20230816-windows\pnnx.exe"
    # opt_exe = r"F:\Projects\CLionProjects\third_party\ncnn\build_vs2022_release\install\bin\ncnnoptimize.exe"
    frame_len, frame_hop, hidden_size, encoder_size = 1024, 512, 128, 512
    in_pt_path = "../data/models/drb_only/DTLN_0827_wSDR_drb_only_pre75ms_none_ep41.pth"
    out_onnx_dir = "../data/export"

    torch.set_grad_enabled(False)
    FileUtils.ensure_dir(out_onnx_dir)

    model1 = Pytorch_DTLN_P1_stateful(
        frame_len=frame_len, frame_hop=frame_hop, hidden_size=hidden_size
    )
    model2 = Pytorch_DTLN_P2_stateful(
        frame_len=frame_len, hidden_size=hidden_size, encoder_size=encoder_size
    )
    model1.load_state_dict(torch.load(in_pt_path, "cpu"), strict=False)
    model2.load_state_dict(torch.load(in_pt_path, "cpu"), strict=False)
    model1.eval()
    model2.eval()

    input1 = torch.randn(1, 1, frame_len // 2 + 1)
    input2 = torch.randn(1, frame_len, 1)

    h_00, c_00, h_01, c_01 = [
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
    ]
    h_10, c_10, h_11, c_11 = [
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
        torch.randn(1, 1, hidden_size),
    ]

    print("=========== exporting onnx model ===========")
    out_ncnn_paths = [
        Path(out_onnx_dir) / (Path(in_pt_path).stem + f"_p{i+1}.pt") for i in range(2)
    ]
    model1 = torch.jit.trace(model1, (input1, h_00, c_00, h_01, c_01))
    model2 = torch.jit.trace(model2, (input2, h_10, c_10, h_11, c_11))
    model1.save(out_ncnn_paths[0])
    model2.save(out_ncnn_paths[1])

    def get_shapes(*inputs):
        return ",".join(map(lambda x: str(list(x.shape)), inputs)).replace(" ", "")

    cmds = (
        f"{pnnx_exe} {out_ncnn_paths[0]} inputshape={get_shapes(input1, h_00, c_00, h_01, c_01)}",
        f"{pnnx_exe} {out_ncnn_paths[1]} inputshape={get_shapes(input2, h_10, c_10, h_11, c_11)}",
    )
    print("####### exporting ncnn model ######")
    for i in range(2):
        print(cmds[i])
        os.system(cmds[i])

    cwd = Path.cwd().absolute()
    os.remove(f"{cwd}/debug.bin")
    os.remove(f"{cwd}/debug.param")
    os.remove(f"{cwd}/debug2.bin")
    os.remove(f"{cwd}/debug2.param")

    # print("####### begin ncnn model optimization ######")
    # for i in range(2):
    #     parent_name = Path(out_ncnn_paths[i]).parent / Path(out_ncnn_paths[i]).stem
    #     in_ncnn_param = parent_name.as_posix() + ".ncnn.param"
    #     in_ncnn_bin = parent_name.as_posix() + ".ncnn.bin"
    #
    #     out_ncnn_param = parent_name.as_posix() + "_opt.ncnn.param"
    #     out_ncnn_bin = parent_name.as_posix() + "_opt.ncnn.bin"
    #     os.system(f"{opt_exe} {in_ncnn_param} {in_ncnn_bin} {out_ncnn_param} {out_ncnn_bin} 0")
    ...
