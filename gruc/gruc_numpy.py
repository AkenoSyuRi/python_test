import numpy as np
import torch
from torch import nn


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

        return ht, ht


class Conv2dBNRelu:
    def __init__(self, cnn_weight, cnn_bias, bn_weight, bn_bias, stride=(1, 2)):
        assert cnn_weight.ndim == 4
        self.in_channels = cnn_weight.shape[1]
        self.out_channels = cnn_weight.shape[0]
        self.kernel_size = cnn_weight.shape[2:]
        self.stride = stride

        self.cnn_weight = cnn_weight
        self.cnn_bias = cnn_bias
        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        ...

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
                out[:, i, :, j] = np.vdot(self.cnn_weight[i], inputs[0, :, :, idx])
        ...


if __name__ == "__main__":
    # inputs = np.random.randn(1, 1, 2, 300).astype(np.float32)
    inputs = torch.randn(1, 1, 2, 300)

    net1 = nn.Conv2d(1, 2, kernel_size=(2, 3), stride=(1, 2))
    net2 = Conv2dBNRelu(*net1.state_dict().values(), None, None)

    out1 = net1(inputs)
    out2 = net2(inputs)
    ...
