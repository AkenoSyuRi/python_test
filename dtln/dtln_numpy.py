import numpy as np
import torch


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


class Conv1dNoBias:
    def __init__(self, weight):
        """
        weight: (out_channels, in_channels, 1)
        """
        assert weight.ndim == 3

        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]

        self.weight = weight
        ...

    def __call__(self, inputs):
        """
        inputs: (1, in_channels, 1)
        """
        assert inputs.shape[1] == self.in_channels

        output = inputs * self.weight
        output = np.sum(output, axis=1).reshape([1, self.out_channels, -1])
        return output


class LstmCell:

    def __init__(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
        weight_ih: (input_size, hidden_size * 4)
        weight_ih: (hidden_size, hidden_size * 4)
        bias: (hidden_size * 4)
        """
        assert weight_ih_l0.ndim == weight_hh_l0.ndim == 2
        assert bias_ih_l0.ndim == bias_hh_l0.ndim == 1

        self.input_size = weight_ih_l0.shape[0]
        self.hidden_size = weight_hh_l0.shape[0]

        assert weight_ih_l0.shape[-1] == weight_hh_l0.shape[-1] == bias_ih_l0.shape[-1] == bias_hh_l0.shape[-1] == \
               self.hidden_size * 4

        self.w_ii, self.w_if, self.w_ig, self.w_io = np.split(weight_ih_l0, 4, axis=-1)
        self.w_hi, self.w_hf, self.w_hg, self.w_ho = np.split(weight_hh_l0, 4, axis=-1)

        self.b_ii, self.b_if, self.b_ig, self.b_io = np.split(bias_ih_l0, 4, axis=-1)
        self.b_hi, self.b_hf, self.b_hg, self.b_ho = np.split(bias_hh_l0, 4, axis=-1)
        ...

    def __call__(self, inputs, h0, c0):
        """
        inputs: (1, 1, input_size)
        h0, c0: (1, 1, hidden_size)
        """
        it = sigmoid(np.matmul(inputs, self.w_ii) + self.b_ii + np.matmul(h0, self.w_hi) + self.b_hi)
        ft = sigmoid(np.matmul(inputs, self.w_if) + self.b_if + np.matmul(h0, self.w_hf) + self.b_hf)
        gt = np.tanh(np.matmul(inputs, self.w_ig) + self.b_ig + np.matmul(h0, self.w_hg) + self.b_hg)
        ot = sigmoid(np.matmul(inputs, self.w_io) + self.b_io + np.matmul(h0, self.w_ho) + self.b_ho)

        ct = ft * c0 + it * gt
        ht = ot * np.tanh(ct)

        return ht, ht, ct


class InstantLayerNorm:
    def __init__(self, gamma, beta, eps=1e-7):
        assert gamma.ndim == beta.ndim == 3

        self.input_size = gamma.shape[-1]

        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        ...

    def __call__(self, inputs):
        """
        inputs: (1, 1, input_size)
        """
        # calculate mean of each frame
        mean = np.mean(inputs, axis=-1, keepdims=True)
        sub = inputs - mean
        # calculate variance of each frame
        variance = np.mean(np.square(sub), axis=-1, keepdims=True)
        # calculate standard deviation
        std = np.sqrt(variance + self.eps)
        # normalize each frame independently
        outputs = sub / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class SeparationBlock:
    def __init__(self, state_dict, sep_str):
        assert sep_str == 'sep1' or sep_str == 'sep2'

        self.rnn1 = LstmCell(
            state_dict[f'{sep_str}.rnn1.weight_ih_l0'].numpy().transpose(),
            state_dict[f'{sep_str}.rnn1.weight_hh_l0'].numpy().transpose(),
            state_dict[f'{sep_str}.rnn1.bias_ih_l0'].numpy(),
            state_dict[f'{sep_str}.rnn1.bias_hh_l0'].numpy(),
        )

        self.rnn2 = LstmCell(
            state_dict[f'{sep_str}.rnn2.weight_ih_l0'].numpy().transpose(),
            state_dict[f'{sep_str}.rnn2.weight_hh_l0'].numpy().transpose(),
            state_dict[f'{sep_str}.rnn2.bias_ih_l0'].numpy(),
            state_dict[f'{sep_str}.rnn2.bias_hh_l0'].numpy(),
        )

        self.dense = FullyConnected(
            state_dict[f'{sep_str}.dense.weight'].numpy().transpose(),
            state_dict[f'{sep_str}.dense.bias'].numpy(),
        )
        ...

    def __call__(self, x, h1_in, c1_in, h2_in, c2_in):
        """

        :param x:  [N, T, input_size]
        :param in_states: [1, 1, 128, 4]
        :return:
        """
        # h1_in, c1_in = in_states[:, :, :, 0], in_states[:, :, :, 1]
        # h2_in, c2_in = in_states[:, :, :, 2], in_states[:, :, :, 3]

        # NCNN not support Gather
        x1, h1, c1 = self.rnn1(x, h1_in, c1_in)
        x2, h2, c2 = self.rnn2(x1, h2_in, c2_in)

        mask = self.dense(x2)
        mask = sigmoid(mask)

        out_states = np.concatenate((h1, c1, h2, c2), axis=0)
        return mask, out_states


class DTLN_numpy:
    def __init__(self, state_dict):
        self.sep1 = SeparationBlock(state_dict, 'sep1')

        self.encoder_conv1 = Conv1dNoBias(state_dict["encoder_conv1.weight"].numpy())
        self.encoder_norm1 = InstantLayerNorm(
            state_dict["encoder_norm1.gamma"].numpy(),
            state_dict["encoder_norm1.beta"].numpy()
        )
        self.sep2 = SeparationBlock(state_dict, 'sep2')
        self.decoder_conv1 = Conv1dNoBias(state_dict["decoder_conv1.weight"].numpy())

    def __call__(self, mag, phase, sep1_states, sep2_states):
        sep1_h1_in, sep1_c1_in, sep1_h2_in, sep1_c2_in = sep1_states
        mask1, out_states1 = self.sep1(mag, sep1_h1_in, sep1_c1_in, sep1_h2_in, sep1_c2_in)
        estimated_mag = mag * mask1

        s1_stft = estimated_mag * np.exp((1j * phase))
        y1 = np.fft.irfft(s1_stft)
        y1 = y1.transpose([0, 2, 1])

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.transpose([0, 2, 1])
        encoded_f_norm = self.encoder_norm1(encoded_f)
        sep2_h1_in, sep2_c1_in, sep2_h2_in, sep2_c2_in = sep2_states
        mask2, out_states2 = self.sep2(encoded_f_norm, sep2_h1_in, sep2_c1_in, sep2_h2_in, sep2_c2_in)

        encoded_f = mask2 * encoded_f
        estimated = encoded_f.transpose([0, 2, 1])

        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states1, out_states2


if __name__ == '__main__':
    from dtln_inference_frame_wise import get_dtln_network

    in_pt_path = r"data/models/dtln_ns_d20230717_dnsdrb_ep91.pth"
    frame_len, frame_hop, hidden_size, encoder_size = 1024, 512, 128, 512
    state_dict = torch.load(in_pt_path, 'cpu')
    np.random.seed(42)
    torch.manual_seed(24)

    net_np = DTLN_numpy(state_dict)
    net_fw = get_dtln_network(in_pt_path, frame_len, frame_hop, hidden_size, encoder_size)

    mag = np.random.randn(1, 1, frame_len // 2 + 1)
    phase = np.random.randn(1, 1, frame_len // 2 + 1)

    in_state1 = torch.randn(2, 1, hidden_size, 2)
    in_state2 = torch.randn(2, 1, hidden_size, 2)
    h_00, c_00, h_01, c_01 = [
        in_state1[None, 0, ..., 0].numpy(),
        in_state1[None, 0, ..., 1].numpy(),
        in_state1[None, 1, ..., 0].numpy(),
        in_state1[None, 1, ..., 1].numpy(),
    ]
    h_10, c_10, h_11, c_11 = [
        in_state2[None, 0, ..., 0].numpy(),
        in_state2[None, 0, ..., 1].numpy(),
        in_state2[None, 1, ..., 0].numpy(),
        in_state2[None, 1, ..., 1].numpy(),
    ]

    out2, out2_states1, out2_states2 = net_fw(torch.FloatTensor(mag), torch.FloatTensor(phase), in_state1, in_state2)

    out2, out2_states1, out2_states2 = out2.numpy(), out2_states1.numpy(), out2_states2.numpy()
    out1, out1_states1, out1_states2 = net_np(mag, phase, (h_00, c_00, h_01, c_01), (h_10, c_10, h_11, c_11))

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    ...
