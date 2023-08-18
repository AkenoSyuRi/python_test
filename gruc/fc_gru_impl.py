import torch.nn
from torch import nn


class CustomGRU_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rz_gate = nn.Linear(input_size + hidden_size, hidden_size * 2)
        self.in_gate = nn.Linear(input_size, hidden_size)
        self.hn_gate = nn.Linear(hidden_size, hidden_size)

        # Initialize the weights and biases with zeros
        self._init_weights_with_zeros()
        ...

    def _init_weights_with_zeros(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, xt, ht_1):
        assert 3 == xt.ndim == ht_1.ndim
        inputs = torch.cat([xt, ht_1], -1)

        rz_gate_out = torch.sigmoid(self.rz_gate(inputs))
        # r_gate_out, z_gate_out = torch.split(rz_gate_out, self.hidden_size, -1)  # Unsupported split axis !
        r_gate_out, z_gate_out = rz_gate_out[..., :self.hidden_size], rz_gate_out[..., self.hidden_size:]

        in_out = self.in_gate(xt)
        hn_out = self.hn_gate(ht_1)
        n_gate_out = torch.tanh(in_out + r_gate_out * hn_out)

        ht = (1 - z_gate_out) * n_gate_out + z_gate_out * ht_1
        return ht

    def set_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        slice_index = self.hidden_size * 2
        rz_weight = torch.cat((weight_ih_l0[:slice_index], weight_hh_l0[:slice_index]), -1)
        rz_bias = bias_ih_l0[:slice_index] + bias_hh_l0[:slice_index]

        self.rz_gate.weight.data.copy_(rz_weight)
        self.rz_gate.bias.data.copy_(rz_bias)

        self.in_gate.weight.data.copy_(weight_ih_l0[slice_index:])
        self.in_gate.bias.data.copy_(bias_ih_l0[slice_index:])

        self.hn_gate.weight.data.copy_(weight_hh_l0[slice_index:])
        self.hn_gate.bias.data.copy_(bias_hh_l0[slice_index:])
        ...


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.ModuleList([
            CustomGRU_Cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        ...

    def forward(self, inputs, states):
        # assert inputs.shape == (1, 1, self.input_size)
        # assert states.shape == (self.num_layers, 1, self.hidden_size)

        # state_list = torch.split(states, 1)  # Unsupported split axis !
        state_list = [states[i, None] for i in range(self.num_layers)]
        gru_out, h_states_list = inputs, []
        for i in range(self.num_layers):
            gru_out = self.gru[i](gru_out, state_list[i])
            h_states_list.append(gru_out)
        h_states = torch.cat(h_states_list)
        return gru_out, h_states


if __name__ == '__main__':
    # torch.manual_seed(1)
    input_size, hidden_size, num_layers = 400, 300, 3

    xt = torch.randn(1, 1, input_size)
    ht_1 = torch.randn(num_layers, 1, hidden_size)

    gru1 = nn.GRU(input_size, hidden_size, num_layers)
    out1, _1 = gru1(xt, ht_1)

    gru2 = CustomGRU(input_size, hidden_size, num_layers)
    for i in range(num_layers):
        keys = [f"weight_ih_l{i}", f"weight_hh_l{i}", f"bias_ih_l{i}", f"bias_hh_l{i}"]
        gru2.gru[i].set_weights(*map(gru1.state_dict().get, keys))
    out2, _2 = gru2(xt, ht_1)

    torch.allclose(out1, out2)
    ...
