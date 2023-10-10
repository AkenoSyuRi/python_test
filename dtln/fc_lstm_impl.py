import torch.nn
from torch import nn


class CustomLSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gate = nn.Linear(input_size + hidden_size, hidden_size * 4)

        # Initialize the weights and biases with zeros
        self._init_weights_with_zeros()
        ...

    def _init_weights_with_zeros(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, xt, ht_1, ct_1):
        assert 3 == xt.ndim == ht_1.ndim == ct_1.ndim
        inputs = torch.cat([xt, ht_1], -1)

        in_gate, forget_gate, cell_gate, out_gate = torch.split(self.gate(inputs), self.hidden_size, -1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        ct = forget_gate * ct_1 + in_gate * cell_gate
        ht = out_gate * torch.tanh(ct)
        return ht, ct

    def set_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        weight = torch.cat((weight_ih_l0, weight_hh_l0), -1)
        bias = bias_ih_l0 + bias_hh_l0

        self.gate.weight.data.copy_(weight)
        self.gate.bias.data.copy_(bias)
        ...


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.ModuleList(
            [
                CustomLSTM_Cell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        ...

    def forward(self, inputs, states):
        # assert inputs.shape == (1, 1, self.input_size)
        # assert states.shape == 2 * (self.num_layers, 1, self.hidden_size)

        h_state_list = torch.split(states[0], 1)
        c_state_list = torch.split(states[1], 1)
        lstm_out, h_states_out, c_states_out = inputs, [], []
        for i in range(self.num_layers):
            lstm_out, c_state = self.lstm[i](lstm_out, h_state_list[i], c_state_list[i])
            h_states_out.append(lstm_out)
            c_states_out.append(c_state)
        ht = torch.cat(h_states_out)
        ct = torch.cat(c_states_out)
        return lstm_out, (ht, ct)


if __name__ == "__main__":
    # torch.manual_seed(1)
    input_size, hidden_size, num_layers = 385, 128, 3

    xt = torch.randn(1, 1, input_size)
    ht_1 = torch.randn(num_layers, 1, hidden_size)
    ct_1 = torch.randn(num_layers, 1, hidden_size)

    lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
    out1, (_11, _12) = lstm1(xt, (ht_1, ct_1))

    lstm2 = CustomLSTM(input_size, hidden_size, num_layers)
    for i in range(num_layers):
        keys = [f"weight_ih_l{i}", f"weight_hh_l{i}", f"bias_ih_l{i}", f"bias_hh_l{i}"]
        lstm2.lstm[i].set_weights(*map(lstm1.state_dict().get, keys))
    out2, (_21, _22) = lstm2(xt, (ht_1, ct_1))

    assert torch.allclose(out1, out2, atol=1e-7)
    ...
