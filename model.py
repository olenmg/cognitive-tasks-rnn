import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_activation_map = {
    'relu': F.relu,
    'tanh': F.tanh,
    'softplus': F.softplus
}


class RNNCellWithNoise(nn.Module):
    """ RNN cell with input bias containing Gaussian noise """
    def __init__(
        self,
        input_size,
        hidden_size,
        alpha,
        nonlinearlity='relu',
        noise_std=None
    ):
        super().__init__()
        if nonlinearlity not in _activation_map:
            raise ValueError("nonlinearlity candidates: relu, tanh, softplus")

        self.hidden_size = hidden_size
        self.alpha = alpha
        self.nonlinearity = _activation_map[nonlinearlity]
        if noise_std is not None:
            self.noise_coef = np.sqrt(2 / alpha * (noise_std ** 2))
        else:
            self.noise_coef = 0.

        self.wih = nn.Linear(input_size, hidden_size)
        self.whh = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.empty(hidden_size))

    def forward(self, input, hidden):
        batch_size = input.size(0)
        out = self.wih(input) + self.whh(hidden) + \
                self.bias + self.noise_coef * torch.normal(mean=0., std=1., size=(self.hidden_size, ))
        out = self.nonlinearity(out)

        return out


## TODO: Initialize strategy? ... W_ih: Unknown / W_hh: I / 2 (to be updated)
class CTRNN(nn.Module):
    """ Continuous-time RNN """
    def __init__(
        self,
        input_size,
        hidden_size,
        dt=20,
        tau=100,
        nonlinearlity='relu',
        noise_std=0,
        device=None
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = dt / tau
        self.noise_coef = np.sqrt(2 / self.alpha * noise_std)

        self.rnn_cell = RNNCellWithNoise(
            input_size=input_size,
            hidden_size=hidden_size,
            alpha=self.alpha,
            nonlinearlity='relu',
            noise_std=noise_std
        )
        self.device = device

    def forward_step(self, input, hidden):
        """
        Args:
            input (torch.Tensor): (batch_size, input_size)
            hidden (torch.Tensor): (batch_size, hidden_size)
        Return:
            out (torch.Tensor): Next hidden state, (batch_size, hidden_size)
        """
        out = (1 - self.alpha) * hidden + self.alpha * self.rnn_cell(input, hidden)

        return out

    def forward(self, input, hidden=None):
        """
        Args:
            input (torch.Tensor): (time_length, batch_size, input_size)
            hidden (torch.Tensor): (time_length, batch_size, input_size)
        Return:
            out (torch.Tensor): Hidden states at each time step, 
                                (time_length, batch_size, hidden_size)
            hidden (torch.Tensor): Last hidden state, (batch_size, hidden_size)
        """
        if hidden is None:
            hidden = torch.zeros(input.size(1), self.hidden_size, device=self.device)

        out = torch.stack([self.rnn_cell(input_step, hidden) for input_step in input], dim=0)

        return out, out[-1]


class CTRNNWithHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.ctrnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        """
        Args:
            input (torch.Tensor): (time_length, batch_size, input_size)
            hidden (torch.Tensor): (time_length, batch_size, input_size)
        Return:
            out (torch.Tensor): Logits, (time_length, batch_size, output_size)
        """
        out, _ = self.ctrnn(input, hidden)
        out = self.fc(out)

        return out
