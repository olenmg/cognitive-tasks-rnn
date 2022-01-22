import torch
import torch.nn as nn


## TODO: Initialize strategy? ... W_ih: Unknown / W_hh: I / 2 (to be updated)
class CTRNN(nn.Module):
    """ Continuous-time RNN """
    def __init__(self, input_size, hidden_size, dt=20, tau=100, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = dt / tau

        self.cell = nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            nonlinearity='relu'
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
        out = (1 - self.alpha) * hidden + self.alpha * self.cell(input, hidden)
        
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

        out = torch.stack([self.cell(input_step, hidden) for input_step in input], dim=0)

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