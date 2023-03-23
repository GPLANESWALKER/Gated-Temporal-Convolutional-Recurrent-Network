import torch
from torch import nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation=1, padding=0, dropout=0):
        super(TemporalBlock, self).__init__()
        """
        :param n_inputs: int
        :param n_outputs: int
        :param kernel_size: int
        :param stride: int
        :param dilation: int
        :param padding: int
        :param dropout: float, dropout rate
        """
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        self.downsample = None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        """
        :param num_inputs: int,
        :param num_channels: list,
        :param kernel_size: int,
        :param dropout: float, drop_out rate
        """
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GTCRCell(nn.Module):
    """GTCRNCell"""
    def __init__(self, input_size, hidden_size):
        super(GTCRCell, self).__init__()
        self.reset_gate
        self.update_gate
        self.output_gate
        self.sigmoidR = nn.Sigmoid()
        self.sigmoidZ = nn.Sigmoid()
        self.tanhH = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        self.reset_gate.weight.data.normal_(0, 0.01)
        self.update_gate.weight.data.normal_(0, 0.01)
        self.output_gate.weight.data.normal_(0, 0.01)

    def forward(self, x_r, x_u, x_i):
        hidden_marix = torch.zeros(x_i.shape[1], x_i.shape[0])
        return hidden_marix
