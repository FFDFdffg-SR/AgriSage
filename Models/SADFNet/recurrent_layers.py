import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)
