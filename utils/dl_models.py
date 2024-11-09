import torch
import torch.nn as nn


# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.3, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.rnn(x)
        h_last = h[:, -1, :]  # Last time step output
        out = self.fc(h_last)
        return out


# Define a more complex LSTM model
class ComplexLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, dropout=0.3, output_size=1):
        super(ComplexLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]  # Get the output of the last time step
        out = self.fc(h_last)
        return out
