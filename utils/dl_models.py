import torch
import torch.nn as nn
import math

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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout, output_size):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Set batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_size]
        src = self.input_projection(src)  # Shape: [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)  # Shape: [batch_size, seq_len, d_model]
        output = self.decoder(output)  # Shape: [batch_size, seq_len, output_size]
        output = output[:, -1, :]  # Use the last time step
        return output  # Shape: [batch_size, output_size]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)
