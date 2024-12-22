import torch
import torch.nn as nn
import math


# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.3, output_size=69, direct=False):
        super(SimpleRNN, self).__init__()
        self.direct = direct
        self.output_size = output_size if self.direct else 1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.rnn(x)
        h_last = h[:, -1, :]  # Last time step output
        out = self.fc(h_last)
        return out


# Define a more complex LSTM model
class ComplexLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, dropout=0.3, output_size=69, direct=False):
        super(ComplexLSTM, self).__init__()
        self.direct = direct
        self.output_size = output_size if self.direct else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]  # Get the output of the last time step
        out = self.fc(h_last)
        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout, output_size=69, direct=False):
        super(TimeSeriesTransformer, self).__init__()
        self.direct = direct
        self.output_size = output_size if self.direct else 1
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_size]
        src = self.input_projection(src)  # Shape: [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)  # Shape: [batch_size, seq_len, d_model]
        output = output[:, -1, :]  # Use the last time step
        output = self.decoder(output)  # Shape: [batch_size, output_size]
        return output.squeeze(-1)  # Shape: [batch_size]


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


class xLSTMTimeSeriesModel(nn.Module):
    def __init__(self, xlstm_stack, output_size, embedding_dim, input_size, direct=False):
        super().__init__()
        self.direct = direct
        self.output_size = output_size if self.direct else 1
        self.input_projection = nn.Linear(input_size, embedding_dim)  # Project input to embedding_dim
        self.xlstm_stack = xlstm_stack
        self.output_layer = nn.Linear(embedding_dim, output_size)

    def forward(self, x):
        x = self.input_projection(x)  # Expand input to embedding_dim
        x = self.xlstm_stack(x)
        x = x[:, -1, :]  # Select last timestep
        x = self.output_layer(x)
        return x

    #
    # def register_hooks(self):
    #     """
    #     Registers backward hooks to monitor gradients in key components.
    #     """
    #     def grad_hook(module_name):
    #         def hook(module, grad_input, grad_output):
    #             print(f"Gradient Hook - {module_name}:")
    #             if grad_input and grad_input[0] is not None:
    #                 print(f"  grad_input[0]: {grad_input[0].norm().item():.4f}")
    #             if grad_output and grad_output[0] is not None:
    #                 print(f"  grad_output[0]: {grad_output[0].norm().item():.4f}")
    #         return hook
    #
    #     # Register hooks on the main components of the model
    #     self.input_layer_norm.register_backward_hook(grad_hook("input_layer_norm"))
    #     self.output_layer.register_backward_hook(grad_hook("output_layer"))
    #
    #     # Register hooks for each block in the xLSTMStack
    #     for i, block in enumerate(self.xlstm_stack.blocks):
    #         block.register_backward_hook(grad_hook(f"xlstm_stack.blocks[{i}]"))
    #         if hasattr(block, 'xlstm') and block.xlstm is not None:
    #             block.xlstm.register_backward_hook(grad_hook(f"xlstm_stack.blocks[{i}].xlstm"))
    #         if hasattr(block, 'ffn') and block.ffn is not None:
    #             block.ffn.register_backward_hook(grad_hook(f"xlstm_stack.blocks[{i}].ffn"))
    #
    #     # Optionally, hooks for post-blocks normalization
    #     if hasattr(self.xlstm_stack, "post_blocks_norm"):
    #         self.xlstm_stack.post_blocks_norm.register_backward_hook(grad_hook("post_blocks_norm"))
