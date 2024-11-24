import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import (
    create_train_windows,
    create_test_windows
)
from utils.dl_models import xLSTMTimeSeriesModel
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from utils.preprocess_xlstm import train_and_evaluate_xlstm

# Choose the frequency
freq = 'Hourly'  # or 'Daily'

train, test = train_test_split(freq)

num_series = 1
filtered_series = train["unique_id"].unique()[:num_series]
train = train[train["unique_id"].isin(filtered_series)]
test = test[test["unique_id"].isin(filtered_series)]

# Set parameters based on frequency
if freq == 'Daily':
    look_back = 30  # Number of previous time steps for input
    horizon = 14  # Forecast horizon
    max_length = 200  # For truncating series
    lstm_hidden_size = 50
elif freq == 'Hourly':
    look_back = 120
    horizon = 48
    max_length = None  # Do not truncate series
    lstm_hidden_size = 100
else:
    raise ValueError("Unsupported frequency. Choose 'Daily' or 'Hourly'.")


# Create train and test windows
X_train_xlstm, y_train_xlstm, scalers_xlstm = create_train_windows(train, look_back, horizon)
X_test_xlstm = create_test_windows(train, look_back, scalers_xlstm)

# Prepare data tensors
X_train_xlstm = torch.tensor(X_train_xlstm, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
y_train_xlstm = torch.tensor(y_train_xlstm, dtype=torch.float32)
X_test_xlstm = torch.tensor(X_test_xlstm, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension

# Set up device
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# Move data to device
X_train_xlstm = X_train_xlstm.to(device)
y_train_xlstm = y_train_xlstm.to(device)
X_test_xlstm = X_test_xlstm.to(device)

# Define embedding dimension
embedding_dim = X_train_xlstm.shape[2]  # Should be 1 for univariate time series

# Define xLSTM configuration
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=1,  # Set to 1
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla" if torch.cuda.is_available() else "vanilla",
            num_heads=1,  # Set to 1
            conv1d_kernel_size=4,
            bias_init="constant",  #powerlaw_blockdependent
        ),
        feedforward=FeedForwardConfig(
            proj_factor=1.3,
            act_fn="gelu",
        ),
    ),
    context_length=look_back+1,
    num_blocks=7,
    embedding_dim=embedding_dim,
    slstm_at=[1, 2],
)

print("Checking for NaNs and Infs in data:")
print("X_train_xlstm contains NaN:", torch.isnan(X_train_xlstm).any())
print("X_train_xlstm contains Inf:", torch.isinf(X_train_xlstm).any())
print("y_train_xlstm contains NaN:", torch.isnan(y_train_xlstm).any())
print("y_train_xlstm contains Inf:", torch.isinf(y_train_xlstm).any())


# Create xLSTM stack
xlstm_stack = xLSTMBlockStack(cfg).to(device)

# Instantiate the model
output_size = 1
model = xLSTMTimeSeriesModel(xlstm_stack, output_size, cfg).to(device)

# Pass data through individual blocks
x = X_train_xlstm[:10].clone().detach()
for i, block in enumerate(xlstm_stack.blocks):
    x = block(x)
    print(f"After block {i}, NaN in output: {torch.isnan(x).any()}, shape: {x.shape}")
    if torch.isnan(x).any():
        break  # Stop if NaNs are detected

from torch.nn import LayerNorm

x = X_train_xlstm[:10].clone().detach()
layer_norm = LayerNorm(normalized_shape=x.shape[-1]).to(device)
x = layer_norm(x)
print(f"After LayerNorm, NaN in output: {torch.isnan(x).any()}")


class DummyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x[:, -1, :])  # Use the last time step

# model = DummyModel(input_size=1, output_size=1).to(device)

# Training parameters
epochs = 10
batch_size = 32
criterion = nn.MSELoss()

X_dummy = torch.rand(100, 120, 1)  # Random input
y_dummy = torch.rand(100)  # Random target



# Train and evaluate xLSTM model
print("\nTraining and Evaluating xLSTM Model...")
y_pred_xlstm = train_and_evaluate_xlstm(
    device,
    model,
    X_train_xlstm,
    y_train_xlstm,
    X_test_xlstm,
    scalers_xlstm,
    10,
    batch_size,
    horizon,
    test,
    freq,
    criterion
)

y_true = test['y'].values.reshape(1, horizon)

calculate_smape(y_true, y_pred_xlstm)
