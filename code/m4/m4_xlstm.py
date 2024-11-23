import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import (
    create_rnn_windows,
    create_test_windows
)
from utils.dl_models import ComplexLSTM, SimpleRNN, TimeSeriesTransformer, xLSTMTimeSeriesModel
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

# Prepare data for xLSTM
X_train_xlstm, y_train_xlstm, scalers_xlstm = create_rnn_windows(train, look_back, horizon)
X_test_xlstm, series_ids_xlstm = create_test_windows(train, look_back, scalers_xlstm)

# Convert to tensors and move to device
X_train_xlstm = torch.tensor(X_train_xlstm, dtype=torch.float32)
y_train_xlstm = torch.tensor(y_train_xlstm, dtype=torch.float32)
X_test_xlstm = torch.tensor(X_test_xlstm, dtype=torch.float32)

# Add feature dimension if necessary
if len(X_train_xlstm.shape) == 2:
    X_train_xlstm = X_train_xlstm.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
    X_test_xlstm = X_test_xlstm.unsqueeze(-1)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            backend="vanilla",
            num_heads=1,  # Set to 1
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(
            proj_factor=1.3,
            act_fn="gelu",
        ),
    ),
    context_length=look_back+1,
    num_blocks=7,
    embedding_dim=embedding_dim,
    slstm_at=[1],
)


# Create xLSTM stack
xlstm_stack = xLSTMBlockStack(cfg).to(device)

# Instantiate the model
output_size = 1
model = xLSTMTimeSeriesModel(xlstm_stack, output_size, cfg).to(device)

# Training parameters
epochs = 10
batch_size = 32

# Train and evaluate xLSTM model
print("\nTraining and Evaluating xLSTM Model...")
y_pred_xlstm = train_and_evaluate_xlstm(
    device,
    model,
    X_train_xlstm,
    y_train_xlstm,
    X_test_xlstm,
    scalers_xlstm,
    series_ids_xlstm,
    epochs,
    batch_size,
    horizon,
    test,
    freq
)
