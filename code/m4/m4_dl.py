import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from utils.m4_preprocess_dl import (
    create_train_windows,
    create_test_windows,
    train_and_evaluate
)
from utils.dl_models import ComplexLSTM, SimpleRNN, TimeSeriesTransformer

# Choose the frequency
freq = 'Hourly'  # or 'Daily'

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

# Load data
train, test = train_test_split(freq)
if max_length:
    train = truncate_series(train, max_length)

# Create unified training and test datasets
X_train, y_train, scalers = create_train_windows(train, look_back, horizon)
X_test = create_test_windows(train, look_back, scalers)

# Add feature dimension and send data to device (unified for all models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = X_train.unsqueeze(-1).to(device)  # Shape: [samples, look_back, 1]
X_test = X_test.unsqueeze(-1).to(device)    # Shape: [num_series, look_back, 1]
y_train = y_train.to(device)

# Define common parameters
criterion = nn.MSELoss()
epochs, batch_size = 10, 32

# Train and evaluate LSTM model
print("Training and Evaluating LSTM Model...")
y_pred_lstm = train_and_evaluate(
    device,
    lambda: ComplexLSTM(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    'LSTM',
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    criterion,
    horizon,
    test,
    freq
)

# Train and evaluate RNN model
print("\nTraining and Evaluating RNN Model...")
y_pred_rnn = train_and_evaluate(
    device,
    lambda: SimpleRNN(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    'RNN',
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    criterion,
    horizon,
    test,
    freq
)

# Train and evaluate Transformer model
print("\nTraining and Evaluating Transformer Model...")
y_pred_trans = train_and_evaluate(
    device,
    lambda: TimeSeriesTransformer(
        input_size=1,
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        output_size=1
    ),
    'Transformer',
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    criterion,
    horizon,
    test,
    freq
)
