import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import (
    create_rnn_windows,
    create_transformer_windows,
    create_test_windows,
    create_test_windows_transformer,
    recursive_predict_rnn,
    recursive_predict_transformer,
    nn_train
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

# Truncate series if necessary
if max_length is not None:
    train = truncate_series(train, max_length)

# Generate training data windows
X_train_rnn, y_train_rnn, scalers_rnn = create_rnn_windows(train, look_back, horizon)

# Generate test data windows
X_test_rnn, series_ids_rnn = create_test_windows(train, look_back, scalers_rnn)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move data to device
X_train_rnn, y_train_rnn = X_train_rnn.to(device), y_train_rnn.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()

# Training loop parameters
epochs = 10
batch_size = 32


# Function to train and evaluate a model
def train_and_evaluate(model_class, model_name, X_train, y_train, X_test, scalers, series_ids, epochs, batch_size, model_type='RNN'):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    nn_train(model, epochs, X_train, y_train, batch_size, optimizer, criterion, model_type=model_type)

    # Make predictions
    if model_name == 'Transformer':
        y_pred = recursive_predict_transformer(model, X_test, horizon, device, scalers, series_ids)
    else:
        y_pred = recursive_predict_rnn(model, X_test, horizon, device, scalers, series_ids)

    # Reshape predictions to match the expected shape
    y_pred = y_pred.reshape(test.unique_id.nunique(), horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(test.unique_id.nunique(), horizon)
    smape = calculate_smape(y_true, y_pred)
    print(f"\nsMAPE for {model_name}: {smape:.4f}")
    print(f"{model_name} Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred


# Train and evaluate LSTM model
print("Training and Evaluating LSTM Model...")
y_pred_lstm = train_and_evaluate(
    lambda: ComplexLSTM(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    'LSTM',
    X_train_rnn,
    y_train_rnn,
    X_test_rnn,
    scalers_rnn,
    series_ids_rnn,
    epochs,
    batch_size
)


# Train and evaluate RNN model
print("\nTraining and Evaluating RNN Model...")
y_pred_rnn = train_and_evaluate(
    lambda: SimpleRNN(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    'RNN',
    X_train_rnn,
    y_train_rnn,
    X_test_rnn,
    scalers_rnn,
    series_ids_rnn,
    epochs,
    batch_size
)


# Prepare data for the Transformer model
X_train_trans, y_train_trans, scalers_trans = create_transformer_windows(train, look_back, horizon)
X_test_trans, series_ids_trans = create_test_windows_transformer(train, look_back, scalers_trans)

# Move data to device
X_train_trans, y_train_trans = X_train_trans.to(device), y_train_trans.to(device)

# Train and evaluate Transformer model
print("\nTraining and Evaluating Transformer Model...")
y_pred_trans = train_and_evaluate(
    lambda: TimeSeriesTransformer(
        input_size=1,
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        output_size=horizon  # Transformer outputs the full horizon
    ),
    'Transformer',
    X_train_trans,
    y_train_trans,
    X_test_trans,
    scalers_trans,
    series_ids_trans,
    epochs,
    batch_size,
    model_type='Transformer'  # Specify model type
)
