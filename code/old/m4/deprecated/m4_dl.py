import torch
import torch.nn as nn
from datasetsforecast.m4 import M4Evaluation

from utils.preprocess_m4 import train_test_split, truncate_series
from utils.preprocess_dl import create_train_windows, create_test_windows
from utils.train_dl import train_and_predict
from utils.models_dl import ComplexLSTM, SimpleRNN, TimeSeriesTransformer

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
# num_series = 1
# filtered_series = train["unique_id"].unique()[:num_series]
# train = train[train["unique_id"].isin(filtered_series)]
# test = test[test["unique_id"].isin(filtered_series)]

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
y_pred_lstm, duration = train_and_predict(
    device,
    lambda: ComplexLSTM(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    horizon,
    test,
    criterion
)
print(f"LSTM Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred_lstm))
print(f"Training completed in {duration:.2f} seconds")

# Train and evaluate RNN model
print("\nTraining and Evaluating RNN Model...")
y_pred_rnn, duration = train_and_predict(
    device,
    lambda: SimpleRNN(
        input_size=1,
        hidden_size=lstm_hidden_size,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ),
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    horizon,
    test,
    criterion
)
print(f"RNN Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred_rnn))
print(f"Training completed in {duration:.2f} seconds")

# Train and evaluate Transformer model
print("\nTraining and Evaluating Transformer Model...")
y_pred_trans, duration = train_and_predict(
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
    X_train,
    y_train,
    X_test,
    scalers,
    1,
    batch_size,
    horizon,
    test,
    criterion
)
print(f"Transformer Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred_trans))
print(f"Training completed in {duration:.2f} seconds")
