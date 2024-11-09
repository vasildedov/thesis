import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import create_rnn_windows, create_test_windows, recursive_predict_rnn
from utils.dl_models import ComplexLSTM, SimpleRNN


# Parameters
look_back = 120  # Number of previous time steps for input
horizon = 48  # Forecast horizon

# Load data
train, test = train_test_split('Hourly')

# Generate training data windows
X_train_rnn, y_train_rnn, scalers = create_rnn_windows(train, look_back, horizon)

X_test_rnn = create_test_windows(train, look_back, scalers)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexLSTM(hidden_size=100, num_layers=3, dropout=0.3, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
batch_size = 32
X_train_rnn, y_train_rnn = X_train_rnn.to(device), y_train_rnn.to(device)

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_rnn.size(0))
    epoch_loss = 0
    for i in range(0, X_train_rnn.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_rnn[indices], y_train_rnn[indices]

        optimizer.zero_grad()
        outputs = model(batch_X.unsqueeze(-1)).squeeze(-1)  # Add feature dimension for RNN and adjust output
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


# Make predictions
y_pred_lstm = recursive_predict_rnn(model, X_test_rnn, horizon, device, scalers, test.unique_id.unique())

# Reshape predictions to match the expected shape (414 series, 48-hour horizon)
y_pred_lstm = y_pred_lstm.reshape(test.unique_id.nunique(), horizon)

# Evaluate using sMAPE
y_true = test['y'].values.reshape(test.unique_id.nunique(), horizon)
print(f"sMAPE for LSTM: {calculate_smape(y_true, y_pred_lstm):.4f}")
print("LSTM Model Evaluation:\n", M4Evaluation.evaluate('data', 'Hourly', y_pred_lstm))


# RNN
# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(hidden_size=50, num_layers=2, dropout=0.3, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_rnn.size(0))
    epoch_loss = 0
    for i in range(0, X_train_rnn.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_rnn[indices], y_train_rnn[indices]

        optimizer.zero_grad()
        outputs = model(batch_X.unsqueeze(-1)).squeeze(-1)  # Add feature dimension for RNN and adjust output
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# Make predictions
y_pred_rnn = recursive_predict_rnn(model, X_test_rnn, horizon, device, scalers, test.unique_id.unique())

# Reshape predictions to match the expected shape (414 series, 48-hour horizon)
y_pred_rnn = y_pred_rnn.reshape(test.unique_id.nunique(), horizon)

# Evaluate using sMAPE
y_true = test['y'].values.reshape(test.unique_id.nunique(), horizon)
print(f"sMAPE for RNN: {calculate_smape(y_true, y_pred_rnn):.4f}")
print("RNN Model Evaluation:\n", M4Evaluation.evaluate('data', 'Hourly', y_pred_rnn))
