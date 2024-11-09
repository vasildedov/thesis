import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.ml_preprocess_M4 import train_test_split
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape


# Parameters
look_back = 60  # Number of previous time steps for input
horizon = 48  # Forecast horizon

# Load data
train, test = train_test_split('Hourly')


# Data Preprocessing: Create input/output windows
def create_rnn_windows(df, look_back, horizon):
    X, y = [], []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values
        for i in range(len(series) - look_back - horizon + 1):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back:i + look_back + horizon])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


# Generate training data windows
X_train_rnn, y_train_rnn = create_rnn_windows(train, look_back, horizon)


# Prepare test data: Use the last 60 hours from each series in train
def create_test_windows(df, look_back):
    X_test = []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values[-look_back:]
        X_test.append(series)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    return X_test.unsqueeze(-1)  # Adding feature dimension for RNN


X_test_rnn = create_test_windows(train, look_back)


# Define a simple LSTM model
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=horizon):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]  # Last time step output
        out = self.fc(h_last)
        return out


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN().to(device)
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
        outputs = model(batch_X.unsqueeze(-1))  # Add feature dimension for RNN
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


# Recursive prediction function for the test set
def recursive_predict_rnn(model, X_input, horizon):
    model.eval()
    predictions = []
    X_current = X_input.to(device)
    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current)
            predictions.append(y_pred.cpu().numpy()[:, -1])  # Last step prediction
            X_current = torch.cat((X_current[:, 1:], y_pred[:, -1].unsqueeze(-1).unsqueeze(-1)), dim=1)
    return np.hstack(predictions)


# Make predictions
y_pred_rnn = recursive_predict_rnn(model, X_test_rnn, horizon)

# Reshape predictions to match the expected shape (414 series, 48-hour horizon)
y_pred_rnn = y_pred_rnn.reshape(414, horizon)

# Evaluate using sMAPE
y_true = test['y'].values.reshape(414, horizon)
print(f"sMAPE for RNN: {calculate_smape(y_true, y_pred_rnn):.4f}")
print("RNN Model Evaluation:\n", M4Evaluation.evaluate('data', 'Hourly', y_pred_rnn))