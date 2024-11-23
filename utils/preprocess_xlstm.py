import torch
import numpy as np
import torch.nn as nn
from datasetsforecast.m4 import M4, M4Info, M4Evaluation


# Training and evaluation functions
def train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)  # Shape: [batch_size]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


def recursive_predict_xlstm(model, X_input, horizon, device, scalers, series_ids):
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)

    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current).unsqueeze(-1)  # Predict one step ahead
            predictions.append(y_pred.cpu().numpy())
            X_current = torch.cat((X_current[:, 1:], y_pred), dim=1)  # Update sequence

    predictions = np.hstack(predictions)  # Shape: [num_series, horizon]

    # Reverse scaling for each series
    predictions_rescaled = []
    for i, series_id in enumerate(series_ids):
        scaler = scalers[series_id]
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)


def evaluate_xlstm(model, X_test, scalers, series_ids):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.cpu().numpy()

    # Reverse scaling for each series
    predictions_rescaled = []
    for i, series_id in enumerate(series_ids):
        scaler = scalers[series_id]
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)


def train_and_evaluate_xlstm(device, model, X_train, y_train, X_test, scalers, series_ids, epochs, batch_size, horizon,
                             test, freq):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion)

    # Evaluate the model using recursive forecasting
    y_pred = recursive_predict_xlstm(model, X_test, horizon, device, scalers, series_ids)

    # Reshape predictions to match the expected shape
    y_pred = y_pred.reshape(test.unique_id.nunique(), horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(test.unique_id.nunique(), horizon)
    print(f"xLSTM Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred