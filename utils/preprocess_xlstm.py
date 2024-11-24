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
            batch_X = X_train[indices].to(device)
            batch_y = y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)

            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                print("NaN loss encountered! Check your data or model outputs.")
                return  # Exit training loop

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')



def recursive_predict_xlstm(model, X_input, horizon, device, scalers, series_ids):
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)

    with torch.no_grad():
        for step in range(horizon):
            print(f"Step {step}: X_current shape: {X_current.shape}")  # Debugging shape

            # Ensure X_current is 3D before passing to model
            if X_current.dim() != 3:
                X_current = X_current.squeeze(-1)

            y_pred = model(X_current)  # Expected shape: [Batch Size, 1]
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print(f"Invalid prediction at step {step}.")
            print(f"Step {step}: y_pred shape: {y_pred.shape}")  # Debugging shape

            y_pred = y_pred.unsqueeze(-1)  # Shape: [Batch Size, 1, 1]
            X_current = torch.cat((X_current[:, 1:, :], y_pred), dim=1)  # Update sequence

            print(f"Step {step}: X_current shape after concat: {X_current.shape}")

            predictions.append(y_pred.squeeze(-1).cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)  # Shape: [Batch Size, Horizon]

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


def train_and_evaluate_xlstm(device, model, X_train, y_train, X_test, scalers, series_ids, epochs, batch_size, horizon, test, freq):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Train the model
    train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion)

    # Evaluate the model using recursive forecasting
    print(f"X_test_xlstm shape before prediction: {X_test.shape}, Horizon: {horizon}")
    y_pred = recursive_predict_xlstm(model, X_test, horizon, device, scalers, series_ids)

    print(f"Predictions generated: {y_pred.size}")
    print(f"Expected reshape dimensions: ({test.unique_id.nunique()}, {horizon})")

    # Check for alignment
    num_series = test.unique_id.nunique()
    if y_pred.size != num_series * horizon:
        raise ValueError(f"Mismatch: y_pred size {y_pred.size} does not match expected ({num_series} * {horizon})")

    # Reshape predictions
    y_pred = y_pred.reshape(num_series, horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(num_series, horizon)
    from utils.ml_models import calculate_smape
    calculate_smape(y_true, y_pred)
    # print(f"xLSTM Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred
