import torch
import numpy as np
import torch.nn as nn
from datasetsforecast.m4 import M4, M4Info, M4Evaluation


# Training function
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
            outputs = model(batch_X).squeeze(-1)  # Expected shape: [batch_size]

            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                print(f"NaN loss encountered in epoch {epoch}, batch {i // batch_size}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")


def recursive_predict_xlstm(model, X_input, horizon, device, scalers, test):
    """
    Perform recursive forecasting for multi-step horizons.
    """
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)

    with torch.no_grad():
        for step in range(horizon):
            # Debug X_current before prediction
            print(f"Step {step}: X_current sample (before): {X_current[0, :, 0]}")

            # Model prediction
            y_pred = model(X_current)  # Expected shape: [batch_size, 1]
            print(f"Step {step}: y_pred sample: {y_pred[0].item()}")  # Debug y_pred

            # Detect repeated predictions
            if step > 0 and (y_pred[0].item() == predictions[-1][0]):
                print(f"Warning: Repeated prediction detected at step {step}: {y_pred[0].item()}")

            # Ensure correct dimensions for update
            y_pred = y_pred.unsqueeze(-1)  # Shape: [batch_size, 1, 1]
            X_current = torch.cat((X_current[:, 1:, :], y_pred), dim=1)  # Update input sequence

            # Debug X_current after update
            # print(f"Step {step}: X_current sample (after): {X_current[0, :, 0]}")

            predictions.append(y_pred.cpu().numpy())

    # Concatenate predictions along horizon
    predictions = np.concatenate(predictions, axis=1)  # Shape: [batch_size, horizon]

    # Reverse scaling for each series
    predictions_rescaled = []
    for i, (scaler, series_id) in enumerate(zip(scalers.values(), test["unique_id"].unique())):
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


def train_and_evaluate_xlstm(device, model, X_train, y_train, X_test, scalers, epochs, batch_size, horizon, test, freq,
                             criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Train the model
    train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion)

    # Predict
    y_pred = recursive_predict_xlstm(model, X_test, horizon, device, scalers, test)

    # Reshape predictions for evaluation
    num_series = test["unique_id"].nunique()
    y_pred = y_pred.reshape(num_series, horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(num_series, horizon)
    from utils.ml_models import calculate_smape
    calculate_smape(y_true, y_pred)
    # print(f"xLSTM Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred
