import torch
import numpy as np
import torch.nn as nn
from datasetsforecast.m4 import M4, M4Info, M4Evaluation

# Additional utility functions for diagnostics
def log_gradients(model):
    print("\nGradient Statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: Grad Min={param.grad.min().item()}, Max={param.grad.max().item()}, Mean={param.grad.mean().item()}")
        else:
            print(f"{name}: No gradients (possibly unused in computation)")

def log_weights(model):
    print("\nWeight Update Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: Weight Min={param.min().item()}, Max={param.max().item()}, Mean={param.mean().item()}")

def log_activations(x, name):
    print(f"{name}: Min={x.min().item()}, Max={x.max().item()}, Mean={x.mean().item()}")
    return x

# Training function
def train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion):
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

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
                print("Loss is NaN. Investigate inputs and outputs.")
            else:
                print(f"Loss: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

            optimizer.step()
            epoch_loss += loss.item()

            # Log sample outputs and gradients during training
            print(f"Epoch {epoch + 1}, Batch {i // batch_size}: Sample outputs: {outputs[:5].detach().cpu().numpy()}")
            log_gradients(model)

        scheduler.step(loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        log_weights(model)

# Recursive predict with diagnostics
def recursive_predict_xlstm(model, X_input, horizon, device, scalers, test):
    """
    Perform recursive forecasting for multi-step horizons.
    """
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)

    with torch.no_grad():
        for step in range(horizon):
            # print(f"Step {step}: X_current sample (before): {X_current[0, :, 0]}")
            y_pred = model(X_current)  # Expected shape: [batch_size, 1]
            print(f"Step {step}: y_pred sample: {y_pred[0].item()}")

            if step > 0 and (y_pred[0].item() == predictions[-1][0]):
                print(f"Warning: Repeated prediction detected at step {step}: {y_pred[0].item()}")

            y_pred = y_pred.unsqueeze(-1)  # Shape: [batch_size, 1, 1]
            X_current = torch.cat((X_current[:, 1:, :], y_pred), dim=1)

            predictions.append(y_pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)
    predictions_rescaled = []
    for i, (scaler, series_id) in enumerate(zip(scalers.values(), test["unique_id"].unique())):
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)

# Sanity checks for input data
def sanity_check_data(X_train, y_train, X_test):
    print("Sanity Check: Input Data")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_train contains NaN: {torch.isnan(X_train).any().item()}")
    print(f"y_train contains NaN: {torch.isnan(y_train).any().item()}")
    print(f"X_train contains Inf: {torch.isinf(X_train).any().item()}")
    print(f"y_train contains Inf: {torch.isinf(y_train).any().item()}")

# Sanity checks for model initialization
def sanity_check_model(model):
    print("\nSanity Check: Model Initialization")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: Min={param.min().item():.4f}, Max={param.max().item():.4f}, Mean={param.mean().item():.4f}, Requires Grad={param.requires_grad}")

# Add diagnostics to training and evaluation
def train_and_evaluate_xlstm(device, model, X_train, y_train, X_test, scalers, epochs, batch_size, horizon, test, freq, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # Sanity checks
    sanity_check_data(X_train, y_train, X_test)
    sanity_check_model(model)

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

    return y_pred
