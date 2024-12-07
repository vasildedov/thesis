import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


def recursive_predict(model, X_input, horizon, device, scalers):
    """
    Perform recursive forecasting for multi-step horizons.

    Args:
        model (torch.nn.Module): Trained model for prediction.
        X_input (torch.Tensor): Input data for prediction (shape: [batch_size, seq_len, input_size]).
        horizon (int): Number of steps to forecast.
        device (torch.device): Device for computation.
        scalers (dict): Scalers for inverse transformation.

    Returns:
        np.ndarray: Rescaled predictions.
    """
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)  # Ensure input is on the correct device

    with torch.no_grad():
        for _ in range(horizon):
            # Model output (assumed to be [batch_size] or [batch_size, 1])
            y_pred = model(X_current)

            if _ > 0 and (y_pred[0].item() == predictions[-1][0]):
                print(f"Warning: Repeated prediction detected at step {_}: {y_pred[0].item()}")

            # Ensure y_pred has shape [batch_size, 1, input_size]
            if y_pred.dim() == 1:  # Flattened output [batch_size]
                y_pred = y_pred.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            elif y_pred.dim() == 2:  # [batch_size, 1]
                y_pred = y_pred.unsqueeze(-1)  # [batch_size, 1, 1]

            # Slide the input window forward
            X_current = torch.cat((X_current[:, 1:, :], y_pred), dim=1)  # Maintain sequence length
            predictions.append(y_pred.cpu().numpy())  # Store predictions

    # Concatenate predictions and reverse scaling
    predictions = np.concatenate(predictions, axis=1)  # Shape: [batch_size, horizon]
    predictions_rescaled = []
    for i, scaler in enumerate(scalers.values()):
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)


def train_model(model, X_train, y_train, batch_size, optimizer, criterion, epochs):
    """
    Train a neural network model.

    Args:
        model (torch.nn.Module): The model to train.
        X_train (torch.Tensor): Training inputs.
        y_train (torch.Tensor): Training targets.
        batch_size (int): Batch size for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()

            # Ensure correct input dimensions for RNN/Transformer
            if batch_X.dim() == 2:  # If the input is [batch_size, seq_len], add the feature dimension
                batch_X = batch_X.unsqueeze(-1)  # Shape becomes [batch_size, seq_len, 1]
            outputs = model(batch_X).squeeze(-1)  # Squeeze to [batch_size]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


def train_and_predict(device, model_class, X_train, y_train, X_test, scalers, epochs,
                       batch_size, criterion, horizon, test):
    """
    Train a model and evaluate its performance.

    Args:
        device (torch.device): Device for computation.
        model_class (class): Class of the model to instantiate.
        model_name (str): Name of the model for display.
        X_train (torch.Tensor): Training inputs.
        y_train (torch.Tensor): Training targets.
        X_test (torch.Tensor): Test inputs.
        scalers (dict): Scalers for inverse transformation.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        criterion (torch.nn.Module): Loss function.
        horizon (int): Forecasting horizon.
        test (pd.DataFrame): Test DataFrame.

    Returns:
        np.ndarray: Predicted values.
    """
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, X_train, y_train, batch_size, optimizer, criterion, epochs)

    # Make predictions using recursive forecasting
    y_pred = recursive_predict(model, X_test, horizon, device, scalers)

    # Reshape predictions to match the expected shape
    num_series = test['unique_id'].nunique()
    y_pred = y_pred.reshape(num_series, horizon)
    return y_pred
