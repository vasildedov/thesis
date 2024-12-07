import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import time


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

def train_model(model, X_train, y_train, batch_size, optimizer, criterion, epochs,
                device=None, clip_grad_norm=None):
    """
    Train a neural network model with support for gradient clipping and dynamic input shape handling.

    Args:
        model (torch.nn.Module): The model to train.
        X_train (torch.Tensor): Training inputs.
        y_train (torch.Tensor): Training targets.
        batch_size (int): Batch size for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        device (torch.device, optional): Device to train on ('cpu' or 'cuda'). Defaults to None (no transfer).
        clip_grad_norm (float, optional): Max norm for gradient clipping. Defaults to None (no clipping).
    """
    if device:
        model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            # Adjust input dimensions for RNNs/Transformers if necessary
            if batch_X.dim() == 2:  # If input is [batch_size, seq_len], add feature dimension
                batch_X = batch_X.unsqueeze(-1)  # Shape becomes [batch_size, seq_len, 1]

            outputs = model(batch_X).squeeze(-1)  # Ensure output matches expected shape [batch_size]

            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                print("Loss is NaN. Investigate inputs and outputs.")
                return

            loss.backward()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()
            # Log sample outputs and gradients during training
            # print(f"Epoch {epoch + 1}, Batch {i // batch_size}: Sample outputs: {outputs[:5].detach().cpu().numpy()}")
            # log_gradients(model)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        # log_weights(model)


def train_and_predict(
    device, model_class_or_instance, X_train, y_train, X_test, scalers, epochs,
    batch_size, horizon, test, criterion, optimizer_class=None, learning_rate=1e-3,
    clip_grad_norm=None, perform_sanity_checks=False
):
    """
    Train a model and evaluate its performance with optional diagnostics.

    Args:
        device (torch.device): Device for computation.
        model_class_or_instance (class or torch.nn.Module): Model class to instantiate or a pre-initialized model.
        X_train (torch.Tensor): Training inputs.
        y_train (torch.Tensor): Training targets.
        X_test (torch.Tensor): Test inputs.
        scalers (dict): Scalers for inverse transformation.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        horizon (int): Forecasting horizon.
        test (pd.DataFrame): Test DataFrame.
        criterion (torch.nn.Module): Loss function.
        optimizer_class (torch.optim.Optimizer, optional): Optimizer class (default: AdamW).
        learning_rate (float, optional): Learning rate for the optimizer.
        clip_grad_norm (float, optional): Max norm for gradient clipping (default: None, no clipping).
        perform_sanity_checks (bool, optional): Perform sanity checks on inputs and model (default: True).

    Returns:
        tuple: (Predicted values, Training time)
    """
    # Initialize model
    if isinstance(model_class_or_instance, torch.nn.Module):
        model = model_class_or_instance.to(device)
    else:
        model = model_class_or_instance().to(device)

    # Initialize optimizer
    optimizer_class = optimizer_class or torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Perform sanity checks if enabled
    if perform_sanity_checks:
        sanity_check_data(X_train, y_train, X_test)

    # Start timing
    start_time = time.time()

    # Train the model
    train_model(
        model, X_train, y_train, batch_size, optimizer, criterion, epochs,
        device=device, clip_grad_norm=clip_grad_norm
    )

    # End timing
    end_time = time.time()

    # Predict using recursive forecasting
    y_pred = recursive_predict(model, X_test, horizon, device, scalers)

    # Reshape predictions to match the expected shape
    num_series = test["unique_id"].nunique()
    y_pred = y_pred.reshape(num_series, horizon)

    # Return predictions and training time
    return y_pred, end_time - start_time


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


def sanity_check_model(model):
    print("\nSanity Check: Model Initialization")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: Min={param.min().item():.4f}, Max={param.max().item():.4f}, Mean={param.mean().item():.4f}, Requires Grad={param.requires_grad}")


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
