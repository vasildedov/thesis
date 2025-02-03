import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import os

def load_existing_model(model_path, device, model_class, model_kwargs):
    """
    Load an existing model from the given path if it exists.

    Parameters:
        model_path (str): Path to the saved model.
        device (torch.device): Device to load the model onto.
        model_class (callable): Callable to instantiate the model class.
        model_kwargs (dict): Arguments required to instantiate the model.

    Returns:
        model (torch.nn.Module): Loaded model or None if the file doesn't exist.
    """
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}. Loading...")
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model
    return None


def predict(model, X_input, horizon, device, scalers=None, series_ids=None, batch_size=None, direct=False,
            multivariate=False):
    model.eval()
    num_series = X_input.size(0)
    batch_size = num_series if multivariate else batch_size
    predictions_rescaled = []

    with torch.no_grad():
        for start_idx in range(0, num_series, batch_size or num_series):
            # Select the batch
            end_idx = min(start_idx + batch_size, num_series)
            X_batch = X_input[start_idx:end_idx].clone().to(device)  # [batch_size, look_back, input_size]
            batch_series_ids = series_ids[start_idx:end_idx]

            if direct:
                # Direct multi-step prediction
                batch_predictions = model(X_batch).cpu().numpy()  # Shape: [batch_size, horizon]
            else:
                # Recursive forecasting
                batch_predictions = []
                for _ in range(horizon):
                    y_pred = model(X_batch)

                    if y_pred.dim() == 1:  # Flattened output [batch_size]
                        y_pred = y_pred.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                    elif y_pred.dim() == 2:  # [batch_size, 1]
                        y_pred = y_pred.unsqueeze(-1)  # [batch_size, 1, 1]

                    # Slide the input window forward
                    X_batch = torch.cat((X_batch[:, 1:, :], y_pred), dim=1)
                    batch_predictions.append(y_pred.cpu().numpy())

                batch_predictions = np.concatenate(batch_predictions, axis=1)  # [batch_size, horizon]

    try:
        # Reverse scaling for each series
        if scalers is not None:
            for i, series_id in enumerate(batch_series_ids):
                scaler = scalers[series_id]
                pred_scaled = scaler.inverse_transform(batch_predictions[i].reshape(-1, 1)).flatten()
                predictions_rescaled.append(pred_scaled)
            return np.array(predictions_rescaled).reshape(num_series, horizon)  # [num_series, horizon]
        else:
            return np.array(batch_predictions).reshape(num_series, horizon)
    except ValueError as e:
        print(f"Error reshaping predictions. Expected shape: ({num_series}, {horizon}), "
              f"but got {len(predictions_rescaled)} elements.")
        raise e


def train_model(model, X_train, y_train, batch_size, optimizer, criterion, epochs,
                device=None, clip_grad_norm=None, direct=False):
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

            # Handle potential extra dimensions in input
            if batch_X.dim() == 4 and batch_X.shape[-1] == 1:  # Check for [batch_size, seq_len, num_features, 1]
                batch_X = batch_X.squeeze(-1)  # Remove the last dimension

            if direct:
                # Direct multi-step prediction logic
                outputs = model(batch_X)  # Shape: [batch_size, horizon]
                loss = criterion(outputs, batch_y)  # Loss over the entire horizon
            else:
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


def train_and_save(device, model_class_or_instance, X_train, y_train, epochs, batch_size, criterion,
                      model_path=None, optimizer_class=None, learning_rate=1e-3, clip_grad_norm=None,
                      perform_sanity_checks=False, direct=False):
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
        sanity_check_data(X_train, y_train)

    # Start timing
    start_time = time.time()

    # Train the model
    train_model(model, X_train, y_train, batch_size, optimizer, criterion, epochs, device, clip_grad_norm, direct)

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # End timing
    end_time = time.time()

    return model, end_time - start_time


# Sanity checks for input data
def sanity_check_data(X_train, y_train):
    print("Sanity Check: Input Data")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
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
