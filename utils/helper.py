import os
import json
import torch
from datetime import datetime
import numpy as np


def calculate_smape(y_true, y_pred, epsilon=1e-10):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon))


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
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    return None


def save_metadata(metadata, metadata_path):
    """
    Save metadata to a JSON file.

    Parameters:
        metadata (dict): Metadata to save.
        metadata_path (str): Path to save the metadata.
    """
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The MAPE value as a percentage.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return mape
