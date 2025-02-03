import os
import json
from datetime import datetime
import numpy as np


def calculate_smape(y_true, y_pred, epsilon=1e-10):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon))


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
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return mape


def evaluate(y_true, y_pred):
    evaluation_dict = {
        "SMAPE": round(calculate_smape(y_true, y_pred), 2),
        "MAPE": round(calculate_mape(y_true, y_pred), 3),
        "MAE": round(np.mean(np.abs(y_true - y_pred)), 3),
        "MSE": round(np.mean((y_true - y_pred) ** 2), 3)
    }
    for metric, value in evaluation_dict.items():
        print(f"{metric}: {value}")
    return evaluation_dict

