import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


# Data Preprocessing: Create input/output windows with scaling and integer series identifier
def create_train_windows(df, look_back, horizon=1, direct=False):
    """
    Create training windows for RNN and Transformer models.

    Args:
        df (pd.DataFrame): DataFrame with columns `unique_id` and `y`.
        look_back (int): Number of past time steps to use as input.
        horizon (int): Forecasting horizon.

    Returns:
        X (torch.Tensor): Input tensor of shape [samples, look_back].
        y (torch.Tensor): Target tensor of shape [samples].
        scalers (dict): Dictionary of scalers keyed by `unique_id`.
    """
    X, y = [], []
    scalers = {}

    for unique_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()  # Scale each series individually
        scalers[unique_id] = scaler

        for i in range(len(scaled_series) - look_back - horizon + 1):
            seq_x = scaled_series[i: i + look_back]
            if direct:
                seq_y = scaled_series[i + look_back: i + look_back + horizon]  # Multi-step targets
            else:
                seq_y = scaled_series[i + look_back]  # single-step target
            X.append(seq_x)
            y.append(seq_y)

    X = torch.tensor(np.array(X), dtype=torch.float32)  # Shape: [samples, look_back]
    y = torch.tensor(np.array(y), dtype=torch.float32)  # Shape: [samples]  or [samples, horizon] if multi-step targets
    return X, y, scalers


def create_test_windows(df, look_back, scalers):
    """
    Create test windows for both RNN and Transformer models.

    Args:
        df (pd.DataFrame): DataFrame with columns `unique_id` and `y`.
        look_back (int): Number of past time steps to use as input.
        scalers (dict): Scalers for each unique_id.

    Returns:
        X_test (torch.Tensor): Input tensor of shape [num_series, look_back].
    """
    X_test = []

    for unique_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = scalers[unique_id]  # Retrieve scaler for the series
        scaled_series = scaler.transform(series).flatten()  # Scale the series

        X_test.append(scaled_series[-look_back:])  # Use the last `look_back` steps

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)  # Shape: [num_series, look_back]
    return X_test

