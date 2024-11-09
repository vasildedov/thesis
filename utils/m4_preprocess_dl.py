import torch
import numpy as np


def create_rnn_windows(df, look_back, horizon):
    X, y = [], []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values
        for i in range(len(series) - look_back - horizon + 1):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back:i + look_back + horizon])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


# Prepare test data: Use the last 60 hours from each series in train
def create_test_windows(df, look_back):
    X_test = []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values[-look_back:]
        X_test.append(series)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    return X_test.unsqueeze(-1)  # Adding feature dimension for RNN


# Recursive prediction function for the test set
def recursive_predict_rnn(model, X_input, horizon, device):
    model.eval()
    predictions = []
    X_current = X_input.to(device)
    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current)
            predictions.append(y_pred.cpu().numpy()[:, -1])  # Last step prediction
            X_current = torch.cat((X_current[:, 1:], y_pred[:, -1].unsqueeze(-1).unsqueeze(-1)), dim=1)
    return np.hstack(predictions)