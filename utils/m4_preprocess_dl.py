import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


# Data Preprocessing: Create input/output windows with scaling
def create_rnn_windows(df, look_back, horizon):
    X, y = [], []
    scalers = {}

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()  # Scale each series individually
        scalers[series_id] = scaler  # Save the scaler for each series

        for i in range(len(scaled_series) - look_back - horizon + 1):
            X.append(scaled_series[i:i + look_back])
            y.append(scaled_series[i + look_back])  # Only take the next step as target

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scalers  # Return scalers to reverse the scaling for each series


# Prepare test data: Use the last `look_back` steps from each series in train
def create_test_windows(df, look_back, scalers):
    X_test = []

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = scalers[series_id]  # Retrieve scaler for the series
        scaled_series = scaler.transform(series).flatten()  # Scale using the training scaler
        X_test.append(scaled_series[-look_back:])

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    return X_test.unsqueeze(-1)  # Adding feature dimension for RNN


# Recursive prediction function for multi-step horizon with reverse scaling
def recursive_predict_rnn(model, X_input, horizon, device, scalers, series_ids):
    model.eval()
    predictions = []
    X_current = X_input.to(device)

    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current).unsqueeze(-1)  # Predict one step ahead
            predictions.append(y_pred.cpu().numpy())  # Collect prediction
            X_current = torch.cat((X_current[:, 1:], y_pred), dim=1)  # Update sequence

    predictions = np.hstack(predictions)  # Flatten predictions into a single array

    # Reverse scaling for each series in the test set
    predictions_rescaled = []
    for i, series_id in enumerate(series_ids):
        scaler = scalers[series_id]
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)  # Return rescaled predictions
