import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation

# Data Preprocessing: Create input/output windows with scaling and integer series identifier
def create_rnn_windows(df, look_back, horizon):
    X, y, series_ids = [], [], []
    scalers = {}

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()  # Scale each series individually
        # Store scaler with integer key (e.g., 'D1' -> 1, 'H2' -> 2)
        series_id_int = int(series_id[1:])  # Extract integer part of the ID
        scalers[series_id_int] = scaler

        for i in range(len(scaled_series) - look_back - horizon + 1):
            X.append(np.concatenate(([series_id_int], scaled_series[i: i + look_back])))
            y.append(scaled_series[i + look_back])  # Only take the next step as target
            series_ids.append(series_id_int)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scalers  # Return scalers with integer keys


def create_transformer_windows(df, look_back):
    X, y = [], []
    scalers = {}

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()
        scalers[series_id] = scaler

        for i in range(len(scaled_series) - look_back):
            seq_x = scaled_series[i: i + look_back]
            seq_y = scaled_series[i + look_back]
            X.append(seq_x)
            y.append(seq_y)

    X = torch.tensor(np.array(X), dtype=torch.float32)  # Shape: [samples, look_back]
    y = torch.tensor(np.array(y), dtype=torch.float32)  # Shape: [samples]
    X = X.unsqueeze(-1)  # Shape: [samples, look_back, 1]
    return X, y, scalers


# Prepare test data: Use the last `look_back` steps from each series in train and include series ID
def create_test_windows(df, look_back, scalers):
    X_test, series_ids = [], []

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        # Convert `series_id` to integer for scaler lookup
        series_id_int = int(series_id[1:])
        scaler = scalers[series_id_int]  # Retrieve scaler using the integer part of the ID
        scaled_series = scaler.transform(series).flatten()  # Scale using the training scaler

        # Concatenate the series identifier as the first element in the test window
        window_with_id = np.concatenate(([series_id_int], scaled_series[-look_back:]))
        X_test.append(window_with_id)
        series_ids.append(series_id_int)  # Store the integer identifier only

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    return X_test.unsqueeze(-1), np.array(series_ids)  # Adding feature dimension for RNN


def create_test_windows_transformer(df, look_back, scalers):
    X_test, series_ids = [], []

    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = scalers[series_id]
        scaled_series = scaler.transform(series).flatten()

        seq_x = scaled_series[-look_back:]
        X_test.append(seq_x)
        series_ids.append(series_id)

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)  # Shape: [num_series, look_back]
    X_test = X_test.unsqueeze(-1)  # Shape: [num_series, look_back, 1]
    return X_test, np.array(series_ids)


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
        scaler = scalers[series_id]  # Direct lookup with integer `series_id`
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)  # Return rescaled predictions


def recursive_predict_transformer(model, X_input, horizon, device, scalers, series_ids):
    model.eval()
    predictions = []
    X_current = X_input.clone().to(device)

    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current)            # Shape: [batch_size]
            y_pred = y_pred.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
            predictions.append(y_pred.cpu().numpy())
            X_current = torch.cat((X_current[:, 1:, :], y_pred), dim=1)  # Update sequence

    # Concatenate predictions along the time dimension
    predictions = np.concatenate(predictions, axis=1)  # Shape: [batch_size, horizon, 1]
    predictions = predictions.squeeze(-1)              # Shape: [batch_size, horizon]

    # Reverse scaling for each series
    predictions_rescaled = []
    for i, series_id in enumerate(series_ids):
        scaler = scalers[series_id]
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)


def nn_train(model, epochs, X_train, y_train, batch_size, optimizer, criterion, model_type='RNN'):
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            if model_type == 'Transformer':
                # No need to unsqueeze for Transformer
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            else:
                # Unsqueeze for RNN/LSTM
                outputs = model(batch_X.unsqueeze(-1)).squeeze(-1)
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


def train_and_evaluate(device, model_class, model_name, X_train, y_train, X_test, scalers, series_ids, epochs,
                       batch_size, criterion, horizon, test, freq, model_type='RNN'):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    nn_train(model, epochs, X_train, y_train, batch_size, optimizer, criterion, model_type=model_type)

    # Make predictions using recursive forecasting
    if model_type == 'Transformer':
        y_pred = recursive_predict_transformer(model, X_test, horizon, device, scalers, series_ids)
    else:
        y_pred = recursive_predict_rnn(model, X_test, horizon, device, scalers, series_ids)

    # Reshape predictions to match the expected shape
    y_pred = y_pred.reshape(test.unique_id.nunique(), horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(test.unique_id.nunique(), horizon)
    print(f"{model_name} Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred
