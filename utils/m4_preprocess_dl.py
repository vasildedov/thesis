import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation

# Data Preprocessing: Create input/output windows with scaling and integer series identifier
def create_rnn_windows(df, look_back, horizon):
    X, y = [], []
    scalers = {}

    for _, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()  # Scale each series individually
        scalers[group["unique_id"].iloc[0]] = scaler  # Store scaler using the unique ID

        for i in range(len(scaled_series) - look_back - horizon + 1):
            X.append(scaled_series[i: i + look_back])
            y.append(scaled_series[i + look_back])  # Target is the next step after look_back

    X = torch.tensor(np.array(X), dtype=torch.float32)  # Shape: [samples, look_back]
    y = torch.tensor(np.array(y), dtype=torch.float32)  # Shape: [samples]
    return X, y, scalers


def create_transformer_windows(df, look_back):
    X, y = [], []
    scalers = {}

    for _, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series).flatten()  # Scale the series
        scalers[group["unique_id"].iloc[0]] = scaler  # Store scaler keyed by `unique_id`

        for i in range(len(scaled_series) - look_back):
            seq_x = scaled_series[i: i + look_back]  # Input window
            seq_y = scaled_series[i + look_back]    # Target value
            X.append(seq_x)
            y.append(seq_y)

    X = torch.tensor(np.array(X), dtype=torch.float32)  # Shape: [samples, look_back]
    y = torch.tensor(np.array(y), dtype=torch.float32)  # Shape: [samples]
    X = X.unsqueeze(-1)  # Add feature dimension -> Shape: [samples, look_back, 1]
    return X, y, scalers


# Prepare test data: Use the last `look_back` steps from each series in train and include series ID
def create_test_windows(df, look_back, scalers):
    X_test = []

    for _, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = scalers[group["unique_id"].iloc[0]]  # Retrieve scaler for the series
        scaled_series = scaler.transform(series).flatten()  # Scale using the training scaler

        X_test.append(scaled_series[-look_back:])  # Use the last `look_back` steps

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)  # Shape: [num_series, look_back]
    X_test = X_test.unsqueeze(-1)  # Shape: [num_series, look_back, 1]
    return X_test


def create_test_windows_transformer(df, look_back, scalers):
    X_test = []

    for _, group in df.groupby("unique_id"):
        series = group["y"].values.reshape(-1, 1)
        scaler = scalers[group["unique_id"].iloc[0]]  # Retrieve scaler for the series
        scaled_series = scaler.transform(series).flatten()  # Scale the series

        seq_x = scaled_series[-look_back:]  # Use the last `look_back` steps
        X_test.append(seq_x)

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)  # Shape: [num_series, look_back]
    X_test = X_test.unsqueeze(-1)  # Add feature dimension -> Shape: [num_series, look_back, 1]
    return X_test


# Recursive prediction function for multi-step horizon with reverse scaling
def recursive_predict_rnn(model, X_input, horizon, device, scalers):
    model.eval()
    predictions = []
    X_current = X_input.to(device)

    with torch.no_grad():
        for _ in range(horizon):
            y_pred = model(X_current).unsqueeze(-1)  # Predict one step ahead
            predictions.append(y_pred.cpu().numpy())  # Collect prediction
            X_current = torch.cat((X_current[:, 1:], y_pred), dim=1)  # Update sequence

    predictions = np.hstack(predictions)  # Flatten predictions into a single array

    # Reverse scaling for each series
    predictions_rescaled = []
    for i, scaler in enumerate(scalers.values()):
        pred_scaled = scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
        predictions_rescaled.append(pred_scaled)

    return np.array(predictions_rescaled)  # Return rescaled predictions


def recursive_predict_transformer(model, X_input, horizon, device, scalers):
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
    for i, scaler in enumerate(scalers.values()):
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


def train_and_evaluate(device, model_class, model_name, X_train, y_train, X_test, scalers, epochs,
                       batch_size, criterion, horizon, test, freq, model_type='RNN'):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    nn_train(model, epochs, X_train, y_train, batch_size, optimizer, criterion, model_type=model_type)

    # Make predictions using recursive forecasting
    if model_type == 'Transformer':
        y_pred = recursive_predict_transformer(model, X_test, horizon, device, scalers)
    else:
        y_pred = recursive_predict_rnn(model, X_test, horizon, device, scalers)

    # Reshape predictions to match the expected shape
    num_series = test['unique_id'].nunique()
    y_pred = y_pred.reshape(num_series, horizon)

    # Evaluate using sMAPE
    y_true = test['y'].values.reshape(num_series, horizon)
    print(f"{model_name} Model Evaluation:\n", M4Evaluation.evaluate('data', freq, y_pred))
    return y_pred
