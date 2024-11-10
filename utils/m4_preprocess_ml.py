from datasetsforecast.m4 import M4, M4Info
import numpy as np
import pandas as pd

# Load the dataset
def train_test_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid

# Function to truncate series to a maximum length
def truncate_series(df, max_length=300):
    truncated_dfs = []
    for _, group in df.groupby('unique_id'):
        if len(group) > max_length:
            group = group.tail(max_length)
        truncated_dfs.append(group)
    return pd.concat(truncated_dfs).reset_index(drop=True)

# Modify this function to exclude series ID as a feature
def create_train_windows(df, look_back, horizon):
    X, y = [], []
    for _, group in df.groupby("unique_id"):
        series = group["y"].values
        for i in range(len(series) - look_back - horizon + 1):
            X.append(series[i: i + look_back])  # Exclude series ID
            y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)

# Modify this function to exclude series ID as a feature
def create_test_windows(df, look_back):
    X_test = []
    for _, group in df.groupby("unique_id"):
        series = group["y"].values[-look_back:]  # Only use look-back window
        X_test.append(series)
    return np.array(X_test)

# Recursive prediction function for multi-step prediction
def recursive_predict(model, X_input, horizon):
    predictions = []
    X_current = X_input
    for _ in range(horizon):
        y_pred = model.predict(X_current)
        predictions.append(y_pred)
        X_current = np.concatenate((X_current[:, 1:], y_pred.reshape(-1, 1)), axis=1)
    return np.hstack(predictions)
