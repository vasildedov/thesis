import numpy as np
import pandas as pd


def create_train_windows(df, look_back, horizon):
    X, y = [], []
    for _, group in df.groupby("unique_id"):
        series = group["y"].values
        for i in range(len(series) - look_back - horizon + 1):
            X.append(series[i: i + look_back])  # Exclude series ID
            y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


def create_test_windows(df, look_back):
    X_test = []
    for _, group in df.groupby("unique_id"):
        series = group["y"].values[-look_back:]  # Only use look-back window
        X_test.append(series)
    return np.array(X_test)
