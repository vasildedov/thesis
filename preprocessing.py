from datasetsforecast.m4 import M4, M4Evaluation, M4Info
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import lightgbm as lgb
import xgboost as xgb


def train_test_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid


hourly_train, hourly_test = train_test_split('Hourly')


# Function to standardize training and validation sets separately
def standardize_series(train_df, valid_df):
    scaler = StandardScaler()

    # Fit the scaler on the training data only
    train_df['y'] = scaler.fit_transform(train_df[['y']])

    # Transform the validation data using the scaler from the training data
    valid_df['y'] = scaler.transform(valid_df[['y']])

    return train_df, valid_df, scaler


# Function to ensure consistent time indexing
def ensure_time_index(df):
    df['ds'] = pd.to_datetime(df['ds'])  # Convert to datetime if not already
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    return df


# Usage example (assuming `train` and `valid` dataframes)
# train = ensure_time_index(hourly_train)
# valid = ensure_time_index(hourly_valid)
train, test, scaler = standardize_series(hourly_train, hourly_test)


# Function to create training windows
def create_train_windows(df, look_back, horizon):
    X, y, series_ids = [], [], []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values
        series_id_int = int(series_id[1:])  # Assuming unique_id is of the form 'H1', 'H2', etc.
        for i in range(len(series) - look_back - horizon + 1):
            X.append(np.concatenate(([series_id_int], series[i: i + look_back])))
            y.append(series[i + look_back: i + look_back + horizon])
            series_ids.append(series_id_int)
    return np.array(X), np.array(y), np.array(series_ids)

# Parameters for the sliding window
look_back = 60  # Look-back window of 60 hours
horizon = 48  # Forecast horizon of 48 hours

# Generate windows for training
X_train, y_train, train_ids = create_train_windows(train, look_back, horizon)

# LightGBM model setup
class LGBMModel:
    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)

# Instantiate and train the model
lgbm_model = LGBMModel()
lgbm_model.fit(X_train, y_train)


# Prepare the test window (last 60 hours of each series from the training set)
def create_test_windows(df, look_back):
    X_test = []
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values[-look_back:]
        series_id_int = int(series_id[1:])
        X_test.append(np.concatenate(([series_id_int], series)))
    return np.array(X_test)

X_test = create_test_windows(train, look_back)

# Recursive prediction function for multi-step prediction
def recursive_predict(model, X_input, horizon):
    predictions = []
    X_current = X_input
    for _ in range(horizon):
        y_pred = model.predict(X_current)
        predictions.append(y_pred)
        X_current = np.concatenate((X_current[:, 1:], y_pred.reshape(-1, 1)), axis=1)
    return np.hstack(predictions)

# Predict on the test set
y_pred = recursive_predict(lgbm_model, X_test, horizon)


# Function to calculate sMAPE
def calculate_smape(y_true, y_pred, epsilon=1e-10):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon))

# Calculate sMAPE on the test dataset
y_true = test['y'].values.reshape(-1, horizon)
smape = calculate_smape(y_true, y_pred)

print(f"sMAPE: {smape:.4f}")
