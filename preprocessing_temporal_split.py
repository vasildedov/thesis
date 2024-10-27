from datasetsforecast.m4 import M4, M4Evaluation, M4Info
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


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
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


# Function to generate input windows for cross-validation using temporal splits
def create_temporal_cv_windows(df, look_back, horizon, n_splits):
    train_splits, valid_splits = [], []

    # Iterate over each unique time series in the dataframe
    for series_id, group in df.groupby("unique_id"):
        series = group["y"].values
        # Extract the integer part from the unique_id (e.g., 'H1' -> 1)
        series_id_int = int(series_id[1:])  # Assuming unique_id is in the format 'H<number>'

        # Define the split size based on the number of splits
        split_size = len(series) // (n_splits + 1)

        # Create train-validation pairs for each fold
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            valid_start = train_end
            valid_end = valid_start + split_size

            # Ensure validation set does not exceed series length
            if valid_end > len(series):
                break

            # Generate training windows for the current fold
            X_train, y_train = [], []
            for j in range(train_end - look_back):
                X_train.append(np.concatenate(([series_id_int], series[j: j + look_back])))
                y_train.append(series[j + look_back: j + look_back + horizon])

            # Generate validation windows for the current fold
            X_valid, y_valid = [], []
            for j in range(valid_start, valid_end - look_back):
                X_valid.append(np.concatenate(([series_id_int], series[j: j + look_back])))
                y_valid.append(series[j + look_back: j + look_back + horizon])

            train_splits.append((np.array(X_train), np.array(y_train)))
            valid_splits.append((np.array(X_valid), np.array(y_valid)))

    return train_splits, valid_splits


# Cross-validation for temporal splits
def temporal_cross_validation(df, look_back, horizon, model, n_splits=5):
    """
    Train and validate a global model using temporal cross-validation splits.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns `unique_id`, `ds`, and `y`.
    - look_back (int): Number of time steps to look back for each input window.
    - horizon (int): Number of time steps to predict.
    - model: Forecasting model.
    - n_splits (int): Number of cross-validation splits.

    Returns:
    - results (list): List of evaluation metrics for each fold.
    """
    train_splits, valid_splits = create_temporal_cv_windows(df, look_back, horizon, n_splits)
    results = []

    for i, ((X_train, y_train), (X_valid, y_valid)) in enumerate(zip(train_splits, valid_splits)):
        # Fit the model on the training set
        model.fit(X_train, y_train)  # Assuming a fit method exists for the model
        # Predict on the validation set using recursive prediction
        y_pred = recursive_predict(model, X_valid, horizon)
        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, y_pred.shape)

        # Evaluate the model
        smape = calculate_smape(y_valid, y_pred)
        mae = np.mean(np.abs(y_valid - y_pred))
        rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))
        mse = np.mean((y_valid - y_pred) ** 2)

        results.append({
            "fold": i + 1,
            "sMAPE": smape,
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse
        })

        print(f"Fold {i + 1}: sMAPE = {smape:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, MSE = {mse:.4f}")

    return results


# Recursive prediction function for multi-step prediction (same as before)
def recursive_predict(model, X_input, horizon):
    predictions = []
    X_current = X_input

    for _ in range(horizon):
        y_pred = model.predict(X_current)
        predictions.append(y_pred)

        # Update X_current for next prediction (shift left and add y_pred as the new last value)
        X_current = np.concatenate((X_current[:, 1:], y_pred.reshape(-1, 1)), axis=1)

    return np.hstack(predictions)


# Evaluation function (same as before)
def calculate_smape(y_true, y_pred):
    """Calculate sMAPE."""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))


# Example usage
look_back = 24  # Use 24 hours for look-back
horizon = 48  # Forecast horizon of 48 hours
n_splits = 5  # Number of cross-validation splits


# LightGBM model setup
class LGBMModel:
    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X, y):
        # Flatten y to match LGBMRegressor requirements (predicting one step ahead)
        y_flat = y[:, 0]  # Only predict the first horizon value for simplicity
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)  # Predict only the first horizon value


# XGBoost model setup
class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror')

    def fit(self, X, y):
        # Flatten y to match XGBRegressor requirements (predicting one step ahead)
        y_flat = y[:, 0]  # Only predict the first horizon value for simplicity
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)  # Predict only the first horizon value


# Instantiate models
lgbm_model = LGBMModel()
xgb_model = XGBModel()

# Assume `train` is the preprocessed dataframe with columns `unique_id`, `ds`, and `y`
print("LightGBM Model Cross-Validation Results:")
lgbm_results = temporal_cross_validation(train, look_back, horizon, lgbm_model, n_splits)

print("\nXGBoost Model Cross-Validation Results:")
xgb_results = temporal_cross_validation(train, look_back, horizon, xgb_model, n_splits)
