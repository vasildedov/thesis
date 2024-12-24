import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.m4_preprocess import train_test_split, truncate_series
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from joblib import Parallel, delayed
import os
import time
import json
from datetime import datetime

# Choose the frequency
freq = 'Monthly'  # Options: 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'
# Model type can be 'ARIMA' or 'SARIMA'
model_type = 'ARIMA'

if freq == 'Yearly':
    order, seasonal_order, max_length = (2, 1, 1), (1, 1, 0, 12), None
elif freq == 'Quarterly':
    order, seasonal_order, max_length = (4, 1, 1), (1, 1, 0, 4), None
elif freq == 'Monthly':
    order, seasonal_order, max_length = (6, 1, 1), (1, 1, 0, 12), 120
elif freq == 'Weekly':
    order, seasonal_order, max_length = (5, 1, 1), (1, 1, 0, 52), 260
elif freq == 'Daily':
    order, seasonal_order, max_length = (5, 1, 1), (1, 1, 0, 7), 200
elif freq == 'Hourly':
    order, seasonal_order, max_length = (24, 1, 1), (0, 1, 1, 24), None
else:
    raise ValueError("Unsupported frequency. Choose a valid M4 frequency.")

if model_type == 'ARIMA':
    seasonal_order = None

# Load data
train, test, horizon = train_test_split(freq)
# Truncate series if necessary
if max_length is not None:
    train = truncate_series(train, max_length)

# Define the folder to save all models
model_folder = f"models/m4/stats_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)


def train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon):
    """Train the model and forecast, saving/loading models with parameter validation."""
    model_path = os.path.join(model_folder, f"{model_type.lower()}_{unique_id.lower()}.json")

    # Ensure the series is not empty
    if series.empty:
        raise ValueError(f"Time series data for {unique_id} is empty. Cannot train or forecast.")

    if os.path.exists(model_path):
        print(f"Loading existing model for {unique_id}...")
        with open(model_path, "r") as f:
            model_data = json.load(f)

        model = SARIMAX(
            series,
            order=tuple(model_data["order"]),
            seasonal_order=tuple(model_data["seasonal_order"]),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        expected_length = model.k_params  # Expected number of parameters
        params = np.array(model_data["params"])

        if len(params) != expected_length:
            print(f"Parameter mismatch for {unique_id}. Expected {expected_length}, got {len(params)}.")
            print(f"Re-training model for {unique_id}...")
            os.remove(model_path)  # Delete malformed file
            return train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon)

        # Apply the saved parameters
        fitted_model = model.filter(params)
    else:
        print(f"Training new model for {unique_id}...")
        fitted_model = train_arima_model(series, model_type=model_type, order=order, seasonal_order=seasonal_order)

        model_data = {
            "params": fitted_model.params.tolist(),  # Save model parameters
            "order": order,
            "seasonal_order": seasonal_order
        }
        with open(model_path, "w") as f:
            json.dump(model_data, f)
        print(f"Model for {unique_id} saved to {model_path}.")

    forecast = recursive_predict_arima(fitted_model, steps=horizon)
    return forecast


def train_arima_model(series, model_type='ARIMA', order=(1, 1, 1), seasonal_order=None):
    if model_type == 'ARIMA':
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:  # Default to ARIMA
        model = SARIMAX(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    fitted_model = model.fit(disp=False, method='powell')
    return fitted_model


def recursive_predict_arima(fitted_model, steps):
    forecast = fitted_model.forecast(steps=steps)
    return forecast


# Using parallel processing to speed up training and forecasting
start_overall_time = time.time()

# Using parallel processing to speed up training and forecasting
forecasts = Parallel(n_jobs=-1)(
    delayed(train_and_forecast)(
        train[train['unique_id'] == uid]['y'],
        unique_id=uid,
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        horizon=horizon
    )
    for uid in train['unique_id'].unique()
)

end_overall_time = time.time()

# Convert forecasts to numpy array
y_pred = np.array(forecasts)

# Reshape true values
y_true = test['y'].values.reshape(-1, horizon)

# Evaluate forecasts
evaluation = M4Evaluation.evaluate('data', freq, y_pred)
print('Evaluation:\n', evaluation)

# Save evaluation metadata
metadata_path = os.path.join(model_folder, f"{model_type.lower()}_metadata.json")
metadata = {
    "model_name": model_type,
    "frequency": freq.lower(),
    "order": order,
    "seasonal_order": seasonal_order,
    "horizon": horizon,
    "SMAPE": evaluation['SMAPE'][0],
    "time_to_train": round(end_overall_time-start_overall_time, 2),
    "timestamp": datetime.now().isoformat()
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {metadata_path}")
