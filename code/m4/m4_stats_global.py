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
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder

# Choose the frequency
freq = 'Hourly'  # or 'Daily'
# Model type can be 'ARIMA' or 'SARIMA'
model_type = 'ARIMA'

# Set parameters based on frequency
if freq == 'Daily':
    order = (5, 1, 1)
    seasonal_order = (1, 1, 0, 7)  # Weekly seasonality
    horizon = 14
    max_length = 200  # Truncate long series to speed up training
elif freq == 'Hourly':
    order = (10, 1, 1)
    seasonal_order = (0, 1, 0, 24)  # Daily seasonality
    horizon = 48
    max_length = None  # Do not truncate series
else:
    raise ValueError("Unsupported frequency. Choose 'Daily' or 'Hourly'.")

# Load data
train, test = train_test_split(freq)
# Truncate series if necessary
if max_length is not None:
    train = truncate_series(train, max_length)

# Define the folder to save all models
model_folder = f"models/stats_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)


def convert_integer_to_datetime(train, freq='H', start_date='2023-01-01'):
    # Convert the integer `ds` to a datetime using the reference start date
    train['ds'] = pd.to_datetime(start_date) + pd.to_timedelta(train['ds'], unit=freq[0])
    return train


def prepare_global_with_identity(train, freq='H'):
    # Pivot the data to align all series into columns
    global_data = train.pivot(index='ds', columns='unique_id', values='y')
    global_data = global_data.asfreq(freq)

    # One-hot encode the `unique_id` for exogenous variables
    unique_ids = global_data.columns.to_list()  # Series identifiers (unique_id)
    encoder = OneHotEncoder(sparse=False)
    encoded_ids = encoder.fit_transform(np.array(unique_ids).reshape(-1, 1))  # One-hot encode series IDs

    # Repeat the one-hot encoded `unique_id` for the length of the time index
    repeated_encoded_ids = np.repeat(encoded_ids[np.newaxis, :, :], global_data.shape[0], axis=0)
    repeated_encoded_ids = repeated_encoded_ids.reshape(global_data.shape[0], -1)

    # Create an exogenous variable DataFrame
    exog = pd.DataFrame(
        repeated_encoded_ids,
        index=global_data.index,
        columns=[f"id_{i}" for i in range(repeated_encoded_ids.shape[1])]
    )

    # Align `exog` with `global_data` and drop NaNs
    global_data.dropna(inplace=True)
    exog = exog.loc[global_data.index]

    return global_data, exog

def train_global_sarimax_with_exog(y, exog, order, seasonal_order):
    """
    Train a global SARIMAX model using exogenous variables.

    Args:
        y (pd.DataFrame): Target matrix (time series values).
        exog (pd.DataFrame): Exogenous feature matrix (e.g., one-hot encoded `unique_id`).
        order (tuple): SARIMAX order (p, d, q).
        seasonal_order (tuple): SARIMAX seasonal order (P, D, Q, s).

    Returns:
        SARIMAXResults: Fitted SARIMAX model.
    """
    y_flat = y.values.flatten()
    exog_flat = exog.values

    # Fit the SARIMAX model
    model = SARIMAX(
        y_flat,
        exog=exog_flat,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)

    return fitted_model


def forecast_global_sarimax_with_exog(fitted_model, exog, horizon):
    """
    Forecast all series using the global SARIMAX model with exogenous variables.

    Args:
        fitted_model (SARIMAXResults): Trained global SARIMAX model.
        exog (pd.DataFrame): Exogenous variables for forecasting.
        horizon (int): Forecast horizon (e.g., 48 hours).

    Returns:
        np.ndarray: Forecasted values reshaped to match the series.
    """
    # Repeat exogenous variables for the forecast horizon
    future_exog = np.tile(exog.values[:1], (horizon, 1))

    # Forecast using the fitted model
    forecast = fitted_model.forecast(steps=horizon, exog=future_exog)

    # Reshape forecast to match the number of series
    num_series = exog.shape[1]
    return forecast.reshape(horizon, num_series)


# Convert integer `ds` to datetime
train = convert_integer_to_datetime(train, freq)
test = convert_integer_to_datetime(test, freq)

# Prepare global dataset
global_data, exog = prepare_global_with_identity(train, freq[0])

# Using parallel processing to speed up training and forecasting
start_overall_time = time.time()
# Train SARIMAX
fitted_model = train_global_sarimax_with_exog(global_data, exog, order, seasonal_order)
end_overall_time = time.time()

# Forecast
forecast = forecast_global_sarimax_with_exog(fitted_model, global_data, horizon)






# Convert forecasts to numpy array
y_pred = np.array(forecast)

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
