import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.m4_preprocess_ml import train_test_split, truncate_series
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed

# Choose the frequency
freq = 'Hourly'  # or 'Daily'

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

def train_and_forecast(series, model_type, order, seasonal_order, horizon):
    fitted_model = train_arima_model(series, model_type=model_type, order=order, seasonal_order=seasonal_order)
    forecast = recursive_predict_arima(fitted_model, steps=horizon)
    return forecast

def train_arima_model(series, model_type='ARIMA', order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    if model_type == 'SARIMA':
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

# Model type can be 'ARIMA' or 'SARIMA'
model_type = 'ARIMA'  # Change as desired

# Using parallel processing to speed up training and forecasting
forecasts = Parallel(n_jobs=-1)(
    delayed(train_and_forecast)(
        train[train['unique_id'] == uid]['y'],
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        horizon=horizon
    )
    for uid in train['unique_id'].unique()
)

# Convert forecasts to numpy array
y_pred = np.array(forecasts)

# Reshape true values
y_true = test['y'].values.reshape(-1, horizon)

# Evaluate forecasts
print('Evaluation:\n', M4Evaluation.evaluate('data', freq, y_pred))
