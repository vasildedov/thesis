import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_preprocess_M4 import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed
from utils.ml_preprocess_M4 import truncate_series

train, test = train_test_split('Daily')
train = truncate_series(train, 200)


def train_and_forecast(series, model_type, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), horizon=48):
    fitted_model = train_arima_model(series, model_type=model_type, order=order, seasonal_order=seasonal_order)
    forecast = recursive_predict_arima(fitted_model, steps=horizon)
    return forecast


def train_arima_model(series, model_type='ARIMA', order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    if model_type == 'SARIMA':
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)
    else:  # Default to ARIMA
        model = SARIMAX(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False, method='powell')
    return fitted_model


def recursive_predict_arima(fitted_model, steps):
    forecast = fitted_model.forecast(steps=steps)
    return forecast


# Train and forecast for each time series
order = (5, 1, 1)  # Reduced look-back to 10 to speed up training  # Look-back of 60, similar to ML models
seasonal_order = (1, 1, 0, 7)  # Seasonal component reflecting weekly pattern for daily dataset
model_type = 'ARIMA'  # Change to 'ARIMA', 'SARIMA', or 'STL' as desired
horizon = 14


# Using parallel processing to speed up training and forecasting
forecasts = Parallel(n_jobs=-1)(
    delayed(train_and_forecast)(train[train['unique_id'] == uid]['y'], model_type,
                                order=order, seasonal_order=seasonal_order, horizon=horizon)
    for uid in train['unique_id'].unique()
)

# Calculate sMAPE for the forecasts
y_true = test['y'].values.reshape(-1, horizon)
y_pred = np.array(forecasts)

print('evaluation:\n', M4Evaluation.evaluate('data', 'Daily', y_pred))
