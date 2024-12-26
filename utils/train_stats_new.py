from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import os
import json
import numpy as np
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split as arima_train_test_split
from pmdarima.arima import ARIMA


# Function to train and forecast
def train_and_forecast(series, seasonal, m, horizon):
    """Train and forecast using pmdarima."""
    # Train the model
    print("Training model...")
    model = auto_arima(
        series,
        seasonal=seasonal,  # Enable seasonal ARIMA
        m=m,           # Seasonal period (12 for monthly, 4 for quarterly, etc.)
        suppress_warnings=True,
        error_action="ignore"  # Ignore errors during training
    )

    # Forecast
    forecast = model.predict(n_periods=horizon)
    return forecast


def train_arima_model(series, model_type='ARIMA', order=(1, 1, 1), seasonal_order=None):
    if model_type == 'ARIMA':
        model = SARIMAX(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    fitted_model = model.fit(disp=False, method='powell')
    return fitted_model


def recursive_predict_arima(fitted_model, steps):
    forecast = fitted_model.forecast(steps=steps)
    return forecast
