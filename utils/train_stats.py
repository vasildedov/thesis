from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import os
import json
import numpy as np


def train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon, model_folder):
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
            return train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon, model_folder)

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
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:  # Default to ARIMA
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
