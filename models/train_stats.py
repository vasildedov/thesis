from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import os
import json
import numpy as np


def train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon, model_folder,
                       exogenous_train=None, exogenous_test=None):
    """Train the model and forecast, saving/loading models with parameter validation."""
    model_path = os.path.join(model_folder, f"{model_type.lower()}_{unique_id.lower()}.json")

    # try loading
    if os.path.exists(model_path):
        print(f"Loading existing model for {unique_id}...")
        with open(model_path, "r") as f:
            model_data = json.load(f)

        model = SARIMAX(
            series,
            exog=exogenous_train,
            order=tuple(model_data["order"]),
            seasonal_order=tuple(model_data["seasonal_order"]) if model_data["seasonal_order"] is not None else None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        expected_length = model.k_params  # Expected number of parameters
        params = np.array(model_data["params"])

        if len(params) != expected_length:  # if the model was not saved correctly, delete entry and retrain
            print(f"Parameter mismatch for {unique_id}. Expected {expected_length}, got {len(params)}.")
            print(f"Re-training model for {unique_id}...")
            os.remove(model_path)  # Delete malformed file
            return train_and_forecast(series, unique_id, model_type, order, seasonal_order, horizon, model_folder,
                                      exogenous_train, exogenous_test)

        # Apply the saved parameters if model was saved and loaded correctly
        fitted_model = model.filter(params)
    else:
        print(f"Training new model for {unique_id}...")
        fitted_model = train_arima_model(series, order=order, seasonal_order=seasonal_order, exogenous=exogenous_train)

        model_data = {
            "params": fitted_model.params.tolist(),  # Save model parameters
            "order": order,
            "seasonal_order": seasonal_order
        }
        with open(model_path, "w") as f:
            json.dump(model_data, f)
        print(f"Model for {unique_id} saved to {model_path}.")

    forecast = recursive_predict_arima(fitted_model, exog=exogenous_test, steps=horizon)
    return forecast


def train_arima_model(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), exogenous=None):
    model = SARIMAX(
        series,
        exog=exogenous,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False, method='powell')
    return fitted_model


def recursive_predict_arima(fitted_model, steps, exog=None):
    forecast = fitted_model.forecast(steps=steps, exog=exog)
    return forecast
