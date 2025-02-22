import numpy as np
from models.train_stats import train_and_forecast
import os
import time
import json
from datetime import datetime
from data_preprocessing.params_stats import get_params
from utils.helpers import evaluate

for model_type in ['ARIMA', 'SARIMA']:
    for dataset in ['tourism', 'm3', 'm4', 'etth1', 'etth2']:
        multivariate = False
        if dataset == 'm3':
            from data_preprocessing.preprocess_m3 import train_test_split
            freqs = ['Other', 'Monthly', 'Quarterly', 'Yearly']
        elif dataset == 'm4':
            from data_preprocessing.preprocess_m4 import train_test_split
            freqs = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
        elif dataset == 'tourism':
            from data_preprocessing.preprocess_tourism import train_test_split
            freqs = ['monthly', 'quarterly', 'yearly']
        else:
            from data_preprocessing.preprocess_ett import train_test_split, get_windows
            freqs = ['default']
            multivariate = True


        for freq in freqs:

            # don't train unnecessary yearly model for SARIMA - assume no seasonality for yearly data
            if (model_type == 'SARIMA' and freq.lower() == 'yearly'):
                continue

            order, seasonal_order, asfreq = get_params(freq, model_type)

            # Load data
            if not multivariate:
                train, test, horizon = train_test_split(freq)
                if dataset != 'm4':
                    train.set_index('ds', inplace=True)
            else:
                train, val, test = train_test_split(group=dataset, multivariate=True)
                horizon = 24
                X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train, val, test, 720, horizon,
                                                                             get_exog=True)
                train['unique_id'] = dataset

            # Define the folder to save all models
            model_folder = f"models/{dataset}/recursive/stats_{freq.lower()}/"
            os.makedirs(model_folder, exist_ok=True)

            start_overall_time = time.time()
            # Using parallel processing to speed up training and forecasting
            if not multivariate:
                forecasts = [
                    train_and_forecast(
                        train[train['unique_id'] == uid]['y'].asfreq(asfreq) if (asfreq and dataset != 'm4') else
                        train[train['unique_id'] == uid]['y'],
                        unique_id=uid,
                        model_type=model_type,
                        order=order,
                        seasonal_order=seasonal_order,
                        horizon=horizon,
                        model_folder=model_folder
                    )
                    for uid in train['unique_id'].unique()
                ]
            else:
                forecasts = [
                    train_and_forecast(
                        X_test[i, :, -1],
                        unique_id=f'{i}',
                        model_type=model_type,
                        order=order,
                        seasonal_order=seasonal_order,
                        horizon=horizon,
                        model_folder=model_folder,
                        exogenous_train=X_test[i, :, :-1],
                        exogenous_test=y_test[i, :, :-1]
                    )
                    for i in range(X_test.shape[0])
                ]
            end_overall_time = time.time()

            # Convert forecasts to numpy array
            y_pred = np.array(forecasts)

            # Reshape true values
            y_true = test['y'].values.reshape(-1, horizon) if not multivariate else y_test[:, :, -1].copy()

            # Evaluate forecasts
            evaluation = evaluate(y_true, y_pred)

            # Save evaluation metadata
            metadata_path = os.path.join(model_folder, f"{model_type.lower()}_metadata.json")
            metadata = {
                "model_name": model_type,
                "frequency": freq.lower(),
                "order": order,
                "seasonal_order": seasonal_order,
                "horizon": horizon,
                **evaluation,
                "time_to_train": round(end_overall_time-start_overall_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to {metadata_path}")
