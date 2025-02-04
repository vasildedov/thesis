import numpy as np
from utils.train_stats import train_and_forecast
import os
import time
import json
from datetime import datetime
from utils.params_stats import get_params
from utils.helper import evaluate

dataset = 'etth2'
freq = 'hourly'.capitalize()  # Options: 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'
model_type = 'SARIMA'
multivariate = False

order, seasonal_order, asfreq = get_params(freq, model_type)

if dataset == 'm3':
    from utils.preprocess_m3 import train_test_split
elif dataset == 'm4':
    from utils.preprocess_m4 import train_test_split
elif dataset == 'tourism':
    from utils.preprocess_tourism import train_test_split
    freq = freq.lower()
else:
    from utils.preprocess_ett import train_test_split, get_windows
    multivariate = True

# Load data
if not multivariate:
    train, test, horizon = train_test_split(freq)
    if dataset != 'm4':
        train.set_index('ds', inplace=True)
else:
    train, val, test = train_test_split(group=dataset, multivariate=False)
    horizon = 96
    X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train, val, test, 720, horizon)
    train['unique_id'] = 'etth1'

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
            X_test[i],
            unique_id=f'{i}',
            model_type=model_type,
            order=order,
            seasonal_order=seasonal_order,
            horizon=horizon,
            model_folder=model_folder,

        )
        # for start_idx, end_idx in range(0, len(test), 720)
        for i in range(X_test.shape[0])
    ]
end_overall_time = time.time()

# Convert forecasts to numpy array
y_pred = np.array(forecasts)

# Reshape true values
y_true = y_test.copy()

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
