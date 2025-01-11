import numpy as np
from datasetsforecast.m4 import M4Evaluation
from utils.train_stats import train_and_forecast
import os
import time
import json
from datetime import datetime
from utils.params_stats import get_params
from utils.helper import calculate_smape, calculate_mape

dataset = 'tourism'
freq = 'monthly'  # Options: 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'
model_type = 'SARIMA'

order, seasonal_order, asfreq = get_params(dataset, freq)

if dataset == 'm3':
    from utils.preprocess_m3 import train_test_split
elif dataset == 'm4':
    from utils.preprocess_m4 import train_test_split
elif dataset == 'tourism':
    from utils.preprocess_tourism import train_test_split

# Load data
train, test, horizon = train_test_split(freq)
if dataset != 'm4':
    train.set_index('ds', inplace=True)

# Define the folder to save all models
model_folder = f"models/{dataset}/recursive/stats_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)

start_overall_time = time.time()
# Using parallel processing to speed up training and forecasting
forecasts = [
    train_and_forecast(
        train[train['unique_id'] == uid]['y'].asfreq(asfreq) if (asfreq and dataset != 'm4') else train[train['unique_id'] == uid]['y'],
        unique_id=uid,
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        horizon=horizon,
        model_folder=model_folder
    )
    for uid in train['unique_id'].unique()
]

end_overall_time = time.time()

# Convert forecasts to numpy array
y_pred = np.array(forecasts)

# Reshape true values
y_true = test['y'].values.reshape(-1, horizon)

# Evaluate forecasts
evaluation={}
evaluation['SMAPE'] = calculate_smape(y_true, y_pred)
print('SMAPE:\n', round(evaluation['SMAPE'], 2))
evaluation['MAPE'] = calculate_mape(y_true, y_pred)
print('MAPE:\n', round(evaluation['MAPE'], 2))

# Save evaluation metadata
metadata_path = os.path.join(model_folder, f"{model_type.lower()}_metadata.json")
metadata = {
    "model_name": model_type,
    "frequency": freq.lower(),
    "order": order,
    "seasonal_order": seasonal_order,
    "horizon": horizon,
    "SMAPE": round(evaluation['SMAPE'], 2),
    "MAPE": round(evaluation['MAPE'], 2),
    "time_to_train": round(end_overall_time-start_overall_time, 2),
    "timestamp": datetime.now().isoformat()
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {metadata_path}")
