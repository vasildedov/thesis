import numpy as np
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.preprocess_m4 import train_test_split, truncate_series
from utils.train_stats import train_and_forecast
import os
import time
import json
from datetime import datetime

# Choose the frequency
freq = 'Daily'  # Options: 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'
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

# Using parallel processing to speed up training and forecasting
start_overall_time = time.time()

forecasts = [
    train_and_forecast(
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
