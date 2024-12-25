import numpy as np
from joblib import Parallel, delayed
import os
import time
import json
from datetime import datetime
from utils.train_stats import train_and_forecast
from utils.preprocess_m3 import train_test_split
from utils.helper import calculate_smape

# Choose the frequency
freq = 'Quarterly'  # Options: 'Yearly', 'Quarterly', 'Monthly', 'Other'
# Model type can be 'ARIMA' or 'SARIMA'
model_type = 'ARIMA'

if freq == 'Yearly':
    order, seasonal_order = (1, 1, 0), (0, 0, 0, 0)  # Minimal seasonality, focus on trend
elif freq == 'Quarterly':
    order, seasonal_order = (2, 1, 2), (0, 1, 1, 4)  # Quarterly seasonality
elif freq == 'Monthly':
    order, seasonal_order = (3, 1, 2), (0, 1, 1, 12)  # Monthly seasonality
elif freq == 'Other':
    order, seasonal_order = (1, 1, 1), (0, 0, 0, 0)  # Minimal, adapt based on data
else:
    raise ValueError("Unsupported frequency. Choose a valid M3 frequency.")

if model_type == 'ARIMA':
    seasonal_order = (0, 0, 0, 0)  # Explicitly set to zero-seasonality

# Load data
train, test, horizon = train_test_split(freq)

# Define the folder to save all models
model_folder = f"models/m3/stats_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)

# Using parallel processing to speed up training and forecasting
start_overall_time = time.time()

# Using parallel processing to speed up training and forecasting
forecasts = Parallel(n_jobs=-1)(
    delayed(train_and_forecast)(
        train[train['unique_id'] == uid]['y'],
        unique_id=uid,
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        horizon=horizon,
        model_folder=model_folder
    )
    for uid in train['unique_id'].unique()
)

end_overall_time = time.time()

# Convert forecasts to numpy array
y_pred = np.array(forecasts)

# Reshape true values
y_true = test['y'].values.reshape(-1, horizon)

# Evaluate forecasts
evaluation = calculate_smape(y_true, y_pred)
print('SMAPE:\n', round(evaluation, 2))

# Save evaluation metadata
metadata_path = os.path.join(model_folder, f"{model_type.lower()}_metadata.json")
metadata = {
    "model_name": model_type,
    "frequency": freq.lower(),
    "order": order,
    "seasonal_order": seasonal_order,
    "horizon": horizon,
    "SMAPE": round(evaluation, 2),
    "time_to_train": round(end_overall_time-start_overall_time, 2),
    "timestamp": datetime.now().isoformat()
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {metadata_path}")
