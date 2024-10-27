import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.preprocess_M4 import train_test_split, create_train_windows, create_test_windows, recursive_predict
from utils.ml_models import LGBMModel, calculate_smape, XGBModel

train, test = train_test_split('Hourly')

# Parameters for the sliding window
look_back = 60  # Look-back window of 60 hours
horizon = 48  # Forecast horizon of 48 hours

# Generate windows for training
X_train, y_train, train_ids = create_train_windows(train, look_back, horizon)

# Instantiate and train the model
lgbm_model = LGBMModel()
lgbm_model.fit(X_train, y_train)

xgb_model = XGBModel()
xgb_model.fit(X_train, y_train)

# Prepare the test window (last 60 hours of each series from the training set)
X_test = create_test_windows(train, look_back)

# Predict on the test set
y_pred = recursive_predict(lgbm_model, X_test, horizon)
y_pred_xgb = recursive_predict(xgb_model, X_test, horizon)
# Function to calculate sMAPE

# Calculate sMAPE on the test dataset
y_true = test['y'].values.reshape(-1, horizon)
smape = calculate_smape(y_true, y_pred)

print(f"sMAPE: {smape:.4f}")

# equivalent
print('LGBM evaluation:')
M4Evaluation.evaluate('data', 'Hourly', y_pred)

print('XGB evaluation:')
M4Evaluation.evaluate('data', 'Hourly', y_pred_xgb)
