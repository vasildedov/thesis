import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.m4_preprocess import train_test_split, truncate_series
from utils.m4_preprocess_ml import create_train_windows, create_test_windows, recursive_predict
from utils.models_ml import LGBMModel, XGBModel, CatBoostModel

# Choose the frequency
freq = 'Daily'  # or 'Hourly'

# Parameters based on frequency
if freq == 'Hourly':
    look_back = 60   # Look-back window of 60 hours
    horizon = 48     # Forecast horizon of 48 hours
elif freq == 'Daily':
    look_back = 30   # Look-back window of 30 days
    horizon = 14     # Forecast horizon of 14 days
else:
    raise ValueError("Unsupported frequency. Choose 'Hourly' or 'Daily'.")

# Load train and test data
train, test = train_test_split(freq)

# Truncate long series for 'Daily' data
if freq == 'Daily':
    train = truncate_series(train, max_length=200)

# Generate windows for training
X_train, y_train = create_train_windows(train, look_back, horizon)

# Instantiate and train the models
lgbm_model = LGBMModel()
lgbm_model.fit(X_train, y_train)

xgb_model = XGBModel()
xgb_model.fit(X_train, y_train)

# Adjust CatBoost parameters based on frequency
if freq == 'Daily':
    catboost_model = CatBoostModel(hyper_parametrize=False)
else:
    catboost_model = CatBoostModel()
catboost_model.fit(X_train, y_train)

# Prepare the test window (last 'look_back' points of each series)
X_test = create_test_windows(train, look_back)

# Predict on the test set
y_pred_lgbm = recursive_predict(lgbm_model, X_test, horizon)
y_pred_xgb = recursive_predict(xgb_model, X_test, horizon)
y_pred_cb = recursive_predict(catboost_model, X_test, horizon)

# Ensemble prediction using averaging
def ensemble_predict(models, X_input, horizon):
    model_predictions = [recursive_predict(model, X_input, horizon) for model in models]
    return np.mean(model_predictions, axis=0)

y_pred_ens = ensemble_predict([lgbm_model, xgb_model, catboost_model], X_test, horizon)


# Evaluate and print results
print('LGBM evaluation:\n', M4Evaluation.evaluate('data', freq, y_pred_lgbm))
print('XGB evaluation:\n', M4Evaluation.evaluate('data', freq, y_pred_xgb))
print('CatBoost evaluation:\n', M4Evaluation.evaluate('data', freq, y_pred_cb))
print('Ensemble evaluation:\n', M4Evaluation.evaluate('data', freq, y_pred_ens))
