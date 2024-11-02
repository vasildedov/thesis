import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_preprocess_M4 import train_test_split, create_train_windows, create_test_windows, recursive_predict
from utils.ml_models import LGBMModel, calculate_smape, XGBModel, CatBoostModel

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

catboost_model = CatBoostModel()
catboost_model.fit(X_train, y_train)

# Ensemble prediction using averaging
# Prepare the test window (last 60 hours of each series from the training set)
X_test = create_test_windows(train, look_back)

# Predict on the test set
y_pred = recursive_predict(lgbm_model, X_test, horizon)
y_pred_xgb = recursive_predict(xgb_model, X_test, horizon)
y_pred_cb = recursive_predict(catboost_model, X_test, horizon)
def ensemble_predict(models, X_input, horizon):
    model_predictions = [recursive_predict(model, X_input, horizon) for model in models]
    return np.mean(model_predictions, axis=0)
# Predict on the test set using the ensemble
y_pred_ens = ensemble_predict([lgbm_model, xgb_model, catboost_model], X_test, horizon)
# Calculate sMAPE on the test dataset
y_true = test['y'].values.reshape(-1, horizon)
print(f"sMAPE: {calculate_smape(y_true, y_pred):.4f}")

# equivalent
print('LGBM evaluation:\n', M4Evaluation.evaluate('data', 'Hourly', y_pred))

print('XGB evaluation:\n', M4Evaluation.evaluate('data', 'Hourly', y_pred_xgb))

print('Catboost evaluation:\n', M4Evaluation.evaluate('data', 'Hourly', y_pred_cb))

print('Ensemble evaluation:\n', M4Evaluation.evaluate('data', 'Hourly', y_pred_ens))
