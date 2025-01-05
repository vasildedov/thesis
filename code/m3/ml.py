import time
import json
from utils.preprocess_m3 import train_test_split
from utils.preprocess_ml import create_train_windows, create_test_windows
from utils.train_ml import train_and_save_model, ensemble_predict
from utils.models_ml import LGBMModel, XGBModel
from utils.helper import calculate_smape

# Choose the frequency
freq = 'Other'
retrain = False
direct = True

# Load train and test data
train, test, horizon = train_test_split(freq)
look_back = 2*horizon

# Generate windows for training
X_train, y_train = create_train_windows(train, look_back, horizon)
# Prepare the test window (last 'look_back' points of each series)
X_test = create_test_windows(train, look_back)
y_test = test['y'].values.reshape(test['unique_id'].nunique(), horizon)

# ===== Models =====
# LightGBM
lgbm_model = LGBMModel(direct=direct)
y_pred_lgbm, eval_lgbm = train_and_save_model(lgbm_model, "LGBM", X_train, y_train, X_test, y_test, horizon, freq, look_back, retrain, 'm3', direct)

# XGBoost
xgb_model = XGBModel(direct=direct)
y_pred_xgb, eval_xgb = train_and_save_model(xgb_model, "XGB", X_train, y_train, X_test, y_test, horizon, freq, look_back, retrain, 'm3', direct)

# Ensemble predictions
y_pred_ens = ensemble_predict([lgbm_model, xgb_model], X_test, horizon)
eval_ensemble = calculate_smape(y_test, y_pred_ens)

# Save ensemble metadata
ensemble_metadata_path = f"models/m3/ml_{freq.lower()}/ensemble_metadata.json"
ensemble_metadata = {
    "model_name": "Ensemble",
    "frequency": freq.lower(),
    "look_back": look_back,
    "horizon": horizon,
    "SMAPE": eval_ensemble,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(ensemble_metadata_path, "w") as f:
    json.dump(ensemble_metadata, f, indent=4)
print(f"Ensemble metadata saved to {ensemble_metadata_path}")
