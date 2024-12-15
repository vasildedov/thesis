import time
import json
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.m4_preprocess import train_test_split, truncate_series
from utils.m4_preprocess_ml import create_train_windows, create_test_windows
from utils.m4_train_ml import train_and_save_model, ensemble_predict
from utils.models_ml import LGBMModel, XGBModel

# Choose the frequency
freq = 'Hourly'
retrain = True

if freq == 'Yearly':
    look_back, horizon = 12, 6
elif freq == 'Quarterly':
    look_back, horizon = 16, 8
elif freq == 'Monthly':
    look_back, horizon = 36, 18
elif freq == 'Weekly':
    look_back, horizon = 26, 13
elif freq == 'Daily':
    look_back, horizon = 28, 14
elif freq == 'Hourly':
    look_back, horizon = 96, 48
else:
    raise ValueError("Unsupported frequency. Choose from 'Yearly', 'Quarterly', 'Monthly', 'Daily', or 'Hourly'.")


# Load train and test data
train, test = train_test_split(freq)

# Truncate long series for 'Daily' data
# if freq == 'Daily':
#     train = truncate_series(train, max_length=200)

# Generate windows for training
X_train, y_train = create_train_windows(train, look_back, horizon)
# Prepare the test window (last 'look_back' points of each series)
X_test = create_test_windows(train, look_back)

# ===== Models =====
# LightGBM
lgbm_model = LGBMModel()
y_pred_lgbm, eval_lgbm = train_and_save_model(lgbm_model, "LGBM", X_train, y_train, X_test, horizon, freq, look_back, retrain)

# XGBoost
xgb_model = XGBModel()
y_pred_xgb, eval_xgb = train_and_save_model(xgb_model, "XGB", X_train, y_train, X_test, horizon, freq, look_back, retrain)

# Ensemble predictions
y_pred_ens = ensemble_predict([lgbm_model, xgb_model], X_test, horizon)
eval_ensemble = M4Evaluation.evaluate('data', freq, y_pred_ens)

# Save ensemble metadata
ensemble_metadata_path = f"models/ensemble_{freq.lower()}_metadata.json"
ensemble_metadata = {
    "model_name": "Ensemble",
    "frequency": freq.lower(),
    "look_back": look_back,
    "horizon": horizon,
    "SMAPE": eval_ensemble['SMAPE'][0],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(ensemble_metadata_path, "w") as f:
    json.dump(ensemble_metadata, f, indent=4)
print(f"Ensemble metadata saved to {ensemble_metadata_path}")
