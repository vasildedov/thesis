import time
import json
from utils.preprocess_ml import create_train_windows, create_test_windows
from utils.train_ml import train_and_save_model, ensemble_predict
from utils.models_ml import LGBMModel, XGBModel
from utils.helper import calculate_smape


# ===== Dataset =====
dataset = 'm3'
freq = 'Monthly'

if dataset == 'm3':
    from utils.preprocess_m3 import train_test_split
    freq = freq.capitalize()
elif dataset == 'm4':
    from utils.preprocess_m4 import train_test_split
    freq = freq.capitalize()
elif dataset == 'tourism':
    from utils.preprocess_tourism import train_test_split
    freq = freq.lower()
elif dataset == 'etth1':
    from utils.preprocess_ett import train_test_split, get_windows

# ===== Parameters =====
retrain = True
direct = True

# Load train and test data
if dataset != 'etth1':
    train, test, horizon = train_test_split(freq)
    look_back = 2*horizon if not (dataset == 'tourism' and freq == 'yearly') else 7
    # Generate windows for training
    X_train, y_train = create_train_windows(train, look_back, horizon)
    # Prepare the test window (last 'look_back' points of each series)
    X_test = create_test_windows(train, look_back)
    y_test = test['y'].values.reshape(test['unique_id'].nunique(), horizon)
else:
    train, val, test = train_test_split()
    look_back = 720
    horizon = 96
    X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train, val, test, look_back, horizon)

# ===== Models =====
# LightGBM
lgbm_model = LGBMModel(direct=direct)
y_pred_lgbm, eval_lgbm = train_and_save_model(lgbm_model, "LGBM", X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, horizon, freq, look_back, retrain, dataset, direct)

# XGBoost
xgb_model = XGBModel(direct=direct)
y_pred_xgb, eval_xgb = train_and_save_model(xgb_model, "XGB", X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, horizon, freq, look_back, retrain, dataset, direct)
