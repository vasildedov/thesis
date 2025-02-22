from data_preprocessing.preprocess_ml import create_train_windows, create_test_windows
from models.train_ml import train_and_save_model
from models.models_ml import LGBMModel, XGBModel

# ===== Parameters =====
retrain_mode = False

for direct in [True, False]:
    for dataset in ['tourism', 'm3', 'm4', 'etth1', 'etth2']:
        multivariate = False
        if dataset == 'm3':
            from data_preprocessing.preprocess_m3 import train_test_split
            freqs = ['Other', 'Monthly', 'Quarterly', 'Yearly']
        elif dataset == 'm4':
            from data_preprocessing.preprocess_m4 import train_test_split
            freqs = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
        elif dataset == 'tourism':
            from data_preprocessing.preprocess_tourism import train_test_split
            freqs = ['monthly', 'quarterly', 'yearly']
        else:
            from data_preprocessing.preprocess_ett import train_test_split, get_windows
            freqs = ['default']
            multivariate = True
        
        for freq in freqs:                
            # Load train and test data
            if not multivariate:
                train, test, horizon = train_test_split(freq)
                look_back = 2*horizon if not (dataset == 'tourism' and freq == 'yearly') else 7
                # Generate windows for training
                X_train, y_train = create_train_windows(train, look_back, horizon)
                # Prepare the test window (last 'look_back' points of each series)
                X_test = create_test_windows(train, look_back)
                y_test = test['y'].values.reshape(test['unique_id'].nunique(), horizon)
            else:
                train, val, test = train_test_split(group=dataset)
                look_back = 720
                horizon = 24
                X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train, val, test, look_back, horizon)
            
            # ===== Models =====
            # LightGBM
            lgbm_model = LGBMModel(direct=direct)
            y_pred_lgbm, eval_lgbm = train_and_save_model(lgbm_model, "LGBM", X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, horizon, freq, look_back, retrain_mode, dataset, direct)
            
            # XGBoost
            xgb_model = XGBModel(direct=direct)
            y_pred_xgb, eval_xgb = train_and_save_model(xgb_model, "XGB", X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, horizon, freq, look_back, retrain_mode, dataset, direct)
