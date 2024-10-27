import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def calculate_smape(y_true, y_pred, epsilon=1e-10):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon))


# LightGBM model setup
class LGBMModel:
    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class LGBMHyperModel():
    # LightGBM model setup with hyperparameter tuning
    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        param_grid = {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X, y_flat)
        self.model = grid_search.best_estimator_

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


# XGBoost model setup
class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror')

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)
