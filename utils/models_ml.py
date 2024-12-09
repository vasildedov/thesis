import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import catboost as cb


class LGBMModel:
    def __init__(self, hyper_parametrize=False):
        self.model = lgb.LGBMRegressor()
        self.hyper_parametrize = hyper_parametrize

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        if self.hyper_parametrize:
            param_grid = {
                'num_leaves': [31, 50],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_absolute_error')
            grid_search.fit(X, y_flat)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X, y_flat)

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


# CatBoost model setup with GPU and hyperparameter tuning
class CatBoostModel:
    def __init__(self, hyper_parametrize=False):
        self.model = cb.CatBoostRegressor(verbose=0, task_type='GPU')
        self.hyper_parametrize = hyper_parametrize

    def fit(self, X, y):
        y_flat = y[:, 0]  # Predict the first value of the horizon
        if self.hyper_parametrize:
            param_grid = {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [500, 1000]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_absolute_error')
            grid_search.fit(X, y_flat)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)
