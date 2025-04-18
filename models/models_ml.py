import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor


class LGBMModel:
    def __init__(self, hyper_parametrize=False, direct=False):
        self.model = MultiOutputRegressor(lgb.LGBMRegressor()) if direct else lgb.LGBMRegressor()
        self.hyper_parametrize = hyper_parametrize
        self.direct = direct

    def fit(self, X, y):
        y_flat = y[:, 0] if not self.direct else y.copy()  # Predict the first value of the horizon
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
        return self.model.predict(X).reshape(-1, 1) if not self.direct else self.model.predict(X)


# XGBoost model setup
class XGBModel:
    def __init__(self, direct=False):
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
        self.direct = direct

    def fit(self, X, y):
        y_flat = y[:, 0] if not self.direct else y.copy()  # Predict the first value of the horizon
        self.model.fit(X, y_flat)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1) if not self.direct else self.model.predict(X)

