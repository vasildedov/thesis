import lightgbm as lgb
import numpy as np


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
