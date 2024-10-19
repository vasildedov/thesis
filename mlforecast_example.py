# https://www.kaggle.com/code/lemuz90/m4-competition
import random

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from mlforecast import MLForecast
from window_ops.expanding import expanding_mean
from window_ops.ewm import ewm_mean
from window_ops.rolling import rolling_mean, seasonal_rolling_mean

def train_valid_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid

hourly_train, hourly_valid = train_valid_split('Hourly')

%%time
lgb_params = {
    'n_estimators': 200,
    'bagging_freq': 1,
    'learning_rate': 0.05,
    'verbose': -1,
    'force_col_wise': True,
    'num_leaves': 2500,
    'lambda_l1': 0.03,
    'lambda_l2': 0.5,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.8,
}
hourly_fcst = MLForecast(
    models=lgb.LGBMRegressor(**lgb_params),
    freq=1,
    lags=[24 * i for i in range(1, 15)],
    lag_transforms={
        24: [(ewm_mean, 0.3), (rolling_mean, 7 * 24), (rolling_mean, 7 * 48)],
        48: [(ewm_mean, 0.3), (rolling_mean, 7 * 24), (rolling_mean, 7 * 48)],
    },
    num_threads=4,
)
hourly_fcst.fit(
    hourly_train,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)

%time hourly_preds = hourly_fcst.predict(48)
M4Evaluation.evaluate('data', 'Hourly', hourly_preds['LGBMRegressor'].values.reshape(-1, 48))

