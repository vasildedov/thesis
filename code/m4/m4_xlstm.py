import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import create_train_windows, create_test_windows

from utils.dl_models import xLSTMTimeSeriesModel
from utils.preprocess_xlstm import train_and_predict, get_stack_cfg

# args
# Choose the frequency
freq = 'Hourly'  # or 'Daily'

train, test = train_test_split(freq)

num_series = 1  # train.unique_id.nunique() # for filtering

# Training parameters
# Define embedding dimension
embedding_dim = 64
epochs = 10
batch_size = 128
criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss(beta=1.0)
output_size = 1
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# filtering series
filtered_series = train["unique_id"].unique()[:num_series]
train = train[train["unique_id"].isin(filtered_series)]
test = test[test["unique_id"].isin(filtered_series)]

# Set parameters based on frequency
if freq == 'Daily':
    look_back = 30  # Number of previous time steps for input
    horizon = 14  # Forecast horizon
    max_length = 200  # For truncating series
elif freq == 'Hourly':
    look_back = 120
    horizon = 48
    max_length = None  # Do not truncate series
else:
    raise ValueError("Unsupported frequency. Choose 'Daily' or 'Hourly'.")

if max_length:
    train = truncate_series(train, max_length)

# Create train and test windows
X_train_xlstm, y_train_xlstm, scalers_xlstm = create_train_windows(train, look_back, horizon)
X_test_xlstm = create_test_windows(train, look_back, scalers_xlstm)

# Prepare data tensors
X_train_xlstm = X_train_xlstm.clone().detach().unsqueeze(-1).float().requires_grad_()  # Add feature dimension
y_train_xlstm = y_train_xlstm.clone().detach().unsqueeze(-1).float()
X_test_xlstm = X_test_xlstm.clone().detach().unsqueeze(-1).float().requires_grad_()  # Add feature dimension

# Move data to device
X_train_xlstm = X_train_xlstm.to(device)
y_train_xlstm = y_train_xlstm.to(device)
X_test_xlstm = X_test_xlstm.to(device)

# Instantiate model
xlstm_stack = get_stack_cfg(embedding_dim, look_back, device, fix_inits_bool=False)

# model checks and init
model = xLSTMTimeSeriesModel(xlstm_stack, output_size, embedding_dim).to(device)

# Train and evaluate xLSTM model
print("\nTraining and Evaluating xLSTM Model...")
y_pred_xlstm = train_and_predict(
    device,
    model,
    X_train_xlstm,
    y_train_xlstm,
    X_test_xlstm,
    scalers_xlstm,
    1,
    batch_size,
    horizon,
    test,
    freq,
    criterion
)

y_true = test['y'].values.reshape(num_series, horizon)

print('SMAPE:', round(calculate_smape(y_true, y_pred_xlstm), 2))
