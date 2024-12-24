from datetime import datetime
import json
import torch
import torch.nn as nn

from utils.m4_preprocess import train_test_split, truncate_series
from utils.preprocess_dl import create_train_windows, create_test_windows
from utils.models_dl import xLSTMTimeSeriesModel
from utils.train_dl_xlstm import get_stack_cfg
from utils.train_dl import train_and_predict
from utils.helper import load_existing_model, save_metadata, calculate_smape

# args
freq = 'Hourly'  # or 'Daily'
num_series = 4  # train.unique_id.nunique()
embedding_dim = 64
epochs = 10
batch_size = 128
criterion = nn.MSELoss()  # nn.SmoothL1Loss(beta=1.0)
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train, test = train_test_split(freq)
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


# get model configuration
xlstm_stack = get_stack_cfg(embedding_dim, look_back, device, fix_inits_bool=False)

# load up model
model_path = f'models/xlstm_{freq.lower()}_{num_series}_series.pth'
model = load_existing_model(model_path, device, xLSTMTimeSeriesModel, {'xlstm_stack': xlstm_stack,
                                                               'output_size': output_size, 'embedding_dim': embedding_dim})
# define true values
y_true = test['y'].values.reshape(num_series, horizon)
X_test_xlstm = X_test_xlstm.to(device)

if not model:
    print(f'Model at {model_path} not found. Training new model...')
    # Move data to device
    X_train_xlstm = X_train_xlstm.to(device)
    y_train_xlstm = y_train_xlstm.to(device)

    # model checks and init
    model = xLSTMTimeSeriesModel(xlstm_stack, output_size, embedding_dim).to(device)

    # Train and evaluate xLSTM model
    print("\nTraining and Evaluating xLSTM Model...")
    y_pred_xlstm, duration = train_and_predict(
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
        criterion
    )

    # Log the duration
    print(f"Training completed in {duration:.2f} seconds")

    smape = round(calculate_smape(y_true, y_pred_xlstm), 2)
    print('sMAPE:', smape)

    # Save the model
    model_path = f'models/xlstm_{freq.lower()}_{num_series}_series.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metadata = {"model_name": "xLSTMTimeSeriesModel", "frequency": freq.lower(), "embedding_dim": embedding_dim,
                "look_back": look_back, "horizon": horizon, "batch_size": batch_size, "epochs": epochs,
                "criterion": str(criterion), "device": str(device), "SMAPE": smape, "model_path": model_path,
                "timestamp": datetime.now().isoformat(), "num_series": num_series, "series": filtered_series.tolist(),
                "time_to_train": round(duration, 2)}

    # Save metadata to JSON
    save_metadata(metadata, model_path.split('.')[0]+'_metadata.json')

else:
    from utils.train_dl import recursive_predict
    # for inferences
    with torch.no_grad():  # Disable gradient computation
        predictions = recursive_predict(model, X_test_xlstm, horizon, device, scalers_xlstm)
        print("Inferences from loaded model made successfully.")

    preds = predictions.reshape(num_series, horizon)

    smape = round(calculate_smape(y_true, preds), 2)
    print('sMAPE:', smape)
