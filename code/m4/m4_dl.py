import torch
import torch.nn as nn
from datetime import datetime
import json
from utils.m4_preprocess import train_test_split, truncate_series
from utils.m4_preprocess_dl import create_train_windows, create_test_windows
from utils.models_dl import ComplexLSTM, SimpleRNN, TimeSeriesTransformer, xLSTMTimeSeriesModel
from utils.m4_train_dl import train_and_predict, recursive_predict
from utils.m4_train_xlstm import get_stack_cfg
from utils.helper import load_existing_model, save_metadata, calculate_smape
from datasetsforecast.m4 import M4Evaluation

# ===== Parameters =====
retrain_mode = True
full_load = True
freq = 'Yearly'
embedding_dim = 64
epochs = 10
batch_size = 32
criterion = nn.MSELoss()  # Can use nn.SmoothL1Loss(beta=1.0) as alternative
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if freq == 'Yearly':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 12, 6, None, 23000 if full_load else 10, 16
elif freq == 'Quarterly':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 16, 8, None, 24000 if full_load else 10, 50
elif freq == 'Monthly':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 36, 18, 120, 48000 if full_load else 10, 50
elif freq == 'Weekly':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 26, 13, 260, 359 if full_load else 10, 64
elif freq == 'Daily':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 28, 14, 200, 4227 if full_load else 10, 50
elif freq == 'Hourly':
    look_back, horizon, max_length, num_series, lstm_hidden_size = 96, 48, None, 414 if full_load else 10, 100
else:
    raise ValueError("Unsupported frequency. Choose from 'Yearly', 'Quarterly', 'Monthly', 'Daily', or 'Hourly'.")

# ===== Load Data =====
train, test = train_test_split(freq)
filtered_series = train["unique_id"].unique()[:num_series]
train = train[train["unique_id"].isin(filtered_series)]
test = test[test["unique_id"].isin(filtered_series)]

if max_length:
    train = truncate_series(train, max_length)

# Create datasets
X_train, y_train, scalers = create_train_windows(train, look_back, horizon)
X_test = create_test_windows(train, look_back, scalers)
X_train = X_train.unsqueeze(-1).to(device)  # Add feature dimension
X_test = X_test.unsqueeze(-1).to(device)
y_train = y_train.to(device)

# ===== Models and Configurations =====
models = [
    ("ComplexLSTM", ComplexLSTM, {"input_size": 1, "hidden_size": lstm_hidden_size, "num_layers": 3, "dropout": 0.3, "output_size": 1}),
    ("SimpleRNN", SimpleRNN, {"input_size": 1, "hidden_size": lstm_hidden_size, "num_layers": 3, "dropout": 0.3, "output_size": 1}),
    ("TimeSeriesTransformer", TimeSeriesTransformer, {"input_size": 1, "d_model": 64, "nhead": 8, "num_layers": 3, "dim_feedforward": 128, "dropout": 0.1, "output_size": 1}),
    ("xLSTM", xLSTMTimeSeriesModel, None)  # xLSTM requires additional configuration
]

# ===== Train and Evaluate Models =====
for model_name, model_class, model_kwargs in models:
    print(f"\nTraining and Evaluating {model_name}...")

    # Model-specific configurations
    model_path = f"models/dl_{freq.lower()}/{model_name.lower()}_{freq.lower()}_{num_series}_series.pth"
    metadata_path = model_path.replace(".pth", "_metadata.json")

    if model_name == "xLSTM":
        xlstm_stack = get_stack_cfg(embedding_dim, look_back, device, fix_inits_bool=False)
        model_kwargs = {"xlstm_stack": xlstm_stack, "output_size": output_size, "embedding_dim": embedding_dim}

    # Check for existing model
    model = load_existing_model(model_path, device, model_class, model_kwargs) if not retrain_mode else None
    y_true = test['y'].values.reshape(num_series, horizon)

    if model is None:
        print(f"No existing model found or retrain mode was enabled. Training a new {model_name}...")
        model = model_class(**model_kwargs).to(device)

        # Train and evaluate
        y_pred, duration = train_and_predict(
            device,
            lambda: model,
            X_train,
            y_train,
            X_test,
            scalers,
            epochs,
            batch_size,
            horizon,
            test,
            criterion
        )

        print(f"Training completed in {duration:.2f} seconds")

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"{model_name} saved to {model_path}")

        # Calculate SMAPE and save metadata
        smape = round(calculate_smape(y_true, y_pred), 2)
        print('sMAPE', smape)
        metadata = {
            "model_name": model_name,
            "frequency": freq.lower(),
            "look_back": look_back,
            "horizon": horizon,
            "batch_size": batch_size,
            "epochs": epochs,
            "criterion": str(criterion),
            "device": str(device),
            "SMAPE": smape,
            "model_path": model_path,
            "time_to_train": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
            "num_series": num_series
        }
        save_metadata(metadata, metadata_path)
        print(f"{model_name} Metadata saved to {metadata_path}")
    else:
        print(f"Loaded {model_name} from {model_path}. Performing inference...")
        with torch.no_grad():
            predictions = recursive_predict(model, X_test, horizon, device, scalers)
            smape = round(calculate_smape(y_true, predictions.reshape(num_series, horizon)), 2)
        print(f"{model_name} SMAPE from loaded model: {smape}")
