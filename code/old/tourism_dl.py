import os
import torch
import torch.nn as nn
from datetime import datetime
import json
from utils.preprocess_tourism import train_test_split
from utils.preprocess_dl import create_train_windows, create_test_windows
from utils.models_dl import ComplexLSTM, SimpleRNN, TimeSeriesTransformer, xLSTMTimeSeriesModel
from utils.train_dl import train_and_predict, predict, load_existing_model
from utils.train_dl_xlstm import get_stack_cfg
from utils.helper import save_metadata, calculate_smape, calculate_mape
import numpy as np

torch.cuda.empty_cache()

# ===== Parameters =====
retrain_mode = False
direct = True  # direct or recursive prediction of horizon steps
freq = 'yearly'
embedding_dim = 64
epochs = 10
batch_size = 256
criterion = nn.MSELoss()  # Can use nn.SmoothL1Loss(beta=1.0) as alternative
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Data =====
train, test, horizon = train_test_split(freq)
look_back = 2*horizon if not freq=='yearly' else 7
lstm_hidden_size = 100

# Create datasets
X_train, y_train, scalers = create_train_windows(train, look_back, horizon, direct=direct)
X_test = create_test_windows(train, look_back, scalers)
X_train = X_train.unsqueeze(-1).to(device)  # Add feature dimension
X_test = X_test.unsqueeze(-1).to(device)
y_train = y_train.to(device)

# ===== Models and Configurations =====
models = [
    ("ComplexLSTM", ComplexLSTM, {"input_size": 1, "hidden_size": lstm_hidden_size, "num_layers": 3, "dropout": 0.3, "output_size": horizon if direct else 1, "direct": direct}),
    ("SimpleRNN", SimpleRNN, {"input_size": 1, "hidden_size": lstm_hidden_size, "num_layers": 3, "dropout": 0.3, "output_size": horizon if direct else 1, "direct": direct}),
    ("TimeSeriesTransformer", TimeSeriesTransformer, {"input_size": 1, "d_model": 64, "nhead": 8, "num_layers": 3, "dim_feedforward": 128, "dropout": 0.1, "output_size": horizon if direct else 1, "direct": direct}),
    ("xLSTM", xLSTMTimeSeriesModel, {"input_size": 1, "output_size": horizon if direct else 1, "embedding_dim": embedding_dim,  "direct": direct})
]

ensemble_predictions = []

# Define the folder to save all models
ending = 'direct' if direct else 'recursive'
model_folder = f"models/tourism/{ending}/dl_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)

# ===== Train and Evaluate Models =====
for model_name, model_class, model_kwargs in models:
    print(f"\nTraining and Evaluating {model_name}...")

    # Model-specific configurations
    model_path = f"{model_folder}{model_name.lower()}.pth"
    metadata_path = model_path.replace(".pth", "_metadata.json")

    if model_name == "xLSTM":
        xlstm_stack = get_stack_cfg(embedding_dim, look_back, device, fix_inits_bool=False)
        model_kwargs["xlstm_stack"] = xlstm_stack

    # Check for existing model
    model = load_existing_model(model_path, device, model_class, model_kwargs) if not retrain_mode else None
    y_true = test['y'].values.reshape(test['unique_id'].nunique(), horizon)

    if model is None:
        print(f"No existing model found or retrain mode was enabled. Training a new {model_name}...")
        model = model_class(**model_kwargs).to(device)

        # Train and evaluate
        y_pred, duration = train_and_predict(
            device,
            model,
            X_train,
            y_train,
            X_test,
            scalers,
            epochs,
            batch_size,
            horizon,
            test,
            criterion,
            direct=direct
        )

        print(f"Training completed in {duration:.2f} seconds")

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"{model_name} saved to {model_path}")

        # Calculate SMAPE and MAPE and save metadata
        smape = round(calculate_smape(y_true, y_pred), 2)
        mape = round(calculate_mape(y_true, y_pred), 2)
        print(f'sMAPE: {smape}, MAPE: {mape}')
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
            "MAPE": mape,
            "model_path": model_path,
            "time_to_train": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }
        save_metadata(metadata, metadata_path)
        print(f"{model_name} Metadata saved to {metadata_path}")
    else:
        print(f"Loaded {model_name} from {model_path}. Performing inference...")
        # Predict using recursive forecasting
        series_ids = test["unique_id"].unique()
        num_series = len(series_ids)
        with torch.no_grad():
            y_pred = predict(model, X_test, horizon, device, scalers, series_ids,
                             2500 if num_series > 2500 else num_series, direct=direct)
            smape = round(calculate_smape(y_true, y_pred), 2)
            mape = round(calculate_mape(y_true, y_pred), 2)
        print(f"{model_name} SMAPE from loaded model: {smape}")
        print(f"{model_name} MAPE from loaded model: {mape}")

    # Collect predictions for ensemble
    ensemble_predictions.append(y_pred)

# ===== Simple Average Ensemble =====
print("\nCalculating Simple Average Ensemble...")
ensemble_avg = np.mean(ensemble_predictions, axis=0)
ensemble_smape = round(calculate_smape(y_true, ensemble_avg), 2)
ensemble_mape = round(calculate_mape(y_true, ensemble_avg), 2)
print(f"Simple Average Ensemble SMAPE: {ensemble_smape}")
print(f"Simple Average Ensemble MAPE: {ensemble_mape}")

# Model-specific configurations
model_path = f"{model_folder}ensemble.pth"
metadata_path = model_path.replace(".pth", "_metadata.json")
metadata = {
            "model_name": 'ensemble',
            "frequency": freq.lower(),
            "look_back": look_back,
            "horizon": horizon,
            "batch_size": batch_size,
            "epochs": epochs,
            "criterion": str(criterion),
            "device": str(device),
            "SMAPE": ensemble_smape,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat()
        }
save_metadata(metadata, metadata_path)
