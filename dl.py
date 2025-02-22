import os
import torch
import numpy as np
from datetime import datetime
from utils.preprocess_dl import create_train_windows, create_test_windows
from utils.models_dl import ComplexLSTM, SimpleRNN, TimeSeriesTransformer, xLSTMTimeSeriesModel
from utils.train_dl import train_and_save, predict, load_existing_model
from utils.train_dl_xlstm import get_stack_cfg
from utils.helper import save_metadata, evaluate

torch.cuda.empty_cache()

# ===== Dataset =====
dataset = 'etth2'
freq = 'hourly'.capitalize()
multivariate = False

if dataset == 'm3':
    from utils.preprocess_m3 import train_test_split
elif dataset == 'm4':
    from utils.preprocess_m4 import train_test_split
elif dataset == 'tourism':
    from utils.preprocess_tourism import train_test_split
    freq = freq.lower()
else:
    from utils.preprocess_ett import train_test_split, get_windows
    freq = 'default'
    multivariate = True

# ===== Parameters =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrain_mode = False
full_load = True
direct = True  # direct or recursive prediction of horizon steps

epochs = 10
batch_size = 32
num_layers = 3
dropout = 0.1
criterion = torch.nn.MSELoss()  # Can use nn.SmoothL1Loss(beta=1.0) as alternative

# rnns and lstms
hidden_size = 128  # xlstm embedding_dim and transformers d_model are 1/2 of lstm_hidden_size
# transformers
d_model = 64
n_heads = 8
dim_feedforward = 128
# xlstm
embedding_dim = 64

# ===== Load Data =====
if not multivariate:
    train, test, horizon = train_test_split(freq)
    look_back = 2*horizon if not (dataset == 'tourism' and freq == 'yearly') else 7
else:
    train, val, test = train_test_split(group=dataset)
    look_back = 720
    horizon = 24
output_size = horizon if direct else 1

if not full_load:
    num_series = 10
    filtered_series = train["unique_id"].unique()[:num_series]
    train = train[train["unique_id"].isin(filtered_series)]
    test = test[test["unique_id"].isin(filtered_series)]

# Create datasets
if not multivariate:
    X_train, y_train, scalers = create_train_windows(train, look_back, horizon, direct=direct)
    X_test = create_test_windows(train, look_back, scalers)
    X_train = X_train.unsqueeze(-1).to(device)  # Add feature dimension
    X_test = X_test.unsqueeze(-1).to(device)
    y_train = y_train.to(device)
else:
    X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train, val, test, look_back=look_back, horizon=horizon)
    X_train = torch.tensor(X_train).to(device).float()  # Add feature dimension
    y_train = torch.tensor(y_train).to(device).float()
    X_test = torch.tensor(X_test).to(device).float()  # Add feature dimension
    train['unique_id'] = dataset
    test['unique_id'] = dataset
    scalers = None
input_size = 1 if not multivariate else X_train.shape[-1]


# ===== Models and Configurations =====
models = [
    ("SimpleRNN", SimpleRNN,
     {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
      "output_size": output_size, "direct": direct}),
    ("ComplexLSTM", ComplexLSTM,
     {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
      "output_size": output_size, "direct": direct}),
    ("TimeSeriesTransformer", TimeSeriesTransformer,
     {"input_size": input_size, "d_model": d_model, "nhead": n_heads, "num_layers": num_layers,
      "dim_feedforward": dim_feedforward, "dropout": dropout, "output_size": output_size, "direct": direct}),
    ("xLSTM_1_0", xLSTMTimeSeriesModel,
     {"input_size": input_size, "output_size": output_size, "embedding_dim": embedding_dim, "direct": direct,
      "xlstm_stack": get_stack_cfg(embedding_dim, look_back, device, dropout, ratio='1_0')}),
    ("xLSTM_1_1", xLSTMTimeSeriesModel,
     {"input_size": input_size, "output_size": output_size, "embedding_dim": embedding_dim, "direct": direct,
      "xlstm_stack": get_stack_cfg(embedding_dim, look_back, device, dropout, ratio='1_1')}),
    ("xLSTM_0_1", xLSTMTimeSeriesModel,
     {"input_size": input_size, "output_size": output_size, "embedding_dim": embedding_dim, "direct": direct,
      "xlstm_stack": get_stack_cfg(embedding_dim, look_back, device, dropout, ratio='0_1')}),
]

# ensemble_predictions = []

# Define the folder to save all models
ending = 'direct' if direct else 'recursive'
model_folder = f"models/{dataset}/{ending}/dl_{freq.lower()}/"
os.makedirs(model_folder, exist_ok=True)

# ===== Train and Evaluate Models =====
for model_name, model_class, model_kwargs in models:
    print(f"\nTraining and Evaluating {model_name}...")

    # Model-specific configurations
    model_path = f"{model_folder}{model_name.lower()}.pth" if full_load else \
        f"{model_folder}{model_name.lower()}_{num_series}_series.pth"
    metadata_path = model_path.replace(".pth", "_metadata.json")

    # Check for existing model
    model = load_existing_model(model_path, device, model_class, model_kwargs) if not retrain_mode else None
    y_true = test['y'].values.reshape(num_series if not full_load else train['unique_id'].nunique(), horizon) if not multivariate else y_test.copy()

    if model is None:
        print(f"No existing model found or retrain mode was enabled for {dataset} - {freq} - {ending}. Training a new {model_name}...")
        model = model_class(**model_kwargs).to(device)

        # Train and evaluate
        model, duration = train_and_save(
            device,
            lambda: model,
            X_train,
            y_train,
            epochs,
            batch_size,
            criterion,
            direct=direct,
            model_path=model_path
        )
        print(f"Training completed in {duration:.2f} seconds")
    else:
        duration = None
        print(f"Loaded {model_name} from {model_path}. Performing inference...")

    # Predict either way
    series_ids = test["unique_id"].unique()
    len_series = len(series_ids)
    with torch.no_grad():
        y_pred = predict(model, X_test, horizon, device, scalers, series_ids,
                                   2500 if len_series > 2500 else len_series, direct=direct, multivariate=multivariate)

        evaluation = evaluate(y_true, y_pred)

    if duration:
        metadata = {
            "model_name": model_name,
            "frequency": freq.lower(),
            "look_back": look_back,
            "horizon": horizon,
            "batch_size": batch_size,
            "epochs": epochs,
            "criterion": str(criterion),
            "device": str(device),
            **evaluation,
            "model_path": model_path,
            "time_to_train": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
            "num_series": num_series if not full_load else train['unique_id'].nunique(),
            "model_kwargs": str(model_kwargs)
        }
        save_metadata(metadata, metadata_path)
        print(f"{model_name} Metadata saved to {metadata_path}")


# ===== Simple Average Ensemble =====
# print("\nCalculating Simple Average Ensemble...")
# ensemble_avg = np.mean(ensemble_predictions, axis=0)
# ensemble_smape = round(calculate_smape(y_true, ensemble_avg), 2)
# print(f"Simple Average Ensemble SMAPE: {ensemble_smape}")
#
# # Model-specific configurations
# model_path = f"{model_folder}ensemble.pth"
# metadata_path = model_path.replace(".pth", "_metadata.json")
# metadata = {
#             "model_name": 'ensemble',
#             "frequency": freq.lower(),
#             "look_back": look_back,
#             "horizon": horizon,
#             "batch_size": batch_size,
#             "epochs": epochs,
#             "criterion": str(criterion),
#             "device": str(device),
#             "SMAPE": ensemble_smape,
#             "model_path": model_path,
#             "timestamp": datetime.now().isoformat()
#         }
# save_metadata(metadata, metadata_path)
