import json
import os
import time
import numpy as np
from utils.helpers import evaluate


def train_and_save_model(model, model_name, X_train, y_train, X_test, y_test, horizon, freq, look_back,
                         retrain=True, dataset='m4', direct=False):
    """
    Train a model, save it, and save its metadata.
    """
    if direct:
        sufix = 'direct'
    else:
        sufix = 'recursive'
    model_folder = f'models/{dataset}/{sufix}/ml_{freq.lower()}/'
    os.makedirs(model_folder, exist_ok=True)

    model_path = f'{model_folder}{model_name.lower()}.txt'
    metadata_path = model_path.replace('.txt', '_metadata.json')

    # Check if the model already exists
    if os.path.exists(model_path) and model_name != "LGBM" and not retrain:
        print(f"{model_name} model found at {model_path}. Loading existing model...")
        if model_name == "XGB":
            model.model.load_model(model_path)
        elif model_name == "CatBoost":
            model.model.load_model(model_path, format="json")
        print(f"{model_name} model loaded successfully.")
    else:
        print(f"No existing {model_name} model found. Training a new model...")
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        # Save model
        if model_name == "LGBM" and not direct:
            model.model.booster_.save_model(model_path)
        elif model_name == "XGB":
            model.model.save_model(model_path)
        elif model_name == "CatBoost":
            model.model.save_model(model_path, format="json")
        print(f"{model_name} model saved to {model_path}")

    # Predict and evaluate
    y_pred = recursive_predict(model, X_test, horizon) if not direct else model.predict(X_test)

    evaluation = evaluate(y_test, y_pred)

    if not os.path.exists(metadata_path) or retrain:
        # Save metadata
        metadata = {
            "model_name": model_name,
            "frequency": freq.lower(),
            "look_back": look_back,
            "horizon": horizon,
            "time_to_train": round(end_time - start_time, 2),
            **evaluation,
            "model_path": model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"{model_name} metadata saved to {metadata_path}")

    return y_pred, evaluation


# Recursive prediction function for multi-step prediction
def recursive_predict(model, X_input, horizon):
    predictions = []
    X_current = X_input
    for _ in range(horizon):
        y_pred = model.predict(X_current)
        predictions.append(y_pred)
        X_current = np.concatenate((X_current[:, 1:], y_pred.reshape(-1, 1)), axis=1)
    return np.hstack(predictions)


def ensemble_predict(models, X_input, horizon):
    model_predictions = [recursive_predict(model, X_input, horizon) for model in models]
    return np.mean(model_predictions, axis=0)
