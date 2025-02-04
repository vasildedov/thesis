import os
import json
import pandas as pd
import scipy.stats as stats

def load_metadata(base_path, model, model_type, frequency):
    folder = f'{model_type}_{frequency}'
    metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')

    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
            smape = metadata.get('SMAPE', float('nan'))
            mape = metadata.get('MAPE', float('nan'))
            mae = metadata.get('MAE', float('nan'))
            mse = metadata.get('MSE', float('nan'))
            training_time = metadata.get('time_to_train', float('nan'))
    except FileNotFoundError:
        if model == 'sarima':  # Fallback to ARIMA if SARIMA is missing
            print(f"SARIMA metadata not found for {model} at {metadata_path}. Attempting ARIMA...")
            return load_metadata(base_path, 'arima', model_type, frequency)
        print(f"Metadata file not found: {metadata_path}")
        smape, mape, mae, mse, training_time = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    return smape, mape, mae, mse, training_time


def calculate_weighted_metrics(models, dataset, frequencies, weights, include_all=False):
    data = []

    # Iterate over suffixes: 'direct' and 'recursive'
    suffixes = ['recursive', 'direct']

    for suffix in suffixes:
        base_path = os.path.join(os.getcwd(), f'models/{dataset}/{suffix}')
        for model, model_type in models.items():
            smape_weighted_sum = 0.0
            if include_all:
                mape_weighted_sum = 0.0
                mae_weighted_sum = 0.0
                mse_weighted_sum = 0.0
            training_time_weighted_sum = 0.0

            for freq in frequencies:
                smape, mape, mae, mse, training_time = load_metadata(base_path, model, model_type, freq)
                weight = weights.get(freq, 0)
                smape_weighted_sum += (smape * weight) if not pd.isna(smape) else 0.0
                if include_all:
                    mape_weighted_sum += (mape * weight) if not pd.isna(mape) else 0.0
                    mae_weighted_sum += (mae * weight) if not pd.isna(mae) else 0.0
                    mse_weighted_sum += (mse * weight) if not pd.isna(mse) else 0.0
                training_time_weighted_sum += (training_time * weight) if not pd.isna(training_time) else 0.0

            # Append results for this model and suffix
            data.extend([
                {'Model': model, 'Suffix': suffix, 'Metric': 'sMAPE', 'Value': smape_weighted_sum},
                {'Model': model, 'Suffix': suffix, 'Metric': 'Training Time (s)', 'Value': training_time_weighted_sum}
            ])
            if include_all:
                data.extend([
                    {'Model': model, 'Suffix': suffix, 'Metric': 'MAPE', 'Value': mape_weighted_sum},
                    {'Model': model, 'Suffix': suffix, 'Metric': 'MAE', 'Value': mae_weighted_sum},
                    {'Model': model, 'Suffix': suffix, 'Metric': 'MSE', 'Value': mse_weighted_sum}])

    metrics_df = pd.DataFrame(data).pivot(index='Model', columns=['Suffix', 'Metric'], values='Value')
    return metrics_df.reindex(models.keys())


# Function to calculate SMAPE per frequency for each model
def calculate_metrics_per_frequency(models, dataset, frequencies):
    # Initialize a dictionary to store SMAPE data per frequency
    suffixes = ['recursive', 'direct']
    data = []

    for suffix in suffixes:
        base_path = os.path.join(os.getcwd(), f'models/{dataset}/{suffix}')

        for model, model_type in models.items():
            for freq in frequencies:
                smape, training_time = load_metadata(base_path, model, model_type, freq)

                data.extend([
                    {'Frequency': freq, 'Model': model, 'Suffix': suffix, 'Metric': 'sMAPE', 'Value': smape},
                    {'Frequency': freq, 'Model': model, 'Suffix': suffix, 'Metric': 'Training Time (s)', 'Value': training_time}
                ])

    metrics_df = pd.DataFrame(data).pivot(index=['Frequency', 'Model'], columns=['Suffix', 'Metric'], values='Value')
    return metrics_df.reindex(models.keys(), level='Model')
