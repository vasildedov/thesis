import os
import json
import pandas as pd


# Function to calculate weighted SMAPE and return a DataFrame with SMAPE per frequency
def calculate_weighted_smape_and_df(model, model_type, dataset, frequencies, weights):
    metadatas = {}
    base_path = os.path.join(os.getcwd(), f'models/{dataset}')
    for freq in frequencies:
        folder = f'{model_type}_{freq}'
        metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadatas[freq] = json.load(f)

    # Initialize a DataFrame to store SMAPE values
    smape_data = []

    # Calculate weighted SMAPE
    total_smape = 0.0
    print(f"Calculating SMAPE for {model}...")
    for freq, data in metadatas.items():
        smape = data['SMAPE']
        training_time = data['time_to_train']
        weight = weights[freq]
        contribution = smape * weight
        print(f"Frequency: {freq}")
        print(f"SMAPE: {smape}")
        print(f"Weighted Contribution: {contribution:.4f}")
        total_smape += contribution

        # Append to DataFrame
        smape_data.append({'freq': freq, 'smape': smape, 'training_time': training_time})

    # Create DataFrame for the model
    smape_df = pd.DataFrame(smape_data)
    print(f"Total Weighted SMAPE for {model}: {total_smape:.4f}\n")

    return total_smape, smape_df


# Function to calculate SMAPE per frequency for each model
def calculate_smape_per_frequency(models, dataset, frequencies, sufix):
    # Initialize a dictionary to store SMAPE data per frequency
    metrics_per_frequency = {freq: {} for freq in frequencies}

    for model, model_type in models.items():
        base_path = os.path.join(os.getcwd(), f'models/{dataset}/{sufix}' if model_type=='dl' else f'models/{dataset}')
        for freq in frequencies:
            folder = f'{model_type}_{freq}'
            metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    smape = metadata.get('SMAPE', float('nan'))
                    training_time = metadata.get('time_to_train', float('nan'))
                    metrics_per_frequency[freq][model] = {
                        'SMAPE': smape,
                        'Training Time': training_time
                    }
            else:
                metrics_per_frequency[freq][model] = {
                    'SMAPE': None,
                    'Training Time': None  # Indicate missing data
                }
    # Convert to DataFrame for better readability
    data = []
    for freq, models_data in metrics_per_frequency.items():
        for model, metrics in models_data.items():
            data.append({
                'Frequency': freq,
                'Model': model,
                'sMAPE': metrics['SMAPE'],
                'Training Time (s)': metrics['Training Time']
            })

    metrics_df = pd.DataFrame(data)
    return metrics_df
