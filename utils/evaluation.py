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
        try:
            with open(metadata_path, 'r') as f:
                metadatas[freq] = json.load(f)
        except FileNotFoundError:
            print(f"Metadata file not found: {metadata_path}")
            metadatas[freq] = {'SMAPE': None, 'time_to_train': None}

    # Initialize a DataFrame to store SMAPE values
    smape_data = []

    # Calculate weighted SMAPE
    total_smape = 0.0
    print(f"Calculating SMAPE for {model}...")
    for freq, data in metadatas.items():
        smape = data['SMAPE']
        training_time = data['time_to_train']
        weight = weights[freq]
        contribution = smape * weight if not pd.isna(smape) else 0.0
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
def calculate_smape_per_frequency(models, dataset, frequencies):
    # Initialize a dictionary to store SMAPE data per frequency
    suffixes = ['direct', 'recursive']
    data = []

    for suffix in suffixes:
        for model, model_type in models.items():
            base_path = os.path.join(
                os.getcwd(),
                f'models/{dataset}/{suffix}'
            )
            for freq in frequencies:
                folder = f'{model_type}_{freq}'
                metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        smape = metadata.get('SMAPE', float('nan'))
                        training_time = metadata.get('time_to_train', float('nan'))
                except FileNotFoundError:
                    print(f"Metadata file not found: {metadata_path}")
                    smape = None
                    training_time = None

                data.append({
                    'Frequency': freq,
                    'Model': model,
                    'Suffix': suffix,
                    'Metric': 'sMAPE',
                    'Value': smape
                })
                data.append({
                    'Frequency': freq,
                    'Model': model,
                    'Suffix': suffix,
                    'Metric': 'Training Time (s)',
                    'Value': training_time
                })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(data)

    # Pivot table to create MultiIndex columns
    metrics_df = metrics_df.pivot(index=['Frequency', 'Model'], columns=['Suffix', 'Metric'], values='Value')
    # Sort the DataFrame by model order
    # Ensure the models are ordered based on the input dictionary
    ordered_models = list(models.keys())
    metrics_df = metrics_df.reindex(ordered_models, level='Model')

    return metrics_df
