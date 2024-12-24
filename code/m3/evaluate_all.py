import json
import os
import pandas as pd

dataset = 'm3'
# Define frequencies and corresponding weights
frequencies = ['other', 'monthly', 'quarterly', 'yearly']
num_series = [174, 1428, 756, 645]
all_series = sum(num_series)

# Construct the weights dictionary
weights = {freq: num / all_series for freq, num in zip(frequencies, num_series)}


# Function to calculate weighted SMAPE and return a DataFrame with SMAPE per frequency
def calculate_weighted_smape_and_df(model, model_type, dataset):
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
        weight = weights[freq]
        contribution = smape * weight
        print(f"Frequency: {freq}")
        print(f"SMAPE: {smape}")
        print(f"Weighted Contribution: {contribution:.4f}")
        total_smape += contribution

        # Append to DataFrame
        smape_data.append({'freq': freq, 'smape': smape})

    # Create DataFrame for the model
    smape_df = pd.DataFrame(smape_data)
    print(f"Total Weighted SMAPE for {model}: {total_smape:.4f}\n")

    return total_smape, smape_df


# Calculate SMAPE for all models
results = {}
dfs = {}

models = {
    'lgbm': 'ml',
    'xgb': 'ml',
    'simplernn': 'dl',
    'complexlstm': 'dl',
    'timeseriestransformer': 'dl',
    'xlstm': 'dl'
}

for model, model_type in models.items():
    total_smape, smape_df = calculate_weighted_smape_and_df(model, model_type, dataset)
    results[model] = total_smape
    dfs[model] = smape_df

# Compare the results
print("\nComparison:")
for model, smape in results.items():
    print(f"{model.upper()} Total Weighted SMAPE: {smape:.2f}")


# Optionally save all DataFrames to CSVs
output_folder = os.path.join(os.getcwd(), f'models/smape_results/{dataset}')
os.makedirs(output_folder, exist_ok=True)
for model, smape_df in dfs.items():
    output_path = os.path.join(output_folder, f"{model}_smape.csv")
    smape_df.to_csv(output_path, index=False)
    print(f"Saved SMAPE DataFrame for {model} to {output_path}")
