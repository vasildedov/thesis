import json
import os
import pandas as pd

dataset = 'tourism'
# Define frequencies and corresponding weights
frequencies = ['monthly', 'quarterly', 'yearly']
num_series = [366, 427, 518]
all_series = sum(num_series)

# Construct the weights dictionary
weights = {freq: num / all_series for freq, num in zip(frequencies, num_series)}


# Function to calculate weighted mape and return a DataFrame with mape per frequency
def calculate_weighted_mape_and_df(model, model_type, dataset):
    metadatas = {}
    base_path = os.path.join(os.getcwd(), f'models/{dataset}')
    for freq in frequencies:
        folder = f'{model_type}_{freq}'
        metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadatas[freq] = json.load(f)

    # Initialize a DataFrame to store mape values
    mape_data = []

    # Calculate weighted mape
    total_mape = 0.0
    print(f"Calculating MAPE for {model}...")
    for freq, data in metadatas.items():
        mape = data['MAPE']
        weight = weights[freq]
        contribution = mape * weight
        print(f"Frequency: {freq}")
        print(f"MAPE: {mape}")
        print(f"Weighted Contribution: {contribution:.4f}")
        total_mape += contribution

        # Append to DataFrame
        mape_data.append({'freq': freq, 'mape': mape})

    # Create DataFrame for the model
    mape_df = pd.DataFrame(mape_data)
    print(f"Total Weighted mape for {model}: {total_mape:.4f}\n")

    return total_mape, mape_df


# Calculate mape for all models
results = {}
dfs = {}

models = {
    'lgbm': 'ml',
    'xgb': 'ml',
    'simplernn': 'dl',
    'complexlstm': 'dl',
    'timeseriestransformer': 'dl',
    'xlstm': 'dl',
    'sarima': 'stats',
    'arima': 'stats'
}

for model, model_type in models.items():
    total_mape, mape_df = calculate_weighted_mape_and_df(model, model_type, dataset+'/direct' if model_type=='dl' else dataset)
    results[model] = total_mape
    dfs[model] = mape_df

# Compare the results
print("\nComparison:")
for model, mape in results.items():
    print(f"{model.upper()} Total Weighted mape: {mape:.2f}")


# Optionally save all DataFrames to CSVs
output_folder = os.path.join(os.getcwd(), f'models/mape_results/{dataset}')
os.makedirs(output_folder, exist_ok=True)
for model, mape_df in dfs.items():
    output_path = os.path.join(output_folder, f"{model}_mape.csv")
    mape_df.to_csv(output_path, index=False)
    print(f"Saved mape DataFrame for {model} to {output_path}")
