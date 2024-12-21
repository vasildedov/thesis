import json
import os

# Define frequencies and corresponding weights
frequencies = ['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
num_series = [414, 4227, 359, 48000, 24000, 23000]
all_series = sum(num_series)
# Construct the weights dictionary
weights = {freq: num / all_series for freq, num in zip(frequencies, num_series)}


# Function to calculate weighted SMAPE for a given metadata type
def calculate_weighted_smape(model, model_type):
    datas = {}
    base_path = os.path.join(os.getcwd(), 'models')
    for freq in frequencies:
        folder = f'{model_type}_{freq}'
        metadata_path = os.path.join(base_path, folder, f'{model}_metadata.json')
        with open(metadata_path, 'r') as f:
            datas[freq] = json.load(f)

    # Calculate weighted SMAPE
    total_smape = 0.0
    print(f"Calculating SMAPE for {model}...")
    for freq, data in datas.items():
        smape = data['SMAPE']
        weight = weights[freq]
        contribution = smape * weight
        print(f"Frequency: {freq}")
        print(f"SMAPE: {smape}")
        print(f"Weighted Contribution: {contribution:.4f}")
        total_smape += contribution

    print(f"Total Weighted SMAPE for {model}: {total_smape:.4f}\n")
    return total_smape

# Calculate for both LGBM and XGB
lgbm_total_smape = calculate_weighted_smape('lgbm', 'ml')
xgb_total_smape = calculate_weighted_smape('xgb', 'ml')
simplernn_total_smape = calculate_weighted_smape('simplernn', 'dl')

# Compare the results
print(f"Comparison:")
print(f"LGBM Total Weighted SMAPE: {lgbm_total_smape:.4f}")
print(f"XGB Total Weighted SMAPE: {xgb_total_smape:.4f}")
