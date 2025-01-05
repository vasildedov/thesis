import json
import os
import pandas as pd
from utils.evaluation import calculate_weighted_smape_and_df, calculate_smape_per_frequency

dataset = 'm3'
direct = False
if direct:
    sufix = 'direct'
else:
    sufix = 'recursive'
# Define frequencies and corresponding weights
if dataset == 'm3':
    frequencies = ['other', 'monthly', 'quarterly', 'yearly']
    num_series = [174, 1428, 756, 645]
elif dataset == 'm4':
    frequencies = ['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    num_series = [414, 4227, 359, 48000, 24000, 23000]
else:
    frequencies = ['monthly', 'quarterly', 'yearly']
    num_series = [366, 427, 518]

all_series = sum(num_series)

# Construct the weights dictionary
weights = {freq: num / all_series for freq, num in zip(frequencies, num_series)}

# Calculate SMAPE for all models
results = {}
dfs = {}

models = {
    'lgbm': 'ml',
    'xgb': 'ml',
    'simplernn': 'dl',
    'complexlstm': 'dl',
    'timeseriestransformer': 'dl',
    'xlstm': 'dl',
    # 'ensemble': 'dl',
    # 'sarima': 'stats',
    'arima': 'stats'
}

for model, model_type in models.items():
    total_smape, smape_df = calculate_weighted_smape_and_df(model, model_type,
                                                            dataset+'/'+sufix if model_type!='stats' else dataset,
                                                            frequencies, weights)
    results[model] = total_smape
    dfs[model] = smape_df

# Compare the results
print("\nComparison:")
for model, smape in results.items():
    print(f"{model.upper()} Total Weighted SMAPE: {smape:.2f}")


# # Optionally save all DataFrames to CSVs
# output_folder = os.path.join(os.getcwd(), f'models/smape_results/{dataset}')
# os.makedirs(output_folder, exist_ok=True)
# for model, smape_df in dfs.items():
#     output_path = os.path.join(output_folder, f"{model}_smape.csv")
#     smape_df.to_csv(output_path, index=False)
#     print(f"Saved SMAPE DataFrame for {model} to {output_path}")

# Calculate SMAPE per frequency
metrics_df = calculate_smape_per_frequency(models, dataset, frequencies)
# Display the DataFrame
print("SMAPE per Frequency for Each Model:")
print(metrics_df)

# Split metrics_df into separate DataFrames based on frequency
frequency_dfs = {freq: group.droplevel('Frequency') for freq, group in metrics_df.groupby('Frequency')}

# Define output folder for LaTeX files
output_folder = os.path.join(os.getcwd(), f'models/metrics_results/{dataset}/latex_by_frequency')
os.makedirs(output_folder, exist_ok=True)

# Save each frequency-specific DataFrame as a LaTeX file
for freq, df in frequency_dfs.items():
    # Reset index and prepare the DataFrame for LaTeX
    df = df.reset_index().fillna('N/A')

    # Convert DataFrame to LaTeX with multicolumn for MultiIndex columns
    latex_table = df.to_latex(index=False,
                              caption=f"Metrics for {freq.capitalize()} Frequency",
                              label=f"tab:metrics_{freq}",
                              multicolumn=True,
                              float_format="%.2f"  # Ensures numbers have at most 2 decimal places
                              )

    # Save LaTeX table to a file
    output_path = os.path.join(output_folder, f"metrics_{freq}.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"Saved LaTeX table for {freq} to {output_path}")
