import json
import os
import pandas as pd
from utils.helpers import calculate_metrics_per_frequency, calculate_weighted_metrics

all_results = {}
multivariate = False

for dataset in ['m4', 'm3', 'tourism', 'etth1', 'etth2']:
    # args

    # Define frequencies and corresponding weights
    if dataset == 'm3':
        frequencies = ['other', 'monthly', 'quarterly', 'yearly']
        num_series = [174, 1428, 756, 645]
    elif dataset == 'm4':
        frequencies = ['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        num_series = [414, 4227, 359, 48000, 24000, 23000]
    elif dataset == 'tourism':
        frequencies = ['monthly', 'quarterly', 'yearly']
        num_series = [366, 427, 518]
    else:
        frequencies = ['default']
        num_series = [1]
        multivariate = True

    all_series = sum(num_series)
    # Construct the weights dictionary
    weights = {freq: num / all_series for freq, num in zip(frequencies, num_series)}

    models = {
        'arima': 'stats',
        'sarima': 'stats',
        'lgbm': 'ml',
        'xgb': 'ml',
        'simplernn': 'dl',
        'complexlstm': 'dl',
        'timeseriestransformer': 'dl',
        'xlstm_1_0': 'dl',
        'xlstm_0_1': 'dl',
        'xlstm_1_1': 'dl'
        # 'ensemble': 'dl'
    }

    models_names_dict = {'arima': 'ARIMA', 'sarima': 'SARIMA', 'complexlstm': 'LSTM', 'simplernn': 'RNN',
                         'timeseriestransformer': 'Transformer', 'xgb': 'XGBoost', 'lgbm': 'LightGBM',
                         'xlstm_1_0': 'xLSTM(1:0)', 'xlstm_0_1': 'xLSTM(0:1)', 'xlstm_1_1': 'xLSTM(1:1)'}

    # Define output folder for LaTeX files
    output_folder = os.path.join(os.getcwd(), f'models/metrics_results/{dataset}/latex_by_frequency')
    os.makedirs(output_folder, exist_ok=True)

    # calculations
    if multivariate:
        metrics_per_dataset = calculate_weighted_metrics(models, dataset, frequencies, weights, include_all=True, coalesce_suffixes=True).reset_index()
        metrics_per_dataset = metrics_per_dataset.drop(columns=['MAPE', 'sMAPE'])
        metrics_per_dataset = metrics_per_dataset[['Model', 'MAE', 'MSE', 'Training Time (s)']]
    else:
        metrics_per_dataset = calculate_weighted_metrics(models, dataset, frequencies, weights).reset_index()
    print(metrics_per_dataset)

    # save to LateX
    metrics_per_dataset['Model'] = metrics_per_dataset['Model'].replace(models_names_dict)

    # Convert DataFrame to LaTeX with multicolumn for MultiIndex columns
    latex_table = metrics_per_dataset.to_latex(index=False,
                                               caption=f"Metrics for {dataset.capitalize()} dataset",
                                               label=f"tab:metrics_{dataset}",
                                               multicolumn=True,
                                               float_format="%.2f")  # Ensures numbers have at most 2 decimal places

    # Save LaTeX table to a file
    output_path = os.path.join(output_folder, f"metrics_overall.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"Saved LaTeX table for {dataset} to {output_path}")

    # metrics per frequency
    metrics_per_frequency = calculate_metrics_per_frequency(models, dataset, frequencies)
    # Display the DataFrame
    print("SMAPE per Frequency for Each Model:")
    print(metrics_per_frequency)

    all_results[dataset] = metrics_per_frequency

    # Split metrics_df into separate DataFrames based on frequency
    frequency_dfs = {freq: group.droplevel('Frequency') for freq, group in metrics_per_frequency.groupby('Frequency')}

    # Save each frequency-specific DataFrame as a LaTeX file
    for freq, df in frequency_dfs.items():
        # Reset index and prepare the DataFrame for LaTeX
        df = df.reset_index().fillna('N/A')
        df['Model'] = df['Model'].replace(models_names_dict)

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
