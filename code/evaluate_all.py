import json
import os
import pandas as pd
from utils.evaluation import calculate_metrics_per_frequency, calculate_weighted_metrics


for dataset in ['m4', 'm3', 'tourism']:
    # args

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
    # overall metrics
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

    #
    # # Define the stats model parameters as a DataFrame
    # stats_params = pd.DataFrame({
    #     'Frequency': ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Other'],
    #     'Order (p, d, q)': [(2, 1, 1), (2, 1, 1), (2, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
    #     'Seasonal Order (P, D, Q, s)': [(0, 0, 0, 0), (0, 1, 1, 4), (0, 1, 1, 12), (0, 1, 1, 52), (0, 1, 1, 7), (0, 1, 1, 24), (0, 0, 0, 0)],
    # })
    #
    # # Display the DataFrame
    # print(stats_params)
    #
    # latex_table = stats_params.to_latex(index=False,
    #                               caption=f"Parameters for (S)ARIMA models based on frequency",
    #                               label=f"tab:stats_params",
    #                               float_format="%.2f"  # Ensures numbers have at most 2 decimal places
    #                               )
    #
    # output_folder = os.path.join(os.getcwd(), f'models/metrics_results')
    # os.makedirs(output_folder, exist_ok=True)
    # # Save LaTeX table to a file
    # output_path = os.path.join(output_folder, f"stats_params.tex")
    # with open(output_path, 'w') as f:
    #     f.write(latex_table)
    #


import os
import json
import pandas as pd
import scipy.stats as stats
from utils.evaluation import calculate_metrics_per_frequency, calculate_weighted_metrics


def compare_models_statistical_significance(metrics_df, metric, test_type='t-test'):
    """
    Perform pairwise statistical tests on model metrics.
    :param metrics_df: DataFrame containing metrics for each model and frequency.
    :param test_type: Statistical test to use ('t-test' or 'wilcoxon').
    :return: DataFrame with pairwise statistical test results.
    """
    results = []

    # Extract sMAPE values for both recursive and direct suffixes
    for suffix in metrics_df.columns.get_level_values('Suffix').unique():
        smape_df = metrics_df.xs(key=metric, axis=1, level='Metric')[suffix].dropna(how='all')

        # Get unique models from the index
        models = smape_df.index.get_level_values('Model').unique()

        for i, model_1 in enumerate(models):
            for model_2 in models[i + 1:]:
                # Extract sMAPE values for the two models
                smape_1 = smape_df.loc[(slice(None), model_1)]
                smape_2 = smape_df.loc[(slice(None), model_2)]

                # Align the data and drop NaN values
                aligned_data = pd.concat([smape_1, smape_2], axis=1, keys=[model_1, model_2]).dropna()

                if aligned_data.empty:
                    print(f"No data available for comparison between {model_1} and {model_2} for suffix '{suffix}'.")
                    continue

                # Perform the specified statistical test
                if test_type == 't-test':
                    stat, p_value = stats.ttest_rel(aligned_data[model_1], aligned_data[model_2], nan_policy='omit')
                elif test_type == 'wilcoxon':
                    stat, p_value = stats.wilcoxon(aligned_data[model_1], aligned_data[model_2], zero_method='pratt',
                                                   nan_policy='omit')
                else:
                    raise ValueError("Invalid test_type. Choose 't-test' or 'wilcoxon'.")

                results.append({
                    'Model 1': model_1,
                    'Model 2': model_2,
                    'Suffix': suffix,
                    'Test Statistic': stat,
                    'p-value': p_value
                })

    return pd.DataFrame(results)


# Main Execution
for dataset in ['m4', 'm3', 'tourism']:
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
        'xlstm_0_1': 'dl'
    }

    models_names_dict = {'arima': 'ARIMA', 'sarima': 'SARIMA', 'complexlstm': 'LSTM', 'simplernn': 'RNN',
                         'timeseriestransformer': 'Transformer', 'xgb': 'XGBoost', 'lgbm': 'LightGBM',
                         'xlstm_1_0': 'xLSTM(1:0)', 'xlstm_0_1': 'xLSTM(0:1)'}

    # Output folder for LaTeX results
    output_folder = os.path.join(os.getcwd(), f'models/metrics_results/{dataset}/latex_significance_tests')
    os.makedirs(output_folder, exist_ok=True)

    # Calculate metrics per frequency
    metrics_per_frequency = calculate_metrics_per_frequency(models, dataset, frequencies)

    # Perform statistical significance tests
    test_results = compare_models_statistical_significance(metrics_per_frequency, 'sMAPE')

    # Save test results to LaTeX
    latex_table = test_results.to_latex(index=False,
                                        caption=f"Statistical Significance Tests for {dataset.capitalize()} Dataset",
                                        label=f"tab:significance_{dataset}",
                                        float_format="%.4f")

    output_path = os.path.join(output_folder, f"significance_tests.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"Saved statistical significance results for {dataset} to {output_path}")


def compare_recursive_vs_direct(metrics_df, test_type='t-test'):
    """
    Compare recursive vs direct forecasting approaches across all models and frequencies.
    :param metrics_df: DataFrame with MultiIndex columns ['Suffix', 'Metric'] containing metrics like sMAPE.
    :param test_type: Statistical test to use ('t-test' or 'wilcoxon').
    :return: Test statistic and p-value for recursive vs direct comparison.
    """
    # Extract recursive and direct sMAPE values
    recursive_smape = metrics_df.xs(key=('recursive', 'sMAPE'), axis=1, drop_level=False)
    direct_smape = metrics_df.xs(key=('direct', 'sMAPE'), axis=1, drop_level=False)

    # Group by Frequency and calculate mean
    recursive_smape_mean = recursive_smape.groupby(level='Frequency').mean()
    direct_smape_mean = direct_smape.groupby(level='Frequency').mean()

    # Align data for paired comparison
    aligned_data = pd.concat([recursive_smape_mean, direct_smape_mean], axis=1, keys=['Recursive', 'Direct']).dropna()

    if aligned_data.empty:
        raise ValueError("No data available for recursive vs direct comparison.")

    # Perform the specified statistical test
    if test_type == 't-test':
        stat, p_value = stats.ttest_rel(aligned_data['Recursive'], aligned_data['Direct'], nan_policy='omit')
    elif test_type == 'wilcoxon':
        stat, p_value = stats.wilcoxon(aligned_data['Recursive'], aligned_data['Direct'], zero_method='pratt', nan_policy='omit')
    else:
        raise ValueError("Invalid test_type. Choose 't-test' or 'wilcoxon'.")

    return stat, p_value

# Perform recursive vs direct comparison
stat, p_value = compare_recursive_vs_direct(metrics_per_frequency, test_type='t-test')

# Output results
print(f"Test Statistic: {stat}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between recursive and direct forecasting approaches.")
else:
    print("No significant difference between recursive and direct forecasting approaches.")











def compare_recursive_vs_direct_across_datasets(datasets, models, frequencies, weights_func, calculate_metrics_func, test_type='t-test'):
    """
    Compare recursive vs direct forecasting approaches across multiple datasets.
    :param datasets: List of dataset names.
    :param models: Dictionary of models with their types.
    :param frequencies: List of frequencies for each dataset.
    :param weights_func: Function to calculate weights for each dataset.
    :param calculate_metrics_func: Function to calculate metrics per frequency for each dataset.
    :param test_type: Statistical test to use ('t-test' or 'wilcoxon').
    :return: Test statistic and p-value for recursive vs direct comparison across datasets.
    """
    combined_metrics = []

    # Process each dataset
    for dataset in datasets:
        # Calculate weights for the current dataset
        weights = weights_func(dataset)

        # Calculate metrics per frequency for the current dataset
        metrics_per_frequency = calculate_metrics_func(models, dataset, frequencies, weights)

        # Add a 'Dataset' level to the index
        metrics_per_frequency['Dataset'] = dataset
        metrics_per_frequency = metrics_per_frequency.set_index('Dataset', append=True)

        # Append to combined metrics
        combined_metrics.append(metrics_per_frequency)

    # Concatenate all metrics into a single DataFrame
    combined_metrics_df = pd.concat(combined_metrics)

    # Extract recursive and direct sMAPE values
    recursive_smape = combined_metrics_df.xs(key=('recursive', 'sMAPE'), axis=1, drop_level=False)
    direct_smape = combined_metrics_df.xs(key=('direct', 'sMAPE'), axis=1, drop_level=False)

    # Align and group by Dataset and Frequency
    recursive_mean = recursive_smape.groupby(level=['Dataset', 'Frequency']).mean()
    direct_mean = direct_smape.groupby(level=['Dataset', 'Frequency']).mean()

    # Align data for paired comparison
    aligned_data = pd.concat([recursive_mean, direct_mean], axis=1, keys=['Recursive', 'Direct']).dropna()

    if aligned_data.empty:
        raise ValueError("No data available for recursive vs direct comparison across datasets.")

    # Perform the specified statistical test
    if test_type == 't-test':
        stat, p_value = stats.ttest_rel(aligned_data['Recursive'], aligned_data['Direct'], nan_policy='omit')
    elif test_type == 'wilcoxon':
        stat, p_value = stats.wilcoxon(aligned_data['Recursive'], aligned_data['Direct'], zero_method='pratt', nan_policy='omit')
    else:
        raise ValueError("Invalid test_type. Choose 't-test' or 'wilcoxon'.")

    return stat, p_value

# Define datasets and other parameters
datasets = ['m4', 'm3', 'tourism']
models = {
    'arima': 'stats',
    'sarima': 'stats',
    'lgbm': 'ml',
    'xgb': 'ml',
    'simplernn': 'dl',
    'complexlstm': 'dl',
    'timeseriestransformer': 'dl',
    'xlstm_1_0': 'dl',
    'xlstm_0_1': 'dl'
}

# Example weight function
def calculate_weights(dataset):
    if dataset == 'm3':
        frequencies = ['other', 'monthly', 'quarterly', 'yearly']
        num_series = [174, 1428, 756, 645]
    elif dataset == 'm4':
        frequencies = ['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        num_series = [414, 4227, 359, 48000, 24000, 23000]
    else:  # 'tourism'
        frequencies = ['monthly', 'quarterly', 'yearly']
        num_series = [366, 427, 518]

    all_series = sum(num_series)
    return {freq: num / all_series for freq, num in zip(frequencies, num_series)}

# Perform the comparison
stat, p_value = compare_recursive_vs_direct_across_datasets(
    datasets=datasets,
    models=models,
    frequencies=['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'],  # Adjust as needed
    weights_func=calculate_weights,
    calculate_metrics_func=calculate_metrics_per_frequency,  # Replace with your actual function
    test_type='t-test'
)

# Output results
print(f"Test Statistic: {stat}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between recursive and direct approaches across datasets.")
else:
    print("No significant difference between recursive and direct approaches across datasets.")
