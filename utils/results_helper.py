import os
import json
import pandas as pd
import scipy.stats as stats
import numpy as np
from tabulate import tabulate


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


def calculate_weighted_metrics(models, dataset, frequencies, weights, include_all=False, coalesce_suffixes=False):
    data = []
    suffixes = ['recursive', 'direct']

    for model, model_type in models.items():
        # Initialize storage for metrics
        metric_sums = {"sMAPE": float("nan"), "Training Time (s)": float("nan")}
        if include_all:
            metric_sums.update({"MAPE": float("nan"), "MAE": float("nan"), "MSE": float("nan")})

        # Separate storage if keeping suffixes
        separate_metrics = {suffix: {} for suffix in suffixes} if not coalesce_suffixes else None

        for suffix in suffixes:
            base_path = os.path.join(os.getcwd(), f'models/{dataset}/{suffix}')

            smape_tmp, training_time_tmp = float("nan"), float("nan")
            mape_tmp, mae_tmp, mse_tmp = (float("nan"), float("nan"), float("nan")) if include_all else (
            None, None, None)

            for freq in frequencies:
                smape, mape, mae, mse, training_time = load_metadata(base_path, model, model_type, freq)
                weight = weights.get(freq, 0)

                if not pd.isna(smape):
                    smape_tmp = smape_tmp + (smape * weight) if not pd.isna(smape_tmp) else (smape * weight)
                if not pd.isna(training_time):
                    training_time_tmp = training_time_tmp + (training_time * weight) if not pd.isna(
                        training_time_tmp) else (training_time * weight)

                if include_all:
                    if not pd.isna(mape):
                        mape_tmp = mape_tmp + (mape * weight) if not pd.isna(mape_tmp) else (mape * weight)
                    if not pd.isna(mae):
                        mae_tmp = mae_tmp + (mae * weight) if not pd.isna(mae_tmp) else (mae * weight)
                    if not pd.isna(mse):
                        mse_tmp = mse_tmp + (mse * weight) if not pd.isna(mse_tmp) else (mse * weight)

            if coalesce_suffixes:
                # Use the first non-NaN value
                metric_sums["sMAPE"] = smape_tmp if pd.isna(metric_sums["sMAPE"]) else metric_sums["sMAPE"]
                metric_sums["Training Time (s)"] = training_time_tmp if pd.isna(metric_sums["Training Time (s)"]) else \
                metric_sums["Training Time (s)"]

                if include_all:
                    metric_sums["MAPE"] = mape_tmp if pd.isna(metric_sums["MAPE"]) else metric_sums["MAPE"]
                    metric_sums["MAE"] = mae_tmp if pd.isna(metric_sums["MAE"]) else metric_sums["MAE"]
                    metric_sums["MSE"] = mse_tmp if pd.isna(metric_sums["MSE"]) else metric_sums["MSE"]
            else:
                # Store separately
                separate_metrics[suffix] = {
                    "sMAPE": smape_tmp,
                    "Training Time (s)": training_time_tmp
                }
                if include_all:
                    separate_metrics[suffix].update({
                        "MAPE": mape_tmp,
                        "MAE": mae_tmp,
                        "MSE": mse_tmp
                    })

        # Convert collected metrics into DataFrame rows
        if coalesce_suffixes:
            for metric_name, value in metric_sums.items():
                data.append({"Model": model, "Metric": metric_name, "Value": value})
        else:
            for suffix, metrics in separate_metrics.items():
                for metric_name, value in metrics.items():
                    data.append({"Model": model, "Suffix": suffix, "Metric": metric_name, "Value": value})

    # Convert to DataFrame
    metrics_df = pd.DataFrame(data)


    if coalesce_suffixes:
        # Pivot without 'Suffix'
        metrics_df = metrics_df.pivot(index='Model', columns='Metric', values='Value')

        # Reorder columns
        ordered_columns = ['sMAPE', 'Training Time (s)']
        if include_all:
            ordered_columns.extend(['MAPE', 'MAE', 'MSE'])

        # Keep only available columns (in case some are missing)
        ordered_columns = [col for col in ordered_columns if col in metrics_df.columns]
        metrics_df = metrics_df[ordered_columns]

    else:
        metrics_df = metrics_df.pivot(index='Model', columns=['Suffix', 'Metric'], values='Value')

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
                smape, mape, mae, mse, training_time = load_metadata(base_path, model, model_type, freq)

                data.extend([
                    {'Frequency': freq, 'Model': model, 'Suffix': suffix, 'Metric': 'sMAPE', 'Value': smape},
                    {'Frequency': freq, 'Model': model, 'Suffix': suffix, 'Metric': 'Training Time (s)', 'Value': training_time}
                ])

    metrics_df = pd.DataFrame(data).pivot(index=['Frequency', 'Model'], columns=['Suffix', 'Metric'], values='Value')
    return metrics_df.reindex(models.keys(), level='Model')


def stats_test_table(data_A, data_B, metric='sMAPE'):
    t_stat, p_value = stats.ttest_rel(data_A, data_B)

    # Compute means and standard deviations
    mean_A, std_A = np.mean(data_A), np.std(data_A, ddof=1)
    mean_B, std_B = np.mean(data_B), np.std(data_B, ddof=1)

    # Create DataFrame for the results
    table_data = pd.DataFrame({
        "Metric": [metric],
        "Model A (Mean ± SD)": [f"{mean_A:.2f} ± {std_A:.2f}"],
        "Model B (Mean ± SD)": [f"{mean_B:.2f} ± {std_B:.2f}"],
        "t-statistic": [f"{t_stat:.2f}"],
        "p-value": [f"{p_value:.2f}"]})

    # Print the table
    print("\nStatistical Comparison of Model Performance:\n")
    print(tabulate(table_data, headers='keys', tablefmt='grid'))
    return table_data

