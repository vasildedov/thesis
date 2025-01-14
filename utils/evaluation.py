import os
import json
import pandas as pd
import scipy.stats as stats


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


# Function to calculate pairwise statistical significance
def compare_models_statistical_significance(dfs, test_type='t-test'):
    """
    Perform pairwise statistical tests on model results.
    :param dfs: Dictionary of DataFrames containing sMAPE per frequency for each model.
    :param test_type: 't-test' for paired t-test, 'wilcoxon' for Wilcoxon signed-rank test.
    :return: DataFrame with pairwise comparison results.
    """
    models = list(dfs.keys())
    pairwise_results = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_1, model_2 = models[i], models[j]

            # Extract sMAPE values for both models
            smape_1 = dfs[model_1]['smape']
            smape_2 = dfs[model_2]['smape']

            # Ensure data is aligned by frequency
            aligned_data = pd.merge(smape_1, smape_2, left_index=True, right_index=True, suffixes=('_1', '_2'))

            # Drop rows with NaN values
            aligned_data = aligned_data.dropna()

            if aligned_data.empty:
                print(f"No data available for comparison between {model_1} and {model_2}.")
                continue

            # Perform the statistical test
            if test_type == 't-test':
                stat, p_value = stats.ttest_rel(aligned_data['smape_1'], aligned_data['smape_2'], nan_policy='omit')
            elif test_type == 'wilcoxon':
                stat, p_value = stats.wilcoxon(aligned_data['smape_1'], aligned_data['smape_2'], zero_method='pratt', nan_policy='omit')
            else:
                raise ValueError("Invalid test_type. Choose 't-test' or 'wilcoxon'.")

            pairwise_results.append({
                'Model 1': model_1,
                'Model 2': model_2,
                'Test Statistic': stat,
                'p-value': p_value
            })

    # Convert pairwise results to a DataFrame
    results_df = pd.DataFrame(pairwise_results)
    return results_df


def calculate_weighted_metrics(models, dataset, frequencies, weights):
    data = []

    # Iterate over suffixes: 'direct' and 'recursive'
    suffixes = ['recursive', 'direct']

    for suffix in suffixes:
        for model, model_type in models.items():
            smape_weighted_sum = 0.0
            training_time_weighted_sum = 0.0

            base_path = os.path.join(os.getcwd(), f'models/{dataset}/{suffix}')
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
                    smape = float('nan')
                    training_time = float('nan')

                weight = weights.get(freq, 0)
                smape_weighted_sum += (smape * weight) if not pd.isna(smape) else 0.0
                training_time_weighted_sum += (training_time * weight) if not pd.isna(training_time) else 0.0

            # Append results for this model and suffix
            data.append({
                'Model': model,
                'Suffix': suffix,
                'Metric': 'sMAPE',
                'Value': smape_weighted_sum
            })
            data.append({
                'Model': model,
                'Suffix': suffix,
                'Metric': 'Training Time (s)',
                'Value': training_time_weighted_sum
            })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(data)

    # Pivot table to create MultiIndex columns for Suffix and Metric
    metrics_df = metrics_df.pivot(index='Model', columns=['Suffix', 'Metric'], values='Value')

    # Ensure the models are ordered based on the input dictionary
    ordered_models = list(models.keys())
    metrics_df = metrics_df.reindex(ordered_models)

    return metrics_df


# Function to calculate SMAPE per frequency for each model
def calculate_smape_per_frequency(models, dataset, frequencies):
    # Initialize a dictionary to store SMAPE data per frequency
    suffixes = ['recursive', 'direct']
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
