import json
import os
import pandas as pd
from utils.results_helper import calculate_metrics_per_frequency, calculate_weighted_metrics , stats_test_table
import scipy.stats as stats

all_results = {}

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

    if dataset == 'etth1':
        # Calculate metrics for the first dataset (etth1)
        metrics_per_dataset = calculate_weighted_metrics(models, 'etth1', frequencies, weights, coalesce_suffixes=True)
        metrics_per_dataset.columns = pd.MultiIndex.from_product([['etth1'], metrics_per_dataset.columns])

        # Calculate metrics for the second dataset (etth2)
        metrics_per_dataset2 = calculate_weighted_metrics(models, 'etth2', frequencies, weights, coalesce_suffixes=True)
        metrics_per_dataset2.columns = pd.MultiIndex.from_product([['etth2'], metrics_per_dataset2.columns])

        # Merge both datasets using MultiIndex on columns
        metrics_per_dataset = pd.concat([metrics_per_dataset, metrics_per_dataset2], axis=1)

        # Reset index for better visualization
        metrics_per_dataset = metrics_per_dataset.reset_index()

        # Print output
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


for dataset in ['m4', 'm3', 'tourism', 'etth1', 'etth2']:
    all_results[dataset]['dataset'] = dataset
    all_results[dataset] = all_results[dataset].reset_index()
    all_results[dataset] = all_results[dataset].melt(id_vars=['Model', 'Frequency', 'dataset'], var_name=['Suffix', 'Metric'], value_name='Value')

df_all_results = pd.concat(all_results)
df_all_results['type_of_model'] = df_all_results['Model'].map(models)

# recursive vs direct
recursive_smape = df_all_results[(df_all_results['Suffix'] == 'recursive') & (df_all_results['Metric']=='sMAPE')]
direct_smape = df_all_results[(df_all_results['Suffix'] == 'direct') & (df_all_results['Metric']=='sMAPE')]

recursive_smape = recursive_smape[(recursive_smape['type_of_model']!='stats') & ~(recursive_smape['dataset'].isin(['etth1', 'etth2']))]
direct_smape = direct_smape[(direct_smape['type_of_model']!='stats') & ~(direct_smape['dataset'].isin(['etth1', 'etth2']))]

table_data = stats_test_table(recursive_smape['Value'].values, direct_smape['Value'].values)
output_folder = os.path.join(os.getcwd(), f'models/metrics_results/latex_t_tests')
# Convert DataFrame to LaTeX with multicolumn for MultiIndex columns
latex_table = table_data.to_latex(index=False,
                                  caption=f"T-test recursive vs. direct approach",
                                  label=f"tab:t_rec_dir",
                                  float_format="%.2f")  # Ensures numbers have at most 2 decimal places

# Save LaTeX table to a file
output_path = os.path.join(output_folder, f"t_rec_dir.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)


# training time
recursive_tr_time = df_all_results[(df_all_results['Suffix'] == 'recursive') & (df_all_results['Metric']=='Training Time (s)')]
direct_tr_time = df_all_results[(df_all_results['Suffix'] == 'direct') & (df_all_results['Metric']=='Training Time (s)')]

recursive_tr_time = recursive_tr_time[(recursive_tr_time['type_of_model']!='stats') & ~(recursive_tr_time['dataset'].isin(['etth1', 'etth2']))]
direct_tr_time = direct_tr_time[(direct_tr_time['type_of_model']!='stats') & ~(direct_tr_time['dataset'].isin(['etth1', 'etth2']))]

table_data = stats_test_table(recursive_tr_time['Value'].values, direct_tr_time['Value'].values, metric='Training Time (s)')
output_folder = os.path.join(os.getcwd(), f'models/metrics_results/latex_t_tests')
# Convert DataFrame to LaTeX with multicolumn for MultiIndex columns
latex_table = table_data.to_latex(index=False,
                                  caption=f"T-test recursive vs. direct approach",
                                  label=f"tab:t_rec_dir",
                                  float_format="%.2f")  # Ensures numbers have at most 2 decimal places

# Save LaTeX table to a file
output_path = os.path.join(output_folder, f"t_rec_dir_tr_time.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)


# ML vs DL
ml = df_all_results[df_all_results['type_of_model']=='ml']
dl = df_all_results[df_all_results['type_of_model']=='dl']

ml = ml.dropna()
dl = dl.dropna()

ml = ml.groupby(['dataset', 'Frequency', 'Suffix', 'Metric']).agg({'Value':'mean'})
dl = dl.groupby(['dataset', 'Frequency', 'Suffix', 'Metric']).agg({'Value':'mean'})

ml_smape = ml.reset_index()[ml.reset_index()['Metric']=='sMAPE']
dl_smape = dl.reset_index()[dl.reset_index()['Metric']=='sMAPE']

table = stats_test_table(ml_smape['Value'].values, dl_smape['Value'].values)
latex_table = table.to_latex(index=False,
                                  caption=f"T-test gradient boosting vs. DL",
                                  label=f"tab:t_gb_dl",
                                  float_format="%.2f")  # Ensures numbers have at most 2 decimal places

# Save LaTeX table to a file
output_path = os.path.join(output_folder, f"t_gb_dl.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)

"""
import numpy as np
import scipy.stats as stats

# Sample sMAPE scores
model_A = [0.12, 0.15, 0.14, 0.10, 0.13]
model_B = [0.18, 0.20, 0.21, 0.17, 0.19]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(model_A, model_B)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

T-statistic: -5.4772
P-value: 0.0006
Interpretation
T-statistic = -5.48

The negative value means model_A has a lower mean sMAPE than model_B.
The magnitude 5.48 is large, suggesting a significant difference.
P-value = 0.0006

Since p-value < 0.05, we reject the null hypothesis.
Conclusion: Model A's sMAPE is significantly different from Model B's sMAPE.
"""

#
# models = {
#     'ARIMA': 'stats',
#     'SARIMA': 'stats',
#     'LightGBM': 'ml',
#     'XGBoost': 'ml',
#     'RNN': 'dl',
#     'LSTM': 'dl',
#     'Transformer': 'dl',
#     'xLSTM(1:0)': 'dl',
#     'xLSTM(0:1)': 'dl',
#     'xLSTM(1:1)': 'dl'
# }
