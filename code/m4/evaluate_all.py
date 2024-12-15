import pandas as pd
import json
import os

datas = {}
for folder in ['ml_' + freq for freq in ['daily', 'hourly', 'monthly', 'quarterly', 'weekly', 'yearly']]:
    with open(os.getcwd()+f'\models\{folder}\lgbm_metadata.json') as f:
        datas[folder] = json.load(f)

all_series = 100000
yearly_weight = 23000/all_series
quarterly_weight = 24000/all_series
monthly_weight = 48000/all_series
weekly_weight = 359/all_series
daily_weight = 4227/all_series
hourly_weight = 414/all_series

weights = {'yearly_weight': yearly_weight, 'quarterly_weight': quarterly_weight, 'monthly_weight': monthly_weight,
           'weekly_weight': weekly_weight, 'daily_weight': daily_weight, 'hourly_weight': hourly_weight}

total_smape = float()
for key in datas:
    print(key.split('_')[1])
    print(datas[key]['SMAPE'])
    contribution = datas[key]['SMAPE'] * weights[key.split('_')[1]+'_weight']
    print(contribution)
    total_smape += contribution

