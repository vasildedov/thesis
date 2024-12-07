from datasetsforecast.m4 import M4, M4Info
import pandas as pd


# Load the dataset
def train_test_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid

# Function to truncate series to a maximum length
def truncate_series(df, max_length=300):
    truncated_dfs = []
    for _, group in df.groupby('unique_id'):
        if len(group) > max_length:
            group = group.tail(max_length)
        truncated_dfs.append(group)
    return pd.concat(truncated_dfs).reset_index(drop=True)
