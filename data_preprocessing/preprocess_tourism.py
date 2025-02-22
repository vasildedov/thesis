import datasets
import pandas as pd


def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
    """Convert dataset to long data frame format."""
    sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
    return ds.to_pandas().explode(sequence_columns).infer_objects()


def train_test_split(group):
    ds = datasets.load_dataset("autogluon/chronos_datasets", f"monash_tourism_{group}")
    df = to_pandas(ds['train']).reset_index(drop=True)
    df = df.rename(columns={'id': 'unique_id', 'timestamp': 'ds', 'target': 'y'})

    if group == 'monthly':
        horizon = 24
    elif group == 'quarterly':
        horizon = 8
    elif group == 'yearly':
        horizon = 4

    test = df.groupby('unique_id').tail(horizon)
    train = df.drop(test.index)
    return train, test, horizon

"""from datasetsforecast import long_horizon

y_df, X_df, S_df = long_horizon.LongHorizon.load(directory='data', group='ETTh1')
horizon = long_horizon.ETTh1.horizons[0]

test = X_df.groupby('unique_id').tail(0.2*len(X_df))
train = X_df.drop(test.index)


len(train)+len(val)+len(test) == len(X_df)

long_horizon.ETTh1.test_size


import datasets
from datasets import load_dataset
dataset = load_dataset("Monash-University/monash_tsf", "solar_weekly")

dataset1 = load_dataset("AutonLab/Timeseries-PILE", "ETT")

ds = load_dataset("autogluon/chronos_datasets", "m4_daily", split="train")
ds.set_format("numpy")  # sequences returned as numpy arrays
ds_p = ds.to_pandas(ds)

ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTh", split="train", trust_remote_code=True)
ds.set_format("numpy")  # sequences returned as numpy arrays
ds_p = ds.to_pandas(ds)

ds = datasets.load_dataset("monash_tsf", "tourism_monthly")
ds_p = to_pandas(ds)
"""
