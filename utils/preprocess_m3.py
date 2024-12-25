from datasetsforecast.m3 import M3, M3Info
import pandas as pd


# Load the dataset
def train_test_split(group):
    df, *_ = M3.load(directory='data', group=group)
    horizon = M3Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid, horizon
