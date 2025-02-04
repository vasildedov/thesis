import datasets
import pandas as pd
import numpy as np


def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
    """Convert dataset to long data frame format."""
    sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
    return ds.to_pandas().explode(sequence_columns).infer_objects()

# add so y is only the target
def create_sliding_windows(data, look_back, horizon, step=1, target='OT'):
    # Check if the target column exists
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the data.")

    inputs, outputs = [], []
    for i in range(0, len(data) - look_back - horizon + 1, step):
        # Include target column as part of the input
        X_window = data.iloc[i:i + look_back].values  # All features, including the target
        y_window = data[target].iloc[i + look_back:i + look_back + horizon].values  # Only the target for prediction

        inputs.append(X_window)
        outputs.append(y_window)

    return np.array(inputs), np.array(outputs)

def train_test_split(group="etth1", multivariate=True):
    ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTh", split="train", trust_remote_code=True)
    ds.set_format("numpy")  # sequences returned as numpy arrays
    ds_p = to_pandas(ds)
    if group == 'etth1':
        df = ds_p[ds_p['id']=='ETTh1']
    else:
        df = ds_p[ds_p['id']=='ETTh2']
    df = df.drop(columns=['id'])
    if multivariate:
        df = df.drop(columns=['timestamp'])
        df = df.reset_index(drop=True)
    else:
        df = df[['OT', 'timestamp']]
        df = df.set_index('timestamp')

    train_ind = int(0.6*len(df))
    val_ind = train_ind+int(0.2*len(df))
    # Split data into train, validation, test
    train_data = df[:train_ind]
    val_data = df[train_ind:val_ind]
    test_data = df[val_ind:]
    return train_data, val_data, test_data

def get_windows(train_data, val_data, test_data,
                look_back=720, horizon=96, step_train=1):  # Overlapping windows
    step_val_test = horizon  # Non-overlapping windows
    # Generate sliding windows
    X_train, y_train = create_sliding_windows(train_data, look_back, horizon, step_train)
    X_val, y_val = create_sliding_windows(val_data.reset_index(drop=True), look_back, horizon, step_val_test)
    X_test, y_test = create_sliding_windows(test_data.reset_index(drop=True), look_back, horizon, step_val_test)

    print(f"Training Data: {X_train.shape}, {y_train.shape}")
    print(f"Validation Data: {X_val.shape}, {y_val.shape}")
    print(f"Testing Data: {X_test.shape}, {y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# train_data, val_data, test_data = train_test_split()
#
# X_train, y_train, X_val, y_val, X_test, y_test = get_windows(train_data, val_data, test_data)


# https://arxiv.org/pdf/2308.11200v1
# convert to 12-4-4 train-val-test
# horizons and look-backs can be the same as in the paper - 96, 192, 336, 720 - best to go for 720
# normal sliding windows for train set:
"""
Training Sliding Windows
Overlapping Windows: The input windows slide forward by one step (e.g., one hour at a time), creating highly overlapping input-output pairs.
Why?: This maximizes the number of training samples, which is important for model learning. More windows provide more data for the model to generalize from.
Example:
Look-back: 720 hours.

Horizon: 96 hours.

Data: [x₀, x₁, ..., xₙ].

First Window:

Input: x₀ → x₇₁₉.
Output: x₇₂₀ → x₈₁₅.
Second Window:

Input: x₁ → x₇₂₀.
Output: x₇₂₁ → x₈₁₆.
Sliding by One Step: Each new input-output pair overlaps significantly with the previous one.

"""

# for validation and test:
"""
Given:
Validation Period: 4 months (e.g., 2880 hours).
Test Period: 4 months (e.g., 2880 hours).
Look-back Window: 720 hours.
Forecast Horizon: 96 hours.
Validation Windows:
First window:
Input: Hours 0 to 719 (from validation set).
Output: Hours 720 to 815.
Second window:
Input: Hours 96 to 815.
Output: Hours 816 to 911.
Repeat until the end of the validation set.
Testing Windows:
The same process is applied to the testing set:

First window:
Input: Hours 0 to 719 (from testing set).
Output: Hours 720 to 815.
Second window:
Input: Hours 96 to 815.
Output: Hours 816 to 911.
Continue until the end of the testing set.
"""