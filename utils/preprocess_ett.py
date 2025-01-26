import datasets
import pandas as pd
import numpy as np


def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
    """Convert dataset to long data frame format."""
    sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
    return ds.to_pandas().explode(sequence_columns).infer_objects()

ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTh", split="train", trust_remote_code=True)
ds.set_format("numpy")  # sequences returned as numpy arrays
ds_p = to_pandas(ds)
etth1 = ds_p[ds_p['id']=='ETTh1']


# add so y is only the target
def create_sliding_windows(data, look_back, horizon, step=1):
    inputs, outputs = [], []
    for i in range(0, len(data) - look_back - horizon + 1, step):
        inputs.append(data[i:i + look_back])
        outputs.append(data[i + look_back:i + look_back + horizon])
    return np.array(inputs), np.array(outputs)

# Example Parameters
look_back = 720
horizon = 96
step_train = 1  # Overlapping windows
step_val_test = horizon  # Non-overlapping windows

# Split data into train, validation, test
train_data = etth1[:8640]  # First 12 months
val_data = etth1[8640:11520]  # Next 4 months
test_data = etth1[11520:]  # Last 4 months

# Generate sliding windows
X_train, y_train = create_sliding_windows(train_data, look_back, horizon, step_train)
X_val, y_val = create_sliding_windows(val_data, look_back, horizon, step_val_test)
X_test, y_test = create_sliding_windows(test_data, look_back, horizon, step_val_test)

print(f"Training Data: {X_train.shape}, {y_train.shape}")
print(f"Validation Data: {X_val.shape}, {y_val.shape}")
print(f"Testing Data: {X_test.shape}, {y_test.shape}")



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