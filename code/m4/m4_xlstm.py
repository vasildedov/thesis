import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.m4_preprocess_ml import train_test_split, truncate_series
from datasetsforecast.m4 import M4, M4Info, M4Evaluation
from utils.ml_models import calculate_smape
from utils.m4_preprocess_dl import (
    create_train_windows,
    create_test_windows
)
from utils.dl_models import xLSTMTimeSeriesModel
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from utils.preprocess_xlstm import train_and_evaluate_xlstm

# Choose the frequency
freq = 'Hourly'  # or 'Daily'

train, test = train_test_split(freq)

num_series = 1
filtered_series = train["unique_id"].unique()[:num_series]
train = train[train["unique_id"].isin(filtered_series)]
test = test[test["unique_id"].isin(filtered_series)]

# Set parameters based on frequency
if freq == 'Daily':
    look_back = 30  # Number of previous time steps for input
    horizon = 14  # Forecast horizon
    max_length = 200  # For truncating series
    lstm_hidden_size = 50
elif freq == 'Hourly':
    look_back = 120
    horizon = 48
    max_length = None  # Do not truncate series
    lstm_hidden_size = 100
else:
    raise ValueError("Unsupported frequency. Choose 'Daily' or 'Hourly'.")


# Create train and test windows
X_train_xlstm, y_train_xlstm, scalers_xlstm = create_train_windows(train, look_back, horizon)
X_test_xlstm = create_test_windows(train, look_back, scalers_xlstm)

# Prepare data tensors
X_train_xlstm = X_train_xlstm.clone().detach().unsqueeze(-1).float().requires_grad_()  # Add feature dimension
y_train_xlstm = y_train_xlstm.clone().detach().unsqueeze(-1).float()
X_test_xlstm = X_test_xlstm.clone().detach().unsqueeze(-1).float().requires_grad_()  # Add feature dimension

# Set up device
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# Move data to device
X_train_xlstm = X_train_xlstm.to(device)
y_train_xlstm = y_train_xlstm.to(device)
X_test_xlstm = X_test_xlstm.to(device)

# Define embedding dimension
embedding_dim = X_train_xlstm.shape[2]  # Should be 1 for univariate time series

# Define xLSTM configuration
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,  # Set to 1
            dropout=0.2
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla" if torch.cuda.is_available() else "vanilla",
            num_heads=1,  # Set to 1
            conv1d_kernel_size=8,
            bias_init="powerlaw_blockdependent",  # constant
        ),
        feedforward=FeedForwardConfig(
            proj_factor=4.0,
            act_fn="gelu",
            dropout=0.2
        ),
    ),
    context_length=look_back+1,
    num_blocks=50,
    embedding_dim=embedding_dim,
    slstm_at=[x for x in range(50) if x % 2 == 0],  # Alternate placement
)
#
# cfg = xLSTMBlockStackConfig(
#     mlstm_block=mLSTMBlockConfig(
#         mlstm=mLSTMLayerConfig(
#             conv1d_kernel_size=4,
#             qkv_proj_blocksize=4,
#             num_heads=1,
#             proj_factor=1.0,
#             dropout=0.1,
#             embedding_dim=embedding_dim,
#         )
#     ),
#     slstm_block=sLSTMBlockConfig(
#         slstm=sLSTMLayerConfig(
#             backend="vanilla",
#             num_heads=1,
#             conv1d_kernel_size=4,
#             bias_init="constant",
#             dropout=0.1,
#             embedding_dim=embedding_dim,
#         ),
#         feedforward=FeedForwardConfig(
#             proj_factor=1.5,
#             act_fn="gelu",
#             dropout=0.1,
#         ),
#     ),
#     context_length=look_back + 1,
#     num_blocks=4,  # Fewer blocks for simplicity
#     embedding_dim=embedding_dim,
#     slstm_at=[1, 3],
# )


# Instantiate model
xlstm_stack = xLSTMBlockStack(cfg).to(device)

from torch.nn.init import xavier_uniform_

# Fix initialization
for name, param in xlstm_stack.named_parameters():
    if param.dim() > 1 and "weight" in name:  # Apply Xavier only for tensors with >1 dimension
        xavier_uniform_(param)
    elif "bias" in name:  # Initialize biases to 0
        torch.nn.init.zeros_(param)
    elif "norm.weight" in name:  # Set norm weights to 1
        torch.nn.init.ones_(param)
    elif "learnable_skip" in name:  # Ensure learnable_skip parameters are trainable
        param.requires_grad = True


# Define a hook function for debugging
# def debug_hook(module, input, output):
#     print(f"{module}: Output Mean={output.mean().item()}, Std={output.std().item()}")
#
# # Register hooks to inspect intermediate activations
# for name, module in xlstm_stack.named_modules():
#     if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
#         module.register_forward_hook(debug_hook)

output_size = 1
model = xLSTMTimeSeriesModel(xlstm_stack, output_size, cfg).to(device)

# Training parameters
epochs = 10
batch_size = 32
criterion = nn.MSELoss()

# Check for NaNs and Infs
def check_for_nans_infs(data, name):
    print(f"{name} contains NaN:", torch.isnan(data).any())
    print(f"{name} contains Inf:", torch.isinf(data).any())

check_for_nans_infs(X_train_xlstm, "X_train_xlstm")
check_for_nans_infs(y_train_xlstm, "y_train_xlstm")

# Debug through blocks
x = X_train_xlstm[:10].clone().detach()
for i, block in enumerate(xlstm_stack.blocks):
    x = block(x)
    if torch.isnan(x).any():
        print(f"NaN detected after Block {i}! Investigating...")
        raise ValueError(f"NaN detected in block {i}")
    print(f"After Block {i}: Min={x.min().item()}, Max={x.max().item()}, Mean={x.mean().item()}, Shape={x.shape}")
#
#
# # Training parameters
# epochs = 10
# batch_size = 32
# criterion = nn.MSELoss()
#
# X_dummy = torch.rand(100, 120, 1)  # Random input
# y_dummy = torch.rand(100)  # Random target
#
#
# # Train and evaluate xLSTM model
# print("\nTraining and Evaluating xLSTM Model...")
# y_pred_xlstm = train_and_evaluate_xlstm(
#     device,
#     model,
#     X_dummy,
#     y_dummy,
#     X_dummy,
#     scalers_xlstm,
#     2,
#     batch_size,
#     horizon,
#     test,
#     freq,
#     criterion
# )
#
# y_true = test['y'].values.reshape(num_series, horizon)
#
# calculate_smape(y_true, y_pred_xlstm)
#


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=1000):
    # Generate x values
    x = torch.linspace(-1, 1, num_samples).unsqueeze(1)  # shape: (num_samples, 1)
    # Generate y values (e.g., a simple quadratic function)
    y = x ** 2 + 0.1 * torch.randn_like(x)  # Add some noise
    return x, y

# Step 3: Training Loop
def train_model(model, x_train, y_train, num_epochs=100, lr=1e-3):
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Step 4: Plot Results
def plot_predictions(model, x, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
    plt.figure(figsize=(8, 6))
    plt.scatter(x.numpy(), y_true.numpy(), label="True Data", color="blue", alpha=0.5)
    plt.scatter(x.numpy(), y_pred.numpy(), label="Predicted Data", color="red", alpha=0.5)
    plt.legend()
    plt.show()


# Step 5: Execute
# Generate synthetic data
x_train, y_train = generate_synthetic_data(num_samples=500)

x_train = x_train.unsqueeze(-1)

# Define and train the model
train_model(model, x_train, y_train, num_epochs=100, lr=1e-3)

# Plot results
plot_predictions(model, x_train, y_train)
