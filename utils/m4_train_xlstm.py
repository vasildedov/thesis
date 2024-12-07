import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from utils.m4_train_dl import recursive_predict

# Additional utility functions for diagnostics
def log_gradients(model):
    print("\nGradient Statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: Grad Min={param.grad.min().item()}, Max={param.grad.max().item()}, Mean={param.grad.mean().item()}")
        else:
            print(f"{name}: No gradients (possibly unused in computation)")


def log_weights(model):
    print("\nWeight Update Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: Weight Min={param.min().item()}, Max={param.max().item()}, Mean={param.mean().item()}")


def log_activations(x, name):
    print(f"{name}: Min={x.min().item()}, Max={x.max().item()}, Mean={x.mean().item()}")
    return x


# debug through blocks
def debug_blocks(X, embedding_dim, stack):
    x = X[:10].clone().detach()
    input_projection = nn.Linear(1, embedding_dim)
    x = input_projection(x)

    for i, block in enumerate(stack.blocks):
        x = block(x)
        if torch.isnan(x).any():
            print(f"NaN detected after Block {i}! Investigating...")
            raise ValueError(f"NaN detected in block {i}")
        print(f"After Block {i}: Min={x.min().item()}, Max={x.max().item()}, Mean={x.mean().item()}, Shape={x.shape}")


# Training function
def train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train[indices].to(device)
            batch_y = y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)  # Expected shape: [batch_size]

            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                print("Loss is NaN. Investigate inputs and outputs.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

            optimizer.step()
            epoch_loss += loss.item()

            # Log sample outputs and gradients during training
            # print(f"Epoch {epoch + 1}, Batch {i // batch_size}: Sample outputs: {outputs[:5].detach().cpu().numpy()}")
            # log_gradients(model)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        # log_weights(model)

# Add diagnostics to training and evaluation
def train_and_predict(device, model, X_train, y_train, X_test, scalers, epochs, batch_size, horizon, test, criterion):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Sanity checks
    sanity_check_data(X_train, y_train, X_test)
    # sanity_check_model(model)

    # Start timing
    start_time = time.time()

    # Train the model
    train_xlstm(device, model, epochs, X_train, y_train, batch_size, optimizer, criterion)

    # End timing
    end_time = time.time()

    # Predict
    y_pred = recursive_predict(model, X_test, horizon, device, scalers)

    # Reshape predictions for evaluation
    num_series = test["unique_id"].nunique()
    y_pred = y_pred.reshape(num_series, horizon)
    return y_pred, end_time - start_time


# Sanity checks for input data
def sanity_check_data(X_train, y_train, X_test):
    print("Sanity Check: Input Data")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_train contains NaN: {torch.isnan(X_train).any().item()}")
    print(f"y_train contains NaN: {torch.isnan(y_train).any().item()}")
    print(f"X_train contains Inf: {torch.isinf(X_train).any().item()}")
    print(f"y_train contains Inf: {torch.isinf(y_train).any().item()}")


# Sanity checks for model initialization
def sanity_check_model(model):
    print("\nSanity Check: Model Initialization")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: Min={param.min().item():.4f}, Max={param.max().item():.4f}, Mean={param.mean().item():.4f}, Requires Grad={param.requires_grad}")


def get_stack_cfg(embedding_dim, look_back, device, fix_inits_bool=False):
    # Define an enhanced xLSTM configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=6,  # Larger kernel for capturing broader patterns
                qkv_proj_blocksize=8,  # Increased projection blocksize for better feature learning
                num_heads=8,  # More attention heads for complex patterns
                dropout=0.3,  # Slightly higher dropout for regularization
                embedding_dim=embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend='vanilla',
                num_heads=4,  # More heads for better focus on features
                conv1d_kernel_size=10,  # Larger convolution for feature aggregation
                bias_init="powerlaw_blockdependent",
                embedding_dim=embedding_dim
            ),
            feedforward=FeedForwardConfig(
                proj_factor=5.0,  # Larger projection factor for expanded representations
                act_fn="gelu",  # Smooth activation for stability
                dropout=0.3,
                embedding_dim=embedding_dim
            ),
        ),
        context_length=look_back + 1,
        num_blocks=8,  # Increased blocks for deeper model
        embedding_dim=embedding_dim,
        slstm_at=[0, 2, 4, 6]  # Strategically placed sLSTM blocks
    )

    stack = xLSTMBlockStack(cfg).to(device)
    if fix_inits_bool:
        fix_inits(stack)
    return stack


# model checks
def fix_inits(stack):
    # Fix initialization
    for name, param in stack.named_parameters():
        if param.dim() > 1 and "weight" in name:  # Apply Xavier only for tensors with >1 dimension
            xavier_uniform_(param)
        elif "bias" in name:  # Initialize biases to 0
            torch.nn.init.zeros_(param)
        elif "norm.weight" in name:  # Set norm weights to 1
            torch.nn.init.ones_(param)
        elif "learnable_skip" in name:  # Ensure learnable_skip parameters are trainable
            param.requires_grad = True
