import torch
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


def get_stack_cfg(embedding_dim, look_back, device, dropout, ratio="1_1", fix_inits_bool=False):
    """
    Returns an xLSTM block stack configuration based on the specified mLSTM-to-sLSTM ratio.

    Args:
        embedding_dim (int): The embedding dimension for both mLSTM and sLSTM layers.
        look_back (int): The look-back window size.
        device (torch.device): The device to which the stack should be moved.
        dropout (float): The dropout rate for regularization.
        ratio (str): The mLSTM-to-sLSTM ratio, e.g., "7:1", "1:1", or "1:0".
        fix_inits_bool (bool): Whether to apply fixed initializations for stability.

    Returns:
        stack: Configured xLSTM block stack.
    """
    # Parse the ratio string
    try:
        mlstm_count, slstm_count = map(int, ratio.split("_"))
    except ValueError:
        raise ValueError(f"Invalid ratio format: {ratio}. Must be in 'm:s' format, e.g., '7:1', '1:1', or '1:0'.")

    # Handle special cases for ratios
    if ratio == "1_0":
        total_blocks = 4  # Fixed number of blocks
    else:
        total_blocks = mlstm_count + slstm_count
    slstm_indices = (
        list(range(1, total_blocks, total_blocks // slstm_count))[:slstm_count]
        if slstm_count > 0
        else []
    )

    # Create xLSTM configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=6,
                qkv_proj_blocksize=8,
                num_heads=8,
                dropout=dropout,
                embedding_dim=embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=1,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
                embedding_dim=embedding_dim,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=1.5,
                act_fn="gelu",
                dropout=dropout,
                embedding_dim=embedding_dim,
            ),
        ),
        context_length=look_back + 1,
        num_blocks=total_blocks,
        embedding_dim=embedding_dim,
        slstm_at=slstm_indices
    )

    # Build the stack and move to the device
    stack = xLSTMBlockStack(cfg).to(device)

    # Apply fixed initializations if requested
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
