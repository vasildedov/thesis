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
        # slstm_block=sLSTMBlockConfig(
        #     slstm=sLSTMLayerConfig(
        #         backend='vanilla',
        #         num_heads=2,  # More heads for better focus on features
        #         conv1d_kernel_size=10,  # Larger convolution for feature aggregation
        #         bias_init="powerlaw_blockdependent",
        #         embedding_dim=embedding_dim
        #     ),
        #     feedforward=FeedForwardConfig(
        #         proj_factor=3.0,  # Larger projection factor for expanded representations
        #         act_fn="gelu",  # Smooth activation for stability
        #         dropout=0.3,
        #         embedding_dim=embedding_dim
        #     ),
        # ),
        context_length=look_back + 1,
        num_blocks=4,  # Increased blocks for deeper model
        embedding_dim=embedding_dim,
        # slstm_at=[0]  # Strategically placed sLSTM blocks
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
