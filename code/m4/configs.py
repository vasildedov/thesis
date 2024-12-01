# Define xLSTM configuration
# cfg = xLSTMBlockStackConfig(
#     mlstm_block=mLSTMBlockConfig(
#         mlstm=mLSTMLayerConfig(
#             conv1d_kernel_size=4,
#             qkv_proj_blocksize=4,
#             num_heads=4,  # Set to 1
#             dropout=0.2
#         )
#     ),
#     slstm_block=sLSTMBlockConfig(
#         slstm=sLSTMLayerConfig(
#             backend="vanilla" if torch.cuda.is_available() else "vanilla",
#             num_heads=1,  # Set to 1
#             conv1d_kernel_size=8,
#             bias_init="powerlaw_blockdependent",  # constant
#         ),
#         feedforward=FeedForwardConfig(
#             proj_factor=4.0,
#             act_fn="gelu",
#             dropout=0.2
#         ),
#     ),
#     context_length=look_back+1,
#     num_blocks=2,
#     embedding_dim=embedding_dim,
#     slstm_at=[x for x in range(2) if x % 2 == 0],  # Alternate placement
# )
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
#             bias_init="powerlaw_blockdependent",
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
#
# cfg = xLSTMBlockStackConfig(
#     mlstm_block=None,  # Use only sLSTM blocks
#     slstm_block=sLSTMBlockConfig(
#         slstm=sLSTMLayerConfig(
#             backend="vanilla",
#             num_heads=1,
#             conv1d_kernel_size=4,
#             bias_init="constant",
#         )
#     ),
#     context_length=look_back + 1,
#     num_blocks=5,  # Reduce number of blocks
#     embedding_dim=embedding_dim,
# )
