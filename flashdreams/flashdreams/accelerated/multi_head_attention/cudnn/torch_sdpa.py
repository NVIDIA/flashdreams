# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch scaled-dot-product attention forced to cuDNN."""

import torch
import torch.nn.functional as F
from torch import Tensor


def torch_cudnn_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Apply PyTorch scaled-dot-product attention with the cuDNN backend.

    Args:
        query: Queries in ``[B, H, L, D]`` layout.
        key: Keys in ``[B, H, S, D]`` layout.
        value: Values in ``[B, H, S, D]`` layout.
        attn_mask: Optional dense boolean visibility mask shaped ``[L, S]``.
        enable_gqa: Share each key/value head across a group of query heads.

    Returns:
        Attention output shaped like ``query``.
    """
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            enable_gqa=enable_gqa,
        )
