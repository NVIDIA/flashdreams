# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared scaled-dot-product attention over projected token-major tensors."""

from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor


class SDPABackend(str, Enum):
    """Scaled-dot-product attention implementation."""

    TORCH = "torch"
    """Use PyTorch's native device- and dtype-aware dispatcher."""

    CUDNN = "cudnn"
    """Use Torch cuDNN or native cuDNN Frontend for FP8."""

    FA2 = "fa2"
    """Use Triton FlashAttention2."""


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_causal: bool = False,
    backend: SDPABackend = SDPABackend.TORCH,
    use_tma: bool = False,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Attend to projected Q/K/V in ``[B, L/S, H, D]`` layout.

    Args:
        query: Projected query tokens.
        key: Projected key tokens.
        value: Projected value tokens.
        is_causal: Apply a causal mask; supported by the native Torch backend.
        backend: Implementation policy, independent of model projections.
        use_tma: Prefer TMA when the FA2 backend and hardware support it.
        output_dtype: Optional output storage dtype.

    Returns:
        Attention output in token-major layout.

    Raises:
        ValueError: Tensor geometry or backend policy is invalid.
    """
    if not isinstance(backend, SDPABackend):
        raise TypeError(f"unsupported SDPA backend: {backend!r}")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("Q/K/V must have shape [B, L/S, H, D]")
    if (
        key.shape != value.shape
        or query.shape[0] != key.shape[0]
        or query.shape[2:] != key.shape[2:]
    ):
        raise ValueError("Q/K/V batch, head and feature dimensions must match")
    if is_causal and backend is not SDPABackend.TORCH:
        raise ValueError("causal attention requires the Torch SDPA backend")
    if backend is SDPABackend.TORCH:
        if query.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Torch SDPA does not support FP8 inputs")
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=is_causal,
        ).transpose(1, 2)
    elif backend is SDPABackend.CUDNN:
        from flashdreams.accelerated.multi_head_attention.cudnn import (
            native_cudnn_fp8_sdpa,
            torch_cudnn_sdpa,
        )

        attention = (
            native_cudnn_fp8_sdpa
            if query.dtype is torch.float8_e4m3fn
            else torch_cudnn_sdpa
        )
        output = attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
    else:
        from flashdreams.accelerated.multi_head_attention.triton import (
            flash_attention_2,
            flash_attention_2_tma,
            is_tma_flash_attention_supported,
        )

        attention = (
            flash_attention_2_tma
            if use_tma and is_tma_flash_attention_supported(query, key, value)
            else flash_attention_2
        )
        return attention(query, key, value, output_dtype=output_dtype)
    return output if output_dtype is None else output.to(output_dtype)
