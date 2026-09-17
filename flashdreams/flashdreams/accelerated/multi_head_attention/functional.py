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

"""Functional dense and block-sparse attention behind one mask contract."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from flashdreams.accelerated.multi_head_attention import AttentionMask
from flashdreams.accelerated.multi_head_attention.flex import (
    FlexAttentionOptions,
    compiled_flex_attention,
)


class DenseSDPABackend(str, Enum):
    """PyTorch scaled-dot-product attention backend for dense masks."""

    CUDNN = "cudnn"
    """Use cuDNN on CUDA and PyTorch's automatic backend on CPU."""

    EFFICIENT = "efficient"
    """Use memory-efficient attention on CUDA and the math backend on CPU."""


def cudnn_attention() -> AbstractContextManager[None]:
    """Select cuDNN scaled-dot-product attention within a context."""
    return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)


def backend_for(
    query: Tensor,
    backend: DenseSDPABackend = DenseSDPABackend.CUDNN,
) -> AbstractContextManager[None]:
    """Select a dense SDPA backend appropriate for the query device."""
    if not isinstance(backend, DenseSDPABackend):
        raise TypeError(f"backend must be a DenseSDPABackend; got {backend!r}.")
    if backend is DenseSDPABackend.CUDNN:
        return cudnn_attention() if query.is_cuda else nullcontext()
    return torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        if query.is_cuda
        else torch.nn.attention.SDPBackend.MATH
    )


def dense_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    *,
    enable_gqa: bool = False,
    backend: DenseSDPABackend = DenseSDPABackend.CUDNN,
) -> Tensor:
    """Apply dense attention with an explicitly selected PyTorch backend.

    The memory-efficient backend groups query heads around zero-stride K/V
    views when grouped-query attention is enabled. This preserves native GQA
    memory use even when the selected PyTorch kernel does not accept
    ``enable_gqa`` directly.

    Args:
        query: Query heads shaped ``[B, Hq, Q, D]``.
        key: Key heads shaped ``[B, Hkv, K, D]``.
        value: Value heads shaped ``[B, Hkv, K, Dv]``.
        mask: Optional boolean visibility mask shaped ``[Q, K]``.
        enable_gqa: Share each K/V head across a group of query heads.
        backend: Dense PyTorch SDPA backend policy.

    Returns:
        Attention output shaped ``[B, Hq, Q, Dv]``.
    """
    _validate_qkv(query, key, value, enable_gqa=enable_gqa)
    _validate_dense_mask(mask, query=query, key=key)
    if not isinstance(backend, DenseSDPABackend):
        raise TypeError(f"backend must be a DenseSDPABackend; got {backend!r}.")

    output_shape: tuple[int, ...] | None = None
    use_native_gqa = enable_gqa
    if backend is DenseSDPABackend.EFFICIENT and enable_gqa:
        repeats = query.shape[1] // key.shape[1]
        output_shape = (*query.shape[:3], value.shape[-1])
        batch_heads = key.shape[0] * key.shape[1]
        query = query.reshape(batch_heads, repeats, *query.shape[2:])
        key = key.reshape(batch_heads, 1, *key.shape[2:]).expand(-1, repeats, -1, -1)
        value = value.reshape(batch_heads, 1, *value.shape[2:]).expand(
            -1, repeats, -1, -1
        )
        use_native_gqa = False

    with backend_for(query, backend):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            enable_gqa=use_native_gqa,
        )
    return output if output_shape is None else output.reshape(output_shape)


def masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: AttentionMask,
    *,
    enable_gqa: bool = False,
    dense_backend: DenseSDPABackend = DenseSDPABackend.CUDNN,
    flex_options: FlexAttentionOptions = FlexAttentionOptions(),
) -> Tensor:
    """Apply dense or block-sparse attention selected by ``mask``.

    Args:
        query: Query heads shaped ``[B, Hq, Q, D]``.
        key: Key heads shaped ``[B, Hkv, K, D]``.
        value: Value heads shaped ``[B, Hkv, K, Dv]``.
        mask: A boolean ``[Q, K]`` tensor or equivalent :class:`BlockMask`.
        enable_gqa: Share each K/V head across a group of query heads.
        dense_backend: PyTorch backend used for a dense mask. Ignored for a
            block-sparse mask.
        flex_options: Mask geometry, compilation, and kernel policy used by
            FlexAttention. Ignored for a dense mask.

    Returns:
        Attention output shaped ``[B, Hq, Q, Dv]``.

    Raises:
        TypeError: ``mask`` is neither a boolean tensor nor a block mask.
        ValueError: Q/K/V or mask geometry is incompatible.
    """
    _validate_qkv(query, key, value, enable_gqa=enable_gqa)
    if not isinstance(dense_backend, DenseSDPABackend):
        raise TypeError(
            f"dense_backend must be a DenseSDPABackend; got {dense_backend!r}."
        )
    if not isinstance(flex_options, FlexAttentionOptions):
        raise TypeError(
            f"flex_options must be a FlexAttentionOptions; got {flex_options!r}."
        )
    expected = (query.shape[-2], key.shape[-2])
    if isinstance(mask, BlockMask):
        if tuple(mask.shape[-2:]) != expected:
            raise ValueError(
                f"block mask shape {tuple(mask.shape[-2:])} does not match {expected}."
            )
        return compiled_flex_attention(dynamic=flex_options.compile_dynamic)(
            query,
            key,
            value,
            block_mask=mask,
            enable_gqa=enable_gqa,
            kernel_options=flex_options.kernel_options or None,
        )
    if not isinstance(mask, Tensor):
        raise TypeError(f"mask must be a Tensor or BlockMask; got {type(mask)!r}.")
    return dense_attention(
        query,
        key,
        value,
        mask,
        enable_gqa=enable_gqa,
        backend=dense_backend,
    )


def _validate_dense_mask(
    mask: Tensor | None,
    *,
    query: Tensor,
    key: Tensor,
) -> None:
    if mask is None:
        return
    if not isinstance(mask, Tensor):
        raise TypeError(f"mask must be a Tensor or None; got {type(mask)!r}.")
    if mask.ndim != 2:
        raise ValueError(
            "dense mask must be a two-dimensional [Q, KV] mask; "
            f"got {tuple(mask.shape)}."
        )
    expected = (query.shape[-2], key.shape[-2])
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"dense mask shape {tuple(mask.shape)} does not match {expected}."
        )
    if mask.dtype is not torch.bool:
        raise ValueError(f"dense mask must be boolean; got {mask.dtype}.")
    if mask.device != query.device:
        raise ValueError("dense mask must be on the query device.")


def _validate_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    enable_gqa: bool,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, S, D].")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value batch/token axes must agree.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must agree.")
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if enable_gqa:
        if query_heads % kv_heads:
            raise ValueError(
                f"query heads {query_heads} must be divisible by K/V heads {kv_heads}."
            )
    elif query_heads != kv_heads:
        raise ValueError(
            "query and K/V heads must agree unless enable_gqa is true; "
            f"got {query_heads} and {kv_heads}."
        )


__all__ = [
    "backend_for",
    "compiled_flex_attention",
    "cudnn_attention",
    "dense_attention",
    "DenseSDPABackend",
    "FlexAttentionOptions",
    "masked_attention",
]
