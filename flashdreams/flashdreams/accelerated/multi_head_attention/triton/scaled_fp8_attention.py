# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scale-aware INT8-QK/FP8-PV attention built on the in-tree FA2 kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashdreams.accelerated.multi_head_attention.triton.flash_attention_2_kernel import (
    flash_attention_2,
)
from flashdreams.accelerated.multi_head_attention.triton.flash_attention_2_tma_kernel import (
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)


@triton.jit
def _quantize_qk_rows_kernel(
    input_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    input_stride_b,
    input_stride_s,
    input_stride_h,
    mean_stride_b,
    mean_stride_h,
    sequence_length,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
):
    """Quantize one Q/K row and optionally fuse sequence-mean subtraction."""
    row = tl.program_id(0)
    head = row % num_heads
    sequence = (row // num_heads) % sequence_length
    batch = row // (num_heads * sequence_length)
    feature_offsets = tl.arange(0, BLOCK_D)
    mask = feature_offsets < HEAD_DIM
    values = tl.load(
        input_ptr
        + batch * input_stride_b
        + sequence * input_stride_s
        + head * input_stride_h
        + feature_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if SUBTRACT_MEAN:
        values -= tl.load(
            mean_ptr + batch * mean_stride_b + head * mean_stride_h + feature_offsets,
            mask=mask,
            other=0.0,
        )
    scale = tl.maximum(
        tl.max(tl.abs(values), axis=0) / 127.0,
        torch.finfo(torch.float32).tiny,
    )
    quantized = tl.maximum(
        tl.minimum(libdevice.rint(libdevice.div_rn(values, scale)), 127.0), -127.0
    )
    tl.store(
        output_ptr
        + batch * input_stride_b
        + sequence * input_stride_s
        + head * input_stride_h
        + feature_offsets,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row, scale)


def _quantize_qk_rows(
    input: Tensor, mean: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize Q/K rows, optionally subtracting K mean in-kernel."""
    batch_size, sequence_length, num_heads, head_dim = input.shape
    output = torch.empty_like(input, dtype=torch.int8)
    scale = torch.empty(
        (batch_size, sequence_length, num_heads, 1),
        device=input.device,
        dtype=torch.float32,
    )
    if mean is None:
        mean = input
    _quantize_qk_rows_kernel[(batch_size * sequence_length * num_heads,)](
        input,
        mean,
        output,
        scale,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        mean.stride(0),
        mean.stride(2),
        sequence_length,
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SUBTRACT_MEAN=mean is not input,
        num_warps=4,
    )
    return output, scale


@triton.jit
def _value_channel_partial_max_kernel(
    value_ptr,
    partial_ptr,
    value_stride_b,
    value_stride_s,
    value_stride_h,
    partial_stride_b,
    partial_stride_h,
    partial_stride_s,
    sequence_length,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Reduce one coalesced sequence tile to per-channel absolute maxima."""
    sequence_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    sequence_offsets = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
    feature_offsets = tl.arange(0, HEAD_DIM)
    values = tl.load(
        value_ptr
        + batch * value_stride_b
        + sequence_offsets[:, None] * value_stride_s
        + head * value_stride_h
        + feature_offsets[None, :],
        mask=sequence_offsets[:, None] < sequence_length,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        partial_ptr
        + batch * partial_stride_b
        + head * partial_stride_h
        + sequence_block * partial_stride_s
        + feature_offsets,
        tl.max(tl.abs(values), axis=0),
    )


@triton.jit
def _value_channel_scale_kernel(
    partial_ptr,
    scale_ptr,
    partial_stride_b,
    partial_stride_h,
    partial_stride_s,
    scale_stride_b,
    scale_stride_h,
    partial_count,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PARTIAL: tl.constexpr,
):
    """Reduce tile maxima to one FP32 FP8 scale per value channel."""
    channel = tl.program_id(0)
    head = (channel // HEAD_DIM) % num_heads
    batch = channel // (num_heads * HEAD_DIM)
    feature = channel % HEAD_DIM
    partial_offsets = tl.arange(0, BLOCK_PARTIAL)
    maxima = tl.load(
        partial_ptr
        + batch * partial_stride_b
        + head * partial_stride_h
        + partial_offsets * partial_stride_s
        + feature,
        mask=partial_offsets < partial_count,
        other=0.0,
    )
    scale = tl.maximum(
        tl.max(maxima, axis=0) / 448.0,
        torch.finfo(torch.float32).tiny,
    )
    tl.store(
        scale_ptr + batch * scale_stride_b + head * scale_stride_h + feature,
        scale,
    )


@triton.jit
def _quantize_value_channels_kernel(
    value_ptr,
    scale_ptr,
    output_ptr,
    value_stride_b,
    value_stride_s,
    value_stride_h,
    scale_stride_b,
    scale_stride_h,
    sequence_length,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Quantize one coalesced value tile with its per-channel FP32 scales."""
    sequence_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    sequence_offsets = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
    feature_offsets = tl.arange(0, HEAD_DIM)
    mask = sequence_offsets[:, None] < sequence_length
    offsets = (
        batch * value_stride_b
        + sequence_offsets[:, None] * value_stride_s
        + head * value_stride_h
        + feature_offsets[None, :]
    )
    values = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scales = tl.load(
        scale_ptr + batch * scale_stride_b + head * scale_stride_h + feature_offsets
    )
    quantized = tl.maximum(
        tl.minimum(libdevice.div_rn(values, scales[None, :]), 448.0), -448.0
    )
    tl.store(output_ptr + offsets, quantized, mask=mask)


def _quantize_value_channels(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize token-major V coalescently with one scale per channel."""
    batch_size, sequence_length, num_heads, head_dim = value.shape
    block_s = 64
    partial_count = triton.cdiv(sequence_length, block_s)
    partial = torch.empty(
        (batch_size, num_heads, partial_count, head_dim),
        device=value.device,
        dtype=torch.float32,
    )
    scale = torch.empty(
        (batch_size, 1, num_heads, head_dim),
        device=value.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    grid = (partial_count, batch_size * num_heads)
    _value_channel_partial_max_kernel[grid](
        value,
        partial,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        sequence_length,
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_S=block_s,
        num_warps=8,
    )
    _value_channel_scale_kernel[(batch_size * num_heads * head_dim,)](
        partial,
        scale,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        scale.stride(0),
        scale.stride(2),
        partial_count,
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_PARTIAL=triton.next_power_of_2(partial_count),
        num_warps=4,
    )
    _quantize_value_channels_kernel[grid](
        value,
        scale,
        output,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        scale.stride(0),
        scale.stride(2),
        sequence_length,
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_S=block_s,
        num_warps=8,
    )
    return output, scale


@torch.compiler.disable
def scaled_fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    output_dtype: torch.dtype | None = None,
    smooth_key: bool = True,
    use_tma: bool = True,
) -> Tensor:
    """Apply Sage-inspired scale-aware low-precision attention.

    Q and K use dynamic INT8 scales per token/head row. V uses dynamic FP8
    e4m3 scales per batch/head/channel across the sequence. The FA2 kernel
    consumes those scales directly, keeps its online softmax state and output
    accumulation in FP32, and uses compensated FP8 probabilities for ``P @ V``.
    Subtracting the sequence mean from K is softmax-invariant and reduces
    quantization outliers, matching the accuracy-preserving idea used by
    SageAttention.

    This is a generic in-tree implementation, not a wrapper around the
    SageAttention package. It supports non-causal, unmasked attention with equal
    Q/K/V head counts.

    Args:
        query: CUDA FP16/BF16 queries shaped ``[B, L, H, D]``.
        key: CUDA FP16/BF16 keys shaped ``[B, S, H, D]``.
        value: CUDA FP16/BF16 values shaped ``[B, S, H, D]``.
        scale: QK softmax scale; ``None`` uses ``1 / sqrt(D)``.
        output_dtype: FP16/BF16 output dtype; ``None`` uses ``query.dtype``.
        smooth_key: Subtract the sequence mean from K before quantization.
        use_tma: Prefer the TMA kernel when tensor layouts support it.

    Returns:
        Attention output shaped ``[B, L, H, D]``.

    Raises:
        ValueError: Q/K/V shapes are incompatible.
        RuntimeError: Placement or dtype is unsupported.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, L, H, D]")
    if key.shape[0] != query.shape[0] or key.shape[2:] != query.shape[2:]:
        raise ValueError("query and key batch, head, and feature dimensions differ")
    if value.shape != key.shape:
        raise ValueError("key and value must have identical shapes")
    if key.shape[1] == 0:
        raise ValueError("key and value sequence length must be positive")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("scaled FP8 attention requires CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise RuntimeError("query, key, and value must occupy the same CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise RuntimeError("query, key, and value must have the same dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("scaled FP8 attention requires FP16 or BF16 inputs")
    head_dim = query.shape[-1]
    if not (16 <= head_dim <= 256 and head_dim & (head_dim - 1) == 0):
        raise RuntimeError(
            "scaled FP8 attention requires a power-of-two head_dim in [16, 256]"
        )
    if output_dtype is None:
        output_dtype = query.dtype
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("scaled FP8 attention requires an FP16 or BF16 output")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    quantized_query, query_scale = _quantize_qk_rows(query)
    key_mean = key.mean(dim=1, keepdim=True) if smooth_key else None
    quantized_key, key_scale = _quantize_qk_rows(key, key_mean)
    quantized_value, value_scale = _quantize_value_channels(value)

    attention = (
        flash_attention_2_tma
        if use_tma
        and is_tma_flash_attention_supported(
            quantized_query, quantized_key, quantized_value
        )
        else flash_attention_2
    )
    return attention(
        quantized_query,
        quantized_key,
        quantized_value,
        scale=scale,
        output_dtype=output_dtype,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
    )


__all__ = ["scaled_fp8_attention"]
