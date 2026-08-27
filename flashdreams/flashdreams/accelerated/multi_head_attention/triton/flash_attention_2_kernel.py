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

"""Pointer-based Triton FlashAttention2 for projected attention tensors."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_ATTENTION_CONFIGS = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m, block_n, num_warps, num_stages in (
        (16, 32, 4, 2),
        (32, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 64, 4, 3),
        (64, 64, 8, 3),
        (128, 32, 4, 3),
        (128, 64, 4, 2),
        (128, 64, 4, 3),
        (128, 64, 8, 3),
        (128, 128, 8, 3),
    )
]
"""Candidate query/key tile geometries for FlashAttention autotuning.

``BLOCK_M`` controls query rows and the FP32 output-accumulator footprint;
``BLOCK_N`` controls each streamed K/V tile. Warp and stage variants let Triton
balance parallel dot products against load-pipeline resource use."""


def _prune_attention_configs(
    configs: list[triton.Config],
    named_args: dict[str, object],
    **meta: object,
) -> list[triton.Config]:
    """Drop tiles that waste work or exceed wide-head shared memory.

    This callback runs before benchmarking so short sequences and wide heads do
    not compile configurations whose padded work or accumulator footprint cannot
    be competitive.

    Args:
        configs: Candidate autotuning configurations.
        named_args: Runtime arguments containing ``query_length`` and
            ``key_length``.
        **meta: Compile-time metadata containing ``HEAD_DIM``.

    Returns:
        Configurations whose query and key tiles fit the input geometry.
    """
    query_length = named_args["query_length"]
    key_length = named_args["key_length"]
    head_dim = meta["HEAD_DIM"]
    assert isinstance(query_length, int)
    assert isinstance(key_length, int)
    assert isinstance(head_dim, int)
    # Bound each tile by its sequence axis. Wide ``[D]`` accumulators use at
    # most 64 query rows to limit SRAM consumption.
    maximum_block_m = min(128, max(16, int(triton.next_power_of_2(query_length))))
    if head_dim > 128:
        maximum_block_m = min(maximum_block_m, 64)
    maximum_block_n = min(128, max(32, int(triton.next_power_of_2(key_length))))
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] <= maximum_block_m
        and config.kwargs["BLOCK_N"] <= maximum_block_n
    ]


# Cache the winning tile by logical geometry and sequence strides. Pointer values
# and the numeric softmax scale do not change scheduling,
# so they intentionally do not create new autotuning entries.


@triton.autotune(
    configs=_ATTENTION_CONFIGS,
    key=[
        "num_heads",
        "query_length",
        "key_length",
        "query_stride_l",
        "key_stride_s",
        "value_stride_s",
        "HEAD_DIM",
    ],
    prune_configs_by={"early_config_prune": _prune_attention_configs},
    cache_results=True,
)
@triton.jit
def _flash_attention_2_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_stride_b,
    query_stride_h,
    query_stride_l,
    query_stride_d: tl.constexpr,
    key_stride_b,
    key_stride_h,
    key_stride_s,
    key_stride_d: tl.constexpr,
    value_stride_b,
    value_stride_h,
    value_stride_s,
    value_stride_d: tl.constexpr,
    output_stride_b,
    output_stride_h,
    output_stride_l,
    output_stride_d: tl.constexpr,
    num_heads: tl.constexpr,
    query_length: tl.constexpr,
    key_length: tl.constexpr,
    scale,
    HEAD_DIM: tl.constexpr,
    QUANTIZED_SDPA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply tiled non-causal FlashAttention2 with pointer loads and stores.

    Inputs are Q ``[B, L, H, D]`` and K/V ``[B, S, H, D]``. Element
    strides describe metadata-only ``[B, H, L|S, D]`` views over that
    storage. Each program loads one ``[BLOCK_M, D]`` query tile, streams
    every ``[BLOCK_N, D]`` K/V tile, and writes the matching output tile.

    Args:
        query_ptr: Base pointer for logical queries ``[B, L, H, D]``.
        key_ptr: Base pointer for logical keys ``[B, S, H, D]``.
        value_ptr: Base pointer for logical values ``[B, S, H, D]``.
        output_ptr: Base pointer for logical output ``[B, L, H, D]``.
        query_stride_b: Query batch stride in elements.
        query_stride_h: Query head stride in elements.
        query_stride_l: Query-token stride in elements.
        query_stride_d: Query-feature stride in elements.
        key_stride_b: Key batch stride in elements.
        key_stride_h: Key head stride in elements.
        key_stride_s: Key-token stride in elements.
        key_stride_d: Key-feature stride in elements.
        value_stride_b: Value batch stride in elements.
        value_stride_h: Value head stride in elements.
        value_stride_s: Value-token stride in elements.
        value_stride_d: Value-feature stride in elements.
        output_stride_b: Output batch stride in elements.
        output_stride_h: Output head stride in elements.
        output_stride_l: Output-token stride in elements.
        output_stride_d: Output-feature stride in elements.
        num_heads: Number of attention heads.
        query_length: Logical query-token count ``L``.
        key_length: Logical key/value-token count ``S``.
        scale: Multiplier applied to QK scores before softmax.
        HEAD_DIM: Compile-time head width ``D``.
        QUANTIZED_SDPA: Whether Q/K/V and P use FP8 e4m3.
        BLOCK_M: Compile-time number of query rows owned by one program.
        BLOCK_N: Compile-time number of key/value rows loaded per iteration.
    """
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_base = query_ptr + batch * query_stride_b + head * query_stride_h
    key_base = key_ptr + batch * key_stride_b + head * key_stride_h
    value_base = value_ptr + batch * value_stride_b + head * value_stride_h
    output_base = output_ptr + batch * output_stride_b + head * output_stride_h

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = tl.arange(0, HEAD_DIM)
    query_mask = query_offsets < query_length
    query = tl.load(
        query_base
        + query_offsets[:, None] * query_stride_l
        + feature_offsets[None, :] * query_stride_d,
        mask=query_mask[:, None],
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    qk_scale = scale.to(tl.float32) * 1.4426950408889634

    for key_start in tl.range(0, key_length, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        key = tl.load(
            key_base
            + key_offsets[:, None] * key_stride_s
            + feature_offsets[None, :] * key_stride_d,
            mask=key_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key)) * qk_scale
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        next_row_max = tl.maximum(row_max, tile_max)
        correction = tl.exp2(row_max - next_row_max)
        probabilities = tl.exp2(scores - next_row_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)

        value = tl.load(
            value_base
            + key_offsets[:, None] * value_stride_s
            + feature_offsets[None, :] * value_stride_d,
            mask=key_mask[:, None],
            other=0.0,
        )
        accumulator *= correction[:, None]
        if QUANTIZED_SDPA:
            probabilities = probabilities.to(tl.float8e4nv)
        else:
            probabilities = probabilities.to(value.dtype)
        accumulator = tl.dot(probabilities, value, accumulator)
        row_max = next_row_max

    output = accumulator / denominator[:, None]
    tl.store(
        output_base
        + query_offsets[:, None] * output_stride_l
        + feature_offsets[None, :] * output_stride_d,
        output,
        mask=query_mask[:, None],
    )


def flash_attention_2(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Apply non-causal pointer-based FlashAttention2 to Q/K/V tensors.

    Compute ``softmax(scale * Q @ K.T) @ V`` independently for every
    batch/head plane without dropout or materializing the complete score matrix.
    Empty batch, head, or query axes return an empty output, but the key/value
    sequence axis must be positive.

    Args:
        query: CUDA FP16, BF16, or FP8 e4m3 query tensor with shape
            ``[B, L, H, D]``.
        key: Same-device and same-dtype key tensor with shape
            ``[B, S, H, D]``.
        value: Value tensor matching ``key`` exactly.
        scale: Multiplier applied to QK scores before softmax; ``None`` uses
            ``1 / sqrt(D)``.
        output_dtype: Output storage dtype; ``None`` uses ``query.dtype``.

    Returns:
        Attention result with shape ``[B, L, H, D]`` on the query device and
        in ``output_dtype``.

    Raises:
        ValueError: Q/K/V shapes are incompatible or contain an empty key axis.
        RuntimeError: Placement, dtype, head geometry, or strides do not satisfy
            the pointer-kernel contract.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, L, H, D]")
    batch_size, query_length, num_heads, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[2:] != (num_heads, head_dim):
        raise ValueError("query and key batch, head, and feature dimensions differ")
    if value.shape != key.shape:
        raise ValueError("key and value must have identical shapes")
    key_length = key.shape[1]
    if key_length == 0:
        raise ValueError("key and value sequence length must be positive")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("FlashAttention2 requires CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise RuntimeError("query, key, and value must occupy the same CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise RuntimeError("query, key, and value must have the same dtype")
    if query.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise RuntimeError("FlashAttention2 requires FP16, BF16, or FP8 e4m3 tensors")
    if not (16 <= head_dim <= 256 and head_dim & (head_dim - 1) == 0):
        raise RuntimeError(
            "FlashAttention2 requires a power-of-two head_dim in [16, 256]"
        )
    if any(stride <= 0 for x in (query, key, value) for stride in x.stride()):
        raise RuntimeError("FlashAttention2 requires positive tensor strides")

    if output_dtype is None:
        output_dtype = query.dtype
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise RuntimeError("FlashAttention2 requires an FP16, BF16, or FP8 e4m3 output")
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    if batch_size == 0 or num_heads == 0 or query_length == 0:
        return output

    query_strides = (
        query.stride(0),
        query.stride(2),
        query.stride(1),
        query.stride(3),
    )
    key_strides = (key.stride(0), key.stride(2), key.stride(1), key.stride(3))
    value_strides = (
        value.stride(0),
        value.stride(2),
        value.stride(1),
        value.stride(3),
    )
    output_strides = (
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),
    )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        """Build the two-dimensional launch grid for an autotuned query tile.

        Args:
            meta: Autotuning metadata containing ``BLOCK_M``.

        Returns:
            Query-tile count and flattened batch/head plane count.
        """
        return (
            triton.cdiv(query_length, meta["BLOCK_M"]),
            batch_size * num_heads,
        )

    _flash_attention_2_kernel[grid](
        query,
        key,
        value,
        output,
        *query_strides,
        *key_strides,
        *value_strides,
        *output_strides,
        num_heads,
        query_length,
        key_length,
        1.0 / math.sqrt(head_dim) if scale is None else scale,
        HEAD_DIM=head_dim,
        QUANTIZED_SDPA=query.dtype is torch.float8_e4m3fn,
    )
    return output


__all__ = ["flash_attention_2"]
