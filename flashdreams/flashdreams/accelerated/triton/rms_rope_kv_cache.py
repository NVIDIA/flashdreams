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

"""Fused Q/K RMS normalization, RoPE, and K/V-cache preprocessing.

The public wrapper consumes projected Q/K/V tensors with logical layout
``[B, L, H, D]``. It returns every processed query, then maps a selected source
interval on ``L`` into an in-place destination interval on cache axis ``S``.
Keys are normalized and rotated before storage; values are stored unchanged.
Returned Q and cached K/V use the cache dtype, including E4M3 storage for the
FP8 attention path, while RMS reductions and trigonometry use higher precision.

Three launch topologies match the preprocessing domain. Head scope uses one
program per ``[D]`` head without RoPE and tiles heads when RoPE can share its
trigonometry. Inner and disabled normalization use one program for the packed
``[H * D]`` token width; the disabled scope compiles out the RMS reduction.
Shape comments use ``B`` for batch, ``L`` for current tokens, ``S`` for cache
capacity, ``H`` for heads, and ``D`` for the head dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl
from flashdreams.accelerated.multi_head_attention import QKNormScope


@triton.jit
def _fused_head_rms_rope_kv_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_stride_b,
    query_stride_l,
    query_stride_h,
    query_stride_d,
    key_stride_b,
    key_stride_l,
    key_stride_h,
    key_stride_d,
    value_stride_b,
    value_stride_l,
    value_stride_h,
    value_stride_d,
    query_output_stride_b,
    query_output_stride_l,
    query_output_stride_h,
    query_output_stride_d,
    key_cache_stride_b,
    key_cache_stride_l,
    key_cache_stride_h,
    key_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_l,
    value_cache_stride_h,
    value_cache_stride_d,
    rope_stride_l,
    rope_stride_d,
    sequence_length,
    num_heads,
    cache_read_start,
    cache_write_start,
    cache_write_length,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Process one query/key head and write its current cache slice.

    Q/K/V and processed Q have logical shape ``[B, L, H, D]``; caches have
    logical shape ``[B, S, H, D]``. All supplied strides are element counts,
    so the same pointer arithmetic supports token-major tensors and the
    metadata-only head-major cache view accepted by the wrapper.

    The one-dimensional grid contains ``B * L * H`` programs. Each program
    reduces and rotates one ``[D]`` vector, stores its processed query, and
    conditionally writes its key/value vector when the token belongs to the
    selected cache interval. ``BLOCK_D`` is the power-of-two lane count covering
    ``D``; lanes beyond ``D`` are masked. ``APPLY_NORM``, ``APPLY_ROPE``, and
    ``INTERLEAVED`` are compile-time policies, so disabled work and its pointer
    loads are removed from the generated kernel.
    """
    # One program owns a (batch, token, head) vector: exactly the reduction
    # domain for head-scoped RMSNorm.
    program = tl.program_id(0)
    head = program % num_heads
    token_program = program // num_heads
    token = token_program % sequence_length
    batch = token_program // sequence_length

    # Mask padded lanes in ``[BLOCK_D]`` while loading one physical ``[D]``
    # head. Explicit element strides permit non-contiguous batch, token, and
    # head axes; only the feature axis is required to be contiguous.
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query_base = (
        query_ptr
        + batch * query_stride_b
        + token * query_stride_l
        + head * query_stride_h
    )
    key_base = (
        key_ptr + batch * key_stride_b + token * key_stride_l + head * key_stride_h
    )
    value_base = (
        value_ptr
        + batch * value_stride_b
        + token * value_stride_l
        + head * value_stride_h
    )
    query = tl.load(
        query_base + dim_offsets * query_stride_d,
        mask=dim_mask,
        other=0.0,
    )
    key = tl.load(
        key_base + dim_offsets * key_stride_d,
        mask=dim_mask,
        other=0.0,
    )

    if APPLY_NORM:
        # Reduce RMS statistics in FP32, then cast back after applying the
        # learned ``[D]`` weight so the result matches the projection dtype.
        query_scale = tl.rsqrt(
            tl.sum(query.to(tl.float32) * query.to(tl.float32), axis=0) / HEAD_DIM + EPS
        )
        key_scale = tl.rsqrt(
            tl.sum(key.to(tl.float32) * key.to(tl.float32), axis=0) / HEAD_DIM + EPS
        )
        query_weight = tl.load(
            query_weight_ptr + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        key_weight = tl.load(
            key_weight_ptr + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        query = (query * query_scale * query_weight).to(query.dtype)
        key = (key * key_scale * key_weight).to(key.dtype)

    if APPLY_ROPE:
        # Map every feature lane to its rotation partner within the same head:
        # adjacent pairs for interleaved RoPE, otherwise matching half splits.
        if INTERLEAVED:
            partner_offsets = tl.where(
                dim_offsets % 2 == 0,
                dim_offsets + 1,
                dim_offsets - 1,
            )
            rotation_sign = tl.where(dim_offsets % 2 == 0, -1.0, 1.0)
        else:
            partner_offsets = tl.where(
                dim_offsets < HEAD_DIM // 2,
                dim_offsets + HEAD_DIM // 2,
                dim_offsets - HEAD_DIM // 2,
            )
            rotation_sign = tl.where(dim_offsets < HEAD_DIM // 2, -1.0, 1.0)

        # Reload the unnormalized partner and apply its own affine weight plus
        # the head's shared scalar RMS scale. This avoids materializing the
        # complete normalized Q/K tensors before rotation.
        query_partner = tl.load(
            query_base + partner_offsets * query_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        key_partner = tl.load(
            key_base + partner_offsets * key_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        if APPLY_NORM:
            query_partner_weight = tl.load(
                query_weight_ptr + partner_offsets,
                mask=dim_mask,
                other=0.0,
            )
            key_partner_weight = tl.load(
                key_weight_ptr + partner_offsets,
                mask=dim_mask,
                other=0.0,
            )
            query_partner = (query_partner * query_scale * query_partner_weight).to(
                query_partner.dtype
            )
            key_partner = (key_partner * key_scale * key_partner_weight).to(
                key_partner.dtype
            )

        # RoPE angles ``[L, D]`` are shared across batch and head axes.
        frequencies = tl.load(
            rope_freqs_ptr + token * rope_stride_l + dim_offsets * rope_stride_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        cos_freqs = tl.cos(frequencies).to(query.dtype)
        sin_freqs = tl.sin(frequencies).to(query.dtype)
        query = query * cos_freqs + query_partner * sin_freqs * rotation_sign
        key = key * cos_freqs + key_partner * sin_freqs * rotation_sign

    # Store processed Q for every token. The destination pointer determines the
    # final attention-storage dtype, so this store also performs the E4M3 cast.
    query_output_base = (
        query_output_ptr
        + batch * query_output_stride_b
        + token * query_output_stride_l
        + head * query_output_stride_h
    )
    tl.store(
        query_output_base + dim_offsets * query_output_stride_d,
        query,
        mask=dim_mask,
    )

    # Translate source token ``t`` to destination
    # ``cache_write_start + (t - cache_read_start)``. A nonzero read start occurs
    # when an immutable sink clips a rolling write; the mask leaves all cache
    # positions outside that interval untouched.
    cache_offset = token - cache_read_start
    cache_token = cache_write_start + cache_offset
    cache_mask = dim_mask & (cache_offset >= 0) & (cache_offset < cache_write_length)
    key_cache_base = (
        key_cache_ptr
        + batch * key_cache_stride_b
        + cache_token * key_cache_stride_l
        + head * key_cache_stride_h
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_token * value_cache_stride_l
        + head * value_cache_stride_h
    )
    # K uses its normalized/rotated register value. V is loaded only for tokens
    # that will be written and remains otherwise unprocessed.
    value = tl.load(
        value_base + dim_offsets * value_stride_d,
        mask=cache_mask,
        other=0.0,
    )
    tl.store(
        key_cache_base + dim_offsets * key_cache_stride_d,
        key,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base + dim_offsets * value_cache_stride_d,
        value,
        mask=cache_mask,
    )


@triton.jit
def _fused_tiled_head_rms_rope_kv_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_stride_b,
    query_stride_l,
    query_stride_h,
    query_stride_d,
    key_stride_b,
    key_stride_l,
    key_stride_h,
    key_stride_d,
    value_stride_b,
    value_stride_l,
    value_stride_h,
    value_stride_d,
    query_output_stride_b,
    query_output_stride_l,
    query_output_stride_h,
    query_output_stride_d,
    key_cache_stride_b,
    key_cache_stride_l,
    key_cache_stride_h,
    key_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_l,
    value_cache_stride_h,
    value_cache_stride_d,
    rope_stride_l,
    rope_stride_d,
    sequence_length,
    num_heads,
    cache_read_start,
    cache_write_start,
    cache_write_length,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Process a head tile while sharing RoPE coefficients across its heads.

    Q/K/V and processed Q have logical shape ``[B, L, H, D]``; caches have
    logical shape ``[B, S, H, D]``. The wrapper launches this kernel only for
    head-scoped preprocessing with RoPE and a power-of-two ``D``. Consequently
    ``tl.arange(0, HEAD_DIM)`` covers the feature width exactly, while the final
    head tile still masks lanes whose head index exceeds ``H``.

    The three-dimensional grid is ``[B, L, ceil_div(H, BLOCK_H)]``. Each program
    computes independent ``[D]`` RMS statistics for up to ``BLOCK_H`` heads but
    evaluates the token's sine and cosine vectors once for the whole tile.
    Explicit element strides support both accepted physical cache layouts.
    Processed queries are always stored; K/V writes are masked to the selected
    source and destination cache intervals.
    """
    # Grid axes directly encode batch, token, and head tile, avoiding integer
    # division in the hot tiled-RoPE path.
    batch = tl.program_id(0)
    token = tl.program_id(1)
    head_tile = tl.program_id(2)

    # ``HEAD_DIM`` is exact for this dispatch, so only the potentially partial
    # head tile needs masking. ``[:, None]`` broadcasts that mask across D.
    head_offsets = head_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dim_offsets = tl.arange(0, HEAD_DIM)
    tensor_mask = head_mask[:, None]

    # Form a ``[BLOCK_H, D]`` address tile from independent head and feature
    # strides. This remains valid for token-major Q/K/V and head-major caches.
    query_base = query_ptr + batch * query_stride_b + token * query_stride_l
    key_base = key_ptr + batch * key_stride_b + token * key_stride_l
    value_base = value_ptr + batch * value_stride_b + token * value_stride_l
    query_offsets = (
        head_offsets[:, None] * query_stride_h + dim_offsets[None, :] * query_stride_d
    )
    key_offsets = (
        head_offsets[:, None] * key_stride_h + dim_offsets[None, :] * key_stride_d
    )
    value_offsets = (
        head_offsets[:, None] * value_stride_h + dim_offsets[None, :] * value_stride_d
    )
    query = tl.load(query_base + query_offsets, mask=tensor_mask, other=0.0)
    key = tl.load(key_base + key_offsets, mask=tensor_mask, other=0.0)

    if APPLY_NORM:
        # Axis 1 is D, so every head receives its own FP32 RMS statistic while
        # the learned ``[D]`` affine vector broadcasts across the head tile.
        query_float = query.to(tl.float32)
        key_float = key.to(tl.float32)
        query_scale = tl.rsqrt(
            tl.sum(query_float * query_float, axis=1) / HEAD_DIM + EPS
        )
        key_scale = tl.rsqrt(tl.sum(key_float * key_float, axis=1) / HEAD_DIM + EPS)
        query_weight = tl.load(query_weight_ptr + dim_offsets)
        key_weight = tl.load(key_weight_ptr + dim_offsets)
        query = (query * query_scale[:, None] * query_weight[None, :]).to(query.dtype)
        key = (key * key_scale[:, None] * key_weight[None, :]).to(key.dtype)

    # RoPE accepts one angle per feature lane. Materialize trigonometry once per
    # token and share it across all heads in this tile. Reshaping exposes either
    # adjacent ``(2i, 2i + 1)`` pairs or half-split ``(i, i + D / 2)`` pairs.
    frequencies = tl.load(
        rope_freqs_ptr + token * rope_stride_l + dim_offsets * rope_stride_d
    ).to(tl.float32)
    if INTERLEAVED:
        frequency_pairs = tl.reshape(frequencies, (HEAD_DIM // 2, 2))
    else:
        frequency_pairs = tl.reshape(
            frequencies,
            (2, HEAD_DIM // 2),
        ).permute(1, 0)
    frequencies_a, frequencies_b = tl.split(frequency_pairs)
    cos_a = tl.cos(frequencies_a).to(query.dtype)[None, :]
    sin_a = tl.sin(frequencies_a).to(query.dtype)[None, :]
    cos_b = tl.cos(frequencies_b).to(query.dtype)[None, :]
    sin_b = tl.sin(frequencies_b).to(query.dtype)[None, :]

    # Apply ``(a, b) -> (a cos - b sin, b cos + a sin)`` without reloading
    # partners: both members of every pair are already resident in registers.
    if INTERLEAVED:
        query_pairs = tl.reshape(query, (BLOCK_H, HEAD_DIM // 2, 2))
        key_pairs = tl.reshape(key, (BLOCK_H, HEAD_DIM // 2, 2))
        query_a, query_b = tl.split(query_pairs)
        key_a, key_b = tl.split(key_pairs)
        query = tl.reshape(
            tl.join(
                query_a * cos_a - query_b * sin_a,
                query_b * cos_b + query_a * sin_b,
            ),
            (BLOCK_H, HEAD_DIM),
        )
        key = tl.reshape(
            tl.join(
                key_a * cos_a - key_b * sin_a,
                key_b * cos_b + key_a * sin_b,
            ),
            (BLOCK_H, HEAD_DIM),
        )
    else:
        query_pairs = tl.reshape(
            query,
            (BLOCK_H, 2, HEAD_DIM // 2),
        ).permute(0, 2, 1)
        key_pairs = tl.reshape(
            key,
            (BLOCK_H, 2, HEAD_DIM // 2),
        ).permute(0, 2, 1)
        query_a, query_b = tl.split(query_pairs)
        key_a, key_b = tl.split(key_pairs)
        query = tl.reshape(
            tl.join(
                query_a * cos_a - query_b * sin_a,
                query_b * cos_b + query_a * sin_b,
            ).permute(0, 2, 1),
            (BLOCK_H, HEAD_DIM),
        )
        key = tl.reshape(
            tl.join(
                key_a * cos_a - key_b * sin_a,
                key_b * cos_b + key_a * sin_b,
            ).permute(0, 2, 1),
            (BLOCK_H, HEAD_DIM),
        )

    # Query storage is independent of cache slicing and may cast to E4M3 when
    # ``query_output_ptr`` uses the FP8 cache dtype.
    query_output_base = (
        query_output_ptr + batch * query_output_stride_b + token * query_output_stride_l
    )
    query_output_offsets = (
        head_offsets[:, None] * query_output_stride_h
        + dim_offsets[None, :] * query_output_stride_d
    )
    tl.store(
        query_output_base + query_output_offsets,
        query,
        mask=tensor_mask,
    )

    # Apply the same source-to-destination token translation as the generic
    # kernel. The scalar token predicate broadcasts across ``[BLOCK_H, D]``.
    cache_offset = token - cache_read_start
    cache_token = cache_write_start + cache_offset
    cache_token_mask = (cache_offset >= 0) & (cache_offset < cache_write_length)
    cache_mask = tensor_mask & cache_token_mask
    key_cache_base = (
        key_cache_ptr + batch * key_cache_stride_b + cache_token * key_cache_stride_l
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_token * value_cache_stride_l
    )
    key_cache_offsets = (
        head_offsets[:, None] * key_cache_stride_h
        + dim_offsets[None, :] * key_cache_stride_d
    )
    value_cache_offsets = (
        head_offsets[:, None] * value_cache_stride_h
        + dim_offsets[None, :] * value_cache_stride_d
    )
    # Store processed K and untouched V; destination pointer types perform any
    # native-to-E4M3 cache conversion.
    value = tl.load(
        value_base + value_offsets,
        mask=cache_mask,
        other=0.0,
    )
    tl.store(
        key_cache_base + key_cache_offsets,
        key,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base + value_cache_offsets,
        value,
        mask=cache_mask,
    )


@triton.jit
def _fused_inner_rms_rope_kv_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_stride_b,
    query_stride_l,
    key_stride_b,
    key_stride_l,
    value_stride_b,
    value_stride_l,
    query_output_stride_b,
    query_output_stride_l,
    key_cache_stride_b,
    key_cache_stride_l,
    value_cache_stride_b,
    value_cache_stride_l,
    rope_stride_l,
    rope_stride_d,
    sequence_length,
    cache_read_start,
    cache_write_start,
    cache_write_length,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INNER_DIM: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    """Process one full-inner-width query/key token and update its cache.

    Q/K/V and processed Q use logical shape ``[B, L, H, D]`` but are traversed
    as packed ``[B, L, H * D]`` tensors. The kernel receives only batch/token
    strides because ``+ inner_offsets`` relies on the physical ``H`` and ``D``
    axes being contiguous. Caches use token-major ``[B, S, H, D]`` storage with
    the same packed inner width.

    The grid has ``B * L`` programs. Each program optionally reduces one
    ``[H * D]`` RMS domain, rotates pairs independently within each ``D``-wide
    head, stores processed Q, and conditionally writes the selected K/V cache
    token. ``BLOCK_INNER`` is the power-of-two lane count covering ``H * D``;
    padded lanes are masked. ``QKNormScope.NONE`` uses this topology with
    ``APPLY_NORM=False``, which removes the reduction and affine loads at compile
    time.
    """
    # Inner-scoped RMSNorm couples all heads, so one program owns the complete
    # projected width for one (batch, token) pair.
    program = tl.program_id(0)
    token = program % sequence_length
    batch = program // sequence_length

    # Load packed Q/K ``[H * D]``. The wrapper verifies ``stride(H) == D`` and
    # ``stride(D) == 1``; padded ``BLOCK_INNER`` lanes are masked.
    inner_offsets = tl.arange(0, BLOCK_INNER)
    inner_mask = inner_offsets < INNER_DIM
    query_base = query_ptr + batch * query_stride_b + token * query_stride_l
    key_base = key_ptr + batch * key_stride_b + token * key_stride_l
    value_base = value_ptr + batch * value_stride_b + token * value_stride_l
    query = tl.load(query_base + inner_offsets, mask=inner_mask, other=0.0)
    key = tl.load(key_base + inner_offsets, mask=inner_mask, other=0.0)

    if APPLY_NORM:
        # One FP32 RMS reduction spans all heads, followed by learned
        # ``[H * D]`` weights for Q and K.
        query_scale = tl.rsqrt(
            tl.sum(query.to(tl.float32) * query.to(tl.float32), axis=0) / INNER_DIM
            + EPS
        )
        key_scale = tl.rsqrt(
            tl.sum(key.to(tl.float32) * key.to(tl.float32), axis=0) / INNER_DIM + EPS
        )
        query_weight = tl.load(
            query_weight_ptr + inner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        key_weight = tl.load(
            key_weight_ptr + inner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        query = (query * query_scale * query_weight).to(query.dtype)
        key = (key * key_scale * key_weight).to(key.dtype)

    if APPLY_ROPE:
        # RMS may couple H * D, but RoPE never crosses a head boundary.
        # Reconstruct each flattened lane's head base and within-head partner.
        head_offsets = (inner_offsets // HEAD_DIM) * HEAD_DIM
        dim_offsets = inner_offsets % HEAD_DIM
        if INTERLEAVED:
            partner_dims = tl.where(
                dim_offsets % 2 == 0,
                dim_offsets + 1,
                dim_offsets - 1,
            )
            rotation_sign = tl.where(dim_offsets % 2 == 0, -1.0, 1.0)
        else:
            partner_dims = tl.where(
                dim_offsets < HEAD_DIM // 2,
                dim_offsets + HEAD_DIM // 2,
                dim_offsets - HEAD_DIM // 2,
            )
            rotation_sign = tl.where(dim_offsets < HEAD_DIM // 2, -1.0, 1.0)
        partner_offsets = head_offsets + partner_dims

        query_partner = tl.load(
            query_base + partner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        key_partner = tl.load(
            key_base + partner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        if APPLY_NORM:
            query_partner_weight = tl.load(
                query_weight_ptr + partner_offsets,
                mask=inner_mask,
                other=0.0,
            )
            key_partner_weight = tl.load(
                key_weight_ptr + partner_offsets,
                mask=inner_mask,
                other=0.0,
            )
            query_partner = (query_partner * query_scale * query_partner_weight).to(
                query_partner.dtype
            )
            key_partner = (key_partner * key_scale * key_partner_weight).to(
                key_partner.dtype
            )

        # Convert flattened lanes back to feature indices so ``[L, D]`` RoPE
        # angles broadcast identically across every head.
        frequencies = tl.load(
            rope_freqs_ptr + token * rope_stride_l + dim_offsets * rope_stride_d,
            mask=inner_mask,
            other=0.0,
        ).to(tl.float32)
        cos_freqs = tl.cos(frequencies).to(query.dtype)
        sin_freqs = tl.sin(frequencies).to(query.dtype)
        query = query * cos_freqs + query_partner * sin_freqs * rotation_sign
        key = key * cos_freqs + key_partner * sin_freqs * rotation_sign

    # Store every processed query lane; the output pointer casts to the cache
    # dtype, including E4M3 for the FP8 attention-storage path.
    query_output_base = (
        query_output_ptr + batch * query_output_stride_b + token * query_output_stride_l
    )
    tl.store(query_output_base + inner_offsets, query, mask=inner_mask)

    # Map the selected source interval ``[cache_read_start, ...]`` to the
    # physical cache interval beginning at ``cache_write_start``. The scalar
    # token predicate masks every lane when this program is outside the slice.
    cache_offset = token - cache_read_start
    cache_token = cache_write_start + cache_offset
    cache_mask = inner_mask & (cache_offset >= 0) & (cache_offset < cache_write_length)
    key_cache_base = (
        key_cache_ptr + batch * key_cache_stride_b + cache_token * key_cache_stride_l
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_token * value_cache_stride_l
    )
    # Cache processed K beside unnormalized, unrotated V. Stores cast both to
    # the cache pointer dtype.
    value = tl.load(value_base + inner_offsets, mask=cache_mask, other=0.0)
    tl.store(key_cache_base + inner_offsets, key, mask=cache_mask)
    tl.store(value_cache_base + inner_offsets, value, mask=cache_mask)


def fused_rms_rope_kv_cache_update(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    *,
    query_weight: Tensor | None,
    key_weight: Tensor | None,
    norm_eps: float,
    norm_scope: QKNormScope,
    rope_freqs: Tensor | None,
    rope_interleaved: bool,
    cache_read_start: int,
    cache_write_start: int,
    cache_write_length: int,
) -> Tensor:
    """Normalize and rotate Q/K while writing the current K/V cache slice.

    Normalization precedes RoPE. Processed K and unchanged V from source slice
    ``[cache_read_start:cache_read_start + cache_write_length]`` are cast to the
    cache dtype and written beginning at ``cache_write_start``. Every token still
    produces processed Q, including tokens outside the cache-write slice.

    ``QKNormScope.HEAD`` computes one RMS statistic per ``[D]`` head and uses a
    shared ``[D]`` affine weight. ``QKNormScope.INNER`` computes one statistic
    over each packed ``[H * D]`` token and uses an ``[H * D]`` affine weight.
    ``QKNormScope.NONE`` applies no normalization and requires both weights to be
    absent. RoPE always pairs features within a head, regardless of normalization
    scope. This inference primitive writes caches in place and provides no
    autograd implementation.

    Args:
        query: Projected queries with logical shape ``[B, L, H, D]``. The feature
            axis must be contiguous; inner and disabled normalization also require
            the head axis to be packed with stride ``D``.
        key: Projected keys matching ``query`` in shape, dtype, device, and
            required physical layout.
        value: Projected values matching ``query`` in shape, dtype, device, and
            required physical layout. Values are neither normalized nor rotated.
        key_cache: Mutable dense storage with logical shape ``[B, S, H, D]``.
            Token-major storage is accepted for every scope. Head scope also
            accepts a logical view whose ``transpose(1, 2)`` is contiguous,
            corresponding to physical ``[B, H, S, D]`` storage.
        value_cache: Mutable storage matching ``key_cache`` in shape, dtype,
            device, and physical layout.
        query_weight: Contiguous RMSNorm query weight with shape ``[D]`` for head
            scope or ``[H * D]`` for inner scope; ``None`` for disabled
            normalization.
        key_weight: RMSNorm key weight with the same shape as ``query_weight``;
            ``None`` for disabled normalization. Its device and dtype must match
            Q/K.
        norm_eps: Epsilon used by Q/K RMS normalization; ignored when disabled.
        norm_scope: Normalize each head, normalize the complete projected inner
            width, or disable normalization.
        rope_freqs: Optional full-width RoPE angles with shape ``[L, 1, 1, D]``.
            Angles broadcast across batch and head axes and are converted to FP32
            before evaluating sine and cosine.
        rope_interleaved: Pair adjacent features when ``True``; otherwise pair
            matching positions in the first and second halves of each head.
            Ignored when ``rope_freqs`` is ``None``.
        cache_read_start: First token on the ``L`` axis copied into the cache.
        cache_write_start: First token on the cache ``S`` axis written.
        cache_write_length: Number of consecutive source tokens written. Zero
            computes all processed queries without changing either cache.

    Returns:
        Contiguous processed queries with shape ``[B, L, H, D]`` and the cache
        dtype. Processed K and unchanged V are written in place to their caches.

    Raises:
        ValueError: Tensor shapes, cache bounds, RoPE geometry, or normalization
            inputs violate the kernel contract.
        TypeError: ``norm_scope`` is not a :class:`QKNormScope`.
        RuntimeError: Tensor devices, dtypes, strides, or cache storage layouts
            are incompatible with the kernels.
    """
    # Validate logical Q/K/V ``[B, L, H, D]`` and cache ``[B, S, H, D]``
    # geometry before checking storage properties used by the kernels.
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key, and value must have identical [B, L, H, D] shapes"
        )
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError(
            "key_cache and value_cache must have identical [B, S, H, D] shapes"
        )
    batch_size, sequence_length, num_heads, head_dim = query.shape
    if key_cache.shape[0] != batch_size or key_cache.shape[2:] != (
        num_heads,
        head_dim,
    ):
        raise ValueError("cache batch, head, and feature dimensions differ from Q/K/V")
    tensors = (query, key, value, key_cache, value_cache)
    if not all(x.is_cuda for x in tensors):
        raise RuntimeError("fused RMS/RoPE/cache update requires CUDA tensors")
    if len({x.device for x in tensors}) != 1:
        raise RuntimeError("Q/K/V and cache tensors must share a device")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError("Q/K/V tensors must share a dtype")
    if key_cache.dtype != value_cache.dtype:
        raise RuntimeError("K/V cache tensors must share a dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused RMS/RoPE/cache update requires FP16 or BF16")
    if key_cache.dtype not in (query.dtype, torch.float8_e4m3fn):
        raise RuntimeError(
            "K/V cache storage must match Q/K/V or use torch.float8_e4m3fn"
        )
    if not isinstance(norm_scope, QKNormScope):
        raise TypeError(f"norm_scope must be a QKNormScope; got {norm_scope!r}")

    # Token-major cache strides are ``[S * H * D, H * D, D, 1]``. Head scope
    # additionally accepts a logical ``[B, S, H, D]`` transpose view over dense
    # ``[B, H, S, D]`` storage; transposing S/H back must therefore be contiguous.
    cache_size = key_cache.shape[1]
    token_major = key_cache.is_contiguous() and value_cache.is_contiguous()
    head_major = (
        norm_scope is QKNormScope.HEAD
        and key_cache.transpose(1, 2).is_contiguous()
        and value_cache.transpose(1, 2).is_contiguous()
    )
    if not token_major and not head_major:
        raise RuntimeError(
            "K/V cache storage must be dense token-major, or head-major for "
            "head-scoped RMSNorm"
        )
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise RuntimeError("Q/K/V feature dimensions must be contiguous")

    # The enum is the normalization policy; weight presence must agree with it.
    apply_norm = norm_scope is not QKNormScope.NONE
    if apply_norm and (query_weight is None or key_weight is None):
        raise ValueError("HEAD and INNER normalization require Q/K weights")
    if not apply_norm and (query_weight is not None or key_weight is not None):
        raise ValueError("NONE normalization requires absent Q/K weights")
    inner_dim = num_heads * head_dim
    if apply_norm:
        assert query_weight is not None and key_weight is not None
        expected_weight_size = head_dim if norm_scope is QKNormScope.HEAD else inner_dim
        if query_weight.shape != (expected_weight_size,) or key_weight.shape != (
            expected_weight_size,
        ):
            raise ValueError(
                f"RMSNorm weights must have shape ({expected_weight_size},)"
            )
        if (
            query_weight.device != query.device
            or key_weight.device != query.device
            or query_weight.dtype != query.dtype
            or key_weight.dtype != query.dtype
        ):
            raise RuntimeError("RMSNorm weights must match the Q/K device and dtype")

    # RoPE stores one full-width ``[D]`` angle vector per current token; batch
    # and head axes broadcast inside both kernels.
    if rope_freqs is not None:
        expected_rope_shape = (sequence_length, 1, 1, head_dim)
        if tuple(rope_freqs.shape) != expected_rope_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_rope_shape}; "
                f"got {tuple(rope_freqs.shape)}"
            )
        if rope_freqs.device != query.device:
            raise RuntimeError("rope_freqs must be on the Q/K device")
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

    # The source ``L`` interval and destination ``S`` interval must each fit;
    # they may begin at different offsets when a rolling cache preserves sinks.
    if not (0 <= cache_read_start <= sequence_length):
        raise ValueError("cache_read_start is outside the current sequence")
    if not (0 <= cache_write_start <= cache_size):
        raise ValueError("cache_write_start is outside the cache")
    if cache_write_length < 0:
        raise ValueError("cache_write_length must be non-negative")
    if cache_read_start + cache_write_length > sequence_length:
        raise ValueError("cache source slice exceeds the current sequence")
    if cache_write_start + cache_write_length > cache_size:
        raise ValueError("cache destination slice exceeds cache storage")

    # Match processed Q to cache storage so attention reads Q/K/V uniformly.
    # Triton performs arithmetic in the native Q/K dtype and casts these stores
    # to E4M3 only when the cache uses FP8 storage.
    query_output = torch.empty(
        query.shape,
        device=query.device,
        dtype=key_cache.dtype,
    )
    if batch_size == 0 or sequence_length == 0 or num_heads == 0:
        return query_output

    # Triton launch arguments cannot be None. Safe tensor aliases occupy unused
    # pointer slots; compile-time APPLY_NORM/APPLY_ROPE remove their loads.
    query_weight_ptr = query if query_weight is None else query_weight
    key_weight_ptr = key if key_weight is None else key_weight
    rope_pointer = query if rope_freqs is None else rope_freqs
    # These constexpr values produce separate specialized kernels for each
    # normalization, RoPE, and pairing policy instead of runtime branches.
    common_meta = {
        "EPS": norm_eps,
        "HEAD_DIM": head_dim,
        "APPLY_NORM": apply_norm,
        "APPLY_ROPE": rope_freqs is not None,
        "INTERLEAVED": rope_interleaved,
    }
    if norm_scope is QKNormScope.HEAD:
        if rope_freqs is not None and head_dim & (head_dim - 1) == 0:
            # Power-of-two D permits exact register reshapes into RoPE pairs.
            # Tile heads so one trigonometry pass serves several independent RMS
            # domains without introducing padded feature lanes.
            block_h = min(16, int(triton.next_power_of_2(num_heads)))
            head_tiles = triton.cdiv(num_heads, block_h)
            grid = (batch_size, sequence_length, head_tiles)
            _fused_tiled_head_rms_rope_kv_cache_kernel[grid](
                query,
                key,
                value,
                query_output,
                key_cache,
                value_cache,
                query_weight_ptr,
                key_weight_ptr,
                rope_pointer,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *query_output.stride(),
                *key_cache.stride(),
                *value_cache.stride(),
                rope_pointer.stride(0),
                rope_pointer.stride(-1),
                sequence_length,
                num_heads,
                cache_read_start,
                cache_write_start,
                cache_write_length,
                EPS=norm_eps,
                HEAD_DIM=head_dim,
                APPLY_NORM=apply_norm,
                INTERLEAVED=rope_interleaved,
                BLOCK_H=block_h,
                num_warps=4,
                num_stages=1,
            )
        else:
            # Without RoPE there is no trigonometry to share. A non-power-of-two
            # D also needs the generic kernel's padded feature mask.
            block_d = max(16, int(triton.next_power_of_2(head_dim)))
            grid = (batch_size * sequence_length * num_heads,)
            _fused_head_rms_rope_kv_cache_kernel[grid](
                query,
                key,
                value,
                query_output,
                key_cache,
                value_cache,
                query_weight_ptr,
                key_weight_ptr,
                rope_pointer,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *query_output.stride(),
                *key_cache.stride(),
                *value_cache.stride(),
                rope_pointer.stride(0),
                rope_pointer.stride(-1),
                sequence_length,
                num_heads,
                cache_read_start,
                cache_write_start,
                cache_write_length,
                BLOCK_D=block_d,
                **common_meta,
                num_warps=4,
                num_stages=1,
            )
    else:
        # INNER and NONE share the contiguous inner-width topology. NONE sets
        # APPLY_NORM=False, so compile-time specialization removes the reduction.
        # Flattened loads require H and D to form one contiguous physical span.
        if not all(
            x.stride(-2) == head_dim and x.stride(-1) == 1 for x in (query, key, value)
        ):
            raise RuntimeError(
                "inner-width preprocessing requires contiguous head and feature dimensions"
            )
        # One program per ``(batch, token)`` and one power-of-two block cover the
        # packed ``H * D`` domain. Larger widths use more warps to parallelize
        # the RMS reduction and vector transforms.
        block_inner = max(16, int(triton.next_power_of_2(inner_dim)))
        grid = (batch_size * sequence_length,)
        _fused_inner_rms_rope_kv_cache_kernel[grid](
            query,
            key,
            value,
            query_output,
            key_cache,
            value_cache,
            query_weight_ptr,
            key_weight_ptr,
            rope_pointer,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            query_output.stride(0),
            query_output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            rope_pointer.stride(0),
            rope_pointer.stride(-1),
            sequence_length,
            cache_read_start,
            cache_write_start,
            cache_write_length,
            INNER_DIM=inner_dim,
            BLOCK_INNER=block_inner,
            **common_meta,
            num_warps=8 if inner_dim > 2048 else 4,
            num_stages=1,
        )
    return query_output


__all__ = ["fused_rms_rope_kv_cache_update"]
