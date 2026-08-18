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

"""Reference tests for fused Triton RMSNorm, RoPE, and K/V-cache updates."""

from __future__ import annotations

import pytest
import torch

from flashdreams.accelerated.multi_head_attention import QKNormScope
from flashdreams.accelerated.triton import fused_rms_rope_kv_cache_update

pytestmark = pytest.mark.ci_gpu


def _apply_rope_reference(
    x: torch.Tensor, rope_freqs: torch.Tensor, *, interleaved: bool
) -> torch.Tensor:
    """Apply full-width RoPE with independent PyTorch tensor operations.

    Compute ``x * cos(theta) + rotate(x) * sin(theta)`` using either adjacent
    feature pairs or matching positions from the two head-dimension halves.

    Args:
        x: Head activations with shape ``[B, L, H, D]``.
        rope_freqs: Rotation angles with shape ``[L, 1, 1, D]``.
        interleaved: Pair adjacent features when ``True``; otherwise pair the
            first and second halves of each head.

    Returns:
        Rotated activations with the same shape and dtype as ``x``.
    """
    # Reshape ``[L, 1, 1, D]`` angles to broadcast across batch and head axes.
    freqs = rope_freqs[:, 0, 0, :].reshape(1, x.shape[-3], 1, x.shape[-1])
    cos_freqs = freqs.cos().to(dtype=x.dtype)
    sin_freqs = freqs.sin().to(dtype=x.dtype)
    if interleaved:
        rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    else:
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    return x * cos_freqs + rotated * sin_freqs


@pytest.mark.parametrize(
    ("qk_norm_scope", "rope_interleaved", "head_major_cache", "apply_rope"),
    [
        pytest.param(QKNormScope.HEAD, False, True, True, id="half-split-head-major"),
        pytest.param(QKNormScope.HEAD, True, True, True, id="interleaved-head-major"),
        pytest.param(QKNormScope.NONE, False, False, True, id="no-norm-token-major"),
        pytest.param(QKNormScope.HEAD, False, True, False, id="no-rope-head-major"),
    ],
)
def test_head_preprocessing_layouts_match_torch(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    head_major_cache: bool,
    apply_rope: bool,
) -> None:
    """Match fused head preprocessing and sliced cache writes against PyTorch.

    Cover head-scoped and disabled normalization, both RoPE pair conventions,
    optional RoPE, and token-major or physically head-major cache storage.

    Args:
        tma_device: CUDA device satisfying the shared TMA capability gate.
        qk_norm_scope: RMS normalization scope exercised by the kernel.
        rope_interleaved: RoPE feature-pair convention.
        head_major_cache: Store cache data physically as ``[B, H, S, D]`` while
            exposing the logical ``[B, S, H, D]`` interface.
        apply_rope: Apply rotation when ``True``.
    """
    batch_size, sequence_length, num_heads, head_dim = 2, 3, 3, 16
    cache_size = 6
    generator = torch.Generator(device=tma_device).manual_seed(23)
    query = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        query.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    sentinel = -123.0
    # Build logical ``[B, S, H, D]`` caches in either contiguous token-major
    # storage or a transposed view over contiguous ``[B, H, S, D]`` storage.
    cache_shape = (
        (batch_size, num_heads, cache_size, head_dim)
        if head_major_cache
        else (batch_size, cache_size, num_heads, head_dim)
    )
    key_cache = torch.full(
        cache_shape,
        sentinel,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    if head_major_cache:
        key_cache = key_cache.transpose(1, 2)
    value_cache = torch.full_like(key_cache, sentinel)
    rope_freqs = (
        torch.arange(
            sequence_length * head_dim,
            device=tma_device,
            dtype=torch.float32,
        ).reshape(sequence_length, 1, 1, head_dim)
        / 37
    )
    query_weight = key_weight = None
    if qk_norm_scope is QKNormScope.HEAD:
        query_weight = torch.ones(head_dim, device=tma_device, dtype=torch.bfloat16)
        key_weight = torch.ones_like(query_weight)

    # Process every query token while copying source tokens ``[1:3]`` into cache
    # positions ``[2:4]``; neighboring sentinel rows must remain untouched.
    actual_query = fused_rms_rope_kv_cache_update(
        query,
        key,
        value,
        key_cache,
        value_cache,
        query_weight=query_weight,
        key_weight=key_weight,
        norm_eps=1e-6,
        norm_scope=qk_norm_scope,
        rope_freqs=rope_freqs if apply_rope else None,
        rope_interleaved=rope_interleaved,
        cache_read_start=1,
        cache_write_start=2,
        cache_write_length=2,
    )

    # Compose the reference in kernel order: RMSNorm, then RoPE. Values bypass
    # both operations and are copied directly into the selected cache slice.
    expected_query = query
    expected_key = key
    if qk_norm_scope is QKNormScope.HEAD:
        expected_query = torch.nn.functional.rms_norm(
            query, (head_dim,), weight=query_weight, eps=1e-6
        )
        expected_key = torch.nn.functional.rms_norm(
            key, (head_dim,), weight=key_weight, eps=1e-6
        )
    if apply_rope:
        expected_query = _apply_rope_reference(
            expected_query, rope_freqs, interleaved=rope_interleaved
        )
        expected_key = _apply_rope_reference(
            expected_key, rope_freqs, interleaved=rope_interleaved
        )

    # Allow BF16 reduction and trigonometric ordering differences for processed
    # Q/K; the unmodified V slice must remain exact.
    torch.testing.assert_close(actual_query, expected_query, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        key_cache[:, 2:4], expected_key[:, 1:3], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(value_cache[:, 2:4], value[:, 1:3])

    # Sentinel integrity proves that the fused store honors both write bounds.
    assert torch.all(key_cache[:, :2] == sentinel)
    assert torch.all(key_cache[:, 4:] == sentinel)
    assert torch.all(value_cache[:, :2] == sentinel)
    assert torch.all(value_cache[:, 4:] == sentinel)
