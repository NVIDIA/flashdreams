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

"""Correctness tests for Torch multi-head attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionType,
    QKNormScope,
    RoPEConfig,
    RoPEScope,
    RoPEStyle,
)
from flashdreams.accelerated.multi_head_attention.torch import TorchMultiHeadAttention

pytestmark = pytest.mark.ci_cpu


class _IdentityMHA(TorchMultiHeadAttention):
    """Provide identity projections for direct attention comparisons."""

    @property
    def query_projection(self) -> nn.Linear:
        """Return the query projection."""
        return self.projection

    @property
    def key_projection(self) -> nn.Linear:
        """Return the key projection."""
        return self.projection

    @property
    def value_projection(self) -> nn.Linear:
        """Return the value projection."""
        return self.projection

    @property
    def output_projection(self) -> nn.Linear:
        """Return the output projection."""
        return self.projection

    @property
    def query_norm(self) -> nn.Module:
        """Return identity query normalization."""
        return self.norm

    @property
    def key_norm(self) -> nn.Module:
        """Return identity key normalization."""
        return self.norm

    def __init__(
        self,
        rope_style: RoPEStyle,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
    ) -> None:
        """Initialize identity attention with after-cache RoPE.

        Args:
            rope_style: Feature pairing convention under test.
            attention_type: Relationship between query and context tokens.
        """
        super().__init__(
            attention_type,
            AttentionConfig(
                query_dim=4,
                n_heads=1,
                head_dim=4,
                qk_norm_scope=QKNormScope.NONE,
                rope_config=RoPEConfig(
                    style=rope_style,
                    scope=RoPEScope.AFTER_KV_CACHE,
                ),
            ),
        )
        self.projection = nn.Linear(4, 4, bias=False)
        self.norm = nn.Identity()
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))


def _apply_rope(x: Tensor, rope_freqs: Tensor, style: RoPEStyle) -> Tensor:
    """Apply RoPE independently for the expected attention result.

    Args:
        x: Projected query or key heads.
        rope_freqs: Rotation angles for every token in ``x``.
        style: Feature pairing convention under test.

    Returns:
        Rotated projected heads.
    """
    freqs = rope_freqs[:, 0, 0].reshape(1, x.shape[1], 1, x.shape[-1])
    if style is RoPEStyle.INTERLEAVED:
        rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    else:
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    return x * freqs.cos() + rotated * freqs.sin()


@pytest.mark.parametrize("rope_style", tuple(RoPEStyle), ids=lambda value: value.value)
@torch.inference_mode()
def test_after_kv_cache_rope_matches_visible_cache_positions(
    rope_style: RoPEStyle,
) -> None:
    """Rotate visible keys after cache fill and rolling without changing storage.

    Args:
        rope_style: Feature pairing convention under test.
    """
    attention = _IdentityMHA(rope_style)
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device="cpu",
        dtype=torch.float32,
    )
    chunks = (
        torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0]]]),
        torch.tensor([[[0.5, 1.5, -2.0, 1.0], [3.0, 0.5, 2.0, -1.0]]]),
        torch.tensor([[[-1.0, 2.5, 1.0, 0.5], [1.5, -0.5, 3.0, 2.0]]]),
    )
    rope_freqs = torch.linspace(0.1, 1.6, steps=16).reshape(4, 1, 1, 4)

    for chunk_idx, chunk in enumerate(chunks):
        cache.before_update(chunk_idx)
        write_end = cache.write_end
        write_start = write_end - cache.chunk_size
        actual = attention(chunk, cache, rope_freqs)

        visible = torch.cat(chunks[max(0, chunk_idx - 1) : chunk_idx + 1], dim=1)
        query = _apply_rope(
            chunk.reshape(1, 2, 1, 4),
            rope_freqs[write_start:write_end],
            rope_style,
        )
        key = _apply_rope(
            visible.reshape(1, visible.shape[1], 1, 4),
            rope_freqs[: visible.shape[1]],
            rope_style,
        )
        value = visible.reshape(1, visible.shape[1], 1, 4)
        expected = (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
            )
            .transpose(1, 2)
            .flatten(-2)
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(cache.cached_k(), value)
        cache.after_update(chunk_idx)


@pytest.mark.parametrize("rope_style", tuple(RoPEStyle), ids=lambda value: value.value)
@torch.inference_mode()
def test_cross_attention_after_cache_rope_allows_unequal_lengths(
    rope_style: RoPEStyle,
) -> None:
    """Slice query and key rotations independently when cross lengths differ.

    Args:
        rope_style: Feature pairing convention under test.
    """
    attention = _IdentityMHA(rope_style, AttentionType.CROSS_ATTENTION)
    query = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0], [0.5, 1.5, -2.0, 1.0]]]
    )
    context = torch.tensor([[[3.0, 0.5, 2.0, -1.0], [-1.0, 2.5, 1.0, 0.5]]])
    rope_freqs = torch.linspace(0.1, 1.2, steps=12).reshape(3, 1, 1, 4)

    cache = attention.compute_kv(context)
    actual = attention(query, cache, rope_freqs)

    rotated_query = _apply_rope(query.reshape(1, 3, 1, 4), rope_freqs[:3], rope_style)
    rotated_key = _apply_rope(context.reshape(1, 2, 1, 4), rope_freqs[:2], rope_style)
    expected = (
        F.scaled_dot_product_attention(
            rotated_query.transpose(1, 2),
            rotated_key.transpose(1, 2),
            context.reshape(1, 2, 1, 4).transpose(1, 2),
        )
        .transpose(1, 2)
        .flatten(-2)
    )

    torch.testing.assert_close(actual, expected)
