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

"""CPU tests for call-level model-of-thought attention routing."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionMask,
    AttentionType,
    MoTRoute,
    MultiHeadAttention,
)
from flashdreams.accelerated.multi_head_attention.mot import MoTMultiHeadAttention

pytestmark = pytest.mark.ci_cpu


class _Branch(MultiHeadAttention[Tensor]):
    """Record calls and add a pathway-specific value to outputs."""

    def __init__(self, offset: float) -> None:
        """Initialize one test pathway."""
        super().__init__(
            AttentionType.CROSS_ATTENTION,
            AttentionConfig(query_dim=2, n_heads=1, head_dim=2),
        )
        self.offset = offset
        self.calls = 0
        self.projection = nn.Linear(2, 2, bias=False)
        self.norm = nn.Identity()

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

    def compute_kv(self, context: Tensor, rope_freqs: Tensor | None = None) -> Tensor:
        """Return pathway-tagged context for dispatch testing."""
        del rope_freqs
        return context + self.offset

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor,
        rope_freqs: Tensor | None = None,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> Tensor:
        """Return pathway-tagged queries and record the selected call."""
        del kv_cache, rope_freqs, attn_mask
        self.calls += 1
        return x + self.offset


def test_route_selects_only_one_homogeneous_pathway() -> None:
    """Dispatch each complete call without computing the other pathway."""
    understanding = _Branch(1.0)
    generation = _Branch(2.0)
    attention = MoTMultiHeadAttention(understanding, generation)
    tokens = torch.tensor([[0.0, 1.0]])

    output = attention(
        tokens,
        tokens,
        route=MoTRoute.GENERATION,
        attn_mask=torch.tensor([[True]]),
    )

    assert output.tolist() == [[2.0, 3.0]]
    assert understanding.calls == 0
    assert generation.calls == 1
    assert attention.compute_kv(tokens, route=MoTRoute.UNDERSTANDING).tolist() == [
        [1.0, 2.0]
    ]


def test_route_is_an_enum_not_a_per_token_mask() -> None:
    """Reject an untyped route instead of silently accepting mixed-token routing."""
    attention = MoTMultiHeadAttention(_Branch(1.0), _Branch(2.0))

    with pytest.raises(TypeError, match="MoTRoute"):
        attention(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
            route="generation",
        )
