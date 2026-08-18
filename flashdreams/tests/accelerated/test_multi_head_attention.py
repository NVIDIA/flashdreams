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

"""CPU tests for the abstract multi-head attention interface."""

from __future__ import annotations

from typing import cast

import pytest
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    MultiHeadAttention,
    QKNormScope,
)

pytestmark = pytest.mark.ci_cpu


class _Attention(MultiHeadAttention[object]):
    """Behavior-free concrete attention for base-contract tests.

    The fixture implements the abstract runtime surface without projection
    modules so geometry and policy validation stay isolated from any backend.
    """

    @property
    def query_projection(self) -> nn.Linear:
        """Reject access to the intentionally absent query projection."""
        raise NotImplementedError

    @property
    def key_projection(self) -> nn.Linear:
        """Reject access to the intentionally absent key projection."""
        raise NotImplementedError

    @property
    def value_projection(self) -> nn.Linear:
        """Reject access to the intentionally absent value projection."""
        raise NotImplementedError

    @property
    def output_projection(self) -> nn.Linear:
        """Reject access to the intentionally absent output projection."""
        raise NotImplementedError

    @property
    def query_norm(self) -> nn.Module:
        """Reject access to the intentionally absent query normalization."""
        raise NotImplementedError

    @property
    def key_norm(self) -> nn.Module:
        """Reject access to the intentionally absent key normalization."""
        raise NotImplementedError

    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> object:
        """Return an opaque cache placeholder for interface tests.

        Args:
            context: Unused context tokens accepted by the abstract contract.
            rope_freqs: Unused positional data; ``None`` is also accepted.

        Returns:
            Fresh opaque object standing in for a backend cache.
        """
        del context, rope_freqs
        return object()

    def forward(
        self,
        query: Tensor,
        kv_cache: object,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Return ``query`` unchanged through the abstract runtime contract.

        Args:
            query: Input tokens passed through unchanged.
            kv_cache: Unused opaque cache placeholder.
            rope_freqs: Unused positional data; ``None`` is also accepted.

        Returns:
            Original ``query`` tensor.
        """
        del kv_cache, rope_freqs
        return query


def test_forward_and_logical_modules_are_abstract() -> None:
    """Require one runtime entry point and backend-owned logical modules."""
    # Pin the complete abstract surface so subclasses cannot silently inherit a
    # backend-specific projection or cache operation.
    assert MultiHeadAttention.__abstractmethods__ == {
        "compute_kv",
        "forward",
        "key_norm",
        "key_projection",
        "output_projection",
        "query_norm",
        "query_projection",
        "value_projection",
    }

    # Cache writes and queries remain private implementation details; callers
    # interact with attention only through ``forward``.
    assert "query_kv" not in MultiHeadAttention.__dict__
    assert "update_kv" not in MultiHeadAttention.__dict__


def test_attention_geometry_and_policies_are_stored() -> None:
    """Store default self-attention and explicit cross-attention policies."""
    # Contrast the default square self-attention geometry with an asymmetric
    # cross-attention instance that overrides every optional policy.
    self_attention = _Attention(
        query_dim=48,
        n_heads=3,
        head_dim=16,
    )
    cross_attention = _Attention(
        query_dim=48,
        n_heads=3,
        head_dim=16,
        context_dim=32,
        attention_type=AttentionType.CROSS_ATTENTION,
        qk_norm_eps=1e-5,
        qk_norm_scope=QKNormScope.INNER,
        rope_interleaved=True,
    )

    # Self-attention derives context width and inner width from query geometry.
    assert self_attention.context_dim == 48
    assert self_attention.inner_dim == 48
    assert self_attention.attention_type is AttentionType.SELF_ATTENTION
    assert self_attention.qk_norm_eps == 1e-6
    assert self_attention.qk_norm_scope is QKNormScope.HEAD
    assert self_attention.rope_interleaved is False

    # Cross-attention preserves its independent context width and explicit
    # normalization and RoPE choices.
    assert cross_attention.context_dim == 32
    assert cross_attention.attention_type is AttentionType.CROSS_ATTENTION
    assert cross_attention.qk_norm_eps == 1e-5
    assert cross_attention.qk_norm_scope is QKNormScope.INNER
    assert cross_attention.rope_interleaved is True


def test_attention_type_constructor_policy() -> None:
    """Expose stable enum values and reject invalid attention geometry."""
    # Enum values form configuration-facing strings and must remain stable.
    assert AttentionType.SELF_ATTENTION.value == "self_attention"
    assert AttentionType.CROSS_ATTENTION.value == "cross_attention"
    assert QKNormScope.NONE.value == "none"

    # Reject string lookalikes before backend construction can select the wrong
    # forward branch.
    string_policy = cast(AttentionType, "self_attention")
    with pytest.raises(TypeError, match="AttentionType"):
        _Attention(
            query_dim=16,
            n_heads=2,
            head_dim=8,
            attention_type=string_policy,
        )

    # Self-attention projects one token source, so query and context widths must
    # be identical.
    with pytest.raises(ValueError, match="self-attention requires"):
        _Attention(
            query_dim=16,
            context_dim=8,
            n_heads=2,
            head_dim=8,
        )


@pytest.mark.parametrize(
    "qk_norm_eps",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("-inf"), id="negative-infinite"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_qk_norm_eps_must_be_finite_and_positive(qk_norm_eps: float) -> None:
    """Reject non-positive or non-finite Q/K normalization epsilon values.

    Args:
        qk_norm_eps: Invalid epsilon supplied by the parameterized case.
    """
    with pytest.raises(ValueError, match="qk_norm_eps"):
        _Attention(
            query_dim=16,
            n_heads=2,
            head_dim=8,
            qk_norm_eps=qk_norm_eps,
        )


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        pytest.param("query_dim", 0, id="zero-query-dim"),
        pytest.param("context_dim", 0, id="zero-context-dim"),
        pytest.param("n_heads", 0, id="zero-heads"),
        pytest.param("head_dim", 0, id="zero-head-dim"),
    ],
)
def test_attention_dimensions_must_be_positive(dimension: str, value: int) -> None:
    """Reject a zero value for each attention geometry dimension.

    Args:
        dimension: Constructor argument under validation.
        value: Invalid zero dimension supplied to that argument.
    """
    with pytest.raises(ValueError, match=dimension):
        _Attention(
            query_dim=value if dimension == "query_dim" else 16,
            context_dim=value if dimension == "context_dim" else None,
            n_heads=value if dimension == "n_heads" else 2,
            head_dim=value if dimension == "head_dim" else 8,
        )
