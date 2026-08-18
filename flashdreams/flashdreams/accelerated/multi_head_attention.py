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

"""Multi-head attention interface and shared policy enums."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from torch import Tensor, nn

KVCacheT = TypeVar("KVCacheT")
"""Backend-owned K/V cache type passed to attention."""


class AttentionType(str, Enum):
    """Relationship between query tokens and cached K/V context."""

    SELF_ATTENTION = "self_attention"
    """Update K/V from each query chunk before attention."""

    CROSS_ATTENTION = "cross_attention"
    """Query a precomputed static K/V cache without updating it."""


class QKNormScope(str, Enum):
    """Feature scope for query and key normalization."""

    NONE = "none"
    """Skip query and key normalization."""

    HEAD = "head"
    """Normalize each attention head independently."""

    INNER = "inner"
    """Normalize the complete projected inner width."""


class MultiHeadAttention(nn.Module, ABC, Generic[KVCacheT]):
    """Generic multi-head attention interface over an implementation-owned cache.

    The complete attention operation is the extension point so implementations
    may fuse projection, normalization, RoPE, cache mutation, attention, and
    output projection as needed. Streaming self-attention updates a
    caller-prepared rolling cache from ``x``; cross-attention reads precomputed
    static K/V without changing it.

    Shape descriptions use ``L`` for query tokens, ``S`` for cached context
    tokens, ``H`` for attention heads, and ``D`` for each head's feature
    dimension. Leading ``...`` dimensions describe batch or grouping geometry;
    query and context layouts may differ when implementations flatten them to
    the same batch size.
    """

    query_dim: int
    """Input and output token width in ``[..., L, query_dim]`` tensors."""

    context_dim: int
    """Context token width projected into ``H * D`` key and value features."""

    attention_type: AttentionType
    """Whether forward performs self-attention or static cross-attention."""

    n_heads: int
    """Number of query, key, and value heads."""

    head_dim: int
    """Per-head feature dimension ``D``."""

    inner_dim: int
    """Concatenated head width ``H * D``, equal to ``n_heads * head_dim``."""

    qk_norm_eps: float
    """Epsilon used by query and key RMS normalization when enabled."""

    qk_norm_scope: QKNormScope
    """Feature scope used by query and key normalization."""

    rope_interleaved: bool
    """Whether RoPE rotates adjacent feature pairs instead of half splits."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize shared attention geometry and implementation policies.

        Args:
            query_dim: Feature dimension of input and output tokens.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            context_dim: Feature dimension projected into cached keys and values;
                ``None`` uses ``query_dim``. Self-attention requires the two
                dimensions to match.
            attention_type: Whether :meth:`forward` updates a rolling cache or
                queries precomputed static context.
            qk_norm_eps: Positive finite epsilon used by Q/K RMS normalization.
            qk_norm_scope: Normalize each head, normalize all projected heads
                jointly, or disable Q/K normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.

        Raises:
            TypeError: An enum policy has the wrong type.
            ValueError: A dimension or normalization epsilon is invalid, or
                self-attention has different query and context dimensions.
        """
        super().__init__()

        context_dim = query_dim if context_dim is None else context_dim
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive; got {query_dim}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {n_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}")
        if not isinstance(attention_type, AttentionType):
            raise TypeError(
                f"attention_type must be an AttentionType; got {attention_type!r}"
            )
        if attention_type is AttentionType.SELF_ATTENTION and query_dim != context_dim:
            raise ValueError(
                "self-attention requires query_dim to equal context_dim; "
                f"got {query_dim} and {context_dim}"
            )
        if not isinstance(qk_norm_scope, QKNormScope):
            raise TypeError(
                f"qk_norm_scope must be a QKNormScope; got {qk_norm_scope!r}"
            )
        if not math.isfinite(qk_norm_eps) or qk_norm_eps <= 0:
            raise ValueError(
                f"qk_norm_eps must be finite and positive; got {qk_norm_eps}"
            )

        # Store the projection geometry and policies shared by every backend.
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.attention_type = attention_type
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.qk_norm_eps = qk_norm_eps
        self.qk_norm_scope = qk_norm_scope
        self.rope_interleaved = rope_interleaved

    @property
    @abstractmethod
    def query_projection(self) -> nn.Linear:
        """Return the query projection module.

        This logical accessor does not prescribe the module's registered
        attribute name. Model adapters can therefore expose checkpoint-native
        names while shared attention implementations consume one interface.
        """

    @property
    @abstractmethod
    def key_projection(self) -> nn.Linear:
        """Return the key projection module."""

    @property
    @abstractmethod
    def value_projection(self) -> nn.Linear:
        """Return the value projection module."""

    @property
    @abstractmethod
    def output_projection(self) -> nn.Linear:
        """Return the attention output projection module."""

    @property
    @abstractmethod
    def query_norm(self) -> nn.Module:
        """Return the query normalization module or identity."""

    @property
    @abstractmethod
    def key_norm(self) -> nn.Module:
        """Return the key normalization module or identity."""

    @abstractmethod
    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> KVCacheT:
        """Project context and return a precomputed K/V cache.

        Use this stage to materialize static cross-attention context. The cache
        is ready for repeated :meth:`forward` calls; implementations decide its
        physical layout, precision, and ownership.

        Args:
            context: Key/value source, shape ``[..., S, context_dim]``.
            rope_freqs: Optional key positional data for the ``S`` context
                tokens; ``None`` leaves keys position-independent.

        Returns:
            Precomputed cache containing K/V for all ``S`` context tokens.
        """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        kv_cache: KVCacheT,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Apply the configured attention type to ``x`` and ``kv_cache``.

        Implementations own the complete operation so fused backends need not
        expose independently callable cache-update or cache-query stages.

        Args:
            x: Query tokens, shape ``[..., L, query_dim]``.
            kv_cache: Streaming cache for self-attention or precomputed static
                cache for cross-attention. A streaming cache must already be in
                its current-chunk update phase.
            rope_freqs: Optional query positional data for ``L`` tokens. For
                self-attention, the same data is also applied to current keys;
                ``None`` disables positional rotation in both stages.

        Returns:
            Attention result with shape ``[..., L, query_dim]``.
        """


__all__ = ["AttentionType", "MultiHeadAttention", "QKNormScope"]
