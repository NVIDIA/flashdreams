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
from dataclasses import dataclass
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


class RoPEStyle(str, Enum):
    """Feature pairing convention for rotary position embeddings."""

    INTERLEAVED = "interleaved"
    """Rotate adjacent feature pairs."""

    SPLIT = "split"
    """Rotate corresponding features from the two half-width blocks."""


class RoPEScope(str, Enum):
    """Point at which rotary position embeddings are applied to keys."""

    BEFORE_KV_CACHE = "before_kv_cache"
    """Rotate keys before writing them to the K/V cache."""

    AFTER_KV_CACHE = "after_kv_cache"
    """Rotate cached keys immediately before attention."""


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """Rotary position embedding policy."""

    style: RoPEStyle = RoPEStyle.SPLIT
    """Feature pairing convention used by each rotary embedding."""

    scope: RoPEScope = RoPEScope.BEFORE_KV_CACHE
    """Whether keys are rotated before or after K/V cache storage."""

    def __post_init__(self) -> None:
        """Validate rotary policy enum values."""
        if not isinstance(self.style, RoPEStyle):
            raise TypeError(f"style must be a RoPEStyle; got {self.style!r}")
        if not isinstance(self.scope, RoPEScope):
            raise TypeError(f"scope must be a RoPEScope; got {self.scope!r}")


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Geometry, normalization, and rotary policy shared by attention backends."""

    query_dim: int
    """Input and output token width."""

    n_heads: int = 8
    """Number of query, key, and value heads."""

    head_dim: int = 64
    """Feature width of each attention head."""

    context_dim: int | None = None
    """Context token width; ``None`` uses ``query_dim``."""

    qk_norm_scope: QKNormScope = QKNormScope.HEAD
    """Feature scope used by query and key normalization."""

    qk_norm_eps: float = 1e-6
    """Positive finite epsilon used by query and key normalization."""

    rope_config: RoPEConfig | None = None
    """Rotary policy; ``None`` disables rotary embeddings."""

    @property
    def inner_dim(self) -> int:
        """Return the concatenated width of all attention heads."""
        return self.n_heads * self.head_dim

    def __post_init__(self) -> None:
        """Validate and normalize attention configuration values."""
        context_dim = self.query_dim if self.context_dim is None else self.context_dim
        if self.query_dim <= 0:
            raise ValueError(f"query_dim must be positive; got {self.query_dim}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {self.n_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {self.head_dim}")
        if not isinstance(self.qk_norm_scope, QKNormScope):
            raise TypeError(
                f"qk_norm_scope must be a QKNormScope; got {self.qk_norm_scope!r}"
            )
        if not math.isfinite(self.qk_norm_eps) or self.qk_norm_eps <= 0:
            raise ValueError(
                f"qk_norm_eps must be finite and positive; got {self.qk_norm_eps}"
            )
        if self.rope_config is not None and not isinstance(
            self.rope_config, RoPEConfig
        ):
            raise TypeError(
                f"rope_config must be a RoPEConfig or None; got {self.rope_config!r}"
            )
        object.__setattr__(self, "context_dim", context_dim)


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

    attention_type: AttentionType
    """Whether forward performs self-attention or static cross-attention."""

    attention_config: AttentionConfig
    """Geometry, normalization, and rotary policy used by this module."""

    def __init__(
        self,
        attention_type: AttentionType,
        attention_config: AttentionConfig,
    ) -> None:
        """Initialize shared attention geometry and implementation policies.

        Args:
            attention_type: Select self-attention or static cross-attention.
            attention_config: Shared geometry, normalization, and rotary policy.

        Raises:
            TypeError: An argument has the wrong configuration type.
            ValueError: Self-attention has different query and context dimensions.
        """
        super().__init__()

        if not isinstance(attention_type, AttentionType):
            raise TypeError(
                f"attention_type must be an AttentionType; got {attention_type!r}"
            )
        if not isinstance(attention_config, AttentionConfig):
            raise TypeError(
                f"attention_config must be an AttentionConfig; got {attention_config!r}"
            )
        context_dim = attention_config.context_dim
        assert context_dim is not None
        if (
            attention_type is AttentionType.SELF_ATTENTION
            and attention_config.query_dim != context_dim
        ):
            raise ValueError(
                "self-attention requires query_dim to equal context_dim; "
                f"got {attention_config.query_dim} and {context_dim}"
            )

        self.attention_type = attention_type
        self.attention_config = attention_config

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
                tokens. Applied only by before-cache RoPE; after-cache RoPE
                stores unrotated keys and consumes cache-relative data in
                :meth:`forward`. Ignored when ``rope_config`` is ``None``.

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
            rope_freqs: Optional positional data. Before-cache RoPE expects the
                current ``L`` query/key positions. After-cache RoPE expects
                cache-relative positions for all visible keys and selects the
                current query positions from the cache write interval. Ignored
                when ``rope_config`` is ``None``.

        Returns:
            Attention result with shape ``[..., L, query_dim]``.
        """


__all__ = [
    "AttentionConfig",
    "AttentionType",
    "MultiHeadAttention",
    "QKNormScope",
    "RoPEConfig",
    "RoPEScope",
    "RoPEStyle",
]
