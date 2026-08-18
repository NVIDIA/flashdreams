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

"""PyTorch self- and cross-attention over block K/V caches."""

from __future__ import annotations

from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.core.attention import BlockKVCache


class TorchMultiHeadAttention(MultiHeadAttention[BlockKVCache]):
    """PyTorch reference for self- and cross-attention with a block cache.

    This implementation owns the cache lifecycle dispatched by :meth:`forward`.
    Self-attention writes each current chunk to a prepared rolling cache;
    cross-attention reuses K/V materialized once by :meth:`compute_kv`.

    Shape descriptions use ``B`` for the product of all leading batch or grouping
    dimensions, ``L`` for query tokens, ``S`` for visible cached context, ``H``
    for heads, and ``D`` for head features. Projections collapse leading token
    dimensions into the cache's single ``B`` axis; query outputs restore their
    original leading layout. Query and context layouts may therefore differ as
    long as their flattened batch sizes agree.

    Native PyTorch SDPA supplies the portable reference backend. Concrete
    subclasses own the projection and normalization modules and can override
    :meth:`_attention` while retaining the RoPE and cache contracts.
    """

    @property
    @abstractmethod
    def query_projection(self) -> nn.Linear:
        """Return the query projection module."""

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
        """Initialize shared reference-attention geometry and policies.

        Args:
            query_dim: Feature dimension of input and output tokens.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            context_dim: Feature dimension projected into keys and values;
                ``None`` uses ``query_dim``. Self-attention requires the two
                dimensions to match.
            attention_type: Whether :meth:`forward` updates a rolling cache or
                queries precomputed static context.
            qk_norm_eps: Positive finite epsilon used by Q/K RMS normalization.
            qk_norm_scope: Normalize each head, normalize all projected heads
                jointly, or disable Q/K normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.

        Raises:
            TypeError: ``attention_type`` or ``qk_norm_scope`` has the wrong type.
            ValueError: A dimension or normalization epsilon is invalid, or
                self-attention has unequal query and context widths.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qk_norm_eps=qk_norm_eps,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
        )

    def allocate_kv_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Allocate a native-precision rolling K/V cache.

        The returned cache is empty. For every chunk, call
        :meth:`BlockKVCache.before_update`, invoke self-attention, and call
        :meth:`BlockKVCache.after_update` with the same chunk index.

        Args:
            batch_size: Product ``B`` of the input's leading dimensions.
            chunk_size: Exact number of current tokens ``L`` per update.
            window_size: Number of rolling context tokens retained after the sink.
            sink_size: Number of initial context tokens that are never evicted.
            device: Device on which to allocate K/V storage.
            dtype: Data type used by K/V storage.

        Returns:
            Block cache with K/V storage shaped
            ``[B, sink_size + window_size, H, D]``.
        """
        # BlockKVCache rolls sequence dimension 1 while preserving independent
        # batch and head axes for SDPA.
        cache_shape = (
            batch_size,
            sink_size + window_size,
            self.n_heads,
            self.head_dim,
        )
        return BlockKVCache(
            k_shape=cache_shape,
            v_shape=cache_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Project static context into a finalized, reusable K/V cache.

        Args:
            context: Static key/value source, shape
                ``[..., S, context_dim]``. Leading dimensions flatten into ``B``.
            rope_freqs: Optional key rotations with shape ``[S, 1, 1, D]``;
                ``None`` stores unrotated keys.

        Returns:
            Static cache with K/V shape ``[B, S, H, D]``, ready for repeated
            :meth:`forward` calls without lifecycle bookkeeping.

        Raises:
            ValueError: Context width or RoPE geometry is incompatible with the
                configured attention geometry.
        """
        key, value = self._project_kv(context)
        if rope_freqs is not None:
            key = self._apply_rope(key, rope_freqs)
        # ``from_tensor`` completes its one write internally, so static context
        # can be queried immediately and never enters the rolling lifecycle.
        return BlockKVCache.from_tensor(key, value, seq_dim=1)

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Apply self- or cross-attention using the configured cache lifecycle.

        Args:
            x: Query tokens, shape ``[..., L, query_dim]``.
            kv_cache: Prepared rolling cache for self-attention or precomputed
                static cache for cross-attention.
            rope_freqs: Optional query rotations. Self-attention also applies
                them to the current keys before updating the cache.

        Returns:
            Output-projected tokens with the same shape as ``x``.
        """
        if self.attention_type is AttentionType.SELF_ATTENTION:
            kv_cache = self._update_kv(x, kv_cache, rope_freqs)
        return self._query_kv(x, kv_cache, rope_freqs)

    def _update_kv(
        self,
        context: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Project and write one context chunk to a prepared rolling cache.

        Args:
            context: Current key/value source chunk, shape
                ``[..., L, context_dim]`` where ``L`` equals
                ``kv_cache.chunk_size``.
            kv_cache: Rolling cache after
                :meth:`BlockKVCache.before_update` for the current chunk.
            rope_freqs: Optional key rotations with shape ``[L, 1, 1, D]``;
                ``None`` stores unrotated keys.

        Returns:
            The same cache with the current K/V chunk written and visible to
            attention. The caller still owns final bookkeeping.

        Raises:
            ValueError: Token width, RoPE geometry, cache geometry, or chunk
                length is incompatible with this module.
            RuntimeError: Cache device or dtype differs from the projected
                tensors, or its update lifecycle is inactive.
        """
        key, value = self._project_kv(context)
        if rope_freqs is not None:
            key = self._apply_rope(key, rope_freqs)
        self._validate_cache(kv_cache, key, updating=True)
        kv_cache.update(key, value)
        # Leave the lifecycle open so a following query sees the current chunk;
        # the caller closes it with ``after_update`` after attention completes.
        return kv_cache

    def _query_kv(
        self,
        query: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Query visible K/V without mutating cache storage or bookkeeping.

        Args:
            query: Query tokens, shape ``[..., L, query_dim]``. Its flattened
                leading size must equal the cache batch size ``B``.
            kv_cache: Static or rolling cache exposing K/V as ``[B, S, H, D]``.
            rope_freqs: Optional query rotations with shape ``[L, 1, 1, D]``;
                ``None`` leaves queries unrotated.

        Returns:
            Output-projected tokens with the same leading dimensions and shape
            ``[..., L, query_dim]`` as ``query``.

        Raises:
            ValueError: Query width, RoPE geometry, or cache geometry is
                incompatible with this module.
            RuntimeError: Cache device or dtype differs from projected queries.
        """
        # Projection folds every leading query dimension into the cache's single
        # batch axis; retain the public layout for the final reshape.
        batch_shape = query.shape[:-2]
        query = self._project_query(query)
        if rope_freqs is not None:
            query = self._apply_rope(query, rope_freqs)
        self._validate_cache(kv_cache, query)
        key = kv_cache.cached_k()
        value = kv_cache.cached_v()
        output = self._attention(query, key, value)
        output = self._output_projection(output)
        # ``output`` is ``[B, L, query_dim]`` after head concatenation; restore
        # the exact leading query geometry captured at the module boundary.
        return output.reshape(batch_shape + output.shape[-2:])

    def _validate_tokens(self, x: Tensor, feature_dim: int, name: str) -> None:
        """Validate a public token tensor's rank and feature width.

        Args:
            x: Query or context tokens with expected shape ``[..., L, C]``.
            feature_dim: Required trailing feature width ``C``.
            name: Argument name included in validation errors.

        Raises:
            ValueError: ``x`` has fewer than two dimensions or the wrong trailing
                feature width.
        """
        if x.ndim < 2:
            raise ValueError(
                f"{name} must have shape [..., L, D]; got {tuple(x.shape)}"
            )
        if x.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} feature width must equal {feature_dim}; got {x.shape[-1]}"
            )

    def _validate_cache(
        self,
        kv_cache: BlockKVCache,
        x: Tensor,
        *,
        updating: bool = False,
    ) -> None:
        """Validate cache geometry, placement, and optional write lifecycle.

        Args:
            kv_cache: Block cache expected to expose ``[B, S, H, D]`` K/V.
            x: Projected queries or keys with shape ``[B, L, H, D]``.
            updating: Also require an active update and ``L`` equal to the cache
                chunk size.

        Raises:
            ValueError: Cache layout, batch/head geometry, or update length is
                incompatible with ``x``.
            RuntimeError: Cache device or dtype differs from ``x``, or
                ``updating`` is true outside an active update.
        """
        # The reference cache ABI is token-major ``[B, S, H, D]`` with a
        # dynamically visible ``S``; only the other axes are fixed here.
        if kv_cache.seq_dim != 1 or kv_cache._k.ndim != 4:
            raise ValueError("K/V cache must have shape [B, S, H, D] with seq_dim=1")
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("K/V cache tensors must have identical shapes")
        expected = (x.shape[0], self.n_heads, self.head_dim)
        actual = (
            kv_cache._k.shape[0],
            kv_cache._k.shape[2],
            kv_cache._k.shape[3],
        )
        if actual != expected:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected}; got {actual}"
            )
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        if kv_cache._k.dtype != x.dtype or kv_cache._v.dtype != x.dtype:
            raise RuntimeError("K/V cache tensors must match the input dtype")
        if not updating:
            return
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if x.shape[1] != kv_cache.chunk_size:
            raise ValueError(
                "context sequence length must equal cache chunk_size="
                f"{kv_cache.chunk_size}; got {x.shape[1]}"
            )

    def _project_query(self, query: Tensor) -> Tensor:
        """Project and normalize queries as ``[B, L, H, D]``.

        Args:
            query: Query tokens with shape ``[..., L, query_dim]``.

        Returns:
            Projected queries with all leading dimensions flattened into ``B``.
        """
        self._validate_tokens(query, self.query_dim, "query")
        sequence_length = query.shape[-2]
        # Split ``H * D`` while collapsing arbitrary batch/group dimensions into
        # the one batch axis shared with BlockKVCache.
        query = self.query_projection(query).reshape(
            -1, sequence_length, self.n_heads, self.head_dim
        )
        if self.qk_norm_scope is QKNormScope.INNER:
            # INNER normalization sees concatenated heads; HEAD and NONE operate
            # directly on trailing ``D`` through RMSNorm or Identity.
            return self.query_norm(query.flatten(-2)).reshape(query.shape)
        return self.query_norm(query)

    def _project_kv(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Project context into K/V shaped ``[B, S, H, D]``.

        Args:
            context: Key/value source with shape ``[..., S, context_dim]``.

        Returns:
            Normalized keys and unnormalized values with flattened batch size
            ``B`` and independent ``H`` and ``D`` axes.
        """
        self._validate_tokens(context, self.context_dim, "context")
        sequence_length = context.shape[-2]
        # K/V share the cache ABI even when context leading dimensions differ
        # from the query layout used later.
        head_shape = (-1, sequence_length, self.n_heads, self.head_dim)
        key = self.key_projection(context).reshape(head_shape)
        value = self.value_projection(context).reshape(head_shape)
        if self.qk_norm_scope is QKNormScope.INNER:
            # Values are never Q/K-normalized; only keys follow the configured
            # head or concatenated-inner normalization policy.
            key = self.key_norm(key.flatten(-2)).reshape(key.shape)
        else:
            key = self.key_norm(key)
        return key, value

    def _apply_rope(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply the configured RoPE pairing to projected queries or keys.

        Args:
            x: Projected tensor with shape ``[B, L, H, D]``.
            rope_freqs: Rotation angles with exact shape ``[L, 1, 1, D]``.

        Returns:
            Rotated tensor with the same shape, device, and dtype as ``x``.

        Raises:
            ValueError: ``D`` is odd or ``rope_freqs`` has incompatible geometry.
        """
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim; got {x.shape[-1]}")

        # RoPE lookup shape is ``[L, 1, 1, D]`` for an input shaped
        # ``[..., L, H, D]``.
        expected_shape = (x.shape[-3], 1, 1, x.shape[-1])
        if tuple(rope_freqs.shape) != expected_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_shape}; "
                f"got {tuple(rope_freqs.shape)}"
            )

        # Broadcast positions over leading dimensions and heads:
        # ``[L, 1, 1, D] -> [..., L, 1, D]``.
        freqs = rope_freqs[:, 0, 0, :].reshape(
            (1,) * (x.ndim - 3) + (x.shape[-3], 1, x.shape[-1])
        )

        # Materialize rotation coefficients in activation precision so the
        # elementwise rotation neither promotes projected tensors nor cache data.
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)
        if self.rope_interleaved:
            # Rotate adjacent feature pairs; shape stays ``[..., L, H, D]``.
            rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
        else:
            # Rotate matching half-split features; shape stays ``[..., L, H, D]``.
            first, second = x.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)

        # Apply the elementwise complex rotation: ``[..., L, H, D]``.
        return x * cos_freqs + rotated * sin_freqs

    def _attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply non-causal scaled dot-product attention over visible K/V.

        Args:
            query: Projected queries with shape ``[B, L, H, D]``.
            key: Visible cached keys with shape ``[B, S, H, D]``.
            value: Visible cached values with shape ``[B, S, H, D]``.

        Returns:
            Per-head attention output with shape ``[B, L, H, D]``.
        """
        # Move heads before tokens for SDPA:
        # Q ``[..., L, H, D] -> [..., H, L, D]`` and
        # K/V ``[..., S, H, D] -> [..., H, S, D]``.
        query_heads = query.transpose(-3, -2)
        key_heads = key.transpose(-3, -2)
        value_heads = value.transpose(-3, -2)

        # Let PyTorch dispatch the available SDPA backend so the reference works
        # on CPU and CUDA. Cache visibility defines the allowed context, while
        # zero dropout and a non-causal mask make inference deterministic.
        output = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p=0.0,
            is_causal=False,
        )

        # Restore token-major layout: ``[..., H, L, D] -> [..., L, H, D]``.
        return output.transpose(-3, -2)

    def _output_projection(self, x: Tensor) -> Tensor:
        """Concatenate attention heads and project back to query features.

        Args:
            x: Per-head attention output with shape ``[B, L, H, D]``.

        Returns:
            Projected tokens with shape ``[B, L, query_dim]``.
        """
        # Concatenate heads: ``[..., L, H, D] -> [..., L, H * D]``.
        x = x.flatten(-2)

        # Project attention features: ``[..., L, H * D] -> [..., L, query_dim]``.
        return self.output_projection(x)


__all__ = ["TorchMultiHeadAttention"]
