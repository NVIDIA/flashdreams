# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract seams for KV-cache compression and attention-efficiency methods.

The interfaces here are deliberately small. They let storage codecs, retention
policies, sparse-attention selection, and compute reuse evolve independently
without changing the default :class:`BlockKVCache` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


class RuntimePhase(str, Enum):
    """Runtime phase where a KV/memory-efficiency hook is invoked."""

    DENOISE = "denoise"
    FINALIZE_CLEAN_KV = "finalize_clean_kv"
    RESET = "reset"


@dataclass(frozen=True)
class KVSpan:
    """Token span in the cache sequence dimension, using half-open indexing."""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start >= 0, "start must be non-negative"
        assert self.end >= self.start, "end must be >= start"

    @property
    def length(self) -> int:
        """Number of tokens in the span."""
        return self.end - self.start


@dataclass
class KVCompressionConfig:
    """Active config envelope for KV/memory-efficiency policies.

    Backend-specific parameters belong in ``backend_config`` so Wan, Alpadreams,
    and future recipes can share the same top-level control surface.
    ``kv_compression=None`` is the disabled state.
    """

    backend: str
    schedule: dict[str, Any] = field(default_factory=dict)
    protected_recent_chunks: int = 0
    protected_sink_tokens: int = 0
    backend_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.backend, "backend must be non-empty"
        assert self.protected_recent_chunks >= 0, (
            "protected_recent_chunks must be non-negative"
        )
        assert self.protected_sink_tokens >= 0, (
            "protected_sink_tokens must be non-negative"
        )


@dataclass
class KVStoragePayload:
    """Opaque storage payload for one compressed or uncompressed K/V span."""

    k: Any
    v: Any
    span: KVSpan
    original_dtype: torch.dtype
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KVCacheStats:
    """Common stats emitted by KV compression implementations."""

    bf16_equivalent_bytes: int = 0
    stored_bytes: int = 0
    quantize_ms: float = 0.0
    dequantize_ms: float = 0.0
    num_quantized_spans: int = 0

    @property
    def compression_ratio(self) -> float:
        """BF16-equivalent bytes divided by stored bytes."""
        if self.stored_bytes <= 0:
            return 1.0
        return self.bf16_equivalent_bytes / self.stored_bytes

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-friendly stats."""
        return {
            "bf16_equivalent_bytes": self.bf16_equivalent_bytes,
            "stored_bytes": self.stored_bytes,
            "compression_ratio": self.compression_ratio,
            "quantize_ms": self.quantize_ms,
            "dequantize_ms": self.dequantize_ms,
            "num_quantized_spans": self.num_quantized_spans,
        }


@runtime_checkable
class KVStorageBackend(Protocol):
    """Codec API for storing cached K/V spans more cheaply.

    QVG belongs here: it changes representation but returns dense K/V tensors
    before attention.
    """

    name: str

    def compress_span(
        self,
        k: Tensor,
        v: Tensor,
        *,
        span: KVSpan,
        phase: RuntimePhase,
        config: KVCompressionConfig,
    ) -> KVStoragePayload:
        """Compress one K/V span."""

    def decompress_span(
        self,
        payload: KVStoragePayload,
        *,
        phase: RuntimePhase,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return dense K/V tensors for attention."""

    def estimate_bytes(self, payload: KVStoragePayload) -> int:
        """Estimate payload storage bytes."""


@dataclass
class KVRetentionDecision:
    """Token-retention result for pruning or merging old cache entries."""

    keep_indices: Tensor
    protected_spans: tuple[KVSpan, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class KVRetentionPolicy(Protocol):
    """Policy API for selecting which cached tokens remain addressable."""

    name: str

    def select_tokens(
        self,
        *,
        total_tokens: int,
        phase: RuntimePhase,
        config: KVCompressionConfig,
        device: torch.device | str,
    ) -> KVRetentionDecision:
        """Select retained token positions."""


@runtime_checkable
class SparseAttentionPolicy(Protocol):
    """Query-aware key/value selection before an attention kernel."""

    name: str

    def select_key_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        phase: RuntimePhase,
        config: KVCompressionConfig,
    ) -> tuple[Tensor, Tensor]:
        """Return key/value tensors to attend over."""


@dataclass
class ComputeReuseDecision:
    """Decision for X-Cache-style block output reuse."""

    reuse: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ComputeReusePolicy(Protocol):
    """Policy API for reusing block residuals or outputs across chunks."""

    name: str

    def should_reuse(
        self,
        *,
        block_index: int,
        phase: RuntimePhase,
        fingerprint: Tensor,
        config: KVCompressionConfig,
    ) -> ComputeReuseDecision:
        """Decide whether a block can reuse cached compute."""


class IdentityKVStorageBackend:
    """No-op storage backend used by abstraction tests."""

    name = "identity"

    def compress_span(
        self,
        k: Tensor,
        v: Tensor,
        *,
        span: KVSpan,
        phase: RuntimePhase,
        config: KVCompressionConfig,
    ) -> KVStoragePayload:
        del phase, config
        return KVStoragePayload(k=k, v=v, span=span, original_dtype=k.dtype)

    def decompress_span(
        self,
        payload: KVStoragePayload,
        *,
        phase: RuntimePhase,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        del phase
        k = payload.k
        v = payload.v
        assert isinstance(k, Tensor), "identity payload k must be a Tensor"
        assert isinstance(v, Tensor), "identity payload v must be a Tensor"
        if device is not None:
            k = k.to(device)
            v = v.to(device)
        return k, v

    def estimate_bytes(self, payload: KVStoragePayload) -> int:
        return estimate_tensor_tree_bytes(payload.k) + estimate_tensor_tree_bytes(
            payload.v
        )


def estimate_tensor_tree_bytes(value: Any) -> int:
    """Recursively estimate bytes for tensors in common container types."""
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(estimate_tensor_tree_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tensor_tree_bytes(v) for v in value)
    return 0
