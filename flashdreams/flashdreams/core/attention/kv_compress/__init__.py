# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared interfaces for KV-cache compression and related attention efficiency work."""

from flashdreams.core.attention.kv_compress.base import (
    ComputeReuseDecision,
    ComputeReusePolicy,
    IdentityKVStorageBackend,
    KVCacheStats,
    KVCompressionConfig,
    KVRetentionDecision,
    KVRetentionPolicy,
    KVSpan,
    KVStorageBackend,
    KVStoragePayload,
    RuntimePhase,
    SparseAttentionPolicy,
    estimate_tensor_tree_bytes,
)
from flashdreams.core.attention.kv_compress.qvg import QVGBackend, QVGQuantConfig
from flashdreams.core.attention.kv_compress.quantized_cache import QuantizedKVCache

__all__ = [
    "ComputeReuseDecision",
    "ComputeReusePolicy",
    "IdentityKVStorageBackend",
    "KVCacheStats",
    "KVCompressionConfig",
    "KVRetentionDecision",
    "KVRetentionPolicy",
    "KVSpan",
    "KVStorageBackend",
    "KVStoragePayload",
    "RuntimePhase",
    "SparseAttentionPolicy",
    "QVGBackend",
    "QVGQuantConfig",
    "QuantizedKVCache",
    "estimate_tensor_tree_bytes",
]
