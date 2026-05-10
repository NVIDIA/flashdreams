# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KV-compression abstraction seams."""

import torch

from flashdreams.core.attention.kv_compress import (
    IdentityKVStorageBackend,
    KVCacheStats,
    KVCompressionConfig,
    KVSpan,
    KVStorageBackend,
    RuntimePhase,
    estimate_tensor_tree_bytes,
)


def test_kv_compression_config_is_active_config() -> None:
    config = KVCompressionConfig(backend="qvg")

    assert config.backend == "qvg"


def test_identity_backend_round_trips_tensors() -> None:
    backend = IdentityKVStorageBackend()
    assert isinstance(backend, KVStorageBackend)

    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    payload = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 3),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=KVCompressionConfig(backend="identity"),
    )
    out_k, out_v = backend.decompress_span(
        payload, phase=RuntimePhase.DENOISE, device=torch.device("cpu")
    )

    torch.testing.assert_close(out_k, k)
    torch.testing.assert_close(out_v, v)
    assert backend.estimate_bytes(payload) == k.numel() * k.element_size() * 2


def test_stats_and_tensor_tree_byte_estimate() -> None:
    x = torch.zeros(2, 3, dtype=torch.float32)
    payload = {"x": x, "nested": [x.to(torch.float16), "ignored"]}

    assert estimate_tensor_tree_bytes(payload) == 36

    stats = KVCacheStats(bf16_equivalent_bytes=70, stored_bytes=10)
    assert stats.compression_ratio == 7.0
    assert stats.as_dict()["compression_ratio"] == 7.0
