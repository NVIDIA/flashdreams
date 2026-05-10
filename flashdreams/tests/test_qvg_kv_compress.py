# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for QVG-style KV compression."""

import torch

from flashdreams.core.attention.kv_compress import (
    KVCompressionConfig,
    KVSpan,
    QVGBackend,
    QVGQuantConfig,
    QuantizedKVCache,
    RuntimePhase,
)
from flashdreams.core.attention.kv_compress.qvg import _quantize_residual


def _qvg_config(quant_type: str = "triton-nstages-kmeans-int2") -> KVCompressionConfig:
    return KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 1},
        protected_recent_chunks=1,
        backend_config={
            "quant_type": quant_type,
            "cache_num_k_centroids": 2,
            "cache_num_v_centroids": 2,
            "kmeans_max_iters": 2,
            "quant_block_size": 4,
            "num_prq_stages": 1,
            "scale_dtype": "float16",
            "kernel_impl": "native",
        },
    )


def _append(cache: QuantizedKVCache, chunk_idx: int, values: torch.Tensor) -> None:
    cache.before_update(chunk_idx)
    cache.update(values, values + 100)
    cache.after_update(chunk_idx)
    cache.finalize_clean_chunk(chunk_idx)


def _append_with_rope(
    cache: QuantizedKVCache,
    chunk_idx: int,
    values: torch.Tensor,
    rope_freqs: torch.Tensor,
) -> None:
    cache.before_update(chunk_idx)
    cache.set_pending_rope_freqs(rope_freqs)
    cache.update(values, values + 100)
    cache.after_update(chunk_idx)
    cache.finalize_clean_chunk(chunk_idx)


def test_qvg_backend_roundtrip_shape_and_smaller_payload() -> None:
    config = _qvg_config()
    backend = QVGBackend(QVGQuantConfig.from_config(config))
    k = torch.randn(1, 2, 32, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 32, 8, dtype=torch.float16)

    payload = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )
    out_k, out_v = backend.decompress_span(payload, phase=RuntimePhase.DENOISE)

    assert out_k.shape == k.shape
    assert out_v.shape == v.shape
    assert backend.estimate_bytes(payload) < k.numel() * k.element_size() * 2


def test_qvg_residual_quant_rounds_half_away_from_zero() -> None:
    residual = torch.tensor([[[[-0.5, 0.5, 1.0, -1.0]]]], dtype=torch.float32)

    packed, _scales = _quantize_residual(
        residual,
        block_size=4,
        num_bits=2,
        scale_dtype=torch.float32,
    )

    assert packed.item() == 40


def test_qvg_backend_random_kmeans_init_is_seeded() -> None:
    config = _qvg_config()
    config.backend_config.update({"kmeans_init": "random", "kmeans_seed": 123})
    first_backend = QVGBackend(QVGQuantConfig.from_config(config))
    second_backend = QVGBackend(QVGQuantConfig.from_config(config))
    k = torch.randn(1, 2, 32, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 32, 8, dtype=torch.float16)

    first = first_backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )
    second = second_backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )

    torch.testing.assert_close(
        first.k["cluster_ids_list"][0],
        second.k["cluster_ids_list"][0],
    )


def test_qvg_backend_mixed_kv_bit_widths() -> None:
    config = _qvg_config()
    config.backend_config.update({"cache_k_num_bits": 2, "cache_v_num_bits": 4})
    backend = QVGBackend(QVGQuantConfig.from_config(config))
    k = torch.randn(1, 2, 32, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 32, 8, dtype=torch.float16)

    payload = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )

    assert payload.k["num_bits"] == 2
    assert payload.v["num_bits"] == 4
    assert payload.metadata["k_num_bits"] == 2
    assert payload.metadata["v_num_bits"] == 4


def test_qvg_backend_seeded_random_varies_by_compress_call() -> None:
    config = _qvg_config()
    config.backend_config.update(
        {
            "cache_num_k_centroids": 8,
            "cache_num_v_centroids": 8,
            "kmeans_init": "random",
            "kmeans_seed": 123,
        }
    )
    backend = QVGBackend(QVGQuantConfig.from_config(config))
    k = torch.randn(1, 2, 32, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 32, 8, dtype=torch.float16)

    first = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )
    second = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 32),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )

    assert not torch.equal(
        first.k["cluster_ids_list"][0],
        second.k["cluster_ids_list"][0],
    )


def test_quantized_kv_cache_mixed_bf16_and_quantized_spans() -> None:
    config = _qvg_config()
    cache = QuantizedKVCache(
        k_shape=(1, 8, 2, 8),
        v_shape=(1, 8, 2, 8),
        seq_dim=1,
        chunk_size=2,
        window_size=8,
        sink_size=0,
        device="cpu",
        dtype=torch.float16,
        backend=QVGBackend(QVGQuantConfig.from_config(config)),
        compression_config=config,
    )

    for chunk_idx in range(4):
        values = torch.full((1, 2, 2, 8), float(chunk_idx), dtype=torch.float16)
        _append(cache, chunk_idx, values)

    assert cache.cached_k().shape == (1, 8, 2, 8)
    assert cache.cached_v().shape == (1, 8, 2, 8)
    assert cache._dense_read_cache is None
    assert cache.stats.num_quantized_spans > 0
    assert cache.stats.bf16_equivalent_bytes > 0


def test_quantized_kv_cache_official_qvg_cadence() -> None:
    config = KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 8},
        protected_recent_chunks=0,
        backend_config={
            "quant_type": "triton-nstages-kmeans-int2",
            "cache_num_k_centroids": 4,
            "cache_num_v_centroids": 4,
            "kmeans_max_iters": 2,
            "quant_block_size": 4,
            "num_prq_stages": 1,
            "scale_dtype": "float16",
            "kernel_impl": "native",
        },
    )
    cache = QuantizedKVCache(
        k_shape=(1, 32, 2, 8),
        v_shape=(1, 32, 2, 8),
        seq_dim=1,
        chunk_size=4,
        window_size=32,
        sink_size=0,
        device="cpu",
        dtype=torch.float16,
        backend=QVGBackend(QVGQuantConfig.from_config(config)),
        compression_config=config,
    )

    for chunk_idx in range(8):
        values = torch.full((1, 4, 2, 8), float(chunk_idx), dtype=torch.float16)
        _append(cache, chunk_idx, values)

    assert cache.stats.num_quantized_spans == 1
    assert cache.stats.compression_ratio > 1.0


def test_quantized_kv_cache_roll_preserves_quantized_span() -> None:
    config = KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 2},
        protected_recent_chunks=0,
        backend_config={
            "quant_type": "triton-nstages-kmeans-int2",
            "cache_num_k_centroids": 2,
            "cache_num_v_centroids": 2,
            "kmeans_max_iters": 2,
            "quant_block_size": 4,
            "num_prq_stages": 1,
            "scale_dtype": "float16",
            "kernel_impl": "native",
        },
    )
    cache = QuantizedKVCache(
        k_shape=(1, 8, 2, 8),
        v_shape=(1, 8, 2, 8),
        seq_dim=1,
        chunk_size=4,
        window_size=8,
        sink_size=0,
        device="cpu",
        dtype=torch.float16,
        backend=QVGBackend(QVGQuantConfig.from_config(config)),
        compression_config=config,
    )

    for chunk_idx in range(2):
        values = torch.full((1, 4, 2, 8), float(chunk_idx), dtype=torch.float16)
        _append(cache, chunk_idx, values)
    assert cache._entries[0] is not None
    assert cache._entries[0].kind == "quantized"

    values = torch.full((1, 4, 2, 8), 2.0, dtype=torch.float16)
    _append(cache, 2, values)

    assert cache._entries[0] is not None
    assert cache._entries[0].kind == "quantized"
    assert cache._entries[0].payload_start_chunk == 1
    assert cache._entries[1] is not None
    assert cache._entries[1].kind == "bf16"
    assert cache.cached_k().shape == (1, 8, 2, 8)


def test_quantized_kv_cache_tracks_prerope_freqs_across_compressed_span() -> None:
    config = KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 2},
        protected_recent_chunks=0,
        backend_config={
            "quant_type": "triton-nstages-kmeans-int2",
            "cache_num_k_centroids": 2,
            "cache_num_v_centroids": 2,
            "kmeans_max_iters": 2,
            "quant_block_size": 4,
            "num_prq_stages": 1,
            "scale_dtype": "float16",
            "kernel_impl": "native",
            "store_prerope_keys": True,
        },
    )
    cache = QuantizedKVCache(
        k_shape=(1, 4, 2, 8),
        v_shape=(1, 4, 2, 8),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device="cpu",
        dtype=torch.float16,
        backend=QVGBackend(QVGQuantConfig.from_config(config)),
        compression_config=config,
    )

    rope0 = torch.full((2, 1, 1, 8), 0.0)
    rope1 = torch.full((2, 1, 1, 8), 1.0)
    _append_with_rope(
        cache, 0, torch.full((1, 2, 2, 8), 0.0, dtype=torch.float16), rope0
    )
    _append_with_rope(
        cache, 1, torch.full((1, 2, 2, 8), 1.0, dtype=torch.float16), rope1
    )

    assert cache._entries[0] is not None
    assert cache._entries[0].kind == "quantized"
    torch.testing.assert_close(cache.cached_k_rope_freqs(), torch.cat([rope0, rope1]))


def test_quantized_kv_cache_prerope_freqs_survive_roll() -> None:
    config = KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 2},
        protected_recent_chunks=0,
        backend_config={
            "quant_type": "triton-nstages-kmeans-int2",
            "cache_num_k_centroids": 2,
            "cache_num_v_centroids": 2,
            "kmeans_max_iters": 2,
            "quant_block_size": 4,
            "num_prq_stages": 1,
            "scale_dtype": "float16",
            "kernel_impl": "native",
            "store_prerope_keys": True,
        },
    )
    cache = QuantizedKVCache(
        k_shape=(1, 4, 2, 8),
        v_shape=(1, 4, 2, 8),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device="cpu",
        dtype=torch.float16,
        backend=QVGBackend(QVGQuantConfig.from_config(config)),
        compression_config=config,
    )

    ropes = [torch.full((2, 1, 1, 8), float(i)) for i in range(3)]
    for chunk_idx, rope_freqs in enumerate(ropes):
        _append_with_rope(
            cache,
            chunk_idx,
            torch.full((1, 2, 2, 8), float(chunk_idx), dtype=torch.float16),
            rope_freqs,
        )

    assert cache._entries[0] is not None
    assert cache._entries[0].kind == "quantized"
    assert cache._entries[0].payload_start_chunk == 1
    torch.testing.assert_close(
        cache.cached_k_rope_freqs(), torch.cat([ropes[1], ropes[2]])
    )
