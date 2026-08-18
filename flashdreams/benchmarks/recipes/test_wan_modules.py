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

"""Microbenchmarks for Wan self-attention and DiT blocks.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/recipes/test_wan_modules.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from enum import Enum

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention_triton import SDPABackend
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.recipes.wan.transformer.impl.modules import (
    AttentionBackend,
    Block,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork1pt3BConfig,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Wan DiT module benchmarks require CUDA.",
    ),
]

# Causal-Forcing's chunkwise 480x832 Wan 2.1 1.3B geometry. The VAE produces
# 3x60x104 latents per AR chunk; 1x2x2 DiT patches produce 3x30x52 tokens.
_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_LATENT_HEIGHT = 60
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 3
_ATTENTION_HEIGHT = 30
_ATTENTION_WIDTH = 52
_WINDOW_CHUNKS = 7
_TEXT_TOKENS = 512
_CHUNK_TOKENS = _CHUNK_SIZE_T * _ATTENTION_HEIGHT * _ATTENTION_WIDTH
_WINDOW_TOKENS = _WINDOW_CHUNKS * _CHUNK_TOKENS
_SINK_TOKENS = 0
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 42


class _Implementation(str, Enum):
    """Wan attention implementations covered by the benchmark."""

    WAN_TORCH = "wan_torch"
    TRITON_CUDNN = "triton_cudnn"
    TRITON_FA2 = "triton_fa2"

    @property
    def attention_backend(self) -> AttentionBackend:
        """Return the DiT attention implementation."""
        if self is self.WAN_TORCH:
            return AttentionBackend.WAN
        return AttentionBackend.TRITON

    @property
    def sdpa_backend(self) -> SDPABackend:
        """Return the configured self-attention SDPA implementation."""
        if self is self.TRITON_FA2:
            return SDPABackend.TRITON
        return SDPABackend.CUDNN

    @property
    def self_attention_operator(self) -> str:
        """Return the concrete self-attention operator name."""
        if self is self.WAN_TORCH:
            return "cudnn"
        if self is self.TRITON_CUDNN:
            return "torch_cudnn_sdpa"
        return "triton_fa2"

    @property
    def cross_attention_operator(self) -> str:
        """Return the concrete cross-attention operator name."""
        if self is self.WAN_TORCH:
            return "cudnn"
        return "triton_fa2"


assert {case.attention_backend for case in _Implementation} == set(AttentionBackend)
assert {
    case.sdpa_backend
    for case in _Implementation
    if case.attention_backend is AttentionBackend.TRITON
} == set(SDPABackend)


def _skip_unsupported_device(
    backend: AttentionBackend,
    device: torch.device,
) -> None:
    """Skip Triton attention on devices without tensor-memory acceleration."""
    if backend is AttentionBackend.TRITON and torch.cuda.get_device_capability(
        device
    ) < (9, 0):
        pytest.skip("Triton attention requires compute capability 9.0 or newer.")


def _make_block(
    config: WanDiTNetwork1pt3BConfig,
    backend: AttentionBackend,
) -> Block:
    """Build a backend-selected block with weight-matched random parameters."""

    def make(selected_backend: AttentionBackend) -> Block:
        return Block(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            i2v=config.cross_attn_enable_img,
            apply_rope_before_kvcache=config.apply_rope_before_kvcache,
            cp_method=config.cp_method,
            attention_backend=selected_backend,
            sdpa_backend=config.sdpa_backend,
        )

    torch.manual_seed(_SEED)
    reference = make(AttentionBackend.WAN)
    if backend is AttentionBackend.WAN:
        return reference

    block = make(backend)
    block.load_state_dict(reference.state_dict(), strict=True)
    return block


@pytest.mark.parametrize(
    "implementation",
    tuple(_Implementation),
    ids=lambda implementation: implementation.value.replace("_", "-"),
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    implementation: _Implementation,
) -> None:
    """Benchmark Wan self-attention against a full production KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan self-attention benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    backend = implementation.attention_backend
    _skip_unsupported_device(backend, device)
    dtype = torch.bfloat16
    config = WanDiTNetwork1pt3BConfig(sdpa_backend=implementation.sdpa_backend)
    block = _make_block(config, backend)
    assert block.attention_backend is backend
    assert block.sdpa_backend is implementation.sdpa_backend
    attention = block.self_attn
    attention.to(device=device, dtype=dtype).eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    x = torch.randn(
        (_CHUNK_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=_CHUNK_TOKENS,
        window_size=_WINDOW_TOKENS,
        sink_size=_SINK_TOKENS,
        device=device,
        dtype=dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_t=_CHUNK_SIZE_T,
        len_h=_ATTENTION_HEIGHT,
        len_w=_ATTENTION_WIDTH,
        interleaved=True,
        device=device,
    )

    benchmark_chunk_idx = _WINDOW_CHUNKS
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(_WINDOW_CHUNKS):
        cache.before_update(chunk_idx)
        output = attention(x, cache, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize(device)

    benchmark.group = "wan-dit-self-attention"
    benchmark.extra_info.update(
        {
            "module": "self_attention",
            "model_family": "wan",
            "model_variant": "wan2.1-1.3b",
            "implementation": implementation.value,
            "attention_backend": backend.value,
            "sdpa_backend": implementation.sdpa_backend.value,
            "self_attention_operator": implementation.self_attention_operator,
            "projection_backend": (
                "separate_qkv"
                if backend is AttentionBackend.WAN
                else "row_scaled_fp8_fused_qkv_output"
            ),
            "batch_shape": [],
            "flattened_batch_size": 1,
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "attention_grid": [
                _CHUNK_SIZE_T,
                _ATTENTION_HEIGHT,
                _ATTENTION_WIDTH,
            ],
            "chunk_tokens": _CHUNK_TOKENS,
            "window_chunks": _WINDOW_CHUNKS,
            "window_tokens": _WINDOW_TOKENS,
            "sink_tokens": _SINK_TOKENS,
            "model_channels": config.dim,
            "num_heads": config.num_heads,
            "head_dim": config.dim // config.num_heads,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "cache_dtype": str(cache.dtype),
            "cache_state": "full_window",
            "cache_prefill_chunks": _WINDOW_CHUNKS,
            "benchmark_chunk_idx": benchmark_chunk_idx,
            "cache_update_bookkeeping": "excluded_from_timing",
            "rope_interleaved": True,
            "context_parallel_size": 1,
            "compiled": False,
            "cuda_graph": False,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

    # Roll outside timing, then repeatedly overwrite the same slot to mirror
    # multiple denoising evaluations at one autoregressive position.
    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = attention(x, cache, rope_freqs[benchmark_chunk_idx])
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)
    benchmark.extra_info["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
        device
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "implementation",
    tuple(_Implementation),
    ids=lambda implementation: implementation.value.replace("_", "-"),
)
@torch.inference_mode()
def test_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    implementation: _Implementation,
) -> None:
    """Benchmark a production-configured Wan DiT block at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan DiT block benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    backend = implementation.attention_backend
    _skip_unsupported_device(backend, device)
    dtype = torch.bfloat16
    config = WanDiTNetwork1pt3BConfig(sdpa_backend=implementation.sdpa_backend)
    block = _make_block(config, backend).to(device=device, dtype=dtype).eval()
    assert block.attention_backend is backend
    assert block.sdpa_backend is implementation.sdpa_backend
    block.update_parameters_after_loading_checkpoint()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    x = torch.randn(
        (_CHUNK_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    modulation = torch.randn(
        (6, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (_TEXT_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=_CHUNK_TOKENS,
        window_size=_WINDOW_TOKENS,
        sink_size=_SINK_TOKENS,
        context_text=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_t=_CHUNK_SIZE_T,
        len_h=_ATTENTION_HEIGHT,
        len_w=_ATTENTION_WIDTH,
        interleaved=True,
        device=device,
    )

    def forward(chunk_idx: int, chunk_rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=chunk_rope_freqs,
        )
        cache.after_update(chunk_idx)
        return result

    benchmark_chunk_idx = _WINDOW_CHUNKS
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(_WINDOW_CHUNKS):
        output = forward(chunk_idx, rope_freqs[chunk_idx])
    torch.cuda.synchronize(device)

    benchmark.group = "wan-dit-block"
    benchmark.extra_info.update(
        {
            "module": "Block",
            "model_family": "wan",
            "model_variant": "wan2.1-1.3b",
            "implementation": implementation.value,
            "attention_backend": backend.value,
            "sdpa_backend": implementation.sdpa_backend.value,
            "self_attention_operator": implementation.self_attention_operator,
            "cross_attention_operator": implementation.cross_attention_operator,
            "projection_backend": (
                "separate_qkv"
                if backend is AttentionBackend.WAN
                else "row_scaled_fp8_fused_qkv_output"
            ),
            "batch_shape": [],
            "flattened_batch_size": 1,
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "attention_grid": [
                _CHUNK_SIZE_T,
                _ATTENTION_HEIGHT,
                _ATTENTION_WIDTH,
            ],
            "chunk_tokens": _CHUNK_TOKENS,
            "window_chunks": _WINDOW_CHUNKS,
            "window_tokens": _WINDOW_TOKENS,
            "sink_tokens": _SINK_TOKENS,
            "text_tokens": _TEXT_TOKENS,
            "model_channels": config.dim,
            "ffn_channels": config.ffn_dim,
            "num_heads": config.num_heads,
            "head_dim": config.dim // config.num_heads,
            "parameter_count": sum(
                parameter.numel() for parameter in block.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "self_attention_cache_dtype": str(cache.self_attn.dtype),
            "cross_attention_cache_dtype": str(cache.cross_attn.text.dtype),
            "cache_state": "full_window_static_text",
            "cache_prefill_chunks": _WINDOW_CHUNKS,
            "benchmark_chunk_idx": benchmark_chunk_idx,
            "cache_update_bookkeeping": "excluded_from_timing",
            "rope_interleaved": True,
            "context_parallel_size": 1,
            "compiled": False,
            "cuda_graph": False,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs[benchmark_chunk_idx],
        )
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)
    benchmark.extra_info["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
        device
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
