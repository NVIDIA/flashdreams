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

"""Benchmark the complete Wan 2.1 1.3B DiT network.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/recipes/test_wan_network.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import math
from enum import Enum

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention_triton import SDPABackend
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.infra.acceleration import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.compile import compile_module
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork1pt3BConfig,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Wan DiT network benchmarks require CUDA.",
    ),
]

# Causal-Forcing's production chunkwise Wan 2.1 1.3B geometry.
_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_LATENT_HEIGHT = 60
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 3
_ATTENTION_HEIGHT = 30
_ATTENTION_WIDTH = 52
_WINDOW_SIZE_T = 21
_SINK_SIZE_T = 0
_TEXT_TOKENS = 512
_CHUNK_TOKENS = _CHUNK_SIZE_T * _ATTENTION_HEIGHT * _ATTENTION_WIDTH
_WINDOW_CHUNKS = _WINDOW_SIZE_T // _CHUNK_SIZE_T
_WINDOW_TOKENS = _WINDOW_CHUNKS * _CHUNK_TOKENS
_DIFFUSION_TIMESTEP = 1000.0
_CUDA_GRAPH_WARMUP_ITERS = 2
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


@pytest.mark.parametrize(
    "implementation",
    tuple(_Implementation),
    ids=lambda implementation: implementation.value.replace("_", "-"),
)
@torch.inference_mode()
def test_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    implementation: _Implementation,
) -> None:
    """Benchmark a compiled Wan 2.1 1.3B DiT using CUDA-graph replay."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan DiT network benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    backend = implementation.attention_backend
    _skip_unsupported_device(backend, device)
    dtype = torch.bfloat16
    torch.manual_seed(_SEED)
    config = WanDiTNetwork1pt3BConfig(
        patch_embedding_type="conv3d",
        cp_method="ring",
        attention_backend=backend,
        sdpa_backend=implementation.sdpa_backend,
    )

    # Allocate the real 1.3B network directly in BF16 on its final device.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            network = WanDiTNetwork(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    network.set_context_parallel_group(None)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    assert all(
        block.attention_backend is backend
        and block.sdpa_backend is implementation.sdpa_backend
        for block in network.blocks
    )

    generator = torch.Generator(device=device).manual_seed(_SEED)
    patch_volume = math.prod(config.patch_size)
    x = torch.randn(
        (_CHUNK_TOKENS, config.in_dim * patch_volume),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor(_DIFFUSION_TIMESTEP, device=device, dtype=dtype)
    text_embeddings = torch.randn(
        (_TEXT_TOKENS, config.text_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = network.initialize_cache(
        chunk_size=_CHUNK_TOKENS,
        window_size=_WINDOW_TOKENS,
        sink_size=0,
        text_embeddings=text_embeddings,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_t=_CHUNK_SIZE_T,
        len_h=_ATTENTION_HEIGHT,
        len_w=_ATTENTION_WIDTH,
        interleaved=True,
        device=device,
    )

    network = compile_module(network)
    capture_chunk_idx = cuda_graph_capture_ar_index(
        sink_size_t=_SINK_SIZE_T,
        window_size_t=_WINDOW_SIZE_T,
        len_t=_CHUNK_SIZE_T,
    )
    graph_dispatch = CUDAGraphDispatch(
        network,
        enabled=True,
        capture_ar_idx=capture_chunk_idx,
        warmup_iters=_CUDA_GRAPH_WARMUP_ITERS,
    )

    def forward(chunk_idx: int, chunk_rope_freqs: torch.Tensor) -> torch.Tensor:
        return graph_dispatch.select(chunk_idx, uncond=False)(
            x=x,
            timesteps=timestep,
            cache=cache,
            rope_freqs=chunk_rope_freqs,
            current_chunk_idx=chunk_idx,
            eager_mode=False,
        )

    benchmark_chunk_idx = capture_chunk_idx + 1
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]

    # Drain compile/autotune while filling the cache. At the first full-window
    # index, finish wrapper warmup and capture before benchmarking graph replay.
    for chunk_idx in range(capture_chunk_idx):
        cache.before_update(chunk_idx)
        output = forward(chunk_idx, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    cache.before_update(capture_chunk_idx)
    for _ in range(_CUDA_GRAPH_WARMUP_ITERS + 1):
        output = forward(capture_chunk_idx, rope_freqs[capture_chunk_idx])
    cache.after_update(capture_chunk_idx)
    torch.cuda.synchronize(device)

    benchmark.group = "wan-dit-network"
    benchmark.extra_info.update(
        {
            "network": "WanDiTNetwork1pt3B",
            "model_family": "wan",
            "model_variant": "wan2.1-1.3b",
            "implementation": implementation.value,
            "execution_backend": "pytorch",
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
            "sink_tokens": 0,
            "text_tokens": _TEXT_TOKENS,
            "input_patch_channels": config.in_dim * patch_volume,
            "output_patch_channels": config.out_dim * patch_volume,
            "model_channels": config.dim,
            "ffn_channels": config.ffn_dim,
            "num_blocks": config.num_layers,
            "num_heads": config.num_heads,
            "head_dim": config.dim // config.num_heads,
            "parameter_count": parameter_count,
            "checkpoint": "random_init_seed_matched",
            "dtype": str(dtype),
            "self_attention_cache_dtype": str(cache[0].self_attn.dtype),
            "cross_attention_cache_dtype": str(cache[0].cross_attn.text.dtype),
            "compiled": True,
            "compile_mode": "max-autotune-no-cudagraphs",
            "cuda_graph": True,
            "cuda_graph_warmup_iters": _CUDA_GRAPH_WARMUP_ITERS,
            "cuda_graph_capture_chunk_idx": capture_chunk_idx,
            "cache_state": "full_window_static_text",
            "cache_prefill_chunks": capture_chunk_idx + 1,
            "benchmark_chunk_idx": benchmark_chunk_idx,
            "cache_update_bookkeeping": "excluded_from_timing",
            "diffusion_timestep": _DIFFUSION_TIMESTEP,
            "classifier_free_guidance": False,
            "rope_interleaved": True,
            "context_parallel_size": 1,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "compiler_cache_state": (
                "host-dependent; compile, autotune, and CUDA graph capture "
                "excluded from measured rounds"
            ),
            "seed": _SEED,
        }
    )

    # Roll once outside timing; all fixture warmups and measured calls replay
    # the captured graph against the same steady-state cache slot.
    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = forward(benchmark_chunk_idx, rope_freqs[benchmark_chunk_idx])
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

    expected_output_shape = (_CHUNK_TOKENS, config.out_dim * patch_volume)
    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()
