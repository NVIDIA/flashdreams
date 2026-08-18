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

"""Benchmark the complete Wan 2.1 T2V 1.3B DiT network by backend."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.distributed import init as init_distributed
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
from integrations.wan21.benchmarks.cases import (
    ATTENTION_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Wan 2.1 DiT network benchmark requires CUDA"

# Shipped wan21-t2v-1.3b-480p geometry. The 480x832 output becomes a
# 60x104 latent; 1x2x2 patching yields 30x52 tokens for each of 21 frames.
_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_LATENT_HEIGHT = 60
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 21
_WINDOW_SIZE_T = 21
_SINK_SIZE_T = 0
_ATTENTION_HEIGHT = 30
_ATTENTION_WIDTH = 52
_TEXT_TOKENS = 512
_GLOBAL_CHUNK_TOKENS = _CHUNK_SIZE_T * _ATTENTION_HEIGHT * _ATTENTION_WIDTH
_GLOBAL_WINDOW_TOKENS = _WINDOW_SIZE_T * _ATTENTION_HEIGHT * _ATTENTION_WIDTH
_DIFFUSION_TIMESTEP = 1000.0
_PIPELINE_GUIDANCE_SCALE = 6.0
_AUTOTUNE_DRAIN_ROUNDS = 3
_CUDA_GRAPH_WARMUP_ITERS = 2
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 42


def _benchmark_device() -> torch.device:
    """Initialize context parallelism and return this rank's GPU."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        init_distributed()
    if dist.is_initialized():
        return torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _synchronize_ranks() -> None:
    """Align context-parallel ranks before a benchmark sample."""
    if dist.is_initialized():
        dist.barrier()


def _skip_unsupported_case(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a case where its hardware or execution mode is unsupported."""
    skip_unsupported_device(case, device)
    if case.attention_backend is not AttentionBackend.TRITON:
        return
    if dist.is_initialized() and dist.get_world_size() > 1:
        pytest.skip("Triton attention does not support context parallelism")


@pytest.mark.parametrize(
    "case",
    ATTENTION_CASES,
    ids=lambda case: case.pytest_id,
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.inference_mode()
def test_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark one compiled production-size Wan 2.1 DiT evaluation."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan 2.1 DiT network benchmark requires bfloat16 support")

    dtype = torch.bfloat16
    torch.manual_seed(_SEED)
    _skip_unsupported_case(case, device)
    backend = case.attention_backend
    config = WanDiTNetwork1pt3BConfig(
        patch_embedding_type="conv3d",
        cp_method="ring",
        attention_backend=backend,
        sdpa_backend=case.sdpa_backend,
    )

    # Avoid materializing the 1.3B random initialization as fp32 CPU weights.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            network = WanDiTNetwork(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())

    cp_size = dist.get_world_size() if dist.is_initialized() else 1
    cp_group = dist.group.WORLD if cp_size > 1 else None
    network.set_context_parallel_group(cp_group)
    assert all(
        block.attention_backend is backend and block.sdpa_backend is case.sdpa_backend
        for block in network.blocks
    )
    attention_modules = [
        module
        for module in network.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    assert {attention.backend for attention in attention_modules} == (
        {"cudnn"} if backend is AttentionBackend.WAN else set()
    )
    cp_enabled_attention_modules = [
        attention
        for attention in attention_modules
        if attention.is_context_parallel_enabled()
    ]
    local_attention_methods = {
        attention.method
        for attention in attention_modules
        if not attention.is_context_parallel_enabled()
    }
    assert all(
        attention.context_parallel_size() == cp_size
        for attention in cp_enabled_attention_modules
    )
    assert all(
        attention.method == config.cp_method
        for attention in cp_enabled_attention_modules
    )
    assert bool(cp_enabled_attention_modules) == (cp_size > 1)

    assert _GLOBAL_CHUNK_TOKENS % cp_size == 0
    assert _GLOBAL_WINDOW_TOKENS % cp_size == 0
    chunk_tokens = _GLOBAL_CHUNK_TOKENS // cp_size
    window_tokens = _GLOBAL_WINDOW_TOKENS // cp_size
    patch_volume = config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    generator = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(
        (chunk_tokens, config.in_dim * patch_volume),
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
        chunk_size=chunk_tokens,
        window_size=window_tokens,
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
    rope.set_context_parallel_group(cp_group)
    rope_freqs = rope.shift_t(0)

    network = compile_module(network)
    capture_ar_index = cuda_graph_capture_ar_index(
        sink_size_t=_SINK_SIZE_T,
        window_size_t=_WINDOW_SIZE_T,
        len_t=_CHUNK_SIZE_T,
    )
    graph_dispatch = CUDAGraphDispatch(
        network,
        enabled=True,
        capture_ar_idx=capture_ar_index,
        warmup_iters=_CUDA_GRAPH_WARMUP_ITERS,
    )

    def forward() -> torch.Tensor:
        # The shipped integration generates only AR index 0. Its window is one
        # full chunk, so capture starts at index 1 and production uses drain.
        return graph_dispatch.select(0, uncond=False)(
            x=x,
            timesteps=timestep,
            cache=cache,
            rope_freqs=rope_freqs,
            current_chunk_idx=0,
            eager_mode=False,
        )

    assert capture_ar_index == 1
    cache.before_update(0)
    for _ in range(_AUTOTUNE_DRAIN_ROUNDS):
        output = forward()
    torch.cuda.synchronize(device)
    del output

    benchmark.group = "wan21-t2v-1.3b-dit-network"
    benchmark.extra_info.update(
        {
            "network": "WanDiTNetwork1pt3B",
            "integration": "wan21",
            "model_family": "wan",
            "model_variant": "wan21-t2v-1.3b-480p",
            "implementation": case.implementation,
            "execution_backend": "pytorch",
            "attention_backend": backend.value,
            "sdpa_backend": case.sdpa_backend.value,
            "self_attention_operator": case.self_attention_operator,
            "cross_attention_operator": case.cross_attention_operator,
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
            "global_chunk_tokens": _GLOBAL_CHUNK_TOKENS,
            "local_chunk_tokens": chunk_tokens,
            "global_window_tokens": _GLOBAL_WINDOW_TOKENS,
            "local_window_tokens": window_tokens,
            "global_sink_tokens": 0,
            "local_sink_tokens": 0,
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
            "self_attention_context_parallel_method": (
                config.cp_method if backend is AttentionBackend.WAN else None
            ),
            "local_attention_methods": sorted(local_attention_methods),
            "context_parallel_size": cp_size,
            "context_parallel_attention_modules": len(cp_enabled_attention_modules),
            "local_attention_modules": (
                len(attention_modules) - len(cp_enabled_attention_modules)
            ),
            "distributed_sample_alignment": "barrier_before_each_round",
            "compiled": True,
            "compile_mode": "max-autotune-no-cudagraphs",
            "cuda_graph_configured": True,
            "cuda_graph_selected": False,
            "cuda_graph_dispatch": "drain",
            "cuda_graph_capture_ar_index": capture_ar_index,
            "cuda_graph_warmup_iters": _CUDA_GRAPH_WARMUP_ITERS,
            "cache_state": "single_full_chunk_repeated_scheduler_slot",
            "benchmark_ar_index": 0,
            "cache_update_bookkeeping": "excluded_from_timing",
            "diffusion_timestep": _DIFFUSION_TIMESTEP,
            "pipeline_guidance_scale": _PIPELINE_GUIDANCE_SCALE,
            "classifier_free_guidance_branch": "conditional",
            "rope_interleaved": True,
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "autotune_drain_rounds": _AUTOTUNE_DRAIN_ROUNDS,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "compiler_cache_state": (
                "host-dependent; compile and autotune excluded from measured rounds"
            ),
            "seed": _SEED,
        }
    )

    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = forward()
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        setup=_synchronize_ranks,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(0)
    benchmark.extra_info["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
        device
    )

    expected_output_shape = (chunk_tokens, config.out_dim * patch_volume)
    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()
