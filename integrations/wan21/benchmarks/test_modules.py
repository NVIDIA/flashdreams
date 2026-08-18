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

"""Microbenchmarks for Wan 2.1 self-attention and DiT blocks.

Run all attention cases with::

    uv run --package flashdreams-wan21 --group test pytest \
        integrations/wan21/benchmarks/test_modules.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.distributed import init as init_distributed
from flashdreams.recipes.wan.transformer.impl.modules import (
    AttentionBackend,
    Block,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork1pt3BConfig,
)
from integrations.wan21.benchmarks.cases import (
    ATTENTION_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Wan 2.1 DiT module benchmarks require CUDA"

# The shipped T2V runner generates 480x832 pixels from a single 21-frame
# latent chunk. Wan VAE compression yields 21x60x104 latents, and the DiT's
# 1x2x2 patching produces 21x30x52 attention tokens.
_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_LATENT_HEIGHT = 60
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 21
_WINDOW_SIZE_T = 21
_SINK_SIZE_T = 0
_TEXT_TOKENS = 512
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
    context_parallel_size: int,
) -> None:
    """Skip case and execution combinations unsupported by production code."""
    skip_unsupported_device(case, device)
    if case.attention_backend is AttentionBackend.TRITON and context_parallel_size > 1:
        pytest.skip("Triton attention does not support context parallelism")


def _make_block(
    config: WanDiTNetwork1pt3BConfig,
    case: AttentionBenchmarkCase,
    device: torch.device,
    dtype: torch.dtype,
) -> Block:
    """Build a backend-selected block with shared random weights."""

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

    # Allocate the block directly at its benchmark precision. Initialization
    # and state-dict conversion remain outside the measured region.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            torch.manual_seed(_SEED)
            reference = make(AttentionBackend.WAN)
            if case.attention_backend is AttentionBackend.WAN:
                return reference
            block = make(case.attention_backend)
            block.load_state_dict(reference.state_dict(), strict=True)
            return block
    finally:
        torch.set_default_dtype(previous_dtype)


def _token_geometry(
    config: WanDiTNetwork1pt3BConfig,
    context_parallel_size: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Return global and per-rank token geometry for the shipped T2V shape."""
    patch_t = _CHUNK_SIZE_T // config.patch_size[0]
    patch_h = _LATENT_HEIGHT // config.patch_size[1]
    patch_w = _LATENT_WIDTH // config.patch_size[2]
    tokens_per_frame = patch_h * patch_w
    global_chunk_tokens = patch_t * tokens_per_frame
    global_window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    global_sink_tokens = _SINK_SIZE_T * tokens_per_frame
    assert global_chunk_tokens % context_parallel_size == 0
    assert global_window_tokens % context_parallel_size == 0
    assert global_sink_tokens % context_parallel_size == 0
    return (
        patch_t,
        patch_h,
        patch_w,
        global_chunk_tokens,
        global_window_tokens,
        global_sink_tokens,
        global_chunk_tokens // context_parallel_size,
        global_window_tokens // context_parallel_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    ATTENTION_CASES,
    ids=lambda case: case.pytest_id,
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark Wan 2.1 self-attention over its full single-chunk window."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan 2.1 self-attention benchmark requires bfloat16 support")

    dtype = torch.bfloat16
    cp_size = dist.get_world_size() if dist.is_initialized() else 1
    _skip_unsupported_case(case, device, cp_size)
    backend = case.attention_backend
    config = WanDiTNetwork1pt3BConfig(
        cp_method="ring",
        attention_backend=backend,
        sdpa_backend=case.sdpa_backend,
    )
    block = _make_block(config, case, device, dtype).eval()
    block.update_parameters_after_loading_checkpoint()
    attention = block.self_attn
    assert block.attention_backend is backend
    assert block.sdpa_backend is case.sdpa_backend
    del block

    cp_group = dist.group.WORLD if cp_size > 1 else None
    attention.set_context_parallel_group(cp_group)
    self_attention_cp_enabled = attention.is_context_parallel_enabled()
    assert self_attention_cp_enabled == (
        backend is AttentionBackend.WAN and cp_size > 1
    )

    (
        patch_t,
        patch_h,
        patch_w,
        global_chunk_tokens,
        global_window_tokens,
        global_sink_tokens,
        chunk_tokens,
        window_tokens,
    ) = _token_geometry(config, cp_size)
    sink_tokens = global_sink_tokens // cp_size
    head_dim = config.dim // config.num_heads
    generator = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(
        (chunk_tokens, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=sink_tokens,
        device=device,
        dtype=dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=patch_t,
        len_h=patch_h,
        len_w=patch_w,
        interleaved=True,
        device=device,
    )
    rope.set_context_parallel_group(cp_group)
    rope_freqs = rope.shift_t(0)

    # Populate the one-chunk window before timing. The measured calls overwrite
    # the same cache slot, matching repeated denoising evaluations at AR index 0.
    cache.before_update(0)
    output = attention(x, cache, rope_freqs)
    cache.after_update(0)
    torch.cuda.synchronize(device)
    del output

    benchmark.group = "wan21-dit-self-attention"
    benchmark.extra_info.update(
        {
            "module": "self_attention",
            "module_owner": "flashdreams.recipes.wan",
            "integration": "wan21",
            "model_family": "wan",
            "model_variant": "wan21-t2v-1.3b-480p",
            "implementation": case.implementation,
            "attention_backend": backend.value,
            "sdpa_backend": case.sdpa_backend.value,
            "self_attention_operator": case.self_attention_operator,
            "projection_backend": (
                "separate_qkv"
                if backend is AttentionBackend.WAN
                else "row_scaled_fp8_fused_qkv_output"
            ),
            "batch_shape": [],
            "flattened_batch_size": 1,
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "attention_grid": [patch_t, patch_h, patch_w],
            "global_chunk_tokens": global_chunk_tokens,
            "local_chunk_tokens": chunk_tokens,
            "global_window_tokens": global_window_tokens,
            "local_window_tokens": window_tokens,
            "global_sink_tokens": global_sink_tokens,
            "local_sink_tokens": sink_tokens,
            "model_channels": config.dim,
            "num_heads": config.num_heads,
            "head_dim": head_dim,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "cache_dtype": str(cache.dtype),
            "self_attention_context_parallel_method": (
                config.cp_method if backend is AttentionBackend.WAN else None
            ),
            "context_parallel_size": cp_size,
            "self_attention_context_parallel_enabled": (self_attention_cp_enabled),
            "distributed_sample_alignment": "barrier_before_each_round",
            "cache_state": "full_single_chunk_window",
            "cache_prefill_chunks": 1,
            "benchmark_ar_index": 0,
            "cache_update_bookkeeping": "excluded_from_timing",
            "rope_interleaved": True,
            "compiled": False,
            "cuda_graph": False,
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
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

    cache.before_update(0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = attention(x, cache, rope_freqs)
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

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    ATTENTION_CASES,
    ids=lambda case: case.pytest_id,
)
@torch.inference_mode()
def test_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the Wan 2.1 T2V DiT block over its full attention window."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan 2.1 DiT block benchmark requires bfloat16 support")

    dtype = torch.bfloat16
    cp_size = dist.get_world_size() if dist.is_initialized() else 1
    _skip_unsupported_case(case, device, cp_size)
    backend = case.attention_backend
    config = WanDiTNetwork1pt3BConfig(
        cp_method="ring",
        attention_backend=backend,
        sdpa_backend=case.sdpa_backend,
    )
    block = _make_block(config, case, device, dtype).eval()
    block.update_parameters_after_loading_checkpoint()
    assert block.attention_backend is backend
    assert block.sdpa_backend is case.sdpa_backend

    cp_group = dist.group.WORLD if cp_size > 1 else None
    block.set_context_parallel_group(cp_group)
    self_attention_cp_enabled = block.self_attn.is_context_parallel_enabled()
    cross_attention_cp_enabled = block.cross_attn.is_context_parallel_enabled()
    assert self_attention_cp_enabled == (
        backend is AttentionBackend.WAN and cp_size > 1
    )
    assert not cross_attention_cp_enabled

    (
        patch_t,
        patch_h,
        patch_w,
        global_chunk_tokens,
        global_window_tokens,
        global_sink_tokens,
        chunk_tokens,
        window_tokens,
    ) = _token_geometry(config, cp_size)
    sink_tokens = global_sink_tokens // cp_size
    head_dim = config.dim // config.num_heads
    generator = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(
        (chunk_tokens, config.dim),
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
        (1, _TEXT_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=sink_tokens,
        context_text=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=patch_t,
        len_h=patch_h,
        len_w=patch_w,
        interleaved=True,
        device=device,
    )
    rope.set_context_parallel_group(cp_group)
    rope_freqs = rope.shift_t(0)

    def forward() -> torch.Tensor:
        cache.before_update(0)
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs,
        )
        cache.after_update(0)
        return result

    # Populate the one-chunk window before timing. Cross-attention's projected
    # text context remains static throughout all measured denoising evaluations.
    output = forward()
    torch.cuda.synchronize(device)
    del output

    benchmark.group = "wan21-dit-block"
    benchmark.extra_info.update(
        {
            "module": "Block",
            "module_owner": "flashdreams.recipes.wan",
            "integration": "wan21",
            "model_family": "wan",
            "model_variant": "wan21-t2v-1.3b-480p",
            "benchmark_scope": "whole_wan_t2v_block",
            "implementation": case.implementation,
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
            "attention_grid": [patch_t, patch_h, patch_w],
            "global_chunk_tokens": global_chunk_tokens,
            "local_chunk_tokens": chunk_tokens,
            "global_window_tokens": global_window_tokens,
            "local_window_tokens": window_tokens,
            "global_sink_tokens": global_sink_tokens,
            "local_sink_tokens": sink_tokens,
            "text_tokens": _TEXT_TOKENS,
            "model_channels": config.dim,
            "ffn_channels": config.ffn_dim,
            "num_heads": config.num_heads,
            "head_dim": head_dim,
            "parameter_count": sum(
                parameter.numel() for parameter in block.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "self_attention_cache_dtype": str(cache.self_attn.dtype),
            "cross_attention_cache_dtype": str(cache.cross_attn.text.dtype),
            "self_attention_context_parallel_method": (
                config.cp_method if backend is AttentionBackend.WAN else None
            ),
            "cross_attention_method": (
                config.cp_method if backend is AttentionBackend.WAN else None
            ),
            "context_parallel_size": cp_size,
            "self_attention_context_parallel_enabled": (self_attention_cp_enabled),
            "cross_attention_context_parallel_enabled": (cross_attention_cp_enabled),
            "distributed_sample_alignment": "barrier_before_each_round",
            "cache_state": "full_single_chunk_window_static_text",
            "cache_prefill_chunks": 1,
            "benchmark_ar_index": 0,
            "cache_update_bookkeeping": "excluded_from_timing",
            "rope_interleaved": True,
            "compiled": False,
            "cuda_graph": False,
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
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

    cache.before_update(0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def synchronized_forward() -> torch.Tensor:
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs,
        )
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

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
