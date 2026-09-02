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

"""Microbenchmarks for LingBot transformer modules.

Run the module benchmarks with::

    uv run --group test pytest \
        integrations/lingbot/benchmarks/test_modules.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch
from lingbot.transformer.impl.modules import CamCtrlBlock
from lingbot.transformer.impl.network import LingbotWorldDiTNetwork14BConfig

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from integrations.lingbot.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot transformer module benchmarks require CUDA"

# Production 464x832 geometry becomes 58x104 latents. Wan's 2x2 spatial
# patching produces 29x52 tokens per frame, and each chunk has three frames.
_LATENT_HEIGHT = 58
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 3
_WINDOW_SIZE_T = 15
_SINK_SIZE_T = 3
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


def _implementation_id(optimized_impl_config: OptimizedImplConfig | None) -> str:
    """Return a stable pytest identifier for an attention implementation."""
    if optimized_impl_config is None:
        return "torch"
    backend = optimized_impl_config.sdpa_backend.value
    fusion = optimized_impl_config.qkv_fusion_option.value.replace("_", "-")
    tma = "tma" if optimized_impl_config.use_tma else "no-tma"
    projection_dtype = optimized_impl_config.quantization.projection
    quantization = (
        ""
        if projection_dtype is None
        else f"-projection-{projection_dtype}".replace("torch.", "").replace("_", "-")
    )
    quantized_sdpa = (
        "-quantized-sdpa" if optimized_impl_config.quantization.quantized_sdpa else ""
    )
    return f"optimized-{backend}-{fusion}-{tma}{quantization}{quantized_sdpa}"


_CUDNN_OPTIMIZED_IMPL_CONFIGS = tuple(
    OptimizedImplConfig(
        sdpa_backend=SDPABackend.CUDNN,
        qkv_fusion_option=qkv_fusion_option,
        use_tma=False,
        quantization=QuantizationOption(
            projection=projection_dtype,
            quantized_sdpa=quantized_sdpa,
        ),
    )
    for qkv_fusion_option in QKVFusionOption
    for projection_dtype in (None, torch.float8_e4m3fn)
    for quantized_sdpa in (False, True)
)
"""Every cuDNN fusion and quantization policy; TMA only affects FA2 dispatch."""

_FA2_OPTIMIZED_IMPL_CONFIGS = tuple(
    OptimizedImplConfig(
        sdpa_backend=SDPABackend.FA2,
        qkv_fusion_option=qkv_fusion_option,
        use_tma=use_tma,
        quantization=QuantizationOption(
            projection=projection_dtype,
            quantized_sdpa=quantized_sdpa,
        ),
    )
    for qkv_fusion_option in QKVFusionOption
    for use_tma in (False, True)
    for projection_dtype in (None, torch.float8_e4m3fn)
    for quantized_sdpa in (False, True)
)
"""Every FA2 fusion, TMA, and quantization policy."""

_OPTIMIZED_IMPL_CONFIGS = (
    *_CUDNN_OPTIMIZED_IMPL_CONFIGS,
    *_FA2_OPTIMIZED_IMPL_CONFIGS,
)
"""Every backend-specific policy used by the isolated attention benchmarks."""

_MODULE_ATTENTION_CONFIGS = (None, *_OPTIMIZED_IMPL_CONFIGS)
"""Torch attention plus every optimized implementation policy."""


def _block_case(
    self_config: OptimizedImplConfig | None,
    cross_config: OptimizedImplConfig | None,
) -> AttentionBenchmarkCase:
    """Build one self/cross implementation combination for block timing."""
    reference_config = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.NONE,
        sdpa_backend=SDPABackend.CUDNN,
    )
    needs_hopper = self_config is not None or cross_config is not None
    return AttentionBenchmarkCase(
        implementation=(
            f"self_{_implementation_id(self_config)}_"
            f"cross_{_implementation_id(cross_config)}"
        ),
        self_attention_backend=(
            AttentionBackend.TORCH
            if self_config is None
            else AttentionBackend.OPTIMIZED
        ),
        cross_attention_backend=(
            AttentionBackend.TORCH
            if cross_config is None
            else AttentionBackend.OPTIMIZED
        ),
        self_attn_optimized_impl_config=self_config or reference_config,
        cross_attn_optimized_impl_config=cross_config or reference_config,
        minimum_compute_capability=(9, 0) if needs_hopper else None,
    )


def _module_block_cases(full_policy_search: bool) -> tuple[AttentionBenchmarkCase, ...]:
    """Build representative or exhaustive self/cross pairs for block timing."""
    if not full_policy_search:
        return BENCHMARK_CASES
    return tuple(
        _block_case(self_config, cross_config)
        for self_config in _MODULE_ATTENTION_CONFIGS
        for cross_config in _MODULE_ATTENTION_CONFIGS
    )


_FULL_POLICY_SEARCH = os.environ.get("FLASHDREAMS_RUN_FULL_BENCHMARK") == "1"
"""Whether to benchmark every self/cross policy pair in the full DiT block."""

_MODULE_BLOCK_CASES = _module_block_cases(_FULL_POLICY_SEARCH)
"""Selected block cases; set ``FLASHDREAMS_RUN_FULL_BENCHMARK=1`` for all 1,369."""


def _module_config(case: AttentionBenchmarkCase) -> LingbotWorldDiTNetwork14BConfig:
    """Build the production network config for one module benchmark row."""
    return LingbotWorldDiTNetwork14BConfig(
        in_dim=36,
        patch_embedding_type="conv3d",
        control_type="cam",
        cp_method="ring",
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        self_attn_optimized_impl_config=case.self_attn_optimized_impl_config,
        cross_attn_optimized_impl_config=case.cross_attn_optimized_impl_config,
    )


def _make_block(
    config: LingbotWorldDiTNetwork14BConfig,
    case: AttentionBenchmarkCase,
) -> CamCtrlBlock:
    """Build a backend-selected block with shared random weights."""

    def make(
        self_backend: AttentionBackend,
        cross_backend: AttentionBackend,
    ) -> CamCtrlBlock:
        return CamCtrlBlock(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            cp_method=config.cp_method,
            self_attention_backend=self_backend,
            cross_attention_backend=cross_backend,
            self_attn_optimized_impl_config=config.self_attn_optimized_impl_config,
            cross_attn_optimized_impl_config=config.cross_attn_optimized_impl_config,
        )

    torch.manual_seed(_SEED)
    torch_block = make(AttentionBackend.TORCH, AttentionBackend.TORCH)
    if (
        case.self_attention_backend is AttentionBackend.TORCH
        and case.cross_attention_backend is AttentionBackend.TORCH
    ):
        return torch_block

    block = make(case.self_attention_backend, case.cross_attention_backend)
    block.load_state_dict(torch_block.state_dict(), strict=True)
    return block


def _token_geometry(
    config: LingbotWorldDiTNetwork14BConfig,
) -> tuple[int, int, int, int]:
    """Return patch height, patch width, chunk tokens, and cache tokens."""
    patch_t = _CHUNK_SIZE_T // config.patch_size[0]
    patch_h = _LATENT_HEIGHT // config.patch_size[1]
    patch_w = _LATENT_WIDTH // config.patch_size[2]
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    cache_tokens = (_SINK_SIZE_T + _WINDOW_SIZE_T) * tokens_per_frame
    return patch_h, patch_w, chunk_tokens, cache_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", _MODULE_BLOCK_CASES, ids=lambda case: case.pytest_id)
@torch.inference_mode()
def test_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark a production-configured camera-control block at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot DiT block benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    block = _make_block(config, case).to(device=device, dtype=dtype)
    block.eval()
    block.update_parameters_after_loading_checkpoint()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_h, patch_w, chunk_tokens, cache_tokens = _token_geometry(config)
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
    plucker_embedding = torch.randn(
        (chunk_tokens, config.dim),
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
    tokens_per_frame = patch_h * patch_w
    cache = block.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=_WINDOW_SIZE_T * tokens_per_frame,
        sink_size=_SINK_SIZE_T * tokens_per_frame,
        context_text=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_h=patch_h,
        len_w=patch_w,
        len_t=_CHUNK_SIZE_T // config.patch_size[0],
        interleaved=True,
        device=device,
    )

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        output = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs,
            plucker_embedding=plucker_embedding,
        )
        cache.after_update(chunk_idx)
        return output

    cache_chunks = cache_tokens // chunk_tokens
    benchmark_chunk_idx = cache_chunks - 1
    rope_freqs = [rope.shift_t(idx) for idx in range(cache_chunks)]
    for chunk_idx, chunk_rope_freqs in enumerate(rope_freqs):
        output = forward(chunk_idx, chunk_rope_freqs)
    torch.cuda.synchronize()

    benchmark.group = "lingbot-dit-block"

    def synchronized_forward() -> torch.Tensor:
        result = forward(benchmark_chunk_idx, rope_freqs[benchmark_chunk_idx])
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "optimized_impl_config",
    _MODULE_ATTENTION_CONFIGS,
    ids=_implementation_id,
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
) -> None:
    """Benchmark self-attention against the full production KV cache."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot self-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    case = _block_case(optimized_impl_config, None)
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    attention = _make_block(config, case).self_attn.to(device=device, dtype=dtype)
    attention.eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_h, patch_w, chunk_tokens, cache_tokens = _token_geometry(config)
    x = torch.randn(
        (chunk_tokens, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = attention.initialize_cache(
        batch_size=1,
        chunk_size=chunk_tokens,
        window_size=_WINDOW_SIZE_T * patch_h * patch_w,
        sink_size=_SINK_SIZE_T * patch_h * patch_w,
        device=device,
        dtype=dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_h=patch_h,
        len_w=patch_w,
        len_t=_CHUNK_SIZE_T // config.patch_size[0],
        interleaved=True,
        device=device,
    )

    cache_chunks = cache_tokens // chunk_tokens
    benchmark_chunk_idx = cache_chunks - 1
    rope_freqs = [rope.shift_t(idx) for idx in range(cache_chunks)]
    for chunk_idx, chunk_rope_freqs in enumerate(rope_freqs):
        cache.before_update(chunk_idx)
        output = attention(x, kv_cache=cache, rope_freqs=chunk_rope_freqs)
        cache.after_update(chunk_idx)
    torch.cuda.synchronize()

    benchmark.group = "lingbot-dit-self-attention"
    cache.before_update(benchmark_chunk_idx)

    def synchronized_forward() -> torch.Tensor:
        result = attention(
            x,
            kv_cache=cache,
            rope_freqs=rope_freqs[benchmark_chunk_idx],
        )
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "optimized_impl_config",
    _MODULE_ATTENTION_CONFIGS,
    ids=_implementation_id,
)
@torch.inference_mode()
def test_cross_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
) -> None:
    """Benchmark cross-attention including text KV projection."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot cross-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    case = _block_case(None, optimized_impl_config)
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    attention = _make_block(config, case).cross_attn.to(device=device, dtype=dtype)
    attention.eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    _, _, chunk_tokens, _ = _token_geometry(config)
    x = torch.randn(
        (chunk_tokens, config.dim),
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
    torch.cuda.synchronize()

    benchmark.group = "lingbot-dit-cross-attention"

    def synchronized_forward() -> torch.Tensor:
        cache = attention.initialize_cache(context)
        result = attention(x, kv_cache=cache)
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
