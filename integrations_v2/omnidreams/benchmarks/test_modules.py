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

"""Microbenchmarks for Omnidreams model modules.

Run the module benchmarks with::

    uv run --group test pytest \
        integrations_v2/omnidreams/benchmarks/test_modules.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch
from omnidreams.impl.transformer.modules import (
    AttentionBackend,
    Block,
)
from omnidreams.impl.transformer.network import CosmosDiTNetworkConfig

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from integrations_v2.omnidreams.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Omnidreams DiT module benchmarks require CUDA"

# Production single-view, 720p chunk-2 geometry. The VAE reduces 720x1280 to
# 90x160 latents, and the DiT's 2x2 spatial patching produces 45x80 tokens per
# latent frame. The local window holds three two-frame chunks.
_BATCH_SIZE = 1
_NUM_VIEWS = 1
_LATENT_HEIGHT = 90
_LATENT_WIDTH = 160
_CHUNK_SIZE_T = 2
_WINDOW_SIZE_T = 6
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


def _implementation_id(optimized_impl_config: OptimizedImplConfig | None) -> str:
    """Return a stable pytest identifier for an attention implementation."""
    if optimized_impl_config is None:
        return "omnidreams"
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

_MODULE_SELF_ATTENTION_CONFIGS = (None, *_OPTIMIZED_IMPL_CONFIGS)
"""Reference self-attention plus every Optimized implementation config."""

_MODULE_CROSS_ATTENTION_CONFIGS = (
    None,
    *(
        config
        for config in _OPTIMIZED_IMPL_CONFIGS
        if config.qkv_fusion_option is not QKVFusionOption.FULL
    ),
)
"""Reference cross-attention plus every valid Optimized implementation config.

Full QKV fusion requires equal query and context widths, which production
OmniDreams text cross-attention does not have.
"""


def _block_case(
    self_config: OptimizedImplConfig | None,
    cross_config: OptimizedImplConfig | None,
) -> AttentionBenchmarkCase:
    """Build one self/cross implementation combination for the DiT block."""
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
            AttentionBackend.OMNIDREAMS
            if self_config is None
            else AttentionBackend.OPTIMIZED
        ),
        cross_attention_backend=(
            AttentionBackend.OMNIDREAMS
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
        return tuple(case for case in BENCHMARK_CASES if not case.native_dit)
    return tuple(
        _block_case(self_config, cross_config)
        for self_config in _MODULE_SELF_ATTENTION_CONFIGS
        for cross_config in _MODULE_CROSS_ATTENTION_CONFIGS
    )


_FULL_POLICY_SEARCH = os.environ.get("FLASHDREAMS_RUN_FULL_BENCHMARK") == "1"
"""Whether to benchmark every self/cross pair in the full DiT block."""

_MODULE_BLOCK_CASES = _module_block_cases(_FULL_POLICY_SEARCH)
"""Selected block cases; set FLASHDREAMS_RUN_FULL_BENCHMARK=1 for all 925."""


def _module_config(case: AttentionBenchmarkCase) -> CosmosDiTNetworkConfig:
    """Build the network config for one module benchmark row."""
    return CosmosDiTNetworkConfig(
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        self_attn_optimized_impl_config=case.self_attn_optimized_impl_config,
        cross_attn_optimized_impl_config=case.cross_attn_optimized_impl_config,
    )


def _make_block(
    config: CosmosDiTNetworkConfig,
    case: AttentionBenchmarkCase,
) -> Block:
    """Build a backend-selected block with shared random weights."""

    def make(self_backend: AttentionBackend, cross_backend: AttentionBackend) -> Block:
        # Keep this constructor in lockstep with CosmosDiTNetwork.__init__.
        return Block(
            x_dim=config.model_channels,
            context_dim=config.crossattn_emb_channels,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            use_adaln_lora=config.use_adaln_lora,
            adaln_lora_dim=config.adaln_lora_dim,
            enable_cross_view_attn=config.enable_cross_view_attn,
            cp_method=config.cp_method,
            self_attention_backend=self_backend,
            cross_attention_backend=cross_backend,
            self_attn_optimized_impl_config=config.self_attn_optimized_impl_config,
            cross_attn_optimized_impl_config=config.cross_attn_optimized_impl_config,
        )

    torch.manual_seed(_SEED)
    omnidreams_block = make(AttentionBackend.OMNIDREAMS, AttentionBackend.OMNIDREAMS)
    if (
        case.self_attention_backend is AttentionBackend.OMNIDREAMS
        and case.cross_attention_backend is AttentionBackend.OMNIDREAMS
    ):
        return omnidreams_block

    block = make(case.self_attention_backend, case.cross_attention_backend)
    block.load_state_dict(omnidreams_block.state_dict(), strict=True)
    return block


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", _MODULE_BLOCK_CASES, ids=lambda case: case.pytest_id)
@torch.inference_mode()
def test_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark a production-configured DiT block with a full KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams DiT block benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    block = _make_block(config, case).to(device=device, dtype=dtype)
    block.eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_t = _CHUNK_SIZE_T // config.patch_temporal
    patch_h = _LATENT_HEIGHT // config.patch_spatial
    patch_w = _LATENT_WIDTH // config.patch_spatial
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    head_dim = config.model_channels // config.num_heads

    x = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            patch_t,
            tokens_per_frame,
            config.model_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    emb = torch.randn(
        (_BATCH_SIZE, config.model_channels),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    adaln_lora = torch.randn(
        (_BATCH_SIZE, 3 * config.model_channels),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _TEXT_TOKENS,
            config.crossattn_emb_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=0,
        context=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_h=patch_h,
        len_w=patch_w,
        len_t=patch_t,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        device=device,
    )

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        output = block(
            x=x,
            emb=emb,
            cache=cache,
            rope_freqs=rope_freqs,
            adaln_lora=adaln_lora,
        )
        cache.after_update(chunk_idx)
        return output

    # Fill the rolling cache before timing so every measured call exercises
    # steady-state attention over the full local window. Repeating the final
    # chunk mirrors multiple denoising steps at one autoregressive position.
    steady_chunk_idx = _WINDOW_SIZE_T // _CHUNK_SIZE_T - 1
    rope_freqs = [rope.shift_t(chunk_idx) for chunk_idx in range(steady_chunk_idx + 1)]
    for chunk_idx, chunk_rope_freqs in enumerate(rope_freqs):
        output = forward(chunk_idx, chunk_rope_freqs)
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-block"

    def synchronized_forward() -> torch.Tensor:
        result = forward(steady_chunk_idx, rope_freqs[steady_chunk_idx])
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
    _MODULE_SELF_ATTENTION_CONFIGS,
    ids=_implementation_id,
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
) -> None:
    """Benchmark self-attention against a full production KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams self-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    block_case = _block_case(optimized_impl_config, None)
    skip_unsupported_device(block_case, device)
    dtype = torch.bfloat16
    config = _module_config(block_case)
    attention = _make_block(config, block_case).self_attn.to(device=device, dtype=dtype)
    attention.eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_t = _CHUNK_SIZE_T // config.patch_temporal
    patch_h = _LATENT_HEIGHT // config.patch_spatial
    patch_w = _LATENT_WIDTH // config.patch_spatial
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    x = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            chunk_tokens,
            config.model_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = attention.allocate_kv_cache(
        batch_size=_BATCH_SIZE * _NUM_VIEWS,
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=0,
        device=device,
        dtype=dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.model_channels // config.num_heads,
        len_h=patch_h,
        len_w=patch_w,
        len_t=patch_t,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        device=device,
    )

    steady_chunk_idx = _WINDOW_SIZE_T // _CHUNK_SIZE_T - 1
    rope_freqs = [rope.shift_t(chunk_idx) for chunk_idx in range(steady_chunk_idx + 1)]
    for chunk_idx, chunk_rope_freqs in enumerate(rope_freqs):
        cache.before_update(chunk_idx)
        output = attention(x, kv_cache=cache, rope_freqs=chunk_rope_freqs)
        cache.after_update(chunk_idx)
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-self-attention"

    # Repeated denoising evaluations at one autoregressive position overwrite
    # the final cache chunk while attending over the same full window.
    cache.before_update(steady_chunk_idx)

    def synchronized_forward() -> torch.Tensor:
        result = attention(
            x,
            kv_cache=cache,
            rope_freqs=rope_freqs[steady_chunk_idx],
        )
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(steady_chunk_idx)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "optimized_impl_config",
    _MODULE_CROSS_ATTENTION_CONFIGS,
    ids=_implementation_id,
)
@torch.inference_mode()
def test_cross_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
) -> None:
    """Benchmark cross-attention including production text KV projection."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams cross-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    block_case = _block_case(None, optimized_impl_config)
    skip_unsupported_device(block_case, device)
    dtype = torch.bfloat16
    config = _module_config(block_case)
    attention = _make_block(config, block_case).cross_attn.to(
        device=device, dtype=dtype
    )
    attention.eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_t = _CHUNK_SIZE_T // config.patch_temporal
    patch_h = _LATENT_HEIGHT // config.patch_spatial
    patch_w = _LATENT_WIDTH // config.patch_spatial
    chunk_tokens = patch_t * patch_h * patch_w
    x = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            chunk_tokens,
            config.model_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _TEXT_TOKENS,
            config.crossattn_emb_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-cross-attention"

    def synchronized_forward() -> torch.Tensor:
        cache = attention.compute_kv(context)
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
