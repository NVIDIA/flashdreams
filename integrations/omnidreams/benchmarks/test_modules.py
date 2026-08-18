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
        integrations/omnidreams/benchmarks/test_modules.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import pytest
import torch
from omnidreams.transformer.impl.modules import (
    AttentionBackend,
    Block,
)
from omnidreams.transformer.impl.network import CosmosDiTNetworkConfig
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from integrations.omnidreams.benchmarks.cases import (
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

_OMNIDREAMS_TORCH_CASE = next(
    case for case in BENCHMARK_CASES if case.implementation == "omnidreams_torch"
)
_HYBRID_ATTENTION_CASE = next(
    case
    for case in BENCHMARK_CASES
    if case.implementation == "triton_cudnn_bf16_full_omnidreams_cross"
)

_MODULE_BENCHMARK_CASES = [
    _OMNIDREAMS_TORCH_CASE,
    *[
        AttentionBenchmarkCase(
            implementation=(
                f"triton_{'fa2' if sdpa_backend is SDPABackend.TRITON else 'cudnn'}_"
                f"{'fp8' if use_fp8 else 'bf16'}_{qkv_fusion_option.value}"
            ),
            self_attention_backend=AttentionBackend.TRITON,
            cross_attention_backend=AttentionBackend.TRITON,
            sdpa_backend=sdpa_backend,
            self_attention_operator=(
                "triton_fa2"
                if sdpa_backend is SDPABackend.TRITON
                else "torch_cudnn_sdpa"
            ),
            cross_attention_operator=(
                "triton_fa2"
                if sdpa_backend is SDPABackend.TRITON
                else "torch_cudnn_sdpa"
            ),
            use_fp8=use_fp8,
            self_attn_qkv_fusion_option=qkv_fusion_option,
            cross_attn_qkv_fusion_option=(
                qkv_fusion_option
                if qkv_fusion_option is not QKVFusionOption.FULL
                else QKVFusionOption.FUSE_KV
            ),
            minimum_compute_capability=(
                (9, 0) if use_fp8 or sdpa_backend is SDPABackend.TRITON else None
            ),
        )
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in QKVFusionOption
    ],
    _HYBRID_ATTENTION_CASE,
]
_MODULE_SELF_ATTENTION_CASES = [
    case
    for case in _MODULE_BENCHMARK_CASES
    if case.self_attention_backend is case.cross_attention_backend
]
_MODULE_CROSS_ATTENTION_CASES = [
    case
    for case in _MODULE_BENCHMARK_CASES
    if case.self_attention_backend is AttentionBackend.OMNIDREAMS
    or case.self_attn_qkv_fusion_option is not QKVFusionOption.FULL
]


def _module_config(case: AttentionBenchmarkCase) -> CosmosDiTNetworkConfig:
    """Build the network config for one module benchmark row."""
    return CosmosDiTNetworkConfig(
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        sdpa_backend=case.sdpa_backend,
        cross_attn_sdpa_backend=case.sdpa_backend,
        self_attn_qkv_fusion_option=case.self_attn_qkv_fusion_option,
        cross_attn_qkv_fusion_option=case.cross_attn_qkv_fusion_option,
        use_fp8=case.use_fp8,
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
            sdpa_backend=config.sdpa_backend,
            cross_attn_sdpa_backend=config.cross_attn_sdpa_backend,
            self_attn_qkv_fusion_option=config.self_attn_qkv_fusion_option,
            cross_attn_qkv_fusion_option=config.cross_attn_qkv_fusion_option,
            use_fp8=config.use_fp8,
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
@pytest.mark.parametrize(
    "case", _MODULE_BENCHMARK_CASES, ids=lambda case: case.pytest_id
)
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
    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "num_views": _NUM_VIEWS,
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "chunk_tokens": chunk_tokens,
            "window_tokens": window_tokens,
            "text_tokens": _TEXT_TOKENS,
            "model_channels": config.model_channels,
            "num_heads": config.num_heads,
            "parameter_count": sum(
                parameter.numel() for parameter in block.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "implementation": case.implementation,
            "sdpa_backend": case.sdpa_backend.value,
            "use_fp8": case.use_fp8,
            "self_attn_qkv_fusion_option": case.self_attn_qkv_fusion_option.value,
            "cross_attn_qkv_fusion_option": case.cross_attn_qkv_fusion_option.value,
            "attention_backend": case.self_attention_operator,
            "self_attention_backend": case.self_attention_operator,
            "cross_attention_backend": case.cross_attention_operator,
            "self_attention_cache_dtype": str(cache.self_attn.dtype),
            "cross_attention_cache_dtype": str(cache.cross_attn.dtype),
            "cache_state": "full_window_static_context",
            "cache_prefill_chunks": steady_chunk_idx + 1,
            "benchmark_chunk_idx": steady_chunk_idx,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

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
    "case", _MODULE_SELF_ATTENTION_CASES, ids=lambda case: case.pytest_id
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark self-attention against a full production KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams self-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    attention = _make_block(config, case).self_attn.to(device=device, dtype=dtype)
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
    benchmark.extra_info.update(
        {
            "module": "self_attention",
            "batch_size": _BATCH_SIZE,
            "num_views": _NUM_VIEWS,
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "chunk_tokens": chunk_tokens,
            "window_tokens": window_tokens,
            "model_channels": config.model_channels,
            "num_heads": config.num_heads,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "implementation": case.implementation,
            "sdpa_backend": case.sdpa_backend.value,
            "use_fp8": case.use_fp8,
            "qkv_fusion_option": case.self_attn_qkv_fusion_option.value,
            "attention_backend": case.self_attention_operator,
            "cache_dtype": str(cache.dtype),
            "cache_state": "full_window",
            "cache_prefill_chunks": steady_chunk_idx + 1,
            "benchmark_chunk_idx": steady_chunk_idx,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

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
    "case", _MODULE_CROSS_ATTENTION_CASES, ids=lambda case: case.pytest_id
)
@torch.inference_mode()
def test_cross_attention_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark cross-attention against the cached production text context."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams cross-attention benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    config = _module_config(case)
    attention = _make_block(config, case).cross_attn.to(device=device, dtype=dtype)
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
    cache = attention.compute_kv(context)
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-cross-attention"
    benchmark.extra_info.update(
        {
            "module": "cross_attention",
            "batch_size": _BATCH_SIZE,
            "num_views": _NUM_VIEWS,
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "chunk_tokens": chunk_tokens,
            "text_tokens": _TEXT_TOKENS,
            "model_channels": config.model_channels,
            "context_channels": config.crossattn_emb_channels,
            "num_heads": config.num_heads,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(dtype),
            "implementation": case.implementation,
            "sdpa_backend": case.sdpa_backend.value,
            "use_fp8": case.use_fp8,
            "qkv_fusion_option": case.cross_attn_qkv_fusion_option.value,
            "attention_backend": case.cross_attention_operator,
            "cache_dtype": str(cache.dtype),
            "cache_state": "static_context",
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

    def synchronized_forward() -> torch.Tensor:
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
