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

"""Benchmark the complete Omnidreams DiT network.

Run the benchmark with::

    uv run --group test pytest \
        integrations_v2/omnidreams/benchmarks/test_network.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from omnidreams.apps.interactive_drive.adapter import (
    OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
)
from omnidreams.impl.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.impl.transformer.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.infra.acceleration import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.compile import compile_module
from integrations_v2.omnidreams.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Omnidreams DiT network benchmark requires CUDA"

# Production single-view distilled runner geometry: 704x1280 pixels become
# 88x160 latents, the DiT consumes two latent frames per chunk, and the local
# window retains three chunks. HDMap conditioning uses 16 latent channels.
_BATCH_SIZE = 1
_NUM_VIEWS = 1
_PIXEL_HEIGHT = OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.height
_PIXEL_WIDTH = OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.width
_LATENT_HEIGHT = 88
_LATENT_WIDTH = 160
_CHUNK_SIZE_T = 2
_WINDOW_SIZE_T = 6
_TEXT_TOKENS = 512
_HDMAP_CHANNELS = 16
_DIFFUSION_TIMESTEP = 450.0
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    [case for case in BENCHMARK_CASES if not case.native_dit],
    ids=lambda case: case.pytest_id,
)
@torch.inference_mode()
def test_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark one production compiled PyTorch DiT backend at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams DiT network benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    torch.manual_seed(_SEED)

    config = CosmosDiTNetworkConfig(
        additional_concat_ch=_HDMAP_CHANNELS,
        enable_cross_view_attn=False,
        cp_method="ring",
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        self_attn_optimized_impl_config=case.self_attn_optimized_impl_config,
        cross_attn_optimized_impl_config=case.cross_attn_optimized_impl_config,
    )
    network = CosmosDiTNetwork(config).to(device=device, dtype=dtype)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    assert all(
        block.self_attention_backend is case.self_attention_backend
        for block in network.blocks
    )
    assert all(
        block.cross_attention_backend is case.cross_attention_backend
        for block in network.blocks
    )
    assert all(
        block.self_attn_optimized_impl_config is case.self_attn_optimized_impl_config
        for block in network.blocks
    )
    assert all(
        block.cross_attn_optimized_impl_config is case.cross_attn_optimized_impl_config
        for block in network.blocks
    )
    generator = torch.Generator(device=device).manual_seed(_SEED)

    patch_t = _CHUNK_SIZE_T // config.patch_temporal
    patch_h = _LATENT_HEIGHT // config.patch_spatial
    patch_w = _LATENT_WIDTH // config.patch_spatial
    patch_volume = config.patch_temporal * config.patch_spatial**2
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    head_dim = config.model_channels // config.num_heads

    latent_patch_dim = config.in_channels * patch_volume
    mask_patch_dim = patch_volume
    hdmap_patch_dim = config.additional_concat_ch * patch_volume
    x = torch.randn(
        (_BATCH_SIZE, _NUM_VIEWS, chunk_tokens, latent_patch_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    condition_mask = torch.zeros(
        (_BATCH_SIZE, _NUM_VIEWS, chunk_tokens, mask_patch_dim),
        device=device,
        dtype=dtype,
    )
    hdmap_condition = torch.randn(
        (_BATCH_SIZE, _NUM_VIEWS, chunk_tokens, hdmap_patch_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor(_DIFFUSION_TIMESTEP, device=device, dtype=dtype)
    context = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _TEXT_TOKENS,
            config.crossattn_proj_in_channels,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = network.initialize_cache(
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

    network = compile_module(network)
    capture_chunk_idx = cuda_graph_capture_ar_index(
        sink_size_t=0,
        window_size_t=_WINDOW_SIZE_T,
        len_t=_CHUNK_SIZE_T,
    )
    graph_dispatch = CUDAGraphDispatch(
        network,
        enabled=True,
        capture_ar_idx=capture_chunk_idx,
        warmup_iters=2,
    )

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        return graph_dispatch.select(chunk_idx, uncond=False)(
            x=x,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache,
            condition_video_input_mask=condition_mask,
            current_chunk_idx=chunk_idx,
            hdmap_condition=hdmap_condition,
            view_indices=None,
            eager_mode=False,
        )

    # Fill and roll every per-block KV cache through the production CUDA-graph
    # threshold before timing. Benchmark warmups finish graph capture.
    benchmark_chunk_idx = capture_chunk_idx + 1
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(capture_chunk_idx + 1):
        cache.before_update(chunk_idx)
        output = forward(chunk_idx, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-network"

    # Repeated scheduler evaluations overwrite one production steady-state slot.
    cache.before_update(benchmark_chunk_idx)

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
    cache.after_update(benchmark_chunk_idx)

    expected_output_shape = (
        _BATCH_SIZE,
        _NUM_VIEWS,
        chunk_tokens,
        config.out_channels * patch_volume,
    )
    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    [case for case in BENCHMARK_CASES if case.native_dit],
    ids=lambda case: case.pytest_id,
)
@torch.inference_mode()
def test_native_cuda_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the production native CUDA DiT backend at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams native DiT benchmark requires bfloat16 support")
    if case.native_dit_backend == "fp8_kvcache_cudnn" and not hasattr(
        torch, "float8_e4m3fn"
    ):
        pytest.skip("Omnidreams native DiT benchmark requires float8_e4m3fn")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    torch.manual_seed(_SEED)

    network_config = CosmosDiTNetworkConfig(
        additional_concat_ch=_HDMAP_CHANNELS,
        enable_cross_view_attn=False,
        cp_method="ring",
    )
    transformer_config = CosmosTransformerConfig(
        network=network_config,
        dtype=dtype,
        batch_shape=(_BATCH_SIZE,),
        num_views=_NUM_VIEWS,
        len_t=_CHUNK_SIZE_T,
        window_size_t=_WINDOW_SIZE_T,
        sink_size_t=0,
        compile_network=False,
        use_cuda_graph=True,
        native_dit_acceleration="required",
        native_dit_backend=case.native_dit_backend,
        native_dit_attention_backend=case.native_attention_backend,
    )
    transformer = CosmosTransformer(transformer_config).to(device=device, dtype=dtype)
    transformer.eval()

    x_unpatched = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _CHUNK_SIZE_T,
            network_config.in_channels,
            _LATENT_HEIGHT,
            _LATENT_WIDTH,
        ),
        device=device,
        dtype=dtype,
    )
    hdmap_unpatched = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _CHUNK_SIZE_T,
            network_config.additional_concat_ch,
            _LATENT_HEIGHT,
            _LATENT_WIDTH,
        ),
        device=device,
        dtype=dtype,
    )
    image_embeddings = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            1,
            network_config.in_channels,
            _LATENT_HEIGHT,
            _LATENT_WIDTH,
        ),
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _TEXT_TOKENS,
            network_config.crossattn_proj_in_channels,
        ),
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor(_DIFFUSION_TIMESTEP, device=device, dtype=dtype)

    cache = transformer.initialize_autoregressive_cache(
        height=_LATENT_HEIGHT,
        width=_LATENT_WIDTH,
        text_embeddings=context,
        image_embeddings=image_embeddings,
    )
    x = transformer.patchify_and_maybe_split_cp(x_unpatched)
    hdmap_condition = transformer.patchify_and_maybe_split_cp(hdmap_unpatched)

    patch_t = _CHUNK_SIZE_T // network_config.patch_temporal
    patch_h = _LATENT_HEIGHT // network_config.patch_spatial
    patch_w = _LATENT_WIDTH // network_config.patch_spatial
    patch_volume = network_config.patch_temporal * network_config.patch_spatial**2
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame

    def forward() -> torch.Tensor:
        return transformer.predict_flow(
            noisy_latent=x,
            timestep=timestep,
            cache=cache,
            input=hdmap_condition,
        )

    # Build the native runtime and FP8 weights, then fill and roll the cache
    # through the production CUDA-graph threshold. Benchmark warmups finish
    # graph capture before measured rounds.
    capture_chunk_idx = transformer._cuda_graph_capture_ar_idx
    benchmark_chunk_idx = capture_chunk_idx + 1
    for chunk_idx in range(capture_chunk_idx + 1):
        cache.start(chunk_idx)
        output = forward()
        cache.finalize(chunk_idx)
    torch.cuda.synchronize()

    native_selection = transformer._optimized_dit_selection
    native_executor = transformer._optimized_dit_executor
    assert native_selection is not None and native_selection.enabled
    assert native_executor is not None

    assert native_executor._uses_fp8_dit is (
        case.native_dit_backend == "fp8_kvcache_cudnn"
    )
    assert native_executor._attention_backend == case.native_attention_backend
    benchmark.group = "omnidreams-dit-network"

    # Repeated scheduler evaluations overwrite one production steady-state slot.
    cache.start(benchmark_chunk_idx)

    def synchronized_forward() -> torch.Tensor:
        result = forward()
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.finalize(benchmark_chunk_idx)

    expected_output_shape = (
        _BATCH_SIZE,
        _NUM_VIEWS,
        chunk_tokens,
        network_config.out_channels * patch_volume,
    )
    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()
