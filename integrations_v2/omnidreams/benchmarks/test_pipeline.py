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

"""Steady-state full-pipeline benchmark for OmniDreams streaming inference.

Run the benchmark with::

    uv run --group test pytest \
        integrations_v2/omnidreams/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from omnidreams.apps.interactive_drive.adapter import (
    OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
)
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
from omnidreams.impl.pipeline import OmnidreamsPipeline
from omnidreams.impl.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.impl.vae_native import OmnidreamsWanVAEEncoderConfig

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.taehv import TeahvVAEDecoder, TeahvVAEDecoderConfig
from integrations_v2.omnidreams.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "OmniDreams full-pipeline benchmark requires CUDA"

_BATCH_SIZE = 1
_NUM_VIEWS = 1
_PIXEL_HEIGHT = OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.height
_PIXEL_WIDTH = OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.width
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda case: case.pytest_id)
def test_full_pipeline_generate_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark pipeline generation for one DiT implementation."""
    _run_full_pipeline_benchmark(
        benchmark,
        case=case,
    )


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
) -> None:
    """Run one DiT backend full-pipeline benchmark variant."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("OmniDreams full-pipeline benchmark requires bfloat16 support")

    device = torch.device("cuda")
    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True
    native_dit = case.native_dit
    if (
        native_dit
        and case.native_dit_backend == "fp8_kvcache_cudnn"
        and not hasattr(torch, "float8_e4m3fn")
    ):
        pytest.skip("OmniDreams native DiT benchmark requires float8_e4m3fn")
    self_attention_backend = case.self_attention_backend
    skip_unsupported_device(case, device)

    # One-shot prompt and first-frame encoders run before streaming begins in
    # production. Replace them with correctly shaped precomputed embeddings so
    # the timed path covers the recurring HDMap encoder, diffusion, decoder,
    # and cache-bookkeeping stages.
    native_acceleration = "required" if native_dit else "disabled"
    native_backend = case.native_dit_backend if native_dit else "bf16"
    native_attention = case.native_attention_backend if native_dit else "auto"
    pipeline_config = derive_config(
        OMNIDREAMS_PIPELINE_CONFIG,
        name=f"omnidreams-full-pipeline-{case.pytest_id}-benchmark",
        text_encoder=None,
        image_encoder=None,
        synthetic_text_max_length=_TEXT_TOKENS,
        enable_sync_and_profile=False,
        diffusion_model={
            "seed": _SEED,
            "transformer": {
                "compile_network": True,
                "network": {
                    "self_attention_backend": self_attention_backend,
                    "cross_attention_backend": case.cross_attention_backend,
                    "self_attn_optimized_impl_config": case.self_attn_optimized_impl_config,
                    "cross_attn_optimized_impl_config": case.cross_attn_optimized_impl_config,
                },
                # Keep cache finalization identical across the comparison;
                # this performs the final context-noise DiT update before
                # committing each autoregressive cache position.
                "skip_finalize_kv_cache": False,
                "native_dit_acceleration": native_acceleration,
                "native_dit_backend": native_backend,
                "native_dit_attention_backend": native_attention,
            },
        },
    )
    pipeline = pipeline_config.setup().to(device=device)
    assert isinstance(pipeline, OmnidreamsPipeline)
    pipeline.eval()
    assert pipeline.encoder is not None
    decoder = pipeline.decoder
    assert isinstance(decoder, TeahvVAEDecoder)

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    encoder_config = pipeline_config.encoder
    decoder_config = pipeline_config.decoder
    assert isinstance(transformer_config, CosmosTransformerConfig)
    assert isinstance(scheduler_config, FlowMatchSchedulerConfig)
    assert isinstance(encoder_config, OmnidreamsWanVAEEncoderConfig)
    assert isinstance(decoder_config, TeahvVAEDecoderConfig)
    network_config = transformer_config.network
    assert network_config.self_attention_backend is self_attention_backend
    assert network_config.cross_attention_backend is case.cross_attention_backend
    assert (
        network_config.self_attn_optimized_impl_config
        is case.self_attn_optimized_impl_config
    )
    assert (
        network_config.cross_attn_optimized_impl_config
        is case.cross_attn_optimized_impl_config
    )

    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, CosmosTransformer)
    assert transformer.config is transformer_config
    assert transformer_config.native_dit_acceleration == native_acceleration
    assert transformer_config.native_dit_backend == native_backend
    assert transformer_config.native_dit_attention_backend == native_attention
    assert transformer_config.skip_finalize_kv_cache is False
    dtype = transformer_config.dtype
    spatial_compression = int(decoder.spatial_compression_ratio)
    latent_height = _PIXEL_HEIGHT // spatial_compression
    latent_width = _PIXEL_WIDTH // spatial_compression
    latent_channels = int(network_config.in_channels)
    text_dim = (
        int(network_config.crossattn_proj_in_channels)
        if network_config.use_crossattn_projection
        else int(network_config.crossattn_emb_channels)
    )

    text_embeddings = torch.zeros(
        (_BATCH_SIZE, _NUM_VIEWS, _TEXT_TOKENS, text_dim),
        device=device,
        dtype=dtype,
    )
    image_embeddings = torch.zeros(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            1,
            latent_channels,
            latent_height,
            latent_width,
        ),
        device=device,
        dtype=dtype,
    )
    cache = pipeline.initialize_cache_from_embeddings(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
    )
    del text_embeddings, image_embeddings

    first_chunk_frames = pipeline.get_num_output_frames(0)
    steady_chunk_frames = pipeline.get_num_output_frames(1)
    input_generator = torch.Generator(device=device).manual_seed(_SEED)
    hdmap_first = (
        torch.rand(
            (
                _BATCH_SIZE,
                _NUM_VIEWS,
                first_chunk_frames,
                3,
                _PIXEL_HEIGHT,
                _PIXEL_WIDTH,
            ),
            generator=input_generator,
            device=device,
            dtype=dtype,
        )
        .mul_(2)
        .sub_(1)
    )
    hdmap_steady = (
        torch.rand(
            (
                _BATCH_SIZE,
                _NUM_VIEWS,
                steady_chunk_frames,
                3,
                _PIXEL_HEIGHT,
                _PIXEL_WIDTH,
            ),
            generator=input_generator,
            device=device,
            dtype=dtype,
        )
        .mul_(2)
        .sub_(1)
    )

    def run_chunk(autoregressive_index: int, hdmap: torch.Tensor) -> torch.Tensor:
        output = pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=hdmap,
        )
        pipeline.finalize(autoregressive_index=autoregressive_index, cache=cache)
        return output

    # Fill the local attention window and execute the first steady-state index.
    # This excludes torch.compile, CUDA-graph capture, kernel autotuning, and
    # cache growth from both pytest-benchmark's warmups and measured rounds.
    capture_ar_index = (
        transformer_config.sink_size_t + transformer_config.window_size_t
    ) // transformer_config.len_t
    cache_prefill_chunks = capture_ar_index + 1
    for autoregressive_index in range(cache_prefill_chunks):
        hdmap = hdmap_first if autoregressive_index == 0 else hdmap_steady
        run_chunk(autoregressive_index, hdmap)
    torch.cuda.synchronize()

    native_selection = transformer._optimized_dit_selection
    native_executor = transformer._optimized_dit_executor
    if native_dit:
        assert native_selection is not None and native_selection.enabled
        assert native_executor is not None
        assert native_executor._uses_fp8_dit is (
            case.native_dit_backend == "fp8_kvcache_cudnn"
        )
        assert native_executor._attention_backend == case.native_attention_backend
    else:
        assert native_selection is None
        assert native_executor is None

    benchmark.group = "omnidreams-full-pipeline-generate"

    next_chunk_index = cache_prefill_chunks
    latest_output: torch.Tensor | None = None

    def synchronized_generate() -> torch.Tensor:
        nonlocal latest_output
        latest_output = pipeline.generate(
            autoregressive_index=next_chunk_index,
            cache=cache,
            input=hdmap_steady,
        )
        torch.cuda.synchronize()
        return latest_output

    def teardown_generate() -> None:
        nonlocal next_chunk_index
        pipeline.finalize(
            autoregressive_index=next_chunk_index,
            cache=cache,
        )
        torch.cuda.synchronize()
        next_chunk_index += 1

    output = benchmark.pedantic(
        synchronized_generate,
        teardown=teardown_generate,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output is not None
    assert output.shape == (
        _BATCH_SIZE,
        _NUM_VIEWS,
        steady_chunk_frames,
        3,
        _PIXEL_HEIGHT,
        _PIXEL_WIDTH,
    )
    assert torch.isfinite(output).all()
