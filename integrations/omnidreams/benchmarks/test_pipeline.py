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
        integrations/omnidreams/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH
from omnidreams.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.vae_native import OmnidreamsWanVAEEncoderConfig
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from integrations.omnidreams.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "OmniDreams full-pipeline benchmark requires CUDA"

_BATCH_SIZE = 1
_NUM_VIEWS = 1
_PIXEL_HEIGHT = DEFAULT_VIDEO_HEIGHT
_PIXEL_WIDTH = DEFAULT_VIDEO_WIDTH
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
        stage="generate",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda case: case.pytest_id)
def test_full_pipeline_finalize_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark pipeline finalization for one DiT implementation."""
    _run_full_pipeline_benchmark(
        benchmark,
        case=case,
        stage="finalize",
    )


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
    stage: Literal["generate", "finalize"],
) -> None:
    """Run one DiT backend and pipeline-stage benchmark variant."""
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
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
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
                    "sdpa_backend": case.sdpa_backend,
                    "cross_attn_sdpa_backend": case.sdpa_backend,
                    "self_attn_qkv_fusion_option": (case.self_attn_qkv_fusion_option),
                    "cross_attn_qkv_fusion_option": (case.cross_attn_qkv_fusion_option),
                    "use_fp8": case.use_fp8,
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
    assert pipeline.decoder is not None

    parameter_count = sum(parameter.numel() for parameter in pipeline.parameters())

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
    assert network_config.sdpa_backend is case.sdpa_backend
    assert network_config.cross_attn_sdpa_backend is case.sdpa_backend
    assert (
        network_config.self_attn_qkv_fusion_option is case.self_attn_qkv_fusion_option
    )
    assert (
        network_config.cross_attn_qkv_fusion_option is case.cross_attn_qkv_fusion_option
    )
    assert network_config.use_fp8 is case.use_fp8

    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, CosmosTransformer)
    assert transformer.config is transformer_config
    assert transformer_config.native_dit_acceleration == native_acceleration
    assert transformer_config.native_dit_backend == native_backend
    assert transformer_config.native_dit_attention_backend == native_attention
    assert transformer_config.skip_finalize_kv_cache is False
    dtype = transformer_config.dtype
    spatial_compression = int(pipeline.decoder.spatial_compression_ratio)
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

    first_chunk_frames = pipeline.get_num_frames(0)
    steady_chunk_frames = pipeline.get_num_frames(1)
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
            hdmap=hdmap,
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
        dit_use_fp8 = native_executor._uses_fp8_dit
        assert dit_use_fp8 is (case.native_dit_backend == "fp8_kvcache_cudnn")
        assert native_executor._attention_backend == case.native_attention_backend
        first_block_cache = cache.transformer_cache.network_cache.block_caches[0]
        if dit_use_fp8:
            fp8_runtime = native_executor._fp8_runtime
            assert fp8_runtime is not None
            for cache_name in ("k_self_fp8_caches", "v_self_fp8_caches"):
                fp8_caches = fp8_runtime[cache_name]
                assert fp8_caches
                assert all(
                    cache_tensor.dtype == torch.uint8 for cache_tensor in fp8_caches
                )
            dit_self_kv_cache_dtype = "float8_e4m3fn (uint8 native storage)"
            if case.native_attention_backend == "sage3_fp8":
                for cache_name in (
                    "k_cross_sage3_fp4_caches",
                    "v_cross_sage3_fp4_caches",
                ):
                    fp4_caches = fp8_runtime[cache_name]
                    assert fp4_caches
                    assert all(
                        cache_tensor.dtype == torch.uint8 for cache_tensor in fp4_caches
                    )
                for cache_name in (
                    "k_cross_sage3_sf_caches",
                    "v_cross_sage3_sf_caches",
                ):
                    scale_caches = fp8_runtime[cache_name]
                    assert scale_caches
                    assert all(
                        cache_tensor.dtype == torch.float8_e4m3fn
                        for cache_tensor in scale_caches
                    )
                dit_cross_kv_cache_dtype = "Sage3 FP4 + FP8 scale factors"
            else:
                for cache_name in ("k_cross_fp8_caches", "v_cross_fp8_caches"):
                    fp8_caches = fp8_runtime[cache_name]
                    assert fp8_caches
                    assert all(
                        cache_tensor.dtype == torch.uint8 for cache_tensor in fp8_caches
                    )
                dit_cross_kv_cache_dtype = "float8_e4m3fn (uint8 native storage)"
            if case.native_attention_backend == "sparge":
                dit_self_kv_cache_dtype += " + bfloat16 Sparge storage"
                dit_cross_kv_cache_dtype += " + bfloat16 Sparge storage"
            native_setup = "FP8 conversion, "
        else:
            assert native_executor._bf16_runtime is not None
            dit_self_kv_cache_dtype = str(first_block_cache.self_attn.dtype)
            dit_cross_kv_cache_dtype = str(first_block_cache.cross_attn.dtype)
            native_setup = ""
        dit_execution = "native_cuda"
        dit_self_attn_qkv_fusion_option = "native_cuda"
        dit_cross_attn_qkv_fusion_option = "native_cuda"
        dit_attention_backend = native_executor._attention_backend
        dit_self_attention_backend = native_executor._attention_backend
        dit_cross_attention_backend = native_executor._attention_backend
        dit_sdpa_backend = native_executor._attention_backend
        dit_kv_cache_dtype = (
            f"self={dit_self_kv_cache_dtype}, cross={dit_cross_kv_cache_dtype}"
        )
        native_extension = native_selection.reason
        compiler_cache_state = (
            f"host-dependent; native extension build, {native_setup}CUDA graph "
            "capture, and autotune excluded by prefill"
        )
    else:
        assert native_selection is None
        assert native_executor is None
        dit_execution = "pytorch"
        dit_use_fp8 = case.use_fp8
        dit_self_attn_qkv_fusion_option = case.self_attn_qkv_fusion_option.value
        dit_cross_attn_qkv_fusion_option = case.cross_attn_qkv_fusion_option.value
        dit_attention_backend = case.self_attention_operator
        dit_self_attention_backend = case.self_attention_operator
        dit_cross_attention_backend = case.cross_attention_operator
        dit_sdpa_backend = case.sdpa_backend.value
        first_block_cache = cache.transformer_cache.network_cache.block_caches[0]
        dit_self_kv_cache_dtype = str(first_block_cache.self_attn.dtype)
        dit_cross_kv_cache_dtype = str(first_block_cache.cross_attn.dtype)
        dit_kv_cache_dtype = (
            f"self={dit_self_kv_cache_dtype}, cross={dit_cross_kv_cache_dtype}"
        )
        native_extension = None
        compiler_cache_state = (
            "host-dependent; compile, CUDA graph capture, and autotune excluded "
            "by prefill"
        )

    denoising_timesteps = scheduler_config.denoising_timesteps
    timed_stages = (
        ["hdmap_encode", "diffuse", "decode"] if stage == "generate" else ["finalize"]
    )
    untimed_lifecycle_stage = "finalize" if stage == "generate" else "generate"
    benchmark.group = f"omnidreams-full-pipeline-{stage}"
    benchmark.extra_info.update(
        {
            "pipeline": pipeline_config.name,
            "source_pipeline": (SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF.name),
            "batch_size": _BATCH_SIZE,
            "num_views": _NUM_VIEWS,
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [
                transformer_config.len_t,
                latent_channels,
                latent_height,
                latent_width,
            ],
            "frames_per_chunk": steady_chunk_frames,
            "text_tokens": _TEXT_TOKENS,
            "text_embedding_dim": text_dim,
            "num_inference_steps": scheduler_config.num_inference_steps,
            "denoising_timesteps": (
                list(denoising_timesteps) if denoising_timesteps is not None else None
            ),
            "context_noise": diffusion_config.context_noise,
            "window_size_t": transformer_config.window_size_t,
            "cache_prefill_chunks": cache_prefill_chunks,
            "timed_stage": stage,
            "timed_stages": timed_stages,
            "untimed_lifecycle_stage": untimed_lifecycle_stage,
            "one_shot_inputs": "synthetic_precomputed_embeddings",
            "dit_checkpoint": transformer_config.checkpoint_path,
            "hdmap_encoder_checkpoint": encoder_config.checkpoint_path,
            "decoder_checkpoint": decoder_config.checkpoint_path,
            "dtype": str(dtype),
            "implementation": case.implementation,
            "dit_execution": dit_execution,
            "dit_sdpa_backend": dit_sdpa_backend,
            "dit_use_fp8": dit_use_fp8,
            "dit_self_attn_qkv_fusion_option": dit_self_attn_qkv_fusion_option,
            "dit_cross_attn_qkv_fusion_option": dit_cross_attn_qkv_fusion_option,
            "dit_attention_backend": dit_attention_backend,
            "dit_self_attention_backend": dit_self_attention_backend,
            "dit_cross_attention_backend": dit_cross_attention_backend,
            "dit_kv_cache_dtype": dit_kv_cache_dtype,
            "dit_self_attention_kv_cache_dtype": dit_self_kv_cache_dtype,
            "dit_cross_attention_kv_cache_dtype": dit_cross_kv_cache_dtype,
            "native_dit_acceleration": transformer_config.native_dit_acceleration,
            "native_dit_backend": transformer_config.native_dit_backend,
            "native_dit_attention_backend": (
                transformer_config.native_dit_attention_backend
            ),
            "native_extension": native_extension,
            "dit_compiled": transformer_config.compile_network,
            "dit_cuda_graph": transformer_config.use_cuda_graph,
            "skip_finalize_kv_cache": transformer_config.skip_finalize_kv_cache,
            "hdmap_encoder_compiled": encoder_config.use_compile,
            "hdmap_encoder_cuda_graph": encoder_config.use_cuda_graph,
            "decoder_compiled": decoder_config.use_compile,
            "decoder_cuda_graph": decoder_config.use_cuda_graph,
            "parameter_count": parameter_count,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "startup_timing": "excluded",
            "first_visible_timing": "excluded",
            "compiler_cache_state": compiler_cache_state,
            "num_gpus": torch.cuda.device_count(),
            "seed": _SEED,
        }
    )

    next_chunk_index = cache_prefill_chunks
    latest_output: torch.Tensor | None = None
    stage_peak_cuda_memory_bytes = 0

    def record_stage_peak_memory() -> None:
        nonlocal stage_peak_cuda_memory_bytes
        stage_peak_cuda_memory_bytes = max(
            stage_peak_cuda_memory_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )

    if stage == "generate":

        def setup_generate() -> None:
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_generate() -> torch.Tensor:
            nonlocal latest_output
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                hdmap=hdmap_steady,
            )
            torch.cuda.synchronize()
            return latest_output

        def teardown_generate() -> None:
            nonlocal next_chunk_index
            record_stage_peak_memory()
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize()
            next_chunk_index += 1

        output = benchmark.pedantic(
            synchronized_generate,
            setup=setup_generate,
            teardown=teardown_generate,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
    else:

        def setup_finalize() -> None:
            nonlocal latest_output
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                hdmap=hdmap_steady,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_finalize() -> None:
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize()

        def teardown_finalize() -> None:
            nonlocal next_chunk_index
            record_stage_peak_memory()
            next_chunk_index += 1

        benchmark.pedantic(
            synchronized_finalize,
            setup=setup_finalize,
            teardown=teardown_finalize,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
        output = latest_output

    benchmark.extra_info["peak_cuda_memory_bytes"] = stage_peak_cuda_memory_bytes
    assert benchmark.stats is not None
    sample_times_s = benchmark.stats.stats.sorted_data
    p90_index = math.ceil(0.9 * len(sample_times_s)) - 1
    median_stage_s = benchmark.stats.stats.median
    p90_stage_s = sample_times_s[p90_index]
    benchmark.extra_info.update(
        {
            f"median_{stage}_ms": median_stage_s * 1_000,
            f"p90_{stage}_ms": p90_stage_s * 1_000,
            "median_chunks_per_second": 1.0 / median_stage_s,
            "p90_chunks_per_second": 1.0 / p90_stage_s,
        }
    )
    if stage == "generate":
        benchmark.extra_info.update(
            {
                "median_output_fps": steady_chunk_frames / median_stage_s,
                "p90_output_fps": steady_chunk_frames / p90_stage_s,
            }
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
