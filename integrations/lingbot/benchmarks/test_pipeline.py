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

"""Steady-state full-pipeline benchmarks for LingBot streaming inference.

Run the manual GPU benchmarks with::

    uv run --package flashdreams-lingbot --group test pytest \
        integrations/lingbot/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import math
import os
from typing import Literal

import pytest
import torch
import torch.distributed as dist
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.encoder.camctrl import CamCtrlInput, I2VCamCtrlEncoderConfig
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)
from lingbot.transformer.impl.network import LingbotWorldDiTNetwork
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchScheduler,
    FlowMatchSchedulerConfig,
)
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoderConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipelineCache
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from integrations.lingbot.benchmarks.cases import (
    ATTENTION_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot full-pipeline benchmark requires CUDA"

_PIXEL_HEIGHT = 352
_PIXEL_WIDTH = 640
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
) -> None:
    """Skip a case where its hardware or context-parallel contract is unmet."""
    skip_unsupported_device(case, device)
    if case.attention_backend is not AttentionBackend.TRITON:
        return
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size > 1:
        pytest.skip("Triton attention does not support context parallelism")


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    ATTENTION_CASES,
    ids=lambda case: case.pytest_id,
)
def test_full_pipeline_generate_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark steady-state LingBot encode, diffuse, and decode."""
    _run_full_pipeline_benchmark(benchmark, case=case, stage="generate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    ATTENTION_CASES,
    ids=lambda case: case.pytest_id,
)
def test_full_pipeline_finalize_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the LingBot DiT cache-finalization update."""
    _run_full_pipeline_benchmark(benchmark, case=case, stage="finalize")


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
    stage: Literal["generate", "finalize"],
) -> None:
    """Run one full-pipeline lifecycle-stage benchmark."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot full-pipeline benchmark requires bfloat16 support")
    _skip_unsupported_case(case, device)

    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True

    # UMT5 is a one-shot rollout initializer, so use a correctly shaped
    # precomputed embedding. The recurring pipeline remains production-like:
    # LingBot camera rendering/control, four-step DiT diffusion, and TAEHV
    # decoding. Initializing the 14B DiT directly on this rank's GPU avoids a
    # transient fp32 CPU copy and does not affect timed steady-state stages.
    pipeline_config = derive_config(
        PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
        name=f"lingbot-world-v2-full-pipeline-{case.implementation}-benchmark",
        text_encoder=None,
        enable_sync_and_profile=False,
        diffusion_model={
            "seed": _SEED,
            "transformer": {
                "init_device": str(device),
                "network": {
                    "attention_backend": case.attention_backend,
                    "sdpa_backend": case.sdpa_backend,
                },
            },
        },
    )
    pipeline = pipeline_config.setup().to(device=device)
    assert isinstance(pipeline, LingbotWorldInferencePipeline)
    pipeline.eval()
    assert pipeline.encoder is not None
    assert pipeline.decoder is not None

    recurring_parameter_count = sum(
        parameter.numel() for parameter in pipeline.parameters()
    )
    context_parallel_attention_modules = [
        module
        for module in pipeline.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    context_parallel_attention_backends = {
        attention.backend for attention in context_parallel_attention_modules
    }
    assert context_parallel_attention_backends == (
        {"cudnn"} if case.attention_backend is AttentionBackend.WAN else set()
    )

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    encoder_config = pipeline_config.encoder
    decoder_config = pipeline_config.decoder
    assert isinstance(transformer_config, LingbotWorldTransformerConfig)
    assert isinstance(scheduler_config, FlowMatchSchedulerConfig)
    assert isinstance(encoder_config, I2VCamCtrlEncoderConfig)
    assert isinstance(encoder_config.i2v.encoder, WanVAEEncoderConfig)
    assert isinstance(decoder_config, TeahvVAEDecoderConfig)
    assert transformer_config.network.attention_backend is case.attention_backend
    assert transformer_config.network.sdpa_backend is case.sdpa_backend

    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, LingbotWorldTransformer)
    assert transformer.config is transformer_config
    network = getattr(transformer.network, "_orig_mod", transformer.network)
    assert isinstance(network, LingbotWorldDiTNetwork)
    assert network.blocks
    assert all(
        block.attention_backend is case.attention_backend
        and block.sdpa_backend is case.sdpa_backend
        for block in network.blocks
    )
    cp_size = transformer._cp_size
    cp_enabled_attention_modules = [
        attention
        for attention in context_parallel_attention_modules
        if attention.is_context_parallel_enabled()
    ]
    local_attention_methods = {
        attention.method
        for attention in context_parallel_attention_modules
        if not attention.is_context_parallel_enabled()
    }
    assert all(
        attention.context_parallel_size() == cp_size
        for attention in cp_enabled_attention_modules
    )
    assert all(
        attention.method == transformer_config.network.cp_method
        for attention in cp_enabled_attention_modules
    )
    assert bool(cp_enabled_attention_modules) == (cp_size > 1)
    dtype = transformer_config.dtype
    spatial_compression = int(pipeline.decoder.spatial_compression_ratio)
    latent_height = _PIXEL_HEIGHT // spatial_compression
    latent_width = _PIXEL_WIDTH // spatial_compression
    latent_channels = int(transformer_config.network.out_dim)
    text_dim = int(transformer_config.network.text_dim)

    text_embeddings = torch.zeros(
        (1, _TEXT_TOKENS, text_dim),
        device=device,
        dtype=dtype,
    )
    image = torch.zeros(
        (1, 3, _PIXEL_HEIGHT, _PIXEL_WIDTH),
        device=device,
        dtype=dtype,
    )

    # Bypass only Wan's raw-text one-shot initializer. The base cache builder
    # still creates the real recurring encoder, transformer, and decoder
    # caches, and the Wan cache wrapper retains the first frame for I2V.
    parent_cache = StreamInferencePipeline.initialize_cache(
        pipeline,
        transformer_context={
            "height": latent_height,
            "width": latent_width,
            "text_embeddings": text_embeddings,
            "negative_text_embeddings": None,
            "image_embeddings": None,
        },
    )
    cache = WanInferencePipelineCache(
        transformer_cache=parent_cache.transformer_cache,
        encoder_cache=parent_cache.encoder_cache,
        decoder_cache=parent_cache.decoder_cache,
        image=image,
    )
    del text_embeddings

    first_block_cache = cache.transformer_cache.network_cache.block_caches[0]
    self_attention_cache_dtype = str(first_block_cache.self_attn.dtype)
    cross_attention_cache_dtype = str(first_block_cache.cross_attn.text.dtype)

    first_chunk_frames = pipeline.get_num_input_frames(0)
    steady_input_frames = pipeline.get_num_input_frames(1)
    steady_output_frames = pipeline.get_num_output_frames(1)

    def camera_input(num_frames: int) -> CamCtrlInput:
        intrinsics = torch.tensor(
            [416.0, 416.0, _PIXEL_WIDTH / 2, _PIXEL_HEIGHT / 2],
            device=device,
            dtype=torch.float32,
        ).repeat(num_frames, 1)
        poses = torch.eye(4, device=device, dtype=torch.float32).repeat(
            num_frames, 1, 1
        )
        return CamCtrlInput(
            intrinsics=intrinsics,
            poses=poses,
            world_scale=1.0,
        )

    first_camera_input = camera_input(first_chunk_frames)
    steady_camera_input = camera_input(steady_input_frames)

    def run_chunk(
        autoregressive_index: int,
        input: CamCtrlInput,
    ) -> torch.Tensor:
        output = pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )
        pipeline.finalize(
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        return output

    # Fill the sink/window cache through the first CUDA-graph index. This also
    # advances the reused Wan I2V encoder past its first five real VAE calls;
    # steady-state timed rounds reuse its cached latent and measure only the
    # LingBot camera-control work on that branch.
    capture_ar_index = transformer._cuda_graph_capture_ar_idx
    cache_prefill_chunks = capture_ar_index + 1
    for autoregressive_index in range(cache_prefill_chunks):
        chunk_input = (
            first_camera_input if autoregressive_index == 0 else steady_camera_input
        )
        output = run_chunk(autoregressive_index, chunk_input)
    torch.cuda.synchronize(device)

    scheduler = pipeline.diffusion_model.scheduler
    assert isinstance(scheduler, FlowMatchScheduler)
    resolved_denoising_timesteps = (
        scheduler.denoising_step_list.detach().to(torch.float32).cpu().tolist()
    )
    effective_dit_timesteps = (
        scheduler.denoising_step_list.detach()
        .to(dtype=dtype)
        .to(torch.float32)
        .cpu()
        .tolist()
    )
    denoising_sigmas = (
        scheduler.denoising_sigmas.detach().to(torch.float32).cpu().tolist()
    )
    timed_stages = (
        ["cached_i2v_and_plucker_encode", "diffuse", "taehv_decode"]
        if stage == "generate"
        else ["dit_cache_finalize"]
    )
    untimed_lifecycle_stage = "finalize" if stage == "generate" else "generate"
    measurement_start_ar_index = cache_prefill_chunks + _WARMUP_ROUNDS
    measurement_end_ar_index = measurement_start_ar_index + _BENCHMARK_ROUNDS - 1
    benchmark.group = f"lingbot-full-pipeline-{stage}"
    benchmark.extra_info.update(
        {
            "pipeline": pipeline_config.name,
            "source_pipeline": (
                PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3.name
            ),
            "batch_shape": list(transformer_config.batch_shape),
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [
                transformer_config.len_t,
                latent_channels,
                latent_height,
                latent_width,
            ],
            "global_chunk_tokens": (
                transformer_config.len_t
                * (latent_height // transformer_config.network.patch_size[1])
                * (latent_width // transformer_config.network.patch_size[2])
            ),
            "local_chunk_tokens": transformer.latent_shape[-2],
            "input_frames_per_chunk": steady_input_frames,
            "output_frames_per_chunk": steady_output_frames,
            "text_tokens": _TEXT_TOKENS,
            "text_embedding_dim": text_dim,
            "num_inference_steps": scheduler_config.num_inference_steps,
            "configured_denoising_timesteps": list(
                scheduler_config.denoising_timesteps
            ),
            "resolved_denoising_timesteps_fp32": resolved_denoising_timesteps,
            "effective_dit_timesteps": effective_dit_timesteps,
            "denoising_sigmas_fp32": denoising_sigmas,
            "context_noise": diffusion_config.context_noise,
            "window_size_t": transformer_config.window_size_t,
            "sink_size_t": transformer_config.sink_size_t,
            "cache_prefill_chunks": cache_prefill_chunks,
            "fixture_warmup_start_ar_index": cache_prefill_chunks,
            "measurement_start_ar_index": measurement_start_ar_index,
            "measurement_end_ar_index": measurement_end_ar_index,
            "timed_stage": stage,
            "timed_stages": timed_stages,
            "untimed_lifecycle_stage": untimed_lifecycle_stage,
            "one_shot_text_input": "synthetic_precomputed_embedding",
            "first_frame_image": "zeros",
            "camera_control_schedule": "repeated_static_camera",
            "camera_intrinsics": [
                416.0,
                416.0,
                _PIXEL_WIDTH / 2,
                _PIXEL_HEIGHT / 2,
            ],
            "camera_poses": "identity_4x4_repeated_per_frame",
            "camera_world_scale": 1.0,
            "reused_wan_i2v_vae_timed": False,
            "reused_taehv_decoder_timed": stage == "generate",
            "dit_checkpoint": transformer_config.checkpoint_path,
            "i2v_encoder_checkpoint": encoder_config.i2v.encoder.checkpoint_path,
            "decoder_checkpoint": decoder_config.checkpoint_path,
            "dtype": str(dtype),
            "implementation": case.implementation,
            "dit_execution": "pytorch",
            "configured_attention_backend": case.attention_backend.value,
            "configured_sdpa_backend": case.sdpa_backend.value,
            "dit_attention_backend": case.self_attention_operator,
            "dit_self_attention_backend": case.self_attention_operator,
            "dit_cross_attention_backend": case.cross_attention_operator,
            "projection_backend": (
                "separate_qkv"
                if case.attention_backend is AttentionBackend.WAN
                else "row_scaled_fp8_fused_qkv_output"
            ),
            "dit_self_attention_kv_cache_dtype": self_attention_cache_dtype,
            "dit_cross_attention_kv_cache_dtype": cross_attention_cache_dtype,
            "self_attention_context_parallel_method": (
                transformer_config.network.cp_method
                if case.attention_backend is AttentionBackend.WAN
                else None
            ),
            "local_attention_methods": sorted(local_attention_methods),
            "context_parallel_size": cp_size,
            "context_parallel_attention_modules": len(cp_enabled_attention_modules),
            "local_attention_modules": (
                len(context_parallel_attention_modules)
                - len(cp_enabled_attention_modules)
            ),
            "distributed_sample_alignment": "barrier_before_each_round",
            "dit_compiled": transformer_config.compile_network,
            "dit_cuda_graph": transformer_config.use_cuda_graph,
            "i2v_encoder_compiled": encoder_config.i2v.encoder.use_compile,
            "i2v_encoder_cuda_graph": encoder_config.i2v.encoder.use_cuda_graph,
            "decoder_compiled": decoder_config.use_compile,
            "decoder_cuda_graph": decoder_config.use_cuda_graph,
            "recurring_pipeline_parameter_count": recurring_parameter_count,
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "startup_timing": "excluded",
            "first_visible_timing": "excluded",
            "compiler_cache_state": (
                "host-dependent; checkpoint loading, compile, CUDA graph "
                "capture, and autotune excluded by prefill"
            ),
            "num_gpus_visible": torch.cuda.device_count(),
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
            _synchronize_ranks()
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_generate() -> torch.Tensor:
            nonlocal latest_output
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                input=steady_camera_input,
            )
            torch.cuda.synchronize(device)
            return latest_output

        def teardown_generate() -> None:
            nonlocal next_chunk_index
            record_stage_peak_memory()
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize(device)
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
            _synchronize_ranks()
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                input=steady_camera_input,
            )
            torch.cuda.synchronize(device)
            _synchronize_ranks()
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_finalize() -> None:
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize(device)

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
            f"median_{stage}_chunks_per_second": 1.0 / median_stage_s,
            f"{stage}_chunks_per_second_at_p90_latency": 1.0 / p90_stage_s,
        }
    )
    if stage == "generate":
        benchmark.extra_info.update(
            {
                "median_generate_only_output_fps": (
                    steady_output_frames / median_stage_s
                ),
                "generate_only_output_fps_at_p90_latency": (
                    steady_output_frames / p90_stage_s
                ),
            }
        )

    assert output is not None
    assert output.shape == (
        steady_output_frames,
        3,
        _PIXEL_HEIGHT,
        _PIXEL_WIDTH,
    )
    assert torch.isfinite(output).all()
