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

"""One-shot full-pipeline benchmarks for the shipped Wan 2.1 T2V runner.

Run the manual GPU benchmarks with::

    uv run --package flashdreams-wan21 --group test pytest \
        integrations/wan21/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
import torch
import torch.distributed as dist
from pytest_benchmark.fixture import BenchmarkFixture
from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P
from wan21.runner import DEFAULT_PROMPT

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModel
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCScheduler,
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.wan import (
    NEGATIVE_PROMPT,
    Wan21Transformer,
    Wan21TransformerConfig,
    WanDiTNetwork,
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanVAEDecoderConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerCache
from integrations.wan21.benchmarks.cases import (
    ATTENTION_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Wan 2.1 full-pipeline benchmark requires CUDA"

_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_TEXT_TOKENS = 512
_COMPONENT_PREWARM_ROUNDS = 3
_WARMUP_ROUNDS = 0
_BENCHMARK_ROUNDS = 1
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
    """Align context-parallel ranks outside a benchmark sample."""
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
    """Benchmark the shipped one-shot Wan 2.1 denoise and decode path."""
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
    """Benchmark the matching AR=0 DiT cache-finalization update."""
    _run_full_pipeline_benchmark(benchmark, case=case, stage="finalize")


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
    stage: Literal["generate", "finalize"],
) -> None:
    """Run one backend and one production pipeline lifecycle stage."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan 2.1 full-pipeline benchmark requires bfloat16 support")
    _skip_unsupported_case(case, device)

    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True

    # UMT5 is a one-shot rollout initializer. Use correctly shaped synthetic
    # embeddings so setup measures the checkpoint-backed recurring pipeline:
    # 50-step CFG diffusion followed by the production Wan VAE decoder.
    pipeline_config = derive_config(
        PIPELINE_WAN21_T2V_1PT3B_480P,
        name=f"wan21-t2v-1.3b-full-pipeline-{case.implementation}-benchmark",
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
    assert isinstance(pipeline, WanInferencePipeline)
    pipeline.eval()
    assert pipeline.encoder is None
    assert pipeline.decoder is not None

    recurring_parameter_count = sum(
        parameter.numel() for parameter in pipeline.parameters()
    )
    source_text_encoder_config = PIPELINE_WAN21_T2V_1PT3B_480P.text_encoder
    assert isinstance(source_text_encoder_config, UMT5TextEncoderConfig)

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    decoder_config = pipeline_config.decoder
    assert isinstance(transformer_config, Wan21TransformerConfig)
    assert isinstance(scheduler_config, FlowMatchUniPCSchedulerConfig)
    assert isinstance(decoder_config, WanVAEDecoderConfig)
    assert transformer_config.network.attention_backend is case.attention_backend
    assert transformer_config.network.sdpa_backend is case.sdpa_backend
    assert transformer_config.batch_shape == ()
    assert transformer_config.len_t == 21
    assert transformer_config.window_size_t == 21
    assert transformer_config.sink_size_t == 0
    assert transformer_config.guidance_scale == 6.0
    assert scheduler_config.num_inference_steps == 50

    transformer = pipeline.diffusion_model.transformer
    scheduler = pipeline.diffusion_model.scheduler
    assert isinstance(transformer, Wan21Transformer)
    assert isinstance(scheduler, FlowMatchUniPCScheduler)
    network = getattr(transformer.network, "_orig_mod", transformer.network)
    assert isinstance(network, WanDiTNetwork)
    assert network.blocks
    assert all(
        block.attention_backend is case.attention_backend
        and block.sdpa_backend is case.sdpa_backend
        for block in network.blocks
    )

    context_parallel_attention_modules = [
        module
        for module in pipeline.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    assert {attention.backend for attention in context_parallel_attention_modules} == (
        {"cudnn"} if case.attention_backend is AttentionBackend.WAN else set()
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
    output_frames = pipeline.decoder.get_output_temporal_size(
        0, transformer_config.len_t
    )
    patch_t, patch_h, patch_w = transformer_config.network.patch_size
    global_tokens = (
        (transformer_config.len_t // patch_t)
        * (latent_height // patch_h)
        * (latent_width // patch_w)
    )

    text_embeddings = torch.zeros(
        (1, _TEXT_TOKENS, text_dim),
        device=device,
        dtype=dtype,
    )
    negative_text_embeddings = torch.zeros_like(text_embeddings)

    def initialize_cache() -> WanInferencePipelineCache:
        """Build one production-shaped T2V cache outside sample timing."""
        parent_cache = StreamInferencePipeline.initialize_cache(
            pipeline,
            transformer_context={
                "height": latent_height,
                "width": latent_width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
                "image_embeddings": None,
            },
        )
        cache = WanInferencePipelineCache(
            transformer_cache=parent_cache.transformer_cache,
            encoder_cache=parent_cache.encoder_cache,
            decoder_cache=parent_cache.decoder_cache,
            image=None,
        )
        assert isinstance(cache.transformer_cache, Wan21TransformerCache)
        assert cache.transformer_cache.network_cache_uncond is not None
        return cache

    # Prime lazy compilation and cuDNN selection without paying for another
    # complete 50-step rollout. Fresh measured caches reset the graph wrappers,
    # so their per-rollout AR0 input staging remains inside the timed call.
    probe_cache = initialize_cache()
    assert isinstance(probe_cache.transformer_cache, Wan21TransformerCache)
    first_block_cache = probe_cache.transformer_cache.network_cache.block_caches[0]
    self_attention_cache_dtype = str(first_block_cache.self_attn.dtype)
    cross_attention_cache_dtype = str(first_block_cache.cross_attn.text.dtype)

    probe_transformer_cache = probe_cache.transformer_cache
    probe_transformer_cache.start(0)
    prewarm_latent = torch.zeros(
        transformer.latent_shape,
        device=device,
        dtype=dtype,
    )
    prewarm_timestep = scheduler.timesteps[0].to(device=device, dtype=dtype)
    prewarm_flow = transformer.predict_flow(
        noisy_latent=prewarm_latent,
        timestep=prewarm_timestep,
        cache=probe_transformer_cache,
    )
    for _ in range(1, _COMPONENT_PREWARM_ROUNDS):
        prewarm_flow = transformer.predict_flow(
            noisy_latent=prewarm_latent,
            timestep=prewarm_timestep,
            cache=probe_transformer_cache,
        )
    probe_transformer_cache.finalize(0)

    decoder_prewarm_calls = 0
    prewarm_decode_input: torch.Tensor | None = None
    prewarm_decode_output: torch.Tensor | None = None
    if stage == "generate":
        assert probe_cache.decoder_cache is not None
        prewarm_decode_input = torch.zeros(
            (
                transformer_config.len_t,
                latent_channels,
                latent_height,
                latent_width,
            ),
            device=device,
            dtype=dtype,
        )
        prewarm_decode_output = pipeline.decoder(
            input=prewarm_decode_input,
            autoregressive_index=0,
            cache=probe_cache.decoder_cache,
        )
        decoder_prewarm_calls = 1

    torch.cuda.synchronize(device)
    del prewarm_flow, prewarm_latent, prewarm_timestep
    prewarm_decode_input = None
    del prewarm_decode_output
    probe_cache = None

    resolved_timesteps = scheduler.timesteps.detach().cpu().tolist()
    resolved_sigmas = scheduler.sigmas.detach().cpu().tolist()
    benchmark.group = f"wan21-full-pipeline-{stage}"
    benchmark.extra_info.update(
        {
            "pipeline": pipeline_config.name,
            "source_pipeline": PIPELINE_WAN21_T2V_1PT3B_480P.name,
            "integration": "wan21",
            "model_family": "wan",
            "model_variant": "wan21-t2v-1.3b-480p",
            "batch_shape": list(transformer_config.batch_shape),
            "pixel_resolution": [_PIXEL_HEIGHT, _PIXEL_WIDTH],
            "latent_shape": [
                transformer_config.len_t,
                latent_channels,
                latent_height,
                latent_width,
            ],
            "attention_grid": [
                transformer_config.len_t // patch_t,
                latent_height // patch_h,
                latent_width // patch_w,
            ],
            "global_tokens": global_tokens,
            "local_tokens": transformer.latent_shape[-2],
            "output_frames": output_frames,
            "output_fps": 16,
            "autoregressive_index": 0,
            "rollout_chunks": 1,
            "text_tokens": _TEXT_TOKENS,
            "text_embedding_dim": text_dim,
            "prompt": DEFAULT_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "prompt_embedding_source": "synthetic_precomputed_zeros",
            "raw_prompt_encoding_timed": False,
            "text_encoder_model": source_text_encoder_config.model_id_or_local_path,
            "num_inference_steps": scheduler_config.num_inference_steps,
            "scheduler": type(scheduler).__name__,
            "scheduler_shift": scheduler_config.shift,
            "scheduler_solver_order": scheduler_config.solver_order,
            "resolved_timesteps": resolved_timesteps,
            "resolved_sigmas_fp32": resolved_sigmas,
            "guidance_scale": transformer_config.guidance_scale,
            "cfg_network_branches": 2,
            "context_noise": diffusion_config.context_noise,
            "window_size_t": transformer_config.window_size_t,
            "sink_size_t": transformer_config.sink_size_t,
            "timed_stage": stage,
            "timed_stages": (
                ["unipc_diffuse", "wan_vae_decode"]
                if stage == "generate"
                else ["dit_cache_finalize"]
            ),
            "untimed_lifecycle_stage": (
                "finalize" if stage == "generate" else "synthetic_final_state_setup"
            ),
            "full_generate_calls": 1 if stage == "generate" else 0,
            "finalize_state_source": (
                "generated" if stage == "generate" else "synthetic_zero_clean_latent"
            ),
            "cache_initialization": (
                "cache object allocation excluded; AR0 CUDA graph wrapper "
                "staging included"
            ),
            "checkpoint_loading": "excluded_from_timing",
            "dit_checkpoint": transformer_config.checkpoint_path,
            "decoder_checkpoint": decoder_config.checkpoint_path,
            "dtype": str(dtype),
            "implementation": case.implementation,
            "dit_execution": "pytorch",
            "configured_attention_backend": case.attention_backend.value,
            "configured_sdpa_backend": case.sdpa_backend.value,
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
            "distributed_sample_alignment": "barrier_during_untimed_setup",
            "dit_compiled": transformer_config.compile_network,
            "dit_cuda_graph": transformer_config.use_cuda_graph,
            "dit_cuda_graph_capture_ar_index": transformer._cuda_graph_capture_ar_idx,
            "dit_ar0_execution": "eager_drain_before_steady_state_capture",
            "decoder_compiled": decoder_config.use_compile,
            "decoder_cuda_graph": decoder_config.use_cuda_graph,
            "decoder_ar0_execution": "eager_drain_with_fresh_cache",
            "recurring_pipeline_parameter_count": recurring_parameter_count,
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "targeted_dit_prewarm_calls": _COMPONENT_PREWARM_ROUNDS,
            "targeted_decoder_prewarm_calls": decoder_prewarm_calls,
            "warmup_rounds": _WARMUP_ROUNDS,
            "latency_summary": "single_sample_no_percentiles",
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "startup_timing": (
                "model and checkpoint setup excluded; per-rollout AR0 CUDA "
                "graph wrapper staging included"
            ),
            "compiler_cache_state": (
                "checkpoint loading and targeted component prewarm excluded; "
                "no full-generate warmup"
            ),
            "num_gpus_visible": torch.cuda.device_count(),
            "seed": _SEED,
            "rng_reset_per_sample": True,
        }
    )

    cache: WanInferencePipelineCache | None = None
    latest_output: torch.Tensor | None = None
    stage_peak_cuda_memory_bytes = 0

    def prepare_sample() -> None:
        nonlocal cache, latest_output
        _synchronize_ranks()
        # Drop the prior cache before allocating the next multi-GiB KV cache,
        # allowing the CUDA allocator to reuse its storage without a peak at 2x.
        transformer._cuda_graph_dispatch.reset()
        latest_output = None
        cache = None
        cache = initialize_cache()
        rng = pipeline.diffusion_model.rng
        assert rng is not None
        rng.manual_seed(_SEED)
        torch.cuda.synchronize(device)
        _synchronize_ranks()

    def record_stage_peak_memory() -> None:
        nonlocal stage_peak_cuda_memory_bytes
        stage_peak_cuda_memory_bytes = max(
            stage_peak_cuda_memory_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )

    if stage == "generate":

        def setup_generate() -> None:
            prepare_sample()
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_generate() -> torch.Tensor:
            nonlocal latest_output
            assert cache is not None
            latest_output = pipeline.generate(autoregressive_index=0, cache=cache)
            torch.cuda.synchronize(device)
            return latest_output

        def teardown_generate() -> None:
            assert cache is not None
            record_stage_peak_memory()
            pipeline.finalize(autoregressive_index=0, cache=cache)
            torch.cuda.synchronize(device)

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
            prepare_sample()
            assert cache is not None
            assert isinstance(cache.transformer_cache, Wan21TransformerCache)
            # AR0 replaces the full one-chunk KV window and context_noise is
            # zero, so values do not change which finalize kernels execute.
            cache.autoregressive_index = 0
            cache.transformer_cache.start(0)
            cache.final_state = DiffusionModel.FinalState(
                clean_latent=torch.zeros(
                    transformer.latent_shape,
                    device=device,
                    dtype=dtype,
                ),
                autoregressive_index=0,
                cache=cache.transformer_cache,
            )
            torch.cuda.synchronize(device)
            _synchronize_ranks()
            torch.cuda.reset_peak_memory_stats(device)

        def synchronized_finalize() -> None:
            assert cache is not None
            pipeline.finalize(autoregressive_index=0, cache=cache)
            torch.cuda.synchronize(device)

        def teardown_finalize() -> None:
            record_stage_peak_memory()

        benchmark.pedantic(
            synchronized_finalize,
            setup=setup_finalize,
            teardown=teardown_finalize,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
        output = None

    benchmark.extra_info["peak_cuda_memory_bytes"] = stage_peak_cuda_memory_bytes
    assert benchmark.stats is not None
    single_sample_stage_s = benchmark.stats.stats.median
    benchmark.extra_info.update(
        {
            f"single_sample_{stage}_ms": single_sample_stage_s * 1_000,
            f"single_sample_{stage}_rollouts_per_second": (1.0 / single_sample_stage_s),
        }
    )
    if stage == "generate":
        benchmark.extra_info["single_sample_generate_output_fps"] = (
            output_frames / single_sample_stage_s
        )

    if stage == "generate":
        assert output is not None
        assert output.shape == (
            output_frames,
            3,
            _PIXEL_HEIGHT,
            _PIXEL_WIDTH,
        )
        assert torch.isfinite(output).all()
    else:
        assert cache is not None
        assert cache.final_state is not None
        assert torch.isfinite(cache.final_state.clean_latent).all()
