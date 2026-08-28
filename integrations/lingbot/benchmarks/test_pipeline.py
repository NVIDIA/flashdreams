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

"""Steady-state full-pipeline benchmark for LingBot streaming inference.

Run the benchmark with::

    uv run --group test pytest \
        integrations/lingbot/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from lingbot.config import PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3
from lingbot.encoder.camctrl import CamCtrlInput, I2VCamCtrlEncoder
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.transformer import LingbotWorldTransformer, LingbotWorldTransformerConfig
from lingbot.transformer.impl.network import LingbotWorldDiTNetworkConfig

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.taehv import TeahvVAEDecoder
from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
)
from integrations.lingbot.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot full-pipeline benchmark requires CUDA"

_PIXEL_HEIGHT = 464
_PIXEL_WIDTH = 832
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


def _camera_input(
    *,
    num_frames: int,
    start_frame: int,
    device: torch.device,
) -> CamCtrlInput:
    """Build a deterministic camera trajectory for one autoregressive chunk."""
    intrinsics = torch.tensor(
        [_PIXEL_WIDTH, _PIXEL_WIDTH, _PIXEL_WIDTH / 2, _PIXEL_HEIGHT / 2],
        device=device,
        dtype=torch.float32,
    ).expand(num_frames, -1)
    poses = (
        torch.eye(4, device=device, dtype=torch.float32)
        .expand(num_frames, -1, -1)
        .clone()
    )
    poses[:, 0, 3] = torch.arange(
        start_frame,
        start_frame + num_frames,
        device=device,
        dtype=torch.float32,
    ).mul_(0.01)
    return CamCtrlInput(intrinsics=intrinsics, poses=poses, world_scale=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda case: case.pytest_id)
def test_full_pipeline_generate_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark recurring pipeline generation for one attention configuration."""
    _run_full_pipeline_benchmark(benchmark, case=case)


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
) -> None:
    """Run one full-pipeline benchmark variant."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot full-pipeline benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True

    # Prompt encoding runs once before streaming. Precomputed embeddings keep
    # the measured path focused on recurring camera encoding, diffusion,
    # decoding, and cache bookkeeping.
    pipeline_config = derive_config(
        PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
        name=f"lingbot-full-pipeline-{case.pytest_id}-benchmark",
        text_encoder=None,
        enable_sync_and_profile=False,
        diffusion_model={
            "seed": _SEED,
            "transformer": {
                "init_device": str(device),
                "compile_network": True,
                "use_cuda_graph": True,
                "network": {
                    "cp_method": "ring",
                    "self_attention_backend": case.self_attention_backend,
                    "cross_attention_backend": case.cross_attention_backend,
                    "self_attn_optimized_impl_config": case.self_attn_optimized_impl_config,
                    "cross_attn_optimized_impl_config": case.cross_attn_optimized_impl_config,
                },
            },
        },
    )
    pipeline = pipeline_config.setup().to(device=device)
    assert isinstance(pipeline, LingbotWorldInferencePipeline)
    pipeline.eval()
    assert isinstance(pipeline.encoder, I2VCamCtrlEncoder)
    assert isinstance(pipeline.decoder, TeahvVAEDecoder)
    assert pipeline.text_encoder is None

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    assert isinstance(transformer_config, LingbotWorldTransformerConfig)
    assert isinstance(scheduler_config, FlowMatchSchedulerConfig)
    network_config = transformer_config.network
    assert isinstance(network_config, LingbotWorldDiTNetworkConfig)
    assert network_config.self_attention_backend is case.self_attention_backend
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
    assert isinstance(transformer, LingbotWorldTransformer)
    assert transformer.config is transformer_config
    dtype = transformer_config.dtype
    spatial_compression = int(pipeline.decoder.spatial_compression_ratio)
    latent_height = _PIXEL_HEIGHT // spatial_compression
    latent_width = _PIXEL_WIDTH // spatial_compression

    text_embeddings = torch.zeros(
        (_TEXT_TOKENS, network_config.text_dim),
        device=device,
        dtype=dtype,
    )
    image = torch.zeros(
        (1, 3, _PIXEL_HEIGHT, _PIXEL_WIDTH),
        device=device,
        dtype=dtype,
    )
    parent_cache = super(WanInferencePipeline, pipeline).initialize_cache(
        transformer_context={
            "height": latent_height,
            "width": latent_width,
            "text_embeddings": text_embeddings,
        }
    )
    cache = WanInferencePipelineCache(
        transformer_cache=parent_cache.transformer_cache,
        encoder_cache=parent_cache.encoder_cache,
        decoder_cache=parent_cache.decoder_cache,
        image=image,
    )
    del text_embeddings

    first_input_frames = pipeline.get_num_input_frames(0)
    steady_input_frames = pipeline.get_num_input_frames(1)
    steady_output_frames = pipeline.get_num_output_frames(1)
    capture_ar_index = transformer._cuda_graph_capture_ar_idx
    cache_prefill_chunks = capture_ar_index + 1
    total_chunks = cache_prefill_chunks + _WARMUP_ROUNDS + _BENCHMARK_ROUNDS + 1
    camera_inputs = tuple(
        _camera_input(
            num_frames=(first_input_frames if index == 0 else steady_input_frames),
            start_frame=(
                0
                if index == 0
                else first_input_frames + (index - 1) * steady_input_frames
            ),
            device=device,
        )
        for index in range(total_chunks)
    )

    def run_chunk(autoregressive_index: int) -> torch.Tensor:
        output = pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=camera_inputs[autoregressive_index],
        )
        pipeline.finalize(autoregressive_index=autoregressive_index, cache=cache)
        return output

    # Fill the local KV window and capture the first steady-state graph before
    # pytest-benchmark's warmups and measured rounds.
    for autoregressive_index in range(cache_prefill_chunks):
        output = run_chunk(autoregressive_index)
    torch.cuda.synchronize()

    benchmark.group = "lingbot-full-pipeline-generate"
    next_chunk_index = cache_prefill_chunks

    def synchronized_generate() -> torch.Tensor:
        output = pipeline.generate(
            autoregressive_index=next_chunk_index,
            cache=cache,
            input=camera_inputs[next_chunk_index],
        )
        torch.cuda.synchronize()
        return output

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

    assert output.shape == (
        steady_output_frames,
        3,
        _PIXEL_HEIGHT,
        _PIXEL_WIDTH,
    )
    assert torch.isfinite(output).all()
