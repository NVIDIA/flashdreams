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

"""Benchmark the complete LingBot DiT network.

Run the benchmark with::

    uv run --group test pytest \
        integrations/lingbot/benchmarks/test_network.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from lingbot.transformer.impl.modules import CamCtrlBlock
from lingbot.transformer.impl.network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetwork14BConfig,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.infra.acceleration import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.compile import compile_module
from integrations.lingbot.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot DiT network benchmark requires CUDA"

# Production interactive geometry: 464x832 pixels become 58x104 latents.
_LATENT_HEIGHT = 58
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 3
_WINDOW_SIZE_T = 15
_SINK_SIZE_T = 3
_TEXT_TOKENS = 512
_DIFFUSION_TIMESTEP = 450.0
_WARMUP_ROUNDS = 5
_BENCHMARK_ROUNDS = 50
_SEED = 0


def _build_network(
    config: LingbotWorldDiTNetwork14BConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> LingbotWorldDiTNetwork:
    """Construct the 14B network directly on its benchmark device."""
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            network = LingbotWorldDiTNetwork(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return network.to(device=device, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda case: case.pytest_id)
@torch.inference_mode()
def test_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark one compiled DiT attention configuration at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot DiT network benchmark requires bfloat16 support")

    device = torch.device("cuda")
    skip_unsupported_device(case, device)
    dtype = torch.bfloat16
    torch.manual_seed(_SEED)

    config = LingbotWorldDiTNetwork14BConfig(
        in_dim=36,
        patch_embedding_type="conv3d",
        control_type="cam",
        cp_method="ring",
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        self_attn_optimized_impl_config=case.self_attn_optimized_impl_config,
        cross_attn_optimized_impl_config=case.cross_attn_optimized_impl_config,
    )
    network = _build_network(config, device=device, dtype=dtype)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    assert all(isinstance(block, CamCtrlBlock) for block in network.blocks)
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

    patch_t = _CHUNK_SIZE_T // config.patch_size[0]
    patch_h = _LATENT_HEIGHT // config.patch_size[1]
    patch_w = _LATENT_WIDTH // config.patch_size[2]
    patch_volume = config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    sink_tokens = _SINK_SIZE_T * tokens_per_frame

    x = torch.randn(
        (chunk_tokens, config.in_dim * patch_volume),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    plucker = torch.randn(
        (chunk_tokens, 6 * 64 * patch_volume),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor(_DIFFUSION_TIMESTEP, device=device, dtype=dtype)
    text_embeddings = torch.randn(
        (_TEXT_TOKENS, config.text_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = network.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=sink_tokens,
        text_embeddings=text_embeddings,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_h=patch_h,
        len_w=patch_w,
        len_t=patch_t,
        interleaved=True,
        device=device,
    )

    network = compile_module(network)
    capture_chunk_idx = cuda_graph_capture_ar_index(
        sink_size_t=_SINK_SIZE_T,
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
            cache=cache,
            rope_freqs=rope_freqs,
            plucker=plucker,
            current_chunk_idx=chunk_idx,
            eager_mode=False,
        )

    # Fill the rolling cache and capture the first steady-state CUDA graph
    # before pytest-benchmark starts its own warmups.
    benchmark_chunk_idx = capture_chunk_idx + 1
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(capture_chunk_idx + 1):
        cache.before_update(chunk_idx)
        output = forward(chunk_idx, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize()

    benchmark.group = "lingbot-dit-network"
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

    assert output.shape == (chunk_tokens, config.out_dim * patch_volume)
    assert torch.isfinite(output).all()
