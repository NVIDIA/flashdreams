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

"""CPU-safe contract and numerical tests for LongSana."""

from __future__ import annotations

import pytest
import torch
from longsana.config import LONGSANA_CONFIGS, PIPELINE_LONGSANA_2B_480P
from longsana.impl.constants import (
    DEFAULT_DENOISING_TIMESTEPS,
    FIRST_LATENT_BLOCK_FRAMES,
    LATENT_BLOCK_FRAMES,
    LONGSANA_REVISION,
    LONGSANA_TEXT_CONFIG_PATH,
    MAX_ROPE_POSITION,
    SANA_VIDEO_REVISION,
)
from longsana.impl.model import (
    LongSanaBlockState,
    LongSanaNetworkConfig,
    causal_wan_rope,
)
from longsana.impl.pipeline import LongSanaPipelineConfig
from longsana.impl.scheduler import (
    LongSanaFlowMatchScheduler,
    LongSanaFlowMatchSchedulerConfig,
)
from longsana.impl.transformer import (
    LongSanaConditioning,
    LongSanaTransformer,
    LongSanaTransformerCache,
    LongSanaTransformerConfig,
    longsana_state_dict,
)
from sana_wm.impl.stage1_model import SanaWMStage1Spec
from sana_wm.impl.transformer import _load_inference_config
from torch import Tensor

from flashdreams.infra.diffusion.model import DiffusionModel
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoderConfig

pytestmark = pytest.mark.ci_cpu


def _small_spec() -> SanaWMStage1Spec:
    return SanaWMStage1Spec(
        latent_channels=2,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=2,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        temporal_kernel_size=3,
    )


def test_public_config_uses_runtime_v2_and_release_schedule() -> None:
    """Keep the public pipeline on shared diffusion and Wan components."""
    config = PIPELINE_LONGSANA_2B_480P

    assert isinstance(config, LongSanaPipelineConfig)
    assert config._target.__name__ == "LongSanaPipeline"
    assert config.diffusion_model._target is DiffusionModel
    assert isinstance(config.decoder, WanVAEDecoderConfig)
    assert config.decoder.dtype is torch.float32
    assert isinstance(
        config.diffusion_model.transformer,
        LongSanaTransformerConfig,
    )
    scheduler = config.diffusion_model.scheduler
    assert isinstance(scheduler, LongSanaFlowMatchSchedulerConfig)
    assert scheduler.denoising_timesteps == DEFAULT_DENOISING_TIMESTEPS
    assert scheduler.shift == 7.0
    assert scheduler.warp_denoising_step is False
    assert config.name in LONGSANA_CONFIGS
    assert LONGSANA_REVISION in config.diffusion_model.transformer.checkpoint_path
    assert SANA_VIDEO_REVISION in config.decoder.checkpoint_path


def test_packaged_text_config_matches_release() -> None:
    """Use the released Gemma model, 300-token CHI prompt, and BF16 output."""
    config = _load_inference_config(LONGSANA_TEXT_CONFIG_PATH)

    assert config.model.mixed_precision == "bf16"
    assert config.text_encoder.text_encoder_name == "gemma-2-2b-it"
    assert config.text_encoder.model_max_length == 300
    assert config.text_encoder.y_norm_scale_factor == 0.01
    assert config.text_encoder.chi_prompt[-1] == "User Prompt: "


def test_full_model_schema_has_public_checkpoint_shape() -> None:
    """Construct the 2B schema on meta without allocating model weights."""
    with torch.device("meta"):
        model = LongSanaNetworkConfig().setup()

    assert len(model.state_dict()) == 418
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_057_553_344
    assert model.state_dict()["x_embedder.proj.weight"].shape == (
        2240,
        16,
        1,
        2,
        2,
    )
    assert model.state_dict()["final_layer.linear.weight"].shape == (64, 2240)


def test_checkpoint_normalization_selects_generator_model() -> None:
    """Discard critic/EMA entries and remove the wrapper's model prefix."""
    expected = torch.randn(2, 3)
    payload = {
        "generator": {
            "model.layer.weight": expected,
            "scheduler.buffer": torch.ones(1),
        },
        "generator_ema": {"model.layer.weight": torch.zeros_like(expected)},
        "critic": {"weight": torch.ones(1)},
    }

    state = longsana_state_dict(payload)

    assert set(state) == {"layer.weight"}
    assert state["layer.weight"] is expected


def test_recurrent_state_size_and_storage_are_constant() -> None:
    """Accumulate a second block without retaining its token history."""
    torch.manual_seed(7)
    spec = _small_spec()
    model = LongSanaNetworkConfig(spec=spec).setup().eval()
    condition = model.prepare_condition(torch.randn(1, 1, 5, spec.text_dim))
    mask = torch.ones(1, 5)
    states = [LongSanaBlockState() for _ in range(spec.depth)]
    latent = torch.randn(1, spec.latent_channels, 3, 4, 4)

    first = model(
        latent,
        torch.tensor(727.0),
        condition,
        mask,
        states,
        start_frame=0,
        update_state=True,
    )
    first_bytes = sum(state.num_bytes() for state in states)
    pointers = [
        (
            state.value_key.data_ptr(),
            state.key_sum.data_ptr(),
            state.conv_tail.data_ptr(),
        )
        for state in states
        if state.value_key is not None
        and state.key_sum is not None
        and state.conv_tail is not None
    ]
    second = model(
        latent,
        torch.tensor(727.0),
        condition,
        mask,
        states,
        start_frame=3,
        update_state=True,
    )

    assert first.shape == second.shape == latent.shape
    assert sum(state.num_bytes() for state in states) == first_bytes
    assert [
        (
            state.value_key.data_ptr(),
            state.key_sum.data_ptr(),
            state.conv_tail.data_ptr(),
        )
        for state in states
        if state.value_key is not None
        and state.key_sum is not None
        and state.conv_tail is not None
    ] == pointers


def test_causal_rope_uses_absolute_frame_positions() -> None:
    """A later block's temporal frequencies equal the full table slice."""
    complete = causal_wan_rope(
        head_dim=8,
        start_frame=0,
        frames=5,
        height=1,
        width=1,
        device=torch.device("cpu"),
    )
    later = causal_wan_rope(
        head_dim=8,
        start_frame=3,
        frames=2,
        height=1,
        width=1,
        device=torch.device("cpu"),
    )

    assert later.dtype is torch.complex128
    torch.testing.assert_close(later, complete[:, :, 3:5])


def test_scheduler_matches_upstream_precision_and_noise_order() -> None:
    """Match LongSana's double x0 conversion and flattened B/T/C noise draw."""
    config = LongSanaFlowMatchSchedulerConfig(
        num_inference_steps=4,
        shift=7.0,
        denoising_timesteps=list(DEFAULT_DENOISING_TIMESTEPS),
        warp_denoising_step=False,
    )
    scheduler = config.setup()
    assert isinstance(scheduler, LongSanaFlowMatchScheduler)
    initial = torch.linspace(
        -1,
        1,
        1 * 2 * 3 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(1, 2, 3, 2, 2)

    def predict_flow(noisy: Tensor, timestep: Tensor) -> Tensor:
        return noisy * 0.125 + timestep / 4096

    ours_rng = torch.Generator().manual_seed(123)
    actual = scheduler.sample(initial, predict_flow, ours_rng)

    reference_rng = torch.Generator().manual_seed(123)
    noisy = initial
    expected = initial
    for index, timestep in enumerate(scheduler.denoising_step_list):
        sigma = scheduler.denoising_sigmas[index]
        if index > 0:
            batch, channels, frames, height, width = noisy.shape
            noise = torch.randn(
                (batch * frames, channels, height, width),
                dtype=noisy.dtype,
                generator=reference_rng,
            )
            noise = noise.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4)
            noisy = ((1 - sigma) * expected + sigma * noise).to(initial.dtype)
        flow = predict_flow(noisy, timestep.to(initial.dtype))
        expected = (noisy.double() - sigma.double() * flow.double()).to(initial.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_noise_exposes_tchw_but_preserves_upstream_rng_order() -> None:
    """Keep Runtime V2's public layout without changing LongSana's seeded noise."""
    spec = _small_spec()
    transformer = LongSanaTransformerConfig(
        network=LongSanaNetworkConfig(spec=spec),
        dtype=torch.float32,
        latent_height=4,
        latent_width=4,
        first_block_frames=3,
        block_frames=2,
    ).setup()
    assert isinstance(transformer, LongSanaTransformer)
    cache = transformer.initialize_autoregressive_cache()
    cache.start(0)

    actual_rng = torch.Generator().manual_seed(42)
    actual = transformer.initial_noise(
        latent_shape=transformer.latent_shape,
        rng=actual_rng,
        cache=cache,
    )

    expected_rng = torch.Generator().manual_seed(42)
    expected = torch.randn(
        (spec.latent_channels, 3, 4, 4),
        dtype=torch.float32,
        generator=expected_rng,
    ).permute(1, 0, 2, 3)

    assert actual.shape == (3, spec.latent_channels, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_noise_uses_each_sessions_active_block_shape() -> None:
    """Keep first and steady block lengths isolated across interleaved sessions."""
    spec = _small_spec()
    transformer = LongSanaTransformerConfig(
        network=LongSanaNetworkConfig(spec=spec),
        dtype=torch.float32,
        latent_height=4,
        latent_width=4,
        first_block_frames=3,
        block_frames=2,
    ).setup()
    assert isinstance(transformer, LongSanaTransformer)
    first_session = transformer.initialize_autoregressive_cache()
    steady_session = transformer.initialize_autoregressive_cache()

    steady_session.start(0)
    steady_session.finalize(0)
    steady_session.start(1)
    first_session.start(0)

    steady = transformer.initial_noise(
        latent_shape=transformer.latent_shape,
        rng=torch.Generator().manual_seed(1),
        cache=steady_session,
    )
    first = transformer.initial_noise(
        latent_shape=transformer.latent_shape,
        rng=torch.Generator().manual_seed(2),
        cache=first_session,
    )

    assert steady.shape == (2, spec.latent_channels, 4, 4)
    assert first.shape == (3, spec.latent_channels, 4, 4)


def test_transformer_cache_tracks_release_block_boundaries() -> None:
    """Advance 11 latent frames first and 10 thereafter."""
    cache = LongSanaTransformerCache(
        conditioning=LongSanaConditioning(
            condition=torch.empty(1, 1, 5, 12),
            mask=torch.ones(1, 5),
        ),
        block_states=[LongSanaBlockState()],
    )

    cache.start(0)
    assert cache.active_frames == FIRST_LATENT_BLOCK_FRAMES
    cache.finalize(0)
    assert cache.start_frame == FIRST_LATENT_BLOCK_FRAMES

    cache.start(1)
    assert cache.active_frames == LATENT_BLOCK_FRAMES
    cache.finalize(1)
    assert cache.start_frame == FIRST_LATENT_BLOCK_FRAMES + LATENT_BLOCK_FRAMES


def test_transformer_cache_rejects_rope_overflow_before_generation() -> None:
    """Protect direct pipeline callers from exceeding absolute RoPE positions."""
    cache = LongSanaTransformerCache(
        start_frame=MAX_ROPE_POSITION - LATENT_BLOCK_FRAMES + 1,
        next_index=1,
    )

    with pytest.raises(ValueError, match="exceeds.*RoPE table"):
        cache.start(1)

    assert cache.active_index is None
    assert cache.active_frames == 0


def test_generic_scheduler_would_not_preserve_longsana_rng_layout() -> None:
    """Document why the integration uses its narrow scheduler subclass."""
    generic = FlowMatchSchedulerConfig(
        num_inference_steps=4,
        shift=7.0,
        denoising_timesteps=list(DEFAULT_DENOISING_TIMESTEPS),
        warp_denoising_step=False,
    ).setup()
    assert type(generic).__name__ == "FlowMatchScheduler"
