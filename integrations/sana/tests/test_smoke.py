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

"""CPU-safe smoke tests for the SANA-WM configs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import sana_wm._tools as tools_module
import sana_wm.conditioning as conditioning_module
import sana_wm.decoder as decoder_module
import sana_wm.refiner as refiner_module
import torch

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from sana_wm.conditioning import (
    SanaWMCameraConditioningEncoderConfig,
    SanaWMCameraRequest,
    SanaWMConditioningEncoderConfig,
    SanaWMFirstFrameEncoderConfig,
    SanaWMStreamingConditioningEncoderConfig,
    SanaWMStreamingI2VConditioningRequest,
    SanaWMTextPromptEncoderConfig,
    SanaWMTextPromptRequest,
    streaming_chunk_boundaries,
)
from sana_wm.config import (
    PIPELINE_SANA_WM_BIDIRECTIONAL,
    PIPELINE_SANA_WM_STREAMING,
    RUNNER_CONFIGS,
    RUNNER_SANA_WM_BIDIRECTIONAL,
    RUNNER_SANA_WM_STREAMING,
)
from sana_wm.constants import (
    DEFAULT_STREAMING_DENOISING_STEP_LIST,
    SANA_WM_CONFIG_PATH,
    SANA_WM_HF_REPO,
    SANA_WM_MODEL_PATH,
    SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
    SANA_WM_STREAMING_CONFIG_PATH,
    SANA_WM_STREAMING_HF_REPO,
    SANA_WM_STREAMING_MODEL_PATH,
    SANA_WM_STREAMING_REFINER_GEMMA_ROOT,
    SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES,
    SANA_WM_STREAMING_REFINER_ROOT,
)
from sana_wm.decoder import (
    SanaWMDecodedVideo,
    SanaWMLTX2LatentRefinerConfig,
    SanaWMLTX2VAEDecoderConfig,
    SanaWMStreamingLTX2LatentRefinerConfig,
    SanaWMStreamingLTX2VAEDecoderConfig,
    SanaWMStreamingVideoDecoder,
    SanaWMStreamingVideoDecoderConfig,
    SanaWMVideoDecoderConfig,
)
from sana_wm.diffusion import SanaWMDiffusionModelConfig
from sana_wm.quant import (
    TorchScaledMMFP4Linear,
    TorchScaledMMFP8Linear,
    apply_rht16,
    nvfp4_global_scale,
    replace_linear_with_torch_fp4,
    replace_linear_with_torch_fp8,
)
from sana_wm.refiner import SanaWMLTX2Refiner, _pack_latents, _unpack_latents
from sana_wm.runner import (
    SanaWMRunner,
    SanaWMRunnerConfig,
    SanaWMStreamingRunner,
    SanaWMStreamingRunnerConfig,
    _pipeline_config,
    _resolve_quant_backend,
    _streaming_pipeline_config,
    _validate_precision_request,
)
from sana_wm.scheduler import (
    SanaWMLTXEulerScheduler,
    SanaWMLTXEulerSchedulerConfig,
)
from sana_wm.stage1_model import (
    SANA_WM_STAGE1_SPEC,
    SANA_WM_STREAMING_STAGE1_SPEC,
    GLUMBConvTemp,
    SanaWMStage1Model,
    SanaWMStage1Spec,
    Stage1SelfAttention,
    linearize_stage1_ffn_for_quant,
)
from sana_wm.transformer import (
    SanaWMStage1Conditioning,
    SanaWMStreamingStage1Conditioning,
    SanaWMStreamingTransformerCache,
    SanaWMStreamingTransformerConfig,
    SanaWMTransformerCache,
    SanaWMTransformerConfig,
    _avoid_degenerate_tile_tail,
    _load_inference_config,
    _stage1_quant_include_patterns,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.diffusion.model import DiffusionModel

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runner_config_is_registered() -> None:
    """Expose the SANA-WM runner slugs."""
    assert RUNNER_CONFIGS == {
        "sana-wm-bidirectional": RUNNER_SANA_WM_BIDIRECTIONAL,
        "sana-wm-streaming": RUNNER_SANA_WM_STREAMING,
    }


def test_runner_name_mirrors_pipeline_name() -> None:
    """Keep ``flashdreams-run <slug>`` aligned with the wrapped config slug."""
    assert RUNNER_SANA_WM_BIDIRECTIONAL.runner_name == (
        PIPELINE_SANA_WM_BIDIRECTIONAL.name
    )
    assert RUNNER_SANA_WM_STREAMING.runner_name == PIPELINE_SANA_WM_STREAMING.name


def test_runner_has_description() -> None:
    """Provide non-empty CLI help text for the runner registry."""
    assert RUNNER_SANA_WM_BIDIRECTIONAL.description.strip()
    assert RUNNER_SANA_WM_STREAMING.description.strip()


def test_pipeline_uses_sana_diffusion_model() -> None:
    """Keep the public runner wired to explicit SANA-WM boundaries."""
    pipeline = PIPELINE_SANA_WM_BIDIRECTIONAL
    transformer = PIPELINE_SANA_WM_BIDIRECTIONAL.diffusion_model.transformer

    assert isinstance(pipeline.encoder, SanaWMConditioningEncoderConfig)
    assert isinstance(pipeline.decoder, SanaWMVideoDecoderConfig)
    assert isinstance(pipeline.diffusion_model, SanaWMDiffusionModelConfig)
    assert pipeline.diffusion_model._target is DiffusionModel
    assert isinstance(transformer, SanaWMTransformerConfig)
    assert isinstance(pipeline.diffusion_model.scheduler, SanaWMLTXEulerSchedulerConfig)
    assert isinstance(pipeline.decoder.refiner, SanaWMLTX2LatentRefinerConfig)


def test_streaming_pipeline_uses_sana_streaming_components() -> None:
    """Keep the streaming runner wired to explicit streaming boundaries."""
    pipeline = PIPELINE_SANA_WM_STREAMING
    transformer = pipeline.diffusion_model.transformer

    assert isinstance(pipeline.encoder, SanaWMStreamingConditioningEncoderConfig)
    assert isinstance(pipeline.decoder, SanaWMStreamingVideoDecoderConfig)
    assert isinstance(pipeline.diffusion_model, SanaWMDiffusionModelConfig)
    assert pipeline.diffusion_model._target is DiffusionModel
    assert isinstance(transformer, SanaWMStreamingTransformerConfig)
    assert isinstance(pipeline.diffusion_model.scheduler, SanaWMLTXEulerSchedulerConfig)
    assert pipeline.diffusion_model.scheduler.denoising_step_list == (
        DEFAULT_STREAMING_DENOISING_STEP_LIST
    )
    assert isinstance(pipeline.decoder.vae_decoder, SanaWMStreamingLTX2VAEDecoderConfig)
    assert isinstance(pipeline.decoder.refiner, SanaWMStreamingLTX2LatentRefinerConfig)


def test_sana_decoder_uses_video_decoder_contract() -> None:
    """Expose SANA-WM VAE temporal sizing through the FlashDreams decoder API."""
    decoder = SanaWMVideoDecoderConfig().setup()

    assert isinstance(decoder, StreamingVideoDecoder)
    assert decoder.spatial_compression_ratio == 32
    assert decoder.temporal_compression_ratio == 8
    assert decoder.get_output_temporal_size(0, 21) == 161
    assert decoder.get_input_temporal_size(0, 161) == 21
    with pytest.raises(ValueError, match="8k\\+1"):
        decoder.get_input_temporal_size(0, 160)
    with pytest.raises(ValueError, match="one AR step"):
        decoder.get_output_temporal_size(1, 21)


def test_streaming_decoder_uses_video_decoder_contract() -> None:
    """Expose streaming chunk temporal sizing through the decoder API."""
    decoder = SanaWMStreamingVideoDecoderConfig(refiner=None).setup()

    assert isinstance(decoder, StreamingVideoDecoder)
    assert isinstance(decoder, SanaWMStreamingVideoDecoder)
    assert decoder.spatial_compression_ratio == 32
    assert decoder.temporal_compression_ratio == 8
    assert decoder.get_output_temporal_size(0, 4) == 24
    assert decoder.get_input_temporal_size(0, 24) == 4
    assert decoder.get_output_temporal_size(1, 3) == 24
    assert decoder.get_input_temporal_size(1, 24) == 3
    with pytest.raises(ValueError, match="multiple of 8"):
        decoder.get_input_temporal_size(1, 25)


def test_sana_diffusion_config_instantiates_base_model() -> None:
    """Use FlashDreams' shared diffusion model rather than a Sana-only runner."""
    model = PIPELINE_SANA_WM_BIDIRECTIONAL.diffusion_model.setup()

    assert type(model) is DiffusionModel
    assert isinstance(model.transformer.config, SanaWMTransformerConfig)
    assert isinstance(model.scheduler, SanaWMLTXEulerScheduler)


def test_streaming_diffusion_config_instantiates_base_model() -> None:
    """Keep streaming on the shared diffusion loop with streaming components."""
    model = PIPELINE_SANA_WM_STREAMING.diffusion_model.setup()

    assert type(model) is DiffusionModel
    assert isinstance(model.transformer.config, SanaWMStreamingTransformerConfig)
    assert isinstance(model.scheduler, SanaWMLTXEulerScheduler)
    assert model.scheduler.config.denoising_step_list == (
        DEFAULT_STREAMING_DENOISING_STEP_LIST
    )


def test_runner_pipeline_config_routes_runtime_fields_to_components() -> None:
    """Apply CLI overrides to the component that owns each runtime field."""
    cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        config_path="local_config.yaml",
        model_path="local_model.safetensors",
        stage1_precision="fp8",
        step=7,
        flow_shift=6.5,
        no_refiner=True,
        offload_vae=True,
        offload_text_encoder=True,
        offload_stage1=True,
    )

    pipeline = _pipeline_config(cfg, quant_backend="torch-fp8")
    diffusion_model = pipeline.diffusion_model
    assert isinstance(diffusion_model, SanaWMDiffusionModelConfig)
    transformer = diffusion_model.transformer
    scheduler = diffusion_model.scheduler
    assert isinstance(transformer, SanaWMTransformerConfig)
    assert isinstance(scheduler, SanaWMLTXEulerSchedulerConfig)

    assert isinstance(pipeline.encoder, SanaWMConditioningEncoderConfig)
    assert isinstance(pipeline.decoder, SanaWMVideoDecoderConfig)
    assert pipeline.encoder.config_path == "local_config.yaml"
    assert pipeline.encoder.text_encoder.config_path == "local_config.yaml"
    assert pipeline.encoder.text_encoder.stage1_precision == "fp8"
    assert pipeline.encoder.text_encoder.quant_backend == "torch-fp8"
    assert pipeline.encoder.text_encoder.offload_text_encoder is True
    assert pipeline.encoder.first_frame_encoder.offload_vae is True
    assert pipeline.decoder.vae_decoder.config_path == "local_config.yaml"
    assert pipeline.decoder.vae_decoder.offload_vae is True
    assert pipeline.decoder.refiner is None
    assert transformer.checkpoint_path == "local_model.safetensors"
    assert transformer.offload_stage1 is True
    assert scheduler.num_inference_steps == 7
    assert scheduler.shift == 6.5


def test_streaming_runner_pipeline_config_routes_runtime_fields_to_components() -> None:
    """Apply streaming CLI overrides to the component that owns each field."""
    cfg = derive_config(
        RUNNER_SANA_WM_STREAMING,
        config_path="stream_config.yaml",
        model_path="stream_model.safetensors",
        causal_vae_path="local_causal_vae",
        refiner_root="local_refiner",
        refiner_gemma_root="local_gemma",
        stage1_precision="fp8",
        step=4,
        flow_shift=8.0,
        no_sink_token=True,
        num_frame_per_block=5,
        num_cached_blocks=3,
        denoising_step_list=(1000, 500, 100, 0),
        refiner_block_size=5,
        refiner_kv_max_frames=17,
        offload_vae=True,
        offload_text_encoder=True,
        offload_stage1=True,
    )

    pipeline = _streaming_pipeline_config(cfg, quant_backend="torch-fp8")
    diffusion_model = pipeline.diffusion_model
    assert isinstance(diffusion_model, SanaWMDiffusionModelConfig)
    transformer = diffusion_model.transformer
    scheduler = diffusion_model.scheduler
    assert isinstance(transformer, SanaWMStreamingTransformerConfig)
    assert isinstance(scheduler, SanaWMLTXEulerSchedulerConfig)

    assert isinstance(pipeline.encoder, SanaWMStreamingConditioningEncoderConfig)
    assert isinstance(pipeline.decoder, SanaWMStreamingVideoDecoderConfig)
    assert pipeline.encoder.config_path == "stream_config.yaml"
    assert pipeline.encoder.text_encoder.config_path == "stream_config.yaml"
    assert pipeline.encoder.text_encoder.stage1_precision == "fp8"
    assert pipeline.encoder.text_encoder.quant_backend == "torch-fp8"
    assert pipeline.encoder.text_encoder.offload_text_encoder is True
    assert pipeline.encoder.first_frame_encoder.offload_vae is True
    assert pipeline.decoder.vae_decoder.config_path == "stream_config.yaml"
    assert pipeline.decoder.vae_decoder.vae_path == "local_causal_vae"
    assert pipeline.decoder.vae_decoder.offload_vae is True
    assert isinstance(pipeline.decoder.refiner, SanaWMStreamingLTX2LatentRefinerConfig)
    assert pipeline.decoder.refiner.refiner_root == "local_refiner"
    assert pipeline.decoder.refiner.refiner_gemma_root == "local_gemma"
    assert pipeline.decoder.refiner.kv_max_frames == 17
    assert pipeline.decoder.refiner.block_size == 5
    assert transformer.checkpoint_path == "stream_model.safetensors"
    assert transformer.offload_stage1 is True
    assert transformer.num_frame_per_block == 5
    assert transformer.num_cached_blocks == 3
    assert transformer.sink_token is False
    assert scheduler.num_inference_steps == 4
    assert scheduler.shift == 8.0
    assert scheduler.denoising_step_list == (1000, 500, 100, 0)


def test_sana_ltx_scheduler_step_pins_zero_timestep_tokens() -> None:
    """Keep first-frame tokens fixed in the per-token LTX Euler step."""
    scheduler = SanaWMLTXEulerSchedulerConfig(num_inference_steps=4).setup()

    timesteps = scheduler.timesteps(
        num_inference_steps=4,
        shift=5.0,
        device=torch.device("cpu"),
    )
    sample = torch.ones((1, 2, 1))
    model_output = torch.ones_like(sample)
    stepped = scheduler.step_ltx(
        model_output=model_output,
        timestep=torch.tensor(1000.0),
        next_timestep=torch.tensor(500.0),
        sample=sample,
        per_token_timesteps=torch.tensor([[1000.0, 0.0]]),
        schedule_timesteps=torch.tensor([1000.0, 500.0, 0.0]),
    )

    assert isinstance(scheduler, SanaWMLTXEulerScheduler)
    assert timesteps.shape == (5,)
    assert float(timesteps[-1]) == 0.0
    torch.testing.assert_close(
        stepped,
        torch.tensor([[[1.5], [1.0]]]),
    )


def test_sana_ltx_scheduler_matches_diffusers_per_token_step() -> None:
    """Match diffusers FlowMatch Euler for SANA-WM's per-token branch."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    ours = SanaWMLTXEulerSchedulerConfig(num_inference_steps=4).setup()
    schedule_timesteps = ours.timesteps(
        num_inference_steps=4,
        shift=5.0,
        device=torch.device("cpu"),
    )
    upstream = cast(Any, FlowMatchEulerDiscreteScheduler(shift=5.0))
    upstream.set_timesteps(4, device=torch.device("cpu"))
    torch.testing.assert_close(schedule_timesteps[:-1], upstream.timesteps)
    torch.testing.assert_close(schedule_timesteps / 1000.0, upstream.sigmas)

    sample = torch.tensor([[[1.0, -0.5], [0.25, 0.5], [-1.0, 0.75]]])
    model_output = torch.tensor([[[0.2, 0.4], [-0.3, 0.1], [0.5, -0.2]]])
    per_token_timesteps = torch.stack(
        [
            torch.tensor(
                [
                    float(upstream.timesteps[0]),
                    0.0,
                    float(upstream.timesteps[0]),
                ]
            )
        ]
    )

    expected = upstream.step(
        model_output,
        upstream.timesteps[0],
        sample,
        per_token_timesteps=per_token_timesteps,
        return_dict=False,
    )[0]
    actual = ours.step_ltx(
        model_output=model_output,
        timestep=schedule_timesteps[0],
        next_timestep=schedule_timesteps[1],
        sample=sample,
        per_token_timesteps=per_token_timesteps,
        schedule_timesteps=schedule_timesteps,
    )

    torch.testing.assert_close(actual, expected)


def test_sana_ltx_scheduler_uses_explicit_streaming_timesteps() -> None:
    """Use the distilled streaming timestep list verbatim when configured."""
    scheduler = SanaWMLTXEulerSchedulerConfig(
        denoising_step_list=DEFAULT_STREAMING_DENOISING_STEP_LIST,
    ).setup()

    timesteps = scheduler.timesteps(
        num_inference_steps=999,
        shift=1.0,
        device=torch.device("cpu"),
    )

    assert timesteps.dtype == torch.float32
    torch.testing.assert_close(
        timesteps,
        torch.tensor(DEFAULT_STREAMING_DENOISING_STEP_LIST, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="end with 0"):
        SanaWMLTXEulerSchedulerConfig(
            denoising_step_list=(1000, 500),
        ).setup().timesteps(
            num_inference_steps=2,
            shift=1.0,
            device=torch.device("cpu"),
        )


def test_sana_transformer_keeps_conditioned_frame_fixed_with_generic_scheduler() -> (
    None
):
    """Keep SANA conditioning out of the scheduler and inside the transformer."""
    scheduler = SanaWMLTXEulerSchedulerConfig(num_inference_steps=1).setup()
    transformer = SanaWMTransformerConfig().setup()
    latent_shape = (1, 1, 2, 1, 1)
    initial_noise = torch.zeros(latent_shape, dtype=torch.float32)
    initial_noise[:, :, 0] = 5.0
    conditioning = SanaWMStage1Conditioning(
        condition=torch.ones((1, 1, 1, 1)),
        uncondition=None,
        model_kwargs={"data_info": {"condition_frame_info": {0: 0.0}}},
        first_latent=torch.empty((1, 1, 1, 1, 1)),
        latent_shape=latent_shape,
        cfg_scale=1.0,
        flow_shift=5.0,
        steps=1,
        seed=0,
    )

    class DummyModel(torch.nn.Module):
        def forward(
            self,
            noisy_latent: torch.Tensor,
            timestep: torch.Tensor,
            _prompt_embeds: torch.Tensor,
            **_kwargs: object,
        ) -> torch.Tensor:
            assert timestep.shape == (1, 1, 2)
            torch.testing.assert_close(timestep[:, :, 0], torch.zeros((1, 1)))
            assert torch.all(timestep[:, :, 1] > 0)
            return -torch.ones_like(noisy_latent)

    transformer.model = DummyModel()
    transformer._model_built = True

    def predict_flow(
        noisy_latent: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return transformer.predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=SanaWMTransformerCache(),
            input=conditioning,
        )

    sampled = scheduler.sample(
        initial_noise=initial_noise,
        predict_flow=predict_flow,
    )

    torch.testing.assert_close(sampled[:, :, 0], initial_noise[:, :, 0])
    assert torch.all(sampled[:, :, 1] > initial_noise[:, :, 1])


def test_transformer_contract_shape_and_conditioning_guard() -> None:
    """Pin the SANA-WM Stage-1 boundary to the public model layout."""
    transformer_cfg = SanaWMTransformerConfig()
    transformer = transformer_cfg.setup()

    assert transformer.latent_shape == (1, 128, 21, 22, 40)
    cache = transformer.initialize_autoregressive_cache()

    with pytest.raises(RuntimeError, match="without conditioning"):
        transformer.predict_flow(
            noisy_latent=torch.empty(transformer.latent_shape),
            timestep=torch.tensor(1000.0),
            cache=cache,
        )


def test_transformer_initial_noise_uses_conditioning_payload() -> None:
    """Generate first-frame-pinned noise through the shared transformer hook."""
    transformer = SanaWMTransformerConfig().setup()
    transformer.weight_dtype = torch.float32
    first_latent = torch.full((1, 1, 1, 1, 1), 7.0)
    conditioning = SanaWMStage1Conditioning(
        condition=torch.empty((1, 1, 1, 1)),
        uncondition=None,
        model_kwargs={},
        first_latent=first_latent,
        latent_shape=(1, 1, 2, 1, 1),
        cfg_scale=1.0,
        flow_shift=1.0,
        steps=1,
        seed=123,
    )
    cache = transformer.initialize_autoregressive_cache()

    noise = transformer.initial_noise(
        latent_shape=(1, 1, 2, 1, 1),
        rng=None,
        cache=cache,
        input=conditioning,
    )

    assert cache.conditioning is conditioning
    assert noise.shape == (1, 1, 2, 1, 1)
    torch.testing.assert_close(noise[:, :, :1], first_latent)


def test_streaming_transformer_commits_chunks_into_prefix_cache() -> None:
    """Keep chunk state while bounding the per-step Stage-1 model window."""

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def forward(
            self,
            noisy_latent: torch.Tensor,
            timestep: torch.Tensor,
            prompt_embeds: torch.Tensor,
            **_kwargs: object,
        ) -> tuple[torch.Tensor]:
            del prompt_embeds
            camera_conditions = cast(torch.Tensor, _kwargs["camera_conditions"])
            chunk_plucker = cast(torch.Tensor, _kwargs["chunk_plucker"])
            data_info = cast(dict[object, object], _kwargs["data_info"])
            chunk_index = cast(list[int], _kwargs["chunk_index"])
            self.calls.append(
                {
                    "noisy_latent": noisy_latent.detach().clone(),
                    "timestep": timestep.detach().clone(),
                    "camera_conditions": camera_conditions.detach().clone(),
                    "chunk_plucker": chunk_plucker.detach().clone(),
                    "data_info": dict(data_info),
                    "chunk_index": list(chunk_index),
                }
            )
            return (torch.ones_like(noisy_latent),)

    total_frames = 13

    def conditioning_for_chunk(
        *,
        start: int,
        end: int,
        chunk_index: int,
    ) -> SanaWMStreamingStage1Conditioning:
        return SanaWMStreamingStage1Conditioning(
            condition=torch.ones((1, 1, 1)),
            uncondition=None,
            model_kwargs={
                "data_info": {"condition_frame_info": {0: 0.0}},
                "camera_conditions": torch.arange(
                    total_frames, dtype=torch.float32
                ).reshape(1, total_frames, 1),
                "chunk_plucker": torch.arange(
                    total_frames, dtype=torch.float32
                ).reshape(1, 1, total_frames, 1, 1),
                "chunk_index": [0, 4, 7, 10],
            },
            first_latent=torch.full((1, 1, 1, 1, 1), 7.0),
            latent_shape=(1, 1, end - start, 1, 1),
            cfg_scale=1.0,
            flow_shift=8.0,
            steps=4,
            seed=123,
            total_latent_shape=(1, 1, total_frames, 1, 1),
            start_frame=start,
            end_frame=end,
            chunk_index=chunk_index,
            chunk_boundaries=(0, 4, 7, 10, 13),
        )

    transformer = SanaWMStreamingTransformerConfig().setup()
    transformer.weight_dtype = torch.float32
    dummy_model = DummyModel()
    transformer.model = dummy_model
    transformer._model_built = True
    chunks = (
        conditioning_for_chunk(start=0, end=4, chunk_index=0),
        conditioning_for_chunk(start=4, end=7, chunk_index=1),
        conditioning_for_chunk(start=7, end=10, chunk_index=2),
        conditioning_for_chunk(start=10, end=13, chunk_index=3),
    )
    cache = transformer.initialize_autoregressive_cache(conditioning=chunks[0])
    assert isinstance(cache, SanaWMStreamingTransformerCache)

    flows: list[torch.Tensor] = []
    cleans: list[torch.Tensor] = []
    for chunk, timestep, clean_value in (
        (chunks[0], 1000.0, 2.0),
        (chunks[1], 500.0, 4.0),
        (chunks[2], 250.0, 6.0),
        (chunks[3], 125.0, 8.0),
    ):
        noise = transformer.initial_noise(
            latent_shape=chunk.latent_shape,
            rng=None,
            cache=cache,
            input=chunk,
        )
        flows.append(
            transformer.predict_flow(
                noisy_latent=noise,
                timestep=torch.tensor(timestep),
                cache=cache,
                input=chunk,
            )
        )
        clean = torch.full_like(noise, clean_value)
        cleans.append(clean)
        transformer.postprocess_clean_latent(clean, cache, input=chunk)

    assert len(dummy_model.calls) == 4
    assert dummy_model.calls[0]["noisy_latent"].shape == (1, 1, 4, 1, 1)
    assert dummy_model.calls[1]["noisy_latent"].shape == (1, 1, 7, 1, 1)
    assert dummy_model.calls[2]["noisy_latent"].shape == (1, 1, 10, 1, 1)
    assert dummy_model.calls[3]["noisy_latent"].shape == (1, 1, 10, 1, 1)
    torch.testing.assert_close(
        dummy_model.calls[0]["timestep"],
        torch.tensor([[[0.0, 1000.0, 1000.0, 1000.0]]]),
    )
    torch.testing.assert_close(
        dummy_model.calls[1]["timestep"],
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 500.0]]]),
    )
    torch.testing.assert_close(
        dummy_model.calls[3]["timestep"],
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 125.0, 125.0, 125.0]]]),
    )
    camera3 = dummy_model.calls[3]["camera_conditions"]
    assert isinstance(camera3, torch.Tensor)
    torch.testing.assert_close(
        camera3.flatten(),
        torch.tensor([0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )
    plucker3 = dummy_model.calls[3]["chunk_plucker"]
    assert isinstance(plucker3, torch.Tensor)
    torch.testing.assert_close(plucker3.flatten(), camera3.flatten())
    assert dummy_model.calls[3]["data_info"] == {"condition_frame_info": {0: 0.0}}
    assert dummy_model.calls[3]["chunk_index"] == [0, 4, 7]
    torch.testing.assert_close(flows[0][:, :, :1], torch.zeros_like(flows[0][:, :, :1]))
    torch.testing.assert_close(flows[0][:, :, 1:], torch.ones_like(flows[0][:, :, 1:]))
    torch.testing.assert_close(flows[1], torch.ones_like(flows[1]))
    torch.testing.assert_close(flows[3], torch.ones_like(flows[3]))
    assert cache.latent_state is not None
    torch.testing.assert_close(
        cache.latent_state[:, :, :1],
        torch.full((1, 1, 1, 1, 1), 7.0),
    )
    torch.testing.assert_close(cache.latent_state[:, :, 1:4], cleans[0][:, :, 1:])
    torch.testing.assert_close(cache.latent_state[:, :, 4:7], cleans[1])
    torch.testing.assert_close(cache.latent_state[:, :, 7:10], cleans[2])
    torch.testing.assert_close(cache.latent_state[:, :, 10:13], cleans[3])


def test_transformer_predict_flow_applies_cfg_from_conditioning_input() -> None:
    """Keep CFG inside the transformer boundary used by base diffusion."""

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def forward(
            self,
            noisy_latent: torch.Tensor,
            timestep: torch.Tensor,
            prompt_embeds: torch.Tensor,
            *,
            mask: torch.Tensor,
            camera_conditions: torch.Tensor,
            chunk_plucker: torch.Tensor,
            **_kwargs: object,
        ) -> torch.Tensor:
            assert "negative_mask" not in _kwargs
            self.calls.append(
                {
                    "noisy_latent": noisy_latent,
                    "timestep": timestep,
                    "prompt_embeds": prompt_embeds,
                    "mask": mask,
                    "camera_conditions": camera_conditions,
                    "chunk_plucker": chunk_plucker,
                    "data_info": _kwargs["data_info"],
                }
            )
            branch_values = (
                prompt_embeds.flatten(1)
                .mean(dim=1)
                .reshape(
                    -1,
                    1,
                    1,
                    1,
                    1,
                )
            )
            return torch.ones_like(noisy_latent) * branch_values

    transformer = SanaWMTransformerConfig().setup()
    dummy_model = DummyModel()
    transformer.model = dummy_model
    transformer._model_built = True
    cond_mask = torch.ones((1, 1))
    neg_mask = torch.zeros((1, 1))
    camera = torch.zeros((1, 3, 2, 1, 1))
    chunk_plucker = torch.ones((1, 6, 2, 1, 1))
    conditioning = SanaWMStage1Conditioning(
        condition=torch.ones((1, 1, 1, 1)),
        uncondition=torch.zeros((1, 1, 1, 1)),
        model_kwargs={
            "mask": cond_mask,
            "negative_mask": neg_mask,
            "camera_conditions": camera,
            "chunk_plucker": chunk_plucker,
            "data_info": {"condition_frame_info": {0: 0.0}},
        },
        first_latent=torch.empty((1, 1, 1, 1, 1)),
        latent_shape=(1, 1, 2, 1, 1),
        cfg_scale=2.0,
        flow_shift=1.0,
        steps=1,
        seed=0,
    )

    out = transformer.predict_flow(
        noisy_latent=torch.zeros((1, 1, 2, 1, 1)),
        timestep=torch.full((1, 1, 2), 1000.0),
        cache=SanaWMTransformerCache(),
        input=conditioning,
    )

    torch.testing.assert_close(
        out,
        torch.tensor([[[[[0.0]], [[2.0]]]]]),
    )
    assert len(dummy_model.calls) == 1
    call = dummy_model.calls[0]
    assert call["noisy_latent"].shape == (2, 1, 2, 1, 1)
    assert call["timestep"].shape == (2, 1, 2)
    assert call["prompt_embeds"].shape == (2, 1, 1, 1)
    torch.testing.assert_close(call["mask"], torch.cat([neg_mask, cond_mask], dim=0))
    torch.testing.assert_close(call["camera_conditions"], camera)
    torch.testing.assert_close(
        call["chunk_plucker"],
        torch.cat([chunk_plucker, chunk_plucker], dim=0),
    )
    assert call["data_info"] == {"condition_frame_info": {0: 0.0}}


def test_transformer_predict_flow_caches_static_plucker_embedding() -> None:
    """Avoid re-projecting rollout-constant Plucker conditioning each step."""

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_calls = 0
            self.prepare_camera_calls = 0
            self.forward_calls: list[dict[str, object]] = []
            self.camera_cache = object()

        def prepare_plucker_embedding(
            self,
            chunk_plucker: torch.Tensor,
        ) -> torch.Tensor:
            self.prepare_calls += 1
            return chunk_plucker.flatten(2).transpose(1, 2)

        def prepare_camera_projection_cache(
            self,
            camera_conditions: torch.Tensor,
            *,
            frames: int,
            height: int,
            width: int,
        ) -> tuple[torch.Tensor, object]:
            del camera_conditions
            self.prepare_camera_calls += 1
            return torch.empty(frames, height, width), self.camera_cache

        def forward(
            self,
            noisy_latent: torch.Tensor,
            timestep: torch.Tensor,
            prompt_embeds: torch.Tensor,
            *,
            chunk_plucker_emb: torch.Tensor,
            rotary_emb: torch.Tensor,
            camera_cache: object,
            **_kwargs: object,
        ) -> torch.Tensor:
            del timestep, prompt_embeds
            assert "chunk_plucker" not in _kwargs
            self.forward_calls.append(
                {
                    "chunk_plucker_emb": chunk_plucker_emb,
                    "rotary_emb": rotary_emb,
                    "camera_cache": camera_cache,
                }
            )
            return torch.ones_like(noisy_latent)

    transformer = SanaWMTransformerConfig().setup()
    dummy_model = DummyModel()
    transformer.model = dummy_model
    transformer._model_built = True
    chunk_plucker = torch.ones((1, 2, 2, 1, 1))
    conditioning = SanaWMStage1Conditioning(
        condition=torch.ones((1, 1, 1)),
        uncondition=None,
        model_kwargs={
            "chunk_plucker": chunk_plucker,
            "camera_conditions": torch.zeros((1, 2, 20)),
            "data_info": {"condition_frame_info": {0: 0.0}},
        },
        first_latent=torch.empty((1, 1, 1, 1, 1)),
        latent_shape=(1, 1, 2, 1, 1),
        cfg_scale=1.0,
        flow_shift=1.0,
        steps=1,
        seed=0,
    )
    cache = SanaWMTransformerCache()

    for _ in range(2):
        transformer.predict_flow(
            noisy_latent=torch.zeros((1, 1, 2, 1, 1)),
            timestep=torch.full((1, 1, 2), 1000.0),
            cache=cache,
            input=conditioning,
        )

    assert dummy_model.prepare_calls == 1
    assert dummy_model.prepare_camera_calls == 1
    assert "chunk_plucker" not in conditioning.model_kwargs
    assert "camera_cache" in conditioning.model_kwargs
    assert len(dummy_model.forward_calls) == 2
    cached = conditioning.model_kwargs["chunk_plucker_emb"]
    assert isinstance(cached, torch.Tensor)
    assert all(
        call["chunk_plucker_emb"] is cached for call in dummy_model.forward_calls
    )
    assert all(
        call["camera_cache"] is dummy_model.camera_cache
        for call in dummy_model.forward_calls
    )


def test_inference_config_loads_yaml(
    tmp_path: Path,
) -> None:
    """Parse SANA-WM YAML with local config objects."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  model: SanaMSVideoCamCtrl_1600M_P1_D20
  mixed_precision: bf16
  chunk_split_strategy: first_chunk_plus_one
vae:
  vae_type: LTX2VAE_diffusers
  vae_pretrained: hf://example/model
  weight_dtype: bfloat16
  vae_latent_dim: 128
  vae_stride: [8, 32, 32]
text_encoder:
  text_encoder_name: gemma-2-2b-it
  model_max_length: 300
  chi_prompt: ["prefix"]
scheduler:
  flow_shift: 9.95
  inference_flow_shift: 9.8
""",
        encoding="utf-8",
    )

    cfg = _load_inference_config(str(config_path))

    assert cfg.model.model == "SanaMSVideoCamCtrl_1600M_P1_D20"
    assert cfg.model.get("missing", "fallback") == "fallback"
    assert cfg.vae.vae_stride == [8, 32, 32]
    assert cfg.text_encoder.chi_prompt == ["prefix"]
    assert cfg.scheduler.inference_flow_shift == 9.8
    assert cfg.work_dir == ""


def test_inference_config_loads_builtin_streaming_config() -> None:
    """Provide FlashDreams-owned config data for the streaming release."""
    cfg = _load_inference_config(SANA_WM_STREAMING_CONFIG_PATH)

    assert cfg.work_dir == ""
    assert cfg.model.model == "SanaMSVideoCamCtrlStreaming_1600M_P1_D20"
    assert cfg.model.chunk_size == 3
    assert cfg.vae.vae_type == "LTX2VAE_diffusers_causal"
    assert cfg.vae.vae_pretrained == SANA_WM_STREAMING_CAUSAL_VAE_ROOT
    assert cfg.scheduler.inference_flow_shift == 8.0


def test_text_prompt_encoder_outputs_padded_prompt_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encode prompts behind the explicit text component boundary."""

    class DummyTokens:
        def __init__(self, max_length: int) -> None:
            self.input_ids = torch.arange(max_length).reshape(1, max_length)
            self.attention_mask = torch.ones((1, max_length), dtype=torch.long)

        def to(self, _device: torch.device) -> "DummyTokens":
            return self

    class DummyTokenizer:
        def encode(self, _text: str) -> list[int]:
            return [1, 2]

        def __call__(
            self,
            *_args: object,
            max_length: int,
            **_kwargs: object,
        ) -> DummyTokens:
            return DummyTokens(max_length)

    class DummyTextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            del attention_mask
            hidden = torch.ones((1, input_ids.shape[1], 12), dtype=torch.float32)
            return (hidden,)

    monkeypatch.setenv("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8")
    monkeypatch.setattr(
        conditioning_module,
        "_load_inference_config",
        lambda _path: SimpleNamespace(
            model=SimpleNamespace(mixed_precision="bf16"),
            text_encoder=SimpleNamespace(
                text_encoder_name="T5-small",
                model_max_length=5,
                chi_prompt=["prefix"],
            ),
        ),
    )
    monkeypatch.setattr(
        conditioning_module,
        "_get_tokenizer_and_text_encoder",
        lambda **_kwargs: (DummyTokenizer(), DummyTextEncoder()),
    )
    encoder = SanaWMTextPromptEncoderConfig(
        config_path="dummy.yaml",
        stage1_precision="fp4",
    ).setup()

    encoded = encoder(
        SanaWMTextPromptRequest(
            prompt="drive forward",
            negative_prompt="low quality",
        )
    )

    assert encoded.condition.shape == (1, 1, 8, 12)
    assert encoded.condition_mask.shape == (1, 8)
    assert encoded.negative.shape == (1, 1, 8, 12)
    assert encoded.negative_mask.shape == (1, 8)


def test_first_frame_encoder_outputs_latent_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encode the first frame behind the named I2V component."""
    from PIL import Image

    class DummyLatentDist:
        def mode(self) -> torch.Tensor:
            return torch.ones((1, 4, 1, 2, 3), dtype=torch.float32)

    class DummyPosterior:
        latent_dist = DummyLatentDist()

    class DummyVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0))
            self.latents_mean = torch.zeros(4)
            self.latents_std = torch.ones(4)
            self.config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, _images: torch.Tensor) -> DummyPosterior:
            return DummyPosterior()

    monkeypatch.setattr(
        conditioning_module,
        "_load_inference_config",
        lambda _path: SimpleNamespace(
            model=SimpleNamespace(mixed_precision="bf16"),
            vae=SimpleNamespace(
                weight_dtype="float32",
                vae_type="LTX2VAE_diffusers",
                vae_pretrained="local",
            ),
        ),
    )
    monkeypatch.setattr(
        conditioning_module,
        "_get_vae",
        lambda *_args, **_kwargs: DummyVAE(),
    )
    encoder = SanaWMFirstFrameEncoderConfig(config_path="dummy.yaml").setup()

    latent = encoder(Image.new("RGB", (4, 4)))

    assert latent.shape == (1, 4, 1, 2, 3)
    assert latent.dtype == torch.bfloat16


def test_camera_conditioning_encoder_outputs_sana_schema() -> None:
    """Build raymap/chunk-Plucker tensors behind the camera component."""
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (17, 4, 4)).copy()
    intrinsics = np.broadcast_to(
        np.array([900.0, 900.0, 640.0, 352.0], dtype=np.float32),
        (17, 4),
    ).copy()
    encoder = SanaWMCameraConditioningEncoderConfig().setup()

    camera = encoder(
        SanaWMCameraRequest(
            poses_c2w=poses,
            intrinsics_vec4=intrinsics,
        )
    )

    assert camera["raymap"].shape == (3, 20)
    assert camera["chunk_plucker"].shape == (48, 3, 22, 40)


def test_streaming_chunk_boundaries_use_sink_plus_fixed_blocks() -> None:
    """Represent streaming AR chunks as sink+block, then block-only steps."""
    assert streaming_chunk_boundaries(total_frames=10, chunk_size=3) == (
        0,
        4,
        7,
        10,
    )
    with pytest.raises(ValueError, match="more than one latent frame"):
        streaming_chunk_boundaries(total_frames=1, chunk_size=3)
    with pytest.raises(ValueError, match="divide the chunk size"):
        streaming_chunk_boundaries(total_frames=9, chunk_size=3)


def test_video_decoder_returns_structured_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode Stage-1 latents through the explicit decoder component."""
    decoder = SanaWMVideoDecoderConfig(refiner=None).setup()
    monkeypatch.setattr(
        decoder.vae_decoder,
        "decode_latents",
        lambda _latents: np.zeros((1, 2, 2, 3), dtype=np.uint8),
    )

    decoded = decoder(
        torch.zeros((1, 4, 1, 1, 1)),
        autoregressive_index=0,
        cache=None,
    )

    assert isinstance(decoded, SanaWMDecodedVideo)
    assert decoded.video_hwc.shape == (1, 2, 2, 3)
    assert decoded.stage1_video_hwc is None


def test_streaming_video_decoder_emits_only_new_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode chunks with context while returning only the active frames."""
    decoder = SanaWMStreamingVideoDecoderConfig(
        vae_decoder=SanaWMStreamingLTX2VAEDecoderConfig(
            use_streaming_decode_cache=False,
        ),
        refiner=None,
    ).setup()

    def decode_with_context(latents: torch.Tensor) -> np.ndarray:
        frames = 1 + (latents.shape[2] - 1) * 8
        values = np.arange(frames, dtype=np.uint8)
        return np.broadcast_to(values[:, None, None, None], (frames, 2, 2, 3)).copy()

    monkeypatch.setattr(decoder.vae_decoder, "decode_latents", decode_with_context)
    cache = decoder.initialize_autoregressive_cache()

    first = decoder(
        torch.zeros((1, 4, 4, 1, 1)),
        autoregressive_index=0,
        cache=cache,
    )
    second = decoder(
        torch.zeros((1, 4, 3, 1, 1)),
        autoregressive_index=1,
        cache=cache,
    )

    assert first.video_hwc.shape == (24, 2, 2, 3)
    assert second.video_hwc.shape == (24, 2, 2, 3)
    assert first.video_hwc[0, 0, 0, 0] == 1
    assert second.video_hwc[0, 0, 0, 0] == 25
    assert cache.stage1_sink is not None
    assert cache.stage1_sink.shape[2] == 1
    assert cache.stage1_chunks[0].shape[2] == 3
    assert cache.stage1_chunks[1].shape[2] == 3
    assert cache.refined_chunks[0].shape[2] == 3
    assert cache.refined_chunks[1].shape[2] == 3
    assert first.stage1_video_hwc is None
    assert second.stage1_video_hwc is None


def test_streaming_video_decoder_uses_causal_vae_chunk_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode only active latents after the causal VAE cache is initialized."""
    decoder = SanaWMStreamingVideoDecoderConfig(refiner=None).setup()
    calls: list[tuple[torch.Tensor, bool]] = []

    def decode_chunk(latents: torch.Tensor, *, reset_cache: bool) -> np.ndarray | None:
        calls.append((latents.clone(), reset_cache))
        frames = 1 + (latents.shape[2] - 1) * 8 if reset_cache else latents.shape[2] * 8
        value = 10 if reset_cache else 20
        return np.full((frames, 2, 2, 3), value, dtype=np.uint8)

    monkeypatch.setattr(decoder.vae_decoder, "decode_streaming_chunk", decode_chunk)
    cache = decoder.initialize_autoregressive_cache()

    first = decoder(
        torch.zeros((1, 4, 4, 1, 1)),
        autoregressive_index=0,
        cache=cache,
    )
    second = decoder(
        torch.zeros((1, 4, 3, 1, 1)),
        autoregressive_index=1,
        cache=cache,
    )

    assert [tuple(call[0].shape) for call in calls] == [
        (1, 4, 4, 1, 1),
        (1, 4, 3, 1, 1),
    ]
    assert [call[1] for call in calls] == [True, False]
    assert first.video_hwc.shape == (24, 2, 2, 3)
    assert second.video_hwc.shape == (24, 2, 2, 3)
    assert first.video_hwc[0, 0, 0, 0] == 10
    assert second.video_hwc[0, 0, 0, 0] == 20
    assert cache.vae_streaming_cache_ready is True


def test_streaming_video_decoder_refines_against_rolling_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use prior refined chunks as clean context for later streaming chunks."""
    decoder = SanaWMStreamingVideoDecoderConfig(
        vae_decoder=SanaWMStreamingLTX2VAEDecoderConfig(
            use_streaming_decode_cache=False,
        ),
        refiner=None,
    ).setup()
    calls: list[torch.Tensor] = []

    class DummyRefiner:
        def refine_chunk(
            self,
            *,
            context_latents: torch.Tensor,
            active_latents: torch.Tensor,
            prompt: str,
            prompt_embeds: torch.Tensor | None,
            prompt_attention_mask: torch.Tensor | None,
            fps: int,
            generator: torch.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del prompt, prompt_embeds, prompt_attention_mask, fps, generator
            calls.append(context_latents.clone())
            return active_latents + 1, torch.ones((1, 1, 1)), torch.ones((1, 1))

    def decode_with_context(latents: torch.Tensor) -> np.ndarray:
        frames = 1 + (latents.shape[2] - 1) * 8
        return np.zeros((frames, 2, 2, 3), dtype=np.uint8)

    decoder.refiner = DummyRefiner()
    monkeypatch.setattr(decoder.vae_decoder, "decode_latents", decode_with_context)
    cache = decoder.initialize_autoregressive_cache(
        prompt="demo",
        refiner_kv_max_frames=11,
    )

    decoder(torch.zeros((1, 1, 4, 1, 1)), autoregressive_index=0, cache=cache)
    decoder(torch.full((1, 1, 3, 1, 1), 2.0), autoregressive_index=1, cache=cache)
    decoder(torch.full((1, 1, 3, 1, 1), 4.0), autoregressive_index=2, cache=cache)
    decoder(torch.full((1, 1, 3, 1, 1), 6.0), autoregressive_index=3, cache=cache)

    assert [call.shape[2] for call in calls] == [1, 4, 7, 8]
    torch.testing.assert_close(calls[0], torch.zeros((1, 1, 1, 1, 1)))
    torch.testing.assert_close(
        calls[1],
        torch.cat(
            [
                torch.zeros((1, 1, 1, 1, 1)),
                torch.ones((1, 1, 3, 1, 1)),
            ],
            dim=2,
        ),
    )
    torch.testing.assert_close(
        calls[3],
        torch.tensor([0.0, 1.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0]).reshape(1, 1, 8, 1, 1),
    )
    assert cache.refiner_prompt_embeds is not None
    assert cache.refiner_prompt_attention_mask is not None


def test_stage1_model_matches_checkpoint_schema() -> None:
    """Pin the Stage-1 module to the public checkpoint schema."""
    state = SanaWMStage1Model().state_dict()

    assert SANA_WM_STAGE1_SPEC.chunk_size is None
    assert SANA_WM_STAGE1_SPEC.chunk_split_strategy == "first_chunk_plus_one"
    assert SANA_WM_STREAMING_STAGE1_SPEC.chunk_size == 3
    assert SANA_WM_STREAMING_STAGE1_SPEC.chunk_split_strategy == (
        "first_chunk_plus_one"
    )
    assert SANA_WM_STREAMING_STAGE1_SPEC.depth == SANA_WM_STAGE1_SPEC.depth
    assert len(state) == 872
    assert tuple(state["x_embedder.proj.weight"].shape) == (2240, 128, 1, 1, 1)
    assert tuple(state["raymap_embedder.proj.weight"].shape) == (2240, 3, 1, 1, 1)
    assert tuple(state["plucker_embedder.proj.weight"].shape) == (
        2240,
        48,
        1,
        1,
        1,
    )
    assert tuple(state["pos_embed"].shape) == (1, 484, 2240)
    assert tuple(state["y_embedder.y_embedding"].shape) == (300, 2304)
    assert tuple(state["blocks.0.attn.qkv.weight"].shape) == (6720, 2240)
    assert tuple(state["blocks.0.attn.conv_k.weight"].shape) == (2240, 1, 4)
    assert tuple(state["blocks.0.cross_attn.kv_linear.weight"].shape) == (
        4480,
        2240,
    )
    assert tuple(state["blocks.19.mlp.t_conv.weight"].shape) == (2240, 2240, 3, 1)
    assert tuple(state["final_layer.linear.weight"].shape) == (128, 2240)

    for block_index in range(SANA_WM_STAGE1_SPEC.depth):
        has_gdn_conv = f"blocks.{block_index}.attn.conv_k.weight" in state
        assert has_gdn_conv is SANA_WM_STAGE1_SPEC.block_uses_gdn(block_index)


def test_stage1_forward_preserves_latent_shape() -> None:
    """Exercise the Stage-1 forward path on a small CPU-safe spec."""
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=2,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=2,
    )
    model = SanaWMStage1Model(spec)
    latents = torch.randn(1, 4, 3, 2, 2)
    timesteps = torch.ones(1, 1, 3)
    text = torch.randn(1, 1, 5, 12)
    mask = torch.ones(1, 5)
    plucker = torch.randn(1, 6, 3, 2, 2)

    out = model(latents, timesteps, text, mask=mask, chunk_plucker=plucker)

    assert out.shape == latents.shape


def test_stage1_forward_accepts_precomputed_camera_cache() -> None:
    """Precomputed camera cache must match the raw camera-conditioning path."""
    torch.manual_seed(0)
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=1,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=1,
    )
    model = SanaWMStage1Model(spec).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(mean=0.0, std=0.02)
    latents = torch.randn(1, 4, 3, 2, 2)
    timesteps = torch.ones(1, 1, 3)
    text = torch.randn(1, 1, 5, 12)
    mask = torch.ones(1, 5)
    plucker = torch.randn(1, 6, 3, 2, 2)
    camera = torch.zeros(1, 3, 20)
    camera[..., :16] = torch.eye(4).flatten()
    camera[..., 16:] = torch.tensor([1.0, 1.0, 0.5, 0.5])
    rotary_emb, camera_cache = model.prepare_camera_projection_cache(
        camera,
        frames=3,
        height=2,
        width=2,
    )

    raw = model(
        latents,
        timesteps,
        text,
        mask=mask,
        chunk_plucker=plucker,
        camera_conditions=camera,
    )
    cached = model(
        latents,
        timesteps,
        text,
        mask=mask,
        chunk_plucker=plucker,
        camera_conditions=camera,
        rotary_emb=rotary_emb,
        camera_cache=camera_cache,
    )

    torch.testing.assert_close(cached, raw)


def test_stage1_self_attention_uses_camera_conditions() -> None:
    """Changing camera conditions must affect camera attention."""
    torch.manual_seed(0)
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=1,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=2,
    )
    attn = Stage1SelfAttention(spec, use_gdn_convs=True).eval()
    hidden = torch.randn(1, 8, 16)
    camera = torch.zeros(1, 2, 20)
    camera[..., :16] = torch.eye(4).flatten()
    camera[..., 16:] = torch.tensor([1.0, 1.0, 0.5, 0.5])
    shifted_camera = camera.clone()
    shifted_camera[..., 3] = 0.25

    base = attn(hidden, HW=(2, 2, 2), camera_conditions=camera)
    shifted = attn(hidden, HW=(2, 2, 2), camera_conditions=shifted_camera)

    assert base.shape == hidden.shape
    assert not torch.allclose(base, shifted)


def test_stage1_self_attention_broadcasts_single_camera_batch() -> None:
    """CFG can reuse one camera projection cache for both prompt branches."""
    torch.manual_seed(1)
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=1,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=2,
    )
    attn = Stage1SelfAttention(spec, use_gdn_convs=True).eval()
    hidden = torch.randn(2, 8, 16)
    camera = torch.zeros(1, 2, 20)
    camera[..., :16] = torch.eye(4).flatten()
    camera[..., 16:] = torch.tensor([1.0, 1.0, 0.5, 0.5])

    broadcast = attn(hidden, HW=(2, 2, 2), camera_conditions=camera)
    duplicated = attn(
        hidden,
        HW=(2, 2, 2),
        camera_conditions=torch.cat([camera, camera], dim=0),
    )

    assert broadcast.shape == hidden.shape
    torch.testing.assert_close(broadcast, duplicated)


def test_stage1_self_attention_uses_gdn_main_without_gdn_convs() -> None:
    """Blocks without GDN conv weights still route main attention through GDN."""
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=1,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=1,
    )
    attn = Stage1SelfAttention(spec, use_gdn_convs=False).eval()
    hidden = torch.randn(1, 4, 16)
    called = {"gdn_main": False}

    def fake_gdn_main(
        x: torch.Tensor,
        *,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        del HW, rotary_emb
        called["gdn_main"] = True
        assert precomputed_gates[0].shape == (1, 2, 1, 4)
        return torch.zeros_like(x)

    def fail_softmax_main(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("main attention should not use softmax")

    setattr(attn, "_forward_gdn_main", fake_gdn_main)
    setattr(attn, "_forward_softmax_main", fail_softmax_main)

    out = attn(hidden, HW=(1, 2, 2), apply_output_gate=False)

    assert called["gdn_main"]
    torch.testing.assert_close(out, torch.zeros_like(hidden))


def test_transformer_releases_stage1_runtime() -> None:
    """Free Stage-1-only modules and conditioning before decode/refine."""
    transformer = SanaWMTransformerConfig().setup()
    transformer.model = torch.nn.Linear(1, 1)
    transformer._model_built = True
    transformer._stage1_quantized = True
    cache = SanaWMTransformerCache(
        conditioning=SanaWMStage1Conditioning(
            condition=torch.empty(1),
            uncondition=None,
            model_kwargs={},
            first_latent=torch.empty(1),
            latent_shape=(1, 1, 1, 1, 1),
            cfg_scale=1.0,
            flow_shift=1.0,
            steps=1,
            seed=0,
        )
    )

    transformer.release_stage1_runtime(cache)

    assert cache.conditioning is None
    assert transformer.model is None
    assert transformer._model_built is False
    assert transformer._stage1_quantized is False


def _stage1_conditioning_cache() -> SanaWMTransformerCache:
    return SanaWMTransformerCache(
        conditioning=SanaWMStage1Conditioning(
            condition=torch.empty(1),
            uncondition=None,
            model_kwargs={},
            first_latent=torch.empty(1),
            latent_shape=(1, 1, 1, 1, 1),
            cfg_scale=1.0,
            flow_shift=1.0,
            steps=1,
            seed=0,
        )
    )


def test_postprocess_keeps_stage1_resident_by_default() -> None:
    """Default keeps the Stage-1 DiT resident so a reused pipeline never reloads."""
    transformer = SanaWMTransformerConfig().setup()
    assert transformer.config.offload_stage1 is False
    model = torch.nn.Linear(1, 1)
    transformer.model = model
    transformer._model_built = True
    transformer._stage1_quantized = True
    cache = _stage1_conditioning_cache()
    latent = torch.zeros(1)

    result = transformer.postprocess_clean_latent(latent, cache)

    assert result is latent
    assert transformer.model is model
    assert transformer._model_built is True
    assert transformer._stage1_quantized is True
    # Per-rollout conditioning is still dropped even when the model stays warm.
    assert cache.conditioning is None


def test_postprocess_releases_stage1_when_offloading() -> None:
    """offload_stage1 restores the free-before-decode behavior."""
    transformer = SanaWMTransformerConfig(offload_stage1=True).setup()
    transformer.model = torch.nn.Linear(1, 1)
    transformer._model_built = True
    transformer._stage1_quantized = True
    cache = _stage1_conditioning_cache()

    transformer.postprocess_clean_latent(torch.zeros(1), cache)

    assert transformer.model is None
    assert transformer._model_built is False
    assert cache.conditioning is None


def test_vae_tiling_defaults_match_upstream_fast_path() -> None:
    """Keep VAE decode on the upstream-style fast tiles by default."""

    class DummyVAE:
        def __init__(self) -> None:
            self.calls: list[dict[str, int]] = []
            self.tile_sample_min_height = 512
            self.tile_sample_stride_height = 448
            self.tile_sample_min_width = 512
            self.tile_sample_stride_width = 448
            self.tile_sample_min_num_frames = 96
            self.tile_sample_stride_num_frames = 64
            self.use_framewise_encoding = False
            self.use_framewise_decoding = False
            self.spatial_compression_ratio = 32

        def enable_tiling(self, **kwargs: int) -> None:
            self.calls.append(kwargs)

    decoder = SanaWMLTX2VAEDecoderConfig().setup()
    decoder.vae = DummyVAE()

    decoder._configure_vae_tiling()

    assert decoder.vae.calls == [
        {
            "tile_sample_min_height": 512,
            "tile_sample_stride_height": 448,
            "tile_sample_min_width": 512,
            "tile_sample_stride_width": 448,
            "tile_sample_min_num_frames": 96,
            "tile_sample_stride_num_frames": 64,
        }
    ]
    assert decoder.vae.tile_sample_min_height == 512
    assert decoder.vae.tile_sample_stride_height == 448
    assert decoder.vae.tile_sample_min_width == 512
    assert decoder.vae.tile_sample_stride_width == 448
    assert decoder.vae.tile_sample_min_num_frames == 96
    assert decoder.vae.tile_sample_stride_num_frames == 64
    assert decoder.vae.use_framewise_encoding is True
    assert decoder.vae.use_framewise_decoding is True


def test_decode_retries_vae_oom_with_smaller_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry VAE decode with smaller tiles after a CUDA OOM."""

    class DummyVAE:
        def __init__(self) -> None:
            self.tile_sample_min_height = 256
            self.tile_sample_stride_height = 224
            self.tile_sample_min_width = 256
            self.tile_sample_stride_width = 224
            self.tile_sample_min_num_frames = 24
            self.tile_sample_stride_num_frames = 8
            self.use_framewise_encoding = False
            self.use_framewise_decoding = False
            self.spatial_compression_ratio = 32
            self.temporal_compression_ratio = 8

        def to(self, *_args: object, **_kwargs: object) -> "DummyVAE":
            return self

        def enable_tiling(self, **kwargs: int) -> None:
            for name, value in kwargs.items():
                setattr(self, name, value)

    decoder = SanaWMLTX2VAEDecoderConfig().setup()
    decoder.vae = DummyVAE()
    decoder.vae_dtype = torch.float32
    monkeypatch.setattr(decoder, "_ensure_vae", lambda: None)
    calls = 0
    inference_modes: list[bool] = []

    def decode_once(_samples: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        inference_modes.append(torch.is_inference_mode_enabled())
        if calls == 1:
            raise torch.OutOfMemoryError("test OOM")
        return torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(decoder, "_vae_decode", decode_once)

    video = decoder.decode_latents(torch.zeros((1, 4, 1, 1, 1)))

    assert calls == 2
    assert inference_modes == [True, True]
    assert video.shape == (1, 2, 2, 3)
    assert decoder.vae.tile_sample_min_height == 128
    assert decoder.vae.tile_sample_stride_height == 64
    assert decoder.vae.tile_sample_min_width == 128
    assert decoder.vae.tile_sample_stride_width == 64
    assert decoder.vae.tile_sample_min_num_frames == 16
    assert decoder.vae.tile_sample_stride_num_frames == 8


def test_vae_tiling_avoids_degenerate_latent_tails() -> None:
    """Avoid last spatial VAE tiles with size one after compression."""
    assert (
        _avoid_degenerate_tile_tail(
            sample_extent=704,
            sample_tile_min=256,
            sample_stride=224,
            compression_ratio=32,
        )
        == 192
    )
    assert (
        _avoid_degenerate_tile_tail(
            sample_extent=1280,
            sample_tile_min=128,
            sample_stride=112,
            compression_ratio=32,
        )
        == 64
    )


def test_resolve_hf_path_preloads_on_rank0_then_reads_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route hf:// resolution through the shared rank-0 download helper."""
    preloads: list[tuple[str, object]] = []
    snapshots: list[dict[str, object]] = []

    def _preload(repo_id: str, **kwargs: object) -> None:
        preloads.append((repo_id, kwargs.get("allow_patterns")))

    def _snapshot_download(**kwargs: object) -> str:
        snapshots.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(tools_module, "maybe_download_hf_repo_on_rank0", _preload)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _snapshot_download,
    )

    resolved = tools_module.resolve_hf_path(SANA_WM_STREAMING_REFINER_ROOT)

    assert resolved == str(tmp_path / "snapshot" / "refiner_diffusers")
    assert preloads == [
        (
            SANA_WM_STREAMING_HF_REPO,
            [
                "refiner_diffusers",
                "refiner_diffusers/*",
                "refiner_diffusers/**",
            ],
        )
    ]
    # The download itself is the helper's job; resolution only reads the cache.
    assert snapshots[0]["local_files_only"] is True


def test_resolve_hf_path_leaves_local_paths_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep local roots download-free."""

    def _fail_preload(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local paths must not trigger a download")

    monkeypatch.setattr(tools_module, "maybe_download_hf_repo_on_rank0", _fail_preload)

    assert tools_module.resolve_hf_path(str(tmp_path)) == str(tmp_path)


def test_hf_defaults_point_at_bidirectional_release() -> None:
    """Pin every default SANA-WM artefact to the bidirectional HF repo."""
    assert SANA_WM_HF_REPO == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert SANA_WM_MODEL_PATH == (
        "https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional/"
        "resolve/main/dit/sana_wm_1600m_720p.safetensors"
    )
    assert SANA_WM_CONFIG_PATH == (
        "hf://Efficient-Large-Model/SANA-WM_bidirectional/config.yaml"
    )


def test_hf_defaults_point_at_streaming_release() -> None:
    """Pin every default streaming artefact to the streaming HF repo."""
    assert SANA_WM_STREAMING_HF_REPO == "Efficient-Large-Model/SANA-WM_streaming"
    assert SANA_WM_STREAMING_MODEL_PATH == (
        "https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming/"
        "resolve/main/sana_dit/model.pt"
    )
    assert SANA_WM_STREAMING_CONFIG_PATH == (
        "flashdreams://sana-wm-streaming-1600m-720p"
    )
    assert SANA_WM_STREAMING_CAUSAL_VAE_ROOT == (
        "hf://Efficient-Large-Model/SANA-WM_streaming/ltx2_causal_vae"
    )
    assert SANA_WM_STREAMING_REFINER_ROOT == (
        "hf://Efficient-Large-Model/SANA-WM_streaming/refiner_diffusers"
    )
    assert SANA_WM_STREAMING_REFINER_GEMMA_ROOT == (
        "hf://Efficient-Large-Model/SANA-WM_streaming/gemma3_12b"
    )
    assert SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES == 11


def test_runner_setup_preserves_cli_fields() -> None:
    """Construct the runner and preserve CLI override fields."""
    cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        image_path=Path("missing.png"),
        prompt="demo",
    )

    runner = cfg.setup()

    assert isinstance(runner, SanaWMRunner)
    assert runner.config.image_path == Path("missing.png")


def test_streaming_runner_setup_preserves_cli_fields() -> None:
    """Construct the streaming runner and preserve CLI override fields."""
    cfg = derive_config(
        RUNNER_SANA_WM_STREAMING,
        image_path=Path("missing.png"),
        prompt="demo",
        num_frame_per_block=5,
    )

    runner = cfg.setup()

    assert isinstance(runner, SanaWMStreamingRunner)
    assert runner.config.image_path == Path("missing.png")
    assert runner.config.num_frame_per_block == 5


def test_runner_derives_intrinsics_when_omitted(tmp_path: Path) -> None:
    """Omitting --intrinsics-path derives per-frame intrinsics from the frame."""
    from PIL import Image

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(10, 20, 30)).save(image_path)
    cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        image_path=image_path,
        prompt="demo",
        intrinsics_path=None,
        action="w-4",
        num_frames=25,
    )
    runner = cfg.setup()

    _image, c2w, intrinsics_vec4, num_frames = runner._prepare_inputs()

    assert num_frames == 25
    assert c2w.shape[0] == num_frames
    assert intrinsics_vec4.shape == (num_frames, 4)
    # Derived focal lengths are finite and positive after the crop transform.
    assert np.all(np.isfinite(intrinsics_vec4))
    assert np.all(intrinsics_vec4[:, :2] > 0)


def test_runner_fits_camera_path_to_requested_frames(tmp_path: Path) -> None:
    """Do not let short explicit trajectories cap the output frame count."""
    from PIL import Image

    image_path = tmp_path / "frame.png"
    camera_path = tmp_path / "camera.npy"
    Image.new("RGB", (640, 360), color=(10, 20, 30)).save(image_path)
    c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    c2w[:, 2, 3] = [0.0, 1.0, 2.0]
    np.save(camera_path, c2w)
    cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        image_path=image_path,
        prompt="demo",
        intrinsics_path=None,
        camera_path=camera_path,
        num_frames=25,
    )
    runner = cfg.setup()

    _image, fitted_c2w, intrinsics_vec4, num_frames = runner._prepare_inputs()

    assert num_frames == 25
    assert fitted_c2w.shape[0] == num_frames
    assert intrinsics_vec4.shape == (num_frames, 4)
    np.testing.assert_allclose(fitted_c2w[[0, -1], 2, 3], [0.0, 2.0])


def test_runner_config_type() -> None:
    """Keep the exported literal on the SANA-WM runner config subclass."""
    assert isinstance(RUNNER_SANA_WM_BIDIRECTIONAL, SanaWMRunnerConfig)
    assert isinstance(RUNNER_SANA_WM_STREAMING, SanaWMStreamingRunnerConfig)


def test_runner_defaults_to_bf16_precision() -> None:
    """Keep the default runner on BF16 precision."""
    assert RUNNER_SANA_WM_BIDIRECTIONAL.stage1_precision == "bf16"
    assert RUNNER_SANA_WM_BIDIRECTIONAL.refiner_precision == "bf16"
    assert RUNNER_SANA_WM_BIDIRECTIONAL.quant_backend == "auto"
    assert RUNNER_SANA_WM_BIDIRECTIONAL.no_refiner is False
    assert RUNNER_SANA_WM_STREAMING.stage1_precision == "bf16"
    assert RUNNER_SANA_WM_STREAMING.refiner_precision == "bf16"
    assert RUNNER_SANA_WM_STREAMING.quant_backend == "auto"
    assert RUNNER_SANA_WM_STREAMING.no_refiner is False
    assert RUNNER_SANA_WM_STREAMING.num_frame_per_block == 3
    assert RUNNER_SANA_WM_STREAMING.denoising_step_list == (
        DEFAULT_STREAMING_DENOISING_STEP_LIST
    )


def test_stage1_quant_scope_matches_upstream_precision_cli() -> None:
    """Keep FP8 and FP4 on upstream's self-attn + cross-attn + FFN scope."""
    patterns = _stage1_quant_include_patterns()

    assert patterns is not None
    assert r"^blocks\.\d+\.attn\.qkv$" in patterns
    assert r"^blocks\.\d+\.attn\.output_gate$" in patterns
    assert r"^blocks\.\d+\.cross_attn\." in patterns
    assert r"^blocks\.\d+\.mlp\.inverted_conv\.linear$" in patterns
    assert r"^blocks\.\d+\.mlp\.point_conv\.linear$" in patterns


def test_stage1_ffn_linearization_preserves_forward() -> None:
    """Expose pointwise FFN convs as Linear modules without changing math."""
    torch.manual_seed(0)
    spec = SanaWMStage1Spec(
        latent_channels=4,
        hidden_size=16,
        text_dim=12,
        timestep_dim=8,
        depth=1,
        num_heads=2,
        head_dim=8,
        max_text_length=5,
        latent_grid_size=(2, 2),
        mlp_ratio=1,
        conv_kernel_size=3,
        temporal_kernel_size=3,
        plucker_channels=6,
        raymap_channels=3,
        softmax_every_n=2,
    )
    mlp = GLUMBConvTemp(spec).eval()
    inputs = torch.randn(1, 8, 16)
    reference = mlp(inputs, frames=2, height=2, width=2)

    converted, skipped = linearize_stage1_ffn_for_quant(mlp)
    output = mlp(inputs, frames=2, height=2, width=2)

    assert converted == 2
    assert skipped == 0
    assert hasattr(mlp.inverted_conv, "linear")
    assert hasattr(mlp.point_conv, "linear")
    torch.testing.assert_close(output, reference)


def test_fp4_rht16_is_orthogonal() -> None:
    """The tiled Hadamard rotation must preserve dot products before quantization."""
    inputs = torch.randn(3, 4, 32)
    weights = torch.randn(5, 32)

    rotated_inputs = apply_rht16(inputs)
    rotated_weights = apply_rht16(weights)

    torch.testing.assert_close(
        rotated_inputs.reshape(-1, 32) @ rotated_weights.t(),
        inputs.reshape(-1, 32) @ weights.t(),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_fp4_rht16_rejects_unaligned_last_dimension() -> None:
    """NVFP4 RHT is tiled in groups of sixteen values."""
    with pytest.raises(ValueError, match="divisible by 16"):
        apply_rht16(torch.randn(2, 15))


def test_fp4_global_scale_uses_nvfp4_dynamic_range() -> None:
    """The hierarchical FP4 global scale should target E4M3 x E2M1 range."""
    inputs = torch.tensor([-2688.0, 0.0, 1344.0])

    assert nvfp4_global_scale(inputs).item() == pytest.approx(1.0)


def test_refiner_is_flashdreams_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the refiner adapter."""
    calls: dict[str, object] = {}

    class DummyRefiner:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

    monkeypatch.setattr(refiner_module, "SanaWMLTX2Refiner", DummyRefiner)
    monkeypatch.setattr(
        decoder_module,
        "resolve_hf_path",
        lambda value: f"/resolved/{value}",
    )
    refiner = SanaWMLTX2LatentRefinerConfig().setup()

    refiner._ensure_refiner()

    assert refiner._refiner_built is True
    assert isinstance(refiner.refiner, DummyRefiner)
    assert calls["refiner_root"] == (
        "/resolved/hf://Efficient-Large-Model/SANA-WM_bidirectional/refiner"
    )
    assert calls["gemma_root"] == (
        "/resolved/hf://Efficient-Large-Model/SANA-WM_bidirectional/refiner/text_encoder"
    )
    assert calls["dtype"] is torch.bfloat16
    assert calls["precision"] == "bf16"
    assert calls["quant_backend"] == "torch"


def test_refiner_latent_pack_round_trips() -> None:
    """Preserve LTX-2 latent layout across token packing and unpacking."""
    latents = torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(
        1,
        4,
        3,
        2,
        2,
    )

    packed = _pack_latents(latents, patch_size=1, patch_size_t=1)
    unpacked = _unpack_latents(
        packed,
        num_frames=3,
        height=2,
        width=2,
        patch_size=1,
        patch_size_t=1,
    )

    torch.testing.assert_close(unpacked, latents)


def test_refiner_sink_bidirectional_path_runs_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the supported LTX-2 refiner path explicit and inference-only."""

    class DummyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"patch_size": 1, "patch_size_t": 1})()

    refiner = SanaWMLTX2Refiner.__new__(SanaWMLTX2Refiner)
    torch.nn.Module.__init__(refiner)
    refiner.device = torch.device("cpu")
    refiner.dtype = torch.float32
    refiner.transformer = DummyTransformer()
    inference_modes: list[bool] = []
    monkeypatch.setattr(refiner, "_prepare_quantization", lambda: None)
    monkeypatch.setattr(
        refiner,
        "_encode_prompt",
        lambda _prompt: (torch.zeros((1, 1, 1)), torch.ones((1, 1))),
    )

    def predict_current_x0(
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
        packed_context_tokens: torch.Tensor | None,
        context_cache: object | None,
    ) -> torch.Tensor:
        del sink, prompt_embeds, prompt_attention_mask, sigma, fps, context_cache
        assert packed_context_tokens is not None
        inference_modes.append(torch.is_inference_mode_enabled())
        return torch.zeros_like(_pack_latents(noisy_current))

    monkeypatch.setattr(refiner, "_predict_current_x0", predict_current_x0)
    latents = torch.zeros((1, 2, 3, 1, 1))

    refined = refiner.refine_latents(
        latents,
        "demo",
        fps=16.0,
        progress=False,
        sigmas=(0.5, 0.0),
    )

    assert refined.shape == latents.shape
    assert inference_modes == [True]


def test_auto_quant_backend_resolves_to_torch_backend() -> None:
    """Keep default quantization on the Torch backend."""
    assert _resolve_quant_backend("auto", ["fp4"]) == "torch"
    assert _resolve_quant_backend("auto", ["fp8", "fp4"]) == "torch"
    assert _resolve_quant_backend("auto", []) == "torch"


def test_quantized_precision_requires_cuda() -> None:
    """Reject fp8/fp4 before loading checkpoints on CPU-only devices."""
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _validate_precision_request(
            device=torch.device("cpu"),
            stage1_precision="fp8",
            refiner_precision="bf16",
            refiner_enabled=True,
            quant_backend="torch",
        )


def test_bf16_precision_does_not_require_cuda_or_quant_backend() -> None:
    """Keep BF16 as the TE-free compatibility path."""
    _validate_precision_request(
        device=torch.device("cpu"),
        stage1_precision="bf16",
        refiner_precision="bf16",
        refiner_enabled=True,
        quant_backend="auto",
    )


def test_fp8_precision_requires_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject FP8 on pre-Hopper GPUs before checking quantization backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))

    with pytest.raises(ValueError, match="requires a Hopper or newer GPU"):
        _validate_precision_request(
            device=torch.device("cuda:0"),
            stage1_precision="fp8",
            refiner_precision="bf16",
            refiner_enabled=True,
            quant_backend="torch",
        )


def test_fp4_precision_requires_blackwell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject NVFP4 on Hopper-class GPUs before checking quantization backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    with pytest.raises(ValueError, match="requires a Blackwell GPU"):
        _validate_precision_request(
            device=torch.device("cuda:0"),
            stage1_precision="fp4",
            refiner_precision="bf16",
            refiner_enabled=True,
            quant_backend="torch",
        )


def test_torch_fp8_backend_validates_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow FP8 validation with the required PyTorch primitives."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch, "_scaled_mm", object(), raising=False)
    monkeypatch.setattr(torch, "float8_e4m3fn", torch.uint8, raising=False)

    _validate_precision_request(
        device=torch.device("cuda:0"),
        stage1_precision="fp8",
        refiner_precision="bf16",
        refiner_enabled=True,
        quant_backend="torch-fp8",
    )


def test_auto_fp8_backend_uses_torch_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route default FP8 to the Torch backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch, "_scaled_mm", object(), raising=False)
    monkeypatch.setattr(torch, "float8_e4m3fn", torch.uint8, raising=False)

    _validate_precision_request(
        device=torch.device("cuda:0"),
        stage1_precision="fp8",
        refiner_precision="bf16",
        refiner_enabled=True,
        quant_backend="auto",
    )


def test_runner_reaches_normal_input_validation() -> None:
    """Reach normal file input validation."""
    cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        image_path=Path("missing.png"),
        prompt="demo",
    )
    runner = cfg.setup()

    with pytest.raises(FileNotFoundError, match="missing.png"):
        runner.run()


def test_torch_fp8_backend_rejects_fp4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently route FP4 to a backend without an NVFP4 kernel."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))

    with pytest.raises(ValueError, match="accepts fp8 only"):
        _validate_precision_request(
            device=torch.device("cuda:0"),
            stage1_precision="fp4",
            refiner_precision="bf16",
            refiner_enabled=True,
            quant_backend="torch-fp8",
        )


def test_torch_fp4_backend_validates_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow FP4 validation with the required PyTorch primitives."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(torch, "_scaled_mm", object(), raising=False)
    monkeypatch.setattr(torch, "float4_e2m1fn_x2", torch.uint8, raising=False)
    monkeypatch.setattr(torch, "float8_e4m3fn", torch.uint8, raising=False)

    _validate_precision_request(
        device=torch.device("cuda:0"),
        stage1_precision="fp4",
        refiner_precision="bf16",
        refiner_enabled=True,
        quant_backend="torch-fp4",
    )


def test_torch_fp4_backend_rejects_fp8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep precision-specific backends explicit."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))

    with pytest.raises(ValueError, match="accepts fp4 only"):
        _validate_precision_request(
            device=torch.device("cuda:0"),
            stage1_precision="fp8",
            refiner_precision="bf16",
            refiner_enabled=True,
            quant_backend="torch-fp4",
        )


def test_torch_backend_allows_mixed_fp8_fp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow mixed FP8/FP4 requests on the Torch backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch, "_scaled_mm", object(), raising=False)
    monkeypatch.setattr(torch, "float4_e2m1fn_x2", torch.uint8, raising=False)
    monkeypatch.setattr(torch, "float8_e4m3fn", torch.uint8, raising=False)

    _validate_precision_request(
        device=torch.device("cuda:0"),
        stage1_precision="fp8",
        refiner_precision="fp4",
        refiner_enabled=True,
        quant_backend="torch",
    )


def test_torch_fp8_linear_replaces_matching_modules() -> None:
    """Provide a Torch replacement for eligible Linear modules."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for FP8 replacement")

    module = torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=True),
        torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False)),
        torch.nn.Linear(15, 32, bias=False),
    )

    converted, skipped = replace_linear_with_torch_fp8(
        module,
        recipe=None,
        params_dtype=torch.bfloat16,
        skip_patterns=(),
        include_patterns=("^0$", "^1\\.0$", "^2$"),
    )

    first = module[0]
    nested = cast(torch.nn.Sequential, module[1])
    third = module[2]
    assert converted == 2
    assert skipped == 1
    assert isinstance(first, TorchScaledMMFP8Linear)
    assert isinstance(nested[0], TorchScaledMMFP8Linear)
    assert isinstance(third, torch.nn.Linear)
    assert first.weight.shape == (32, 16)
    assert first.bias is not None


def test_torch_fp4_linear_replaces_matching_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route eligible FP4 modules through the replacement helper."""
    module = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=True),
        torch.nn.Sequential(torch.nn.Linear(32, 64, bias=False)),
        torch.nn.Linear(16, 64, bias=False),
    )

    @classmethod
    def from_linear(
        cls: type[TorchScaledMMFP4Linear],
        source: torch.nn.Linear,
        *,
        out_dtype: torch.dtype,
        use_rht: bool = True,
        use_global_scale: bool = True,
        weight_scale_2d: bool = True,
    ) -> TorchScaledMMFP4Linear:
        del out_dtype, use_rht, use_global_scale, weight_scale_2d
        instance = cls.__new__(cls)
        torch.nn.Module.__init__(instance)
        instance.in_features = source.in_features
        instance.out_features = source.out_features
        return instance

    monkeypatch.setattr(TorchScaledMMFP4Linear, "from_linear", from_linear)

    converted, skipped = replace_linear_with_torch_fp4(
        module,
        recipe=None,
        params_dtype=torch.bfloat16,
        skip_patterns=(),
        include_patterns=("^0$", "^1\\.0$", "^2$"),
    )

    first = module[0]
    nested = cast(torch.nn.Sequential, module[1])
    third = module[2]
    assert converted == 2
    assert skipped == 1
    assert isinstance(first, TorchScaledMMFP4Linear)
    assert isinstance(nested[0], TorchScaledMMFP4Linear)
    assert isinstance(third, torch.nn.Linear)


def test_pyproject_entry_point_matches_runner_literal() -> None:
    """Keep the package entry point aligned with ``RUNNER_CONFIGS``."""
    pyproject = tomllib.loads(
        Path("integrations/sana/pyproject.toml").read_text(encoding="utf-8")
    )
    entry_points = pyproject["project"]["entry-points"][ENTRY_POINT_GROUP]

    assert entry_points == {
        "sana-wm-bidirectional": "sana_wm.config:RUNNER_SANA_WM_BIDIRECTIONAL",
        "sana-wm-streaming": "sana_wm.config:RUNNER_SANA_WM_STREAMING",
    }


def test_pyproject_package_selection() -> None:
    """Keep the package metadata aligned with the integration package."""
    pyproject = tomllib.loads(
        Path("integrations/sana/pyproject.toml").read_text(encoding="utf-8")
    )

    packages = pyproject["tool"]["setuptools"]["packages"]["find"]
    dependencies = set(pyproject["project"]["dependencies"])

    assert "accelerate>=1.0" in dependencies
    assert packages["include"] == ["sana_wm*"]
    assert "diffusers>=0.36" in dependencies
    assert "transformers>=5.0,<6" in dependencies
