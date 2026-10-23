# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the Cosmos Interactive hand V2 pipeline."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, cast

import cosmos3_nano_sim_bimanual.pipeline as pipeline_module
import pytest
import torch
import torch.nn as nn
from cosmos3_nano_sim_bimanual import action_format
from cosmos3_nano_sim_bimanual.config import PIPELINE_COSMOS3_NANO_SIM_BIMANUAL
from cosmos3_nano_sim_bimanual.decoder import HandDecoder, HandDecoderConfig
from cosmos3_nano_sim_bimanual.encoder import (
    HandActionEncoder,
    HandActionEncoderConfig,
    HandSceneEncoder,
)
from cosmos3_nano_sim_bimanual.patchify import unpatchify_latent
from cosmos3_nano_sim_bimanual.pipeline import (
    HandStreamInferencePipeline,
    HandStreamInferencePipelineConfig,
    validate_pipeline_config,
)
from cosmos3_nano_sim_bimanual.spec import SPEC
from cosmos3_nano_sim_bimanual.transformer import (
    HandTransformer,
    HandTransformerConfig,
)
from cosmos3_nano_sim_bimanual.transformer.network import CosmosMoTNetwork
from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)

pytestmark = pytest.mark.ci_cpu


class _FakeCache:
    def __init__(self) -> None:
        self.scene = SimpleNamespace(name="unset")
        self.last_timing = {"model_step_s": 0.125}
        self.waits: list[bool] = []
        self.reset_count = 0
        self.profile = False

    def wait_for_pending_cache_update(self, *, block: bool = False) -> float:
        self.waits.append(block)
        return 0.0

    def reset_rollout(self) -> None:
        self.wait_for_pending_cache_update(block=True)
        self.reset_count += 1


class _FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(2, 4)


class _FakeWanDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.ones(1))
        self.inputs: list[tuple[Tensor, int, object]] = []
        self.caches: list[object] = []

    def initialize_autoregressive_cache(self) -> object:
        cache = object()
        self.caches.append(cache)
        return cache

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int,
        cache: object,
    ) -> Tensor:
        self.inputs.append((input, autoregressive_index, cache))
        frames = 1 if autoregressive_index == 0 else 4
        return input.new_zeros(
            frames,
            3,
            input.shape[-2] * action_format.SPATIAL_COMPRESSION_FACTOR,
            input.shape[-1] * action_format.SPATIAL_COMPRESSION_FACTOR,
        )


def test_config_topology_uses_v2_components() -> None:
    config = PIPELINE_COSMOS3_NANO_SIM_BIMANUAL

    assert config.name == "cosmos3-nano-sim-bimanual"
    assert isinstance(config, HandStreamInferencePipelineConfig)
    assert isinstance(config, StreamInferencePipelineConfig)
    assert isinstance(config.encoder, HandActionEncoderConfig)
    assert isinstance(config.decoder, HandDecoderConfig)
    assert isinstance(config.diffusion_model, DiffusionModelConfig)
    assert isinstance(config.diffusion_model.transformer, HandTransformerConfig)
    assert isinstance(config.diffusion_model.scheduler, FlowMatchSchedulerConfig)
    assert config.diffusion_model.scheduler.shift == 5.0
    assert config.diffusion_model.scheduler.num_inference_steps == 4
    assert config.diffusion_model.scheduler.denoising_timesteps == [
        1000,
        750,
        500,
        250,
    ]
    assert config.diffusion_model.seed == 0
    assert config.diffusion_model._target is DiffusionModel
    assert config.diffusion_model.transformer.chunk_size == 2
    assert config.diffusion_model.transformer.model_fps is None
    assert config.decoder.backend == "wan"
    assert config.compile_prewarm_steps == 0
    assert issubclass(HandStreamInferencePipeline, StreamInferencePipeline)
    assert issubclass(HandDecoder, StreamingVideoDecoder)


def test_unbounded_cat_cache_remains_supported() -> None:
    config = HandTransformerConfig(device="cpu", cache_mode="cat", window_frames=0)
    transformer = HandTransformer(
        config, network=cast(CosmosMoTNetwork, _FakeNetwork())
    )
    assert transformer.config.window_frames == 0


def test_transformer_runtime_validation_precedes_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HandTransformerConfig(
        device="cpu",
        cache_mode="block",
        compile_denoise=True,
        cuda_graph_denoise=True,
    )
    loaded = False

    def load_network(self: HandTransformer) -> _FakeNetwork:
        nonlocal loaded
        loaded = True
        return _FakeNetwork()

    monkeypatch.setattr(HandTransformer, "_load_network", load_network)

    with pytest.raises(ValueError, match="CUDA device"):
        HandTransformer(config)

    assert not loaded


def test_action_encoder_validates_and_moves_actions() -> None:
    encoder = HandActionEncoder(HandActionEncoderConfig())
    actions = torch.ones(16, action_format.MAX_ACTION_DIM, dtype=torch.float64)

    encoded = encoder(actions)

    assert encoded.actions.shape == actions.shape
    assert encoded.actions.dtype == torch.float32
    assert encoded.actions.is_contiguous()
    assert encoded.domain_ids.tolist() == [action_format.AGIBOT_DOMAIN_ID] * 16
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.ones(16, action_format.MAX_ACTION_DIM - 1))


def test_scene_encoder_keeps_training_latent_crop_without_extra_token_row() -> None:
    # Exercise the real patchifier without loading the VAE weights.
    encoder = HandSceneEncoder.__new__(HandSceneEncoder)
    nn.Module.__init__(encoder)
    channels = SPEC.vfm.vae_latent_channels
    latent = torch.arange(channels * 30 * 52, dtype=torch.float32).reshape(
        channels, 1, 30, 52
    )

    tokens = encoder.patchify_encoded(latent, content_height=360, content_width=640)
    restored = unpatchify_latent(
        tokens,
        channels=channels,
        grid_t=1,
        grid_h=11,
        grid_w=20,
        patch_spatial=2,
    )

    assert tokens.shape == (220, SPEC.vfm.patch_latent_dim)
    assert restored.shape == (channels, 1, 22, 40)
    torch.testing.assert_close(restored, latent[..., :22, :40], rtol=0, atol=0)
    assert (action_format.LATENT_HEIGHT, action_format.LATENT_WIDTH) == (22, 40)


def test_transformer_owns_and_reuses_scene_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = HandTransformer(
        HandTransformerConfig(device="cpu"),
        network=cast(CosmosMoTNetwork, _FakeNetwork()),
    )
    created: list[tuple[Any, _FakeCache]] = []

    def new_cache(scene: Any) -> _FakeCache:
        cache = _FakeCache()
        cache.scene = scene
        created.append((scene, cache))
        return cache

    monkeypatch.setattr(transformer, "_new_cache", new_cache)
    first_scene = SimpleNamespace(name="first")
    first = transformer.initialize_autoregressive_cache(scene=first_scene)
    assert first is created[0][1]

    second_scene = SimpleNamespace(name="second")
    second = transformer.initialize_autoregressive_cache(scene=second_scene)
    assert second is created[1][1]
    assert first.waits == [True]
    assert first.reset_count == 1

    transformer.release_active_cache(first)
    assert second.waits == []
    transformer.release_active_cache(second)
    assert second.waits == [True]
    assert second.reset_count == 1

    reused = transformer.initialize_autoregressive_cache(scene=second_scene)
    assert reused is second
    assert len(created) == 2


@pytest.mark.parametrize("failure_stage", [None, "generate", "decode", "finalize"])
@pytest.mark.parametrize("repeat_fill", [False, True])
def test_scene_prewarm_runs_full_pipeline_and_preserves_only_successful_cache(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str | None, repeat_fill: bool
) -> None:
    transformer = HandTransformer(
        HandTransformerConfig(device="cpu", compile_denoise=False, chunk_size=2),
        network=cast(CosmosMoTNetwork, _FakeNetwork()),
    )
    created: list[_FakeCache] = []
    decoder_caches: list[Any] = []
    stages: list[tuple[str, int]] = []

    def new_cache(scene: Any) -> _FakeCache:
        cache = _FakeCache()
        cache.scene = scene
        created.append(cache)
        return cache

    def visit(stage: str, index: int) -> None:
        stages.append((stage, index))
        if stage == failure_stage and index == 1:
            raise RuntimeError(f"failed {stage}")

    class FakeDiffusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer
            self.device = torch.device("cpu")

        def generate(self, autoregressive_index, cache, input):
            assert cache.profile
            assert input.actions.shape == (8, action_format.MAX_ACTION_DIM)
            assert input.actions.dtype == torch.float32
            assert not input.actions.count_nonzero()
            assert input.domain_ids.tolist() == [action_format.AGIBOT_DOMAIN_ID] * 8
            visit("generate", autoregressive_index)
            return torch.zeros(1), autoregressive_index

        def finalize(self, final_state):
            visit("finalize", final_state)

    class FakeDecoder(nn.Module):
        def initialize_autoregressive_cache(self, *, scene):
            cache = SimpleNamespace(scene=scene, history=[])
            decoder_caches.append(cache)
            return cache

        def forward(self, input, autoregressive_index, cache):
            visit("decode", autoregressive_index)
            cache.history.append(autoregressive_index)
            return input

    monkeypatch.setattr(transformer, "_new_cache", new_cache)
    pipeline = HandStreamInferencePipeline.__new__(HandStreamInferencePipeline)
    nn.Module.__init__(pipeline)
    pipeline.config = derive_config(
        PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
        compile_prewarm_steps=4 if repeat_fill else 2,
        diffusion_model={
            "scheduler": {"num_inference_steps": 2, "denoising_timesteps": [1000, 500]},
            "transformer": {
                "compile_denoise": True,
                "cache_mode": "block",
                "chunk_size": 2,
                "window_frames": 2,
                "fixed_shape_cache_fill": repeat_fill,
                "cuda_graph_denoise": repeat_fill,
            },
        },
    )
    pipeline.encoder = HandActionEncoder(HandActionEncoderConfig())
    pipeline.decoder = FakeDecoder()
    pipeline.diffusion_model = FakeDiffusion()
    scene = SimpleNamespace(name="warm-scene")

    if failure_stage is not None:
        with pytest.raises(RuntimeError, match=f"failed {failure_stage}"):
            pipeline.prepare_scene(scene)
        assert not transformer.has_prepared_cache(scene)
        assert transformer._active_cache is None
        assert created[0].reset_count == (2 if repeat_fill else 1)
        assert not created[0].profile
        failure_stage = None
        stages.clear()

    pipeline.prepare_scene(scene)

    assert stages == [
        (stage, index)
        for index in ([0, 0, 1, 2] if repeat_fill else [0, 1])
        for stage in ("generate", "decode", "finalize")
    ]
    warmed_cache = created[-1]
    assert warmed_cache.reset_count == (2 if repeat_fill else 1)
    assert not warmed_cache.profile
    assert transformer.has_prepared_cache(scene)
    assert transformer._active_cache is None
    warmed_decoder = decoder_caches[-1]
    assert warmed_decoder.history == ([0, 1, 2] if repeat_fill else [0, 1])
    if repeat_fill:
        assert decoder_caches[-2].history == [0]
    pipeline.prepare_scene(scene)
    assert decoder_caches[-1] is warmed_decoder

    visible = pipeline.initialize_cache(
        transformer_context={"scene": scene}, decoder_context={"scene": scene}
    )
    assert visible.transformer_cache is warmed_cache
    assert visible.autoregressive_index is None
    assert visible.decoder_cache is not warmed_decoder
    assert visible.decoder_cache.history == []


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], 0),
        (["--compile-denoise"], 33),
        (["--compile-denoise", "--cuda-graph-denoise"], 33),
        (["--compile-denoise", "--fixed-shape-cache-fill", "--cuda-graph-denoise"], 33),
        (["--compile-denoise", "--valid-length-cache-attention"], 1),
        (
            [
                "--compile-denoise",
                "--valid-length-cache-attention",
                "--cuda-graph-denoise",
            ],
            1,
        ),
        (["--compile-denoise", "--window-frames", "5"], 4),
        (
            [
                "--compile-denoise",
                "--cuda-graph-denoise",
                "--cuda-graph-warmup-iters",
                "0",
            ],
            33,
        ),
        (
            [
                "--compile-denoise",
                "--cuda-graph-denoise",
                "--cuda-graph-warmup-iters",
                "5",
            ],
            34,
        ),
        (["--compile-denoise", "--compile-prewarm-steps", "0"], 0),
        (["--compile-denoise", "--compile-prewarm-steps", "2"], 2),
        (
            [
                "--compile-denoise",
                "--checkpoint",
                "cosmos3_nano_sim_bimanual_ga",
                "--denoise-steps",
                "4",
                "--cuda-graph-denoise",
            ],
            33,
        ),
    ],
)
def test_cli_prewarm_covers_compile_and_capture_regimes(options, expected):
    from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import (
        _add_model_arguments,
        _resolve_pipeline_config,
    )

    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    # Exercise each regime independently of the interactive demo's optimized preset.
    config = _resolve_pipeline_config(
        vars(
            parser.parse_args(
                [
                    "--no-compile-denoise",
                    "--no-valid-length-cache-attention",
                    "--no-cuda-graph-denoise",
                    *options,
                ]
            )
        )
    )
    assert config.compile_prewarm_steps == expected


@pytest.mark.parametrize(
    "warmup_iters, automatic_steps", [(0, 2), (2, 2), (3, 2), (4, 4)]
)
@pytest.mark.parametrize("explicit_steps", [None, 0, 1, 3])
def test_cli_short_window_prewarm_repeats_fill_without_changing_explicit_counts(
    warmup_iters, automatic_steps, explicit_steps
):
    from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import (
        _add_model_arguments,
        _resolve_pipeline_config,
    )

    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    options = [
        "--compile-denoise",
        "--fixed-shape-cache-fill",
        "--no-valid-length-cache-attention",
        "--cuda-graph-denoise",
        "--window-frames",
        "2",
        "--cuda-graph-warmup-iters",
        str(warmup_iters),
    ]
    if explicit_steps is not None:
        options.extend(["--compile-prewarm-steps", str(explicit_steps)])
    config = _resolve_pipeline_config(vars(parser.parse_args(options)))
    assert config.compile_prewarm_steps == (
        automatic_steps if explicit_steps is None else explicit_steps
    )


def test_decoder_primes_scene_state_and_preserves_temporal_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeWanDecoder()
    monkeypatch.setattr(HandDecoder, "_setup_backend", lambda self: backend)
    decoder = HandDecoder(HandDecoderConfig(device="cpu", backend="wan"))
    scene = SimpleNamespace(
        init_latent=torch.zeros(
            action_format.TOKEN_GRID[0] * action_format.TOKEN_GRID[1],
            SPEC.vfm.patch_latent_dim,
        )
    )
    cache = decoder.initialize_autoregressive_cache(scene=scene)

    assert backend.inputs[0][0].shape == (
        1,
        SPEC.vfm.vae_latent_channels,
        action_format.LATENT_HEIGHT,
        action_format.LATENT_WIDTH,
    )
    output = decoder(
        torch.zeros(
            1,
            SPEC.vfm.vae_latent_channels,
            4,
            action_format.LATENT_HEIGHT,
            action_format.LATENT_WIDTH,
        ),
        cache=cache,
    )
    assert output.shape == (
        16,
        3,
        action_format.OUTPUT_HEIGHT,
        action_format.OUTPUT_WIDTH,
    )
    assert decoder.get_output_temporal_size(0, 4) == 16
    assert decoder.get_input_temporal_size(0, 16) == 4
    assert [entry[1] for entry in backend.inputs] == [0, 1, 2, 3, 4]
    assert all(entry[2] is cache.backend_cache for entry in backend.inputs)

    second = decoder.initialize_autoregressive_cache(scene=scene)
    assert decoder.backend is backend
    assert second.backend_cache is not cache.backend_cache
    assert len(backend.caches) == 2
    assert backend.inputs[-1][1] == 0

    decoder.eval().to(dtype=torch.float64)
    assert not backend.training
    assert backend.probe.dtype == torch.float64
    assert dict(decoder.named_modules())["backend"] is backend


def test_pipeline_wires_owned_components_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSceneEncoder(nn.Module):
        def __init__(self, *, device: str) -> None:
            super().__init__()
            self.device = torch.device(device)

    monkeypatch.setattr(pipeline_module, "HandSceneEncoder", FakeSceneEncoder)
    monkeypatch.setattr(
        HandTransformer,
        "_load_network",
        lambda self: _FakeNetwork(),
    )
    monkeypatch.setattr(HandDecoder, "_setup_backend", lambda self: _FakeWanDecoder())
    config = derive_config(
        PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
        diffusion_model={
            "seed": 73,
            "transformer": {"device": "cpu", "chunk_size": 4},
        },
        decoder={"device": "cpu"},
    )

    pipeline = HandStreamInferencePipeline(config)

    assert isinstance(pipeline.diffusion_model, DiffusionModel)
    assert isinstance(pipeline.transformer, HandTransformer)
    assert isinstance(pipeline.decoder, HandDecoder)
    assert isinstance(pipeline.encoder, HandActionEncoder)
    assert pipeline.actions_per_step == 16
    assert config.diffusion_model.seed == 73
    assert isinstance(config.diffusion_model.transformer, HandTransformerConfig)
    assert isinstance(config.diffusion_model.scheduler, FlowMatchSchedulerConfig)
    assert config.diffusion_model.transformer.device == "cpu"
    assert config.diffusion_model.scheduler.denoising_timesteps == [
        1000,
        750,
        500,
        250,
    ]


def test_decoder_config_rejects_an_unknown_backend_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_called = False

    def setup_backend(self: HandDecoder) -> nn.Module:
        nonlocal setup_called
        setup_called = True
        return nn.Identity()

    monkeypatch.setattr(HandDecoder, "_setup_backend", setup_backend)
    with pytest.raises(ValueError, match="unknown decoder"):
        HandDecoder(HandDecoderConfig(device="cpu", backend="invalid"))  # ty: ignore[invalid-argument-type]
    assert not setup_called


def test_pipeline_rejects_negative_compile_prewarm() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_pipeline_config(
            derive_config(
                PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
                compile_prewarm_steps=-1,
            ),
            check_runtime=False,
        )


def test_pipeline_config_rejects_invalid_components_before_setup() -> None:
    with pytest.raises(ValueError, match="unknown decoder"):
        validate_pipeline_config(
            derive_config(
                PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
                decoder={"backend": "invalid"},
            ),
            check_runtime=False,
        )
    with pytest.raises(ValueError, match="number of denoising_timesteps"):
        validate_pipeline_config(
            derive_config(
                PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
                diffusion_model={"scheduler": {"num_inference_steps": 3}},
            ),
            check_runtime=False,
        )
    with pytest.raises(ValueError, match="compile prewarm requires"):
        validate_pipeline_config(
            derive_config(
                PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
                compile_prewarm_steps=1,
            ),
            check_runtime=False,
        )
