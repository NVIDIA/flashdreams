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

"""Cosmos3 Nano Sim Bimanual binding to the reusable HandDreams application."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from handdreams.app import ControllerPipelineFactory, HandDreamsV2Application
from handdreams.config import HandDreamsApplicationDefaults, HanddreamsScenario
from handdreams.scenes import ScenePreview
from handdreams.teleop import IsaacTeleopControllerPipeline
from torch import Tensor

from cosmos3_nano_sim_bimanual.action_format import (
    AGIBOT_DOMAIN_ID,
    CAMERA_DOMAIN_ID,
)
from cosmos3_nano_sim_bimanual.config import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    DENOISING_TIMESTEPS,
)
from cosmos3_nano_sim_bimanual.decoder import DECODER_BACKENDS, HandDecoderConfig
from cosmos3_nano_sim_bimanual.encoder import HandActionInput
from cosmos3_nano_sim_bimanual.fetch_checkpoint import checkpoint_denoise_steps
from cosmos3_nano_sim_bimanual.pipeline import (
    HandStreamInferencePipeline,
    HandStreamInferencePipelineConfig,
    _compile_prewarm_plan,
    validate_pipeline_config,
)
from cosmos3_nano_sim_bimanual.sdpa_backends import SDPA_BACKENDS
from cosmos3_nano_sim_bimanual.transformer import HandTransformerConfig
from cosmos3_nano_sim_bimanual.transformer.config import (
    CACHE_MODES,
    CACHE_REFRESH_MODES,
    COMPILE_DENOISE_MODES,
)
from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineCache

from .actions import (
    ControllerActionState,
    encode_controller_actions,
    stationary_camera_mask,
)
from .agibot_scenes import default_scene_directory, prepare_default_scenes
from .config import DEFAULT_HAND_PIPELINE_CONFIG
from .scenes import Scene, discover_scenes, load_scene, scene_previews

InferencePipelineFactory = Callable[[HandStreamInferencePipelineConfig], Any]
SceneFactory = Callable[[HanddreamsScenario, Any], Any]


@dataclass(kw_only=True, slots=True)
class CosmosHandDreamsRolloutBuilder:
    """Scene encoding and rollout operations for the application-owned Cosmos model."""

    pipeline: HandStreamInferencePipeline
    """Loaded model shared by the application's sessions."""
    scene: Scene
    """Encoded scene and recorded actions for this session."""
    _action_state: ControllerActionState = field(
        default_factory=ControllerActionState, init=False
    )
    """Chunk anchors advanced only when diffusion accepts the action window."""
    _pending_action_state: ControllerActionState | None = field(
        default=None, init=False
    )
    """Proposed anchors for the current controller transaction."""
    _pending_domain_ids: Tensor | None = field(default=None, init=False)
    """Per-row domains for the same pending action transaction."""

    @property
    def actions_per_step(self) -> int:
        """Return the number of action rows consumed by one model step."""
        return self.pipeline.actions_per_step

    def prepare_scene(self) -> None:
        """Warm the selected scene on the shared pipeline."""
        self.pipeline.prepare_scene(self.scene)

    def build_cache(self) -> StreamInferencePipelineCache[Any, Any, Any]:
        """Build a session cache using the scene's conditioning."""
        self._action_state = ControllerActionState()
        self._pending_action_state = None
        self._pending_domain_ids = None
        context = {"scene": self.scene}
        return self.pipeline.initialize_cache(
            transformer_context=context,
            decoder_context=context,
        )

    def encode_actions(self, frames: list[dict[str, Any]]) -> Tensor:
        """Encode calibrated controller targets for the Cosmos action pathway."""
        self._pending_action_state = None
        self._pending_domain_ids = None
        actions, pending = encode_controller_actions(
            frames, self.scene, self._action_state
        )
        camera_rows, pending = stationary_camera_mask(frames, pending)
        # Unified action projection: retain globally normalized camera channels,
        # mask every hand/gripper channel AFTER normalization for domain 2.
        actions[camera_rows, 9:] = 0
        self._pending_domain_ids = torch.where(
            camera_rows, CAMERA_DOMAIN_ID, AGIBOT_DOMAIN_ID
        )
        self._pending_action_state = pending
        return actions

    def generate(
        self,
        step_index: int,
        cache: Any,
        actions: Tensor,
        on_accepted: Callable[[], None],
    ) -> Tensor:
        """Generate video and accept the control transaction after diffusion."""
        pending = self._pending_action_state
        domain_ids = self._pending_domain_ids
        if pending is None or domain_ids is None:
            raise RuntimeError("encode controller actions before generating video")

        def accept_actions() -> None:
            self._action_state = pending
            on_accepted()

        try:
            return self.pipeline.generate(
                step_index,
                cache,
                input=HandActionInput(actions, domain_ids),
                on_diffusion_complete=accept_actions,
            )
        finally:
            self._pending_action_state = None
            self._pending_domain_ids = None

    def finalize(self, step_index: int, cache: Any) -> Mapping[str, Any]:
        """Finalize one model step and return its timings."""
        return self.pipeline.finalize(step_index, cache) or {}

    def release_cache(self, cache: Any) -> None:
        """Release the pipeline's active cache when this session stops."""
        self._action_state = ControllerActionState()
        self._pending_action_state = None
        self._pending_domain_ids = None
        self.pipeline.release_session(cache)


def _setup_inference_pipeline(
    config: HandStreamInferencePipelineConfig,
) -> HandStreamInferencePipeline:
    pipeline = config.setup()
    pipeline.eval()
    return pipeline


def _load_scene(scenario: HanddreamsScenario, pipeline: Any) -> Scene:
    name, source = _resolve_scene_source(scenario)
    return load_scene(
        name,
        source,
        encode=pipeline.encode_scene,
        patchify=pipeline.patchify_scene_latent,
        device=pipeline.device,
    )


def _resolve_scene_source(scenario: HanddreamsScenario) -> tuple[str, str]:
    """Resolve one selected scene from a prepared or user-supplied catalog."""
    sources = _scene_sources(
        scenario, (scenario.sample,) if scenario.sample is not None else None
    )
    if len(sources) != 1:
        raise ValueError(f"expected exactly one HandDreams scene, got {len(sources)}")
    return sources[0]


def _discover_scene_catalog(scenario: HanddreamsScenario) -> tuple[str, ...]:
    # Called only by application.init; session selection never downloads assets.
    if scenario.scene_path is None and scenario.sample_dir is None:
        prepare_default_scenes()
    sources = _scene_sources(scenario)
    names = [name for name, _ in sources]
    if scenario.sample is not None and scenario.sample not in names:
        raise ValueError(f"requested HandDreams sample is absent: {scenario.sample}")
    return tuple(dict.fromkeys(names))


def _scene_previews(
    scenario: HanddreamsScenario, names: tuple[str, ...]
) -> tuple[ScenePreview, ...]:
    return scene_previews(_scene_sources(scenario, names))


def _scene_sources(
    scenario: HanddreamsScenario, names: tuple[str, ...] | None = None
) -> list[tuple[str, str]]:
    directory = scenario.sample_dir
    if directory is None and scenario.scene_path is None:
        directory = default_scene_directory()
    return discover_scenes(
        sample=scenario.sample,
        samples=names,
        scene_path=scenario.scene_path,
        sample_dir=directory,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = DEFAULT_HAND_PIPELINE_CONFIG
    transformer = cast(HandTransformerConfig, defaults.diffusion_model.transformer)
    decoder = cast(HandDecoderConfig, defaults.decoder)
    scheduler = cast(FlowMatchSchedulerConfig, defaults.diffusion_model.scheduler)
    parser.add_argument("--checkpoint", default=transformer.checkpoint)
    parser.add_argument("--credential-path", default=transformer.credential_path)
    parser.add_argument("--device", default=transformer.device)
    parser.add_argument("--seed", type=int, default=defaults.diffusion_model.seed)
    parser.add_argument("--decoder", choices=DECODER_BACKENDS, default=decoder.backend)
    parser.add_argument("--chunk-size", type=int, default=transformer.chunk_size)
    parser.add_argument(
        "--denoise-steps",
        type=int,
        choices=tuple(DENOISING_TIMESTEPS),
        default=scheduler.num_inference_steps,
    )
    parser.add_argument("--window-frames", type=int, default=transformer.window_frames)
    parser.add_argument(
        "--native-gqa-attention",
        action=argparse.BooleanOptionalAction,
        default=transformer.native_gqa_attention,
    )
    parser.add_argument(
        "--sdpa-backend", choices=SDPA_BACKENDS, default=transformer.sdpa_backend
    )
    parser.add_argument(
        "--cache-mode", choices=CACHE_MODES, default=transformer.cache_mode
    )
    parser.add_argument(
        "--cache-refresh-mode",
        choices=CACHE_REFRESH_MODES,
        default=transformer.cache_refresh_mode,
    )
    parser.add_argument(
        "--overlap-cache-update",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-denoise",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-denoise-mode",
        choices=COMPILE_DENOISE_MODES,
        default=transformer.compile_denoise_mode,
    )
    parser.add_argument(
        "--fixed-shape-cache-fill",
        action=argparse.BooleanOptionalAction,
        default=transformer.fixed_shape_cache_fill,
    )
    parser.add_argument(
        "--valid-length-cache-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cuda-graph-denoise",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cuda-graph-warmup-iters",
        type=int,
        default=transformer.cuda_graph_warmup_iters,
    )
    parser.add_argument(
        "--compile-prewarm-steps",
        type=int,
        default=None,
        help=(
            "Hidden initialization chunks; defaults to warming every compiled "
            "denoise regime and CUDA graph. Use 0 to disable or a positive "
            "number to override."
        ),
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=transformer.profile,
    )


def _resolve_pipeline_config(
    args: Mapping[str, Any],
) -> HandStreamInferencePipelineConfig:
    checkpoint = str(args["checkpoint"])
    denoise_steps = int(args["denoise_steps"])
    expected_steps = checkpoint_denoise_steps(checkpoint)
    if expected_steps is not None and denoise_steps != expected_steps:
        raise ValueError(
            f"{checkpoint!r} expects {expected_steps} denoising steps, "
            f"got {denoise_steps}"
        )
    credential_path = (
        None if args.get("credential_path") is None else str(args["credential_path"])
    )
    device = str(args["device"])
    cuda_defaults = torch.device(device).type == "cuda"
    preset = cast(
        HandTransformerConfig, DEFAULT_HAND_PIPELINE_CONFIG.diffusion_model.transformer
    )
    optimizations = {
        name: cuda_defaults and getattr(preset, name)
        if args[name] is None
        else bool(args[name])
        for name in (
            "overlap_cache_update",
            "compile_denoise",
            "valid_length_cache_attention",
            "cuda_graph_denoise",
        )
    }
    prewarm_steps = args["compile_prewarm_steps"]
    config = derive_config(
        DEFAULT_HAND_PIPELINE_CONFIG,
        compile_prewarm_steps=0 if prewarm_steps is None else int(prewarm_steps),
        decoder={
            "backend": str(args["decoder"]),
            "credential_path": credential_path,
            "device": device,
        },
        diffusion_model={
            "seed": None if args["seed"] is None else int(args["seed"]),
            "scheduler": {
                "num_inference_steps": denoise_steps,
                "denoising_timesteps": list(DENOISING_TIMESTEPS[denoise_steps]),
            },
            "transformer": {
                "checkpoint": checkpoint,
                "credential_path": credential_path,
                "device": device,
                "chunk_size": int(args["chunk_size"]),
                "model_fps": args.get("control_fps"),
                "window_frames": int(args["window_frames"]),
                "native_gqa_attention": bool(args["native_gqa_attention"]),
                "sdpa_backend": str(args["sdpa_backend"]),
                "cache_mode": str(args["cache_mode"]),
                "cache_refresh_mode": str(args["cache_refresh_mode"]),
                **optimizations,
                "compile_denoise_mode": str(args["compile_denoise_mode"]),
                "fixed_shape_cache_fill": bool(args["fixed_shape_cache_fill"]),
                "cuda_graph_warmup_iters": int(args["cuda_graph_warmup_iters"]),
                "profile": bool(args["profile"]),
            },
        },
    )
    transformer = validate_pipeline_config(config, check_runtime=False)
    if prewarm_steps is None and transformer.compile_denoise:
        _, _, prewarm_steps = _compile_prewarm_plan(config)
        config = replace(config, compile_prewarm_steps=prewarm_steps)
    return config


COSMOS_HANDDREAMS_DEFAULTS = HandDreamsApplicationDefaults(
    video_width=DEFAULT_VIDEO_WIDTH,
    video_height=DEFAULT_VIDEO_HEIGHT,
    add_model_arguments=_add_model_arguments,
    resolve_pipeline_config=_resolve_pipeline_config,
    inference_pipeline_factory=_setup_inference_pipeline,
    discover_scene_catalog=_discover_scene_catalog,
    scene_previews=_scene_previews,
    scene_factory=_load_scene,
    rollout_builder_factory=CosmosHandDreamsRolloutBuilder,
)
"""Cosmos model and scene bindings used by the HandDreams V2 entry point."""


class CosmosHandDreamsV2Application(HandDreamsV2Application):
    """Cosmos specialization of the reusable HandDreams application."""

    def __init__(
        self,
        *,
        inference_pipeline_factory: InferencePipelineFactory = _setup_inference_pipeline,
        pipeline_factory: ControllerPipelineFactory = IsaacTeleopControllerPipeline,
        scene_factory: SceneFactory | None = None,
    ) -> None:
        super().__init__(
            defaults=replace(
                COSMOS_HANDDREAMS_DEFAULTS,
                inference_pipeline_factory=inference_pipeline_factory,
                scene_factory=scene_factory or _load_scene,
            ),
            pipeline_factory=pipeline_factory,
        )


def create_app() -> IApplication:
    """Return the Cosmos HandDreams application registered as ``handdreams``."""
    return CosmosHandDreamsV2Application()


__all__ = [
    "COSMOS_HANDDREAMS_DEFAULTS",
    "CosmosHandDreamsRolloutBuilder",
    "CosmosHandDreamsV2Application",
    "create_app",
]
