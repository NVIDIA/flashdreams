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

"""OmniDreams model runtime and browser hooks for the shared WebRTC demo."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from omnidreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
    OmnidreamsConditioningState,
    OmnidreamsConditioningWrapper,
    TextPrompt,
)
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.scenes import (
    SCENE_CLIPGT_DIRNAME,
    SCENE_PROMPT_FILENAME,
    SCENE_VARIANT_DEFAULT,
    ensure_hf_scene_synced,
    extract_local_scene,
    prepare_clipgt_dir,
    resolve_scene_assets,
)
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.runtime import InferenceConfig, StepResult
from flashdreams.runtime.demo import DemoSpec, WebRTCAppResources, WebRTCOutputSpec
from flashdreams.runtime.demo.webrtc import (
    CreateWebRTCApp,
    RunWebRTCServer,
    serve_webrtc_demo,
)
from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    PoseSegment,
)
from flashdreams.serving.webrtc.encoders import EncoderBackend
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import ThreadAffineDistributedWebRTCRuntime
from flashdreams.serving.webrtc.server import create_webrtc_app

from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    OMNIDREAMS_MODEL_ID,
    resolve_webrtc_scenario,
)

WebRTCRuntimeFactory = Callable[..., Any]


class OmnidreamsWebRTCModelRuntimeError(RuntimeError):
    """Raised when the OmniDreams demo runtime is used incorrectly."""


@dataclass(frozen=True, slots=True)
class OmnidreamsWebRTCModelRuntimeConfig:
    """Configuration for one scene-driven OmniDreams WebRTC runtime."""

    pipeline_config_name: str
    """User-facing name of the selected OmniDreams pipeline."""

    pipeline_config: Any
    """Resolved single-view OmniDreams pipeline configuration."""

    scene_dir: Path | None = None
    """Local scene root; ``None`` downloads the selected Hugging Face scene."""

    scene_uuid: str | None = DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID
    """Scene UUID used for remote lookup or local archive selection."""

    scene_variant: str = SCENE_VARIANT_DEFAULT
    """Weather variant selected from the scene assets."""

    seed: int | None = 42
    """Per-rollout seed; ``None`` selects fresh entropy for every session."""

    device: str = "cuda:0"
    """Device used for rendering and model inference."""

    video_height: int = 704
    """Generated video height in pixels."""

    video_width: int = 1280
    """Generated video width in pixels."""

    fps: int = 30
    """Input sampling and output playback frame rate."""

    camera_name: str = "camera_front_wide_120fov"
    """Scene camera controlled by browser keyboard input."""

    move_speed_per_s: float = 6.0
    """Forward and reverse translation speed in scene units per second."""

    rotate_speed_rad_per_s: float = float(np.deg2rad(35.0))
    """Left and right rotation speed in radians per second."""

    warmup_chunks: int = 10
    """Number of synthetic chunks generated before accepting sessions."""

    warmup_timeout_s: float = 600.0
    """Maximum duration for WebRTC loopback warmup."""

    debug_serve_hdmaps: bool = False
    """Stream rendered conditioning frames without running video generation."""

    encoder_backend: EncoderBackend = "auto"
    """WebRTC video encoder selection policy."""

    encoder_bitrate_bps: int = 6_000_000
    """Target WebRTC video bitrate in bits per second."""

    encoder_gop: int = 30
    """WebRTC video encoder group-of-pictures length."""


class OmnidreamsWebRTCModelRuntime(
    ThreadAffineDistributedWebRTCRuntime[
        OmnidreamsWebRTCModelRuntimeConfig,
        None,
    ]
):
    """Run one single-view OmniDreams scene with browser camera controls."""

    def __init__(self, *, config: OmnidreamsWebRTCModelRuntimeConfig) -> None:
        super().__init__(
            config=config,
            runtime_error_type=OmnidreamsWebRTCModelRuntimeError,
            thread_name="omnidreams-demo-runtime",
        )
        self.pose_integrator = self._new_pose_integrator()
        self._wrapper: OmnidreamsConditioningWrapper | None = None
        self._state: OmnidreamsConditioningState | None = None
        self._renderer: Any | None = None
        self._scene_data: Any | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._camera_to_rig: torch.Tensor | None = None
        self._initial_ego_pose: np.ndarray | None = None
        self._step_index = 0
        self._next_timestamp_us = 0
        self._clipgt_temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _new_pose_integrator(self) -> CameraPoseIntegrator:
        return CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )

    def _is_runtime_initialized(self) -> bool:
        return self._wrapper is not None and self._renderer is not None

    def _runtime_step_index(self) -> int:
        return self._step_index

    def _next_input_frame_count(self) -> int:
        wrapper = self._require_wrapper()
        if self._state is None:
            return int(wrapper.initial_frame_chunk_size)
        return int(wrapper.frame_chunk_size)

    def _steady_output_frame_count(self) -> int:
        return int(self._require_wrapper().frame_chunk_size)

    def _initialize_sync(self) -> None:
        if self._wrapper is not None:
            return

        init_t0 = time.perf_counter()
        cfg = self.config
        transformer_cfg = cfg.pipeline_config.diffusion_model.transformer
        if not isinstance(transformer_cfg, CosmosTransformerConfig):
            raise TypeError(
                "OmniDreams WebRTC requires a CosmosTransformerConfig pipeline."
            )
        if transformer_cfg.num_views != 1:
            raise ValueError(
                "OmniDreams WebRTC supports only single-view configs; "
                f"{cfg.pipeline_config_name!r} has num_views="
                f"{transformer_cfg.num_views}."
            )
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniDreams WebRTC inference.")

        scene_dir = self._prepare_scene()
        clipgt_dir, first_frame_path, prompt_path = resolve_scene_assets(
            scene_dir,
            prompt_filename=SCENE_PROMPT_FILENAME,
            clipgt_dirname=SCENE_CLIPGT_DIRNAME,
            camera_name=cfg.camera_name,
            variant=cfg.scene_variant,
        )
        self._initial_rgb_frames = self._load_first_frame(first_frame_path)
        prompt = prompt_path.read_text(encoding="utf-8").strip() or AV_POSITIVE_PROMPT
        self._text_prompts = [TextPrompt(positive=prompt)]

        loadable_clipgt_dir, self._clipgt_temp_dir = prepare_clipgt_dir(clipgt_dir)
        logger.info("Loading OmniDreams scene data from {}", loadable_clipgt_dir)
        scene_data = load_scene(
            loadable_clipgt_dir,
            camera_names=[cfg.camera_name],
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=(cfg.video_height, cfg.video_width),
        )
        scene_data = load_and_attach_ludus_scene(
            loadable_clipgt_dir,
            scene_data,
            device=self._device,
        )
        self._validate_scene_data(scene_data, scene_dir=loadable_clipgt_dir)

        logger.info(
            "Setting up OmniDreams pipeline {} on {}.",
            cfg.pipeline_config_name,
            self._device,
        )
        wrapper = OmnidreamsConditioningWrapper(
            pipeline_config_name=cfg.pipeline_config_name,
            pipeline_config=cfg.pipeline_config,
            resolution_wh=(cfg.video_width, cfg.video_height),
            seed_for_every_rollout=cfg.seed,
            device=self._device,
        )
        renderer = wrapper.create_renderer(scene_data, [cfg.camera_name])

        self._wrapper = wrapper
        self._renderer = renderer
        self._scene_data = scene_data
        self._camera_to_rig = torch.as_tensor(
            scene_data.camera_extrinsics[cfg.camera_name],
            device=self._device,
            dtype=torch.float32,
        )
        self._initial_ego_pose = scene_data.ego_poses[0].transformation_matrix
        self._next_timestamp_us = int(scene_data.ego_poses[0].timestamp)
        self._reset_rollout_sync()
        self._initialize_video_encoder_sync()
        logger.info(
            "OmniDreams runtime initialization complete in {:.1f}s.",
            time.perf_counter() - init_t0,
        )

    def _prepare_scene(self) -> Path:
        cfg = self.config
        if cfg.scene_dir is None:
            return ensure_hf_scene_synced(
                cfg.scene_uuid or DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
                variant=cfg.scene_variant,
                clipgt_dirname=SCENE_CLIPGT_DIRNAME,
            )
        return extract_local_scene(
            cfg.scene_dir,
            scene_uuid=cfg.scene_uuid,
            variant=cfg.scene_variant,
            clipgt_dirname=SCENE_CLIPGT_DIRNAME,
        )

    def _load_first_frame(self, path: Path) -> torch.Tensor:
        logger.info("Loading OmniDreams first frame from {}", path)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (self.config.video_width, self.config.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        return (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.uint8)
        )

    def _validate_scene_data(self, scene_data: Any, *, scene_dir: Path) -> None:
        camera_name = self.config.camera_name
        if not scene_data.ego_poses:
            raise ValueError(f"Scene {scene_dir} has no ego poses.")
        if camera_name not in scene_data.camera_models:
            raise ValueError(f"Camera {camera_name!r} was not loaded from {scene_dir}.")
        if camera_name not in scene_data.camera_extrinsics:
            raise ValueError(
                f"Camera {camera_name!r} has no extrinsics in {scene_dir}."
            )

    def _reset_rollout_sync(self, session_input: None = None) -> None:
        del session_input
        wrapper = self._require_wrapper()
        if self._renderer is None or self._scene_data is None:
            raise OmnidreamsWebRTCModelRuntimeError("Scene state is not initialized.")
        if self._initial_ego_pose is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "Initial camera pose is unavailable."
            )

        if self._state is not None and self._state.pipeline_cache is not None:
            del self._state.pipeline_cache
        self._state = None
        self._step_index = 0
        self.pose_integrator = self._new_pose_integrator()
        self.pose_integrator.reset(self._initial_ego_pose)
        self._next_timestamp_us = int(self._scene_data.ego_poses[0].timestamp)
        wrapper.set_rollout_seed(self.config.seed)

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> StepResult:
        wrapper = self._require_wrapper()
        if (
            self._renderer is None
            or self._initial_rgb_frames is None
            or self._text_prompts is None
            or self._camera_to_rig is None
        ):
            raise OmnidreamsWebRTCModelRuntimeError("Runtime is not initialized.")
        if len(frame_times) != self._next_input_frame_count():
            raise OmnidreamsWebRTCModelRuntimeError(
                f"Expected {self._next_input_frame_count()} frame times for "
                f"step {self._step_index}, got {len(frame_times)}."
            )
        if not segments:
            raise OmnidreamsWebRTCModelRuntimeError(
                f"Step {self._step_index} received no control segments."
            )

        ego_poses = self.pose_integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        ego_poses_t = torch.from_numpy(ego_poses).to(
            device=self._device,
            dtype=torch.float32,
        )
        camera_poses = torch.einsum("nij,jk->nik", ego_poses_t, self._camera_to_rig)
        frame_timestamps_us = self._consume_timestamps(len(frame_times))
        serve_hdmaps = self.config.debug_serve_hdmaps

        if self._state is None:
            output = wrapper.start_generation(
                text_prompts=self._text_prompts,
                initial_rgb_frames=self._initial_rgb_frames,
                renderer=self._renderer,
                camera_names=[self.config.camera_name],
                camera_poses_per_view={self.config.camera_name: camera_poses},
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
        else:
            output = wrapper.continue_generation(
                state=self._state,
                camera_names=[self.config.camera_name],
                camera_poses_per_view={self.config.camera_name: camera_poses},
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
        self._state = output.state
        if self._state.pipeline_cache is not None:
            wrapper.finalize_block_generation(
                self._state.pipeline_cache,
                output.finalization_state,
            )

        metadata = {"stream": "hdmap" if serve_hdmaps else "rgb"}
        if serve_hdmaps:
            video_chunk = output.condition_frames
        else:
            if output.rgb_frames is None:
                raise OmnidreamsWebRTCModelRuntimeError(
                    "OmniDreams generation produced no RGB frames."
                )
            video_chunk = output.rgb_frames
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=video_chunk.detach(),
            layout="bvtchw",
            metadata=metadata,
        )
        expected_frames = len(frame_times)
        if result.frame_count != expected_frames:
            raise OmnidreamsWebRTCModelRuntimeError(
                f"Expected generated chunk to contain {expected_frames} frames, "
                f"got {result.frame_count}."
            )
        self._step_index += 1
        return result

    def _consume_timestamps(self, num_frames: int) -> list[int]:
        step_us = int(round(1_000_000 / self.config.fps))
        timestamps = [
            self._next_timestamp_us + frame_index * step_us
            for frame_index in range(num_frames)
        ]
        self._next_timestamp_us += num_frames * step_us
        return timestamps

    def _close_sync(self) -> None:
        if self._wrapper is not None and self._state is not None:
            self._wrapper.cleanup(self._state)
        elif self._renderer is not None:
            self._renderer.cleanup()
        self._state = None
        self._wrapper = None
        self._renderer = None
        self._scene_data = None
        self._initial_rgb_frames = None
        self._text_prompts = None
        self._camera_to_rig = None
        self._initial_ego_pose = None
        if self._clipgt_temp_dir is not None:
            self._clipgt_temp_dir.cleanup()
            self._clipgt_temp_dir = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _require_wrapper(self) -> OmnidreamsConditioningWrapper:
        if self._wrapper is None:
            raise OmnidreamsWebRTCModelRuntimeError("Runtime is not initialized.")
        return self._wrapper


def serve_omnidreams_webrtc_demo(
    *,
    spec: DemoSpec,
    world_rank: int = 0,
    runtime_factory: WebRTCRuntimeFactory = OmnidreamsWebRTCModelRuntime,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> object:
    """Create OmniDreams' runtime and serve it through the shared WebRTC transport."""
    if spec.input_mode != "keyboard-driving":
        raise ValueError(
            "OmniDreams WebRTC requires input_mode='keyboard-driving', "
            f"got {spec.input_mode!r}."
        )
    if not isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("OmniDreams WebRTC requires WebRTC output.")
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    if config.model_id != OMNIDREAMS_MODEL_ID:
        raise ValueError(
            f"OmniDreams WebRTC requires model_id={OMNIDREAMS_MODEL_ID!r}, "
            f"got {config.model_id!r}."
        )
    scenario = resolve_webrtc_scenario(spec.scenario)
    preset_id = _preset_id(config)
    seed = _option(config, "seed", 42)
    runtime_config = OmnidreamsWebRTCModelRuntimeConfig(
        pipeline_config_name=preset_id,
        pipeline_config=_pipeline_config(config),
        scene_dir=scenario.scene_dir,
        scene_uuid=scenario.scene_uuid,
        scene_variant=scenario.scene_variant,
        seed=None if seed is None else int(seed),
        device=config.device or str(_option(config, "device", "cuda:0")),
        video_height=spec.output.video_height,
        video_width=spec.output.video_width,
        fps=spec.output.fps,
        camera_name=scenario.camera_name,
        warmup_chunks=spec.output.warmup_chunks,
        warmup_timeout_s=spec.output.warmup_timeout_s,
        debug_serve_hdmaps=scenario.debug_serve_hdmaps,
        encoder_backend="default" if scenario.prefer_sw_encoder else "auto",
    )
    runtime_config = _apply_runtime_options(runtime_config, config.runtime_options)
    runtime = runtime_factory(config=runtime_config)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        identity=runtime_config.pipeline_config_name,
        busy_message="An OmniDreams session is already active.",
        warmup_label="OmniDreams WebRTC",
        supported_control_keys=WSAD_SUPPORTED_KEYS,
        fatal_generation_errors=True,
        client_liveness_timeout_s=spec.output.client_liveness_timeout_s,
    )
    from importlib.resources import files

    return serve_webrtc_demo(
        output=spec.output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("omnidreams.demo").joinpath("web"),
            preload_name="OmniDreams",
        ),
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


def _preset_id(config: InferenceConfig | None) -> str:
    return (
        DEFAULT_OMNIDREAMS_PRESET
        if config is None or config.preset_id is None
        else config.preset_id
    )


def _pipeline_config(config: InferenceConfig) -> Any:
    custom = config.runtime_options.get("pipeline_config")
    if custom is not None:
        return custom
    preset_id = _preset_id(config)
    try:
        return OMNIDREAMS_CONFIGS[preset_id]
    except KeyError as exc:
        supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
        raise ValueError(
            f"Unsupported OmniDreams preset_id={preset_id!r}. "
            f"Supported presets: {supported}."
        ) from exc


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


def _apply_runtime_options(
    runtime_config: OmnidreamsWebRTCModelRuntimeConfig,
    options: Any,
) -> OmnidreamsWebRTCModelRuntimeConfig:
    if not isinstance(options, dict):
        options = dict(options)
    overrides = {
        name: options[name]
        for name in (
            "move_speed_per_s",
            "rotate_speed_rad_per_s",
            "encoder_bitrate_bps",
            "encoder_gop",
        )
        if name in options
    }
    return replace(runtime_config, **overrides) if overrides else runtime_config


__all__ = [
    "OmnidreamsWebRTCModelRuntime",
    "OmnidreamsWebRTCModelRuntimeConfig",
    "OmnidreamsWebRTCModelRuntimeError",
    "WebRTCRuntimeFactory",
    "serve_omnidreams_webrtc_demo",
]
