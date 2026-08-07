# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from omnidreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
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
)
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.webrtc.model_session import OmnidreamsConditioningSessionCore
from omnidreams.webrtc.postprocess import validate_requested_postprocess_preset
from omnidreams.webrtc.scene_assets import (
    ensure_hf_scene_synced,
    extract_local_scene,
    prepare_clipgt_dir,
    resolve_scene_assets,
)

from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    create_video_postprocess_stream,
)
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepResult
from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    PoseSegment,
)
from flashdreams.serving.webrtc.encoders import EncoderBackend
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
)
from flashdreams.serving.webrtc.runtime import (
    ThreadAffineDistributedWebRTCRuntime,
)

# Default scene (clear-weather base archive). Weather siblings are selected
# via OmnidreamsRuntimeConfig.scene_variant / the server's --scene-variant.
DEFAULT_WEBRTC_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"


class OmnidreamsRuntimeError(RuntimeError):
    """Raised when the Omnidreams WebRTC runtime is used incorrectly."""


@dataclass(slots=True)
class OmnidreamsRuntimeConfig:
    pipeline_config_name: str = (
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    )
    pipeline_config: Any | None = None
    manifest_path: Path | None = None
    scene_dir: Path | None = None
    scene_uuid: str | None = None
    # Weather variant slug (default/rain/snow): picks the sibling USDZ + prompt.
    scene_variant: str = SCENE_VARIANT_DEFAULT
    seed: int | None = 42
    device: str = "cuda:0"
    video_height: int = 704
    video_width: int = 1280
    fps: int = 30
    camera_name: str = "camera_front_wide_120fov"
    prompt_filename: str = SCENE_PROMPT_FILENAME
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME
    move_speed_per_s: float = 6.0
    rotate_speed_rad_per_s: float = float(np.deg2rad(35.0))
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0
    debug_serve_hdmaps: bool = False
    postprocess: VideoPostprocessChainConfig = field(
        default_factory=VideoPostprocessChainConfig
    )
    # Video encoder selection. ``"auto"`` prefers NVENC when the driver
    # reports support at the target resolution (Stage-1 probe via
    # ``PyNvVideoCodec.GetEncoderCaps``) and falls back to aiortc's
    # software encoder otherwise. ``"nvenc"`` fails startup if NVENC
    # cannot be initialized. ``"default"`` skips the probe entirely.
    encoder_backend: EncoderBackend = "auto"
    encoder_bitrate_bps: int = 6_000_000
    encoder_gop: int = 30


@dataclass(frozen=True, slots=True)
class OmnidreamsSessionInput:
    """Browser-selectable settings applied to the next WebRTC rollout."""

    postprocess_preset: str | None = None
    """Launched preset selection; ``None`` keeps the CLI default and ``""`` disables it."""


class OmnidreamsInferenceRuntime(
    ThreadAffineDistributedWebRTCRuntime[
        OmnidreamsRuntimeConfig,
        OmnidreamsSessionInput,
    ]
):
    """Single-scene, single-view Omnidreams runtime for WebRTC control."""

    def __init__(self, config: OmnidreamsRuntimeConfig | None = None) -> None:
        super().__init__(
            config=config or OmnidreamsRuntimeConfig(),
            runtime_error_type=OmnidreamsRuntimeError,
            thread_name="omnidreams-webrtc-runtime",
        )

        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self._wrapper: OmnidreamsConditioningWrapper | None = None
        self._model_session: OmnidreamsConditioningSessionCore | None = None
        self._renderer: Any | None = None
        self._scene_data: Any | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._camera_to_rig: torch.Tensor | None = None
        self._initial_ego_pose: np.ndarray | None = None
        self._next_timestamp_us: int = 0
        self._postprocess_preset = self.config.postprocess.preset
        self._clipgt_temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _is_runtime_initialized(self) -> bool:
        return self._wrapper is not None and self._model_session is not None

    def _runtime_step_index(self) -> int:
        if self._model_session is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return self._model_session.step_index

    def _next_input_frame_count(self) -> int:
        if self._model_session is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return self._model_session.next_num_frames()

    def _steady_output_frame_count(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return int(self._wrapper.frame_chunk_size)

    def _initialize_sync(self) -> None:
        if self._wrapper is not None:
            return

        init_t0 = time.perf_counter()
        cfg = self.config
        if cfg.scene_dir is None:
            scene_uuid = cfg.scene_uuid or DEFAULT_WEBRTC_SCENE_UUID
            scene_dir = ensure_hf_scene_synced(
                scene_uuid,
                variant=cfg.scene_variant,
                clipgt_dirname=cfg.clipgt_dirname,
            )
        else:
            scene_dir = extract_local_scene(
                cfg.scene_dir,
                scene_uuid=cfg.scene_uuid,
                variant=cfg.scene_variant,
                clipgt_dirname=cfg.clipgt_dirname,
            )

        cfg.scene_dir = scene_dir
        clipgt_dir, first_frame_path, prompt_path = resolve_scene_assets(
            scene_dir,
            prompt_filename=cfg.prompt_filename,
            clipgt_dirname=cfg.clipgt_dirname,
            camera_name=cfg.camera_name,
            variant=cfg.scene_variant,
        )
        if (
            cfg.pipeline_config is None
            and cfg.pipeline_config_name not in OMNIDREAMS_CONFIGS
        ):
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unknown pipeline_config_name={cfg.pipeline_config_name!r}. "
                f"Supported: {supported}"
            )

        pipeline_cfg = (
            cfg.pipeline_config or OMNIDREAMS_CONFIGS[cfg.pipeline_config_name]
        )
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        if not isinstance(transformer_cfg, CosmosTransformerConfig):
            raise TypeError(
                "Omnidreams WebRTC requires a CosmosTransformerConfig pipeline."
            )
        if transformer_cfg.num_views != 1:
            raise ValueError(
                "Omnidreams WebRTC v1 only supports single-view configs; "
                f"{cfg.pipeline_config_name!r} has num_views={transformer_cfg.num_views}."
            )

        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Omnidreams WebRTC runtime.")

        logger.info("Loading Omnidreams first frame from {}", first_frame_path)
        image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (cfg.video_width, cfg.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        self._initial_rgb_frames = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.uint8)
        )

        prompt = prompt_path.read_text(encoding="utf-8").strip() or AV_POSITIVE_PROMPT
        self._text_prompts = [TextPrompt(positive=prompt)]

        loadable_clipgt_dir, self._clipgt_temp_dir = prepare_clipgt_dir(clipgt_dir)
        logger.info("Loading Omnidreams scene data from {}", loadable_clipgt_dir)
        scene_t0 = time.perf_counter()
        scene_data = load_scene(
            loadable_clipgt_dir,
            camera_names=[cfg.camera_name],
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=(cfg.video_height, cfg.video_width),
        )
        logger.info(
            "Loaded Omnidreams scene data in {:.1f}s; attaching Ludus scene.",
            time.perf_counter() - scene_t0,
        )
        ludus_t0 = time.perf_counter()
        scene_data = load_and_attach_ludus_scene(
            loadable_clipgt_dir,
            scene_data,
            device=self._device,
        )
        logger.info(
            "Attached Omnidreams Ludus scene in {:.1f}s.",
            time.perf_counter() - ludus_t0,
        )
        if not scene_data.ego_poses:
            raise ValueError(f"Scene {loadable_clipgt_dir} has no ego poses.")
        if cfg.camera_name not in scene_data.camera_models:
            raise ValueError(
                f"Camera {cfg.camera_name!r} was not loaded from {loadable_clipgt_dir}."
            )
        if cfg.camera_name not in scene_data.camera_extrinsics:
            raise ValueError(
                f"Camera {cfg.camera_name!r} has no extrinsics in {loadable_clipgt_dir}."
            )

        logger.info(
            "Setting up Omnidreams pipeline {} on {}. This may load checkpoints, "
            "compile modules, and initialize CUDA graphs.",
            cfg.pipeline_config_name,
            self._device,
        )
        pipeline_t0 = time.perf_counter()
        self._wrapper = OmnidreamsConditioningWrapper(
            pipeline_config_name=cfg.pipeline_config_name,
            pipeline_config=cfg.pipeline_config,
            resolution_wh=(cfg.video_width, cfg.video_height),
            seed_for_every_rollout=cfg.seed,
            device=self._device,
        )
        logger.info(
            "Omnidreams pipeline setup complete in {:.1f}s.",
            time.perf_counter() - pipeline_t0,
        )
        self._scene_data = scene_data
        logger.info("Creating Omnidreams renderer for camera {}", cfg.camera_name)
        renderer_t0 = time.perf_counter()
        self._renderer = self._wrapper.create_renderer(scene_data, [cfg.camera_name])
        logger.info(
            "Omnidreams renderer ready in {:.1f}s.",
            time.perf_counter() - renderer_t0,
        )
        self._camera_to_rig = torch.as_tensor(
            scene_data.camera_extrinsics[cfg.camera_name],
            device=self._device,
            dtype=torch.float32,
        )
        self._initial_ego_pose = scene_data.ego_poses[0].transformation_matrix
        self._next_timestamp_us = int(scene_data.ego_poses[0].timestamp)
        self._model_session = OmnidreamsConditioningSessionCore(
            wrapper=self._wrapper,
            output_stream_factory=lambda: VideoOutputStream(
                postprocess_stream=None,
                output_layout="bvtchw",
            ),
        )
        self._reset_rollout_sync()
        self._initialize_video_encoder_sync()
        logger.info(
            "Omnidreams runtime initialization complete in {:.1f}s.",
            time.perf_counter() - init_t0,
        )

    def _reset_rollout_sync(
        self, session_input: OmnidreamsSessionInput | None = None
    ) -> None:
        if (
            self._wrapper is None
            or self._model_session is None
            or self._renderer is None
        ):
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._initial_ego_pose is None or self._scene_data is None:
            raise OmnidreamsRuntimeError("Scene state is not initialized.")

        self._reset_postprocess_stream(session_input)
        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.pose_integrator.reset(self._initial_ego_pose)
        self._next_timestamp_us = int(self._scene_data.ego_poses[0].timestamp)
        self._wrapper.set_rollout_seed(self.config.seed)
        if self._text_prompts is None or self._initial_rgb_frames is None:
            raise OmnidreamsRuntimeError("Runtime conditioning is not initialized.")
        self._model_session.reset(
            renderer=self._renderer,
            text_prompts=self._text_prompts,
            initial_rgb_frames=self._initial_rgb_frames,
        )

    def _close_sync(self) -> None:
        model_session = self._model_session
        wrapper = self._wrapper
        self._model_session = None
        self._wrapper = None
        self._renderer = None
        self._scene_data = None
        self._initial_rgb_frames = None
        self._text_prompts = None
        self._camera_to_rig = None
        self._initial_ego_pose = None
        if model_session is not None:
            model_session.close()
        if wrapper is not None:
            del wrapper
        if self._clipgt_temp_dir is not None:
            self._clipgt_temp_dir.cleanup()
            self._clipgt_temp_dir = None

        if self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _reset_postprocess_stream(
        self, session_input: OmnidreamsSessionInput | None
    ) -> None:
        configured = self.config.postprocess
        preset = (
            session_input.postprocess_preset
            if session_input is not None
            and session_input.postprocess_preset is not None
            else configured.preset
        )
        if preset:
            validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=configured.preset,
            )
        postprocess = VideoPostprocessChainConfig(
            processors=configured.processors,
            preset=preset,
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._postprocess_preset = preset
        postprocess_stream = create_video_postprocess_stream(
            postprocess=postprocess,
            output_layout="bvtchw",
            fps=self.config.fps,
            per_view=False,
            world_size=world_size,
            is_rank_zero=self.is_master,
        )
        if self._model_session is None:
            raise OmnidreamsRuntimeError("Runtime model session is not initialized.")
        self._model_session.replace_output_stream(
            lambda: VideoOutputStream(
                postprocess_stream=postprocess_stream,
                output_layout="bvtchw",
            )
        )
        if postprocess_stream is not None:
            logger.info(
                "Omnidreams WebRTC post-processing enabled with preset {!r}.",
                preset,
            )

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> StepResult:
        if (
            self._wrapper is None
            or self._model_session is None
            or self._renderer is None
            or self._initial_rgb_frames is None
            or self._text_prompts is None
            or self._camera_to_rig is None
        ):
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        step_index = self._runtime_step_index()
        num_frames = self._next_input_frame_count()
        if len(frame_times) != num_frames:
            raise OmnidreamsRuntimeError(
                f"Expected {num_frames} frame_times for chunk={step_index}, "
                f"got {len(frame_times)}."
            )
        if not segments:
            raise OmnidreamsRuntimeError(f"Chunk={step_index} received empty segments.")

        ego_poses = self.pose_integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        ego_poses_t = torch.from_numpy(ego_poses).to(
            device=self._device, dtype=torch.float32
        )
        camera_poses = torch.einsum("nij,jk->nik", ego_poses_t, self._camera_to_rig)
        frame_timestamps_us = self._consume_timestamps(num_frames)

        serve_hdmaps = self.config.debug_serve_hdmaps
        try:
            return self._model_session.step(
                camera_names=[self.config.camera_name],
                camera_poses_per_view={self.config.camera_name: camera_poses},
                frame_timestamps_us=frame_timestamps_us,
                serve_hdmaps=serve_hdmaps,
                metadata={
                    "stream": "hdmap" if serve_hdmaps else "rgb",
                    "postprocess_preset": self._postprocess_preset,
                },
            )
        except RuntimeError as exc:
            raise OmnidreamsRuntimeError(str(exc)) from exc

    def _consume_timestamps(self, num_frames: int) -> list[int]:
        step_us = int(round(1_000_000 / self.config.fps))
        timestamps = [self._next_timestamp_us + i * step_us for i in range(num_frames)]
        self._next_timestamp_us += num_frames * step_us
        return timestamps


def create_omnidreams_webrtc_session_manager(
    *,
    runtime: OmnidreamsInferenceRuntime | None = None,
    runtime_config: OmnidreamsRuntimeConfig | None = None,
    client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
) -> BaseWebRTCSessionManager[
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
]:
    """Configure the shared WebRTC manager for the OmniDreams runtime."""
    runtime_config = runtime_config or getattr(runtime, "config", None)
    if not isinstance(runtime_config, OmnidreamsRuntimeConfig):
        runtime_config = OmnidreamsRuntimeConfig()
    runtime = runtime or OmnidreamsInferenceRuntime(config=runtime_config)
    return BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        identity=runtime_config.pipeline_config_name,
        busy_message="An Omnidreams session is already active.",
        warmup_label="Omnidreams WebRTC",
        supported_control_keys=WSAD_SUPPORTED_KEYS,
        fatal_generation_errors=True,
        client_liveness_timeout_s=client_liveness_timeout_s,
    )
