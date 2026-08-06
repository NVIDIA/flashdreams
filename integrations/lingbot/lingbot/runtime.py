# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot runtime API adapter and replay session implementation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    runner_artifact_path,
    write_runner_stats,
    write_video_tensor,
)
from flashdreams.infra.video_output import RunnerVideoOutputStream, VideoStepResult
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    OutputArtifact,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.types import StepRequest, StepResult, TimeWindow
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.encoder.utils import get_Ks_transformed, preprocess_example_poses
from lingbot.example_data import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    EXAMPLE_DATA_DIR_LOCAL,
    EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS,
    ensure_example_data_downloaded,
    example_asset_urls,
    example_data_dirname,
)

LINGBOT_MODEL_ID = "lingbot"
DEFAULT_LINGBOT_PRESET = "lingbot-world-fast-taehv-window15-sink3"
DEFAULT_PIXEL_HEIGHT = 464
DEFAULT_PIXEL_WIDTH = 832
DEFAULT_FPS = 16

_INTRINSICS_REFERENCE_HEIGHT = 480
_INTRINSICS_REFERENCE_WIDTH = 832
_INSTALL_HINT = "Install the lingbot plugin: pip install flashdreams-lingbot."

FIELD_PROMPT = "prompt"
FIELD_FIRST_FRAME_PATH = "first_frame_path"
FIELD_CAMERA_POSES_PATH = "camera_poses_path"
FIELD_CAMERA_INTRINSICS_PATH = "camera_intrinsics_path"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"
FIELD_WORLD_SCALE = "world_scale"

PipelineFactory = Callable[[Any, str], Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotReplayInputs:
    """Resolved model-facing Lingbot replay inputs."""

    prompt: str
    first_frame_path: Path
    camera_poses_path: Path
    camera_intrinsics_path: Path
    total_blocks: int = 20
    pixel_height: int = DEFAULT_PIXEL_HEIGHT
    pixel_width: int = DEFAULT_PIXEL_WIDTH
    fps: int = DEFAULT_FPS
    world_scale: float | None = None

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("LingbotReplayInputs.total_blocks must be > 0.")
        if self.pixel_height <= 0 or self.pixel_width <= 0:
            raise ValueError("LingbotReplayInputs pixel dimensions must be > 0.")
        if self.fps <= 0:
            raise ValueError("LingbotReplayInputs.fps must be > 0.")
        if self.world_scale is not None and self.world_scale <= 0:
            raise ValueError("LingbotReplayInputs.world_scale must be > 0.")
        object.__setattr__(self, "prompt", " ".join(self.prompt.split()))
        object.__setattr__(self, "first_frame_path", Path(self.first_frame_path))
        object.__setattr__(self, "camera_poses_path", Path(self.camera_poses_path))
        object.__setattr__(
            self,
            "camera_intrinsics_path",
            Path(self.camera_intrinsics_path),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotReplayRuntimeOptions:
    """Construction knobs for the Lingbot replay runtime."""

    pipeline_config: Any
    pipeline: Any | None = None
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = "tchw"


class LingbotModelAdapter:
    """Model adapter exposing Lingbot through ``flashdreams.runtime``."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[..., InferenceRuntime] | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or LingbotReplayRuntime
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()

    @property
    def model_id(self) -> str:
        return LINGBOT_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            description="Lingbot replay model inputs.",
            global_conditioning_fields=(
                InputField(
                    name=FIELD_PROMPT,
                    input_modality="text",
                    frequency_consumed="once",
                    description="Prompt text for the rollout.",
                ),
                InputField(
                    name=FIELD_FIRST_FRAME_PATH,
                    input_modality="image/path",
                    frequency_consumed="once",
                    description="First-frame RGB image path.",
                ),
                InputField(
                    name=FIELD_CAMERA_POSES_PATH,
                    input_modality="camera/poses-path",
                    frequency_consumed="per_step",
                    description="Path to [T,4,4] camera-to-world poses.",
                ),
                InputField(
                    name=FIELD_CAMERA_INTRINSICS_PATH,
                    input_modality="camera/intrinsics-path",
                    frequency_consumed="per_step",
                    description="Path to [T,4] camera intrinsics.",
                ),
                InputField(name=FIELD_TOTAL_BLOCKS, input_modality="count"),
                InputField(name=FIELD_PIXEL_HEIGHT, input_modality="pixel-height"),
                InputField(name=FIELD_PIXEL_WIDTH, input_modality="pixel-width"),
                InputField(name=FIELD_FPS, input_modality="fps"),
                InputField(
                    name=FIELD_WORLD_SCALE,
                    required=False,
                    input_modality="scale",
                    frequency_consumed="once",
                ),
            ),
        )

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return CanonicalInputSchema()

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"Lingbot adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self.pipeline_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self._runtime_factory(
            config=config,
            options=LingbotReplayRuntimeOptions(
                pipeline_config=self.pipeline_config(config),
                pipeline=config.runtime_options.get("pipeline"),
                pipeline_factory=self._pipeline_factory,
                output_layout=str(config.runtime_options.get("output_layout", "tchw")),
            ),
        )

    def preset_id(self, config: InferenceConfig | None) -> str:
        return (
            DEFAULT_LINGBOT_PRESET
            if config is None or config.preset_id is None
            else config.preset_id
        )

    def pipeline_config(self, config: InferenceConfig) -> Any:
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        preset_id = self.preset_id(config)
        from lingbot.config import PIPELINE_CONFIGS  # noqa: PLC0415

        try:
            return PIPELINE_CONFIGS[preset_id]
        except KeyError as exc:
            supported = ", ".join(sorted(PIPELINE_CONFIGS))
            raise ValueError(
                f"Unsupported Lingbot preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def default_replay_prompt(self, config: InferenceConfig | None) -> str:
        from lingbot.config import RUNNER_CONFIGS  # noqa: PLC0415

        runner = RUNNER_CONFIGS.get(self.preset_id(config))
        return "" if runner is None else str(getattr(runner, "prompt", ""))


class LingbotReplayRuntime:
    """Heavyweight Lingbot runtime consumed by the standard loop."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: LingbotReplayRuntimeOptions,
    ) -> None:
        self.config = config
        self.options = options
        if _is_torchrun_env() and not dist.is_initialized():
            init_distributed()

        if dist.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            device = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            device = config.device or "cuda"

        self.is_rank_zero = self.global_rank == 0
        if options.pipeline is not None:
            self.pipeline = options.pipeline
            self._owns_pipeline = False
        else:
            factory = options.pipeline_factory or _default_pipeline_factory
            self.pipeline = factory(options.pipeline_config, device)
            self._owns_pipeline = True

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        replay_inputs = replay_inputs_from_inference_input(inputs)
        return LingbotReplaySession(
            pipeline=self.pipeline,
            replay_inputs=replay_inputs,
            device=torch.device(f"cuda:{self.local_rank}")
            if dist.is_initialized()
            else torch.device(self.config.device or "cuda"),
            is_rank_zero=self.is_rank_zero,
            output_layout=self.options.output_layout,
        )

    def close(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if self._owns_pipeline and pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
            del self.pipeline
        device = torch.device(self.config.device or "cuda")
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class LingbotReplaySession:
    """One Lingbot rollout over resolved replay inputs."""

    def __init__(
        self,
        *,
        pipeline: Any,
        replay_inputs: LingbotReplayInputs,
        device: torch.device,
        is_rank_zero: bool,
        output_layout: VideoTensorLayout,
    ) -> None:
        self.pipeline = pipeline
        self.inputs = replay_inputs
        self.device = device
        self.is_rank_zero = is_rank_zero
        self.output_layout = output_layout
        self.dtype = torch.bfloat16
        self._closed = False
        self._step_index = 0
        self._frame_start = 0
        self._cache = self._initialize_cache()
        (
            self._camera_intrinsics,
            self._camera_poses,
            self._world_scale,
        ) = self._load_camera_controls()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if dist.is_initialized():
            dist.barrier()

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        if self._step_index >= self.inputs.total_blocks:
            return None
        num_frames = int(self.pipeline.get_num_output_frames(self._step_index))
        if self._frame_start + num_frames > self._camera_poses.shape[0]:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        if self._closed:
            raise RuntimeError("Lingbot replay session is closed.")

        step_index = self._step_index
        num_frames = int(self.pipeline.get_num_output_frames(step_index))
        frame_start = self._frame_start
        frame_end = frame_start + num_frames
        if frame_end > self._camera_poses.shape[0]:
            raise RuntimeError("Lingbot replay inputs ran out of camera frames.")

        if self.is_rank_zero:
            logger.info(
                "Lingbot runtime step {} frames=[{}, {})",
                step_index,
                frame_start,
                frame_end,
            )
        camctrl_input = CamCtrlInput(
            intrinsics=self._camera_intrinsics[frame_start:frame_end],
            poses=self._camera_poses[frame_start:frame_end],
            world_scale=self._world_scale,
        )
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self._cache,
            input=camctrl_input,
        )
        stats = self.pipeline.finalize(
            autoregressive_index=step_index,
            cache=self._cache,
        )
        elapsed_s = time.perf_counter() - start_t
        self._step_index += 1
        self._frame_start = frame_end

        metrics = _numeric_stats(stats)
        metrics.setdefault("model_step_s", elapsed_s)
        return StepResult(
            step_index=step_index,
            output=VideoStepResult.from_video_chunk(
                chunk_index=step_index,
                video_chunk=video_chunk,
                layout=self.output_layout,
                stats=metrics,
            ),
            frame_count=num_frames,
            output_window=TimeWindow(
                start_s=frame_start / self.inputs.fps,
                end_s=frame_end / self.inputs.fps,
            ),
            metrics=metrics,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            replay_inputs = replay_inputs_from_inference_input(inputs)
            if replay_inputs != self.inputs:
                raise ValueError("Lingbot replay reset cannot swap inputs.")
        cache = getattr(self, "_cache", None)
        if cache is not None:
            del self._cache
        self._cache = self._initialize_cache()
        self._step_index = 0
        self._frame_start = 0

    def close(self) -> None:
        self._closed = True
        cache = getattr(self, "_cache", None)
        if cache is not None:
            del self._cache

    def _initialize_cache(self) -> Any:
        first_frames = load_first_frame_tensor(
            self.inputs.first_frame_path,
            pixel_height=self.inputs.pixel_height,
            pixel_width=self.inputs.pixel_width,
            device=self.device,
            dtype=self.dtype,
            interpolation="cubic",
            install_hint=_INSTALL_HINT,
        )
        return self.pipeline.initialize_cache(
            text=[self.inputs.prompt],
            image=first_frames,
        )

    def _load_camera_controls(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        intrinsics = np.load(self.inputs.camera_intrinsics_path)
        intrinsics_t = torch.from_numpy(np.asarray(intrinsics)).to(
            device=self.device,
            dtype=torch.float32,
        )
        camera_intrinsics_t = get_Ks_transformed(
            intrinsics_t,
            height_org=_INTRINSICS_REFERENCE_HEIGHT,
            width_org=_INTRINSICS_REFERENCE_WIDTH,
            height_resize=self.inputs.pixel_height,
            width_resize=self.inputs.pixel_width,
            height_final=self.inputs.pixel_height,
            width_final=self.inputs.pixel_width,
        )

        poses = np.load(self.inputs.camera_poses_path)
        poses, inferred_world_scale = preprocess_example_poses(np.asarray(poses))
        camera_poses_t = torch.from_numpy(poses).to(
            device=self.device,
            dtype=torch.float32,
        )
        if self.is_rank_zero:
            logger.info(
                "Loaded Lingbot camera controls intrinsics={} poses={}",
                tuple(camera_intrinsics_t.shape),
                tuple(camera_poses_t.shape),
            )
        return (
            camera_intrinsics_t,
            camera_poses_t,
            float(
                self.inputs.world_scale
                if self.inputs.world_scale is not None
                else inferred_world_scale
            ),
        )


@dataclass(slots=True)
class LingbotRunnerOutputTarget:
    """Runner-compatible MP4/stats output target for Lingbot replay results."""

    output_stream: RunnerVideoOutputStream
    output_dir: Path
    runner_name: str
    fps: int | float
    install_hint: str = _INSTALL_HINT
    _opened: bool = False

    def open(self) -> None:
        self._opened = True

    def write(self, result: StepResult) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed Lingbot output target.")
        video_result = result.output
        if not isinstance(video_result, VideoStepResult):
            raise TypeError(
                "LingbotRunnerOutputTarget requires VideoStepResult output, "
                f"got {type(video_result).__name__}."
            )
        self.output_stream.process(
            video_result.video_chunk,
            autoregressive_index=video_result.chunk_index,
            stats=video_result.stats or dict(result.metrics),
        )

    def close(self) -> tuple[OutputArtifact, ...]:
        self._opened = False
        artifacts: list[OutputArtifact] = []
        video = self.output_stream.finish()
        if video is None:
            return ()

        ensure_output_dir(self.output_dir)
        video_path = runner_artifact_path(self.output_dir, self.runner_name, "mp4")
        write_video_tensor(
            video,
            video_path,
            fps=self.fps,
            layout="tchw",
            install_hint=self.install_hint,
        )
        logger.info(
            "[{}] wrote video {} -> {}",
            self.runner_name,
            tuple(video.shape),
            video_path.resolve(),
        )
        artifacts.append(
            OutputArtifact(kind="video/mp4", uri=str(video_path.resolve()))
        )
        if self.output_stream.stats_history:
            stats_path = write_runner_stats(
                self.output_dir,
                self.runner_name,
                self.output_stream.stats_history,
            )
            logger.info(
                "[{}] wrote per-AR-step stats -> {}",
                self.runner_name,
                stats_path.resolve(),
            )
            artifacts.append(
                OutputArtifact(kind="application/json", uri=str(stats_path.resolve()))
            )
        return tuple(artifacts)


def inference_config_from_runner_config(
    runner_config: Any,
    *,
    device: str,
    pipeline: Any | None = None,
) -> InferenceConfig:
    """Build the runtime config directly from a Lingbot runner config."""
    runtime_options: dict[str, Any] = {
        "pipeline_config": runner_config.pipeline,
        "output_layout": runner_config.postprocess_output_layout or "tchw",
    }
    if pipeline is not None:
        runtime_options["pipeline"] = pipeline
    compile_network = getattr(
        runner_config.pipeline.diffusion_model.transformer,
        "compile_network",
        None,
    )
    return InferenceConfig(
        model_id=LINGBOT_MODEL_ID,
        preset_id=str(runner_config.pipeline.name),
        device=device,
        compile=None if compile_network is None else bool(compile_network),
        runtime_options=runtime_options,
    )


def inference_input_from_runner_config(
    runner_config: Any,
    *,
    is_rank_zero: bool,
) -> InferenceInput:
    """Build model-facing runtime inputs directly from a Lingbot runner config."""
    replay_inputs = replay_inputs_from_mapping(
        {
            FIELD_PROMPT: getattr(runner_config, "prompt", ""),
            "prompt_path": getattr(runner_config, "prompt_path", None),
            FIELD_FIRST_FRAME_PATH: getattr(runner_config, "image_path", None),
            FIELD_CAMERA_POSES_PATH: getattr(runner_config, "pose_path", None),
            FIELD_CAMERA_INTRINSICS_PATH: getattr(
                runner_config,
                "intrinsic_path",
                None,
            ),
            FIELD_TOTAL_BLOCKS: getattr(runner_config, "total_blocks", 20),
            FIELD_PIXEL_HEIGHT: getattr(
                runner_config,
                "pixel_height",
                DEFAULT_PIXEL_HEIGHT,
            ),
            FIELD_PIXEL_WIDTH: getattr(
                runner_config,
                "pixel_width",
                DEFAULT_PIXEL_WIDTH,
            ),
            FIELD_FPS: getattr(runner_config, "fps", DEFAULT_FPS),
            "example_data": getattr(runner_config, "example_data", False),
            "example_idx": getattr(runner_config, "example_idx", 0),
        },
        is_rank_zero=is_rank_zero,
    )
    return inference_input_from_replay_inputs(replay_inputs)


def inference_input_from_replay_inputs(
    replay_inputs: LingbotReplayInputs,
) -> InferenceInput:
    """Encode resolved Lingbot replay inputs into ``InferenceInput``."""
    payload: dict[str, Any] = {
        FIELD_PROMPT: replay_inputs.prompt,
        FIELD_FIRST_FRAME_PATH: replay_inputs.first_frame_path,
        FIELD_CAMERA_POSES_PATH: replay_inputs.camera_poses_path,
        FIELD_CAMERA_INTRINSICS_PATH: replay_inputs.camera_intrinsics_path,
        FIELD_TOTAL_BLOCKS: replay_inputs.total_blocks,
        FIELD_PIXEL_HEIGHT: replay_inputs.pixel_height,
        FIELD_PIXEL_WIDTH: replay_inputs.pixel_width,
        FIELD_FPS: replay_inputs.fps,
    }
    if replay_inputs.world_scale is not None:
        payload[FIELD_WORLD_SCALE] = replay_inputs.world_scale
    return InferenceInput(global_conditioning=payload)


def replay_inputs_from_inference_input(inputs: InferenceInput) -> LingbotReplayInputs:
    """Decode and validate model-facing Lingbot replay inputs."""
    missing = LingbotModelAdapter().inference_input_schema.missing_global_conditioning(
        inputs
    )
    if missing:
        raise ValueError(f"Lingbot replay inputs missing required fields: {missing}.")
    gc = inputs.global_conditioning
    return LingbotReplayInputs(
        prompt=str(gc[FIELD_PROMPT]),
        first_frame_path=Path(gc[FIELD_FIRST_FRAME_PATH]),
        camera_poses_path=Path(gc[FIELD_CAMERA_POSES_PATH]),
        camera_intrinsics_path=Path(gc[FIELD_CAMERA_INTRINSICS_PATH]),
        total_blocks=int(gc[FIELD_TOTAL_BLOCKS]),
        pixel_height=int(gc[FIELD_PIXEL_HEIGHT]),
        pixel_width=int(gc[FIELD_PIXEL_WIDTH]),
        fps=int(gc[FIELD_FPS]),
        world_scale=(
            None
            if FIELD_WORLD_SCALE not in gc or gc[FIELD_WORLD_SCALE] is None
            else float(gc[FIELD_WORLD_SCALE])
        ),
    )


def replay_inputs_from_mapping(
    value: Any,
    *,
    default_prompt: str = "",
    is_rank_zero: bool = True,
) -> LingbotReplayInputs:
    """Resolve app/CLI replay values into direct Lingbot runtime inputs."""
    if isinstance(value, LingbotReplayInputs):
        _require_existing_replay_paths(value)
        return value
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Lingbot replay inputs must be a LingbotReplayInputs, mapping, or None."
        )

    example_idx = int(value.get("example_idx", 0))
    if example_idx not in EXAMPLE_DATA_AVAILABLE_IDXS:
        raise ValueError(
            f"Lingbot replay example_idx must be one of {EXAMPLE_DATA_AVAILABLE_IDXS}."
        )

    first_frame_path = _optional_path(
        value.get(FIELD_FIRST_FRAME_PATH, value.get("image_path"))
    )
    poses_path = _optional_path(
        value.get(FIELD_CAMERA_POSES_PATH, value.get("pose_path"))
    )
    intrinsics_path = _optional_path(
        value.get(FIELD_CAMERA_INTRINSICS_PATH, value.get("intrinsic_path"))
    )
    prompt_path = _optional_path(value.get("prompt_path"))
    example_data = _resolve_example_data_default(value)

    if example_data and (
        first_frame_path is None
        or poses_path is None
        or intrinsics_path is None
        or (
            prompt_path is None
            and not _has_nonempty_value(value, FIELD_PROMPT)
            and example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        )
    ):
        example_dir = ensure_example_data_downloaded(
            is_rank_zero=is_rank_zero,
            example_idx=example_idx,
        )
        first_frame_path = first_frame_path or example_dir / "image.jpg"
        poses_path = poses_path or example_dir / "poses.npy"
        intrinsics_path = intrinsics_path or example_dir / "intrinsics.npy"
        if (
            prompt_path is None
            and not _has_nonempty_value(value, FIELD_PROMPT)
            and example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        ):
            prompt_path = example_dir / "prompt.txt"

    replay_inputs = LingbotReplayInputs(
        prompt=_resolve_prompt(
            value,
            prompt_path=prompt_path,
            default_prompt=default_prompt,
        ),
        first_frame_path=_require_path_value(
            first_frame_path,
            label=FIELD_FIRST_FRAME_PATH,
        ),
        camera_poses_path=_require_path_value(
            poses_path,
            label=FIELD_CAMERA_POSES_PATH,
        ),
        camera_intrinsics_path=_require_path_value(
            intrinsics_path,
            label=FIELD_CAMERA_INTRINSICS_PATH,
        ),
        total_blocks=int(value.get(FIELD_TOTAL_BLOCKS, 20)),
        pixel_height=int(value.get(FIELD_PIXEL_HEIGHT, DEFAULT_PIXEL_HEIGHT)),
        pixel_width=int(value.get(FIELD_PIXEL_WIDTH, DEFAULT_PIXEL_WIDTH)),
        fps=int(value.get(FIELD_FPS, DEFAULT_FPS)),
        world_scale=(
            None
            if FIELD_WORLD_SCALE not in value or value[FIELD_WORLD_SCALE] is None
            else float(value[FIELD_WORLD_SCALE])
        ),
    )
    _require_existing_replay_paths(replay_inputs)
    if prompt_path is not None:
        _require_existing_path(prompt_path, label="prompt_path")
    return replay_inputs


def build_lingbot_webrtc_runtime_config(
    *,
    preset_id: str,
    pipeline_config: Any,
    device: str,
    seed: int,
    compile_network: bool,
    context_parallel_size: int,
    video_height: int,
    video_width: int,
    fps: int,
    warmup_chunks: int,
    warmup_timeout_s: float,
    example_idx: int,
    prefer_sw_encoder: bool,
    runtime_options: Mapping[str, Any] | None = None,
) -> Any:
    """Build the Lingbot WebRTC runtime config from shared runtime inputs."""
    from lingbot.webrtc.session import LingbotRuntimeConfig  # noqa: PLC0415

    example_dirname = example_data_dirname(example_idx)
    example_dir = EXAMPLE_DATA_DIR_LOCAL / example_dirname
    if (
        example_idx == 0
        and not example_dir.exists()
        and (EXAMPLE_DATA_DIR_LOCAL / "image.jpg").exists()
    ):
        example_dir = EXAMPLE_DATA_DIR_LOCAL
    urls = example_asset_urls(example_idx)
    runtime_config = LingbotRuntimeConfig(
        config_name=preset_id,
        pipeline_config=pipeline_config,
        compile_network=compile_network,
        seed=seed,
        context_parallel_size=context_parallel_size,
        device=device,
        video_height=video_height,
        video_width=video_width,
        fps=fps,
        warmup_chunks=warmup_chunks,
        warmup_timeout_s=warmup_timeout_s,
        encoder_backend="default" if prefer_sw_encoder else "auto",
        example_data_dir=example_dir,
        default_image_url=urls["image"],
        default_intrinsics_url=urls["intrinsics"],
        default_poses_url=urls["poses"],
    )
    return _apply_webrtc_runtime_options(runtime_config, runtime_options or {})


def _apply_webrtc_runtime_options(runtime_config: Any, options: Mapping[str, Any]) -> Any:
    overrides: dict[str, Any] = {}
    for name in (
        "world_scale",
        "default_intrinsics",
        "default_prompt",
        "default_image_url",
        "default_intrinsics_url",
        "default_poses_url",
        "encoder_bitrate_bps",
        "encoder_gop",
        "text_events",
    ):
        if name in options:
            overrides[name] = options[name]
    return replace(runtime_config, **overrides) if overrides else runtime_config


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _numeric_stats(stats: Any) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(key): value
        for key, value in stats.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def _resolve_prompt(
    value: Mapping[str, Any],
    *,
    prompt_path: Path | None,
    default_prompt: str,
) -> str:
    prompt = str(value.get(FIELD_PROMPT, value.get("prompt", ""))).strip()
    if prompt:
        return prompt
    if prompt_path is not None:
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
        if lines:
            prompt = lines[0].strip()
            if prompt:
                return prompt
    return default_prompt.strip()


def _resolve_example_data_default(value: Mapping[str, Any]) -> bool:
    explicit = value.get("example_data")
    if explicit is not None:
        return _bool_value(explicit)
    return not (
        _has_nonempty_value(value, FIELD_FIRST_FRAME_PATH)
        or _has_nonempty_value(value, "image_path")
    ) or not (
        _has_nonempty_value(value, FIELD_CAMERA_POSES_PATH)
        or _has_nonempty_value(value, "pose_path")
    ) or not (
        _has_nonempty_value(value, FIELD_CAMERA_INTRINSICS_PATH)
        or _has_nonempty_value(value, "intrinsic_path")
    )


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _has_nonempty_value(value: Mapping[str, Any], key: str) -> bool:
    if key not in value:
        return False
    raw = value[key]
    return raw is not None and raw != ""


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _require_path_value(value: Path | None, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"Lingbot replay inputs require {label}.")
    return value


def _require_existing_replay_paths(replay_inputs: LingbotReplayInputs) -> None:
    _require_existing_path(replay_inputs.first_frame_path, label=FIELD_FIRST_FRAME_PATH)
    _require_existing_path(replay_inputs.camera_poses_path, label=FIELD_CAMERA_POSES_PATH)
    _require_existing_path(
        replay_inputs.camera_intrinsics_path,
        label=FIELD_CAMERA_INTRINSICS_PATH,
    )


def _require_existing_path(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Lingbot replay inputs missing {label}: {path}")


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_LINGBOT_PRESET",
    "DEFAULT_PIXEL_HEIGHT",
    "DEFAULT_PIXEL_WIDTH",
    "FIELD_CAMERA_INTRINSICS_PATH",
    "FIELD_CAMERA_POSES_PATH",
    "FIELD_FIRST_FRAME_PATH",
    "FIELD_FPS",
    "FIELD_PIXEL_HEIGHT",
    "FIELD_PIXEL_WIDTH",
    "FIELD_PROMPT",
    "FIELD_TOTAL_BLOCKS",
    "FIELD_WORLD_SCALE",
    "LINGBOT_MODEL_ID",
    "LingbotModelAdapter",
    "LingbotReplayInputs",
    "LingbotReplayRuntime",
    "LingbotReplayRuntimeOptions",
    "LingbotReplaySession",
    "LingbotRunnerOutputTarget",
    "PipelineFactory",
    "build_lingbot_webrtc_runtime_config",
    "inference_config_from_runner_config",
    "inference_input_from_replay_inputs",
    "inference_input_from_runner_config",
    "replay_inputs_from_inference_input",
    "replay_inputs_from_mapping",
]
