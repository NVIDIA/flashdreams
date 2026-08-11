# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native runtime API implementation for FlashVSR video upscaling."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
)
from flashdreams.runtime.demo.outputs import SessionInfo
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.types import StepRequest, StepRequirements, StepResult

FLASHVSR_MODEL_ID = "flashvsr"
DEFAULT_FLASHVSR_PRESET = "flashvsr-v1.1-sparse-ratio-2.0"

FIELD_INPUT_HEIGHT = "input_height"
FIELD_INPUT_WIDTH = "input_width"
FIELD_FPS = "fps"
FIELD_TOTAL_FRAMES = "total_frames"
FIELD_CHUNK_SIZE = "chunk_size"
FIELD_TAIL_POLICY = "tail_policy"
FIELD_VIDEO_CHUNK = "video_chunk"
FIELD_VALID_FRAME_COUNT = "valid_frame_count"

TailPolicy = Literal["drop", "pad"]
PipelineFactory = Callable[[Any, str], Any]

_CHUNK_MODES: dict[int, tuple[int, int]] = {
    8: (5, 8),
    16: (13, 16),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class FlashVSRSessionInputs:
    """Session-wide shape and timing information for one input video."""

    input_height: int
    input_width: int
    fps: float
    chunk_size: Literal[8, 16]
    total_frames: int | None = None
    tail_policy: TailPolicy = "drop"

    def __post_init__(self) -> None:
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("FlashVSR input dimensions must be > 0.")
        if self.fps <= 0:
            raise ValueError("FlashVSR fps must be > 0.")
        if self.chunk_size not in _CHUNK_MODES:
            raise ValueError("FlashVSR chunk_size must be 8 or 16.")
        if self.total_frames is not None and self.total_frames <= 0:
            raise ValueError("FlashVSR total_frames must be > 0 when provided.")
        if self.tail_policy not in {"drop", "pad"}:
            raise ValueError("FlashVSR tail_policy must be 'drop' or 'pad'.")

    @property
    def cold_frame_count(self) -> int:
        """Return the raw frame count required by the first model step."""
        return _CHUNK_MODES[self.chunk_size][0]

    @property
    def steady_frame_count(self) -> int:
        """Return the raw frame count required after the first model step."""
        return _CHUNK_MODES[self.chunk_size][1]


@dataclass(frozen=True, kw_only=True, slots=True)
class FlashVSRRuntimeOptions:
    """Construction options for the reusable FlashVSR runtime."""

    pipeline_config: Any
    sparse_ratio: float = 2.0
    scale: Literal[2, 4] = 2
    pipeline: Any | None = None
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = "bcthw"
    compile_network: bool | None = None
    use_cuda_graph: bool | None = None
    color_corrector_implementation: Literal["cuda", "torch"] | None = None

    def __post_init__(self) -> None:
        if self.sparse_ratio <= 0:
            raise ValueError("FlashVSR sparse_ratio must be > 0.")
        if self.scale not in {2, 4}:
            raise ValueError("FlashVSR scale must be 2 or 4.")
        if self.output_layout != "bcthw":
            raise ValueError("FlashVSR native runtime output_layout must be 'bcthw'.")


class FlashVSRModelAdapter:
    """Expose FlashVSR through the transport-neutral inference runtime API."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[..., InferenceRuntime] | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or FlashVSRInferenceRuntime
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()

    @property
    def model_id(self) -> str:
        return FLASHVSR_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            description="FlashVSR streaming RGB video inputs.",
            global_conditioning_fields=(
                InputField(
                    name=FIELD_INPUT_HEIGHT,
                    input_modality="pixel-height",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_INPUT_WIDTH,
                    input_modality="pixel-width",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_FPS,
                    input_modality="fps",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_CHUNK_SIZE,
                    input_modality="frame-count",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_TOTAL_FRAMES,
                    required=False,
                    input_modality="frame-count",
                    frequency_consumed="once",
                    description="Absent for an unbounded looping source.",
                ),
                InputField(
                    name=FIELD_TAIL_POLICY,
                    input_modality="policy",
                    frequency_consumed="once",
                ),
            ),
            step_fields=(
                InputField(
                    name=FIELD_VIDEO_CHUNK,
                    input_modality="video/rgb",
                    frequency_consumed="per_step",
                    metadata={
                        "shape": "[B,3,T,H,W]",
                        "layout": "bcthw",
                        "range": "[-1,1]",
                    },
                    description="One normalized low-resolution RGB chunk.",
                ),
            ),
        )

    @property
    def canonical_input_schema(self) -> None:
        return None

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def preset_id(self, config: InferenceConfig | None) -> str:
        """Return the requested preset or the stable FlashVSR default."""
        if config is None or config.preset_id is None:
            return DEFAULT_FLASHVSR_PRESET
        return config.preset_id

    def pipeline_config(self, config: InferenceConfig) -> Any:
        """Resolve the dimension-independent pipeline scaffold."""
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        from flashvsr.config import RUNNER_CONFIGS  # noqa: PLC0415

        preset_id = self.preset_id(config)
        try:
            return RUNNER_CONFIGS[preset_id].pipeline
        except KeyError as exc:
            supported = ", ".join(sorted(RUNNER_CONFIGS))
            raise ValueError(
                f"Unsupported FlashVSR preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"FlashVSR adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self.pipeline_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        pipeline_config = self.pipeline_config(config)
        options = config.runtime_options
        configured_scale = getattr(
            getattr(pipeline_config, "encoder", None),
            "scale",
            2,
        )
        scale = int(options.get("scale", configured_scale))
        sparse_ratio = options.get("sparse_ratio")
        if sparse_ratio is None:
            from flashvsr.config import RUNNER_CONFIGS  # noqa: PLC0415

            runner = RUNNER_CONFIGS.get(self.preset_id(config))
            sparse_ratio = getattr(runner, "sparse_ratio", 2.0)
        return self._runtime_factory(
            config=config,
            options=FlashVSRRuntimeOptions(
                pipeline_config=pipeline_config,
                sparse_ratio=float(sparse_ratio),
                scale=cast(Literal[2, 4], scale),
                pipeline=options.get("pipeline"),
                pipeline_factory=self._pipeline_factory,
                compile_network=config.compile,
                use_cuda_graph=_optional_bool(options.get("use_cuda_graph")),
                color_corrector_implementation=_optional_color_corrector(
                    options.get("color_corrector_implementation")
                ),
            ),
        )


class FlashVSRInferenceRuntime:
    """Own one reusable, resolution-specific FlashVSR model pipeline."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: FlashVSRRuntimeOptions,
    ) -> None:
        self.config = config
        self.options = options
        if _is_torchrun_env() and not dist.is_initialized():
            init_distributed()
        if dist.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device(config.device or "cuda")
        if self.world_size != 1:
            raise NotImplementedError(
                "The native FlashVSR demo runtime currently supports one GPU. "
                "Use the existing full-attention runner for context parallelism."
            )

        self.is_rank_zero = self.global_rank == 0
        self.pipeline: Any | None = options.pipeline
        self._owns_pipeline = options.pipeline is None
        self._pipeline_shape: tuple[int, int] | None = None
        self._reusable_cache: Any | None = None
        self._active_session: FlashVSRInferenceSession | None = None
        self._closed = False

    def preload(self) -> None:
        """Keep the lazy runtime host hook explicit; dimensions arrive per session."""
        if self._closed:
            raise RuntimeError("FlashVSR runtime is closed.")

    def peek_input_fps(self) -> float:
        """Return transport timing before a concrete session has started."""
        return float(self.config.runtime_options.get("fps", 30.0))

    def peek_steady_output_num_frames(self) -> int:
        """Return the steady chunk size used to bound WebRTC output queues."""
        return int(self.config.runtime_options.get("chunk_size", 16))

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        if self._closed:
            raise RuntimeError("FlashVSR runtime is closed.")
        if self._active_session is not None:
            raise RuntimeError("FlashVSR runtime already has an active session.")
        session_inputs = session_inputs_from_inference_input(inputs)
        pipeline = self._pipeline_for(session_inputs)
        cache = self._acquire_cache(pipeline)
        _reset_pipeline_rng(pipeline, self.config.seed)
        session = FlashVSRInferenceSession(
            pipeline=pipeline,
            cache=cache,
            inputs=session_inputs,
            device=self.device,
            output_layout=self.options.output_layout,
            rollout_seed=self.config.seed,
            on_close=self._release_session,
        )
        self._active_session = session
        return session

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_session is not None:
            self._active_session.close()
        pipeline = self.pipeline
        self.pipeline = None
        self._reusable_cache = None
        if self._owns_pipeline and pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pipeline_for(self, inputs: FlashVSRSessionInputs) -> Any:
        shape = (inputs.input_height, inputs.input_width)
        if self.pipeline is not None:
            if self._pipeline_shape is None:
                self._pipeline_shape = shape
            elif self._pipeline_shape != shape:
                if not self._owns_pipeline:
                    raise ValueError(
                        "An injected FlashVSR pipeline cannot change resolution: "
                        f"loaded {self._pipeline_shape}, requested {shape}."
                    )
                self._dispose_pipeline()
            if self.pipeline is not None:
                return self.pipeline

        pipeline_config = self._derive_pipeline_config(inputs)
        factory = self.options.pipeline_factory or _default_pipeline_factory
        logger.info(
            "Building native FlashVSR runtime for input={}x{} scale={}.",
            inputs.input_height,
            inputs.input_width,
            self.options.scale,
        )
        self.pipeline = factory(pipeline_config, str(self.device))
        self._pipeline_shape = shape
        return self.pipeline

    def _derive_pipeline_config(self, inputs: FlashVSRSessionInputs) -> Any:
        scale = self.options.scale
        target_height = (inputs.input_height * scale // 128) * 128
        target_width = (inputs.input_width * scale // 128) * 128
        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                "FlashVSR scaled dimensions must each contain a 128-pixel block; "
                f"got input={inputs.input_height}x{inputs.input_width}, scale={scale}."
            )
        topk_ratio = (
            self.options.sparse_ratio * 768 * 1280 / (target_height * target_width)
        )
        encoder_updates: dict[str, Any] = {
            "input_H": inputs.input_height,
            "input_W": inputs.input_width,
            "scale": scale,
        }
        decoder_updates: dict[str, Any] = {}
        transformer_updates: dict[str, Any] = {"topk_ratio": topk_ratio}
        if self.options.compile_network is not None:
            enabled = self.options.compile_network
            encoder_updates["use_compile"] = enabled
            decoder_updates["use_compile"] = enabled
            transformer_updates["compile_network"] = enabled
        if self.options.use_cuda_graph is not None:
            enabled = self.options.use_cuda_graph
            encoder_updates["use_cuda_graph"] = enabled
            decoder_updates["use_cuda_graph"] = enabled
            transformer_updates["use_cuda_graph"] = enabled
        if self.options.color_corrector_implementation is not None:
            decoder_updates["color_corrector_implementation"] = (
                self.options.color_corrector_implementation
            )
        diffusion_updates: dict[str, Any] = {"transformer": transformer_updates}
        if self.config.seed is not None:
            diffusion_updates["seed"] = int(self.config.seed)
        return derive_config(
            self.options.pipeline_config,
            encoder=encoder_updates,
            decoder=decoder_updates,
            diffusion_model=diffusion_updates,
        )

    def _acquire_cache(self, pipeline: Any) -> Any:
        cache = self._reusable_cache
        self._reusable_cache = None
        if cache is None:
            return pipeline.initialize_cache()
        reset = getattr(pipeline, "reset_cache_in_place", None)
        if not callable(reset):
            return pipeline.initialize_cache()
        reset(cache)
        return cache

    def _release_session(
        self,
        session: "FlashVSRInferenceSession",
        cache: Any,
    ) -> None:
        if self._active_session is session:
            self._active_session = None
            if not self._closed:
                self._reusable_cache = cache

    def _dispose_pipeline(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self._pipeline_shape = None
        self._reusable_cache = None
        if pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()


class FlashVSRInferenceSession:
    """Run one FlashVSR video stream from new-API frame chunks."""

    def __init__(
        self,
        *,
        pipeline: Any,
        cache: Any,
        inputs: FlashVSRSessionInputs,
        device: torch.device,
        output_layout: VideoTensorLayout,
        rollout_seed: int | None,
        on_close: Callable[["FlashVSRInferenceSession", Any], None],
    ) -> None:
        self.pipeline = pipeline
        self.cache = cache
        self.inputs = inputs
        self.device = device
        self.output_layout = output_layout
        self.rollout_seed = rollout_seed
        self._on_close = on_close
        self._step_index = 0
        self._frame_start = 0
        self._closed = False

    def session_info(self) -> SessionInfo:
        """Return video-output facts known after cache initialization."""
        return SessionInfo(
            output_layout=self.output_layout,
            steady_output_frame_count=self.inputs.steady_frame_count,
            metadata={
                "input_height": self.inputs.input_height,
                "input_width": self.inputs.input_width,
            },
        )

    def next_step_requirements(self) -> StepRequirements | None:
        if self._closed:
            return None
        requested = (
            self.inputs.cold_frame_count
            if self._step_index == 0
            else self.inputs.steady_frame_count
        )
        valid = requested
        if self.inputs.total_frames is not None:
            remaining = self.inputs.total_frames - self._frame_start
            if remaining <= 0:
                return None
            if remaining < requested:
                if self.inputs.tail_policy == "drop":
                    return None
                valid = remaining
        return StepRequirements(
            step_index=self._step_index,
            input_frame_count=requested,
            steady_output_frame_count=self.inputs.steady_frame_count,
            inference_input_schema=FlashVSRModelAdapter().inference_input_schema,
            metadata={
                FIELD_VALID_FRAME_COUNT: valid,
                "frame_start": self._frame_start,
            },
        )

    def next_step_request(self) -> StepRequest | None:
        """Expose the protocol-compatible request alongside the native requirements."""
        requirements = self.next_step_requirements()
        if requirements is None:
            return None
        metadata = dict(requirements.metadata)
        metadata["input_frame_count"] = requirements.input_frame_count
        metadata["steady_output_frame_count"] = requirements.steady_output_frame_count
        return StepRequest(
            step_index=requirements.step_index,
            inference_input_schema=requirements.inference_input_schema,
            metadata=metadata,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        if self._closed:
            raise RuntimeError("FlashVSR inference session is closed.")
        request = self.next_step_requirements()
        if request is None:
            raise RuntimeError("FlashVSR inference session has no remaining step.")
        valid = int(
            inputs.metadata.get(
                FIELD_VALID_FRAME_COUNT,
                request.metadata[FIELD_VALID_FRAME_COUNT],
            )
        )
        expected_valid = int(request.metadata[FIELD_VALID_FRAME_COUNT])
        if valid != expected_valid:
            raise ValueError(
                "FlashVSR valid frame count mismatch: "
                f"expected {expected_valid}, got {valid}."
            )
        video = _require_video_chunk(
            inputs,
            expected_frames=request.input_frame_count,
            expected_height=self.inputs.input_height,
            expected_width=self.inputs.input_width,
        )
        dtype = self.pipeline.diffusion_model.dtype
        video = video.to(device=self.device, dtype=dtype)
        step_index = self._step_index
        output = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self.cache,
            input=video,
        )
        stats = self.pipeline.finalize(
            autoregressive_index=step_index,
            cache=self.cache,
        )
        if output.ndim != 5 or output.shape[2] < valid:
            raise ValueError(
                "FlashVSR pipeline output must be [B,C,T,H,W] with at least "
                f"{valid} frames; got {tuple(output.shape)}."
            )
        if output.shape[2] != valid:
            output = output[:, :, :valid]
        frame_start = self._frame_start
        frame_end = frame_start + valid
        self._frame_start = frame_end
        self._step_index += 1
        return StepResult.from_video_chunk(
            step_index=step_index,
            video_chunk=output,
            layout=self.output_layout,
            output_window=TimeWindow(
                start_s=frame_start / self.inputs.fps,
                end_s=frame_end / self.inputs.fps,
            ),
            metrics=_numeric_metrics(stats),
            metadata={
                "input_frame_count": request.input_frame_count,
                FIELD_VALID_FRAME_COUNT: valid,
                "resolution": {
                    "width": int(output.shape[-1]),
                    "height": int(output.shape[-2]),
                },
            },
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if self._closed:
            raise RuntimeError("FlashVSR inference session is closed.")
        if inputs is not None:
            replacement = session_inputs_from_inference_input(inputs)
            if replacement != self.inputs:
                raise ValueError(
                    "FlashVSR session reset cannot change video shape or timing."
                )
        reset = getattr(self.pipeline, "reset_cache_in_place", None)
        if not callable(reset):
            raise RuntimeError(
                "FlashVSR pipeline does not support in-place cache reset."
            )
        reset(self.cache)
        _reset_pipeline_rng(self.pipeline, self.rollout_seed)
        self._step_index = 0
        self._frame_start = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close(self, self.cache)


def session_inputs_from_inference_input(
    inputs: InferenceInput,
) -> FlashVSRSessionInputs:
    """Validate and decode session-global FlashVSR inputs."""
    values = inputs.global_conditioning
    required = (
        FIELD_INPUT_HEIGHT,
        FIELD_INPUT_WIDTH,
        FIELD_FPS,
        FIELD_CHUNK_SIZE,
        FIELD_TAIL_POLICY,
    )
    missing = tuple(name for name in required if name not in values)
    if missing:
        raise ValueError(f"Missing FlashVSR global conditioning field(s): {missing}.")
    chunk_size = int(values[FIELD_CHUNK_SIZE])
    return FlashVSRSessionInputs(
        input_height=int(values[FIELD_INPUT_HEIGHT]),
        input_width=int(values[FIELD_INPUT_WIDTH]),
        fps=float(values[FIELD_FPS]),
        chunk_size=cast(Literal[8, 16], chunk_size),
        total_frames=(
            None
            if values.get(FIELD_TOTAL_FRAMES) is None
            else int(values[FIELD_TOTAL_FRAMES])
        ),
        tail_policy=cast(TailPolicy, str(values[FIELD_TAIL_POLICY])),
    )


def _require_video_chunk(
    inputs: InferenceInput,
    *,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> torch.Tensor:
    value = inputs.step.get(FIELD_VIDEO_CHUNK)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"FlashVSR step input {FIELD_VIDEO_CHUNK!r} must be a torch.Tensor."
        )
    expected = (1, 3, expected_frames, expected_height, expected_width)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"FlashVSR video chunk must have shape {expected}, got {tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise TypeError("FlashVSR video chunk must use a floating-point dtype.")
    return value


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _reset_pipeline_rng(pipeline: Any, seed: int | None) -> None:
    if seed is None:
        return
    rng = getattr(getattr(pipeline, "diffusion_model", None), "rng", None)
    if rng is not None:
        rng.manual_seed(int(seed))


def _numeric_metrics(value: Any) -> dict[str, float | int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): metric
        for key, metric in value.items()
        if isinstance(metric, int | float) and not isinstance(metric, bool)
    }


def _optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _optional_color_corrector(
    value: Any,
) -> Literal["cuda", "torch"] | None:
    if value is None:
        return None
    if value not in {"cuda", "torch"}:
        raise ValueError("color_corrector_implementation must be 'cuda' or 'torch'.")
    return cast(Literal["cuda", "torch"], value)


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


__all__ = [
    "DEFAULT_FLASHVSR_PRESET",
    "FIELD_CHUNK_SIZE",
    "FIELD_FPS",
    "FIELD_INPUT_HEIGHT",
    "FIELD_INPUT_WIDTH",
    "FIELD_TAIL_POLICY",
    "FIELD_TOTAL_FRAMES",
    "FIELD_VALID_FRAME_COUNT",
    "FIELD_VIDEO_CHUNK",
    "FLASHVSR_MODEL_ID",
    "FlashVSRInferenceRuntime",
    "FlashVSRInferenceSession",
    "FlashVSRModelAdapter",
    "FlashVSRRuntimeOptions",
    "FlashVSRSessionInputs",
    "PipelineFactory",
    "TailPolicy",
    "session_inputs_from_inference_input",
]
