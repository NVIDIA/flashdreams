# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-generation loop and session shared by camera-to-video apps."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from loguru import logger

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.postprocess import VideoPostprocessStream
from flashdreams.infra.runner_io import ResizeInterpolation, load_first_frame_tensor
from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
from flashdreams.runtime_v2.keyboard_input import KeyboardStateTrack
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.recent_frame_rate import RecentFrameRateTracker
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .controls import CameraPoseIntegrator, KeyboardResampler
from .defaults import Cam2VConditioning
from .ui import (
    RECENT_MODEL_FPS_WINDOW_SECONDS,
    Cam2VSlangPyUILoop,
    Cam2VUIState,
    Cam2VUIStatus,
)


@dataclass(kw_only=True, slots=True)
class CameraControlInput:
    """Model-neutral per-step camera payload."""

    intrinsics: torch.Tensor
    """Per-frame intrinsics shaped ``[T, 4]``."""

    poses: torch.Tensor
    """Per-frame camera-to-world matrices shaped ``[T, 4, 4]``."""

    world_scale: float
    """Scale applied to camera translations by the model camera encoder."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Cam2VSessionConfig:
    """Resolved immutable settings for one camera-to-video rollout."""

    conditioning: Cam2VConditioning
    """Prompt, first frame, and calibrated camera values."""

    total_blocks: int
    """Number of model steps generated before the rollout completes."""

    device: torch.device
    """Device holding model inputs and cache state."""

    model_video_width: int | None = None
    """Raw model width when presentation is postprocessed to another size."""

    model_video_height: int | None = None
    """Raw model height when presentation is postprocessed to another size."""

    first_frame_dtype: torch.dtype
    """Tensor dtype required by the model's first-frame input."""

    first_frame_interpolation: ResizeInterpolation
    """Resize interpolation required by the model's image preprocessor."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state FPS."""

    log_model_timing: bool = False
    """Write one synchronized wall-time record for each AR model step."""

    install_hint: str = ""
    """Optional first-frame loader hint for missing integration dependencies."""

    postprocess_enabled: bool = False
    """Whether the configured application-owned postprocessor starts enabled."""

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("Cam2VSessionConfig.total_blocks must be > 0.")
        if self.warmup_blocks < 0:
            raise ValueError("Cam2VSessionConfig.warmup_blocks must be >= 0.")
        if not isinstance(self.log_model_timing, bool):
            raise TypeError("Cam2VSessionConfig.log_model_timing must be bool.")
        if not isinstance(self.postprocess_enabled, bool):
            raise TypeError("Cam2VSessionConfig.postprocess_enabled must be bool.")
        if (self.model_video_width is None) != (self.model_video_height is None):
            raise ValueError("Cam2V model dimensions must be set together.")
        if self.model_video_width is not None:
            assert self.model_video_height is not None
            if self.model_video_width <= 0 or self.model_video_height <= 0:
                raise ValueError("Cam2V model dimensions must be positive.")


@dataclass(slots=True)
class Cam2VModelState:
    """Mutable rollout state owned exclusively by the model-generation-thread."""

    pipeline: Any
    """Application-owned, loaded model pipeline."""

    session_desc: SessionDesc
    """Output shape, layout, and rates accepted for this session."""

    config: Cam2VSessionConfig
    """Resolved inputs and rollout controls."""

    postprocess_stream: VideoPostprocessStream | None = None
    """Application-owned postprocess stream retained across rollout resets."""

    postprocess_enabled: bool = False
    """Whether generated chunks currently pass through post-processing."""

    keyboard_resampler: KeyboardResampler | None = None
    """Legacy combined input view retained for construction compatibility."""

    input_timeline: RealtimeInputTimeline = field(init=False)
    """Session-relative sampling windows owned by the model thread."""

    keyboard_track: KeyboardStateTrack = field(init=False)
    """Timestamped held-key state projected into camera-control segments."""

    cache: Any | None = None
    """Session-local autoregressive model cache."""

    blocks_generated: int = 0
    """Number of completed autoregressive model steps."""

    frames_generated: int = 0
    """Number of generated video frames on the virtual camera clock."""

    pose_integrator: CameraPoseIntegrator = field(default_factory=CameraPoseIntegrator)
    """Session-local continuous camera state."""

    steady_started_at: float | None = None
    """Wall-clock origin immediately after excluded warmup blocks."""

    steady_frames_generated: int = 0
    """Frames generated since :attr:`steady_started_at`."""

    _recent_model_frame_rate_tracker: RecentFrameRateTracker = field(
        default_factory=lambda: RecentFrameRateTracker(
            window_seconds=RECENT_MODEL_FPS_WINDOW_SECONDS
        )
    )
    """Trailing post-warmup AR-step throughput shown by the UI."""

    ui_loop: Cam2VSlangPyUILoop | None = None
    """Registered UI-loop handle used only through ``invoke_async``."""

    def __post_init__(self) -> None:
        """Expose decomposed input state behind the legacy constructor field."""
        keyboard_resampler = self.keyboard_resampler
        if keyboard_resampler is None:
            keyboard_resampler = KeyboardResampler(
                fps=self.session_desc.frames_per_second_for_step,
            )
            self.keyboard_resampler = keyboard_resampler
        self.input_timeline = keyboard_resampler.input_timeline
        self.keyboard_track = keyboard_resampler.keyboard_track

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Toggle the resident postprocessor and reset only its temporal state."""
        enabled = bool(enabled)
        if enabled and self.postprocess_stream is None:
            raise RuntimeError(
                "Cannot enable post-processing without --postprocess-preset."
            )
        if enabled == self.postprocess_enabled:
            return
        if self.postprocess_stream is not None:
            self.postprocess_stream.reset()
        self.postprocess_enabled = enabled


class Cam2VModelLoop(IModelLoop[Cam2VModelState]):
    """Generate one camera-controlled video chunk per model-loop iteration."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Apply new keyboard edges and generate one autoregressive block."""
        state = self.state
        step_started_at = time.perf_counter()
        if state.blocks_generated == state.config.warmup_blocks:
            state.steady_started_at = step_started_at

        conditioning = state.config.conditioning
        if state.cache is None:
            model_height = (
                state.config.model_video_height or state.session_desc.video_height
            )
            model_width = (
                state.config.model_video_width or state.session_desc.video_width
            )
            first_frame = load_first_frame_tensor(
                conditioning.first_frame_path,
                pixel_height=model_height,
                pixel_width=model_width,
                device=state.config.device,
                dtype=state.config.first_frame_dtype,
                interpolation=state.config.first_frame_interpolation,
                install_hint=state.config.install_hint,
            )
            state.cache = state.pipeline.initialize_cache(
                text=[conditioning.prompt],
                image=first_frame,
            )
        assert state.cache is not None

        frame_count = int(state.pipeline.get_num_output_frames(step_index))
        if frame_count <= 0:
            raise ValueError(
                "Cam2V pipelines must generate at least one frame per step."
            )
        keyboard_events = state.keyboard_track.ingest(events)
        input_window = state.input_timeline.next_window(
            frame_count,
            input_times_s=(
                result.timestamp_s for result in keyboard_events if result.tracked
            ),
        )
        segments = state.keyboard_track.segments(input_window)
        frame_times = list(input_window.sample_times_s)
        poses = state.pose_integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        camera_input = CameraControlInput(
            intrinsics=conditioning.base_intrinsics.repeat(frame_count, 1).to(
                device=state.config.device,
                dtype=torch.float32,
            ),
            poses=torch.from_numpy(poses).to(
                device=state.config.device,
                dtype=torch.float32,
            ),
            world_scale=conditioning.world_scale,
        )
        frames = state.pipeline.generate(
            autoregressive_index=step_index,
            cache=state.cache,
            input=camera_input,
        )
        metrics = _numeric_metrics(
            state.pipeline.finalize(
                autoregressive_index=step_index,
                cache=state.cache,
            )
        )
        _synchronize_output(frames)
        model_completed_at = time.perf_counter()
        model_step_wall_s = model_completed_at - step_started_at

        output_frames = frames.detach()
        postprocess_stats = None
        postprocess_stream = state.postprocess_stream
        if state.postprocess_enabled:
            assert postprocess_stream is not None
            output_frames = postprocess_stream.process(
                output_frames,
                autoregressive_index=step_index,
            )
            postprocess_stats = postprocess_stream.last_process_stats
            if state.blocks_generated + 1 == state.config.total_blocks:
                tail = postprocess_stream.finish()
                if tail is not None:
                    output_frames = _concatenate_video(
                        output_frames,
                        tail,
                        layout=state.session_desc.output_layout,
                    )
            # Presentation pacing observes the complete model-loop wall time.
            # Do not let asynchronous post-processing spill into the next step.
            _synchronize_output(output_frames)
        step_completed_at = time.perf_counter()
        postprocess_step_wall_s = step_completed_at - model_completed_at
        output_frame_count = _video_frame_count(
            output_frames,
            layout=state.session_desc.output_layout,
        )
        if output_frame_count <= 0:
            raise RuntimeError(
                "Post-processing buffered the entire Cam2V chunk. Use a "
                "postprocess chunk size no larger than the model cadence."
            )

        state.blocks_generated += 1
        state.frames_generated += frame_count
        metrics.update(
            {
                "model_step_wall_s": model_step_wall_s,
                "chunk_fps": frame_count / model_step_wall_s,
                "model_loop_wall_s": step_completed_at - step_started_at,
                "postprocess_step_wall_s": postprocess_step_wall_s,
                "postprocess_enabled": int(state.postprocess_enabled),
                "postprocess_output_frames": output_frame_count,
            }
        )
        if postprocess_stats is not None:
            metrics["postprocess_ms"] = postprocess_stats.elapsed_ms
        if state.steady_started_at is not None:
            state.steady_frames_generated += frame_count
            steady_elapsed_s = step_completed_at - state.steady_started_at
            metrics["steady_state_fps"] = (
                state.steady_frames_generated / steady_elapsed_s
            )
            metrics["recent_model_fps"] = (
                state._recent_model_frame_rate_tracker.observe(
                    completed_at=step_completed_at,
                    frame_count=frame_count,
                    elapsed_s=model_step_wall_s,
                )
            )
        if state.config.log_model_timing:
            phase = "warmup" if step_index < state.config.warmup_blocks else "steady"
            logger.info(
                "Cam2V AR {} [{}] | {} frames | step wall {:.1f} ms | {:.2f} fps",
                step_index,
                phase,
                frame_count,
                model_step_wall_s * 1_000.0,
                metrics["chunk_fps"],
            )
        _publish_ui_status(state, metrics)
        return [
            StepResult(
                step_index=step_index,
                output=output_frames,
                frame_count=output_frame_count,
                output_layout=state.session_desc.output_layout,
                metrics=metrics,
            )
        ]

    def is_finished(self) -> bool:
        """Return whether this rollout generated its requested blocks."""
        return self.state.blocks_generated >= self.state.config.total_blocks

    def reset(self) -> None:
        """Discard model and camera state for a new session generation."""
        state = self.state
        state.cache = None
        state.blocks_generated = 0
        state.frames_generated = 0
        state.input_timeline.reset(start_s=0.0)
        state.keyboard_track.reset()
        state.pose_integrator.reset()
        state.steady_started_at = None
        state.steady_frames_generated = 0
        state._recent_model_frame_rate_tracker.reset()
        if state.postprocess_stream is not None:
            state.postprocess_stream.reset()

    def close(self) -> None:
        """Release session-owned tensors while retaining the application model."""
        self.state.cache = None
        if self.state.postprocess_stream is not None:
            self.state.postprocess_stream.reset()


class Cam2VSession(ISession):
    """One camera-controlled rollout sharing its application's loaded model."""

    def __init__(
        self,
        *,
        pipeline: Any,
        postprocess_stream: VideoPostprocessStream | None = None,
        config: Cam2VSessionConfig,
        session_desc: SessionDesc,
        use_ui: bool = True,
    ) -> None:
        """Configure one rollout without initializing model or UI resources.

        Args:
            pipeline: Application-owned model pipeline.
            postprocess_stream: Application-owned generated-video processor.
            config: Resolved inputs and rollout controls.
            session_desc: Output dimensions, layout, and loop rates.
            use_ui: Whether to register the shared Cam2V overlay.
        """
        self._pipeline = pipeline
        self._postprocess_stream = postprocess_stream
        self._config = config
        self._session_desc = session_desc
        self._use_ui = use_ui

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved output dimensions and loop rates."""
        return self._session_desc

    @cached_property
    def _presentation_manager(self) -> PresentationManager:
        """Return a frame manager initialized on the model device."""
        return PresentationManager(device=self._config.device)

    def init(self) -> None:
        """Register the UI and model-generation loops with isolated state."""
        ui_loop = None
        if self._use_ui:
            registered_ui = self.register_ui_loop(
                Cam2VSlangPyUILoop,
                state=Cam2VUIState(
                    total_blocks=self._config.total_blocks,
                    target_fps=self._session_desc.frames_per_second_for_step,
                    warmup_blocks=self._config.warmup_blocks,
                    show_postprocess_toggle=self._postprocess_stream is not None,
                    postprocess_enabled=self._config.postprocess_enabled,
                ),
                width=self._session_desc.video_width,
                height=self._session_desc.video_height,
            )
            assert isinstance(registered_ui, Cam2VSlangPyUILoop)
            ui_loop = registered_ui
        self.register_model_loop(
            Cam2VModelLoop,
            state=Cam2VModelState(
                pipeline=self._pipeline,
                postprocess_stream=self._postprocess_stream,
                postprocess_enabled=self._config.postprocess_enabled,
                session_desc=self._session_desc,
                config=self._config,
                keyboard_resampler=KeyboardResampler(
                    fps=self._session_desc.frames_per_second_for_step,
                ),
                ui_loop=ui_loop,
            ),
        )


def _publish_ui_status(
    state: Cam2VModelState,
    metrics: Mapping[str, float | int],
) -> None:
    """Send immutable model status to the UI loop without sharing state."""
    ui_loop = state.ui_loop
    if ui_loop is None:
        return
    recent_model_fps = metrics.get("recent_model_fps")
    status = Cam2VUIStatus(
        completed_blocks=state.blocks_generated,
        frames_generated=state.frames_generated,
        chunk_fps=float(metrics["chunk_fps"]),
        recent_model_rate_snapshot=(
            None
            if recent_model_fps is None
            else state._recent_model_frame_rate_tracker.snapshot()
        ),
        model_step_wall_s=float(metrics["model_step_wall_s"]),
    )
    invoke_async(
        ui_loop,
        lambda ui_state, status=status: ui_state.update_status(status),
    )


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    """Keep numeric pipeline metrics accepted by the v2 result contract."""
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _video_frame_count(
    video: torch.Tensor,
    *,
    layout: VideoTensorLayout,
) -> int:
    """Return a v2 video tensor's temporal extent."""
    time_dim = {
        VideoTensorLayout.tchw: 0,
        VideoTensorLayout.btchw: 1,
        VideoTensorLayout.bcthw: 2,
        VideoTensorLayout.bvtchw: 2,
    }[layout]
    return int(video.shape[time_dim])


def _synchronize_output(output: torch.Tensor) -> None:
    """Wait for CUDA output so step wall time covers completed production."""
    if output.is_cuda:
        torch.cuda.current_stream(output.device).synchronize()


def _concatenate_video(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    layout: VideoTensorLayout,
) -> torch.Tensor:
    """Join postprocess output and its final tail in presentation order."""
    time_dim = {
        VideoTensorLayout.tchw: 0,
        VideoTensorLayout.btchw: 1,
        VideoTensorLayout.bcthw: 2,
        VideoTensorLayout.bvtchw: 2,
    }[layout]
    if first.shape[time_dim] == 0:
        return second
    return torch.cat((first, second), dim=time_dim)


__all__ = [
    "Cam2VModelState",
    "Cam2VModelLoop",
    "Cam2VSession",
    "Cam2VSessionConfig",
    "CameraControlInput",
]
