# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-generation loop and session shared by camera-to-video apps."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.runner_io import ResizeInterpolation, load_first_frame_tensor
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .controls import CameraPoseIntegrator, KeyboardResampler
from .defaults import Cam2VConditioning
from .ui import Cam2VSlangPyUILoop, Cam2VUIState, Cam2VUIStatus


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

    first_frame_dtype: torch.dtype
    """Tensor dtype required by the model's first-frame input."""

    first_frame_interpolation: ResizeInterpolation
    """Resize interpolation required by the model's image preprocessor."""

    log_every_blocks: int
    """Interval between live timing records after warmup."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state FPS."""

    install_hint: str = ""
    """Optional first-frame loader hint for missing integration dependencies."""

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("Cam2VSessionConfig.total_blocks must be > 0.")
        if self.log_every_blocks <= 0:
            raise ValueError("Cam2VSessionConfig.log_every_blocks must be > 0.")
        if self.warmup_blocks < 0:
            raise ValueError("Cam2VSessionConfig.warmup_blocks must be >= 0.")


@dataclass(slots=True)
class Cam2VModelState:
    """Mutable rollout state owned exclusively by the model-generation-thread."""

    pipeline: Any
    """Application-owned, loaded model pipeline."""

    session_desc: SessionDesc
    """Output shape, layout, and rates accepted for this session."""

    config: Cam2VSessionConfig
    """Resolved inputs and rollout controls."""

    keyboard_resampler: KeyboardResampler
    """Timestamped camera-control state sampled on the model frame clock."""

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

    ui_loop: Cam2VSlangPyUILoop | None = None
    """Registered UI-loop handle used only through ``invoke_async``."""


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
            first_frame = load_first_frame_tensor(
                conditioning.first_frame_path,
                pixel_height=state.session_desc.video_height,
                pixel_width=state.session_desc.video_width,
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
        event_times = _buffer_keyboard_events(state.keyboard_resampler, events)
        _catch_up_keyboard_timeline(
            state.keyboard_resampler,
            frame_count=frame_count,
            event_times=event_times,
        )
        segments, frame_times = state.keyboard_resampler.sample_chunk(frame_count)
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
        input_preparation_s = time.perf_counter() - step_started_at

        generate_started_at = time.perf_counter()
        frames = state.pipeline.generate(
            autoregressive_index=step_index,
            cache=state.cache,
            input=camera_input,
        )
        generate_call_s = time.perf_counter() - generate_started_at

        finalize_started_at = time.perf_counter()
        metrics = _numeric_metrics(
            state.pipeline.finalize(
                autoregressive_index=step_index,
                cache=state.cache,
            )
        )
        if state.config.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.current_stream(state.config.device).synchronize()
        finalize_and_sync_s = time.perf_counter() - finalize_started_at
        model_step_wall_s = time.perf_counter() - step_started_at

        state.blocks_generated += 1
        state.frames_generated += frame_count
        metrics.update(
            {
                "input_prepare_s": input_preparation_s,
                "generate_call_s": generate_call_s,
                "finalize_and_sync_s": finalize_and_sync_s,
                "model_step_wall_s": model_step_wall_s,
                "chunk_fps": frame_count / model_step_wall_s,
            }
        )
        if state.steady_started_at is not None:
            state.steady_frames_generated += frame_count
            steady_elapsed_s = time.perf_counter() - state.steady_started_at
            metrics["steady_state_fps"] = (
                state.steady_frames_generated / steady_elapsed_s
            )
        if (
            state.steady_started_at is not None
            and state.blocks_generated % state.config.log_every_blocks == 0
        ):
            _log_step_timing(
                step_index=step_index,
                frame_count=frame_count,
                metrics=metrics,
            )
        _publish_ui_status(state, metrics)
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=frame_count,
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
        state.keyboard_resampler.reset(start_v=0.0)
        state.pose_integrator.reset()
        state.steady_started_at = None
        state.steady_frames_generated = 0

    def close(self) -> None:
        """Release session-owned tensors while retaining the application model."""
        self.state.cache = None


class Cam2VSession(ISession):
    """One camera-controlled rollout sharing its application's loaded model."""

    def __init__(
        self,
        *,
        pipeline: Any,
        config: Cam2VSessionConfig,
        session_desc: SessionDesc,
        use_ui: bool = True,
    ) -> None:
        """Configure one rollout without initializing model or UI resources.

        Args:
            pipeline: Application-owned model pipeline.
            config: Resolved inputs and rollout controls.
            session_desc: Output dimensions, layout, and loop rates.
            use_ui: Whether to register the shared Cam2V overlay.
        """
        self._pipeline = pipeline
        self._config = config
        self._session_desc = session_desc
        self._use_ui = use_ui

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved output dimensions and loop rates."""
        return self._session_desc

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
                session_desc=self._session_desc,
                config=self._config,
                keyboard_resampler=KeyboardResampler(
                    fps=self._session_desc.frames_per_second_for_step,
                ),
                ui_loop=ui_loop,
            ),
        )


def _buffer_keyboard_events(
    keyboard_resampler: KeyboardResampler,
    events: UserInputEvents,
) -> list[float]:
    """Queue timestamped WebRTC keyboard and focus edges for resampling."""
    event_times: list[float] = []
    for event in events.get_events():
        event_t = float(event.get_timestamp()) / 1_000_000.0
        if isinstance(event, FocusUserInputEvent):
            if not event.focused:
                keyboard_resampler.release_all(arrival_t=event_t)
                event_times.append(event_t)
            continue
        if not isinstance(event, KeyboardUserInputEvent):
            continue
        keyboard_resampler.on_edge(
            arrival_t=event_t,
            event=("keydown" if event.state is KeyboardInputState.PRESSED else "keyup"),
            key=event.key,
        )
        event_times.append(event_t)
    return event_times


def _catch_up_keyboard_timeline(
    keyboard_resampler: KeyboardResampler,
    *,
    frame_count: int,
    event_times: list[float],
) -> None:
    """Keep stale wall-clock input from waiting behind the model clock.

    Model warm-up and slow generation can leave the virtual camera timeline
    behind WebRTC's session clock. If the newest unread edge lies beyond the
    next chunk, skip only stale virtual time and retain up to one chunk of the
    batch's original edge timing.
    """
    if not event_times:
        return
    chunk_duration = frame_count * keyboard_resampler.dt
    chunk_end = keyboard_resampler.next_chunk_start_v + chunk_duration
    latest_event_t = max(event_times)
    if latest_event_t <= chunk_end:
        return
    earliest_event_t = min(event_times)
    keyboard_resampler.next_chunk_start_v = max(
        keyboard_resampler.next_chunk_start_v,
        earliest_event_t,
        latest_event_t - chunk_duration,
    )


def _publish_ui_status(
    state: Cam2VModelState,
    metrics: Mapping[str, float | int],
) -> None:
    """Send immutable model status to the UI loop without sharing state."""
    ui_loop = state.ui_loop
    if ui_loop is None:
        return
    steady_state_fps = metrics.get("steady_state_fps")
    status = Cam2VUIStatus(
        completed_blocks=state.blocks_generated,
        frames_generated=state.frames_generated,
        chunk_fps=float(metrics["chunk_fps"]),
        steady_state_fps=(
            None if steady_state_fps is None else float(steady_state_fps)
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


def _log_step_timing(
    *,
    step_index: int,
    frame_count: int,
    metrics: Mapping[str, float | int],
) -> None:
    """Log one chunk's wall-time breakdown and steady-state throughput."""
    logger.info(
        "Cam2V block={} frames={} steady_state_fps={:.2f} chunk_fps={:.2f} "
        "wall={:.3f}s input={:.3f}s generate_call={:.3f}s "
        "finalize_and_sync={:.3f}s",
        step_index,
        frame_count,
        metrics["steady_state_fps"],
        metrics["chunk_fps"],
        metrics["model_step_wall_s"],
        metrics["input_prepare_s"],
        metrics["generate_call_s"],
        metrics["finalize_and_sync_s"],
    )


__all__ = [
    "Cam2VModelState",
    "Cam2VModelLoop",
    "Cam2VSession",
    "Cam2VSessionConfig",
    "CameraControlInput",
]
