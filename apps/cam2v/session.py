# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-generation loop and session shared by camera-to-video apps."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .controls import CameraPoseIntegrator
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

    cache: Any | None = None
    """Session-local autoregressive model cache."""

    first_frame: torch.Tensor | None = None
    """Session-local first-frame tensor retained by the cache."""

    blocks_generated: int = 0
    """Number of completed autoregressive model steps."""

    frames_generated: int = 0
    """Number of generated video frames on the virtual camera clock."""

    held_keys: set[str] = field(default_factory=set)
    """Camera-control keys currently held by the WebRTC client."""

    pose_integrator: CameraPoseIntegrator = field(default_factory=CameraPoseIntegrator)
    """Session-local continuous camera state."""

    steady_started_at: float | None = None
    """Wall-clock origin immediately after excluded warmup blocks."""

    steady_frames_generated: int = 0
    """Frames generated since :attr:`steady_started_at`."""

    ui_loop: Cam2VSlangPyUILoop | None = None
    """Registered UI-loop handle used only through ``invoke_async``."""


class _GPUStageTimer:
    """Measure GPU generation and finalization without intermediate syncs."""

    def __init__(self, device: torch.device) -> None:
        self._enabled = device.type == "cuda" and torch.cuda.is_available()
        self._generate_start: torch.cuda.Event | None = None
        self._generate_end: torch.cuda.Event | None = None
        self._finalize_start: torch.cuda.Event | None = None
        self._finalize_end: torch.cuda.Event | None = None
        self._stream = torch.cuda.current_stream(device) if self._enabled else None
        if self._enabled:
            self._generate_start = torch.cuda.Event(enable_timing=True)
            self._generate_end = torch.cuda.Event(enable_timing=True)
            self._finalize_start = torch.cuda.Event(enable_timing=True)
            self._finalize_end = torch.cuda.Event(enable_timing=True)

    def mark_generate_start(self) -> None:
        """Record the beginning of pipeline generation."""
        if self._generate_start is not None:
            self._generate_start.record(self._stream)

    def mark_generate_end(self) -> None:
        """Record the end of pipeline generation."""
        if self._generate_end is not None:
            self._generate_end.record(self._stream)

    def mark_finalize_start(self) -> None:
        """Record the beginning of pipeline finalization."""
        if self._finalize_start is not None:
            self._finalize_start.record(self._stream)

    def mark_finalize_end(self) -> None:
        """Record the end of pipeline finalization."""
        if self._finalize_end is not None:
            self._finalize_end.record(self._stream)

    def elapsed_seconds(self) -> tuple[float | None, float | None]:
        """Synchronize once and return generation and finalization durations."""
        if self._finalize_end is None:
            return None, None
        assert self._generate_start is not None
        assert self._generate_end is not None
        assert self._finalize_start is not None
        self._finalize_end.synchronize()
        return (
            self._generate_start.elapsed_time(self._generate_end) / 1_000.0,
            self._finalize_start.elapsed_time(self._finalize_end) / 1_000.0,
        )


class Cam2VModelLoop(IModelLoop[Cam2VModelState]):
    """Generate one camera-controlled video chunk per model-loop iteration."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Apply new keyboard edges and generate one autoregressive block."""
        state = self.state
        step_started_at = time.perf_counter()
        if state.blocks_generated == state.config.warmup_blocks:
            state.steady_started_at = step_started_at

        _apply_keyboard_events(state.held_keys, events)
        _ensure_rollout_initialized(state)
        assert state.cache is not None

        frame_count = int(state.pipeline.get_num_output_frames(step_index))
        if frame_count <= 0:
            raise ValueError(
                "Cam2V pipelines must generate at least one frame per step."
            )
        fps = state.session_desc.frames_per_second_for_step
        start_s = state.frames_generated / fps
        end_s = (state.frames_generated + frame_count) / fps
        poses = state.pose_integrator.integrate_chunk(
            segments=[(start_s, end_s, frozenset(state.held_keys))],
            frame_times=[
                start_s + (frame_index + 1) / fps for frame_index in range(frame_count)
            ],
        )
        conditioning = state.config.conditioning
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

        gpu_timer = _GPUStageTimer(state.config.device)
        generate_started_at = time.perf_counter()
        gpu_timer.mark_generate_start()
        frames = state.pipeline.generate(
            autoregressive_index=step_index,
            cache=state.cache,
            input=camera_input,
        )
        gpu_timer.mark_generate_end()
        generate_submit_s = time.perf_counter() - generate_started_at

        finalize_started_at = time.perf_counter()
        gpu_timer.mark_finalize_start()
        metrics = _numeric_metrics(
            state.pipeline.finalize(
                autoregressive_index=step_index,
                cache=state.cache,
            )
        )
        gpu_timer.mark_finalize_end()
        generate_gpu_s, finalize_gpu_s = gpu_timer.elapsed_seconds()
        finalize_submit_s = time.perf_counter() - finalize_started_at
        model_step_wall_s = time.perf_counter() - step_started_at

        state.blocks_generated += 1
        state.frames_generated += frame_count
        metrics.update(
            {
                "input_prepare_s": input_preparation_s,
                "generate_submit_s": generate_submit_s,
                "finalize_submit_s": finalize_submit_s,
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
        if generate_gpu_s is not None:
            metrics["generate_gpu_s"] = generate_gpu_s
        if finalize_gpu_s is not None:
            metrics["finalize_gpu_s"] = finalize_gpu_s
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
        state.first_frame = None
        state.blocks_generated = 0
        state.frames_generated = 0
        state.held_keys.clear()
        state.pose_integrator.reset()
        state.steady_started_at = None
        state.steady_frames_generated = 0

    def close(self) -> None:
        """Release session-owned tensors while retaining the application model."""
        self.state.cache = None
        self.state.first_frame = None


class Cam2VSession(ISession):
    """One camera-controlled rollout sharing its application's loaded model."""

    model_loop_type: type[Cam2VModelLoop] = Cam2VModelLoop
    """Model-generation-loop type registered by :meth:`init`."""

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
            self.model_loop_type,
            state=Cam2VModelState(
                pipeline=self._pipeline,
                session_desc=self._session_desc,
                config=self._config,
                ui_loop=ui_loop,
            ),
        )


def _apply_keyboard_events(held_keys: set[str], events: UserInputEvents) -> None:
    """Update held camera keys from new WebRTC keyboard edges."""
    for event in events.get_events():
        data = event.get_event_data()
        if not isinstance(data, KeyboardUserInputEventData):
            continue
        key = data.key.lower()
        if data.state is KeyboardInputState.PRESSED:
            held_keys.add(key)
        else:
            held_keys.discard(key)


def _ensure_rollout_initialized(state: Cam2VModelState) -> None:
    """Initialize first-frame and cache state on the model-generation-thread."""
    if state.cache is not None:
        return
    conditioning = state.config.conditioning
    state.first_frame = _load_first_frame(
        conditioning.first_frame_path,
        session_desc=state.session_desc,
        device=state.config.device,
        install_hint=state.config.install_hint,
    )
    state.cache = state.pipeline.initialize_cache(
        text=[conditioning.prompt],
        image=state.first_frame,
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


def _load_first_frame(
    path: Path,
    *,
    session_desc: SessionDesc,
    device: torch.device,
    install_hint: str,
) -> torch.Tensor:
    """Load a first frame using the framework's runner-compatible path."""
    return load_first_frame_tensor(
        path,
        pixel_height=session_desc.video_height,
        pixel_width=session_desc.video_width,
        device=device,
        dtype=torch.bfloat16,
        interpolation="cubic",
        install_hint=install_hint,
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
        "wall={:.3f}s input={:.3f}s generate_submit={:.3f}s "
        "finalize_submit={:.3f}s generate_gpu={} finalize_gpu={}",
        step_index,
        frame_count,
        metrics["steady_state_fps"],
        metrics["chunk_fps"],
        metrics["model_step_wall_s"],
        metrics["input_prepare_s"],
        metrics["generate_submit_s"],
        metrics["finalize_submit_s"],
        _format_optional_seconds(metrics.get("generate_gpu_s")),
        _format_optional_seconds(metrics.get("finalize_gpu_s")),
    )


def _format_optional_seconds(value: float | int | None) -> str:
    """Format an optional stage duration for live logging."""
    return "n/a" if value is None else f"{float(value):.3f}s"


__all__ = [
    "Cam2VModelState",
    "Cam2VModelLoop",
    "Cam2VSession",
    "Cam2VSessionConfig",
    "CameraControlInput",
]
