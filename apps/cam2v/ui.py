# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy status and camera-control overlay for Cam2V applications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.runtime.keyboard import KeyboardState, normalize_key
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import InputEventTrace
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_CAMERA_KEY_ORDER = ("w", "s", "q", "e", "a", "d", "j", "l", "i", "k")
"""Stable order used when active camera controls are displayed."""

_CAMERA_KEYS = frozenset(_CAMERA_KEY_ORDER)
"""Keyboard controls recognized by the shared camera pose integrator."""

RECENT_MODEL_FPS_WINDOW_SECONDS = 2.0
"""Trailing AR-step completion window displayed in the model status panel."""

_IDLE_STATUS_REFRESH_SECONDS = 0.25
"""Maximum four-Hz UI refresh while a recent model rate can expire."""


@dataclass(frozen=True, slots=True)
class Cam2VModelStepTiming:
    """One completed autoregressive model step used for recent throughput."""

    completed_at: float
    """Timestamp recorded after the step and required CUDA synchronization."""

    frame_count: int
    """Generated frames returned by this autoregressive step."""

    wall_s: float
    """Wall time for input preparation, generation, finalization, and CUDA sync."""


@dataclass(frozen=True, slots=True)
class Cam2VUIStatus:
    """Latest model-generation status copied to the UI loop."""

    completed_blocks: int
    """Number of autoregressive blocks completed in this rollout."""

    frames_generated: int
    """Number of video frames generated in this rollout."""

    chunk_fps: float
    """Frame throughput measured across the latest model step."""

    recent_model_steps: tuple[Cam2VModelStepTiming, ...] | None
    """Post-warmup model steps completed in the recent sampling window."""

    model_step_wall_s: float
    """Wall time spent producing the latest model chunk."""

    def recent_model_fps(self, now: float | None = None) -> float | None:
        """Return recent post-warmup model-step throughput at ``now``."""
        steps = self.recent_model_steps
        if steps is None:
            return None
        return _recent_model_fps(
            steps,
            now=time.perf_counter() if now is None else now,
            window_seconds=RECENT_MODEL_FPS_WINDOW_SECONDS,
        )


@dataclass(slots=True)
class Cam2VUIState:
    """Mutable Cam2V overlay state owned exclusively by the UI loop."""

    total_blocks: int
    """Number of autoregressive blocks requested for the rollout."""

    target_fps: int
    """Configured generated-video frame rate."""

    warmup_blocks: int
    """Leading blocks excluded from recent model throughput."""

    held_keys: set[str] = field(default_factory=set)
    """Camera-control keys currently held by the client."""

    _keyboard_state: KeyboardState = field(
        default_factory=lambda: KeyboardState(supported_keys=_CAMERA_KEYS),
    )
    """UI-thread-owned source-aware keyboard state."""

    status: Cam2VUIStatus | None = None
    """Latest model status received from the model-generation loop."""

    frames_presented: int = 0
    """Number of model frames selected by the UI thread in this rollout."""

    window: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy controls window."""

    status_widgets: list[Any] = field(default_factory=list, init=False, repr=False)
    """Retained SlangPy text widgets for model status."""

    active_keys_widget: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy text widget for active camera controls."""

    def update_status(self, status: Cam2VUIStatus) -> None:
        """Replace the displayed model-generation status."""
        self.status = status

    def reset(self) -> None:
        """Clear transient controls and model status for a new generation."""
        self.held_keys.clear()
        self._keyboard_state = KeyboardState(supported_keys=_CAMERA_KEYS)
        self.status = None
        self.frames_presented = 0


class Cam2VSlangPyUILoop(SlangPyUILoop[Cam2VUIState]):
    """Draw Cam2V controls and model throughput over generated video."""

    _last_redraw_at: float | None = None
    _last_rendered_recent_fps: float | None = None

    def should_redraw_on_idle(self) -> bool:
        """Refresh an expired recent-model rate at most four times per second."""
        status = self.state.status
        last_redraw_at = self._last_redraw_at
        if status is None or last_redraw_at is None:
            return False
        now = time.perf_counter()
        if now - last_redraw_at < _IDLE_STATUS_REFRESH_SECONDS:
            return False
        recent_model_fps = status.recent_model_fps(now)
        rounded_fps = None if recent_model_fps is None else round(recent_model_fps, 1)
        return rounded_fps != self._last_rendered_recent_fps

    def step_ui(
        self,
        ui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Update retained widgets and return the current model frame."""
        del step_index
        _apply_ui_input(self.state, events)
        frame = self.presented_model_frame()
        self.state.frames_presented = self._presentation_manager.presented_frame_count
        sampled_at = time.perf_counter()
        _ensure_widgets(ui, self.state, sampled_at=sampled_at)
        recent_model_fps = (
            None
            if self.state.status is None
            else self.state.status.recent_model_fps(sampled_at)
        )
        _refresh_widgets(self.state, sampled_at=sampled_at)
        self._last_redraw_at = sampled_at
        self._last_rendered_recent_fps = (
            None if recent_model_fps is None else round(recent_model_fps, 1)
        )

        if frame is None:
            return None
        if frame.is_floating_point():
            return frame
        return frame.to(torch.float32).mul_(2.0 / 255.0).sub_(1.0)

    def input_event_traces(
        self,
        events: UserInputEvents,
    ) -> tuple[InputEventTrace, ...]:
        """Acknowledge correlated camera controls drawn by this UI step."""
        return _ui_input_event_traces(events)

    def reset(self) -> None:
        """Clear UI-loop state for a new generation."""
        self.state.reset()
        self._last_redraw_at = None
        self._last_rendered_recent_fps = None
        super().reset()


def _ensure_widgets(
    ui: Any,
    state: Cam2VUIState,
    *,
    sampled_at: float,
) -> None:
    if state.window is not None:
        return
    state.window = ui.Window(
        ui.screen,
        "Camera controls",
        position=(16, 16),
        size=(360, 280),
    )
    state.status_widgets = [
        ui.Text(state.window, line)
        for line in _status_lines(state, sampled_at=sampled_at)
    ]
    ui.Text(state.window, "Move: W/S    Strafe: Q/E")
    ui.Text(state.window, "Yaw: A/D or J/L    Pitch: I/K")
    state.active_keys_widget = ui.Text(state.window, _active_keys_text(state))
    ui.Text(state.window, "Click the video before using keyboard controls.")


def _refresh_widgets(state: Cam2VUIState, *, sampled_at: float) -> None:
    for widget, line in zip(
        state.status_widgets,
        _status_lines(state, sampled_at=sampled_at),
        strict=True,
    ):
        widget.text = line
    if state.active_keys_widget is not None:
        state.active_keys_widget.text = _active_keys_text(state)


def _status_lines(
    state: Cam2VUIState,
    *,
    sampled_at: float | None = None,
) -> tuple[str, ...]:
    status = state.status
    if status is None:
        return (
            "Waiting for the first generated chunk...",
            f"Presented: {state.frames_presented} frames",
            "Latest model rate: waiting",
            f"Recent model rate: warming up (0/{state.warmup_blocks})",
            f"Target video rate: {state.target_fps} FPS",
            "Latest model step: waiting",
        )

    recent_model_fps = status.recent_model_fps(sampled_at)
    if recent_model_fps is None:
        warmup_done = min(status.completed_blocks, state.warmup_blocks)
        recent_model_rate = (
            f"Recent model rate: warming up ({warmup_done}/{state.warmup_blocks})"
        )
    else:
        recent_model_rate = f"Recent model rate (2 s): {recent_model_fps:.2f} FPS"
    return (
        f"Rollout: {status.completed_blocks}/{state.total_blocks} blocks",
        f"Presented: {state.frames_presented} frames "
        f"({status.frames_generated} generated)",
        f"Latest model rate: {status.chunk_fps:.2f} FPS",
        recent_model_rate,
        f"Target video rate: {state.target_fps} FPS",
        f"Latest model step: {status.model_step_wall_s * 1_000.0:.0f} ms",
    )


def _active_keys_text(state: Cam2VUIState) -> str:
    active = [key.upper() for key in _CAMERA_KEY_ORDER if key in state.held_keys]
    return f"Active keys: {', '.join(active) if active else 'none'}"


def _recent_model_fps(
    steps: tuple[Cam2VModelStepTiming, ...],
    *,
    now: float,
    window_seconds: float,
) -> float:
    """Aggregate model-step throughput completed in a trailing time window."""
    if not steps:
        return 0.0
    now = max(float(now), steps[-1].completed_at)
    cutoff = now - window_seconds
    recent_steps = tuple(step for step in steps if step.completed_at > cutoff)
    elapsed_s = sum(step.wall_s for step in recent_steps)
    if elapsed_s <= 0.0:
        return 0.0
    return sum(step.frame_count for step in recent_steps) / elapsed_s


def _apply_ui_input(state: Cam2VUIState, events: UserInputEvents) -> None:
    for event in events.get_events():
        if isinstance(event, FocusUserInputEvent) and not event.focused:
            state.held_keys.clear()
            state._keyboard_state = KeyboardState(supported_keys=_CAMERA_KEYS)
            continue
        if not isinstance(event, KeyboardUserInputEvent):
            continue
        if not state._keyboard_state.apply_event(
            event=("keydown" if event.state is KeyboardInputState.PRESSED else "keyup"),
            key=event.key,
        ):
            continue
        state.held_keys.clear()
        state.held_keys.update(state._keyboard_state.snapshot())


def _ui_input_event_traces(
    events: UserInputEvents,
) -> tuple[InputEventTrace, ...]:
    """Acknowledge every correlated camera control processed by this UI step."""
    return tuple(
        InputEventTrace(event_id=event.event_id, frame_index=0)
        for event in events.get_events()
        if isinstance(event, KeyboardUserInputEvent)
        and event.event_id is not None
        and normalize_key(event.key) in _CAMERA_KEYS
    )


__all__ = [
    "Cam2VModelStepTiming",
    "Cam2VSlangPyUILoop",
    "Cam2VUIState",
    "Cam2VUIStatus",
]
