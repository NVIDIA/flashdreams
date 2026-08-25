# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy status and camera-control overlay for Cam2V applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_CAMERA_KEYS = frozenset({"w", "s", "q", "e", "a", "d", "j", "l", "i", "k"})
"""Keyboard controls recognized by the shared camera pose integrator."""

_CAMERA_KEY_ORDER = ("w", "s", "q", "e", "a", "d", "j", "l", "i", "k")
"""Stable order used when active camera controls are displayed."""


@dataclass(frozen=True, slots=True)
class Cam2VUIStatus:
    """Latest model-generation status copied to the UI loop."""

    completed_blocks: int
    """Number of autoregressive blocks completed in this rollout."""

    frames_generated: int
    """Number of video frames generated in this rollout."""

    chunk_fps: float
    """Frame throughput measured across the latest model step."""

    steady_state_fps: float | None
    """Cumulative post-warmup throughput, or ``None`` during warmup."""

    model_step_wall_s: float
    """Wall time spent producing the latest model chunk."""


@dataclass(slots=True)
class Cam2VUIState:
    """Mutable Cam2V overlay state owned exclusively by the UI loop."""

    total_blocks: int
    """Number of autoregressive blocks requested for the rollout."""

    target_fps: int
    """Configured generated-video frame rate."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state throughput."""

    held_keys: set[str] = field(default_factory=set)
    """Camera-control keys currently held by the client."""

    status: Cam2VUIStatus | None = None
    """Latest model status received from the model-generation loop."""

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
        self.status = None


class Cam2VSlangPyUILoop(SlangPyUILoop[Cam2VUIState]):
    """Draw Cam2V controls and model throughput over generated video."""

    def step_ui(
        self,
        ui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Update retained widgets and return the current model frame."""
        del step_index
        _apply_ui_input(self.state, events)
        _ensure_widgets(ui, self.state)
        _refresh_widgets(self.state)

        frame = self.presented_model_frame()
        if frame is None:
            return None
        if frame.is_floating_point():
            return frame.to(torch.float32)
        return frame.to(torch.float32).mul_(2.0 / 255.0).sub_(1.0)

    def reset(self) -> None:
        """Clear UI-loop state for a new generation."""
        self.state.reset()
        super().reset()


def _ensure_widgets(ui: Any, state: Cam2VUIState) -> None:
    if state.window is not None:
        return
    state.window = ui.Window(
        ui.screen,
        "Camera controls",
        position=(16, 16),
        size=(360, 280),
    )
    state.status_widgets = [
        ui.Text(state.window, line) for line in _status_lines(state)
    ]
    ui.Text(state.window, "Move: W/S    Strafe: Q/E")
    ui.Text(state.window, "Yaw: A/D or J/L    Pitch: I/K")
    state.active_keys_widget = ui.Text(state.window, _active_keys_text(state))
    ui.Text(state.window, "Click the video before using keyboard controls.")


def _refresh_widgets(state: Cam2VUIState) -> None:
    for widget, line in zip(
        state.status_widgets,
        _status_lines(state),
        strict=True,
    ):
        widget.text = line
    if state.active_keys_widget is not None:
        state.active_keys_widget.text = _active_keys_text(state)


def _status_lines(state: Cam2VUIState) -> tuple[str, ...]:
    status = state.status
    if status is None:
        return (
            "Waiting for the first generated chunk...",
            "Generated: 0 frames",
            "Latest model rate: waiting",
            f"Steady state: warming up (0/{state.warmup_blocks})",
            f"Target video rate: {state.target_fps} FPS",
            "Latest model step: waiting",
        )

    if status.steady_state_fps is None:
        warmup_done = min(status.completed_blocks, state.warmup_blocks)
        steady_state = f"Steady state: warming up ({warmup_done}/{state.warmup_blocks})"
    else:
        steady_state = f"Steady-state model rate: {status.steady_state_fps:.2f} FPS"
    return (
        f"Rollout: {status.completed_blocks}/{state.total_blocks} blocks",
        f"Generated: {status.frames_generated} frames",
        f"Latest model rate: {status.chunk_fps:.2f} FPS",
        steady_state,
        f"Target video rate: {state.target_fps} FPS",
        f"Latest model step: {status.model_step_wall_s * 1_000.0:.0f} ms",
    )


def _active_keys_text(state: Cam2VUIState) -> str:
    active = [key.upper() for key in _CAMERA_KEY_ORDER if key in state.held_keys]
    return f"Active keys: {', '.join(active) if active else 'none'}"


def _apply_ui_input(state: Cam2VUIState, events: UserInputEvents) -> None:
    for event in events.get_events():
        data = event.get_event_data()
        if isinstance(data, FocusUserInputEventData) and not data.focused:
            state.held_keys.clear()
            continue
        if not isinstance(data, KeyboardUserInputEventData):
            continue
        key = data.key.lower()
        if key not in _CAMERA_KEYS:
            logger.info(
                "Cam2V SlangPy UI loop ignored keyboard event "
                "key={} state={} timestamp_us={} reason=unsupported",
                data.key,
                data.state.value,
                int(event.get_timestamp()),
            )
            continue
        if data.state is KeyboardInputState.PRESSED:
            state.held_keys.add(key)
        else:
            state.held_keys.discard(key)
        held_keys = ",".join(
            item for item in _CAMERA_KEY_ORDER if item in state.held_keys
        )
        logger.info(
            "Cam2V SlangPy UI loop processed keyboard event "
            "key={} state={} timestamp_us={} held_keys={}",
            key,
            data.state.value,
            int(event.get_timestamp()),
            held_keys or "none",
        )


__all__ = ["Cam2VSlangPyUILoop", "Cam2VUIState", "Cam2VUIStatus"]
