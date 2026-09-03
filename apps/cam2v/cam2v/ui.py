# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy status and camera-control overlay for Cam2V applications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.runtime.keyboard import KeyboardState
from flashdreams.runtime_v2.recent_frame_rate import RecentFrameRateSnapshot
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

RECENT_MODEL_FPS_WINDOW_SECONDS = 2.0
"""Trailing AR-step completion window displayed in the model status panel."""


@dataclass(frozen=True, slots=True)
class Cam2VControlKey:
    """One keyboard key displayed in a Cam2V control group."""

    key: str
    """Canonical runtime key used to track held state."""

    label: str
    """Short user-facing label rendered in the overlay."""


@dataclass(frozen=True, slots=True)
class Cam2VControlGroup:
    """One action and its associated keyboard controls."""

    action: str
    """User-facing action description."""

    keys: tuple[Cam2VControlKey, ...]
    """Keys that trigger the action."""


DEFAULT_CAM2V_CONTROL_GROUPS = (
    Cam2VControlGroup(
        action="Move forward / backward",
        keys=(Cam2VControlKey("w", "W"), Cam2VControlKey("s", "S")),
    ),
    Cam2VControlGroup(
        action="Strafe left / right",
        keys=(Cam2VControlKey("q", "Q"), Cam2VControlKey("e", "E")),
    ),
    Cam2VControlGroup(
        action="Yaw left / right",
        keys=(
            Cam2VControlKey("a", "A"),
            Cam2VControlKey("d", "D"),
            Cam2VControlKey("j", "J"),
            Cam2VControlKey("l", "L"),
        ),
    ),
    Cam2VControlGroup(
        action="Pitch up / down",
        keys=(Cam2VControlKey("i", "I"), Cam2VControlKey("k", "K")),
    ),
)
"""Default keyboard groups for the shared camera pose integrator."""

DEFAULT_CAM2V_UI_INSTRUCTIONS = (
    "Held controls are shown in brackets.",
    "Click the video before using keyboard controls.",
)
"""Default hints rendered below the Cam2V controls."""


@dataclass(frozen=True, slots=True)
class Cam2VUIStatus:
    """Latest model-generation status copied to the UI loop."""

    completed_blocks: int
    """Number of autoregressive blocks completed in this rollout."""

    frames_generated: int
    """Number of video frames generated in this rollout."""

    chunk_fps: float
    """Frame throughput measured across the latest model step."""

    recent_model_rate_snapshot: RecentFrameRateSnapshot | None
    """Post-warmup model throughput observations shared by the model thread."""

    model_step_wall_s: float
    """Wall time spent producing the latest model chunk."""

    def recent_model_fps(self, now: float | None = None) -> float | None:
        """Return recent post-warmup model-step throughput at ``now``."""
        snapshot = self.recent_model_rate_snapshot
        if snapshot is None:
            return None
        return snapshot.frames_per_second(time.perf_counter() if now is None else now)


@dataclass(slots=True)
class Cam2VUIState:
    """Mutable Cam2V overlay state owned exclusively by the UI loop."""

    total_blocks: int
    """Number of autoregressive blocks requested for the rollout."""

    target_fps: int
    """Configured generated-video frame rate."""

    warmup_blocks: int
    """Leading blocks excluded from recent model throughput."""

    control_groups: tuple[Cam2VControlGroup, ...] = DEFAULT_CAM2V_CONTROL_GROUPS
    """Action-oriented key groups displayed by the overlay."""

    instructions: tuple[str, ...] = DEFAULT_CAM2V_UI_INSTRUCTIONS
    """Short usage hints displayed below the controls."""

    show_status: bool = True
    """Whether model throughput lines are included above the controls."""

    held_keys: set[str] = field(default_factory=set)
    """Keyboard control keys currently held by the client."""

    _keyboard_state: KeyboardState = field(init=False)
    """UI-thread-owned source-aware keyboard state."""

    status: Cam2VUIStatus | None = None
    """Latest model status received from the model-generation loop."""

    frames_presented: int = 0
    """Number of model frames selected by the UI thread in this rollout."""

    window: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy controls window."""

    status_widgets: list[Any] = field(default_factory=list, init=False, repr=False)
    """Retained SlangPy text widgets for model status."""

    control_widgets: list[Any] = field(default_factory=list, init=False, repr=False)
    """Retained SlangPy text widgets for action-oriented key groups."""

    active_keys_widget: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy text widget for active keyboard controls."""

    def __post_init__(self) -> None:
        self._keyboard_state = self._new_keyboard_state()

    def update_status(self, status: Cam2VUIStatus) -> None:
        """Replace the displayed model-generation status."""
        self.status = status

    def reset(self) -> None:
        """Clear transient controls and model status for a new generation."""
        self.held_keys.clear()
        self._keyboard_state = self._new_keyboard_state()
        self.status = None
        self.frames_presented = 0

    def _new_keyboard_state(self) -> KeyboardState:
        keys = frozenset(
            control.key for group in self.control_groups for control in group.keys
        )
        return KeyboardState(supported_keys=keys)


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
        frame = self.presented_model_frame()
        self.state.frames_presented = self._presentation_manager.presented_frame_count
        sampled_at = time.perf_counter()
        _ensure_widgets(ui, self.state, sampled_at=sampled_at)
        _refresh_widgets(self.state, sampled_at=sampled_at)

        return frame

    def reset(self) -> None:
        """Clear UI-loop state for a new generation."""
        self.state.reset()
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
        "Controls",
        position=(16, 16),
        size=(400, 340 if state.show_status else 220),
    )
    if state.show_status:
        state.status_widgets = [
            ui.Text(state.window, line)
            for line in _status_lines(state, sampled_at=sampled_at)
        ]
    state.control_widgets = [
        ui.Text(state.window, _control_group_text(group, state.held_keys))
        for group in state.control_groups
    ]
    state.active_keys_widget = ui.Text(state.window, _active_keys_text(state))
    for instruction in state.instructions:
        ui.Text(state.window, instruction)


def _refresh_widgets(state: Cam2VUIState, *, sampled_at: float) -> None:
    if state.status_widgets:
        for widget, line in zip(
            state.status_widgets,
            _status_lines(state, sampled_at=sampled_at),
            strict=True,
        ):
            widget.text = line
    for widget, group in zip(
        state.control_widgets,
        state.control_groups,
        strict=True,
    ):
        widget.text = _control_group_text(group, state.held_keys)
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
        recent_model_rate_line = (
            f"Recent model rate: warming up ({warmup_done}/{state.warmup_blocks})"
        )
    else:
        assert status.recent_model_rate_snapshot is not None
        window_seconds = status.recent_model_rate_snapshot.window_seconds
        recent_model_rate_line = (
            f"Recent model rate ({window_seconds:g} s): {recent_model_fps:.2f} FPS"
        )
    return (
        f"Rollout: {status.completed_blocks}/{state.total_blocks} blocks",
        f"Presented: {state.frames_presented} frames "
        f"({status.frames_generated} generated)",
        f"Latest model rate: {status.chunk_fps:.2f} FPS",
        recent_model_rate_line,
        f"Target video rate: {state.target_fps} FPS",
        f"Latest model step: {status.model_step_wall_s * 1_000.0:.0f} ms",
    )


def _active_keys_text(state: Cam2VUIState) -> str:
    active = [
        control.label
        for group in state.control_groups
        for control in group.keys
        if control.key in state.held_keys
    ]
    return f"Active keys: {', '.join(active) if active else 'none'}"


def _control_group_text(
    group: Cam2VControlGroup,
    held_keys: set[str],
) -> str:
    labels = [
        f"[{control.label}]" if control.key in held_keys else control.label
        for control in group.keys
    ]
    return f"{group.action}: {' / '.join(labels)}"


def _apply_ui_input(state: Cam2VUIState, events: UserInputEvents) -> None:
    for event in events.get_events():
        if isinstance(event, FocusUserInputEvent) and not event.focused:
            state.held_keys.clear()
            state._keyboard_state = state._new_keyboard_state()
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


__all__ = [
    "Cam2VControlGroup",
    "Cam2VControlKey",
    "Cam2VSlangPyUILoop",
    "Cam2VUIState",
    "Cam2VUIStatus",
]
