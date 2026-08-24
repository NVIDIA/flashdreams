# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui status and camera-control overlay for Cam2V applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.runtime_v2.imgui_thread import ImGUIThread
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
    """Latest model-generation status copied to the UI thread."""

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
    """Mutable Cam2V overlay state owned exclusively by the UI thread."""

    total_blocks: int
    """Number of autoregressive blocks requested for the rollout."""

    target_fps: int
    """Configured generated-video frame rate."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state throughput."""

    held_keys: set[str] = field(default_factory=set)
    """Camera-control keys currently held by the client."""

    status: Cam2VUIStatus | None = None
    """Latest model status received from the model-generation-thread."""

    def update_status(self, status: Cam2VUIStatus) -> None:
        """Replace the displayed model-generation status."""
        self.status = status

    def reset(self) -> None:
        """Clear transient controls and model status for a new generation."""
        self.held_keys.clear()
        self.status = None


class Cam2VImGUIThread(ImGUIThread[Cam2VUIState]):
    """Draw Cam2V controls and model throughput over the generated video."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw one status overlay and return the current model frame beneath it."""
        del step_index
        _apply_ui_input(self.state, events)

        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size((320, 250), imgui.Cond_.once)
        imgui.begin("Camera controls")
        _draw_model_status(imgui, self.state)
        imgui.separator()
        imgui.text("Move: W/S    Strafe: Q/E")
        imgui.text("Yaw: A/D or J/L    Pitch: I/K")
        active = [
            key.upper() for key in _CAMERA_KEY_ORDER if key in self.state.held_keys
        ]
        imgui.text(f"Active keys: {', '.join(active) if active else 'none'}")
        imgui.separator()
        imgui.text("Click the video before using keyboard controls.")
        imgui.end()

        frame = self.presented_model_frame()
        if frame is None:
            return None
        if frame.is_floating_point():
            return frame.to(torch.float32)
        return frame.to(torch.float32).mul_(2.0 / 255.0).sub_(1.0)

    def reset(self) -> None:
        """Clear UI-owned state and renderer input for a new generation."""
        self.state.reset()
        super().reset()


def _draw_model_status(imgui: Any, state: Cam2VUIState) -> None:
    status = state.status
    if status is None:
        imgui.text("Waiting for the first generated chunk...")
        imgui.text(f"Target video rate: {state.target_fps} FPS")
        return

    imgui.text(f"Rollout: {status.completed_blocks}/{state.total_blocks} blocks")
    imgui.text(f"Generated: {status.frames_generated} frames")
    imgui.text(f"Latest model rate: {status.chunk_fps:.2f} FPS")
    if status.steady_state_fps is None:
        warmup_done = min(status.completed_blocks, state.warmup_blocks)
        imgui.text(f"Steady state: warming up ({warmup_done}/{state.warmup_blocks})")
    else:
        imgui.text(f"Steady-state model rate: {status.steady_state_fps:.2f} FPS")
    imgui.text(f"Target video rate: {state.target_fps} FPS")
    imgui.text(f"Latest model step: {status.model_step_wall_s * 1_000.0:.0f} ms")


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
            continue
        if data.state is KeyboardInputState.PRESSED:
            state.held_keys.add(key)
        else:
            state.held_keys.discard(key)


__all__ = ["Cam2VImGUIThread", "Cam2VUIState", "Cam2VUIStatus"]
