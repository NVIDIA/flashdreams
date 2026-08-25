# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImGui HUD and presentation state for Crazy Robotaxi."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from crazy_robotaxi.high_scores import validate_player_name
from crazy_robotaxi.rules import TaxiGameSnapshot
from flashdreams.api_v2.thread import IThread, invoke_async
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_MAX_BUFFERED_HUD_FRAMES = 64
"""Maximum frame-aligned snapshots retained across pending model chunks."""


@dataclass(frozen=True, slots=True)
class TaxiHudFrame:
    """Immutable UI data aligned with one generated video frame."""

    frame_key: int
    """Live tensor data pointer identifying the corresponding video frame."""

    snapshot: TaxiGameSnapshot
    """Taxi rules snapshot for the corresponding simulation frame."""


@dataclass(slots=True)
class _HudWidgets:
    """Retained SlangPy widget handles owned by the UI thread."""

    status_window: Any
    score: Any
    time: Any
    objective: Any
    fare_time: Any
    event: Any
    navigation_window: Any
    navigation: Any
    terminal_window: Any
    terminal_title: Any
    terminal_score: Any
    terminal_rank: Any
    name_input: Any
    submit: Any
    validation: Any
    leaderboard: Any


@dataclass(slots=True)
class TaxiHudState:
    """Mutable ImGui state owned exclusively by the V2 UI thread."""

    width: int
    """Presentation width in pixels."""

    height: int
    """Presentation height in pixels."""

    model_thread: IThread[Any] | None = None
    """Model-thread endpoint used only through ``invoke_async``."""

    _frames: OrderedDict[int, TaxiHudFrame] = field(default_factory=OrderedDict)
    """Recent immutable snapshots keyed by presented tensor-frame identity."""

    _current: TaxiHudFrame | None = None
    """Snapshot aligned with the frame currently beneath ImGui."""

    _widgets: _HudWidgets | None = None
    """Lazily created retained widgets."""

    _validation_message: str = ""
    """Name-entry validation or submission status."""

    _submission_pending: bool = False
    """Whether a validated name is already queued for the model thread."""

    def publish(self, frames: Sequence[TaxiHudFrame]) -> None:
        """Publish immutable model-frame state to the UI-owned lookup."""
        for frame in frames:
            self._frames[frame.frame_key] = frame
            self._frames.move_to_end(frame.frame_key)
        while len(self._frames) > _MAX_BUFFERED_HUD_FRAMES:
            self._frames.popitem(last=False)

    def select_presented_frame(self, frame: Tensor) -> TaxiHudFrame | None:
        """Select the HUD snapshot aligned with ``frame`` when available."""
        selected = self._frames.get(int(frame.data_ptr()))
        if selected is not None:
            if (
                self._current is None
                or selected.snapshot.session_state
                != self._current.snapshot.session_state
            ):
                self._validation_message = ""
                self._submission_pending = False
            self._current = selected
        return self._current

    def draw(self, ui: Any) -> None:
        """Create or update the retained ImGui HUD widget tree."""
        widgets = self._ensure_widgets(ui)
        hud_frame = self._current
        if hud_frame is None:
            widgets.score.text = "LOADING WORLD MODEL"
            widgets.time.text = ""
            widgets.objective.text = ""
            widgets.fare_time.text = ""
            widgets.event.text = ""
            widgets.navigation_window.visible = False
            widgets.terminal_window.visible = False
            return

        snapshot = hud_frame.snapshot
        widgets.status_window.visible = snapshot.session_state == "playing"
        widgets.score.text = _score_label(snapshot)
        widgets.time.text = f"GAME TIME  {snapshot.global_remaining_time_s:05.1f}s"
        objective = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
        widgets.objective.text = f"{objective}  {snapshot.distance_m:04.0f} m"
        widgets.fare_time.text = (
            ""
            if snapshot.remaining_time_s is None
            else f"FARE TIME  {snapshot.remaining_time_s:04.1f}s"
        )
        widgets.event.text = _event_label(snapshot)

        widgets.navigation_window.visible = snapshot.session_state == "playing"
        widgets.navigation.text = _navigation_label(snapshot.relative_bearing_rad)
        self._update_terminal(widgets, snapshot)

    def reset(self) -> None:
        """Clear per-generation HUD snapshots and editable UI state."""
        self._frames.clear()
        self._current = None
        self._validation_message = ""
        self._submission_pending = False
        if self._widgets is not None:
            self._widgets.name_input.value = ""
            self._widgets.name_input.enabled = True
            self._widgets.submit.enabled = True

    def _ensure_widgets(self, ui: Any) -> _HudWidgets:
        if self._widgets is not None:
            return self._widgets
        status_window = ui.Window(
            ui.screen,
            "Crazy Robotaxi",
            position=(14.0, 14.0),
            size=(360.0, 190.0),
        )
        score = ui.Text(status_window, "LOADING WORLD MODEL")
        time = ui.Text(status_window, "")
        objective = ui.Text(status_window, "")
        fare_time = ui.Text(status_window, "")
        event = ui.Text(status_window, "")

        navigation_window = ui.Window(
            ui.screen,
            "Navigation",
            position=(float(self.width - 294), 14.0),
            size=(280.0, 92.0),
        )
        navigation = ui.Text(navigation_window, "")

        terminal_window = ui.Window(
            ui.screen,
            "Game Over",
            position=(
                float(max(20, self.width // 2 - 270)),
                float(max(20, self.height // 2 - 220)),
            ),
            size=(540.0, 440.0),
        )
        terminal_title = ui.Text(terminal_window, "")
        terminal_score = ui.Text(terminal_window, "")
        terminal_rank = ui.Text(terminal_window, "")
        name_input = ui.InputText(
            terminal_window,
            "Driver name",
            value="",
            callback=self._submit_name,
            flags=ui.InputTextFlags.enter_returns_true,
        )
        submit = ui.Button(
            terminal_window,
            "Submit score",
            callback=self._submit_name_from_button,
        )
        validation = ui.Text(terminal_window, "")
        leaderboard = ui.Text(terminal_window, "")
        self._widgets = _HudWidgets(
            status_window=status_window,
            score=score,
            time=time,
            objective=objective,
            fare_time=fare_time,
            event=event,
            navigation_window=navigation_window,
            navigation=navigation,
            terminal_window=terminal_window,
            terminal_title=terminal_title,
            terminal_score=terminal_score,
            terminal_rank=terminal_rank,
            name_input=name_input,
            submit=submit,
            validation=validation,
            leaderboard=leaderboard,
        )
        return self._widgets

    def _update_terminal(
        self, widgets: _HudWidgets, snapshot: TaxiGameSnapshot
    ) -> None:
        awaiting_name = snapshot.session_state == "awaiting_name"
        leaderboard = snapshot.session_state == "leaderboard"
        widgets.terminal_window.visible = awaiting_name or leaderboard
        if not (awaiting_name or leaderboard):
            return
        widgets.terminal_title.text = (
            "NEW HIGH SCORE" if awaiting_name else "LEADERBOARD"
        )
        widgets.terminal_score.text = f"FINAL SCORE  {snapshot.score:06d}"
        widgets.terminal_rank.text = (
            ""
            if snapshot.high_score_rank is None
            else f"RANK  #{snapshot.high_score_rank}"
        )
        widgets.name_input.visible = awaiting_name
        widgets.submit.visible = awaiting_name
        widgets.name_input.enabled = awaiting_name and not self._submission_pending
        widgets.submit.enabled = awaiting_name and not self._submission_pending
        widgets.validation.visible = awaiting_name
        widgets.validation.text = self._validation_message
        widgets.leaderboard.visible = leaderboard
        widgets.leaderboard.text = (
            "\n".join(
                f"{rank:>2}. {entry.name:<12} {entry.score:>7}"
                for rank, entry in enumerate(snapshot.leaderboard, start=1)
            )
            or "NO SCORES YET"
        )

    def _submit_name_from_button(self) -> None:
        assert self._widgets is not None
        self._submit_name(str(self._widgets.name_input.value))

    def _submit_name(self, value: str) -> None:
        if self._submission_pending:
            return
        try:
            normalized = validate_player_name(value)
        except ValueError as error:
            self._validation_message = str(error)
            return
        model_thread = self.model_thread
        if model_thread is None:
            self._validation_message = "Model thread is not ready."
            return
        self._submission_pending = True
        self._validation_message = "Submitting score..."
        if self._widgets is not None:
            self._widgets.name_input.enabled = False
            self._widgets.submit.enabled = False
        invoke_async(
            model_thread,
            lambda state, name=normalized: state.submit_player_name(name),
        )


class CrazyRobotaxiImGUIThread(ImGUIThread[TaxiHudState]):
    """Present generated frames beneath a responsive ImGui taxi HUD."""

    def draw_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw the HUD and return the current generated frame beneath it."""
        del step_index, events
        frames = self.presented_model_frames()
        video = frames[0] if frames else None
        waypoints = frames[1] if len(frames) > 1 else None
        bev = frames[2] if len(frames) > 2 else None
        if video is not None:
            self.state.select_presented_frame(video)
        self.state.draw(ui)
        if video is None:
            return None
        if waypoints is None or waypoints.shape[0] != 4:
            raise ValueError("Waypoint presentation frames must use [4,H,W]")
        world = self._presentation_manager.composite(
            video.to(torch.float32),
            waypoints.to(device=video.device, dtype=torch.float32),
        )
        return _compose_minimap(world, bev)

    def reset(self) -> None:
        """Reset UI-owned state and retained renderer resources."""
        self.state.reset()
        super().reset()


def build_hud_frames(
    video_tchw: Tensor,
    snapshots: Sequence[object],
) -> tuple[TaxiHudFrame, ...]:
    """Build immutable UI messages aligned with generated tensor frames."""
    frame_count = int(video_tchw.shape[0])
    if len(snapshots) != frame_count:
        raise ValueError("Video and game snapshots must align")
    frames = []
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, TaxiGameSnapshot):
            raise TypeError("Taxi HUD received a non-taxi game snapshot")
        frames.append(
            TaxiHudFrame(
                frame_key=int(video_tchw[index].data_ptr()),
                snapshot=snapshot,
            )
        )
    return tuple(frames)


def _compose_minimap(video: Tensor | None, bev: Tensor | None) -> Tensor | None:
    if video is None:
        return None
    output = video.to(torch.float32)
    if bev is None:
        return output
    if bev.ndim != 3 or bev.shape[0] != 3:
        raise ValueError("BEV presentation frames must use [3,H,W]")
    height, width = int(output.shape[-2]), int(output.shape[-1])
    size = min(width // 4, height // 3)
    if size <= 4:
        return output
    panel = functional.interpolate(
        bev.to(device=output.device, dtype=torch.float32).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0]
    output = output.clone()
    margin = max(8, min(width, height) // 80)
    top = height - size - margin
    left = width - size - margin
    target = output[:, top : top + size, left : left + size]
    target.mul_(0.18).add_(panel, alpha=0.82)
    target[:, :2, :] = 1.0
    target[:, -2:, :] = 1.0
    target[:, :, :2] = 1.0
    target[:, :, -2:] = 1.0
    return output


def _score_label(snapshot: TaxiGameSnapshot) -> str:
    label = f"SCORE  {snapshot.score:06d}"
    if snapshot.high_score is not None:
        label += f"    HIGH  {snapshot.high_score:06d}"
    return label


def _event_label(snapshot: TaxiGameSnapshot) -> str:
    if snapshot.event == "pickup_complete":
        return "PASSENGER PICKED UP"
    if snapshot.event == "fare_complete":
        return (
            f"FARE COMPLETE  +{snapshot.awarded_points}  "
            f"+{snapshot.awarded_global_time_s:g}s"
        )
    if snapshot.event == "time_expired":
        return "FARE TIME EXPIRED"
    return ""


def _navigation_label(bearing_rad: float) -> str:
    degrees = math.degrees(math.atan2(math.sin(bearing_rad), math.cos(bearing_rad)))
    if abs(degrees) <= 15.0:
        direction = "AHEAD"
    elif abs(degrees) >= 165.0:
        direction = "BEHIND"
    elif degrees > 0.0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    return f"TARGET {direction}  {abs(degrees):.0f} deg"


__all__ = [
    "CrazyRobotaxiImGUIThread",
    "TaxiHudFrame",
    "TaxiHudState",
    "build_hud_frames",
]
