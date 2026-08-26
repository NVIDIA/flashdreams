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

import logging
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omnidreams_game_engine.types import CameraCalibration
from torch import Tensor

from crazy_robotaxi.high_scores import validate_player_name
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.world_overlay import render_waypoint_layers
from flashdreams.api_v2.loop import ILoop, invoke_async
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_MAX_BUFFERED_HUD_FRAMES = 64
"""Maximum frame-aligned snapshots retained across pending model chunks."""

_MAX_BUFFERED_INPUT_EVENTS = 64
"""Maximum diagnostic event receipts retained before model-frame correlation."""

_PROFILE_DRIVE_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaxiHudFrame:
    """Immutable UI data aligned with one generated video frame."""

    frame_key: int
    """Live tensor data pointer identifying the corresponding video frame."""

    snapshot: TaxiGameSnapshot
    """Taxi rules snapshot for the corresponding simulation frame."""

    rig_pose_world: npt.NDArray[np.float32]
    """Read-only rig pose that generated the corresponding video frame."""

    transition_timestamp_us: int | None = None
    """V2 input transition represented by this frame, when one was received."""

    input_transition_count: int = 0
    """Cumulative resolved drive transitions consumed by the model loop."""

    input_ignored_event_count: int = 0
    """Cumulative redundant drive events ignored by the model loop."""

    input_dropped_transition_count: int = 0
    """Cumulative transitions displaced by fixed-size model chunks."""


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
    input_window: Any | None
    input_state: Any | None
    input_latency: Any | None


@dataclass(slots=True)
class TaxiHudState:
    """Mutable ImGui state owned exclusively by the V2 UI thread."""

    width: int
    """Presentation width in pixels."""

    height: int
    """Presentation height in pixels."""

    calibration: CameraCalibration
    """Camera calibration used to project world markers on the UI thread."""

    profile_input_latency: bool = False
    """Whether input arrival and model-frame latency diagnostics are visible."""

    model_loop: ILoop[Any] | None = None
    """Model-loop endpoint used only through ``invoke_async``."""

    _frames: OrderedDict[int, TaxiHudFrame] = field(default_factory=OrderedDict)
    """Recent immutable snapshots keyed by presented tensor-frame identity."""

    _current: TaxiHudFrame | None = None
    """Snapshot aligned with the frame currently beneath ImGui."""

    _waypoint_source: TaxiHudFrame | None = None
    """Frame metadata used by the cached waypoint layer."""

    _waypoint_layer: Tensor | None = None
    """Cached world overlay for the currently presented generated frame."""

    _widgets: _HudWidgets | None = None
    """Lazily created retained widgets."""

    _validation_message: str = ""
    """Name-entry validation or submission status."""

    _submission_pending: bool = False
    """Whether a validated name is already queued for the model thread."""

    _loading_status: str = "LOADING WORLD MODEL"
    """Current startup phase shown until the first model frame is presented."""

    _loading_started_at_s: float = field(default_factory=time.monotonic)
    """Monotonic timestamp used to make startup progress visibly live."""

    _profile_pressed: set[str] = field(default_factory=set)
    """Normalized drive keys currently held according to UI-thread events."""

    _input_received_at_s: OrderedDict[int, float] = field(default_factory=OrderedDict)
    """UI receipt times keyed by V2 session-relative event timestamp."""

    _reported_input_timestamps_us: set[int] = field(default_factory=set)
    """Input transitions already correlated with a presented model frame."""

    _latest_input_latency_ms: float | None = None
    """Latest UI-ingress-to-model-frame-selection latency measurement."""

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
            self._record_presented_input(selected)
        return self._current

    def consume_input_events(self, events: UserInputEvents) -> None:
        """Track responsive drive state and receipt times on the UI thread."""
        if not self.profile_input_latency:
            return
        for event in events.get_events():
            data = event.get_event_data()
            recognized = False
            if isinstance(data, FocusUserInputEventData) and not data.focused:
                self._profile_pressed.clear()
                recognized = True
            elif isinstance(data, KeyboardUserInputEventData):
                key = _normalize_profile_key(str(data.key))
                if key not in _PROFILE_DRIVE_KEYS:
                    continue
                recognized = True
                if data.state is KeyboardInputState.PRESSED:
                    self._profile_pressed.add(key)
                else:
                    self._profile_pressed.discard(key)
            if not recognized:
                continue
            timestamp_us = int(event.get_timestamp())
            self._input_received_at_s.setdefault(timestamp_us, time.perf_counter())
            self._input_received_at_s.move_to_end(timestamp_us)
        while len(self._input_received_at_s) > _MAX_BUFFERED_INPUT_EVENTS:
            self._input_received_at_s.popitem(last=False)

    def _record_presented_input(self, selected: TaxiHudFrame) -> None:
        if not self.profile_input_latency:
            return
        timestamp_us = selected.transition_timestamp_us
        if timestamp_us is None or timestamp_us in self._reported_input_timestamps_us:
            return
        received_at_s = self._input_received_at_s.pop(timestamp_us, None)
        if received_at_s is None:
            return
        self._reported_input_timestamps_us.add(timestamp_us)
        self._latest_input_latency_ms = (time.perf_counter() - received_at_s) * 1000.0
        _LOGGER.info(
            "[crazy-robotaxi] input-to-model-frame latency: event_us=%d "
            "ui_to_frame_ms=%.1f transitions=%d ignored=%d dropped=%d",
            timestamp_us,
            self._latest_input_latency_ms,
            selected.input_transition_count,
            selected.input_ignored_event_count,
            selected.input_dropped_transition_count,
        )

    def set_loading_status(self, status: str) -> None:
        """Update the startup phase from a model-loop message."""
        self._loading_status = status

    def waypoint_layer(self, frame: Tensor) -> Tensor | None:
        """Return the cached marker layer aligned with ``frame``.

        Rasterizing one presented frame on the UI thread avoids constructing a
        full chunk of large RGBA tensors at the model-step boundary. The V2
        presentation manager may show the same generated frame for multiple UI
        ticks, so the completed layer is retained until frame metadata changes.
        """
        source = self._frames.get(int(frame.data_ptr()))
        if source is None:
            return None
        cached = self._waypoint_layer
        if (
            source is self._waypoint_source
            and cached is not None
            and cached.device == frame.device
        ):
            return cached
        layer = render_waypoint_layers(
            (source.snapshot,),
            source.rig_pose_world[None, ...],
            self.calibration,
            width=self.width,
            height=self.height,
            device=frame.device,
            dtype=torch.float32,
        )[0]
        self._waypoint_source = source
        self._waypoint_layer = layer
        return layer

    def draw(self, ui: Any, ui_tick: int = 0) -> None:
        """Create or update the retained ImGui HUD widget tree."""
        widgets = self._ensure_widgets(ui)
        self._update_input_diagnostic(widgets)
        hud_frame = self._current
        if hud_frame is None:
            dots = "." * (1 + (ui_tick // 15) % 3)
            elapsed_s = max(0, int(time.monotonic() - self._loading_started_at_s))
            widgets.status_window.visible = True
            widgets.score.text = f"{self._loading_status}{dots}"
            widgets.time.text = f"ELAPSED  {elapsed_s}s"
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
        self._waypoint_source = None
        self._waypoint_layer = None
        self._validation_message = ""
        self._submission_pending = False
        self._loading_status = "LOADING WORLD MODEL"
        self._loading_started_at_s = time.monotonic()
        self._profile_pressed.clear()
        self._input_received_at_s.clear()
        self._reported_input_timestamps_us.clear()
        self._latest_input_latency_ms = None
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
        input_window = None
        input_state = None
        input_latency = None
        if self.profile_input_latency:
            input_window = ui.Window(
                ui.screen,
                "Input Latency",
                position=(14.0, float(max(14, self.height - 124))),
                size=(440.0, 110.0),
            )
            input_state = ui.Text(input_window, "")
            input_latency = ui.Text(input_window, "")
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
            input_window=input_window,
            input_state=input_state,
            input_latency=input_latency,
        )
        return self._widgets

    def _update_input_diagnostic(self, widgets: _HudWidgets) -> None:
        if not self.profile_input_latency:
            return
        assert widgets.input_window is not None
        assert widgets.input_state is not None
        assert widgets.input_latency is not None
        pressed = self._profile_pressed
        widgets.input_state.text = "  ".join(
            f"{label} [{'X' if bool(keys & pressed) else ' '}]"
            for label, keys in (
                ("W", {"w", "up"}),
                ("A", {"a", "left"}),
                ("S", {"s", "down"}),
                ("D", {"d", "right"}),
                ("SPACE", {"space"}),
            )
        )
        current = self._current
        latency = self._latest_input_latency_ms
        if current is None:
            counts = "TRANSITIONS  0    IGNORED  0    DROPPED  0"
        else:
            counts = (
                f"TRANSITIONS  {current.input_transition_count}    "
                f"IGNORED  {current.input_ignored_event_count}    "
                f"DROPPED  {current.input_dropped_transition_count}"
            )
        latency_label = (
            "UI TO MODEL FRAME  --"
            if latency is None
            else f"UI TO MODEL FRAME  {latency:.1f} ms"
        )
        widgets.input_latency.text = f"{latency_label}\n{counts}"

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
        model_loop = self.model_loop
        if model_loop is None:
            self._validation_message = "Model loop is not ready."
            return
        self._submission_pending = True
        self._validation_message = "Submitting score..."
        if self._widgets is not None:
            self._widgets.name_input.enabled = False
            self._widgets.submit.enabled = False
        invoke_async(
            model_loop,
            lambda state, name=normalized: state.submit_player_name(name),
        )


class CrazyRobotaxiSlangPyUILoop(SlangPyUILoop[TaxiHudState]):
    """Present generated frames beneath a responsive SlangPy taxi HUD."""

    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw the HUD and return the current generated frame beneath it."""
        self.state.consume_input_events(events)
        frames = self.presented_model_frames()
        video = frames[0] if frames else None
        bev_overlay = frames[1] if len(frames) > 1 else None
        if video is not None:
            self.state.select_presented_frame(video)
        self.state.draw(ui, step_index)
        if video is None:
            return None
        world = video.to(torch.float32)
        waypoints = self.state.waypoint_layer(video)
        if waypoints is not None:
            world = self._presentation_manager.composite(world, waypoints)
        if bev_overlay is None:
            return world
        if bev_overlay.shape[0] != 4:
            raise ValueError("BEV presentation frames must use [4,H,W]")
        return self._presentation_manager.composite(
            world,
            bev_overlay.to(device=video.device, dtype=torch.float32),
        )

    def reset(self) -> None:
        """Reset UI-owned state and retained renderer resources."""
        self.state.reset()
        super().reset()


def build_hud_frames(
    video_tchw: Tensor,
    snapshots: Sequence[object],
    rig_poses_world: npt.NDArray[np.float32],
    *,
    transition_timestamps_us: Sequence[int | None] | None = None,
    input_transition_count: int = 0,
    input_ignored_event_count: int = 0,
    input_dropped_transition_count: int = 0,
) -> tuple[TaxiHudFrame, ...]:
    """Build immutable UI messages aligned with generated tensor frames."""
    frame_count = int(video_tchw.shape[0])
    if len(snapshots) != frame_count:
        raise ValueError("Video and game snapshots must align")
    poses = np.asarray(rig_poses_world, dtype=np.float32)
    if poses.shape != (frame_count, 4, 4):
        raise ValueError("Video and rig poses must align")
    if transition_timestamps_us is None:
        transition_timestamps_us = (None,) * frame_count
    if len(transition_timestamps_us) != frame_count:
        raise ValueError("Input transitions and video frames must align")
    frames = []
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, TaxiGameSnapshot):
            raise TypeError("Taxi HUD received a non-taxi game snapshot")
        pose = poses[index].copy()
        pose.setflags(write=False)
        frames.append(
            TaxiHudFrame(
                frame_key=int(video_tchw[index].data_ptr()),
                snapshot=snapshot,
                rig_pose_world=pose,
                transition_timestamp_us=transition_timestamps_us[index],
                input_transition_count=input_transition_count,
                input_ignored_event_count=input_ignored_event_count,
                input_dropped_transition_count=input_dropped_transition_count,
            )
        )
    return tuple(frames)


def _normalize_profile_key(key: str) -> str:
    if key == " ":
        return "space"
    normalized = key.strip().lower()
    return {
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "spacebar": "space",
    }.get(normalized, normalized)


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
    "CrazyRobotaxiSlangPyUILoop",
    "TaxiHudFrame",
    "TaxiHudState",
    "build_hud_frames",
]
