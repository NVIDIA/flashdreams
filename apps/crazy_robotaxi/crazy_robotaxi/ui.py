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

"""Dear ImGui HUD and presentation state for Crazy Robotaxi."""

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
from crazy_robotaxi.rules import TaxiCameraMarkerProjection, TaxiGameSnapshot
from crazy_robotaxi.world_overlay import (
    draw_waypoints as draw_waypoint_markers,
)
from crazy_robotaxi.world_overlay import (
    project_waypoints,
)
from flashdreams.api_v2.loop import ILoop, invoke_async
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
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


def bev_display_extent(video_width: int, video_height: int) -> tuple[int, int]:
    """Return the largest BEV image extent used by the fixed HUD layout."""
    size = max(1, min(int(video_width) // 4, int(video_height) // 3))
    return size, size


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
class TaxiHudState:
    """Mutable Dear ImGui state owned exclusively by the V2 UI thread."""

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
    """Frame metadata used by the cached waypoint projections."""

    _waypoint_projections: tuple[TaxiCameraMarkerProjection, ...] = ()
    """Cached world-marker projections for the presented generated frame."""

    _name_input: str = ""
    """Immediate-mode name-entry buffer retained by the UI state."""

    _bev_source_key: tuple[int, tuple[int, ...]] | None = None
    """Identity and shape of the raw BEV frame cached for ImGui upload."""

    _bev_pixels: npt.NDArray[np.uint8] | None = None
    """Cached HWC RGB bytes for the currently presented BEV frame."""

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
            recognized = False
            if isinstance(event, FocusUserInputEvent) and not event.focused:
                self._profile_pressed.clear()
                recognized = True
            elif isinstance(event, KeyboardUserInputEvent):
                key = _normalize_profile_key(str(event.key))
                if key not in _PROFILE_DRIVE_KEYS:
                    continue
                recognized = True
                if event.state is KeyboardInputState.PRESSED:
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

    def draw_waypoints(self, imgui: Any, frame: Tensor) -> None:
        """Draw cached world-marker projections aligned with ``frame``."""
        source = self._frames.get(int(frame.data_ptr()))
        if source is None:
            return
        if source is not self._waypoint_source:
            self._waypoint_projections = project_waypoints(
                source.snapshot,
                source.rig_pose_world,
                self.calibration,
                width=self.width,
                height=self.height,
            )
            self._waypoint_source = source
        draw_waypoint_markers(
            imgui,
            self._waypoint_projections,
            phase=source.snapshot.phase,
            width=self.width,
            height=self.height,
        )

    def draw(
        self,
        imgui: Any,
        ui_tick: int = 0,
        *,
        bev_frame: Tensor | None = None,
    ) -> None:
        """Draw one immediate Dear ImGui HUD frame."""
        hud_frame = self._current
        if hud_frame is None:
            dots = "." * (1 + (ui_tick // 15) % 3)
            elapsed_s = max(0, int(time.monotonic() - self._loading_started_at_s))
            self._draw_text_window(
                imgui,
                "Crazy Robotaxi",
                position=(14.0, 14.0),
                size=(360.0, 104.0),
                lines=(f"{self._loading_status}{dots}", f"ELAPSED  {elapsed_s}s"),
            )
            return

        snapshot = hud_frame.snapshot
        if snapshot.session_state == "playing":
            objective = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
            fare_time = (
                ""
                if snapshot.remaining_time_s is None
                else f"FARE TIME  {snapshot.remaining_time_s:04.1f}s"
            )
            self._draw_text_window(
                imgui,
                "Crazy Robotaxi",
                position=(14.0, 14.0),
                size=(360.0, 190.0),
                lines=(
                    _score_label(snapshot),
                    f"GAME TIME  {snapshot.global_remaining_time_s:05.1f}s",
                    f"{objective}  {snapshot.distance_m:04.0f} m",
                    fare_time,
                    _event_label(snapshot),
                ),
            )
            self._draw_text_window(
                imgui,
                "Navigation",
                position=(float(self.width - 294), 14.0),
                size=(280.0, 92.0),
                lines=(_navigation_label(snapshot.relative_bearing_rad),),
            )
            self._draw_bev_window(imgui, bev_frame)
        self._draw_terminal(imgui, snapshot)
        self._draw_input_diagnostic(imgui)

    def reset(self) -> None:
        """Clear per-generation HUD snapshots and editable UI state."""
        self._frames.clear()
        self._current = None
        self._waypoint_source = None
        self._waypoint_projections = ()
        self._validation_message = ""
        self._submission_pending = False
        self._loading_status = "LOADING WORLD MODEL"
        self._loading_started_at_s = time.monotonic()
        self._profile_pressed.clear()
        self._input_received_at_s.clear()
        self._reported_input_timestamps_us.clear()
        self._latest_input_latency_ms = None
        self._name_input = ""
        self._bev_source_key = None
        self._bev_pixels = None

    def _draw_text_window(
        self,
        imgui: Any,
        title: str,
        *,
        position: tuple[float, float],
        size: tuple[float, float],
        lines: Sequence[str],
    ) -> None:
        _prepare_window(imgui, position=position, size=size)
        visible = _begin_window(imgui, title)
        try:
            if visible:
                for line in lines:
                    if line:
                        imgui.text(line)
        finally:
            imgui.end()

    def _draw_bev_window(self, imgui: Any, bev_frame: Tensor | None) -> None:
        if bev_frame is None:
            return
        maximum_width, maximum_height = bev_display_extent(self.width, self.height)
        frame_height, frame_width = (int(value) for value in bev_frame.shape[1:])
        scale = min(maximum_width / frame_width, maximum_height / frame_height)
        image_width = max(1, round(frame_width * scale))
        image_height = max(1, round(frame_height * scale))
        if image_width <= 4 or image_height <= 4:
            return
        pixels = self._bev_image_pixels(bev_frame)
        padding = 16
        title_height = 34
        window_size = (
            float(image_width + padding),
            float(image_height + title_height),
        )
        margin = float(max(8, min(self.width, self.height) // 80))
        position = (
            float(self.width) - window_size[0] - margin,
            float(self.height) - window_size[1] - margin,
        )
        _prepare_window(imgui, position=position, size=window_size, alpha=0.82)
        visible = _begin_window(imgui, "Map")
        try:
            if visible:
                imgui.image(
                    "crazy_robotaxi_bev",
                    pixels,
                    size=(float(image_width), float(image_height)),
                )
        finally:
            imgui.end()

    def _bev_image_pixels(self, frame: Tensor) -> npt.NDArray[np.uint8]:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("BEV presentation frames must use [3,H,W]")
        source_key = (int(frame.data_ptr()), tuple(int(value) for value in frame.shape))
        if source_key == self._bev_source_key and self._bev_pixels is not None:
            return self._bev_pixels
        source = frame.detach()
        if source.dtype == torch.uint8:
            pixels = source.permute(1, 2, 0).contiguous().cpu().numpy()
        elif source.is_floating_point():
            pixels = (
                source.to(dtype=torch.float32)
                .add(1.0)
                .mul(127.5)
                .clamp_(0.0, 255.0)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        else:
            raise ValueError("BEV presentation frames must be uint8 or floating point")
        self._bev_source_key = source_key
        self._bev_pixels = np.ascontiguousarray(pixels)
        return self._bev_pixels

    def _draw_input_diagnostic(self, imgui: Any) -> None:
        if not self.profile_input_latency:
            return
        pressed = self._profile_pressed
        input_state = "  ".join(
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
        self._draw_text_window(
            imgui,
            "Input Latency",
            position=(14.0, float(max(14, self.height - 124))),
            size=(440.0, 110.0),
            lines=(input_state, latency_label, counts),
        )

    def _draw_terminal(self, imgui: Any, snapshot: TaxiGameSnapshot) -> None:
        awaiting_name = snapshot.session_state == "awaiting_name"
        leaderboard = snapshot.session_state == "leaderboard"
        if not (awaiting_name or leaderboard):
            return
        _prepare_window(
            imgui,
            position=(
                float(max(20, self.width // 2 - 270)),
                float(max(20, self.height // 2 - 220)),
            ),
            size=(540.0, 440.0),
        )
        visible = _begin_window(imgui, "Game Over")
        try:
            if not visible:
                return
            imgui.text("NEW HIGH SCORE" if awaiting_name else "LEADERBOARD")
            imgui.text(f"FINAL SCORE  {snapshot.score:06d}")
            if snapshot.high_score_rank is not None:
                imgui.text(f"RANK  #{snapshot.high_score_rank}")
            if awaiting_name:
                disabled = self._submission_pending
                if disabled:
                    imgui.begin_disabled()
                try:
                    submitted, self._name_input = imgui.input_text(
                        "Driver name",
                        self._name_input,
                        flags=imgui.InputTextFlags_.enter_returns_true,
                    )
                    clicked = imgui.button("Submit score")
                finally:
                    if disabled:
                        imgui.end_disabled()
                if submitted or clicked:
                    self._submit_name(self._name_input)
                if self._validation_message:
                    imgui.text(self._validation_message)
            else:
                entries = (
                    "\n".join(
                        f"{rank:>2}. {entry.name:<12} {entry.score:>7}"
                        for rank, entry in enumerate(snapshot.leaderboard, start=1)
                    )
                    or "NO SCORES YET"
                )
                imgui.text(entries)
        finally:
            imgui.end()

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
        invoke_async(
            model_loop,
            lambda state, name=normalized: state.submit_player_name(name),
        )


class CrazyRobotaxiImGuiUILoop(ImGuiUILoop[TaxiHudState]):
    """Present generated frames beneath a responsive Dear ImGui taxi HUD."""

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw the HUD and return the generated world frame beneath it."""
        self.state.consume_input_events(events)
        frames = self.presented_model_frames()
        video = frames[0] if frames else None
        bev_frame = frames[1] if len(frames) > 1 else None
        if video is not None:
            self.state.select_presented_frame(video)
            self.state.draw_waypoints(imgui, video)
        self.state.draw(imgui, step_index, bev_frame=bev_frame)
        if video is None:
            return None
        return video.to(torch.float32)

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


def _prepare_window(
    imgui: Any,
    *,
    position: tuple[float, float],
    size: tuple[float, float],
    alpha: float = 0.72,
) -> None:
    """Set deterministic overlay geometry for the next ImGui window."""
    imgui.set_next_window_pos(imgui.ImVec2(*position), imgui.Cond_.always)
    imgui.set_next_window_size(imgui.ImVec2(*size), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(alpha)


def _begin_window(imgui: Any, title: str) -> bool:
    """Begin a fixed HUD window and normalize ImGui's binding return form."""
    flags = 0
    window_flags = imgui.WindowFlags_
    for name in ("no_move", "no_resize", "no_collapse", "no_saved_settings"):
        flags |= int(getattr(window_flags, name))
    result = imgui.begin(title, flags=flags)
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


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
    "CrazyRobotaxiImGuiUILoop",
    "TaxiHudFrame",
    "TaxiHudState",
    "bev_display_extent",
    "build_hud_frames",
]
