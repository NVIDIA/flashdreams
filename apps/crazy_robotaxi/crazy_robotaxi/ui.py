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
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.config import BevConfig
from omnidreams_game_engine.types import CameraCalibration
from torch import Tensor

from crazy_robotaxi.game_selection import GameMapOption, GameMode, GameSelection
from crazy_robotaxi.high_scores import format_race_time_us, validate_player_name
from crazy_robotaxi.race import RaceGameSnapshot, project_race_gate_to_camera
from crazy_robotaxi.rules import (
    TaxiCameraMarkerProjection,
    TaxiGameSnapshot,
    project_segment_pose_to_bev,
    project_target_pose_to_bev,
    project_target_pose_to_bev_edge,
)
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

_VIDEO_FPS_WINDOW_SECONDS = 2.0
"""Rolling window used to smooth the generated-video frame-rate estimate."""

_BEV_WAYPOINT_ALPHA = 0.5
"""Opacity of visible pickup and drop-off waypoints on the BEV map."""

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

    snapshot: TaxiGameSnapshot | RaceGameSnapshot
    """Game-rules snapshot for the corresponding simulation frame."""

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

    ready_event: torch.cuda.Event | None = None
    """CUDA event recorded after the model produced this frame's tensors."""


@dataclass(slots=True)
class TaxiHudState:
    """Mutable Dear ImGui state owned exclusively by the V2 UI thread."""

    width: int
    """Presentation width in pixels."""

    height: int
    """Presentation height in pixels."""

    calibration: CameraCalibration | None
    """Camera calibration used to project world markers on the UI thread."""

    bev: BevConfig = BevConfig()
    """BEV camera geometry used to place navigation markers on the map."""

    profile_input_latency: bool = False
    """Whether input arrival and model-frame latency diagnostics are visible."""

    show_fps: bool = False
    """Whether to display the measured generated-video frame rate."""

    map_options: tuple[GameMapOption, ...] = ()
    """Lightweight authored-map choices supplied by the application."""

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

    _ready_source: TaxiHudFrame | None = None
    """Presented frame whose model-to-UI CUDA dependency was submitted."""

    _name_input: str = ""
    """Immediate-mode name-entry buffer retained by the UI state."""

    _bev_source_key: tuple[object, ...] | None = None
    """Identity, geometry, and format of the cached GPU BEV panel."""

    _bev_panel: Tensor | None = None
    """Cached normalized CHW BEV panel retained on its source device."""

    _bev_alpha: Tensor | None = None
    """Cached binary road/feature alpha mask retained on the source device."""

    _bev_rect: tuple[int, int, int, int] | None = None
    """Current ImGui content rectangle as ``(top, left, height, width)``."""

    _validation_message: str = ""
    """Name-entry validation or submission status."""

    _submission_pending: bool = False
    """Whether a validated name is already queued for the model thread."""

    _loading_status: str = "LOADING WORLD MODEL"
    """Current startup phase shown until the first model frame is presented."""

    _loading_started_at_s: float = field(default_factory=time.monotonic)
    """Monotonic timestamp used to make startup progress visibly live."""

    _menu_stage: Literal["mode", "map", "loading", "game"] = "mode"
    """Current startup screen owned by the UI thread."""

    _selected_game_mode: GameMode | None = None
    """Mode chosen on the first screen while the map screen is visible."""

    _profile_pressed: set[str] = field(default_factory=set)
    """Normalized drive keys currently held according to UI-thread events."""

    _input_received_at_s: OrderedDict[int, float] = field(default_factory=OrderedDict)
    """UI receipt times keyed by V2 session-relative event timestamp."""

    _reported_input_timestamps_us: set[int] = field(default_factory=set)
    """Input transitions already correlated with a presented model frame."""

    _latest_input_latency_ms: float | None = None
    """Latest UI-ingress-to-model-frame-selection latency measurement."""

    _presented_frame_times_s: deque[float] = field(default_factory=deque)
    """Recent times when distinct generated video frames were selected."""

    _video_fps: float = 0.0
    """Generated-video frame rate estimated from recent selections."""

    def publish(self, frames: Sequence[TaxiHudFrame]) -> None:
        """Publish immutable model-frame state to the UI-owned lookup."""
        for frame in frames:
            self._frames[frame.frame_key] = frame
            self._frames.move_to_end(frame.frame_key)
        while len(self._frames) > _MAX_BUFFERED_HUD_FRAMES:
            self._frames.popitem(last=False)

    def select_presented_frame(self, frame: Tensor) -> TaxiHudFrame | None:
        """Select the HUD snapshot aligned with ``frame`` when available."""
        if self.model_loop is not None and self._menu_stage in {"mode", "map"}:
            return None
        selected = self._frames.get(int(frame.data_ptr()))
        if selected is not None:
            frame_changed = selected is not self._current
            if (
                self._current is None
                or selected.snapshot.session_state
                != self._current.snapshot.session_state
            ):
                self._validation_message = ""
                self._submission_pending = False
            self._current = selected
            self._menu_stage = "game"
            if frame_changed:
                self._record_presented_frame(time.monotonic())
            self._record_presented_input(selected)
        return self._current

    def _record_presented_frame(self, now_s: float) -> None:
        """Update generated-video throughput after selecting a new frame."""
        times = self._presented_frame_times_s
        times.append(now_s)
        cutoff_s = now_s - _VIDEO_FPS_WINDOW_SECONDS
        while len(times) >= 3 and times[1] <= cutoff_s:
            times.popleft()
        if len(times) < 2:
            self._video_fps = 0.0
            return
        elapsed_s = times[-1] - times[0]
        if elapsed_s > 0.0:
            self._video_fps = (len(times) - 1) / elapsed_s

    def consume_input_events(self, events: UserInputEvents) -> None:
        """Track responsive drive state and receipt times on the UI thread."""
        received = events.get_events()
        if any(_is_escape_press(event) for event in received):
            self._handle_escape()
        if not self.profile_input_latency:
            return
        for event in received:
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

    def wait_for_presented_frame(self, stream: torch.cuda.Stream) -> bool:
        """Submit the current model frame's dependency to ``stream`` once."""
        selected = self._current
        if selected is None or selected is self._ready_source:
            return False
        if selected.ready_event is not None:
            stream.wait_event(selected.ready_event)
        self._ready_source = selected
        return selected.ready_event is not None

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

    def activate_scene(self, calibration: CameraCalibration) -> None:
        """Install projection data after the model thread loads the chosen map."""
        self._clear_presented_game()
        self.calibration = calibration
        self._menu_stage = "loading"

    def _handle_escape(self) -> None:
        model_loop = self.model_loop
        if self._menu_stage == "game":
            self.reset()
            self._menu_stage = "map"
            if model_loop is not None:
                invoke_async(
                    model_loop,
                    lambda model_state: model_state.return_to_map_menu(),
                )
        elif self._menu_stage == "map":
            self._selected_game_mode = None
            self._menu_stage = "mode"
        elif self._menu_stage == "mode" and model_loop is not None:
            self._loading_status = "EXITING GAME"
            self._loading_started_at_s = time.monotonic()
            self._menu_stage = "loading"
            invoke_async(model_loop, lambda model_state: model_state.request_exit())

    def _select_mode(self, mode: GameMode) -> None:
        self._selected_game_mode = mode
        self._menu_stage = "map"

    def _select_game(
        self,
        option: GameMapOption,
        *,
        race_course_id: str | None = None,
    ) -> None:
        mode = self._selected_game_mode
        model_loop = self.model_loop
        if mode is None or model_loop is None or self._menu_stage != "map":
            return
        selection = GameSelection(
            mode=mode,
            map_option=option,
            race_course_id=race_course_id,
        )
        self._menu_stage = "loading"
        self._loading_status = f"LOADING {option.name.upper()}"
        self._loading_started_at_s = time.monotonic()
        invoke_async(
            model_loop,
            lambda model_state, value=selection: model_state.select_game(value),
        )

    def draw_waypoints(self, imgui: Any, frame: Tensor) -> None:
        """Draw cached world-marker projections aligned with ``frame``."""
        calibration = self.calibration
        if calibration is None:
            return
        source = self._frames.get(int(frame.data_ptr()))
        if source is None:
            return
        if source is not self._waypoint_source:
            if isinstance(source.snapshot, TaxiGameSnapshot):
                self._waypoint_projections = project_waypoints(
                    source.snapshot,
                    source.rig_pose_world,
                    calibration,
                    width=self.width,
                    height=self.height,
                )
            else:
                self._waypoint_projections = ()
            self._waypoint_source = source
        if isinstance(source.snapshot, TaxiGameSnapshot):
            draw_waypoint_markers(
                imgui,
                self._waypoint_projections,
                phase=source.snapshot.phase,
                width=self.width,
                height=self.height,
            )
        elif source.snapshot.checkpoint_markers:
            camera = FThetaCameraModel(
                calibration,
                output_width=self.width,
                output_height=self.height,
            )
            gate = project_race_gate_to_camera(
                source.snapshot,
                source.rig_pose_world,
                camera,
                image_width=self.width,
                image_height=self.height,
            )
            if gate is not None:
                draw_list = imgui.get_background_draw_list()
                color = int(
                    imgui.color_convert_float4_to_u32(
                        imgui.ImVec4(1.0, 0.18, 0.08, 1.0)
                    )
                )
                draw_list.add_line(
                    imgui.ImVec2(*gate[0]), imgui.ImVec2(*gate[1]), color, 6.0
                )

    def draw(
        self,
        imgui: Any,
        ui_tick: int = 0,
        *,
        bev_frame: Tensor | None = None,
    ) -> None:
        """Draw one immediate Dear ImGui HUD frame."""
        self._bev_rect = None
        self._draw_fps_counter(imgui)
        if self._menu_stage == "mode":
            self._draw_mode_selection(imgui)
            return
        if self._menu_stage == "map":
            self._draw_map_selection(imgui)
            return
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
        if isinstance(snapshot, RaceGameSnapshot) and snapshot.session_state in {
            "awaiting_start",
            "racing",
        }:
            lap = (
                "POINT TO POINT"
                if snapshot.lap_count == 0
                else f"LAP  {snapshot.completed_laps + 1}/{snapshot.lap_count}"
            )
            self._draw_text_window(
                imgui,
                "Crazy Robotaxi Race",
                position=(14.0, 14.0),
                size=(360.0, 190.0),
                lines=(
                    format_race_time_us(snapshot.elapsed_time_us),
                    lap,
                    (
                        f"CHECKPOINT  {snapshot.checkpoint_index + 1}/"
                        f"{snapshot.checkpoint_count}"
                    ),
                    f"{snapshot.target_label}  {snapshot.distance_m:04.0f} m",
                    snapshot.event.replace("_", " ").upper()
                    if snapshot.event is not None
                    else "",
                ),
            )
            self._draw_navigation_arrow(
                imgui,
                snapshot.relative_bearing_rad,
                color_rgb=(1.0, 0.18, 0.08),
            )
            self._draw_bev_window(imgui, bev_frame, hud_frame)
        elif (
            isinstance(snapshot, TaxiGameSnapshot)
            and snapshot.session_state == "playing"
        ):
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
            self._draw_navigation_arrow(
                imgui,
                snapshot.relative_bearing_rad,
                color_rgb=(
                    (118.0 / 255.0, 185.0 / 255.0, 0.0)
                    if snapshot.phase == "seeking_pickup"
                    else (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
                ),
            )
            self._draw_bev_window(imgui, bev_frame, hud_frame)
        self._draw_terminal(imgui, snapshot)
        self._draw_input_diagnostic(imgui)

    def _draw_fps_counter(self, imgui: Any) -> None:
        """Draw the measured generated-video rate when the counter is enabled."""
        if not self.show_fps:
            return
        width = 170.0
        self._draw_text_window(
            imgui,
            "Performance",
            position=(float(max(14.0, self.width - width - 14.0)), 14.0),
            size=(width, 66.0),
            lines=(f"VIDEO FPS  {self._video_fps:5.1f}",),
        )

    def reset(self) -> None:
        """Clear per-generation HUD snapshots and editable UI state."""
        self._clear_presented_game()
        self._validation_message = ""
        self._submission_pending = False
        self._loading_status = "LOADING WORLD MODEL"
        self._loading_started_at_s = time.monotonic()
        self._profile_pressed.clear()
        self._input_received_at_s.clear()
        self._reported_input_timestamps_us.clear()
        self._latest_input_latency_ms = None
        self._name_input = ""

    def _clear_presented_game(self) -> None:
        """Discard frame-aligned HUD and BEV resources from the previous game."""
        self._frames.clear()
        self._current = None
        self._waypoint_source = None
        self._waypoint_projections = ()
        self._ready_source = None
        self._bev_source_key = None
        self._bev_panel = None
        self._bev_alpha = None
        self._bev_rect = None
        self._presented_frame_times_s.clear()
        self._video_fps = 0.0

    def _draw_mode_selection(self, imgui: Any) -> None:
        window_width = max(1.0, min(460.0, float(self.width) - 28.0))
        window_height = max(1.0, min(250.0, float(self.height) - 28.0))
        _prepare_window(
            imgui,
            position=(
                max(14.0, (self.width - window_width) / 2.0),
                max(14.0, (self.height - window_height) / 2.0),
            ),
            size=(window_width, window_height),
            alpha=0.94,
        )
        visible = _begin_window(imgui, "Crazy Robotaxi — Select Game Mode")
        try:
            if not visible:
                return
            imgui.text("SELECT GAME MODE")
            imgui.text("Choose how you want to play.")
            imgui.text("ESC: EXIT")
            if imgui.button("TAXI"):
                self._select_mode("taxi")
            if imgui.button("RACE"):
                self._select_mode("race")
        finally:
            imgui.end()

    def _draw_map_selection(self, imgui: Any) -> None:
        mode = self._selected_game_mode
        if mode is None:
            self._menu_stage = "mode"
            return
        window_width = max(1.0, min(620.0, float(self.width) - 28.0))
        window_height = max(1.0, min(560.0, float(self.height) - 28.0))
        _prepare_window(
            imgui,
            position=(
                max(14.0, (self.width - window_width) / 2.0),
                max(14.0, (self.height - window_height) / 2.0),
            ),
            size=(window_width, window_height),
            alpha=0.94,
        )
        title = (
            "Crazy Robotaxi — Select Map"
            if mode == "taxi"
            else "Crazy Robotaxi — Select Map & Race Course"
        )
        visible = _begin_window(imgui, title)
        try:
            if not visible:
                return
            if imgui.button("BACK"):
                self._selected_game_mode = None
                self._menu_stage = "mode"
                return
            imgui.text("ESC: BACK")
            imgui.text("SELECT MAP" if mode == "taxi" else "SELECT MAP & RACE COURSE")
            available = False
            for index, option in enumerate(self.map_options):
                if mode == "taxi":
                    available = True
                    if imgui.button(f"{option.name}##map-{index}"):
                        self._select_game(option)
                    continue
                if not option.race_course_ids:
                    continue
                available = True
                imgui.text(option.name)
                for course_index, course_id in enumerate(option.race_course_ids):
                    label = course_id.replace("-", " ").replace("_", " ").upper()
                    if imgui.button(f"{label}##course-{index}-{course_index}"):
                        self._select_game(option, race_course_id=course_id)
            if not available:
                imgui.text("NO COMPATIBLE MAPS FOUND")
        finally:
            imgui.end()

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

    def _draw_bev_window(
        self,
        imgui: Any,
        bev_frame: Tensor | None,
        hud_frame: TaxiHudFrame,
    ) -> None:
        if bev_frame is None:
            return
        maximum_width, maximum_height = bev_display_extent(self.width, self.height)
        frame_height, frame_width = (int(value) for value in bev_frame.shape[1:])
        scale = min(maximum_width / frame_width, maximum_height / frame_height)
        image_width = max(1, round(frame_width * scale))
        image_height = max(1, round(frame_height * scale))
        if image_width <= 4 or image_height <= 4:
            return
        padding = 16
        window_size = (
            float(image_width + padding),
            float(image_height + padding),
        )
        margin = float(max(8, min(self.width, self.height) // 80))
        position = (
            float(self.width) - window_size[0] - margin,
            float(self.height) - window_size[1] - margin,
        )
        # The app composites the CUDA BEV beneath this transparent content area.
        # ImGui owns layout and clipping without drawing window chrome.
        _prepare_window(imgui, position=position, size=window_size, alpha=0.0)
        visible = _begin_window(
            imgui,
            "Map",
            extra_flags=("no_title_bar", "no_background"),
        )
        try:
            if visible:
                cursor = imgui.get_cursor_screen_pos()
                left, top = _point_xy(cursor)
                self._bev_rect = (
                    max(0, round(top)),
                    max(0, round(left)),
                    image_height,
                    image_width,
                )
                imgui.dummy(imgui.ImVec2(float(image_width), float(image_height)))
                self._draw_bev_navigation(imgui, hud_frame)
                self._draw_bev_border(imgui)
        finally:
            imgui.end()

    def _draw_bev_border(self, imgui: Any) -> None:
        """Draw an opaque white border at the exact BEV image extent."""
        rect = self._bev_rect
        if rect is None:
            return
        top, left, height, width = rect
        draw_list = imgui.get_background_draw_list()
        draw_list.add_rect(
            imgui.ImVec2(float(left), float(top)),
            imgui.ImVec2(float(left + width), float(top + height)),
            _imgui_color(imgui, (1.0, 1.0, 1.0, 1.0)),
            0.0,
            2.0,
            0,
        )

    def _draw_navigation_arrow(
        self,
        imgui: Any,
        bearing_rad: float,
        *,
        color_rgb: tuple[float, float, float],
    ) -> None:
        """Draw the always-visible target-bearing arrow from the original HUD."""
        draw_list = imgui.get_background_draw_list()
        center_x = float(self.width) * 0.5
        center_y = 74.0
        radius = 30.0
        direction_x = -math.sin(bearing_rad)
        direction_y = -math.cos(bearing_rad)
        perpendicular_x = -direction_y
        perpendicular_y = direction_x
        tip_x = center_x + direction_x * radius
        tip_y = center_y + direction_y * radius
        base_x = center_x + direction_x * radius * 0.25
        base_y = center_y + direction_y * radius * 0.25
        tail = imgui.ImVec2(
            center_x - direction_x * radius * 0.62,
            center_y - direction_y * radius * 0.62,
        )
        left_x = base_x - perpendicular_x * radius * 0.42
        left_y = base_y - perpendicular_y * radius * 0.42
        right_x = base_x + perpendicular_x * radius * 0.42
        right_y = base_y + perpendicular_y * radius * 0.42
        color = _imgui_color(imgui, (*color_rgb, 1.0))
        panel = _imgui_color(
            imgui,
            (12.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0, 0.75),
        )
        center = imgui.ImVec2(center_x, center_y)
        draw_list.add_circle_filled(center, 42.0, panel)
        draw_list.add_circle(center, 42.0, color, 0, 3.0)
        draw_list.add_line(tail, imgui.ImVec2(base_x, base_y), color, 7.0)
        tip = imgui.ImVec2(tip_x, tip_y)
        draw_list.add_triangle_filled(
            tip,
            imgui.ImVec2(left_x, left_y),
            imgui.ImVec2(right_x, right_y),
            color,
        )

    def _draw_bev_navigation(self, imgui: Any, hud_frame: TaxiHudFrame) -> None:
        """Draw target markers and off-map arrows over the composited BEV."""
        rect = self._bev_rect
        if rect is None or not self.bev.enabled:
            return
        top, left, height, width = rect
        if width <= 0 or height <= 0:
            return
        snapshot = hud_frame.snapshot
        pose = hud_frame.rig_pose_world
        draw_list = imgui.get_background_draw_list()

        if isinstance(snapshot, RaceGameSnapshot):
            segment = project_segment_pose_to_bev(
                np.asarray(
                    [snapshot.gate_start_xyz_m, snapshot.gate_end_xyz_m],
                    dtype=np.float32,
                ),
                pose,
                self.bev,
            )
            red = _imgui_color(imgui, (1.0, 0.18, 0.08, 1.0))
            if segment is not None:
                start, end = (
                    imgui.ImVec2(left + uv[0] * width, top + uv[1] * height)
                    for uv in segment
                )
                white = _imgui_color(imgui, (1.0, 1.0, 1.0, 1.0))
                draw_list.add_line(start, end, white, 9.0)
                draw_list.add_line(start, end, red, 6.0)
                return
            self._draw_bev_edge_arrow(
                imgui,
                snapshot.target_xyz_m,
                pose,
                color=red,
            )
            return

        rgb = (
            (118.0 / 255.0, 185.0 / 255.0, 0.0)
            if snapshot.phase == "seeking_pickup"
            else (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
        )
        color = _imgui_color(imgui, (*rgb, _BEV_WAYPOINT_ALPHA))
        targets = (
            snapshot.pickup_targets_xyz_m
            if snapshot.phase == "seeking_pickup" and snapshot.pickup_targets_xyz_m
            else (snapshot.target_xyz_m,)
        )
        visible = False
        white = _imgui_color(imgui, (1.0, 1.0, 1.0, _BEV_WAYPOINT_ALPHA))
        outline = _imgui_color(imgui, (0.08, 0.08, 0.12, _BEV_WAYPOINT_ALPHA))
        for target in targets:
            u, v, inside = project_target_pose_to_bev(target, pose, self.bev)
            if not inside:
                continue
            visible = True
            center = imgui.ImVec2(left + u * width, top + v * height)
            radius = float(max(8, min(width, height) // 16))
            draw_list.add_circle_filled(center, radius + 3.0, white)
            draw_list.add_circle_filled(center, radius, color)
            draw_list.add_circle(center, radius, outline, 0, 2.0)
        if snapshot.phase == "to_dropoff" and not visible:
            self._draw_bev_edge_arrow(
                imgui,
                snapshot.target_xyz_m,
                pose,
                color=_imgui_color(imgui, (*rgb, 1.0)),
            )

    def _draw_bev_edge_arrow(
        self,
        imgui: Any,
        target_xyz_m: tuple[float, float, float],
        pose: npt.NDArray[np.float32],
        *,
        color: int,
    ) -> None:
        rect = self._bev_rect
        assert rect is not None
        projected = project_target_pose_to_bev_edge(target_xyz_m, pose, self.bev)
        if projected is None:
            return
        top, left, height, width = rect
        edge_x = left + projected[0] * width
        edge_y = top + projected[1] * height
        center_x = left + width * 0.5
        center_y = top + height * 0.5
        delta_x, delta_y = edge_x - center_x, edge_y - center_y
        length = math.hypot(delta_x, delta_y)
        if length <= 1.0e-6:
            return
        direction_x, direction_y = delta_x / length, delta_y / length
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        size = float(max(9, min(width, height) // 14))
        arrow_x = edge_x - direction_x * (size + 3.0)
        arrow_y = edge_y - direction_y * (size + 3.0)

        def points(scale: float) -> tuple[Any, Any, Any]:
            tip = imgui.ImVec2(
                arrow_x + direction_x * size * scale,
                arrow_y + direction_y * size * scale,
            )
            base_x = arrow_x - direction_x * size * scale * 0.72
            base_y = arrow_y - direction_y * size * scale * 0.72
            half_width = size * scale * 0.68
            return (
                tip,
                imgui.ImVec2(
                    base_x + perpendicular_x * half_width,
                    base_y + perpendicular_y * half_width,
                ),
                imgui.ImVec2(
                    base_x - perpendicular_x * half_width,
                    base_y - perpendicular_y * half_width,
                ),
            )

        draw_list = imgui.get_background_draw_list()
        white = _imgui_color(imgui, (1.0, 1.0, 1.0, 1.0))
        draw_list.add_triangle_filled(*points(1.0), white)
        draw_list.add_triangle_filled(*points(0.68), color)

    def composite_bev(self, video: Tensor, frame: Tensor | None) -> Tensor:
        """Composite the current BEV into its ImGui-owned rectangle on-device."""
        rect = self._bev_rect
        if frame is None or rect is None:
            return video
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("BEV presentation frames must use [3,H,W]")
        if frame.dtype != torch.uint8 and not frame.is_floating_point():
            raise ValueError("BEV presentation frames must be uint8 or floating point")
        if frame.device != video.device:
            raise ValueError("BEV and video presentation frames must share a device")

        top, left, image_height, image_width = rect
        bottom = min(int(video.shape[-2]), top + image_height)
        right = min(int(video.shape[-1]), left + image_width)
        if bottom <= top or right <= left:
            return video
        source_key = (
            id(self._current),
            int(frame.data_ptr()),
            tuple(int(value) for value in frame.shape),
            frame.dtype,
            frame.device,
            image_height,
            image_width,
        )
        panel = self._bev_panel
        alpha = self._bev_alpha
        if source_key != self._bev_source_key or panel is None or alpha is None:
            source = frame.detach().to(dtype=torch.float32)
            panel = source.div(127.5).sub(1.0) if frame.dtype == torch.uint8 else source
            if frame.dtype == torch.uint8:
                alpha = frame.detach().ne(0).any(dim=0, keepdim=True)
            else:
                alpha = frame.detach().gt(-1.0 + 1.0e-6).any(dim=0, keepdim=True)
            alpha = alpha.to(dtype=torch.float32)
            if tuple(panel.shape[-2:]) != (image_height, image_width):
                panel = functional.interpolate(
                    panel.unsqueeze(0),
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                alpha = functional.interpolate(
                    alpha.unsqueeze(0),
                    size=(image_height, image_width),
                    mode="nearest",
                )[0]
            self._bev_source_key = source_key
            self._bev_panel = panel
            self._bev_alpha = alpha

        output = video.clone()
        target = output[:, top:bottom, left:right]
        source_panel = panel[:, : bottom - top, : right - left]
        source_alpha = alpha[:, : bottom - top, : right - left]
        target.mul_(1.0 - source_alpha).add_(source_panel * source_alpha)
        return output

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

    def _draw_terminal(
        self, imgui: Any, snapshot: TaxiGameSnapshot | RaceGameSnapshot
    ) -> None:
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
            race = isinstance(snapshot, RaceGameSnapshot)
            imgui.text(
                ("NEW BEST TIME" if race else "NEW HIGH SCORE")
                if awaiting_name
                else "LEADERBOARD"
            )
            if race:
                imgui.text(
                    "FINAL TIME  " + format_race_time_us(snapshot.final_time_us or 0)
                )
            else:
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
                    clicked = imgui.button("Submit time" if race else "Submit score")
                finally:
                    if disabled:
                        imgui.end_disabled()
                if submitted or clicked:
                    self._submit_name(self._name_input)
                if self._validation_message:
                    imgui.text(self._validation_message)
            else:
                if race:
                    entries = "\n".join(
                        f"{rank:>2}. {entry.name:<12} "
                        f"{format_race_time_us(entry.elapsed_time_us)}"
                        for rank, entry in enumerate(snapshot.leaderboard, start=1)
                    )
                else:
                    entries = "\n".join(
                        f"{rank:>2}. {entry.name:<12} {entry.score:>7}"
                        for rank, entry in enumerate(snapshot.leaderboard, start=1)
                    )
                entries = entries or "NO SCORES YET"
                imgui.text(entries)
                imgui.text("Press R to play again")
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

    def __init__(
        self,
        *,
        presentation_device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the HUD renderer and its app-owned presentation stream.

        Args:
            presentation_device: Device used for generated and composited
                frames; ``None`` retains the calling thread's current stream.
            **kwargs: Arguments forwarded to :class:`ImGuiUILoop`.
        """
        super().__init__(**kwargs)
        self._presentation_device = (
            None if presentation_device is None else torch.device(presentation_device)
        )
        self._presentation_stream: torch.cuda.Stream | None = None

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw the HUD and return the generated world frame beneath it."""
        presentation_stream = self._activate_presentation_stream()
        self.state.consume_input_events(events)
        frames = self.presented_model_frames()
        video = frames[0] if frames else None
        bev_frame = frames[1] if len(frames) > 1 else None
        if video is not None:
            self.state.select_presented_frame(video)
            if presentation_stream is not None:
                self.state.wait_for_presented_frame(presentation_stream)
                video.record_stream(presentation_stream)
                if bev_frame is not None:
                    bev_frame.record_stream(presentation_stream)
            self.state.draw_waypoints(imgui, video)
        self.state.draw(imgui, step_index, bev_frame=bev_frame)
        if video is None:
            return None
        converted = video.to(torch.float32)
        return self.state.composite_bev(converted, bev_frame)

    def _activate_presentation_stream(self) -> torch.cuda.Stream | None:
        """Make the app-owned CUDA stream current on the V2 UI thread."""
        device = self._presentation_device
        if device is None or device.type != "cuda":
            return None
        with torch.cuda.device(device):
            if self._presentation_stream is None:
                self._presentation_stream = torch.cuda.Stream(device=device)
            torch.cuda.set_stream(self._presentation_stream)
        return self._presentation_stream

    def reset(self) -> None:
        """Reset UI-owned state and retained renderer resources."""
        self.state.reset()
        super().reset()

    def close(self) -> None:
        """Release the renderer and restore the UI thread's default stream."""
        try:
            super().close()
        finally:
            if self._presentation_stream is not None:
                assert self._presentation_device is not None
                with torch.cuda.device(self._presentation_device):
                    torch.cuda.set_stream(
                        torch.cuda.default_stream(self._presentation_device)
                    )
                self._presentation_stream = None


def build_hud_frames(
    video_tchw: Tensor,
    snapshots: Sequence[object],
    rig_poses_world: npt.NDArray[np.float32],
    *,
    transition_timestamps_us: Sequence[int | None] | None = None,
    input_transition_count: int = 0,
    input_ignored_event_count: int = 0,
    input_dropped_transition_count: int = 0,
    ready_event: torch.cuda.Event | None = None,
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
        if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
            raise TypeError("Taxi HUD received an unknown game snapshot")
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
                ready_event=ready_event,
            )
        )
    return tuple(frames)


def _is_escape_press(event: object) -> bool:
    """Return whether an input event is a pressed Escape key."""
    return (
        isinstance(event, KeyboardUserInputEvent)
        and event.state is KeyboardInputState.PRESSED
        and str(event.key).strip().lower() in {"esc", "escape"}
    )


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


def _begin_window(
    imgui: Any,
    title: str,
    *,
    extra_flags: Sequence[str] = (),
) -> bool:
    """Begin a fixed HUD window and normalize ImGui's binding return form."""
    flags = 0
    window_flags = imgui.WindowFlags_
    for name in (
        "no_move",
        "no_resize",
        "no_collapse",
        "no_saved_settings",
        *extra_flags,
    ):
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


def _point_xy(value: Any) -> tuple[float, float]:
    """Return an ImGui vector's coordinates across supported Python bindings."""
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def _imgui_color(
    imgui: Any,
    rgba: tuple[float, float, float, float],
) -> int:
    return int(imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba)))


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
