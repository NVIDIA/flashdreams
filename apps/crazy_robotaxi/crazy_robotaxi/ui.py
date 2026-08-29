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
from importlib.resources import as_file, files
from pathlib import Path
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
from crazy_robotaxi.high_scores import (
    HighScoreEntry,
    RaceTimeEntry,
    format_race_time_us,
    validate_player_name,
)
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
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
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

_MPS_TO_MPH = 2.2369362920544
"""Metres-per-second to miles-per-hour conversion used by the source HUD."""

_TAXI_ACCENT_RGB = (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
_RACE_ACCENT_RGB = (118.0 / 255.0, 185.0 / 255.0, 0.0)

_PROFILE_DRIVE_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
_TRACE_LOGGER = logging.getLogger("flashdreams.runtime_v2.chunk_trace")
_TRACE_PREFIX = "[crazy-robotaxi-chunk-trace]"


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

    speed_mps: float = 0.0
    """Authoritative signed vehicle speed for the corresponding simulation frame."""

    transition_timestamp_us: int | None = None
    """V2 input transition represented by this frame, when one was received."""

    runtime_generation: int = 0
    model_step_index: int = -1
    rollout_epoch: int = 0
    autoregressive_index: int = -1
    frame_index: int = -1
    simulation_timestamp_us: int | None = None
    cache_finalize_returned_ns: int | None = None
    """Chunk-lifecycle correlation fields for input diagnosis."""


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

    initial_game_mode: GameMode | None = None
    """Mode selected explicitly by CLI, skipping the mode screen."""

    initial_map_path: Path | None = None
    """Map selected explicitly by CLI, skipping the map screen."""

    initial_race_course_id: str | None = None
    """Race course selected explicitly by CLI, skipping the course screen."""

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

    _bev_source_key: tuple[object, ...] | None = None
    """Identity, geometry, and format of the cached GPU BEV panel."""

    _bev_panel: Tensor | None = None
    """Cached normalized CHW BEV panel retained on its source device."""

    _bev_alpha: Tensor | None = None
    """Cached binary renderer coverage retained on the source device."""

    _bev_composite_source_key: tuple[object, ...] | None = None
    """Identity and layout of the cached video/BEV back buffer."""

    _bev_composite: Tensor | None = None
    """Cached float32 back buffer for the current presented frame."""

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

    _menu_stage: Literal["mode", "map", "course", "loading", "game"] = "mode"
    """Current startup screen owned by the UI thread."""

    _selected_game_mode: GameMode | None = None
    """Mode chosen on the first screen while the map screen is visible."""

    _selected_map_option: GameMapOption | None = None
    """Map chosen before the separate race-course screen."""

    _profile_pressed: set[str] = field(default_factory=set)
    """Normalized drive keys currently held according to UI-thread events."""

    _input_received_at_ns: OrderedDict[int, int] = field(default_factory=OrderedDict)
    """UI receipt times keyed by V2 session-relative event timestamp."""

    _reported_input_timestamps_us: set[int] = field(default_factory=set)
    """Input transitions already correlated with a presented model frame."""

    _latest_input_latency_ms: float | None = None
    """Latest UI-ingress-to-model-frame-selection latency measurement."""

    _latest_committed_frame: TaxiHudFrame | None = None
    """Newest generated frame metadata received from the model thread."""

    _presented_frame_times_s: deque[float] = field(default_factory=deque)
    """Recent times when distinct generated video frames were selected."""

    _video_fps: float = 0.0
    """Generated-video frame rate estimated from recent selections."""

    _gameplay_font: Any | None = None
    """Droid Sans face used for prominent gameplay feedback."""

    def publish(self, frames: Sequence[TaxiHudFrame]) -> None:
        """Publish immutable model-frame state to the UI-owned lookup."""
        for frame in frames:
            self._frames[frame.frame_key] = frame
            self._frames.move_to_end(frame.frame_key)
        if frames and frames[-1].autoregressive_index >= 0:
            self._latest_committed_frame = frames[-1]
        while len(self._frames) > _MAX_BUFFERED_HUD_FRAMES:
            self._frames.popitem(last=False)

    def select_presented_frame(self, frame: Tensor) -> TaxiHudFrame | None:
        """Select the HUD snapshot aligned with ``frame`` when available."""
        if self.model_loop is not None and self._menu_stage in {
            "mode",
            "map",
            "course",
        }:
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
            presented_at_ns = (
                time.monotonic_ns() if self.profile_input_latency else None
            )
            if frame_changed:
                self._record_presented_frame(
                    time.monotonic()
                    if presented_at_ns is None
                    else presented_at_ns / 1_000_000_000.0
                )
                if presented_at_ns is not None:
                    self._record_presented_trace(selected, presented_at_ns)
            if presented_at_ns is not None:
                self._record_presented_input(selected, presented_at_ns)
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
            elif isinstance(event, (GamepadUserInputEvent, GameWheelUserInputEvent)):
                recognized = True
            if not recognized:
                continue
            timestamp_us = int(event.get_timestamp())
            received_at_ns = time.monotonic_ns()
            self._input_received_at_ns.setdefault(timestamp_us, received_at_ns)
            self._input_received_at_ns.move_to_end(timestamp_us)
            _log_chunk_trace(
                "input_received",
                time_ns=received_at_ns,
                event_us=timestamp_us,
                **_input_event_trace_fields(event),
            )
        while len(self._input_received_at_ns) > _MAX_BUFFERED_INPUT_EVENTS:
            self._input_received_at_ns.popitem(last=False)

    def _record_presented_input(
        self,
        selected: TaxiHudFrame,
        presented_at_ns: int,
    ) -> None:
        if not self.profile_input_latency:
            return
        timestamp_us = selected.transition_timestamp_us
        if timestamp_us is None or timestamp_us in self._reported_input_timestamps_us:
            return
        received_at_ns = self._input_received_at_ns.pop(timestamp_us, None)
        if received_at_ns is None:
            return
        self._reported_input_timestamps_us.add(timestamp_us)
        self._latest_input_latency_ms = (presented_at_ns - received_at_ns) / 1_000_000.0
        _TRACE_LOGGER.info(
            "[crazy-robotaxi] input-to-model-frame latency: "
            "event_us=%d ui_to_frame_ms=%.1f generation=%d step=%d epoch=%d "
            "ar=%d frame=%d",
            timestamp_us,
            self._latest_input_latency_ms,
            selected.runtime_generation,
            selected.model_step_index,
            selected.rollout_epoch,
            selected.autoregressive_index,
            selected.frame_index,
        )

    def _record_presented_trace(
        self,
        selected: TaxiHudFrame,
        presented_at_ns: int,
    ) -> None:
        if not self.profile_input_latency or selected.model_step_index < 0:
            return
        latest = self._latest_committed_frame
        ar_lead: int | str = "unknown"
        step_lead: int | str = "unknown"
        simulation_lead_ms: float | str = "unknown"
        if latest is not None and latest.rollout_epoch == selected.rollout_epoch:
            ar_lead = latest.autoregressive_index - selected.autoregressive_index
            step_lead = latest.model_step_index - selected.model_step_index
            if (
                latest.simulation_timestamp_us is not None
                and selected.simulation_timestamp_us is not None
            ):
                simulation_lead_ms = (
                    latest.simulation_timestamp_us - selected.simulation_timestamp_us
                ) / 1000.0
        finalize_to_present_ms: float | str = "unknown"
        if selected.cache_finalize_returned_ns is not None:
            finalize_to_present_ms = (
                presented_at_ns - selected.cache_finalize_returned_ns
            ) / 1_000_000.0
        _log_chunk_trace(
            "app_frame_presented",
            time_ns=presented_at_ns,
            generation=selected.runtime_generation,
            step=selected.model_step_index,
            epoch=selected.rollout_epoch,
            ar=selected.autoregressive_index,
            frame=selected.frame_index,
            simulation_us=(
                "unknown"
                if selected.simulation_timestamp_us is None
                else selected.simulation_timestamp_us
            ),
            step_lead=step_lead,
            ar_lead=ar_lead,
            simulation_lead_ms=simulation_lead_ms,
            finalize_to_present_ms=finalize_to_present_ms,
        )

    def set_loading_status(self, status: str) -> None:
        """Update the startup phase from a model-loop message."""
        self._loading_status = status

    def activate_scene(self, calibration: CameraCalibration) -> None:
        """Install projection data after the model thread loads the chosen map."""
        self._clear_presented_game()
        self.calibration = calibration
        self._menu_stage = "loading"

    def initialize_selection(self) -> None:
        """Skip selection screens whose values were supplied explicitly by CLI."""
        selected_path = self.initial_map_path
        if selected_path is not None:
            resolved = selected_path.expanduser().resolve()
            self._selected_map_option = next(
                (option for option in self.map_options if option.path == resolved),
                None,
            )
            if self._selected_map_option is None:
                raise ValueError(f"CLI-selected map is unavailable: {resolved}")
        self._selected_game_mode = self.initial_game_mode
        if self._selected_game_mode is None:
            self._menu_stage = "mode"
            return
        self._continue_after_mode_selection()

    def _handle_escape(self) -> None:
        model_loop = self.model_loop
        if self._menu_stage == "game":
            self.reset()
            self._selected_map_option = None
            self._menu_stage = "map"
            if model_loop is not None:
                invoke_async(
                    model_loop,
                    lambda model_state: model_state.return_to_map_menu(),
                )
        elif self._menu_stage == "course":
            self._selected_map_option = None
            self._menu_stage = "map"
        elif self._menu_stage == "map":
            self._selected_map_option = None
            self._selected_game_mode = None
            self._menu_stage = "mode"
        elif self._menu_stage == "mode" and model_loop is not None:
            self._loading_status = "EXITING GAME"
            self._loading_started_at_s = time.monotonic()
            self._menu_stage = "loading"
            invoke_async(model_loop, lambda model_state: model_state.request_exit())

    def _select_mode(self, mode: GameMode) -> None:
        self._selected_game_mode = mode
        self._continue_after_mode_selection()

    def _continue_after_mode_selection(self) -> None:
        option = self._selected_map_option
        if option is None:
            self._menu_stage = "map"
            return
        self._continue_after_map_selection(option)

    def _select_map(self, option: GameMapOption) -> None:
        self._selected_map_option = option
        self._continue_after_map_selection(option)

    def _continue_after_map_selection(self, option: GameMapOption) -> None:
        mode = self._selected_game_mode
        if mode is None:
            self._menu_stage = "mode"
            return
        if mode == "taxi":
            self._start_game(option)
            return
        course_id = self.initial_race_course_id
        if course_id is not None and course_id in option.race_course_ids:
            self._start_game(option, race_course_id=course_id)
            return
        self._menu_stage = "course"

    def _start_game(
        self,
        option: GameMapOption,
        *,
        race_course_id: str | None = None,
    ) -> None:
        mode = self._selected_game_mode
        model_loop = self.model_loop
        if mode is None or model_loop is None:
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
        if self._menu_stage == "course":
            self._draw_course_selection(imgui)
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
            self._draw_race_status(imgui, snapshot)
            self._draw_navigation_arrow(
                imgui,
                snapshot.relative_bearing_rad,
                center_y=110.0,
                color_rgb=(1.0, 0.18, 0.08),
            )
            self._draw_bev_window(imgui, bev_frame, hud_frame)
        elif (
            isinstance(snapshot, TaxiGameSnapshot)
            and snapshot.session_state == "playing"
        ):
            self._draw_taxi_status(imgui, snapshot)
            self._draw_navigation_arrow(
                imgui,
                snapshot.relative_bearing_rad,
                center_y=110.0,
                color_rgb=(
                    (118.0 / 255.0, 185.0 / 255.0, 0.0)
                    if snapshot.phase == "seeking_pickup"
                    else (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
                ),
            )
            self._draw_bev_window(imgui, bev_frame, hud_frame)
        if snapshot.session_state in {"playing", "awaiting_start", "racing"}:
            self._draw_speed(imgui, hud_frame.speed_mps)
        self._draw_terminal(imgui, snapshot)
        self._draw_input_diagnostic(imgui)

    def _draw_taxi_status(self, imgui: Any, snapshot: TaxiGameSnapshot) -> None:
        """Draw the source game's one-line taxi status directly over the frame."""
        phase = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
        fare_time = (
            ""
            if snapshot.remaining_time_s is None
            else f"  {snapshot.remaining_time_s:04.1f}s"
        )
        score = f"SCORE {snapshot.score}"
        if snapshot.high_score is not None:
            score += f"  HIGH {snapshot.high_score}"
        label = (
            f"GAME {snapshot.global_remaining_time_s:04.1f}s  {phase}  "
            f"{snapshot.distance_m:.0f}m{fare_time}  {score}"
        )
        color = (
            (118.0 / 255.0, 185.0 / 255.0, 0.0)
            if snapshot.phase == "seeking_pickup"
            else (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
        )
        self._draw_status_strip(imgui, label, color_rgb=color, top=35.0)
        event = _event_label(snapshot)
        if event:
            self._draw_centered_text(
                imgui,
                event,
                top=160.0,
                font_size=44.0,
                color_rgb=color,
                shadow=True,
                font=self._gameplay_overlay_font(imgui),
            )

    def _draw_race_status(self, imgui: Any, snapshot: RaceGameSnapshot) -> None:
        """Draw the source game's one-line race status directly over the frame."""
        if snapshot.session_state == "awaiting_start":
            progress = "CROSS START LINE TO BEGIN"
        elif snapshot.lap_count == 0:
            progress = (
                f"CHECKPOINT {snapshot.checkpoint_index + 1}/"
                f"{snapshot.checkpoint_count}"
            )
        elif snapshot.target_kind == "start":
            progress = (
                f"RETURN TO START  LAP {snapshot.completed_laps + 1}/"
                f"{snapshot.lap_count}"
            )
        else:
            progress = (
                f"LAP {snapshot.completed_laps + 1}/{snapshot.lap_count}  "
                f"CHECKPOINT {snapshot.checkpoint_index + 1}/"
                f"{snapshot.checkpoint_count}"
            )
        best = (
            ""
            if snapshot.best_time_us is None
            else f"  BEST {format_race_time_us(snapshot.best_time_us)}"
        )
        label = (
            f"RACE {format_race_time_us(snapshot.elapsed_time_us)}  {progress}  "
            f"{snapshot.distance_m:.0f}m{best}"
        )
        self._draw_status_strip(
            imgui,
            label,
            color_rgb=(200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0),
            top=35.0,
            outline=True,
        )

    def _draw_status_strip(
        self,
        imgui: Any,
        label: str,
        *,
        color_rgb: tuple[float, float, float],
        top: float,
        outline: bool = False,
    ) -> None:
        """Draw centered arcade status text without creating an ImGui window."""
        draw_list = imgui.get_background_draw_list()
        font_size = 22.0
        text_width, text_height = _overlay_text_size(imgui, label, font_size)
        available_width = max(1.0, float(self.width) - 28.0)
        if text_width > available_width:
            font_size = max(1.0, font_size * available_width / text_width)
            text_width, text_height = _overlay_text_size(imgui, label, font_size)
        left = (float(self.width) - text_width) * 0.5
        panel_color = _imgui_color(
            imgui,
            (12.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0, 210.0 / 255.0),
        )
        draw_list.add_rect_filled(
            imgui.ImVec2(left - 14.0, top - 6.0),
            imgui.ImVec2(left + text_width + 14.0, top + text_height + 6.0),
            panel_color,
            9.0,
        )
        color = _imgui_color(imgui, (*color_rgb, 1.0))
        if outline:
            draw_list.add_rect(
                imgui.ImVec2(left - 14.0, top - 6.0),
                imgui.ImVec2(left + text_width + 14.0, top + text_height + 6.0),
                color,
                9.0,
                2.0,
            )
        _draw_overlay_text(
            imgui,
            draw_list,
            label,
            position=(left, top),
            font_size=font_size,
            color=color,
        )

    def _draw_centered_text(
        self,
        imgui: Any,
        label: str,
        *,
        top: float,
        font_size: float,
        color_rgb: tuple[float, float, float],
        shadow: bool = False,
        font: Any | None = None,
    ) -> None:
        """Draw centered overlay text at an explicit display size."""
        draw_list = imgui.get_background_draw_list()
        text_width, _ = _overlay_text_size(imgui, label, font_size, font=font)
        available_width = max(1.0, float(self.width) - 28.0)
        if text_width > available_width:
            font_size = max(1.0, font_size * available_width / text_width)
            text_width, _ = _overlay_text_size(imgui, label, font_size, font=font)
        left = (float(self.width) - text_width) * 0.5
        if shadow:
            _draw_overlay_text(
                imgui,
                draw_list,
                label,
                position=(left + 3.0, top + 3.0),
                font_size=font_size,
                color=_imgui_color(imgui, (0.0, 0.0, 0.0, 1.0)),
                font=font,
            )
        _draw_overlay_text(
            imgui,
            draw_list,
            label,
            position=(left, top),
            font_size=font_size,
            color=_imgui_color(imgui, (*color_rgb, 1.0)),
            font=font,
        )

    def _draw_speed(self, imgui: Any, speed_mps: float) -> None:
        """Draw the source HUD's green speed digit directly over the frame."""
        draw_list = imgui.get_background_draw_list()
        font = self._gameplay_overlay_font(imgui)
        font_size = max(28.0, min(76.0, float(self.height) * 0.12))
        speed = str(round(abs(float(speed_mps)) * _MPS_TO_MPH))
        speed_width, speed_height = _overlay_text_size(
            imgui, speed, font_size, font=font
        )
        left = 24.0
        top = max(10.0, float(self.height) - speed_height - 42.0)
        shadow = _imgui_color(imgui, (0.0, 0.0, 0.0, 0.9))
        green = _imgui_color(
            imgui,
            (118.0 / 255.0, 185.0 / 255.0, 0.0, 1.0),
        )
        _draw_overlay_text(
            imgui,
            draw_list,
            speed,
            position=(left + 3.0, top + 3.0),
            font_size=font_size,
            color=shadow,
            font=font,
        )
        _draw_overlay_text(
            imgui,
            draw_list,
            speed,
            position=(left, top),
            font_size=font_size,
            color=green,
            font=font,
        )
        unit_size = max(14.0, font_size * 0.28)
        unit_width, _ = _overlay_text_size(imgui, "mph", unit_size, font=font)
        _draw_overlay_text(
            imgui,
            draw_list,
            "mph",
            position=(
                left + (speed_width - unit_width) * 0.5,
                top + speed_height + 2.0,
            ),
            font_size=unit_size,
            color=_imgui_color(imgui, (0.86, 0.86, 0.9, 1.0)),
            font=font,
        )

    def _gameplay_overlay_font(self, imgui: Any) -> Any:
        """Load and cache imgui-bundle's Droid Sans face."""
        if self._gameplay_font is None:
            resource = (
                files("imgui_bundle")
                .joinpath("assets")
                .joinpath("fonts")
                .joinpath("DroidSans.ttf")
            )
            with as_file(resource) as path:
                self._gameplay_font = imgui.get_io().fonts.add_font_from_file_ttf(
                    str(path), 13.0
                )
        return self._gameplay_font

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
        self._input_received_at_ns.clear()
        self._reported_input_timestamps_us.clear()
        self._latest_input_latency_ms = None
        self._latest_committed_frame = None
        self._name_input = ""

    def _clear_presented_game(self) -> None:
        """Discard frame-aligned HUD and BEV resources from the previous game."""
        self._frames.clear()
        self._current = None
        self._waypoint_source = None
        self._waypoint_projections = ()
        self._bev_source_key = None
        self._bev_panel = None
        self._bev_alpha = None
        self._bev_composite_source_key = None
        self._bev_composite = None
        self._bev_rect = None
        self._presented_frame_times_s.clear()
        self._video_fps = 0.0

    def _draw_mode_selection(self, imgui: Any) -> None:
        window_width = max(1.0, min(500.0, float(self.width) - 28.0))
        window_height = max(1.0, min(390.0, float(self.height) - 28.0))
        scale = min(1.0, window_width / 500.0, window_height / 390.0)
        _draw_arcade_backdrop(imgui, self.width, self.height)
        _prepare_window(
            imgui,
            position=(
                max(14.0, (self.width - window_width) / 2.0),
                max(14.0, (self.height - window_height) / 2.0),
            ),
            size=(window_width, window_height),
            alpha=0.97,
        )
        style_var_count, style_color_count = _push_arcade_card_style(
            imgui, _TAXI_ACCENT_RGB
        )
        visible = _begin_window(
            imgui,
            "Crazy Robotaxi — Select Game Mode",
            extra_flags=("no_title_bar",),
        )
        try:
            if not visible:
                return
            _centered_imgui_text(
                imgui,
                "CRAZY ROBOTAXI",
                font=self._gameplay_overlay_font(imgui),
                font_size=max(24.0, 40.0 * scale),
                color=(*_TAXI_ACCENT_RGB, 1.0),
            )
            _centered_imgui_text(
                imgui,
                "CHOOSE YOUR RIDE",
                font_size=max(13.0, 16.0 * scale),
                color=(0.62, 0.62, 0.68, 1.0),
            )
            imgui.separator()
            button_width = _point_xy(imgui.get_content_region_avail())[0]
            button_height = max(38.0, 54.0 * scale)
            if imgui.button("TAXI", imgui.ImVec2(button_width, button_height)):
                self._select_mode("taxi")
            _centered_imgui_text(
                imgui,
                "PICK UP PASSENGERS. DROP THEM OFF TO SCORE POINTS.",
                font_size=max(12.0, 13.0 * scale),
                color=(0.72, 0.72, 0.76, 1.0),
            )
            for color, alpha in (
                (imgui.Col_.button, 0.78),
                (imgui.Col_.button_hovered, 1.0),
                (imgui.Col_.button_active, 0.62),
            ):
                imgui.push_style_color(color, imgui.ImVec4(*_RACE_ACCENT_RGB, alpha))
            try:
                if imgui.button("RACE", imgui.ImVec2(button_width, button_height)):
                    self._select_mode("race")
            finally:
                imgui.pop_style_color(3)
            _centered_imgui_text(
                imgui,
                "CHASE THE FASTEST TRACK TIME.",
                font_size=max(12.0, 13.0 * scale),
                color=(0.72, 0.72, 0.76, 1.0),
            )
            imgui.separator()
            _centered_imgui_text(
                imgui,
                "ESC  EXIT",
                font_size=max(12.0, 13.0 * scale),
                color=(0.58, 0.58, 0.64, 1.0),
            )
        finally:
            imgui.end()
            imgui.pop_style_color(style_color_count)
            imgui.pop_style_var(style_var_count)

    def _draw_map_selection(self, imgui: Any) -> None:
        mode = self._selected_game_mode
        if mode is None:
            self._menu_stage = "mode"
            return
        window_width = max(1.0, min(620.0, float(self.width) - 28.0))
        window_height = max(1.0, min(560.0, float(self.height) - 28.0))
        scale = min(1.0, window_width / 620.0, window_height / 560.0)
        accent_rgb = _RACE_ACCENT_RGB if mode == "race" else _TAXI_ACCENT_RGB
        _draw_arcade_backdrop(imgui, self.width, self.height)
        _prepare_window(
            imgui,
            position=(
                max(14.0, (self.width - window_width) / 2.0),
                max(14.0, (self.height - window_height) / 2.0),
            ),
            size=(window_width, window_height),
            alpha=0.97,
        )
        style_var_count, style_color_count = _push_arcade_card_style(imgui, accent_rgb)
        visible = _begin_window(
            imgui,
            "Crazy Robotaxi — Select Map",
            extra_flags=("no_title_bar",),
        )
        try:
            if not visible:
                return
            _centered_imgui_text(
                imgui,
                "SELECT MAP",
                font=self._gameplay_overlay_font(imgui),
                font_size=max(24.0, 38.0 * scale),
                color=(*accent_rgb, 1.0),
            )
            _centered_imgui_text(
                imgui,
                "RACE MODE" if mode == "race" else "TAXI MODE",
                font_size=max(13.0, 15.0 * scale),
                color=(0.62, 0.62, 0.68, 1.0),
            )
            imgui.separator()
            button_height = max(36.0, 48.0 * scale)
            list_height = max(
                60.0, _point_xy(imgui.get_content_region_avail())[1] - 92.0
            )
            list_visible = imgui.begin_child(
                "##map-options", imgui.ImVec2(0.0, list_height)
            )
            try:
                if list_visible:
                    button_width = _point_xy(imgui.get_content_region_avail())[0]
                    available = False
                    for index, option in enumerate(self.map_options):
                        if mode == "race" and not option.race_course_ids:
                            continue
                        available = True
                        if imgui.button(
                            f"{option.name}##map-{index}",
                            imgui.ImVec2(button_width, button_height),
                        ):
                            self._select_map(option)
                    if not available:
                        _centered_imgui_text(
                            imgui,
                            "NO COMPATIBLE MAPS FOUND",
                            font_size=max(13.0, 15.0 * scale),
                            color=(0.62, 0.62, 0.68, 1.0),
                        )
            finally:
                imgui.end_child()
            imgui.separator()
            button_width = _point_xy(imgui.get_content_region_avail())[0]
            if imgui.button(
                "BACK", imgui.ImVec2(button_width, max(34.0, 42.0 * scale))
            ):
                self._selected_game_mode = None
                self._menu_stage = "mode"
                return
            _centered_imgui_text(
                imgui,
                "ESC  BACK",
                font_size=max(12.0, 13.0 * scale),
                color=(0.58, 0.58, 0.64, 1.0),
            )
        finally:
            imgui.end()
            imgui.pop_style_color(style_color_count)
            imgui.pop_style_var(style_var_count)

    def _draw_course_selection(self, imgui: Any) -> None:
        option = self._selected_map_option
        if self._selected_game_mode != "race":
            self._menu_stage = "map"
            return
        if option is None:
            self._menu_stage = "map"
            return
        window_width = max(1.0, min(620.0, float(self.width) - 28.0))
        window_height = max(1.0, min(420.0, float(self.height) - 28.0))
        scale = min(1.0, window_width / 620.0, window_height / 420.0)
        _draw_arcade_backdrop(imgui, self.width, self.height)
        _prepare_window(
            imgui,
            position=(
                max(14.0, (self.width - window_width) / 2.0),
                max(14.0, (self.height - window_height) / 2.0),
            ),
            size=(window_width, window_height),
            alpha=0.97,
        )
        style_var_count, style_color_count = _push_arcade_card_style(
            imgui, _RACE_ACCENT_RGB
        )
        visible = _begin_window(
            imgui,
            "Crazy Robotaxi — Select Race Course",
            extra_flags=("no_title_bar",),
        )
        try:
            if not visible:
                return
            _centered_imgui_text(
                imgui,
                "SELECT RACE COURSE",
                font=self._gameplay_overlay_font(imgui),
                font_size=max(22.0, 36.0 * scale),
                color=(*_RACE_ACCENT_RGB, 1.0),
            )
            _centered_imgui_text(
                imgui,
                option.name.upper(),
                font_size=max(13.0, 15.0 * scale),
                color=(0.62, 0.62, 0.68, 1.0),
            )
            imgui.separator()
            button_height = max(36.0, 48.0 * scale)
            list_height = max(
                60.0, _point_xy(imgui.get_content_region_avail())[1] - 92.0
            )
            list_visible = imgui.begin_child(
                "##course-options", imgui.ImVec2(0.0, list_height)
            )
            try:
                if list_visible:
                    button_width = _point_xy(imgui.get_content_region_avail())[0]
                    for course_index, course_id in enumerate(option.race_course_ids):
                        label = course_id.replace("-", " ").replace("_", " ").upper()
                        if imgui.button(
                            f"{label}##course-{course_index}",
                            imgui.ImVec2(button_width, button_height),
                        ):
                            self._start_game(option, race_course_id=course_id)
            finally:
                imgui.end_child()
            imgui.separator()
            button_width = _point_xy(imgui.get_content_region_avail())[0]
            if imgui.button(
                "BACK", imgui.ImVec2(button_width, max(34.0, 42.0 * scale))
            ):
                self._selected_map_option = None
                self._menu_stage = "map"
                return
            _centered_imgui_text(
                imgui,
                "ESC  BACK",
                font_size=max(12.0, 13.0 * scale),
                color=(0.58, 0.58, 0.64, 1.0),
            )
        finally:
            imgui.end()
            imgui.pop_style_color(style_color_count)
            imgui.pop_style_var(style_var_count)

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
        center_y: float,
        color_rgb: tuple[float, float, float],
    ) -> None:
        """Draw the always-visible target-bearing arrow from the original HUD."""
        draw_list = imgui.get_background_draw_list()
        center_x = float(self.width) * 0.5
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
        """Return the cached float32 video and BEV back buffer."""
        if not video.is_floating_point():
            raise ValueError("Video presentation frames must be floating point")
        rect = self._bev_rect
        frame_key = (
            None
            if frame is None
            else (
                int(frame.data_ptr()),
                tuple(int(value) for value in frame.shape),
                frame.dtype,
                frame.device,
            )
        )
        composite_source_key = (
            id(self._current),
            int(video.data_ptr()),
            tuple(int(value) for value in video.shape),
            video.dtype,
            video.device,
            frame_key,
            rect,
        )
        if (
            composite_source_key == self._bev_composite_source_key
            and self._bev_composite is not None
        ):
            return self._bev_composite

        # The shared ImGui overlay is float32. Converting once here avoids a
        # full-frame overlay cast and extra BF16 blend kernels downstream.
        output = video.to(dtype=torch.float32, copy=True)
        if frame is None or rect is None:
            self._bev_composite_source_key = composite_source_key
            self._bev_composite = output
            return output
        if frame.ndim != 3 or frame.shape[0] != 4:
            raise ValueError("BEV presentation frames must use [4,H,W] RGBA")
        if frame.dtype != torch.uint8 and not frame.is_floating_point():
            raise ValueError("BEV presentation frames must be uint8 or floating point")
        if frame.device != video.device:
            raise ValueError("BEV and video presentation frames must share a device")

        top, left, image_height, image_width = rect
        bottom = min(int(video.shape[-2]), top + image_height)
        right = min(int(video.shape[-1]), left + image_width)
        if bottom <= top or right <= left:
            self._bev_composite_source_key = composite_source_key
            self._bev_composite = output
            return output

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
            source = frame[:3].detach().to(dtype=torch.float32)
            panel = source.div(127.5).sub(1.0) if frame.dtype == torch.uint8 else source
            alpha_source = frame[3:4].detach()
            if tuple(panel.shape[-2:]) != (image_height, image_width):
                panel = functional.interpolate(
                    panel.unsqueeze(0),
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                alpha_source = functional.interpolate(
                    alpha_source.unsqueeze(0),
                    size=(image_height, image_width),
                    mode="nearest",
                )[0]
            alpha = alpha_source.ne(0)
            self._bev_source_key = source_key
            self._bev_panel = panel
            self._bev_alpha = alpha

        target = output[:, top:bottom, left:right]
        source_panel = panel[:, : bottom - top, : right - left]
        source_alpha = alpha[:, : bottom - top, : right - left]
        torch.where(source_alpha, source_panel, target, out=target)
        _composite_bev_ego_car(target)
        self._bev_composite_source_key = composite_source_key
        self._bev_composite = output
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
        latency = self._latest_input_latency_ms
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
            lines=(input_state, latency_label),
        )

    def _draw_terminal(
        self, imgui: Any, snapshot: TaxiGameSnapshot | RaceGameSnapshot
    ) -> None:
        awaiting_name = snapshot.session_state == "awaiting_name"
        leaderboard = snapshot.session_state == "leaderboard"
        if not (awaiting_name or leaderboard):
            return
        race = isinstance(snapshot, RaceGameSnapshot)
        accent_rgb = _RACE_ACCENT_RGB if race else _TAXI_ACCENT_RGB
        margin = 16.0
        card_width = max(1.0, min(620.0, float(self.width) - 2.0 * margin))
        card_height = max(1.0, min(540.0, float(self.height) - 2.0 * margin))
        card_left = (float(self.width) - card_width) * 0.5
        card_top = (float(self.height) - card_height) * 0.5
        scale = min(1.0, card_width / 620.0, card_height / 540.0)

        _draw_arcade_backdrop(imgui, self.width, self.height)
        _prepare_window(
            imgui,
            position=(card_left, card_top),
            size=(card_width, card_height),
            alpha=0.97,
        )
        style_var_count, style_color_count = _push_arcade_card_style(imgui, accent_rgb)
        visible = _begin_window(imgui, "Game Over", extra_flags=("no_title_bar",))
        try:
            if not visible:
                return
            imgui.dummy(imgui.ImVec2(0.0, max(2.0, 8.0 * scale)))
            headline = (
                ("NEW BEST TIME" if race else "NEW HIGH SCORE")
                if awaiting_name
                else ("RACE COMPLETE" if race else "GAME OVER")
            )
            _centered_imgui_text(
                imgui,
                headline,
                font=self._gameplay_overlay_font(imgui),
                font_size=max(22.0, 38.0 * scale),
                color=(*accent_rgb, 1.0),
            )
            _centered_imgui_text(
                imgui,
                "FINAL TIME" if race else "FINAL SCORE",
                font_size=max(13.0, 15.0 * scale),
                color=(0.62, 0.62, 0.68, 1.0),
            )
            if race:
                result = format_race_time_us(snapshot.final_time_us or 0)
            else:
                result = f"{snapshot.score:06d}"
            _centered_imgui_text(
                imgui,
                result,
                font=self._gameplay_overlay_font(imgui),
                font_size=max(28.0, 50.0 * scale),
            )
            if snapshot.high_score_rank is not None:
                _centered_imgui_text(
                    imgui,
                    f"RANK #{snapshot.high_score_rank}",
                    font_size=max(13.0, 17.0 * scale),
                    color=(*accent_rgb, 1.0),
                )
            imgui.separator()
            if awaiting_name:
                self._draw_terminal_name_entry(imgui, race, accent_rgb, scale)
            else:
                self._draw_terminal_leaderboard(imgui, snapshot, race, accent_rgb)
            imgui.separator()
            action_width = _point_xy(imgui.get_content_region_avail())[0]
            if imgui.button(
                "PLAY AGAIN",
                imgui.ImVec2(action_width, max(34.0, 44.0 * scale)),
            ):
                self._request_restart()
            _centered_imgui_text(
                imgui,
                "R  RESTART   ·   ESC  MAP",
                font_size=max(12.0, 13.0 * scale),
                color=(0.58, 0.58, 0.64, 1.0),
            )
        finally:
            imgui.end()
            imgui.pop_style_color(style_color_count)
            imgui.pop_style_var(style_var_count)

    def _draw_terminal_name_entry(
        self,
        imgui: Any,
        race: bool,
        accent_rgb: tuple[float, float, float],
        scale: float,
    ) -> None:
        """Draw terminal name entry and submission feedback."""
        _centered_imgui_text(
            imgui,
            "ENTER DRIVER NAME",
            font_size=max(13.0, 16.0 * scale),
        )
        imgui.set_next_item_width(-1.0)
        disabled = self._submission_pending
        if disabled:
            imgui.begin_disabled()
        try:
            submitted, self._name_input = imgui.input_text(
                "##driver-name",
                self._name_input,
                flags=imgui.InputTextFlags_.enter_returns_true,
            )
            submit_width = _point_xy(imgui.get_content_region_avail())[0]
            clicked = imgui.button(
                "SAVE TIME" if race else "SAVE SCORE",
                imgui.ImVec2(submit_width, max(32.0, 40.0 * scale)),
            )
        finally:
            if disabled:
                imgui.end_disabled()
        if submitted or clicked:
            self._submit_name(self._name_input)
        if self._validation_message:
            color = (
                (*accent_rgb, 1.0)
                if self._submission_pending
                else (1.0, 0.38, 0.32, 1.0)
            )
            _centered_imgui_text(
                imgui,
                self._validation_message,
                font_size=max(12.0, 13.0 * scale),
                color=color,
            )

    def _draw_terminal_leaderboard(
        self,
        imgui: Any,
        snapshot: TaxiGameSnapshot | RaceGameSnapshot,
        race: bool,
        accent_rgb: tuple[float, float, float],
    ) -> None:
        """Draw the ranked terminal results table."""
        _centered_imgui_text(imgui, "LEADERBOARD", font_size=16.0)
        entries = snapshot.leaderboard
        if not entries:
            _centered_imgui_text(
                imgui,
                "NO SCORES YET",
                font_size=14.0,
                color=(0.62, 0.62, 0.68, 1.0),
            )
            return
        available_height = _point_xy(imgui.get_content_region_avail())[1]
        table_height = max(90.0, min(250.0, available_height - 92.0))
        table_flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.scroll_y
        )
        if not imgui.begin_table(
            "##leaderboard",
            3,
            flags=table_flags,
            outer_size=imgui.ImVec2(0.0, table_height),
        ):
            return
        try:
            imgui.table_setup_column("RANK", imgui.TableColumnFlags_.width_fixed, 64.0)
            imgui.table_setup_column(
                "DRIVER", imgui.TableColumnFlags_.width_stretch, 1.0
            )
            imgui.table_setup_column(
                "TIME" if race else "SCORE",
                imgui.TableColumnFlags_.width_fixed,
                128.0,
            )
            imgui.table_headers_row()
            for rank, entry in enumerate(entries, start=1):
                imgui.table_next_row(min_row_height=26.0)
                if rank == snapshot.high_score_rank:
                    imgui.table_set_bg_color(
                        imgui.TableBgTarget_.row_bg1,
                        _imgui_color(imgui, (*accent_rgb, 0.24)),
                    )
                if race:
                    assert isinstance(entry, RaceTimeEntry)
                    result = format_race_time_us(entry.elapsed_time_us)
                else:
                    assert isinstance(entry, HighScoreEntry)
                    result = f"{entry.score:>7}"
                values = (
                    f"#{rank}",
                    entry.name,
                    result,
                )
                for column, value in enumerate(values):
                    imgui.table_set_column_index(column)
                    imgui.text(value)
        finally:
            imgui.end_table()

    def _request_restart(self) -> None:
        """Queue a game restart on the model thread."""
        if self.model_loop is not None:
            invoke_async(self.model_loop, lambda state: state.restart_game())

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


def _composite_bev_ego_car(panel: Tensor) -> None:
    """Draw a small heading-up taxi glyph directly on its tensor device."""
    height, width = (int(value) for value in panel.shape[-2:])
    extent = min(height, width)
    if extent < 16:
        return

    car_height = max(8, round(extent * 0.12))
    car_height = min(car_height + (car_height + 1) % 2, height - 2)
    car_width = max(5, round(car_height * 0.55))
    car_width = min(car_width + (car_width + 1) % 2, width - 2)
    top = (height - car_height) // 2
    left = (width - car_width) // 2
    bottom = top + car_height
    right = left + car_width

    white, yellow, glass = panel.new_tensor(
        (
            (1.0, 1.0, 1.0),
            (1.0, 0.6, -1.0),
            (-0.8, -0.2, 0.15),
        )
    ).view(3, 3, 1, 1)
    panel[:, top + 1 : bottom - 1, left:right] = white
    panel[:, top:bottom, left + 1 : right - 1] = white
    panel[:, top + 1 : bottom - 1, left + 1 : right - 1] = yellow

    window_left = left + max(2, car_width // 3)
    window_right = right - max(2, car_width // 3)
    if window_right <= window_left:
        return
    window_height = max(1, car_height // 5)
    window_offset = max(2, car_height // 5)
    panel[
        :,
        top + window_offset : top + window_offset + window_height,
        window_left:window_right,
    ] = glass
    panel[
        :,
        bottom - window_offset - window_height : bottom - window_offset,
        window_left:window_right,
    ] = glass


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
        return self.state.composite_bev(video, bev_frame)

    def reset(self) -> None:
        """Reset UI-owned state and retained renderer resources."""
        self.state.reset()
        super().reset()


def _log_chunk_trace(phase: str, *, time_ns: int, **fields: object) -> None:
    """Emit one grep-friendly chunk lifecycle event."""
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    _TRACE_LOGGER.info(
        "%s phase=%s time_ns=%d %s",
        _TRACE_PREFIX,
        phase,
        time_ns,
        details,
    )


def _input_event_trace_fields(event: object) -> dict[str, object]:
    """Return non-text driving fields for one diagnostic input event."""
    if isinstance(event, KeyboardUserInputEvent):
        return {
            "source": "keyboard",
            "key": _normalize_profile_key(str(event.key)),
            "state": event.state.value,
        }
    if isinstance(event, FocusUserInputEvent):
        return {"source": "focus", "focused": event.focused}
    if isinstance(event, GamepadUserInputEvent):
        return {"source": "gamepad", "action": event.action}
    if isinstance(event, GameWheelUserInputEvent):
        return {"source": "wheel", "action": event.action}
    return {"source": type(event).__name__}


def build_hud_frames(
    video_tchw: Tensor,
    snapshots: Sequence[object],
    rig_poses_world: npt.NDArray[np.float32],
    *,
    speeds_mps: Sequence[float] | None = None,
    transition_timestamps_us: Sequence[int | None] | None = None,
    runtime_generation: int = 0,
    model_step_index: int = -1,
    rollout_epoch: int = 0,
    autoregressive_index: int = -1,
    simulation_timestamps_us: Sequence[int | None] | None = None,
    cache_finalize_returned_ns: int | None = None,
) -> tuple[TaxiHudFrame, ...]:
    """Build immutable UI messages aligned with generated tensor frames."""
    frame_count = int(video_tchw.shape[0])
    if len(snapshots) != frame_count:
        raise ValueError("Video and game snapshots must align")
    poses = np.asarray(rig_poses_world, dtype=np.float32)
    if poses.shape != (frame_count, 4, 4):
        raise ValueError("Video and rig poses must align")
    if speeds_mps is None:
        speeds_mps = (0.0,) * frame_count
    if len(speeds_mps) != frame_count:
        raise ValueError("Vehicle speeds and video frames must align")
    if transition_timestamps_us is None:
        transition_timestamps_us = (None,) * frame_count
    if len(transition_timestamps_us) != frame_count:
        raise ValueError("Input transitions and video frames must align")
    if simulation_timestamps_us is None:
        simulation_timestamps_us = (None,) * frame_count
    if len(simulation_timestamps_us) != frame_count:
        raise ValueError("Simulation timestamps and video frames must align")
    frames = []
    for index, (snapshot, simulation_timestamp_us) in enumerate(
        zip(snapshots, simulation_timestamps_us, strict=True)
    ):
        if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
            raise TypeError("Taxi HUD received an unknown game snapshot")
        pose = poses[index].copy()
        pose.setflags(write=False)
        frames.append(
            TaxiHudFrame(
                frame_key=int(video_tchw[index].data_ptr()),
                snapshot=snapshot,
                rig_pose_world=pose,
                speed_mps=float(speeds_mps[index]),
                transition_timestamp_us=transition_timestamps_us[index],
                runtime_generation=runtime_generation,
                model_step_index=model_step_index,
                rollout_epoch=rollout_epoch,
                autoregressive_index=autoregressive_index,
                frame_index=index,
                simulation_timestamp_us=simulation_timestamp_us,
                cache_finalize_returned_ns=cache_finalize_returned_ns,
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


def _draw_arcade_backdrop(imgui: Any, width: int, height: int) -> None:
    draw_list = imgui.get_background_draw_list()
    draw_list.add_rect_filled(
        imgui.ImVec2(0.0, 0.0),
        imgui.ImVec2(float(width), float(height)),
        _imgui_color(imgui, (0.0, 0.0, 0.0, 0.58)),
    )


def _push_arcade_card_style(
    imgui: Any,
    accent_rgb: tuple[float, float, float],
) -> tuple[int, int]:
    style_vars = (
        (imgui.StyleVar_.window_rounding, 16.0),
        (imgui.StyleVar_.window_border_size, 2.0),
        (imgui.StyleVar_.window_padding, imgui.ImVec2(28.0, 24.0)),
        (imgui.StyleVar_.item_spacing, imgui.ImVec2(10.0, 10.0)),
        (imgui.StyleVar_.frame_rounding, 7.0),
        (imgui.StyleVar_.frame_padding, imgui.ImVec2(10.0, 8.0)),
    )
    style_colors = (
        (imgui.Col_.window_bg, (0.047, 0.047, 0.071, 0.98)),
        (imgui.Col_.border, (*accent_rgb, 0.95)),
        (imgui.Col_.text, (0.94, 0.94, 0.97, 1.0)),
        (imgui.Col_.text_disabled, (0.58, 0.58, 0.64, 1.0)),
        (imgui.Col_.frame_bg, (0.09, 0.09, 0.13, 1.0)),
        (imgui.Col_.frame_bg_hovered, (0.13, 0.13, 0.18, 1.0)),
        (imgui.Col_.frame_bg_active, (0.16, 0.16, 0.22, 1.0)),
        (imgui.Col_.button, (*accent_rgb, 0.78)),
        (imgui.Col_.button_hovered, (*accent_rgb, 1.0)),
        (imgui.Col_.button_active, (*accent_rgb, 0.62)),
    )
    for style_var, value in style_vars:
        imgui.push_style_var(style_var, value)
    for color, value in style_colors:
        imgui.push_style_color(color, imgui.ImVec4(*value))
    return len(style_vars), len(style_colors)


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


def _overlay_text_size(
    imgui: Any,
    text: str,
    font_size: float,
    *,
    font: Any | None = None,
) -> tuple[float, float]:
    """Measure text after applying an explicit ImGui display size."""
    if font is not None:
        imgui.push_font(font, float(font_size))
        try:
            return _point_xy(imgui.calc_text_size(text))
        finally:
            imgui.pop_font()
    width, height = _point_xy(imgui.calc_text_size(text))
    scale = float(font_size) / max(1.0, float(imgui.get_font_size()))
    return width * scale, height * scale


def _centered_imgui_text(
    imgui: Any,
    text: str,
    *,
    font_size: float,
    font: Any | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw one centered ImGui text item."""
    cursor_x = float(imgui.get_cursor_pos_x())
    available_width = _point_xy(imgui.get_content_region_avail())[0]
    imgui.push_font(font, float(font_size))
    if color is not None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
    try:
        text_width = _point_xy(imgui.calc_text_size(text))[0]
        imgui.set_cursor_pos_x(
            cursor_x + max(0.0, (available_width - text_width) * 0.5)
        )
        imgui.text(text)
    finally:
        if color is not None:
            imgui.pop_style_color()
        imgui.pop_font()


def _draw_overlay_text(
    imgui: Any,
    draw_list: Any,
    text: str,
    *,
    position: tuple[float, float],
    font_size: float,
    color: int,
    font: Any | None = None,
) -> None:
    """Draw sized text directly into the shared background overlay."""
    draw_list.add_text(
        imgui.get_font() if font is None else font,
        float(font_size),
        imgui.ImVec2(*position),
        color,
        text,
    )


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


__all__ = [
    "CrazyRobotaxiImGuiUILoop",
    "TaxiHudFrame",
    "TaxiHudState",
    "bev_display_extent",
    "build_hud_frames",
]
