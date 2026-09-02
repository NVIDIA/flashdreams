# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's V2 Dear ImGui UI loop."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import torch
from crazy_robotaxi.controls import (
    ControlsConfig,
    InputBinding,
    load_controls_documents,
)
from crazy_robotaxi.game_selection import (
    GameMapOption,
    GameRaceCourseOption,
    GameSelection,
)
from crazy_robotaxi.high_scores import HighScoreEntry, RaceTimeEntry
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
)
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditAction, LiveEditHudStatus
from crazy_robotaxi.race import RaceGameSnapshot, RaceSessionState
from crazy_robotaxi.rules import (
    TaxiGameSnapshot,
    TaxiSessionState,
    project_taxi_markers_to_camera,
)
from crazy_robotaxi.settings import SettingsDocument
from crazy_robotaxi.ui import (
    _BEV_WAYPOINT_ALPHA,
    _NATIVE_DIT_NOTICE_RGBA,
    _RESTART_NOTICE_RGBA,
    _SAVED_NOTICE_RGBA,
    CrazyRobotaxiImGuiUILoop,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import draw_waypoints, project_waypoints
from omnidreams.config import OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
from omnidreams_game_engine.types import CameraCalibration

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _SettingsPipeline:
    name: str = "test-preset"


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        clipgt_name="front",
        logical_name="front",
        width=160,
        height=96,
        cx=80.0,
        cy=48.0,
        polynomial=np.asarray([0.0, 100.0, 0.0, 0.0], dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )


def _snapshot(*, session_state: TaxiSessionState = "playing") -> TaxiGameSnapshot:
    return TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(25.0, 0.0, 0.0),
        distance_m=25.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=1200,
        high_score=9000,
        global_remaining_time_s=42.5,
        session_state=session_state,
    )


def _race_snapshot(*, session_state: RaceSessionState = "racing") -> RaceGameSnapshot:
    return RaceGameSnapshot(
        map_id="test-city",
        course_id="downtown-sprint",
        session_state=session_state,
        target_kind="start",
        target_element_id="start",
        target_xyz_m=(25.0, 0.0, 0.0),
        gate_start_xyz_m=(25.0, -5.0, 0.0),
        gate_end_xyz_m=(25.0, 5.0, 0.0),
        checkpoint_markers=True,
        distance_m=25.0,
        relative_bearing_rad=0.0,
        checkpoint_index=0,
        checkpoint_count=3,
        completed_laps=1,
        lap_count=1,
        elapsed_time_us=42_345_000,
        best_time_us=41_000_000,
        final_time_us=42_345_000,
    )


class _FakeDrawList:
    def __init__(self) -> None:
        self.commands: list[tuple[str, tuple[Any, ...]]] = []

    def add_line(self, *args: Any) -> None:
        self.commands.append(("line", args))

    def add_circle(self, *args: Any) -> None:
        self.commands.append(("circle", args))

    def add_circle_filled(self, *args: Any) -> None:
        self.commands.append(("circle_filled", args))

    def add_triangle_filled(self, *args: Any) -> None:
        self.commands.append(("triangle_filled", args))

    def add_rect(
        self,
        p_min: Any,
        p_max: Any,
        color: int,
        rounding: float = 0.0,
        thickness: float = 1.0,
        flags: int = 0,
    ) -> None:
        self.commands.append(
            ("rect", (p_min, p_max, color, rounding, thickness, flags))
        )

    def add_rect_filled(self, *args: Any) -> None:
        self.commands.append(("rect_filled", args))

    def add_text(self, *args: Any) -> None:
        self.commands.append(("text", args))


class _FakeFontAtlas:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, float, object]] = []

    def add_font_from_file_ttf(self, path: str, size: float) -> object:
        font = object()
        self.loaded.append((path, size, font))
        return font


class _FakeImGui:
    Cond_ = SimpleNamespace(always=1)
    ChildFlags_ = SimpleNamespace(auto_resize_y=1, always_auto_resize=2)
    WindowFlags_ = SimpleNamespace(
        no_move=1,
        no_resize=2,
        no_collapse=4,
        no_saved_settings=8,
        no_title_bar=16,
        no_background=32,
        always_auto_resize=64,
        no_scrollbar=128,
        no_scroll_with_mouse=256,
        horizontal_scrollbar=512,
    )
    InputTextFlags_ = SimpleNamespace(enter_returns_true=1)
    StyleVar_ = SimpleNamespace(
        window_rounding=1,
        window_border_size=2,
        window_padding=3,
        item_spacing=4,
        frame_rounding=5,
        frame_padding=6,
    )
    Col_ = SimpleNamespace(
        text=1,
        text_disabled=2,
        window_bg=3,
        border=4,
        frame_bg=5,
        frame_bg_hovered=6,
        frame_bg_active=7,
        button=8,
        button_hovered=9,
        button_active=10,
        check_mark=11,
    )
    TableFlags_ = SimpleNamespace(
        row_bg=1,
        borders_inner_h=2,
        no_saved_settings=4,
        sizing_stretch_prop=8,
        scroll_y=16,
        sizing_fixed_same=32,
        sizing_stretch_same=64,
    )
    TableColumnFlags_ = SimpleNamespace(width_fixed=1, width_stretch=2)
    TableBgTarget_ = SimpleNamespace(row_bg1=1)

    def __init__(self) -> None:
        self.windows: dict[str, list[str]] = {}
        self.text_fonts: list[tuple[str, object, float]] = []
        self.text_positions: list[tuple[str, float]] = []
        self.dummies: list[tuple[float, float]] = []
        self.current_window: str | None = None
        self.next_window_position = (0.0, 0.0)
        self.next_window_size = (640.0, 360.0)
        self.cursor_x = 8.0
        self.cursor_y = 8.0
        self.input_value = ""
        self.input_values: dict[str, str] = {}
        self.multiline_inputs: list[tuple[str, str, tuple[float, float], int]] = []
        self.multiline_input_positions: dict[str, float] = {}
        self.checkbox_values: dict[str, bool] = {}
        self.combo_indices: dict[str, int] = {}
        self.submit_input = False
        self.click_submit = False
        self.clicked_buttons: set[str] = set()
        self.buttons: list[str] = []
        self.button_sizes: list[tuple[str, tuple[float, float] | None]] = []
        self.button_positions: list[tuple[str, float]] = []
        self.images: list[tuple[str, np.ndarray, tuple[float, float]]] = []
        self.disabled_depth = 0
        self.disabled_buttons: list[str] = []
        self.background_draw_list = _FakeDrawList()
        self.window_flags: dict[str, int] = {}
        self.child_sizes: dict[str, tuple[float, float]] = {}
        self.child_window_flags: dict[str, int] = {}
        self._child_size_stack: list[tuple[float, float]] = []
        self.current_child_size: tuple[float, float] | None = None
        self.last_item_rect_size = (0.0, 0.0)
        self.tables: dict[str, list[list[str]]] = {}
        self.table_columns: dict[str, list[str]] = {}
        self.table_column_widths: dict[str, list[float]] = {}
        self.table_column_counts: dict[str, int] = {}
        self.table_flags: dict[str, int] = {}
        self.table_outer_sizes: dict[str, tuple[float, float]] = {}
        self.highlighted_rows: list[int] = []
        self.current_table: str | None = None
        self.current_table_column = 0
        self.default_font = object()
        self.current_font = self.default_font
        self.current_font_size = 14.0
        self.font_stack: list[tuple[object, float]] = []
        self.pushed_style_vars: list[tuple[int, object]] = []
        self.pushed_style_colors: list[tuple[int, object]] = []
        self.fonts = _FakeFontAtlas()
        self.io = SimpleNamespace(fonts=self.fonts)

    @staticmethod
    def ImVec2(x: float, y: float) -> tuple[float, float]:
        return x, y

    @staticmethod
    def ImVec4(x: float, y: float, z: float, w: float) -> tuple[float, ...]:
        return x, y, z, w

    @staticmethod
    def color_convert_float4_to_u32(color: tuple[float, ...]) -> int:
        return hash(color)

    @staticmethod
    def calc_text_size(text: str) -> SimpleNamespace:
        return SimpleNamespace(x=float(len(text) * 8), y=14.0)

    def get_font(self) -> object:
        return self.current_font

    def get_font_size(self) -> float:
        return self.current_font_size

    def get_io(self) -> SimpleNamespace:
        return self.io

    def push_font(self, font: object, size: float) -> None:
        self.font_stack.append((self.current_font, self.current_font_size))
        if font is not None:
            self.current_font = font
        self.current_font_size = size

    def pop_font(self) -> None:
        self.current_font, self.current_font_size = self.font_stack.pop()

    def push_style_var(self, style_var: int, value: object) -> None:
        self.pushed_style_vars.append((style_var, value))

    def pop_style_var(self, count: int = 1) -> None:
        del count

    def push_style_color(self, color: int, value: object) -> None:
        self.pushed_style_colors.append((color, value))

    def pop_style_color(self, count: int = 1) -> None:
        del count

    def get_background_draw_list(self) -> _FakeDrawList:
        return self.background_draw_list

    def get_window_draw_list(self) -> _FakeDrawList:
        return self.background_draw_list

    def set_next_window_pos(self, position, condition, pivot=None) -> None:
        self.next_window_position = position
        del condition, pivot

    def set_next_window_size(self, size, condition) -> None:
        self.next_window_size = size
        del condition

    def set_next_window_size_constraints(self, size_min, size_max) -> None:
        del size_min, size_max

    def set_next_window_bg_alpha(self, alpha) -> None:
        del alpha

    def begin(self, title: str, *, flags: int) -> bool:
        self.current_window = title
        self.windows.setdefault(title, [])
        self.window_flags[title] = flags
        return True

    def end(self) -> None:
        self.current_window = None

    def begin_child(
        self,
        child_id: str,
        size: tuple[float, float],
        *,
        child_flags: int = 0,
        window_flags: int = 0,
    ) -> bool:
        del child_flags
        self.child_sizes[child_id] = size
        self.child_window_flags[child_id] = window_flags
        self._child_size_stack.append(size)
        self.current_child_size = size
        return True

    def end_child(self) -> None:
        assert self._child_size_stack
        width, height = self._child_size_stack.pop()
        if height <= 0.0:
            height = 100.0
        self.last_item_rect_size = (width, height)
        self.current_child_size = (
            self._child_size_stack[-1] if self._child_size_stack else None
        )

    def get_item_rect_size(self) -> tuple[float, float]:
        return self.last_item_rect_size

    def get_item_rect_max(self) -> tuple[float, float]:
        window_x, window_y = self.get_window_pos()
        content_bottom = 100.0 if self.current_child_size is not None else 300.0
        return (window_x, window_y + content_bottom)

    @staticmethod
    def get_style() -> SimpleNamespace:
        return SimpleNamespace(
            window_padding=(28.0, 24.0),
            item_spacing=(10.0, 10.0),
            frame_padding=(10.0, 8.0),
            cell_padding=(4.0, 2.0),
            scrollbar_size=14.0,
            display_safe_area_padding=(3.0, 3.0),
        )

    def get_frame_height(self) -> float:
        return self.current_font_size + 16.0

    @staticmethod
    def get_scroll_max_y() -> float:
        return 0.0

    @staticmethod
    def get_scroll_y() -> float:
        return 0.0

    def text(self, value: str) -> None:
        assert self.current_window is not None
        self.windows[self.current_window].append(value)
        self.text_fonts.append((value, self.current_font, self.current_font_size))
        self.text_positions.append((value, self.cursor_y))
        if self.current_table is not None:
            rows = self.tables[self.current_table]
            while len(rows[-1]) <= self.current_table_column:
                rows[-1].append("")
            rows[-1][self.current_table_column] = value
        self.cursor_x = 8.0

    def get_window_pos(self) -> tuple[float, float]:
        return self.next_window_position

    def get_window_size(self) -> tuple[float, float]:
        return self.next_window_size

    def get_cursor_pos_x(self) -> float:
        return self.cursor_x

    def set_cursor_pos_x(self, value: float) -> None:
        self.cursor_x = value

    def get_cursor_pos_y(self) -> float:
        return self.cursor_y

    def set_cursor_pos_y(self, value: float) -> None:
        self.cursor_y = value

    def get_content_region_avail(self) -> tuple[float, float]:
        if self.current_child_size is not None:
            child_width, child_height = self.current_child_size
            available_width = (
                child_width
                if child_width > 0.0
                else max(1.0, float(self.next_window_size[0]) - 56.0)
            )
            if self.current_table is not None:
                available_width /= self.table_column_counts[self.current_table]
            return (
                available_width,
                child_height,
            )
        return (
            max(1.0, float(self.next_window_size[0]) - 56.0),
            max(1.0, float(self.next_window_size[1]) - 48.0),
        )

    def get_cursor_screen_pos(self) -> tuple[float, float]:
        flags = self.window_flags.get(self.current_window or "", 0)
        top_padding = 8.0 if flags & self.WindowFlags_.no_title_bar else 26.0
        return (
            float(self.next_window_position[0]) + 8.0,
            float(self.next_window_position[1]) + top_padding,
        )

    def dummy(self, size: tuple[float, float]) -> None:
        self.dummies.append(size)

    def separator(self) -> None:
        return

    def set_next_item_width(self, width: float) -> None:
        del width

    def input_text(self, label: str, value: str, *, flags: int):
        del flags
        if label in self.input_values:
            return True, self.input_values[label]
        del label, value
        return self.submit_input, self.input_value

    def input_text_multiline(
        self,
        label: str,
        value: str,
        size: tuple[float, float],
        *,
        flags: int,
    ) -> tuple[bool, str]:
        self.multiline_inputs.append((label, value, size, flags))
        self.multiline_input_positions[label] = self.cursor_y
        if label in self.input_values:
            return True, self.input_values[label]
        return False, value

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        if label in self.checkbox_values:
            return True, self.checkbox_values[label]
        return False, value

    def combo(self, label: str, index: int, options: list[str]) -> tuple[bool, int]:
        del options
        if label in self.combo_indices:
            return True, self.combo_indices[label]
        return False, index

    def button(self, label: str, size: tuple[float, float] | None = None) -> bool:
        self.buttons.append(label)
        self.button_sizes.append((label, size))
        self.button_positions.append((label, self.cursor_x))
        self.cursor_x = 8.0
        if self.disabled_depth:
            self.disabled_buttons.append(label)
            return False
        submit = self.click_submit and label in {"SAVE SCORE", "SAVE TIME"}
        return submit or label in self.clicked_buttons

    def image(
        self,
        key: str,
        pixels: np.ndarray,
        *,
        size: tuple[float, float],
    ) -> None:
        self.images.append((key, pixels, size))

    def begin_disabled(self) -> None:
        self.disabled_depth += 1

    def end_disabled(self) -> None:
        self.disabled_depth -= 1

    def same_line(self) -> None:
        return

    def begin_table(
        self,
        table_id: str,
        columns: int,
        *,
        flags: int,
        outer_size: tuple[float, float],
    ) -> bool:
        self.current_table = table_id
        self.tables[table_id] = []
        self.table_columns[table_id] = []
        self.table_column_widths[table_id] = []
        self.table_column_counts[table_id] = columns
        self.table_flags[table_id] = flags
        self.table_outer_sizes[table_id] = outer_size
        return True

    def end_table(self) -> None:
        self.current_table = None

    def table_setup_column(self, label: str, flags: int, width: float) -> None:
        del flags
        assert self.current_table is not None
        self.table_columns[self.current_table].append(label)
        self.table_column_widths[self.current_table].append(width)

    def table_headers_row(self) -> None:
        return

    def table_next_row(self, *, min_row_height: float) -> None:
        del min_row_height
        assert self.current_table is not None
        self.tables[self.current_table].append([])
        self.current_table_column = 0

    def table_set_column_index(self, column: int) -> None:
        self.current_table_column = column

    def table_set_bg_color(self, target: int, color: int) -> None:
        del target, color
        assert self.current_table is not None
        self.highlighted_rows.append(len(self.tables[self.current_table]))


class _CursorBoundaryImGui(_FakeImGui):
    """Model ImGui's requirement that cursor positioning precede an item."""

    def __init__(self) -> None:
        super().__init__()
        self._cursor_position_needs_item = False

    def set_cursor_pos_x(self, value: float) -> None:
        super().set_cursor_pos_x(value)
        self._cursor_position_needs_item = True

    def text(self, value: str) -> None:
        super().text(value)
        self._cursor_position_needs_item = False

    def button(self, label: str, size: tuple[float, float] | None = None) -> bool:
        clicked = super().button(label, size)
        self._cursor_position_needs_item = False
        return clicked

    def end(self) -> None:
        assert not self._cursor_position_needs_item
        super().end()


class _Renderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ui = _FakeImGui()
        self.reset_count = 0
        self.closed = False

    def render(self, step_index, events, step_ui):
        step_ui(self.ui, step_index, events)
        return torch.zeros(4, self.height, self.width)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


@dataclass
class _SubmissionState:
    names: list[str] = field(default_factory=list)

    def submit_player_name(self, name: str) -> None:
        self.names.append(name)


class _SubmissionLoop(IModelLoop[_SubmissionState]):
    def step(self, step_index, events):
        del step_index, events
        return None

    def reset(self) -> None:
        return


@dataclass
class _SelectionState:
    selections: list[GameSelection] = field(default_factory=list)
    live_edit_actions: list[LiveEditAction] = field(default_factory=list)
    return_to_map_count: int = 0
    restart_count: int = 0
    exit_requested: bool = False

    def select_game(self, selection: GameSelection) -> None:
        self.selections.append(selection)

    def return_to_map_menu(self) -> None:
        self.return_to_map_count += 1

    def restart_game(self) -> None:
        self.restart_count += 1

    def request_exit(self) -> None:
        self.exit_requested = True

    def request_live_edit_action(self, action: LiveEditAction) -> None:
        self.live_edit_actions.append(action)


class _SelectionLoop(IModelLoop[_SelectionState]):
    def step(self, step_index, events):
        del step_index, events
        return []

    def reset(self) -> None:
        return


def test_hud_frames_are_immutable_messages_keyed_to_video_storage() -> None:
    video = torch.zeros(2, 3, 96, 160)
    snapshots = (_snapshot(), _snapshot())
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

    frames = build_hud_frames(video, snapshots, poses, speeds_mps=(12.0, -3.0))

    assert [frame.frame_key for frame in frames] == [
        video[index].data_ptr() for index in range(2)
    ]
    assert frames[0].snapshot is snapshots[0]
    assert [frame.speed_mps for frame in frames] == [12.0, -3.0]
    np.testing.assert_array_equal(frames[0].rig_pose_world, poses[0])
    assert not frames[0].rig_pose_world.flags.writeable


def test_hud_frames_preserve_frame_aligned_input_diagnostics() -> None:
    video = torch.zeros(2, 3, 96, 160)
    snapshots = (_snapshot(), _snapshot())
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

    frames = build_hud_frames(
        video,
        snapshots,
        poses,
        transition_timestamps_us=(100, 200),
    )

    assert [frame.transition_timestamp_us for frame in frames] == [100, 200]


def test_hud_frames_preserve_frame_aligned_live_edit_status() -> None:
    video = torch.zeros(2, 3, 96, 160)
    status = LiveEditHudStatus(skin_name="comic", coins_enabled=True)

    frames = build_hud_frames(
        video,
        (_snapshot(), _snapshot()),
        np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        live_edit_status=status,
    )

    assert [frame.live_edit_status for frame in frames] == [status, status]


def test_live_edit_card_dispatches_enabled_actions() -> None:
    live_edit = LiveEditConfig(
        style=LiveEditStyleConfig(enabled=True),
        weather=LiveEditWeatherConfig(enabled=True),
        coins=LiveEditCoinsConfig(enabled=True),
        obstacle=LiveEditObstacleConfig(enabled=True),
    )
    state = TaxiHudState(640, 540, _calibration(), live_edit=live_edit)
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()
    imgui.clicked_buttons = {"K  STYLE", "V  WEATHER", "C  COINS", "O  OBSTACLE"}

    state._draw_live_edit_card(
        imgui,
        LiveEditHudStatus(skin_name="base", weather_name="clear"),
    )
    model_loop._run_message_batch()

    assert model_loop.state.live_edit_actions == [
        "style",
        "weather",
        "coins",
        "obstacle",
    ]
    assert {size for _label, size in imgui.button_sizes} == {(108.0, 34.0)}


def test_live_edit_card_formats_status_and_blocks_weather_during_skin() -> None:
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        live_edit=LiveEditConfig(
            style=LiveEditStyleConfig(enabled=True),
            weather=LiveEditWeatherConfig(enabled=True),
            coins=LiveEditCoinsConfig(enabled=True),
            obstacle=LiveEditObstacleConfig(enabled=True),
        ),
    )
    imgui = _FakeImGui()

    state._draw_live_edit_card(
        imgui,
        LiveEditHudStatus(
            skin_name="comic",
            skin_seconds_remaining=1.5,
            weather_name="rain",
            weather_seconds_remaining=2.5,
            coins_enabled=True,
            coins_collected=3,
            coin_score=30,
            nitro_seconds_remaining=4.0,
            item_flash="NITRO BOOST",
            obstacle_count=2,
            obstacle_hits=1,
        ),
    )

    assert imgui.windows["Live Edit"] == [
        "LIVE EDIT",
        "STYLE  COMIC  1.5s",
        "WEATHER  RAIN  2.5s",
        "COINS  ON  3  +30",
        "NITRO  4.0s",
        "OBSTACLES  2  HITS 1",
        "NITRO BOOST",
    ]
    live_edit_flags = imgui.window_flags["Live Edit"]
    assert live_edit_flags & imgui.WindowFlags_.always_auto_resize
    assert live_edit_flags & imgui.WindowFlags_.no_scrollbar
    assert live_edit_flags & imgui.WindowFlags_.no_scroll_with_mouse
    assert imgui.disabled_buttons == ["V  WEATHER"]


def test_hud_frames_reject_misaligned_input_diagnostics() -> None:
    with pytest.raises(ValueError, match="Input transitions"):
        build_hud_frames(
            torch.zeros(2, 3, 96, 160),
            (_snapshot(), _snapshot()),
            np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
            transition_timestamps_us=(100,),
        )


def test_waypoints_are_projected_and_drawn_on_imgui_background() -> None:
    projections = project_waypoints(
        _snapshot(),
        np.eye(4, dtype=np.float32),
        _calibration(),
        width=160,
        height=96,
    )
    imgui = _FakeImGui()

    draw_waypoints(
        imgui,
        projections,
        phase="seeking_pickup",
        width=160,
        height=96,
    )

    command_names = [name for name, _ in imgui.background_draw_list.commands]
    assert projections
    assert "line" in command_names
    assert "circle" in command_names
    assert "circle_filled" in command_names
    assert "rect_filled" in command_names
    assert "text" in command_names

    terminal = project_waypoints(
        _snapshot(session_state="awaiting_name"),
        np.eye(4, dtype=np.float32),
        _calibration(),
        width=160,
        height=96,
    )
    assert terminal == ()


def test_pickup_waypoint_projection_batches_anchors_and_ring_geometry() -> None:
    class RecordingCamera:
        def __init__(self) -> None:
            self.point_counts: list[int] = []

        def project_world(self, points, rig_to_world):
            del rig_to_world
            points = np.asarray(points)
            self.point_counts.append(len(points))
            uv = np.column_stack(
                (
                    np.full(len(points), 80.0, dtype=np.float32),
                    48.0 - points[:, 2],
                )
            )
            return (
                uv,
                np.ones(len(points), dtype=np.float32),
                np.ones(len(points), dtype=bool),
            )

    camera: Any = RecordingCamera()
    targets = tuple((float(distance), 0.0, 0.0) for distance in range(60, 0, -10))
    snapshot = replace(
        _snapshot(),
        target_xyz_m=targets[-1],
        pickup_targets_xyz_m=targets,
    )

    projections = project_taxi_markers_to_camera(
        snapshot,
        np.eye(4, dtype=np.float32),
        camera,
        image_width=160,
        image_height=96,
    )

    assert camera.point_counts == [6, 102]
    assert [projection.distance_m for projection in projections] == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("show_fps", [False, True])
def test_fps_counter_is_configurable(show_fps: bool) -> None:
    state = TaxiHudState(640, 360, _calibration(), show_fps=show_fps)
    imgui = _FakeImGui()

    state.draw(imgui)

    assert ("Performance" in imgui.windows) is show_fps
    if show_fps:
        assert imgui.windows["Performance"] == ["VIDEO FPS    0.0"]


def test_fps_counter_measures_distinct_generated_video_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_count = 61
    video = torch.zeros(frame_count, 3, 96, 160)
    snapshots = tuple(_snapshot() for _ in range(frame_count))
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    state = TaxiHudState(640, 360, _calibration(), show_fps=True)
    state.publish(build_hud_frames(video, snapshots, poses))
    frame_times = iter(index / 30.0 for index in range(frame_count))
    monkeypatch.setattr(time, "monotonic", lambda: next(frame_times))

    for frame in video:
        state.select_presented_frame(frame)
    state.select_presented_frame(video[-1])
    imgui = _FakeImGui()
    state._draw_fps_counter(imgui)

    assert imgui.windows["Performance"] == ["VIDEO FPS   30.0"]


def test_imgui_ui_loop_draws_waypoints_and_bev_in_the_ui_overlay() -> None:
    width, height = 160, 96
    video = torch.full((1, 3, height, width), -0.5, dtype=torch.bfloat16)
    bev = torch.full((1, 4, 32, 32), 255, dtype=torch.uint8)
    bev[:, :3].fill_(191)
    hud_state = TaxiHudState(width, height, _calibration())
    hud_state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            speeds_mps=(12.0,),
        )
    )
    presentation = PresentationManager()
    presentation.publish(
        0,
        [
            StepResult(0, video, 1, VideoTensorLayout.tchw),
            StepResult(0, bev, 1, VideoTensorLayout.tchw),
        ],
    )
    changed, _ = presentation.advance(0)
    renderer = _Renderer(width, height)
    loop = CrazyRobotaxiImGuiUILoop(
        renderer=renderer,
    )
    loop.register_session_loop_objects(
        state=hud_state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=presentation,
    )

    result = loop.step(0, UserInputEvents([]))

    output = result.read_output()
    assert changed
    assert output.shape == (1, 3, height, width)
    assert output.dtype is torch.float32
    assert hud_state._current is not None
    assert "Crazy Robotaxi" not in renderer.ui.windows
    assert "Navigation" not in renderer.ui.windows
    assert renderer.ui.dummies == [(32.0, 32.0)]
    map_flags = renderer.ui.window_flags["Map"]
    assert map_flags & renderer.ui.WindowFlags_.no_title_bar
    assert map_flags & renderer.ui.WindowFlags_.no_background
    map_borders = [
        command
        for command in renderer.ui.background_draw_list.commands
        if command[0] == "rect" and command[1][4] == 2.0
    ]
    assert len(map_borders) == 1
    command_names = [name for name, _ in renderer.ui.background_draw_list.commands]
    assert "triangle_filled" in command_names
    assert "circle_filled" in command_names
    overlay_text = [
        args[-1]
        for name, args in renderer.ui.background_draw_list.commands
        if name == "text"
    ]
    assert "GAME 42.5s  PICKUP  25m  SCORE 1200  HIGH 9000" in overlay_text
    assert "27" in overlay_text
    assert "mph" in overlay_text
    top, left, panel_height, panel_width = hud_state._bev_rect or (0, 0, 0, 0)
    panel = output[0, :, top : top + panel_height, left : left + panel_width]
    background = 191.0 / 127.5 - 1.0
    assert torch.allclose(panel[:, 0, 0], torch.full_like(panel[:, 0, 0], background))
    assert not torch.allclose(panel, torch.full_like(panel, background))
    torch.testing.assert_close(
        panel[:, panel_height // 2, panel_width // 2], torch.tensor((1.0, 0.6, -1.0))
    )
    outside = output[0].clone()
    outside[:, top : top + panel_height, left : left + panel_width] = -0.5
    assert torch.all(outside == video[0])

    cached_waypoints = hud_state._waypoint_projections
    cached_bev = hud_state._bev_panel
    cached_composite = hud_state._bev_composite
    loop.step(1, UserInputEvents([]))
    assert hud_state._waypoint_projections is cached_waypoints
    assert hud_state._bev_panel is cached_bev
    assert hud_state._bev_composite is cached_composite

    loop.reset()
    assert hud_state._current is None
    assert hud_state._waypoint_projections == ()
    assert hud_state._bev_panel is None
    assert hud_state._bev_composite is None
    assert hud_state._bev_rect is None
    assert renderer.reset_count == 1


def test_bev_compositor_uses_rgba_coverage_for_black_road_pixels() -> None:
    state = TaxiHudState(4, 4, _calibration())
    state._bev_rect = (0, 0, 4, 4)
    video = torch.full((3, 4, 4), -0.5)
    bev = torch.zeros((4, 4, 4), dtype=torch.uint8)
    bev[3, :, 1:3] = 255

    composited = state.composite_bev(video, bev)

    assert torch.all(composited[:, :, (0, 3)] == -0.5)
    assert torch.all(composited[:, :, 1:3] == -1.0)
    assert state._bev_alpha is not None
    assert set(state._bev_alpha.unique().tolist()) == {False, True}


def test_bev_compositor_draws_ego_over_transparent_center() -> None:
    state = TaxiHudState(32, 32, _calibration())
    state._bev_rect = (0, 0, 32, 32)
    video = torch.full((3, 32, 32), -0.5)
    transparent_bev = torch.zeros((4, 32, 32), dtype=torch.uint8)

    composited = state.composite_bev(video, transparent_bev)

    assert composited.device == video.device
    torch.testing.assert_close(composited[:, 16, 16], torch.tensor((1.0, 0.6, -1.0)))
    torch.testing.assert_close(composited[:, 13, 15], torch.tensor((-0.8, -0.2, 0.15)))
    assert torch.all(composited[:, 0, 0] == -0.5)


def test_presentation_back_buffer_is_cached_without_a_bev_frame() -> None:
    state = TaxiHudState(4, 4, _calibration())
    video = torch.full((3, 4, 4), -0.5, dtype=torch.bfloat16)

    first = state.composite_bev(video, None)
    repeated = state.composite_bev(video, None)

    assert first.dtype is torch.float32
    assert repeated is first
    torch.testing.assert_close(first, video.float())


def test_bev_draws_edge_arrow_for_an_offscreen_dropoff() -> None:
    video = torch.zeros(1, 3, 96, 160)
    snapshot = replace(
        _snapshot(),
        phase="to_dropoff",
        target_xyz_m=(500.0, 0.0, 0.0),
        remaining_time_s=20.0,
    )
    state = TaxiHudState(160, 96, _calibration())
    frame = build_hud_frames(
        video,
        (snapshot,),
        np.eye(4, dtype=np.float32)[None],
    )[0]
    state._bev_rect = (0, 0, 96, 96)
    imgui = _FakeImGui()

    state._draw_bev_navigation(imgui, frame)

    triangles = [
        command
        for command in imgui.background_draw_list.commands
        if command[0] == "triangle_filled"
    ]
    assert len(triangles) == 2
    expected_white = imgui.color_convert_float4_to_u32((1.0, 1.0, 1.0, 1.0))
    assert triangles[0][1][-1] == expected_white


def test_bev_draws_visible_waypoints_at_half_opacity() -> None:
    video = torch.zeros(1, 3, 96, 160)
    state = TaxiHudState(160, 96, _calibration())
    frame = build_hud_frames(
        video,
        (_snapshot(),),
        np.eye(4, dtype=np.float32)[None],
    )[0]
    state._bev_rect = (0, 0, 96, 96)
    imgui = _FakeImGui()

    state._draw_bev_navigation(imgui, frame)

    circles = [
        command
        for command in imgui.background_draw_list.commands
        if command[0] == "circle_filled"
    ]
    expected_white = imgui.color_convert_float4_to_u32(
        (1.0, 1.0, 1.0, _BEV_WAYPOINT_ALPHA)
    )
    assert circles[0][1][-1] == expected_white


def test_live_hud_draws_directly_over_the_game_frame() -> None:
    defaults = ControlsConfig()
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        controls=replace(
            defaults,
            keyboard=replace(
                defaults.keyboard,
                restart=(InputBinding("key", "p"), None),
                return_to_menu=(InputBinding("key", "m"), None),
                toggle_hints=(InputBinding("key", "j"), None),
            ),
        ),
    )
    state.publish(
        build_hud_frames(
            torch.zeros(1, 3, 360, 640),
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state._current = next(iter(state._frames.values()))
    state._menu_stage = "game"
    imgui = _FakeImGui()

    state.draw(imgui)

    assert set(imgui.windows) == {"Controls"}
    assert imgui.windows["Controls"][0] == "CONTROLS"
    assert imgui.table_column_counts["##gameplay-control-hints"] == 4
    assert imgui.tables["##gameplay-control-hints"] == [
        ["FORWARD", "W / UP ARROW", "BRAKE / REVERSE", "S / DOWN ARROW"],
        ["STEER LEFT", "A / LEFT ARROW", "STEER RIGHT", "D / RIGHT ARROW"],
        ["HANDBRAKE", "SPACE", "RESTART", "P"],
        ["RETURN TO MENU", "M", "HIDE CONTROLS", "J"],
    ]
    controls_flags = imgui.window_flags["Controls"]
    assert controls_flags & imgui.WindowFlags_.always_auto_resize
    assert controls_flags & imgui.WindowFlags_.no_scrollbar
    assert controls_flags & imgui.WindowFlags_.no_scroll_with_mouse
    overlay_text = [
        args[-1] for name, args in imgui.background_draw_list.commands if name == "text"
    ]
    assert "GAME 42.5s  PICKUP  25m  SCORE 1200  HIGH 9000" in overlay_text
    assert "HIDE CONTROLS" not in overlay_text
    assert "mph" in overlay_text
    assert any(
        name == "triangle_filled" for name, _ in imgui.background_draw_list.commands
    )
    compass = next(
        args
        for name, args in imgui.background_draw_list.commands
        if name == "circle_filled"
    )
    assert compass[0][1] == 110.0


def test_prominent_gameplay_text_uses_droid_sans() -> None:
    state = TaxiHudState(640, 360, _calibration())
    state.publish(
        build_hud_frames(
            torch.zeros(1, 3, 360, 640),
            (replace(_snapshot(), event="pickup_complete"),),
            np.eye(4, dtype=np.float32)[None],
            speeds_mps=(12.0,),
        )
    )
    state._current = next(iter(state._frames.values()))
    state._menu_stage = "game"
    imgui = _FakeImGui()

    state.draw(imgui)

    [(path, size, droid_sans)] = imgui.fonts.loaded
    assert path.endswith("DroidSans.ttf")
    assert size == 13.0
    text_commands = {
        args[-1]: args
        for name, args in imgui.background_draw_list.commands
        if name == "text"
    }
    assert text_commands["PASSENGER PICKED UP"][0] is droid_sans
    assert text_commands["27"][0] is droid_sans
    assert text_commands["mph"][0] is droid_sans
    assert (
        text_commands["GAME 42.5s  PICKUP  25m  SCORE 1200  HIGH 9000"][0]
        is imgui.default_font
    )


def test_compass_arrow_has_no_black_underlay() -> None:
    state = TaxiHudState(160, 96, _calibration())
    imgui = _FakeImGui()

    state._draw_navigation_arrow(
        imgui,
        0.0,
        center_y=198.0,
        color_rgb=(118.0 / 255.0, 185.0 / 255.0, 0.0),
    )

    commands = imgui.background_draw_list.commands
    assert sum(name == "line" for name, _ in commands) == 1
    assert sum(name == "triangle_filled" for name, _ in commands) == 1


def test_hud_animates_prepresentation_warmup_status() -> None:
    state = TaxiHudState(160, 96, _calibration())
    state._menu_stage = "loading"
    state.set_loading_status("WARMING WORLD MODEL  2/4")
    imgui = _FakeImGui()

    state.draw(imgui, ui_tick=30)

    lines = imgui.windows["Crazy Robotaxi"]
    assert lines[0] == "WARMING WORLD MODEL  2/4..."
    assert lines[1].startswith("ELAPSED  ")


def test_two_map_selections_render_in_one_grid_row() -> None:
    preview_path = Path("map-preview.jpg")
    maps = tuple(
        GameMapOption(
            map_id=f"map-{index}",
            name=f"Map {index}",
            path=Path(f"map-{index}.robotaxi.yaml"),
            preview_image_path=preview_path,
        )
        for index in range(2)
    )
    state = TaxiHudState(1280, 720, _calibration(), map_options=maps)
    state._selection_preview_pixels[preview_path] = np.zeros(
        (90, 160, 3), dtype=np.uint8
    )
    state._selected_game_mode = "taxi"
    state._menu_stage = "map"
    imgui = _FakeImGui()

    state.draw(imgui)

    assert imgui.table_column_counts["##map-grid"] == 2
    assert len(imgui.tables["##map-grid"]) == 1


def test_selection_menus_use_arcade_card_layout(tmp_path: Path) -> None:
    map_preview_path = Path("map-preview.jpg")
    course_preview_path = Path("course-preview.jpg")
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
        race_courses=(
            GameRaceCourseOption(
                course_id="downtown-sprint",
                spawn_id="race-start",
                preview_image_path=course_preview_path,
            ),
        ),
        preview_image_path=map_preview_path,
    )
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        native_dit_disabled_for_live_edit=True,
        settings_document=_settings_document(tmp_path / "config.yaml"),
        map_options=(option,),
    )
    state._selection_preview_pixels = {
        map_preview_path: np.zeros((90, 160, 3), dtype=np.uint8),
        course_preview_path: np.zeros((100, 200, 3), dtype=np.uint8),
    }
    state._settings_restart_notice = "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT"
    imgui = _CursorBoundaryImGui()

    state.draw(imgui)
    state._selected_game_mode = "race"
    state._menu_stage = "map"
    state.draw(imgui)
    state._selected_map_option = option
    state._menu_stage = "course"
    state.draw(imgui)

    [(path, _size, droid_sans)] = imgui.fonts.loaded
    assert path.endswith("DroidSans.ttf")
    text_fonts = {text: font for text, font, _size in imgui.text_fonts}
    assert text_fonts["CRAZY ROBOTAXI"] is droid_sans
    assert text_fonts["SELECT MAP"] is droid_sans
    assert text_fonts["SELECT RACE COURSE"] is droid_sans
    for title in (
        "Crazy Robotaxi - Select Game Mode",
        "Crazy Robotaxi - Select Map",
        "Crazy Robotaxi - Select Race Course",
    ):
        flags = imgui.window_flags[title]
        assert flags & imgui.WindowFlags_.no_title_bar
        assert flags & imgui.WindowFlags_.always_auto_resize
        assert flags & imgui.WindowFlags_.no_scrollbar
        assert flags & imgui.WindowFlags_.no_scroll_with_mouse
    button_sizes = dict(imgui.button_sizes)
    button_positions = dict(imgui.button_positions)
    assert button_sizes["TAXI"] == button_sizes["RACE"]
    for label in ("TAXI", "Test City##map-0", "DOWNTOWN SPRINT##course-0"):
        size = button_sizes[label]
        assert size is not None and size[0] > 0.0
    assert [key for key, _pixels, _size in imgui.images] == [
        "selection-preview:map-preview.jpg",
        "selection-preview:course-preview.jpg",
    ]
    map_button_size = button_sizes["Test City##map-0"]
    course_button_size = button_sizes["DOWNTOWN SPRINT##course-0"]
    assert map_button_size is not None
    assert course_button_size is not None
    assert imgui.child_sizes["##map-options"][0] == map_button_size[0]
    assert imgui.child_sizes["##course-options"][0] == course_button_size[0]
    assert imgui.buttons.count("CONTROLS") == 1
    assert imgui.buttons.count("OPTIONS") == 1
    assert imgui.buttons.count("EXIT") == 1
    for label in (
        "TAXI",
        "RACE",
        "CONTROLS",
        "OPTIONS",
        "EXIT",
        "Test City##map-0",
        "DOWNTOWN SPRINT##course-0",
    ):
        assert button_positions[label] > 8.0
    for title in (
        "Crazy Robotaxi - Select Map",
        "Crazy Robotaxi - Select Race Course",
    ):
        lines = imgui.windows[title]
        assert "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT" not in lines
        assert "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES" not in lines
    assert (imgui.Col_.text, _RESTART_NOTICE_RGBA) in imgui.pushed_style_colors
    assert (imgui.Col_.text, _NATIVE_DIT_NOTICE_RGBA) in imgui.pushed_style_colors
    assert [command for command, _args in imgui.background_draw_list.commands].count(
        "rect_filled"
    ) == 3


def test_map_and_course_selections_use_three_column_grids() -> None:
    map_preview_path = Path("map-preview.jpg")
    maps = tuple(
        GameMapOption(
            map_id=f"map-{index}",
            name=f"Map {index}",
            path=Path(f"map-{index}.robotaxi.yaml"),
            race_courses=(
                GameRaceCourseOption(
                    course_id="course-0",
                    spawn_id="race-start",
                ),
            ),
            preview_image_path=map_preview_path,
        )
        for index in range(6)
    )
    state = TaxiHudState(1280, 720, _calibration(), map_options=maps)
    state._selection_preview_pixels[map_preview_path] = np.zeros(
        (90, 160, 3), dtype=np.uint8
    )
    state._selected_game_mode = "race"
    state._menu_stage = "map"
    map_imgui = _FakeImGui()

    state.draw(map_imgui)

    assert map_imgui.table_column_counts["##map-grid"] == 3
    assert (
        map_imgui.table_flags["##map-grid"] & map_imgui.TableFlags_.sizing_stretch_same
    )
    assert len(map_imgui.tables["##map-grid"]) == 2
    map_button_sizes = dict(map_imgui.button_sizes)
    map_button_size = map_button_sizes["Map 0##map-0"]
    assert map_button_size is not None
    assert len(map_imgui.images) == 6

    course_preview_path = Path("course-preview.jpg")
    course_option = GameMapOption(
        map_id="course-map",
        name="Course Map",
        path=Path("course-map.robotaxi.yaml"),
        race_courses=tuple(
            GameRaceCourseOption(
                course_id=f"course-{index}",
                spawn_id=f"race-start-{index}",
                preview_image_path=course_preview_path,
            )
            for index in range(7)
        ),
    )
    state._selection_preview_pixels[course_preview_path] = np.zeros(
        (90, 160, 3), dtype=np.uint8
    )
    state._selected_map_option = course_option
    state._menu_stage = "course"
    course_imgui = _FakeImGui()

    state.draw(course_imgui)

    assert course_imgui.table_column_counts["##course-grid"] == 3
    assert (
        course_imgui.table_flags["##course-grid"]
        & course_imgui.TableFlags_.sizing_stretch_same
    )
    assert len(course_imgui.tables["##course-grid"]) == 3
    course_button_sizes = dict(course_imgui.button_sizes)
    course_button_size = course_button_sizes["COURSE 0##course-0"]
    assert course_button_size is not None
    assert len(course_imgui.images) == 7


def test_controls_menu_edits_and_saves_one_device(tmp_path: Path) -> None:
    documents = load_controls_documents(tmp_path / "controls")
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        live_edit=LiveEditConfig(
            style=LiveEditStyleConfig(enabled=True),
            coins=LiveEditCoinsConfig(enabled=True),
        ),
        settings_document=_settings_document(tmp_path / "config.yaml"),
        control_documents=documents,
    )
    menu_imgui = _FakeImGui()

    state.draw(menu_imgui)

    menu_buttons = [label for label, _size in menu_imgui.button_sizes]
    assert menu_buttons.index("CONTROLS") + 1 == menu_buttons.index("OPTIONS")

    open_imgui = _FakeImGui()
    open_imgui.clicked_buttons.add("CONTROLS")
    state.draw(open_imgui)
    assert state._menu_stage == "controls"

    controls_imgui = _FakeImGui()
    state.draw(controls_imgui)

    assert {"KEYBOARD", "GAMEPAD", "WHEEL", "BACK"} <= set(controls_imgui.buttons)
    landing_positions = dict(controls_imgui.button_positions)
    assert (
        len(
            {
                landing_positions[label]
                for label in ("KEYBOARD", "GAMEPAD", "WHEEL", "BACK")
            }
        )
        == 1
    )
    assert landing_positions["KEYBOARD"] > 8.0

    keyboard_imgui = _FakeImGui()
    keyboard_imgui.clicked_buttons.add("KEYBOARD")
    state.draw(keyboard_imgui)
    assert state._controls_device == "keyboard"

    capture_imgui = _FakeImGui()
    capture_imgui.clicked_buttons.add("R##keyboard-restart-0")
    state.draw(capture_imgui)
    state.consume_input_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(1),
                    key="p",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )
    assert state._controls_draft is not None
    assert state._controls_draft.restart == (InputBinding("key", "p"), None)

    save_imgui = _FakeImGui()
    save_imgui.clicked_buttons.add("SAVE")
    state.draw(save_imgui)
    assert "EXIT WITHOUT SAVING" in save_imgui.buttons
    assert save_imgui.table_columns["##controls-keyboard-table"] == [
        "ACTION",
        "PRIMARY",
        "SECONDARY",
    ]
    assert documents["keyboard"].settings.restart == (
        InputBinding("key", "p"),
        None,
    )
    assert (tmp_path / "controls" / "keyboard.yaml").exists()
    assert state._settings_restart_notice
    assert (save_imgui.Col_.text, _SAVED_NOTICE_RGBA) in save_imgui.pushed_style_colors
    assert (save_imgui.Col_.text, _RESTART_NOTICE_RGBA) in (
        save_imgui.pushed_style_colors
    )

    reset_imgui = _FakeImGui()
    reset_imgui.clicked_buttons.add("RESET TO DEFAULTS")
    state.draw(reset_imgui)
    assert (
        "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT"
        not in reset_imgui.windows["Crazy Robotaxi - Keyboard Controls"]
    )

    save_defaults_imgui = _FakeImGui()
    save_defaults_imgui.clicked_buttons.add("SAVE")
    state.draw(save_defaults_imgui)
    assert not state._settings_restart_notice

    exit_imgui = _FakeImGui()
    exit_imgui.clicked_buttons.add("EXIT")
    state.draw(exit_imgui)
    assert state._controls_device is None

    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    state._menu_stage = "controls"
    state.consume_input_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(1),
                    key="Escape",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )

    assert state._menu_stage == "mode"
    model_loop._run_message_batch()
    assert not model_loop.state.exit_requested


def test_gamepad_controls_show_only_the_configured_button_style(
    tmp_path: Path,
) -> None:
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        gamepad_button_style="PlayStation",
        control_documents=load_controls_documents(tmp_path / "controls"),
    )
    state._open_controls()
    state._open_controls_device("gamepad")
    imgui = _FakeImGui()

    state.draw(imgui)

    assert "R2##gamepad-throttle-0" in imgui.buttons
    assert "L2##gamepad-brake-0" in imgui.buttons
    assert "OPTIONS##gamepad-restart-0" in imgui.buttons
    assert "SHARE##gamepad-return_to_menu-0" in imgui.buttons
    assert not any("##gamepad-reverse-" in label for label in imgui.buttons)
    assert not any(
        "/" in label.split("##", 1)[0]
        for label in imgui.buttons
        if "##gamepad-" in label
    )


def test_missing_selection_thumbnail_keeps_text_button(tmp_path: Path) -> None:
    missing_thumbnail = tmp_path / "missing.png"
    option = GameMapOption(
        map_id="text-only",
        name="Text Only",
        path=Path("text-only.robotaxi.yaml"),
        preview_image_path=missing_thumbnail,
    )
    state = TaxiHudState(640, 540, _calibration(), map_options=(option,))
    state._selected_game_mode = "taxi"
    state._menu_stage = "map"
    imgui = _FakeImGui()

    state.draw(imgui)

    assert "Text Only##map-0" in imgui.buttons
    assert not imgui.images
    assert state._selection_preview_pixels[missing_thumbnail] is None


def test_selection_thumbnail_size_does_not_depend_on_choice_count() -> None:
    first_path = Path("first-preview.jpg")
    second_path = Path("second-preview.jpg")
    first = GameMapOption(
        map_id="first",
        name="First",
        path=Path("first.robotaxi.yaml"),
        preview_image_path=first_path,
    )
    second = GameMapOption(
        map_id="second",
        name="Second",
        path=Path("second.robotaxi.yaml"),
        preview_image_path=second_path,
    )
    state = TaxiHudState(640, 360, _calibration(), map_options=(first,))
    state._selected_game_mode = "taxi"
    state._menu_stage = "map"
    preview = np.zeros((600, 300, 3), dtype=np.uint8)
    state._selection_preview_pixels = {
        first_path: preview,
        second_path: preview,
    }

    one_choice = _FakeImGui()
    state.draw(one_choice)
    state.map_options = (first, second)
    two_choices = _FakeImGui()
    state.draw(two_choices)

    assert one_choice.images[0][2] == two_choices.images[0][2]
    assert two_choices.images[0][2] == two_choices.images[1][2]


def test_startup_menu_selects_taxi_mode_then_map_through_v2_message() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
        race_courses=(
            GameRaceCourseOption(
                course_id="downtown-sprint",
                spawn_id="race-start",
            ),
        ),
    )
    state = TaxiHudState(640, 360, _calibration(), map_options=(option,))
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()
    imgui.clicked_buttons.add("TAXI")

    state.draw(imgui)

    assert state._menu_stage == "map"
    assert "Crazy Robotaxi - Select Game Mode" in imgui.windows
    imgui.clicked_buttons = {"Test City##map-0"}
    state.draw(imgui)

    assert state._menu_stage == "loading"
    model_loop._run_message_batch()
    assert model_loop.state.selections == [
        GameSelection(mode="taxi", map_option=option)
    ]


def test_race_menu_selects_map_then_course() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
        race_courses=(
            GameRaceCourseOption(
                course_id="downtown-sprint",
                spawn_id="race-start",
            ),
        ),
    )
    state = TaxiHudState(640, 360, _calibration(), map_options=(option,))
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()
    imgui.clicked_buttons.add("RACE")
    state.draw(imgui)
    imgui.clicked_buttons = {"Test City##map-0"}

    state.draw(imgui)
    assert state._menu_stage == "course"
    imgui.clicked_buttons = {"DOWNTOWN SPRINT##course-0"}

    state.draw(imgui)
    model_loop._run_message_batch()

    assert model_loop.state.selections == [
        GameSelection(
            mode="race",
            map_option=option,
            race_course_id="downtown-sprint",
        )
    ]


def test_complete_cli_selection_skips_all_selection_screens() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml").resolve(),
        race_courses=(
            GameRaceCourseOption(
                course_id="downtown-sprint",
                spawn_id="race-start",
            ),
        ),
    )
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        map_options=(option,),
        initial_game_mode="race",
        initial_map_path=option.path,
        initial_race_course_id="downtown-sprint",
    )
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop

    state.initialize_selection()

    assert state._menu_stage == "loading"
    model_loop._run_message_batch()
    assert model_loop.state.selections == [
        GameSelection(
            mode="race",
            map_option=option,
            race_course_id="downtown-sprint",
        )
    ]


def test_explicit_race_mode_and_map_skip_to_course_screen() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml").resolve(),
        race_courses=(
            GameRaceCourseOption(
                course_id="downtown-sprint",
                spawn_id="race-start",
            ),
        ),
    )
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        map_options=(option,),
        initial_game_mode="race",
        initial_map_path=option.path,
    )

    state.initialize_selection()

    assert state._menu_stage == "course"
    assert state._selected_map_option is option


def test_escape_navigates_game_to_map_to_mode_then_exits() -> None:
    state = TaxiHudState(640, 360, _calibration())
    state._selected_game_mode = "race"
    state._menu_stage = "game"
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    released = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="Escape",
        state=KeyboardInputState.RELEASED,
    )
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(2),
        key="Escape",
        state=KeyboardInputState.PRESSED,
    )

    state.consume_input_events(UserInputEvents([released]))
    assert state._menu_stage == "game"

    state.consume_input_events(UserInputEvents([pressed]))
    assert state._menu_stage == "map"
    model_loop._run_message_batch()
    assert model_loop.state.return_to_map_count == 1

    state.consume_input_events(UserInputEvents([released]))
    state.consume_input_events(UserInputEvents([pressed]))
    assert state._menu_stage == "mode"
    assert state._selected_game_mode is None

    state.consume_input_events(UserInputEvents([released]))
    state.consume_input_events(UserInputEvents([pressed]))
    assert state._menu_stage == "loading"
    assert state._loading_status == "EXITING GAME"
    model_loop._run_message_batch()
    assert model_loop.state.exit_requested


def test_menu_back_uses_styled_gamepad_cancel_button() -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        gamepad_button_style="Xbox",
    )
    state._selected_game_mode = "taxi"
    state._menu_stage = "map"

    state.consume_input_events(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(1),
                    action="state",
                    pressed=(False, True),
                )
            ]
        )
    )

    assert state._menu_stage == "mode"
    imgui = _FakeImGui()
    state.draw(imgui)
    assert "ESC / B  EXIT" in imgui.windows["Crazy Robotaxi - Select Game Mode"]


def test_gameplay_return_to_map_uses_rebindable_controls() -> None:
    defaults = ControlsConfig()
    controls = replace(
        defaults,
        keyboard=replace(
            defaults.keyboard,
            return_to_menu=(InputBinding("key", "m"), None),
        ),
    )
    state = TaxiHudState(640, 360, _calibration(), controls=controls)
    state._selected_game_mode = "taxi"
    state._menu_stage = "game"
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop

    state.consume_input_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(1),
                    key="Escape",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )
    assert state._menu_stage == "game"

    state.consume_input_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(2),
                    key="m",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )

    assert state._menu_stage == "map"
    model_loop._run_message_batch()
    assert model_loop.state.return_to_map_count == 1


def test_mode_exit_button_requests_exit() -> None:
    state = TaxiHudState(640, 360, _calibration())
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()
    imgui.clicked_buttons.add("EXIT")

    state.draw(imgui)

    assert state._menu_stage == "loading"
    assert state._loading_status == "EXITING GAME"
    model_loop._run_message_batch()
    assert model_loop.state.exit_requested


def test_h_toggles_gameplay_control_tooltips() -> None:
    state = TaxiHudState(640, 360, _calibration())
    released = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="h",
        state=KeyboardInputState.RELEASED,
    )
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(2),
        key="H",
        state=KeyboardInputState.PRESSED,
    )

    state.consume_input_events(UserInputEvents([released]))
    assert state.show_control_tooltips

    state.consume_input_events(UserInputEvents([pressed]))
    assert not state.show_control_tooltips

    state.consume_input_events(UserInputEvents([released]))
    state.consume_input_events(UserInputEvents([pressed]))
    assert state.show_control_tooltips


def test_control_tooltip_card_uses_one_pair_per_row_when_narrow() -> None:
    state = TaxiHudState(160, 96, _calibration())
    imgui = _FakeImGui()

    state._draw_control_tooltips(imgui)

    assert imgui.table_column_counts["##gameplay-control-hints"] == 2
    assert len(imgui.tables["##gameplay-control-hints"]) == 8


def test_connected_gamepad_replaces_keyboard_gameplay_hints() -> None:
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        gamepad_button_style="Xbox",
    )
    state.consume_input_events(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(1),
                    action="state",
                    axes=(0.0,),
                    buttons=(0.0,) * 10,
                )
            ]
        )
    )
    imgui = _FakeImGui()

    state._draw_control_tooltips(imgui)

    hints = [value for row in imgui.tables["##gameplay-control-hints"] for value in row]
    assert "RT" in hints
    assert "LT" in hints
    assert "LEFT STICK X" in hints
    assert "LEFT STICK X (INVERTED)" not in hints
    assert "START / MENU" in hints
    assert "BACK / VIEW" in hints
    assert "W / UP ARROW" not in hints
    column_widths = imgui.table_column_widths["##gameplay-control-hints"]
    assert column_widths[1::2] == [
        max(imgui.calc_text_size(value).x for value in hints[1::2])
    ] * (len(column_widths) // 2)

    terminal_imgui = _FakeImGui()
    state._draw_terminal(
        terminal_imgui,
        _snapshot(session_state="leaderboard"),
    )
    assert (
        "START / MENU RESTART   |   BACK / VIEW MENU"
        in terminal_imgui.windows["Game Over"]
    )


def test_input_latency_profile_correlates_ui_event_with_model_frame() -> None:
    video = torch.zeros(1, 3, 96, 160)
    state = TaxiHudState(
        160,
        96,
        _calibration(),
        profile_input_latency=True,
    )
    state.consume_input_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(100),
                    key="ArrowLeft",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            transition_timestamps_us=(100,),
        )
    )

    state.select_presented_frame(video[0])
    imgui = _FakeImGui()
    state.draw(imgui)

    assert state._latest_input_latency_ms is not None
    diagnostics = imgui.windows["Input Latency"]
    assert "A / LEFT ARROW [X]" in diagnostics[0]
    assert "UI TO MODEL FRAME" in diagnostics[1]

    state.reset()
    assert not state._profile_pressed
    assert state._latest_input_latency_ms is None


def test_input_latency_profile_correlates_gamepad_state() -> None:
    video = torch.zeros(1, 3, 96, 160)
    state = TaxiHudState(
        160,
        96,
        _calibration(),
        profile_input_latency=True,
    )
    state.consume_input_events(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(200),
                    action="state",
                    axes=(0.25,),
                )
            ]
        )
    )
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            transition_timestamps_us=(200,),
        )
    )

    state.select_presented_frame(video[0])

    assert state._latest_input_latency_ms is not None


def test_input_trace_reports_committed_state_ahead_of_presented_frame(caplog) -> None:
    presented_video = torch.zeros(1, 3, 96, 160)
    committed_video = torch.ones(1, 3, 96, 160)
    state = TaxiHudState(
        160,
        96,
        _calibration(),
        profile_input_latency=True,
    )
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(300),
        key="d",
        state=KeyboardInputState.PRESSED,
    )

    with caplog.at_level(logging.INFO, logger="flashdreams.runtime_v2.chunk_trace"):
        state.consume_input_events(UserInputEvents([pressed]))
        state.publish(
            build_hud_frames(
                presented_video,
                (_snapshot(),),
                np.eye(4, dtype=np.float32)[None],
                transition_timestamps_us=(300,),
                runtime_generation=2,
                model_step_index=10,
                rollout_epoch=4,
                autoregressive_index=1,
                simulation_timestamps_us=(1_000,),
                cache_finalize_returned_ns=time.monotonic_ns() - 1_000_000,
            )
        )
        state.publish(
            build_hud_frames(
                committed_video,
                (_snapshot(),),
                np.eye(4, dtype=np.float32)[None],
                runtime_generation=2,
                model_step_index=11,
                rollout_epoch=4,
                autoregressive_index=2,
                simulation_timestamps_us=(9_000,),
                cache_finalize_returned_ns=time.monotonic_ns(),
            )
        )
        state.select_presented_frame(presented_video[0])

    trace = "\n".join(record.getMessage() for record in caplog.records)
    assert "phase=input_received" in trace
    assert "event_us=300 source=keyboard key=d state=Pressed" in trace
    assert "phase=app_frame_presented" in trace
    assert "generation=2 step=10 epoch=4 ar=1 frame=0" in trace
    assert "step_lead=1 ar_lead=1 simulation_lead_ms=8.0" in trace
    assert "event_us=300 ui_to_frame_ms=" in trace


def test_input_trace_is_silent_without_opt_in(caplog) -> None:
    state = TaxiHudState(160, 96, _calibration())
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(400),
        key="d",
        state=KeyboardInputState.PRESSED,
    )

    with caplog.at_level(logging.INFO, logger="flashdreams.runtime_v2.chunk_trace"):
        state.consume_input_events(UserInputEvents([pressed]))

    assert "chunk-trace" not in "\n".join(
        record.getMessage() for record in caplog.records
    )


def test_input_latency_window_is_absent_by_default() -> None:
    state = TaxiHudState(160, 96, _calibration())
    imgui = _FakeImGui()

    state.draw(imgui)

    assert "Input Latency" not in imgui.windows


def test_imgui_name_submission_uses_v2_loop_message_queue() -> None:
    state = TaxiHudState(160, 96, _calibration())
    model_loop = _SubmissionLoop()
    model_loop.register_session_loop_objects(
        state=_SubmissionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    video = torch.zeros(1, 3, 96, 160)
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(session_state="awaiting_name"),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state._menu_stage = "loading"
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()
    imgui.input_value = " DRIVER 7 "
    imgui.click_submit = True

    state.draw(imgui)
    state.draw(imgui)

    assert model_loop.state.names == []
    model_loop._run_message_batch()
    assert model_loop.state.names == ["DRIVER 7"]
    assert state._submission_pending
    assert "Game Over" in imgui.windows


def test_taxi_results_card_draws_ranked_leaderboard() -> None:
    defaults = ControlsConfig()
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        controls=replace(
            defaults,
            keyboard=replace(
                defaults.keyboard,
                restart=(InputBinding("key", "p"), None),
                return_to_menu=(InputBinding("key", "m"), None),
            ),
        ),
    )
    video = torch.zeros(1, 3, 540, 640)
    entries = (
        HighScoreEntry("ACE", 2400, "2026-01-01T00:00:00Z"),
        HighScoreEntry("DRIVER 7", 1200, "2026-01-02T00:00:00Z"),
    )
    state.publish(
        build_hud_frames(
            video,
            (
                replace(
                    _snapshot(session_state="leaderboard"),
                    leaderboard=entries,
                    high_score_rank=2,
                ),
            ),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    [(path, _size, droid_sans)] = imgui.fonts.loaded
    assert path.endswith("DroidSans.ttf")
    text_fonts = {text: font for text, font, _size in imgui.text_fonts}
    assert text_fonts["GAME OVER"] is droid_sans
    assert text_fonts["001200"] is droid_sans
    assert text_fonts["LEADERBOARD"] is imgui.default_font
    assert imgui.table_columns["##leaderboard"] == ["RANK", "DRIVER", "SCORE"]
    assert imgui.tables["##leaderboard"] == [
        ["#1", "ACE", "   2400"],
        ["#2", "DRIVER 7", "   1200"],
    ]
    assert imgui.highlighted_rows == [2]
    rank_width, driver_width, score_width = imgui.table_column_widths["##leaderboard"]
    cell_padding = 2.0 * imgui.get_style().cell_padding[0]
    assert rank_width >= imgui.calc_text_size("RANK").x + cell_padding
    assert driver_width >= imgui.calc_text_size("DRIVER 7").x + cell_padding
    assert score_width >= imgui.calc_text_size("   2400").x + cell_padding
    assert imgui.table_outer_sizes["##leaderboard"][0] >= (
        rank_width + driver_width + score_width + imgui.get_style().scrollbar_size
    )
    assert imgui.table_outer_sizes["##leaderboard"][0] >= state.width * 0.5
    assert "PLAY AGAIN" in imgui.buttons
    assert "P RESTART   |   M MENU" in imgui.windows["Game Over"]
    results_flags = imgui.window_flags["Game Over"]
    assert results_flags & imgui.WindowFlags_.always_auto_resize
    assert results_flags & imgui.WindowFlags_.no_scrollbar
    assert results_flags & imgui.WindowFlags_.no_scroll_with_mouse
    assert imgui.table_flags["##leaderboard"] & imgui.TableFlags_.scroll_y


def test_race_results_card_formats_times() -> None:
    state = TaxiHudState(640, 540, _calibration())
    video = torch.zeros(1, 3, 540, 640)
    entries = (
        RaceTimeEntry(
            "test-city",
            "downtown-sprint",
            "RACER",
            42_345_000,
            "2026-01-01T00:00:00Z",
        ),
    )
    state.publish(
        build_hud_frames(
            video,
            (
                replace(
                    _race_snapshot(session_state="leaderboard"),
                    leaderboard=entries,
                    high_score_rank=1,
                ),
            ),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    assert "RACE COMPLETE" in imgui.windows["Game Over"]
    assert "0:42.345" in imgui.windows["Game Over"]
    assert imgui.table_columns["##leaderboard"] == ["RANK", "DRIVER", "TIME"]
    assert imgui.tables["##leaderboard"] == [["#1", "RACER", "0:42.345"]]
    time_width = imgui.table_column_widths["##leaderboard"][2]
    assert time_width >= (
        imgui.calc_text_size("0:42.345").x + 2.0 * imgui.get_style().cell_padding[0]
    )


@pytest.mark.parametrize("session_state", ["awaiting_name", "leaderboard"])
def test_terminal_play_again_requests_restart(session_state: TaxiSessionState) -> None:
    state = TaxiHudState(640, 360, _calibration())
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    video = torch.zeros(1, 3, 360, 640)
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(session_state=session_state),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state._menu_stage = "loading"
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()
    imgui.clicked_buttons.add("PLAY AGAIN")

    state.draw(imgui)
    model_loop._run_message_batch()

    assert model_loop.state.restart_count == 1


def _settings_document(path: Path) -> SettingsDocument:
    return SettingsDocument.load(
        path,
        pipeline_config=_SettingsPipeline(),
        width=640,
        height=360,
    )


def _perf_settings_document(path: Path) -> SettingsDocument:
    return SettingsDocument.load(
        path,
        pipeline_config=OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
        width=640,
        height=360,
    )


@pytest.mark.parametrize(
    ("stage", "expected_stage"),
    [("mode", "options"), ("map", "map"), ("course", "course")],
)
def test_options_can_only_open_from_mode_selection(
    tmp_path: Path,
    stage: Literal["mode", "map", "course"],
    expected_stage: Literal["options", "map", "course"],
) -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    state._menu_stage = stage

    state._open_options()

    assert state._menu_stage == expected_stage
    if stage == "mode":
        assert state._options_return_stage == "mode"


def test_options_excludes_cli_only_launch_selections(tmp_path: Path) -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    state._open_options()
    imgui = _FakeImGui()

    state.draw(imgui)

    labels = {label for label, _size in imgui.button_sizes}
    options_flags = imgui.window_flags["Crazy Robotaxi - Options"]
    assert options_flags & imgui.WindowFlags_.always_auto_resize
    assert options_flags & imgui.WindowFlags_.no_scrollbar
    assert options_flags & imgui.WindowFlags_.no_scroll_with_mouse
    assert "##options-categories" in imgui.child_sizes
    assert "##options-fields" in imgui.child_sizes
    assert "GAME##options-category-game" in labels
    assert "LAUNCH##options-category-launch" not in labels
    assert "SAVE" in labels
    assert "EXIT" in labels
    assert "EXIT WITHOUT SAVING" not in labels
    assert "RESET TO DEFAULTS" in labels


def test_options_category_click_opens_model_settings(tmp_path: Path) -> None:
    state = TaxiHudState(
        1280,
        720,
        _calibration(),
        settings_document=_perf_settings_document(tmp_path / "config.yaml"),
    )
    state._open_options()
    click_imgui = _FakeImGui()
    click_imgui.clicked_buttons.add("MODEL##options-category-model")

    state.draw(click_imgui)

    assert state._options_category == "model"
    model_imgui = _FakeImGui()
    state.draw(model_imgui)
    lines = model_imgui.windows["Crazy Robotaxi - Options"]
    assert "Device:" in lines
    assert not any("READ ONLY" in line for line in lines)
    assert lines.index("PIPELINE") < lines.index("Enable Sync And Profile:")
    assert lines.index("Enable Sync And Profile:") < lines.index("DIFFUSION MODEL")
    assert lines.index("DIFFUSION MODEL") < lines.index("Seed:")
    assert lines.index("Seed:") < lines.index("TRANSFORMER")
    assert lines.index("TRANSFORMER") < lines.index("Dtype:")


def test_options_booleans_use_compact_native_green_checkboxes(
    tmp_path: Path,
) -> None:
    state = TaxiHudState(
        1280,
        720,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    state._open_options()
    state._options_category = "presentation"
    imgui = _FakeImGui()

    state.draw(imgui)

    native_check_colors = [
        color
        for style, color in imgui.pushed_style_colors
        if style == imgui.Col_.check_mark
    ]
    assert native_check_colors and set(native_check_colors) == {(0.25, 0.85, 0.25, 1.0)}
    compact_padding = [
        value
        for style, value in imgui.pushed_style_vars
        if style == imgui.StyleVar_.frame_padding and value == (10.0, 4.0)
    ]
    assert len(compact_padding) == 3


def test_options_text_fields_wrap_without_resizing_the_submenu(
    tmp_path: Path,
) -> None:
    document = _settings_document(tmp_path / "config.yaml")

    def draw_device(value: str) -> _FakeImGui:
        state = TaxiHudState(
            1280,
            720,
            _calibration(),
            settings_document=document,
        )
        state._open_options()
        state._options_category = "model"
        assert state._options_draft is not None
        state._options_draft = document.update(
            state._options_draft,
            ("model", "device"),
            value,
        )
        imgui = _FakeImGui()
        state.draw(imgui)
        return imgui

    short = draw_device("cuda")
    wrapped = draw_device("wrapped words " * 20)
    unbreakable = draw_device("/" + "long-path-segment" * 20)

    assert {
        short.child_sizes["##options-fields"][0],
        wrapped.child_sizes["##options-fields"][0],
        unbreakable.child_sizes["##options-fields"][0],
    } == {short.child_sizes["##options-fields"][0]}
    short_field = next(
        item for item in short.multiline_inputs if item[0] == "##model.device"
    )
    wrapped_field = next(
        item for item in wrapped.multiline_inputs if item[0] == "##model.device"
    )
    path_field = next(
        item for item in unbreakable.multiline_inputs if item[0] == "##model.device"
    )
    assert wrapped_field[2][1] > short_field[2][1]
    wrapped_label_y = next(y for text, y in wrapped.text_positions if text == "Device:")
    assert wrapped_label_y == pytest.approx(
        wrapped.multiline_input_positions["##model.device"]
        + (wrapped_field[2][1] - wrapped.calc_text_size("Device:").y) / 2.0
    )
    scroll_id = "##model.device-horizontal-scroll"
    assert scroll_id not in wrapped.child_sizes
    assert unbreakable.child_sizes[scroll_id][0] < path_field[2][0]
    assert (
        unbreakable.child_window_flags[scroll_id]
        & unbreakable.WindowFlags_.horizontal_scrollbar
    )


def test_options_reset_to_defaults_remains_unsaved_until_save(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "presentation:\n"
        "  show_fps: true\n"
        "live_edit:\n"
        "  weather:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    document = _perf_settings_document(config_path)
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        show_fps=True,
        native_dit_disabled_for_live_edit=True,
        settings_document=document,
    )
    state._open_options()
    reset_imgui = _FakeImGui()
    reset_imgui.clicked_buttons.add("RESET TO DEFAULTS")

    state.draw(reset_imgui)

    assert state._options_draft == document.defaults
    assert document.settings.presentation.show_fps
    assert state.show_fps
    assert "show_fps: true" in config_path.read_text(encoding="utf-8")

    unsaved_imgui = _FakeImGui()
    state.draw(unsaved_imgui)
    unsaved_labels = {label for label, _size in unsaved_imgui.button_sizes}
    assert "EXIT WITHOUT SAVING" in unsaved_labels
    assert (
        "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES"
        not in unsaved_imgui.windows["Crazy Robotaxi - Options"]
    )

    save_imgui = _FakeImGui()
    save_imgui.clicked_buttons.add("SAVE")
    state.draw(save_imgui)

    assert document.settings == document.defaults
    assert not state.show_fps
    assert "presentation:" not in config_path.read_text(encoding="utf-8")
    saved_imgui = _FakeImGui()
    state.draw(saved_imgui)
    assert (
        "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES"
        not in saved_imgui.windows["Crazy Robotaxi - Options"]
    )


def test_options_save_persists_and_applies_presentation_setting(
    tmp_path: Path,
) -> None:
    document = _settings_document(tmp_path / "config.yaml")
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=document,
    )
    state._open_options()
    state._options_category = "presentation"
    imgui = _FakeImGui()
    imgui.checkbox_values["##presentation.show_fps"] = True
    imgui.clicked_buttons.add("SAVE")

    state.draw(imgui)

    assert state._menu_stage == "options"
    assert state.show_fps
    assert "show_fps: true" in document.path.read_text(encoding="utf-8")
    assert state._settings_notice == f"SAVED {document.path}"
    assert not state._settings_restart_notice
    options_lines = imgui.windows["Crazy Robotaxi - Options"]
    assert "Show Fps:" in options_lines
    assert not any("RESTART REQUIRED" in line for line in options_lines)

    saved_imgui = _FakeImGui()
    state.draw(saved_imgui)
    saved_labels = {label for label, _size in saved_imgui.button_sizes}
    assert state._settings_notice in saved_imgui.windows["Crazy Robotaxi - Options"]
    assert "EXIT" in saved_labels
    assert "EXIT WITHOUT SAVING" not in saved_labels

    exit_imgui = _FakeImGui()
    exit_imgui.clicked_buttons.add("EXIT")
    state.draw(exit_imgui)
    assert state._menu_stage == "mode"

    menu_imgui = _FakeImGui()
    state.draw(menu_imgui)
    menu_lines = menu_imgui.windows["Crazy Robotaxi - Select Game Mode"]
    assert state._settings_notice not in menu_lines
    assert not any("RESTART REQUIRED" in line for line in menu_lines)


def test_options_discard_does_not_write_or_apply_changes(tmp_path: Path) -> None:
    document = _settings_document(tmp_path / "config.yaml")
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=document,
    )
    state._open_options()
    state._options_category = "presentation"
    imgui = _FakeImGui()
    imgui.checkbox_values["##presentation.show_fps"] = True
    imgui.clicked_buttons.add("EXIT WITHOUT SAVING")

    state.draw(imgui)

    assert state._menu_stage == "mode"
    assert not state.show_fps
    assert not document.path.exists()
    labels = {label for label, _size in imgui.button_sizes}
    assert "EXIT WITHOUT SAVING" in labels
    assert "EXIT" not in labels


def test_options_save_notice_expires_five_seconds_after_latest_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr("crazy_robotaxi.ui.time.monotonic", lambda: now[0])
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    state._open_options()

    first_save = _FakeImGui()
    first_save.clicked_buttons.add("SAVE")
    state.draw(first_save)

    now[0] = 104.0
    second_save = _FakeImGui()
    second_save.clicked_buttons.add("SAVE")
    state.draw(second_save)

    now[0] = 105.0
    visible_imgui = _FakeImGui()
    state.draw(visible_imgui)
    assert state._settings_notice in visible_imgui.windows["Crazy Robotaxi - Options"]

    now[0] = 109.0
    expired_imgui = _FakeImGui()
    state.draw(expired_imgui)
    assert not state._settings_notice
    assert not any(
        line.startswith("SAVED ")
        for line in expired_imgui.windows["Crazy Robotaxi - Options"]
    )


def test_options_identifies_restart_required_changes(tmp_path: Path) -> None:
    document = _settings_document(tmp_path / "config.yaml")
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=document,
    )
    state._open_options()
    state._options_category = "runtime"
    imgui = _FakeImGui()
    imgui.input_values["##runtime.prewarm_blocks"] = "9"

    state.draw(imgui)

    assert (
        "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT"
        in imgui.windows["Crazy Robotaxi - Options"]
    )
    assert not any(
        line.startswith("SETTINGS REQUIRING RESTART:")
        for line in imgui.windows["Crazy Robotaxi - Options"]
    )
    assert "Prewarm Blocks:" in imgui.windows["Crazy Robotaxi - Options"]

    imgui.clicked_buttons.add("SAVE")
    state.draw(imgui)

    assert state._menu_stage == "options"
    assert state._settings_notice == f"SAVED {document.path}"
    assert (
        state._settings_restart_notice == "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT"
    )
    assert state._settings_requiring_restart == ("runtime.prewarm_blocks",)

    saved_imgui = _FakeImGui()
    state.draw(saved_imgui)
    assert state._settings_notice in saved_imgui.windows["Crazy Robotaxi - Options"]
    assert (
        state._settings_restart_notice
        in saved_imgui.windows["Crazy Robotaxi - Options"]
    )
    saved_labels = {label for label, _size in saved_imgui.button_sizes}
    assert "EXIT" in saved_labels
    assert "EXIT WITHOUT SAVING" not in saved_labels

    exit_imgui = _FakeImGui()
    exit_imgui.clicked_buttons.add("EXIT")
    state.draw(exit_imgui)
    menu_imgui = _FakeImGui()
    state.draw(menu_imgui)
    menu_lines = menu_imgui.windows["Crazy Robotaxi - Select Game Mode"]
    assert state._settings_notice not in menu_lines
    assert state._settings_restart_notice in menu_lines
    assert not any(
        line.startswith("SETTINGS REQUIRING RESTART:") for line in menu_lines
    )


def test_options_can_show_code_only_restart_setting_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crazy_robotaxi.ui._SHOW_RESTART_REQUIRED_SETTINGS",
        True,
    )
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    state._open_options()
    state._options_category = "runtime"
    imgui = _FakeImGui()
    imgui.input_values["##runtime.prewarm_blocks"] = "9"

    state.draw(imgui)

    assert (
        "SETTINGS REQUIRING RESTART: runtime.prewarm_blocks"
        in imgui.windows["Crazy Robotaxi - Options"]
    )


def test_native_dit_notices_reflect_menu_context(
    tmp_path: Path,
) -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        native_dit_disabled_for_live_edit=True,
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    notice = "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES"

    menu_imgui = _FakeImGui()
    state.draw(menu_imgui)
    assert notice in menu_imgui.windows["Crazy Robotaxi - Select Game Mode"]

    state._open_options()
    options_imgui = _FakeImGui()
    state.draw(options_imgui)
    options_lines = options_imgui.windows["Crazy Robotaxi - Options"]
    assert notice not in options_lines


def test_saving_live_edit_that_disables_native_dit_shows_notice_before_restart(
    tmp_path: Path,
) -> None:
    document = _perf_settings_document(tmp_path / "config.yaml")
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=document,
    )
    state._open_options()
    state._options_category = "live_edit"
    imgui = _FakeImGui()
    imgui.checkbox_values["##live_edit.weather.enabled"] = True

    state.draw(imgui)

    pending_lines = imgui.windows["Crazy Robotaxi - Options"]
    assert "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT" in pending_lines
    assert "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES" in pending_lines

    reverted_imgui = _FakeImGui()
    reverted_imgui.checkbox_values["##live_edit.weather.enabled"] = False
    state.draw(reverted_imgui)
    reverted_lines = reverted_imgui.windows["Crazy Robotaxi - Options"]
    assert "RESTART REQUIRED FOR SETTINGS TO TAKE EFFECT" not in reverted_lines
    assert (
        "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES" not in reverted_lines
    )

    save_imgui = _FakeImGui()
    save_imgui.checkbox_values["##live_edit.weather.enabled"] = True
    save_imgui.clicked_buttons.add("SAVE")
    state.draw(save_imgui)

    assert state._menu_stage == "options"
    saved_imgui = _FakeImGui()
    state.draw(saved_imgui)
    assert (
        "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES"
        in saved_imgui.windows["Crazy Robotaxi - Options"]
    )
    assert (saved_imgui.Col_.text, _SAVED_NOTICE_RGBA) in (
        saved_imgui.pushed_style_colors
    )
    assert (saved_imgui.Col_.text, _RESTART_NOTICE_RGBA) in (
        saved_imgui.pushed_style_colors
    )
    assert (saved_imgui.Col_.text, _NATIVE_DIT_NOTICE_RGBA) in (
        saved_imgui.pushed_style_colors
    )

    exit_imgui = _FakeImGui()
    exit_imgui.clicked_buttons.add("EXIT")
    state.draw(exit_imgui)
    menu_imgui = _FakeImGui()
    state.draw(menu_imgui)
    assert (
        "NATIVE DIT ACCELERATION DISABLED FOR LIVE-EDIT FEATURES"
        not in menu_imgui.windows["Crazy Robotaxi - Select Game Mode"]
    )
