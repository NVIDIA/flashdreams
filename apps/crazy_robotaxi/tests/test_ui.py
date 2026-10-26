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
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from crazy_robotaxi.controls import (
    ControlsConfig,
    InputBinding,
    load_controls_documents,
)
from crazy_robotaxi.free_roam import FreeRoamSnapshot
from crazy_robotaxi.game_selection import (
    GameMapOption,
    GameRaceCourseOption,
    GameSelection,
)
from crazy_robotaxi.high_scores import HighScoreEntry, RaceTimeEntry
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditMapContextConfig,
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
    CrazyRobotaxiImGuiUILoop,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import draw_waypoints, project_waypoints
from omnidreams_game_engine.types import CameraCalibration

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    FullscreenEnabledOptimalWindowSizeUserInputEvent,
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_taxi_ui_requests_fullscreen_dimensions_before_resizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loop = CrazyRobotaxiImGuiUILoop(width=320, height=240)
    document = _settings_document(tmp_path / "config.yaml")
    loop.state = TaxiHudState(
        320, 240, None, presentation_size=(320, 240), settings_document=document
    )
    loop._ui_loop_requests = None
    loop._presentation_manager = Mock()
    loop._presentation_manager.presented_frames.return_value = ()
    draw = TaxiHudState.draw
    monkeypatch.setattr(TaxiHudState, "draw", Mock())
    loop.step_ui(object(), 0, UserInputEvents([]))
    assert loop.get_ui_loop_size() == (320, 240)

    loop.step_ui(
        object(),
        1,
        UserInputEvents(
            [
                FullscreenEnabledOptimalWindowSizeUserInputEvent(
                    timestamp=np.uint64(0), width=1920, height=1080
                )
            ]
        ),
    )
    assert loop.get_ui_loop_size() == (320, 240)
    assert (loop.state.width, loop.state.height) == (320, 240)
    assert loop.state.presentation_size == (1920, 1080)
    requests = loop.flush_ui_loop_requests()
    assert requests is not None and requests.new_window_size == (1920, 1080)

    loop.step_ui(object(), 2, UserInputEvents([]))
    assert loop.get_ui_loop_size() == (1920, 1080)
    assert (loop.state.width, loop.state.height) == (1920, 1080)
    assert loop.flush_ui_loop_requests() is None

    loop.state._open_options()
    loop.state._save_options()
    assert loop.state.presentation_size == (1920, 1080)
    assert "width: 1920" in document.path.read_text(encoding="utf-8")
    assert "height: 1080" in document.path.read_text(encoding="utf-8")

    loop.state._options_category = "presentation"
    imgui = _FakeImGui()
    imgui.input_values["##presentation.width"] = "640"
    imgui.input_values["##presentation.height"] = "480"
    imgui.clicked_buttons.add("SAVE")
    draw(loop.state, imgui)

    saved = _settings_document(document.path).settings.presentation
    assert (saved.width, saved.height) == (640, 480)
    assert loop.state.presentation_size == (640, 480)
    assert (loop.state.width, loop.state.height) == (1920, 1080)
    assert not loop.state._settings_restart_notice
    loop.step_ui(object(), 3, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()
    assert requests is not None and requests.new_window_size == (640, 480)
    loop.step_ui(object(), 4, UserInputEvents([]))
    assert loop.get_ui_loop_size() == (640, 480)
    loop.close()


@dataclass(frozen=True)
class _SettingsTransformer:
    native_dit_acceleration: str = "required"


@dataclass(frozen=True)
class _SettingsDiffusionModel:
    transformer: _SettingsTransformer = field(default_factory=_SettingsTransformer)


@dataclass(frozen=True)
class _SettingsPipeline:
    name: str = "test-preset"
    diffusion_model: _SettingsDiffusionModel = field(
        default_factory=_SettingsDiffusionModel
    )


@dataclass(frozen=True)
class _SettingsEncoder:
    precision: str = "bfloat16"
    """Encoder precision."""


@dataclass(frozen=True)
class _SettingsPipelineWithEncoders:
    image_encoder: _SettingsEncoder = field(
        default_factory=_SettingsEncoder,
    )
    """First instance of the reusable encoder config."""

    encoder: _SettingsEncoder = field(
        default_factory=_SettingsEncoder,
    )
    """Second instance with its own parent field name."""


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
    def add_font_from_file_ttf(self, path: str, size: float) -> object:
        del path, size
        return object()


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
        popup_border_size=7,
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
        popup_bg=12,
        plot_histogram=13,
    )
    TableFlags_ = SimpleNamespace(
        row_bg=1,
        borders_inner_h=2,
        no_saved_settings=4,
        sizing_stretch_prop=8,
        scroll_y=16,
        sizing_stretch_same=64,
    )
    TableColumnFlags_ = SimpleNamespace(width_fixed=1, width_stretch=2)
    TableBgTarget_ = SimpleNamespace(row_bg1=1)

    def __init__(self) -> None:
        self.windows: dict[str, list[str]] = {}
        self.open_popups: set[str] = set()
        self.progress_values: list[float] = []
        self.current_window: str | None = None
        self.next_window_position = (0.0, 0.0)
        self.next_window_size = (640.0, 360.0)
        self.cursor_x = 8.0
        self.cursor_y = 8.0
        self.input_value = ""
        self.input_values: dict[str, str] = {}
        self.checkbox_values: dict[str, bool] = {}
        self.combo_indices: dict[str, int] = {}
        self.submit_input = False
        self.click_submit = False
        self.clicked_buttons: set[str] = set()
        self.buttons: list[str] = []
        self.images: list[tuple[str, np.ndarray, tuple[float, float]]] = []
        self.disabled_depth = 0
        self.disabled_buttons: list[str] = []
        self.background_draw_list = _FakeDrawList()
        self.window_flags: dict[str, int] = {}
        self.window_positions: dict[str, tuple[float, float]] = {}
        self.window_sizes: dict[str, tuple[float, float]] = {}
        self.child_sizes: dict[str, tuple[float, float]] = {}
        self.child_window_flags: dict[str, int] = {}
        self._child_size_stack: list[tuple[float, float]] = []
        self.current_child_size: tuple[float, float] | None = None
        self.last_item_rect_size = (0.0, 0.0)
        self.tables: dict[str, list[list[str]]] = {}
        self.table_columns: dict[str, list[str]] = {}
        self.table_outer_sizes: dict[str, tuple[float, float]] = {}
        self.table_column_indices: dict[str, list[int]] = {}
        self.highlighted_rows: list[int] = []
        self.current_table: str | None = None
        self.current_table_column = 0
        self.default_font = object()
        self.current_font = self.default_font
        self.current_font_size = 14.0
        self.font_stack: list[tuple[object, float]] = []
        self.fonts = _FakeFontAtlas()
        self.io = SimpleNamespace(fonts=self.fonts)
        self.hovered_items: set[str] = set()
        self.tooltips: list[str] = []
        self.wrap_positions: list[float] = []
        self._group_items: list[str] | None = None
        self._group_hovered = False
        self._tooltip_open = False

    def begin_group(self) -> None:
        assert self._group_items is None
        self._group_items = []

    def end_group(self) -> None:
        assert self._group_items is not None
        self._group_hovered = bool(self.hovered_items.intersection(self._group_items))
        self._group_items = None

    def begin_item_tooltip(self) -> bool:
        self._tooltip_open = self._group_hovered
        return self._tooltip_open

    def end_tooltip(self) -> None:
        assert self._tooltip_open
        self._tooltip_open = False

    def push_text_wrap_pos(self, position: float) -> None:
        self.wrap_positions.append(position)

    def pop_text_wrap_pos(self) -> None:
        self.wrap_positions.pop()

    def text_unformatted(self, value: str) -> None:
        assert self._tooltip_open
        assert self.wrap_positions[-1] > 0.0
        self.tooltips.append(value)

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
        del style_var, value

    def pop_style_var(self, count: int = 1) -> None:
        del count

    def push_style_color(self, color: int, value: object) -> None:
        del color, value

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
        self.window_positions[title] = self.next_window_position
        self.window_sizes[title] = self.next_window_size
        return True

    def end(self) -> None:
        self.current_window = None

    def is_popup_open(self, title: str) -> bool:
        return title in self.open_popups

    def open_popup(self, title: str) -> None:
        self.open_popups.add(title)

    def begin_popup_modal(self, title: str, *, flags: int) -> tuple[bool, None]:
        if not self.is_popup_open(title):
            return False, None
        return self.begin(title, flags=flags), None

    def close_current_popup(self) -> None:
        assert self.current_window is not None
        self.open_popups.remove(self.current_window)

    def end_popup(self) -> None:
        self.end()

    def set_item_default_focus(self) -> None:
        pass

    def progress_bar(self, fraction: float, size: Any, *, overlay: str) -> None:
        self.progress_values.append(fraction)

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
        if self._group_items is not None:
            self._group_items.append(value)
        self.windows[self.current_window].append(value)
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
            return (
                child_width
                if child_width > 0.0
                else max(1.0, float(self.next_window_size[0]) - 56.0),
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
        del size

    def separator(self) -> None:
        return

    def set_next_item_width(self, width: float) -> None:
        del width

    def input_text(self, label: str, value: str, *, flags: int):
        del flags
        if self._group_items is not None:
            self._group_items.append(label)
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
        del size, flags
        if self._group_items is not None:
            self._group_items.append(label)
        if label in self.input_values:
            return True, self.input_values[label]
        return False, value

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        if self._group_items is not None:
            self._group_items.append(label)
        if label in self.checkbox_values:
            return True, self.checkbox_values[label]
        return False, value

    def combo(self, label: str, index: int, options: list[str]) -> tuple[bool, int]:
        del options
        if self._group_items is not None:
            self._group_items.append(label)
        if label in self.combo_indices:
            return True, self.combo_indices[label]
        return False, index

    def button(self, label: str, size: tuple[float, float] | None = None) -> bool:
        self.buttons.append(label)
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
        self.table_outer_sizes[table_id] = outer_size
        self.table_column_indices[table_id] = []
        return True

    def end_table(self) -> None:
        self.current_table = None

    def table_setup_column(self, label: str, flags: int, width: float) -> None:
        del flags, width
        assert self.current_table is not None
        self.table_columns[self.current_table].append(label)

    def table_headers_row(self) -> None:
        return

    def table_next_row(self, *, min_row_height: float) -> None:
        del min_row_height
        assert self.current_table is not None
        self.tables[self.current_table].append([])
        self.current_table_column = 0

    def table_set_column_index(self, column: int) -> None:
        self.current_table_column = column
        assert self.current_table is not None
        self.table_column_indices[self.current_table].append(column)

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
    snapshots = (_snapshot(), replace(_snapshot(), score=2400))
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = 10.0
    expected_poses = poses.copy()
    statuses = (
        LiveEditHudStatus(coins_enabled=True, coins_collected=0),
        LiveEditHudStatus(coins_enabled=True, coins_collected=1),
    )
    frames = build_hud_frames(
        video,
        snapshots,
        poses,
        speeds_mps=(12.0, -3.0),
        transition_timestamps_us=(100, 200),
        live_edit_statuses=statuses,
        current_prompt="A taxi driving through a city.",
    )
    poses[:, 0, 3] = 999.0
    state = TaxiHudState(160, 96, _calibration())
    state.publish(frames)
    newer_video = torch.ones(1, 3, 96, 160)
    state.publish(
        build_hud_frames(
            newer_video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            current_prompt="A taxi driving through the countryside.",
        )
    )
    state.select_presented_frame(newer_video[0])

    for index in (1, 0):
        selected = state.select_presented_frame(video[index])
        assert selected is frames[index]
        assert selected.frame_key == video[index].data_ptr()
        assert selected.snapshot is snapshots[index]
        assert selected.speed_mps == (12.0, -3.0)[index]
        assert selected.transition_timestamp_us == (100, 200)[index]
        assert selected.live_edit_status == statuses[index]
        assert selected.current_prompt == "A taxi driving through a city."
        np.testing.assert_array_equal(selected.rig_pose_world, expected_poses[index])
        assert not selected.rig_pose_world.flags.writeable


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
    imgui.clicked_buttons = {
        "CYCLE STYLE (K)",
        "CYCLE WEATHER (V)",
        "TOGGLE COINS (C)",
        "SPAWN OBSTACLE (O)",
    }

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
    assert "STYLE BASE" in " ".join(" ".join(imgui.windows["Live Edit"]).split())
    blocked = _FakeImGui()
    blocked.clicked_buttons.add("CYCLE WEATHER (V)")
    state._draw_live_edit_card(
        blocked, LiveEditHudStatus(skin_name="comic", weather_name="clear")
    )
    model_loop._run_message_batch()

    assert "CYCLE WEATHER (V)" in blocked.disabled_buttons
    blocked_status = " ".join(" ".join(blocked.windows["Live Edit"]).split())
    assert "STYLE COMIC" in blocked_status
    assert "STYLE BASE" not in blocked_status
    assert model_loop.state.live_edit_actions == [
        "style",
        "weather",
        "coins",
        "obstacle",
    ]


def test_live_edit_mapping_location_and_button_visibility() -> None:
    live_edit = LiveEditConfig(
        style=LiveEditStyleConfig(enabled=True),
        weather=LiveEditWeatherConfig(enabled=True),
    )
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        live_edit=live_edit,
        live_edit_mapping_location="control hints",
    )
    card_imgui = _FakeImGui()

    state._draw_live_edit_card(
        card_imgui,
        LiveEditHudStatus(skin_name="base", weather_name="clear"),
    )

    assert card_imgui.buttons == ["CYCLE STYLE", "CYCLE WEATHER"]
    hints_imgui = _FakeImGui()
    state._draw_control_tooltips(hints_imgui)
    hints = [
        value for row in hints_imgui.tables["##gameplay-control-hints"] for value in row
    ]
    assert "CYCLE STYLE" in hints
    assert "K" in hints
    assert "CYCLE WEATHER" in hints
    assert "V" in hints

    state.show_live_edit_buttons = False
    hidden_imgui = _FakeImGui()
    state._draw_live_edit_card(
        hidden_imgui,
        LiveEditHudStatus(skin_name="base", weather_name="clear"),
    )
    assert hidden_imgui.buttons == []
    assert "LIVE EDIT" in hidden_imgui.windows["Live Edit"]


def test_live_edit_card_is_hidden_when_map_context_has_no_visible_content() -> None:
    state = TaxiHudState(
        640,
        540,
        _calibration(),
        live_edit=LiveEditConfig(
            dynamic_prompts=LiveEditMapContextConfig(enabled=True),
        ),
        show_live_edit_buttons=False,
    )
    imgui = _FakeImGui()

    state._draw_live_edit_card(imgui, LiveEditHudStatus())

    assert "Live Edit" not in imgui.windows


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

    assert len(projections) == 1
    assert projections[0].anchor_uv == pytest.approx((80.0, 48.0))
    marker_centers = [
        args[0]
        for name, args in imgui.background_draw_list.commands
        if name == "circle_filled"
    ]
    assert marker_centers == [pytest.approx((80.0, 48.0))]

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


@pytest.mark.parametrize("show_current_prompt", [False, True])
def test_current_prompt_overlay_is_configurable(show_current_prompt: bool) -> None:
    video = torch.zeros(1, 3, 360, 320)
    state = TaxiHudState(
        320,
        360,
        _calibration(),
        show_current_prompt=show_current_prompt,
    )
    state._menu_stage = "game"
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            current_prompt=(
                "A taxi driving through a wide city boulevard with buildings and trees."
            ),
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    assert ("Current Prompt" in imgui.windows) is show_current_prompt
    if show_current_prompt:
        assert len(imgui.windows["Current Prompt"]) > 1


def test_long_prompt_leaves_live_edit_overlays_visible() -> None:
    video = torch.zeros(1, 3, 360, 320)
    prompt = "long prompt " * 100
    state = TaxiHudState(
        320,
        360,
        _calibration(),
        show_current_prompt=True,
        live_edit=LiveEditConfig(coins=LiveEditCoinsConfig(enabled=True)),
    )
    state._menu_stage = "game"
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            live_edit_statuses=(
                LiveEditHudStatus(coins_enabled=True, coins_collected=3),
            ),
            current_prompt=prompt,
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    prompt_bottom = (
        imgui.window_positions["Current Prompt"][1]
        + imgui.window_sizes["Current Prompt"][1]
    )
    assert len(" ".join(imgui.windows["Current Prompt"])) < len(prompt)
    for window in ("Coin Counter", "Live Edit"):
        assert prompt_bottom <= imgui.window_positions[window][1] < state.height


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


def test_presentation_back_buffer_scales_to_ui_resolution() -> None:
    state = TaxiHudState(16, 12, _calibration())
    state._bev_rect = (8, 12, 4, 4)
    video = torch.full((3, 6, 8), -0.5, dtype=torch.bfloat16)
    bev = torch.full((4, 4, 4), 255, dtype=torch.uint8)

    output = state.composite_bev(video, bev)

    assert output.shape == (3, 12, 16)
    assert output.dtype is torch.float32
    torch.testing.assert_close(output[:, 0, 0], torch.full((3,), -0.5))
    torch.testing.assert_close(output[:, 8, 12], torch.ones(3))


def test_presentation_back_buffer_cache_is_invalidated_on_resize() -> None:
    state = TaxiHudState(4, 4, _calibration())
    video = torch.full((3, 4, 4), -0.5, dtype=torch.bfloat16)
    initial = state.composite_bev(video, None)

    state.resize(8, 6)
    resized = state.composite_bev(video, None)

    assert resized is not initial
    assert resized.shape == (3, 6, 8)


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
        args[:3]
        for name, args in imgui.background_draw_list.commands
        if name == "triangle_filled"
    ]
    assert triangles
    # A target straight ahead lies beyond the top edge of the ego-centered BEV.
    for tip, left, right in triangles:
        assert tip[0] == pytest.approx(48.0)
        assert 0.0 <= tip[1] < min(left[1], right[1]) < 48.0
        assert min(left[0], right[0]) < tip[0] < max(left[0], right[0])


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
                return_to_menu=(InputBinding("key", "n"), None),
                toggle_hints=(InputBinding("key", "j"), None),
            ),
        ),
    )
    state.publish(
        build_hud_frames(
            torch.zeros(1, 3, 360, 640),
            (replace(_snapshot(), event="pickup_complete"),),
            np.eye(4, dtype=np.float32)[None],
            speeds_mps=(12.0,),
            live_edit_statuses=(
                LiveEditHudStatus(coins_enabled=True, coins_collected=3),
            ),
        )
    )
    state._current = next(iter(state._frames.values()))
    state._menu_stage = "game"
    imgui = _FakeImGui()

    state.draw(imgui)

    hints = [
        row[index : index + 2]
        for row in imgui.tables["##gameplay-control-hints"]
        for index in range(0, len(row), 2)
    ]
    assert ["RESTART", "P"] in hints
    assert ["RETURN TO MENU", "N"] in hints
    assert ["HIDE CONTROLS", "J"] in hints
    overlay_text = [
        args[-1] for name, args in imgui.background_draw_list.commands if name == "text"
    ]
    status = " ".join(" ".join(overlay_text).split())
    assert "GAME 42.5s" in status
    assert "PICKUP 25m" in status
    assert "SCORE 1200" in status
    assert "HIGH 9000" in status
    assert "27" in overlay_text
    assert "mph" in overlay_text
    assert "PASSENGER PICKED UP" in overlay_text
    assert "COINS 3" in " ".join(" ".join(imgui.windows["Coin Counter"]).split())
    assert any(
        name == "triangle_filled" for name, _ in imgui.background_draw_list.commands
    )


def test_free_roam_hud_has_no_objective_markers() -> None:
    video = torch.zeros(1, 3, 360, 640)
    state = TaxiHudState(640, 360, _calibration())
    frame = build_hud_frames(
        video,
        (FreeRoamSnapshot(),),
        np.eye(4, dtype=np.float32)[None],
    )[0]
    state.publish((frame,))
    state._current = frame
    state._menu_stage = "game"
    imgui = _FakeImGui()

    state.draw(imgui)
    state._bev_rect = (0, 0, 96, 96)
    state._draw_bev_navigation(imgui, frame)

    text = [
        args[-1] for name, args in imgui.background_draw_list.commands if name == "text"
    ]
    assert "FREE-ROAM" in text
    assert not any(
        name == "triangle_filled" for name, _args in imgui.background_draw_list.commands
    )


def test_mode_card_scrolls_within_a_small_viewport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        settings_document=_settings_document(tmp_path / "config.yaml"),
    )
    imgui = _FakeImGui()
    constraints: list[tuple[tuple[float, float], tuple[float, float]]] = []
    monkeypatch.setattr(
        imgui,
        "set_next_window_size_constraints",
        lambda minimum, maximum: constraints.append((minimum, maximum)),
    )

    state.draw(imgui)

    assert constraints == [((1.0, 1.0), (612.0, 332.0))]
    flags = imgui.window_flags["Crazy Robotaxi - Select Game Mode"]
    assert not flags & imgui.WindowFlags_.no_scrollbar
    assert not flags & imgui.WindowFlags_.no_scroll_with_mouse
    assert "OPTIONS" in imgui.buttons
    assert "EXIT" in imgui.buttons


def test_race_map_grid_uses_filtered_positions_for_layout() -> None:
    race_course = (GameRaceCourseOption("course", "race-start"),)
    options = tuple(
        GameMapOption(
            map_id=f"map-{index}",
            name=f"Map {index}",
            path=Path(f"map-{index}.robotaxi.yaml"),
            race_courses=race_course if index % 2 else (),
        )
        for index in range(4)
    )
    state = TaxiHudState(900, 540, _calibration(), map_options=options)
    state._selected_game_mode = "race"
    state._menu_stage = "map"
    imgui = _FakeImGui()

    state.draw(imgui)

    assert imgui.table_column_indices["##map-grid"] == [0, 1]
    assert "Map 1##map-1" in imgui.buttons
    assert "Map 3##map-3" in imgui.buttons


@pytest.mark.parametrize("viewport_width", [640, 1200])
def test_three_column_selection_grids_preserve_natural_width(
    viewport_width: int,
) -> None:
    preview_path = Path("preview.jpg")
    courses = tuple(
        GameRaceCourseOption(
            f"course-{index}",
            f"spawn-{index}",
            preview_path,
        )
        for index in range(5)
    )
    options = tuple(
        GameMapOption(
            map_id=f"map-{index}",
            name=f"Map {index}",
            path=Path(f"map-{index}.robotaxi.yaml"),
            race_courses=courses,
            preview_image_path=preview_path,
        )
        for index in range(5)
    )
    state = TaxiHudState(viewport_width, 720, _calibration(), map_options=options)
    state._selection_preview_pixels[preview_path] = np.zeros(
        (90, 160, 3), dtype=np.uint8
    )
    imgui = _FakeImGui()

    state._selected_game_mode = "taxi"
    state._menu_stage = "map"
    state.draw(imgui)
    state._selected_game_mode = "race"
    state._selected_map_option = options[0]
    state._menu_stage = "course"
    state.draw(imgui)

    for child_id, table_id in (
        ("##map-options", "##map-grid"),
        ("##course-options", "##course-grid"),
    ):
        visible_width = imgui.child_sizes[child_id][0]
        natural_width = imgui.table_outer_sizes[table_id][0]
        horizontal_scroll = (
            imgui.child_window_flags[child_id] & imgui.WindowFlags_.horizontal_scrollbar
        )
        if viewport_width == 640:
            assert natural_width > visible_width
            assert horizontal_scroll
        else:
            assert natural_width == visible_width
            assert not horizontal_scroll


def test_controls_menu_edits_and_saves_one_device(tmp_path: Path) -> None:
    documents = load_controls_documents(tmp_path / "controls")
    for device in ("gamepad", "wheel"):
        documents[device].save(documents[device].settings)
    other_files = {
        documents[device].path: documents[device].path.read_bytes()
        for device in ("gamepad", "wheel")
    }
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
    open_imgui = _FakeImGui()
    open_imgui.clicked_buttons.add("CONTROLS")
    state.draw(open_imgui)
    assert state._menu_stage == "controls"

    controls_imgui = _FakeImGui()
    state.draw(controls_imgui)

    assert {"KEYBOARD", "GAMEPAD", "WHEEL", "BACK"} <= set(controls_imgui.buttons)
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
    reloaded = load_controls_documents(tmp_path / "controls")
    assert reloaded["keyboard"].settings.restart == (InputBinding("key", "p"), None)
    for path, contents in other_files.items():
        assert path.read_bytes() == contents
    assert state._settings_restart_notice
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
    imgui = _CursorBoundaryImGui()
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
    imgui = _CursorBoundaryImGui()
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


def test_free_roam_menu_selects_any_map_without_a_course() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
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
    imgui.clicked_buttons.add("FREE-ROAM")

    state.draw(imgui)

    assert state._menu_stage == "map"
    imgui.clicked_buttons = {"Test City##map-0"}
    state.draw(imgui)
    model_loop._run_message_batch()

    assert state._menu_stage == "loading"
    assert "FREE-ROAM MODE" in imgui.windows["Crazy Robotaxi - Select Map"]
    assert model_loop.state.selections == [
        GameSelection(mode="free-roam", map_option=option)
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


@pytest.mark.parametrize("allow_exit", (False, True))
def test_menu_back_uses_styled_gamepad_cancel_button(allow_exit: bool) -> None:
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        gamepad_button_style="Xbox",
        allow_exit=allow_exit,
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
    assert (
        "ESC / B - EXIT" in imgui.windows["Crazy Robotaxi - Select Game Mode"]
    ) is allow_exit


@pytest.mark.parametrize("allow_exit", (False, True))
def test_gameplay_return_to_map_uses_rebindable_controls(allow_exit: bool) -> None:
    defaults = ControlsConfig()
    controls = replace(
        defaults,
        keyboard=replace(
            defaults.keyboard,
            return_to_menu=(InputBinding("key", "m"), None),
        ),
    )
    state = TaxiHudState(
        640, 360, _calibration(), controls=controls, allow_exit=allow_exit
    )
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


@pytest.mark.parametrize("allow_exit", (False, True))
@pytest.mark.parametrize("exit_input", ("button", "keyboard", "gamepad"))
def test_mode_exit_respects_allow_exit(allow_exit: bool, exit_input: str) -> None:
    state = TaxiHudState(640, 360, _calibration(), allow_exit=allow_exit)
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()
    state.draw(imgui)

    assert ("EXIT" in imgui.buttons) is allow_exit
    assert (
        "ESC / B - EXIT" in imgui.windows["Crazy Robotaxi - Select Game Mode"]
    ) is allow_exit

    if exit_input == "button":
        imgui.clicked_buttons.add("EXIT")
        state.draw(imgui)
    elif exit_input == "keyboard":
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
    else:
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
    assert state._exit_confirmation_pending is allow_exit
    assert not state._exit_requested
    model_loop._run_message_batch()
    assert not model_loop.state.exit_requested

    imgui.clicked_buttons.clear()
    state.draw(imgui)
    assert imgui.is_popup_open("EXIT GAME?") is allow_exit
    imgui.clicked_buttons.add("EXIT GAME")
    state.draw(imgui)

    assert not imgui.is_popup_open("EXIT GAME?")
    assert not state._exit_confirmation_pending
    assert state._menu_stage == ("loading" if allow_exit else "mode")
    if allow_exit:
        assert state._loading_status == "EXITING GAME"
    assert state._exit_requested is allow_exit
    model_loop._run_message_batch()
    assert model_loop.state.exit_requested is allow_exit


def test_exit_confirmation_can_be_cancelled_and_reopened() -> None:
    state = TaxiHudState(640, 360, _calibration())
    imgui = _FakeImGui()
    imgui.clicked_buttons.add("EXIT")
    state.draw(imgui)
    assert imgui.is_popup_open("EXIT GAME?")
    imgui.clicked_buttons.clear()

    imgui.clicked_buttons.add("CANCEL")
    state.draw(imgui)
    assert not imgui.is_popup_open("EXIT GAME?")
    assert not state._exit_confirmation_pending
    assert not state._exit_requested
    assert state._menu_stage == "mode"

    imgui.clicked_buttons.clear()
    imgui.clicked_buttons.add("EXIT")
    state.draw(imgui)
    assert imgui.is_popup_open("EXIT GAME?")
    assert not state._exit_requested


@pytest.mark.parametrize("device", ("keyboard", "gamepad", "analog_gamepad"))
@pytest.mark.parametrize("allow_exit", (False, True))
def test_exit_requires_a_continuous_hold(
    monkeypatch: pytest.MonkeyPatch, device: str, allow_exit: bool
) -> None:
    now = 10.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    state = TaxiHudState(640, 360, _calibration(), allow_exit=allow_exit)
    model_loop = _SelectionLoop()
    model_loop.register_session_loop_objects(
        state=_SelectionState(),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    state.model_loop = model_loop
    imgui = _FakeImGui()

    def send(held: bool) -> None:
        if device == "keyboard":
            event = KeyboardUserInputEvent(
                timestamp=np.uint64(1),
                key="Escape",
                state=(
                    KeyboardInputState.PRESSED if held else KeyboardInputState.RELEASED
                ),
            )
        else:
            event = GamepadUserInputEvent(
                timestamp=np.uint64(1),
                pressed=(False, held),
                buttons=(0.0, float(held)) if device == "analog_gamepad" else (),
            )
        state.consume_input_events(UserInputEvents([event]))

    send(True)
    state.draw(imgui)
    assert state._exit_confirmation_pending is allow_exit
    now += 0.75
    send(True)
    state.draw(imgui)
    assert not state._exit_requested
    if allow_exit:
        assert imgui.progress_values[-1] == pytest.approx(0.5)
        assert "HOLD ESC / B TO EXIT" in imgui.windows["EXIT GAME?"]

    send(False)
    now += 5.0
    state.draw(imgui)
    assert not state._exit_requested
    assert state._exit_confirmation_pending is allow_exit
    if allow_exit:
        assert imgui.progress_values[-1] == 0.0

    send(True)
    now += 1.49
    state.draw(imgui)
    model_loop._run_message_batch()
    assert not state._exit_requested
    assert not model_loop.state.exit_requested
    now += 0.02
    state.draw(imgui)
    model_loop._run_message_batch()
    assert state._exit_requested is allow_exit
    assert model_loop.state.exit_requested is allow_exit


@pytest.mark.parametrize("interruption", ("focus_loss", "disconnect", "reset"))
def test_exit_hold_does_not_survive_input_interruption(
    monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    now = 10.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    state = TaxiHudState(640, 360, _calibration())
    state.consume_input_events(
        UserInputEvents(
            [GamepadUserInputEvent(timestamp=np.uint64(1), pressed=(False, True))]
        )
    )
    state.draw(_FakeImGui())
    now += 1.0
    if interruption == "reset":
        state.reset()
    else:
        event = (
            FocusUserInputEvent(timestamp=np.uint64(2), focused=False)
            if interruption == "focus_loss"
            else GamepadUserInputEvent(timestamp=np.uint64(2), action="disconnected")
        )
        state.consume_input_events(UserInputEvents([event]))
    now += 5.0
    state.draw(_FakeImGui())
    assert not state._exit_requested
    assert not state._exit_hold_started_at_s


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


def test_m_toggles_hdmap_view_only_during_gameplay() -> None:
    state = TaxiHudState(640, 360, _calibration())
    released = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="m",
        state=KeyboardInputState.RELEASED,
    )
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(2),
        key="M",
        state=KeyboardInputState.PRESSED,
    )

    state.consume_input_events(UserInputEvents([pressed]))
    assert not state.show_hdmap

    state.consume_input_events(UserInputEvents([released]))
    state._menu_stage = "game"
    state.consume_input_events(UserInputEvents([pressed]))
    assert state.show_hdmap

    state.consume_input_events(UserInputEvents([released]))
    state.consume_input_events(UserInputEvents([pressed]))
    assert not state.show_hdmap


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
    terminal_imgui = _FakeImGui()
    state._draw_terminal(
        terminal_imgui,
        _snapshot(session_state="leaderboard"),
    )
    assert (
        "START / MENU - RESTART   |   BACK / VIEW - MENU"
        in terminal_imgui.windows["Game Over"]
    )


@pytest.mark.parametrize("source", ("keyboard", "gamepad"))
def test_input_latency_profile_correlates_ui_event_with_model_frame(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    now_ns = 1_000_000_000
    monkeypatch.setattr("crazy_robotaxi.ui.time.monotonic_ns", lambda: now_ns)
    video = torch.zeros(1, 3, 96, 160)
    state = TaxiHudState(160, 96, _calibration(), profile_input_latency=True)
    event = (
        KeyboardUserInputEvent(
            timestamp=np.uint64(100),
            key="ArrowLeft",
            state=KeyboardInputState.PRESSED,
        )
        if source == "keyboard"
        else GamepadUserInputEvent(
            timestamp=np.uint64(100), action="state", axes=(0.25,)
        )
    )
    state.consume_input_events(UserInputEvents([event]))
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
            transition_timestamps_us=(100,),
        )
    )
    now_ns += 25_000_000
    state.select_presented_frame(video[0])
    assert state._latest_input_latency_ms == pytest.approx(25.0)

    imgui = _FakeImGui()
    state.draw(imgui)
    diagnostics = imgui.windows["Input Latency"]
    assert "UI TO MODEL FRAME" in diagnostics[1]
    if source == "keyboard":
        assert "A / LEFT ARROW [X]" in diagnostics[0]

    state.reset()
    assert not state._profile_pressed
    assert state._latest_input_latency_ms is None


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


def test_hidden_gameplay_hud_still_draws_results() -> None:
    state = TaxiHudState(640, 540, _calibration(), hud_enabled=False)
    video = torch.zeros(1, 3, 540, 640)
    state.publish(
        build_hud_frames(
            video,
            (_snapshot(session_state="leaderboard"),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    assert "Game Over" in imgui.windows


def test_race_results_card_draws_leaderboard_without_player_rank() -> None:
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
    assert imgui.highlighted_rows == []


@pytest.mark.parametrize(
    ("snapshot", "expected_rows", "save_label"),
    [
        (
            replace(
                _snapshot(session_state="awaiting_name"),
                high_score_rank=2,
                leaderboard=(HighScoreEntry("ACE", 2400, "2026-01-01T00:00:00Z"),),
            ),
            [["#1", "ACE", "2400"], ["#2", "", "1200"]],
            "SAVE SCORE",
        ),
        (
            replace(
                _race_snapshot(session_state="awaiting_name"),
                high_score_rank=1,
            ),
            [["#1", "", "0:42.345"]],
            "SAVE TIME",
        ),
        (
            replace(
                _snapshot(session_state="leaderboard"),
                leaderboard=tuple(
                    HighScoreEntry(f"DRIVER {index}", 10_000 - index, "")
                    for index in range(10)
                ),
            ),
            [
                [f"#{index + 1}", f"DRIVER {index}", str(10_000 - index)]
                for index in range(10)
            ],
            None,
        ),
    ],
)
def test_terminal_shows_pending_and_saved_results_within_viewport(
    snapshot: TaxiGameSnapshot | RaceGameSnapshot,
    expected_rows: list[list[str]],
    save_label: str | None,
) -> None:
    state = TaxiHudState(320, 180, _calibration())
    video = torch.zeros(1, 3, 180, 320)
    state.publish(
        build_hud_frames(
            video,
            (snapshot,),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state.select_presented_frame(video[0])
    imgui = _FakeImGui()

    state.draw(imgui)

    rows = [[value.strip() for value in row] for row in imgui.tables["##leaderboard"]]
    assert rows == expected_rows
    assert 0.0 < imgui.table_outer_sizes["##leaderboard"][1] <= state.height
    if save_label is not None:
        assert save_label in imgui.buttons
    assert imgui.highlighted_rows == (
        [] if snapshot.high_score_rank is None else [snapshot.high_score_rank]
    )


@pytest.mark.parametrize(
    ("session_state", "button"),
    [
        ("awaiting_name", "PLAY AGAIN"),
        ("leaderboard", "PLAY AGAIN"),
        ("leaderboard", "RETURN TO MENU"),
    ],
)
def test_terminal_result_actions(
    session_state: TaxiSessionState,
    button: str,
) -> None:
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
    imgui.clicked_buttons.add(button)

    state.draw(imgui)
    model_loop._run_message_batch()

    if button == "PLAY AGAIN":
        assert model_loop.state.restart_count == 1
    else:
        assert state._menu_stage == "map"
        assert model_loop.state.return_to_map_count == 1


def _settings_document(path: Path) -> SettingsDocument:
    return SettingsDocument.load(
        path,
        pipeline_config=_SettingsPipeline(),
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


@pytest.mark.parametrize("component", ("image_encoder", "encoder"))
def test_options_tooltips_derive_paths_for_reused_configs(
    tmp_path: Path, component: str
) -> None:
    document = SettingsDocument.load(
        tmp_path / "config.yaml",
        pipeline_config=_SettingsPipelineWithEncoders(),
        width=1280,
        height=704,
    )
    state = TaxiHudState(1280, 720, _calibration(), settings_document=document)
    state._open_options()
    state._options_category = "model"
    imgui = _FakeImGui()
    state.draw(imgui)
    assert imgui.tooltips == []
    imgui.hovered_items.add(f"##model.pipeline.{component}.precision")

    state.draw(imgui)

    assert imgui.tooltips == [
        f"Encoder precision.\n\nYAML: model.pipeline.{component}.precision"
    ]


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
    document = _settings_document(config_path)
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
    unsaved_labels = set(unsaved_imgui.buttons)
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
    category_imgui = _FakeImGui()
    category_imgui.clicked_buttons.add("PRESENTATION##options-category-presentation")
    state.draw(category_imgui)
    assert state._options_category == "presentation"
    imgui = _FakeImGui()
    imgui.checkbox_values["##presentation.show_fps"] = True
    imgui.checkbox_values["##presentation.show_live_edit_buttons"] = False
    imgui.combo_indices["##presentation.live_edit_mapping_location"] = 1
    imgui.checkbox_values["##presentation.show_current_prompt"] = True
    imgui.clicked_buttons.add("SAVE")

    state.draw(imgui)

    assert state._menu_stage == "options"
    assert state.show_fps
    assert not state.show_live_edit_buttons
    assert state.live_edit_mapping_location == "control hints"
    assert state.show_current_prompt
    saved = _settings_document(document.path).settings.presentation
    assert saved.show_fps
    assert not saved.show_live_edit_buttons
    assert saved.live_edit_mapping_location == "control hints"
    assert saved.show_current_prompt
    assert not state._settings_restart_notice
    saved_imgui = _FakeImGui()
    state.draw(saved_imgui)
    saved_labels = set(saved_imgui.buttons)
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
    assert not state._settings_requiring_restart


def test_options_rejects_incomplete_resolution_after_cli_override(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "presentation:\n  width: 1920\n  height: 1080\n",
        encoding="utf-8",
    )
    document = _settings_document(config_path)
    document.cli_overrides[("presentation", "width")] = 2560
    state = TaxiHudState(
        640,
        360,
        _calibration(),
        presentation_size=(2560, 1080),
        settings_document=document,
    )
    state._open_options()
    state._options_category = "presentation"
    imgui = _FakeImGui()
    imgui.input_values["##presentation.width"] = ""
    imgui.input_values["##presentation.height"] = ""
    imgui.clicked_buttons.add("SAVE")

    state.draw(imgui)

    assert state._options_error == "presentation width and height must be set together"
    assert document.settings.presentation.width == 1920
    assert document.settings.presentation.height == 1080
    assert state.presentation_size == (2560, 1080)
    assert config_path.read_text(encoding="utf-8") == (
        "presentation:\n  width: 1920\n  height: 1080\n"
    )


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
    labels = set(imgui.buttons)
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
    document = _settings_document(tmp_path / "config.yaml")
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
    assert state._settings_requiring_restart == ("live_edit.weather.enabled",)
    assert (
        state._settings_restart_notice
        in saved_imgui.windows["Crazy Robotaxi - Options"]
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

    assert (
        state._settings_restart_notice
        in menu_imgui.windows["Crazy Robotaxi - Select Game Mode"]
    )
