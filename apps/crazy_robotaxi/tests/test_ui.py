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
from typing import Any

import numpy as np
import pytest
import torch
from crazy_robotaxi.game_selection import GameMapOption, GameSelection
from crazy_robotaxi.high_scores import HighScoreEntry, RaceTimeEntry
from crazy_robotaxi.race import RaceGameSnapshot, RaceSessionState
from crazy_robotaxi.rules import (
    TaxiGameSnapshot,
    TaxiSessionState,
    project_taxi_markers_to_camera,
)
from crazy_robotaxi.ui import (
    _BEV_WAYPOINT_ALPHA,
    CrazyRobotaxiImGuiUILoop,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import draw_waypoints, project_waypoints
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
    WindowFlags_ = SimpleNamespace(
        no_move=1,
        no_resize=2,
        no_collapse=4,
        no_saved_settings=8,
        no_title_bar=16,
        no_background=32,
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
    )
    TableFlags_ = SimpleNamespace(
        row_bg=1,
        borders_inner_h=2,
        no_saved_settings=4,
        sizing_stretch_prop=8,
        scroll_y=16,
    )
    TableColumnFlags_ = SimpleNamespace(width_fixed=1, width_stretch=2)
    TableBgTarget_ = SimpleNamespace(row_bg1=1)

    def __init__(self) -> None:
        self.windows: dict[str, list[str]] = {}
        self.text_fonts: list[tuple[str, object, float]] = []
        self.dummies: list[tuple[float, float]] = []
        self.current_window: str | None = None
        self.next_window_position = (0.0, 0.0)
        self.next_window_size = (640.0, 360.0)
        self.cursor_x = 8.0
        self.input_value = ""
        self.submit_input = False
        self.click_submit = False
        self.clicked_buttons: set[str] = set()
        self.buttons: list[str] = []
        self.button_sizes: list[tuple[str, tuple[float, float] | None]] = []
        self.background_draw_list = _FakeDrawList()
        self.window_flags: dict[str, int] = {}
        self.tables: dict[str, list[list[str]]] = {}
        self.table_columns: dict[str, list[str]] = {}
        self.highlighted_rows: list[int] = []
        self.current_table: str | None = None
        self.current_table_column = 0
        self.default_font = object()
        self.current_font = self.default_font
        self.current_font_size = 14.0
        self.font_stack: list[tuple[object, float]] = []
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

    def set_next_window_pos(self, position, condition) -> None:
        self.next_window_position = position
        del condition

    def set_next_window_size(self, size, condition) -> None:
        self.next_window_size = size
        del condition

    def set_next_window_bg_alpha(self, alpha) -> None:
        del alpha

    def begin(self, title: str, *, flags: int) -> bool:
        self.current_window = title
        self.windows.setdefault(title, [])
        self.window_flags[title] = flags
        return True

    def end(self) -> None:
        self.current_window = None

    def begin_child(self, child_id: str, size: object) -> bool:
        del child_id, size
        return True

    def end_child(self) -> None:
        return

    def text(self, value: str) -> None:
        assert self.current_window is not None
        self.windows[self.current_window].append(value)
        self.text_fonts.append((value, self.current_font, self.current_font_size))
        if self.current_table is not None:
            rows = self.tables[self.current_table]
            while len(rows[-1]) <= self.current_table_column:
                rows[-1].append("")
            rows[-1][self.current_table_column] = value

    def get_window_pos(self) -> tuple[float, float]:
        return self.next_window_position

    def get_window_size(self) -> tuple[float, float]:
        return self.next_window_size

    def get_cursor_pos_x(self) -> float:
        return self.cursor_x

    def set_cursor_pos_x(self, value: float) -> None:
        self.cursor_x = value

    def get_content_region_avail(self) -> tuple[float, float]:
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
        del label, value, flags
        return self.submit_input, self.input_value

    def button(self, label: str, size: tuple[float, float] | None = None) -> bool:
        self.buttons.append(label)
        self.button_sizes.append((label, size))
        submit = self.click_submit and label in {"SAVE SCORE", "SAVE TIME"}
        return submit or label in self.clicked_buttons

    def begin_disabled(self) -> None:
        return

    def end_disabled(self) -> None:
        return

    def begin_table(
        self,
        table_id: str,
        columns: int,
        *,
        flags: int,
        outer_size: object,
    ) -> bool:
        del columns, flags, outer_size
        self.current_table = table_id
        self.tables[table_id] = []
        self.table_columns[table_id] = []
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

    def table_set_bg_color(self, target: int, color: int) -> None:
        del target, color
        assert self.current_table is not None
        self.highlighted_rows.append(len(self.tables[self.current_table]))


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
    state = TaxiHudState(640, 360, _calibration())
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

    assert not imgui.windows
    overlay_text = [
        args[-1] for name, args in imgui.background_draw_list.commands if name == "text"
    ]
    assert "GAME 42.5s  PICKUP  25m  SCORE 1200  HIGH 9000" in overlay_text
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


def test_selection_menus_use_arcade_card_layout() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
        variant="default",
        race_course_ids=("downtown-sprint",),
    )
    state = TaxiHudState(640, 540, _calibration(), map_options=(option,))
    imgui = _FakeImGui()

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
        "Crazy Robotaxi — Select Game Mode",
        "Crazy Robotaxi — Select Map",
        "Crazy Robotaxi — Select Race Course",
    ):
        assert imgui.window_flags[title] & imgui.WindowFlags_.no_title_bar
    button_sizes = dict(imgui.button_sizes)
    assert button_sizes["TAXI"] == button_sizes["RACE"]
    for label in ("TAXI", "Test City##map-0", "DOWNTOWN SPRINT##course-0"):
        size = button_sizes[label]
        assert size is not None and size[0] > 0.0
    assert [command for command, _args in imgui.background_draw_list.commands].count(
        "rect_filled"
    ) == 3


def test_startup_menu_selects_taxi_mode_then_map_through_v2_message() -> None:
    option = GameMapOption(
        map_id="test-city",
        name="Test City",
        path=Path("test-city.robotaxi.yaml"),
        variant="default",
        race_course_ids=("downtown-sprint",),
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
    assert "Crazy Robotaxi — Select Game Mode" in imgui.windows
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
        variant="default",
        race_course_ids=("downtown-sprint",),
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
        variant="default",
        race_course_ids=("downtown-sprint",),
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
        variant="default",
        race_course_ids=("downtown-sprint",),
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

    state.consume_input_events(UserInputEvents([pressed]))
    assert state._menu_stage == "mode"
    assert state._selected_game_mode is None

    state.consume_input_events(UserInputEvents([pressed]))
    assert state._menu_stage == "loading"
    assert state._loading_status == "EXITING GAME"
    model_loop._run_message_batch()
    assert model_loop.state.exit_requested


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
    assert "A [X]" in diagnostics[0]
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
    state = TaxiHudState(640, 540, _calibration())
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
    assert "PLAY AGAIN" in imgui.buttons
    assert "R  RESTART   ·   ESC  MAP" in imgui.windows["Game Over"]


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
