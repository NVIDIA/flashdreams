# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's V2 Dear ImGui UI loop."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from crazy_robotaxi.rules import (
    TaxiGameSnapshot,
    TaxiSessionState,
    project_taxi_markers_to_camera,
)
from crazy_robotaxi.ui import (
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


class _FakeDrawList:
    def __init__(self) -> None:
        self.commands: list[tuple[str, tuple[Any, ...]]] = []

    def add_line(self, *args: Any) -> None:
        self.commands.append(("line", args))

    def add_circle(self, *args: Any) -> None:
        self.commands.append(("circle", args))

    def add_circle_filled(self, *args: Any) -> None:
        self.commands.append(("circle_filled", args))

    def add_rect(self, *args: Any) -> None:
        self.commands.append(("rect", args))

    def add_rect_filled(self, *args: Any) -> None:
        self.commands.append(("rect_filled", args))

    def add_text(self, *args: Any) -> None:
        self.commands.append(("text", args))


class _FakeImGui:
    Cond_ = SimpleNamespace(always=1)
    WindowFlags_ = SimpleNamespace(
        no_move=1,
        no_resize=2,
        no_collapse=4,
        no_saved_settings=8,
    )
    InputTextFlags_ = SimpleNamespace(enter_returns_true=1)

    def __init__(self) -> None:
        self.windows: dict[str, list[str]] = {}
        self.images: list[tuple[str, np.ndarray, tuple[float, float]]] = []
        self.current_window: str | None = None
        self.input_value = ""
        self.submit_input = False
        self.click_submit = False
        self.background_draw_list = _FakeDrawList()

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

    def get_background_draw_list(self) -> _FakeDrawList:
        return self.background_draw_list

    def set_next_window_pos(self, position, condition) -> None:
        del position, condition

    def set_next_window_size(self, size, condition) -> None:
        del size, condition

    def set_next_window_bg_alpha(self, alpha) -> None:
        del alpha

    def begin(self, title: str, *, flags: int) -> bool:
        del flags
        self.current_window = title
        self.windows.setdefault(title, [])
        return True

    def end(self) -> None:
        self.current_window = None

    def text(self, value: str) -> None:
        assert self.current_window is not None
        self.windows[self.current_window].append(value)

    def image(
        self,
        key: str,
        pixels: np.ndarray,
        *,
        size: tuple[float, float],
    ) -> None:
        self.images.append((key, pixels, size))

    def input_text(self, label: str, value: str, *, flags: int):
        del label, value, flags
        return self.submit_input, self.input_value

    def button(self, label: str) -> bool:
        del label
        return self.click_submit

    def begin_disabled(self) -> None:
        return

    def end_disabled(self) -> None:
        return


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


def test_hud_frames_are_immutable_messages_keyed_to_video_storage() -> None:
    video = torch.zeros(2, 3, 96, 160)
    snapshots = (_snapshot(), _snapshot())
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

    frames = build_hud_frames(video, snapshots, poses)

    assert [frame.frame_key for frame in frames] == [
        video[index].data_ptr() for index in range(2)
    ]
    assert frames[0].snapshot is snapshots[0]
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
        input_transition_count=4,
        input_ignored_event_count=2,
        input_dropped_transition_count=1,
    )

    assert [frame.transition_timestamp_us for frame in frames] == [100, 200]
    assert frames[0].input_transition_count == 4
    assert frames[0].input_ignored_event_count == 2
    assert frames[0].input_dropped_transition_count == 1


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


def test_imgui_ui_loop_draws_waypoints_and_bev_in_the_ui_overlay() -> None:
    width, height = 160, 96
    video = torch.full((1, 3, height, width), -0.5)
    bev = torch.full((1, 3, 32, 32), 0.5)
    hud_state = TaxiHudState(width, height, _calibration())
    hud_state.publish(
        build_hud_frames(
            video,
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
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

    assert changed
    assert result.output.shape == (1, 3, height, width)
    assert result.output.dtype is torch.float32
    assert hud_state._current is not None
    assert "SCORE  001200    HIGH  009000" in renderer.ui.windows["Crazy Robotaxi"]
    assert "TARGET AHEAD  0 deg" in renderer.ui.windows["Navigation"]
    assert len(renderer.ui.images) == 1
    image_key, pixels, image_size = renderer.ui.images[0]
    assert image_key == "crazy_robotaxi_bev"
    assert pixels.shape == (32, 32, 3)
    assert np.all(pixels == 191)
    assert image_size == (32.0, 32.0)
    assert renderer.ui.background_draw_list.commands
    assert torch.all(result.output == video)

    cached_waypoints = hud_state._waypoint_projections
    cached_bev = hud_state._bev_pixels
    loop.step(1, UserInputEvents([]))
    assert hud_state._waypoint_projections is cached_waypoints
    assert hud_state._bev_pixels is cached_bev

    loop.reset()
    assert hud_state._current is None
    assert hud_state._waypoint_projections == ()
    assert hud_state._bev_pixels is None
    assert renderer.reset_count == 1


def test_hud_draws_immediate_imgui_windows() -> None:
    state = TaxiHudState(160, 96, _calibration())
    state.publish(
        build_hud_frames(
            torch.zeros(1, 3, 96, 160),
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state._current = next(iter(state._frames.values()))
    imgui = _FakeImGui()

    state.draw(imgui)

    assert set(imgui.windows) == {"Crazy Robotaxi", "Navigation"}
    assert "SCORE  001200    HIGH  009000" in imgui.windows["Crazy Robotaxi"]


def test_hud_animates_prepresentation_warmup_status() -> None:
    state = TaxiHudState(160, 96, _calibration())
    state.set_loading_status("WARMING WORLD MODEL  2/4")
    imgui = _FakeImGui()

    state.draw(imgui, ui_tick=30)

    lines = imgui.windows["Crazy Robotaxi"]
    assert lines[0] == "WARMING WORLD MODEL  2/4..."
    assert lines[1].startswith("ELAPSED  ")


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
            input_transition_count=1,
            input_ignored_event_count=2,
            input_dropped_transition_count=0,
        )
    )

    state.select_presented_frame(video[0])
    imgui = _FakeImGui()
    state.draw(imgui)

    assert state._latest_input_latency_ms is not None
    diagnostics = imgui.windows["Input Latency"]
    assert "A [X]" in diagnostics[0]
    assert "UI TO MODEL FRAME" in diagnostics[1]
    assert "TRANSITIONS  1" in diagnostics[2]
    assert "IGNORED  2" in diagnostics[2]

    state.reset()
    assert not state._profile_pressed
    assert state._latest_input_latency_ms is None


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
