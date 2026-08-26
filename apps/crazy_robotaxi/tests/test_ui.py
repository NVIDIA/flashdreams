# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's V2 SlangPy UI loop."""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import slangpy as spy
import torch
from crazy_robotaxi.rules import (
    TaxiGameSnapshot,
    TaxiSessionState,
    project_taxi_markers_to_camera,
)
from crazy_robotaxi.ui import (
    CrazyRobotaxiSlangPyUILoop,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import (
    _paint_lines,
    render_bev_overlay,
    render_waypoint_layers,
)
from omnidreams_game_engine.types import CameraCalibration

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
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


def _legacy_line_pixels(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int,
    height: int,
    width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta_x = float(end[0] - start[0])
    delta_y = float(end[1] - start[1])
    sample_count = max(2, int(math.ceil(max(abs(delta_x), abs(delta_y)))) + 1)
    center_x = np.rint(np.linspace(start[0], end[0], sample_count)).astype(np.int32)
    center_y = np.rint(np.linspace(start[1], end[1], sample_count)).astype(np.int32)
    radius = max(0.0, (float(width_px) - 1.0) / 2.0)
    x_parts = []
    y_parts = []
    integer_radius = int(math.ceil(radius))
    for offset_y in range(-integer_radius, integer_radius + 1):
        for offset_x in range(-integer_radius, integer_radius + 1):
            if offset_x * offset_x + offset_y * offset_y > radius * radius:
                continue
            x = center_x + offset_x
            y = center_y + offset_y
            selected = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x_parts.append(x[selected])
            y_parts.append(y[selected])
    return np.concatenate(x_parts), np.concatenate(y_parts)


class _Widget:
    def __init__(
        self,
        parent: object,
        label: str = "",
        *,
        value: Any = None,
        callback: Any = None,
        position: tuple[float, float] = (0.0, 0.0),
        size: tuple[float, float] = (0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        del parent, kwargs
        self.label = label
        self.title = label
        self.text = label
        self.value = value
        self.callback = callback
        self.position = position
        self.size = size
        self.visible = True
        self.enabled = True


class _FakeUI:
    Window = _Widget
    Text = _Widget
    InputText = _Widget
    Button = _Widget
    InputTextFlags = SimpleNamespace(enter_returns_true=1)

    def __init__(self) -> None:
        self.screen = object()


class _Renderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ui = _FakeUI()
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


def test_waypoints_are_rendered_as_frame_aligned_world_layers() -> None:
    poses = np.eye(4, dtype=np.float32)[None]

    layers = render_waypoint_layers(
        (_snapshot(),),
        poses,
        _calibration(),
        width=160,
        height=96,
        device="cpu",
    )

    assert layers.shape == (1, 4, 96, 160)
    assert layers.dtype is torch.float32
    assert torch.any(layers[:, 3] == 1.0)

    terminal = render_waypoint_layers(
        (_snapshot(session_state="awaiting_name"),),
        poses,
        _calibration(),
        width=160,
        height=96,
        device="cpu",
    )
    assert torch.count_nonzero(terminal[:, 3]) == 0


def test_batched_line_rasterizer_matches_the_original_geometry() -> None:
    lines = (
        ((-4.25, 10.5), (35.75, 22.5)),
        ((30.5, -8.0), (2.5, 28.0)),
        ((5.0, 5.0), (5.0, 5.0)),
        ((18.25, 29.75), (45.5, 2.25)),
    )
    expected = np.full((32, 40), -1, dtype=np.int8)
    for start, end in lines:
        x, y = _legacy_line_pixels(start, end, width=40, height=32, width_px=7)
        expected[y, x] = 3
    actual = np.full_like(expected, -1)

    _paint_lines(actual, lines, 40, 32, 7, 3)

    np.testing.assert_array_equal(actual, expected)


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


def test_bev_is_rendered_as_frame_aligned_rgba_overlay() -> None:
    overlay = render_bev_overlay(
        torch.full((1, 3, 32, 32), 0.5),
        width=160,
        height=96,
    )

    assert overlay.shape == (1, 4, 96, 160)
    assert overlay.dtype is torch.float32
    assert torch.count_nonzero(overlay[:, 3, :56, :]) == 0
    assert overlay[0, 3, 56, 120] == 1.0
    assert overlay[0, 3, 58, 122] == pytest.approx(0.82)
    assert torch.all(overlay[0, :3, 58, 122] == 0.5)


def test_slangpy_ui_loop_composites_world_waypoints_then_bev_and_ui() -> None:
    width, height = 160, 96
    video = torch.full((1, 3, height, width), -0.5)
    bev_overlay = render_bev_overlay(
        torch.full((1, 3, 32, 32), 0.5),
        width=width,
        height=height,
    )
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
            StepResult(0, bev_overlay, 1, VideoTensorLayout.tchw),
        ],
    )
    changed, _ = presentation.advance(0)
    renderer = _Renderer(width, height)
    loop = CrazyRobotaxiSlangPyUILoop(
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
    assert hud_state._widgets is not None
    assert hud_state._widgets.score.text == "SCORE  001200    HIGH  009000"
    assert hud_state._widgets.navigation.text == "TARGET AHEAD  0 deg"
    assert not hasattr(hud_state._widgets, "marker_windows")
    assert result.output[0, 0, 48, 80] != -0.5
    assert result.output[0, 0, 58, 122] == pytest.approx(0.32)
    assert torch.any(result.output != video)

    cached_waypoints = hud_state._waypoint_layer
    loop.step(1, UserInputEvents([]))
    assert hud_state._waypoint_layer is cached_waypoints

    loop.reset()
    assert hud_state._current is None
    assert hud_state._waypoint_layer is None
    assert renderer.reset_count == 1


def test_hud_builds_real_slangpy_imgui_widgets_without_a_renderer() -> None:
    state = TaxiHudState(160, 96, _calibration())
    state.publish(
        build_hud_frames(
            torch.zeros(1, 3, 96, 160),
            (_snapshot(),),
            np.eye(4, dtype=np.float32)[None],
        )
    )
    state._current = next(iter(state._frames.values()))
    ui = SimpleNamespace(
        screen=None,
        Window=spy.ui.Window,
        Text=spy.ui.Text,
        InputText=spy.ui.InputText,
        Button=spy.ui.Button,
        InputTextFlags=spy.ui.InputTextFlags,
    )

    state.draw(ui)

    assert state._widgets is not None
    assert isinstance(state._widgets.status_window, spy.ui.Window)
    assert isinstance(state._widgets.name_input, spy.ui.InputText)
    assert state._widgets.score.text == "SCORE  001200    HIGH  009000"


def test_hud_animates_prepresentation_warmup_status() -> None:
    state = TaxiHudState(160, 96, _calibration())
    state.set_loading_status("WARMING WORLD MODEL  2/4")
    ui = SimpleNamespace(
        screen=None,
        Window=spy.ui.Window,
        Text=spy.ui.Text,
        InputText=spy.ui.InputText,
        Button=spy.ui.Button,
        InputTextFlags=spy.ui.InputTextFlags,
    )

    state.draw(ui, ui_tick=30)

    assert state._widgets is not None
    assert state._widgets.status_window.visible
    assert state._widgets.score.text == "WARMING WORLD MODEL  2/4..."
    assert state._widgets.time.text.startswith("ELAPSED  ")


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
                UserInputEvent(
                    timestamp=np.uint64(100),
                    event_data=KeyboardUserInputEventData(
                        key="ArrowLeft",
                        state=KeyboardInputState.PRESSED,
                    ),
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
    state.draw(_FakeUI())

    assert state._latest_input_latency_ms is not None
    assert state._widgets is not None
    assert state._widgets.input_window is not None
    input_state = state._widgets.input_state
    input_latency = state._widgets.input_latency
    assert input_state is not None
    assert input_latency is not None
    assert "A [X]" in input_state.text
    assert "UI TO MODEL FRAME" in input_latency.text
    assert "TRANSITIONS  1" in input_latency.text
    assert "IGNORED  2" in input_latency.text

    state.reset()
    assert not state._profile_pressed
    assert state._latest_input_latency_ms is None


def test_input_latency_widgets_are_absent_by_default() -> None:
    state = TaxiHudState(160, 96, _calibration())

    state.draw(_FakeUI())

    assert state._widgets is not None
    assert state._widgets.input_window is None
    assert state._widgets.input_state is None
    assert state._widgets.input_latency is None


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

    state._submit_name(" DRIVER 7 ")
    state._submit_name("SECOND")

    assert model_loop.state.names == []
    model_loop._run_message_batch()
    assert model_loop.state.names == ["DRIVER 7"]
    assert state._submission_pending
