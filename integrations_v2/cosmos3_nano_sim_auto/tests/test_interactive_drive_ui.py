# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for live sensor channels, HUD wiring, and driving controls."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import interactive_drive.core as core
import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.apps.interactive_drive.adapter import create_app
from cosmos3_nano_sim_auto.apps.interactive_drive.coverage import OUTSIDE_MAP_WARNING
from cosmos3_nano_sim_auto.apps.interactive_drive.ui import (
    MultiviewDriveModelLoop,
    MultiviewDriveSession,
    MultiviewDriveUILoop,
)
from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import CAMERAS as ALL_CAMERAS
from interactive_drive.types import FrameChunk, PresentedFrame, VehicleState

from apps.interactive_drive.tests.test_multiview import RecordingUI
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu
CAMERAS = ALL_CAMERAS[:7]


class CockpitUI(RecordingUI):
    Col_ = Mock()
    StyleVar_ = Mock()
    ImVec4 = staticmethod(lambda *rgba: rgba)
    WindowFlags_ = SimpleNamespace(
        **vars(RecordingUI.WindowFlags_), no_scrollbar=16, no_scroll_with_mouse=32
    )

    def __init__(self):
        super().__init__()
        self.colors = 0
        self.variables = 0
        self.style = SimpleNamespace(font_scale_main=1.0)
        self.drawing = Mock()
        self.uvs = {}

    def get_window_draw_list(self):
        return self.drawing

    def get_font(self):
        return None

    def set_cursor_pos(self, position):
        self.cursor_x, self.cursor_y = position.x, position.y

    def invisible_button(self, label, size):
        self.rectangles[label] = (self.cursor_x, self.cursor_y, size.x, size.y)
        return label in self.clicked

    def button(self, label, *args):
        return label in self.clicked

    def begin_popup(self, label):
        return False

    def image(self, key, pixels, *, size, **kwargs):
        super().image(key, pixels, size=size)
        self.uvs[key] = (kwargs["uv0"], kwargs["uv1"])

    def get_style(self):
        return self.style

    def push_style_color(self, *args):
        self.colors += 1

    def pop_style_color(self, count=1):
        self.colors -= count

    def push_style_var(self, *args):
        self.variables += 1

    def pop_style_var(self, count=1):
        self.variables -= count


def _session(path, cameras, *args):
    app = create_app()
    app.init(
        [
            "--scene",
            str(path),
            "--camera",
            ",".join(cameras),
            *args,
        ]
    )
    desc = app.session_desc()
    assert desc is not None
    session = app.create_session(desc)
    assert isinstance(session, MultiviewDriveSession)
    session.init()
    assert isinstance(session.model_loop, MultiviewDriveModelLoop)
    state = session.model_loop.state
    state.config = replace(
        state.config, app=replace(state.config.app, world_model_device="cpu")
    )
    return session


def _chunk(views):
    vehicle = VehicleState(0, 0, 0, 0, 0, 0)
    frames = tuple(
        PresentedFrame(
            timestamp_us=t * 33_333,
            view_rgb_host_uint8=tuple(
                np.full((4, 6, 3), 10 * t + v, dtype=np.uint8) for v in range(views)
            ),
            view_model_rgb_host_uint8=tuple(
                np.full((4, 6, 3), 100 + 10 * t + v, dtype=np.uint8)
                for v in range(views)
            ),
            depth_host_f32=None,
        )
        for t in range(2)
    )
    return FrameChunk(frames, vehicle, "stub")


@pytest.mark.parametrize("cameras", [CAMERAS[:1], tuple(reversed(CAMERAS))])
@pytest.mark.parametrize("lidar", [False, True])
@pytest.mark.parametrize("view", ["rgb", "hdmap"])
def test_live_step_emits_synchronized_sensors_in_cli_order(
    tmp_path, monkeypatch, cameras, lidar, view
):
    args = ["--view", view, "--present-camera", cameras[-1]]
    if lidar:
        args += ["--lidar-checkpoint", str(tmp_path / "codec")]
    session = _session(tmp_path, cameras, *args)
    ui = session.ui_loop
    assert isinstance(ui, MultiviewDriveUILoop)
    assert tuple(c.name for c in ui.state.cameras) == cameras
    assert ui.state.visible_count == len(cameras)
    assert ui.state.focus_channel == len(cameras) - 1
    assert ui.state.bearing_layout
    assert tuple(v.name for v in ui.state.auxiliary_views) == (
        ("lidar",) if lidar else ()
    )
    assert ui.state.view_modes == ("rgb", "hdmap")

    chunk = replace(_chunk(len(cameras) + int(lidar)), warning=OUTSIDE_MAP_WARNING)
    model = session.model_loop
    model.state.scene = object()
    model.state.vehicle = chunk.boundary_state_after_chunk
    model.state.backend = SimpleNamespace(
        initial_chunk_frames=2,
        chunk_frames=2,
        render_first_chunk=lambda _: chunk,
        finalize=lambda: {"stub": 1.0},
    )
    monkeypatch.setattr(
        core,
        "sample_chunk_trajectory",
        lambda **_: SimpleNamespace(
            boundary_state_after_chunk=chunk.boundary_state_after_chunk,
            timestamps_us=np.array([f.timestamp_us for f in chunk.frames]),
        ),
    )
    results = model.step(0, UserInputEvents([]))
    assert len(results) == len(cameras) + int(lidar)
    for index, result in enumerate(results):
        assert result.step_index == 0
        assert result.frame_count == 2
        assert result.metrics == {"stub": 1.0}
        output = result.read_output()
        assert output.shape == (2, 3, 4, 6)
        offset = (100 if view == "rgb" else 0) + index
        assert output[:, 0, 0, 0].tolist() == [offset, offset + 10]
    ui._run_message_batch()
    assert ui.state.telemetry.blocks_generated == 1
    assert ui.state.telemetry.frames_in_chunk == 2
    assert ui.state.telemetry.warning == OUTSIDE_MAP_WARNING
    imgui = Mock()
    ui._draw_drive_hud(imgui)
    imgui.text_wrapped.assert_has_calls(
        [call("MAP COVERAGE WARNING"), call(OUTSIDE_MAP_WARNING)]
    )
    imgui.push_style_color.assert_called_once()
    imgui.pop_style_color.assert_called_once_with()

    # A recovered chunk clears the persistent warning through the same channel.
    chunk = replace(chunk, warning=None)
    model.state.backend.render_next_chunk = lambda _: chunk
    model.step(1, UserInputEvents([]))
    ui._run_message_batch()
    assert ui.state.telemetry.warning is None
    imgui.reset_mock()
    ui._draw_drive_hud(imgui)
    assert call("MAP COVERAGE WARNING") not in imgui.text_wrapped.call_args_list
    assert call(OUTSIDE_MAP_WARNING) not in imgui.text_wrapped.call_args_list
    imgui.push_style_color.assert_not_called()
    imgui.pop_style_color.assert_not_called()


@pytest.mark.parametrize("prompt", [None, "A quiet residential street."])
@pytest.mark.parametrize("drive_key", ["w", "s"])
@pytest.mark.parametrize("restart", ["button", "r", "R"])
def test_live_controls_toggle_hdmap_and_restart_without_losing_prompt(
    tmp_path, prompt, drive_key, restart
):
    args = [] if prompt is None else ["--prompt", prompt]
    session = _session(tmp_path, CAMERAS, *args)
    ui, model = session.ui_loop, session.model_loop
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=np.uint64(0), key=key, state=KeyboardInputState.PRESSED
            )
            for key in (drive_key, "a")
        ]
    )
    ui._apply_inputs(events)
    model._apply_events(events)
    assert ui.state.drive_input.command() == model._command()
    assert model._command().throttle == 1
    assert model._command().brake == 0
    assert model._command().reverse is (drive_key == "s")
    assert model._command().steer != 0

    imgui = Mock()
    imgui.combo.side_effect = lambda label, index, options: (False, index)
    imgui.button.side_effect = lambda label: label == "View: RGB"
    ui._draw_view_controls(imgui)
    assert ui.state.view_mode == "hdmap"
    assert model.state.view_mode == "rgb"
    model._run_message_batch()
    assert model.state.view_mode == "hdmap"
    if restart == "button":
        ui._restart()
    else:
        model._apply_events(
            UserInputEvents(
                [
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(0),
                        key=restart,
                        state=KeyboardInputState.PRESSED,
                    )
                ]
            )
        )
    model._run_message_batch()
    model.state.agree_on_queued()
    assert model.state.reset_pending
    assert model.state.pending_prompt == (prompt if restart == "button" else None)
    assert model.state.config.app.prompt_override == prompt
    assert ui.state.focus_channel == -1


@pytest.mark.parametrize("focused", [False, True])
def test_no_ui_keeps_one_mosaic_or_selected_camera(tmp_path, focused):
    args = ["--no-ui"]
    if focused:
        args += ["--present-camera", CAMERAS[-1]]
    session = _session(tmp_path, CAMERAS, *args)
    assert session._registered_ui_loop is None
    model = session.model_loop
    model.state.desc = replace(model.state.desc, video_width=18, video_height=12)
    chunk = _chunk(len(CAMERAS))
    outputs = model._frame_chunk_tensors(chunk)
    assert len(outputs) == 1
    expected = core._frame_chunk_tensor(
        chunk,
        "rgb",
        layout=model.state.config.app.presentation,
        output_size=(12, 18),
        output_device="cpu",
    )
    assert torch.equal(outputs[0], expected)


def test_recorded_scene_can_select_fractional_fps_without_rounding(tmp_path):
    app = create_app()
    app.init(["--scene", str(tmp_path), "--fps", "32.7"])
    assert app._config is not None
    assert app._config.app.chunk.fps == 32.7
    assert app.session_desc().frames_per_second_for_step == 32.7


@pytest.mark.parametrize("cameras", [CAMERAS[:1], CAMERAS, tuple(reversed(CAMERAS))])
@pytest.mark.parametrize("size", [(1280, 704), (2496, 1440)])
@pytest.mark.parametrize("lidar", [False, True])
def test_cockpit_keeps_all_labeled_streams_and_reserves_right_sidebar(
    tmp_path, cameras, size, lidar
):
    args = ["--lidar-checkpoint", str(tmp_path / "codec")] if lidar else []
    session = _session(tmp_path, cameras, *args)
    loop = session.ui_loop
    loop.get_ui_loop_size = Mock(return_value=size)
    frames = [
        torch.full((3, 18, 32), i, dtype=torch.uint8)
        for i in range(len(cameras) + int(lidar))
    ]
    loop.presented_model_frame = Mock(
        side_effect=lambda i: frames[i]
        if i < len(frames)
        else torch.zeros(3, 512, 512, dtype=torch.uint8)
    )
    ui = CockpitUI()
    ui.selectable = Mock(wraps=ui.selectable)
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert ui.panels == 0
    assert ui.colors == ui.variables == 0
    assert {key for key in ui.images if key.startswith("stream-")} == {
        f"stream-{i}" for i in range(len(frames))
    }
    metadata = (*loop.state.cameras, *loop.state.auxiliary_views)
    main = cameras.index("camera_front_wide_120fov")
    main_rect = ui.rectangles[f"camera-{main}"]
    map_x, map_y = ui.image_positions["drive-minimap"]
    map_w, map_h = ui.image_sizes["drive-minimap"]
    assert map_x > main_rect[0] + main_rect[2]
    assert map_y > main_rect[1]
    assert map_x + map_w <= size[0]
    assert map_y + map_h <= size[1]
    thumbs = []
    for i, view in enumerate(metadata):
        assert torch.equal(ui.images[f"stream-{i}"], frames[i].permute(1, 2, 0))
        w, h = ui.image_sizes[f"stream-{i}"]
        uv0, uv1 = ui.uvs[f"stream-{i}"]
        assert (32 * (uv1.x - uv0.x)) / (18 * (uv1.y - uv0.y)) == pytest.approx(w / h)
        assert any(
            c.args[-1] == view.label.upper() for c in ui.drawing.add_text.call_args_list
        )
        if i != main:
            rect = ui.rectangles[f"camera-{i}"]
            assert rect[1] > main_rect[1] + main_rect[3]
            assert rect[0] + rect[2] < map_x
            thumbs.append(rect)
    assert len({r[1] for r in thumbs}) <= 1
    assert main_rect[2] * main_rect[3] > sum(r[2] * r[3] for r in thumbs)
    for left, right in zip(thumbs, thumbs[1:]):
        assert left[0] + left[2] < right[0]


def test_cockpit_switches_main_camera_and_updates_pedals_without_generation(tmp_path):
    session = _session(tmp_path, CAMERAS)
    loop = session.ui_loop
    loop.get_ui_loop_size = Mock(return_value=(1920, 1080))
    loop.presented_model_frame = Mock(
        return_value=torch.zeros(3, 18, 32, dtype=torch.uint8)
    )
    ui = CockpitUI()
    ui.clicked.add("camera-1")
    loop.step_ui(
        ui,
        0,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(0), key=key, state=KeyboardInputState.PRESSED
                )
                for key in ("a", "w")
            ]
        ),
    )
    assert loop.state.focus_channel == 1
    assert any(
        c.args[-1] == "THROTTLE  100%" for c in ui.drawing.add_text.call_args_list
    )
    pressed_spokes = [c.args[1] for c in ui.drawing.add_line.call_args_list]
    next_ui = CockpitUI()
    loop.step_ui(
        next_ui,
        1,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(1), key=key, state=KeyboardInputState.RELEASED
                )
                for key in ("a", "w")
            ]
        ),
    )
    assert next_ui.rectangles["camera-1"][1] < next_ui.rectangles["camera-0"][1]
    assert any(
        c.args[-1] == "THROTTLE  0%" for c in next_ui.drawing.add_text.call_args_list
    )
    assert [
        c.args[1] for c in next_ui.drawing.add_line.call_args_list
    ] != pressed_spokes
    next_ui.clicked.add("Front Wide")
    loop.step_ui(next_ui, 2, UserInputEvents([]))
    assert loop.state.focus_channel == -1
    assert next_ui.rectangles["camera-0"][1] < next_ui.rectangles["camera-1"][1]


@pytest.mark.parametrize("lidar", [False, True])
def test_world_scenario_bev_has_its_own_synchronized_channel(tmp_path, lidar):
    args = ["--lidar-checkpoint", str(tmp_path / "codec")] if lidar else []
    session = _session(tmp_path, CAMERAS, *args)
    model, loop = session.model_loop, session.ui_loop
    count = len(CAMERAS) + int(lidar)
    chunk = _chunk(count)
    chunk = replace(
        chunk,
        frames=tuple(
            replace(frame, bev_host_uint8=np.full((8, 8, 3), 30 + i, dtype=np.uint8))
            for i, frame in enumerate(chunk.frames)
        ),
    )
    streams = model._frame_chunk_tensors(chunk)
    assert len(streams) == count
    native_map = model._bev_chunk_tensor(chunk)
    assert native_map.shape == (2, 3, 8, 8)
    assert native_map[:, 0, 0, 0].tolist() == [30, 31]
    loop.presented_model_frame = lambda i: ([s[0] for s in streams] + [native_map[0]])[
        i
    ]
    ui = CockpitUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert torch.equal(ui.images["drive-minimap"], native_map[0].permute(1, 2, 0))
    assert len([key for key in ui.images if key.startswith("stream-")]) == count
    assert not hasattr(loop.state, "map_background")


def test_world_scenario_bev_reuses_native_scene_palette_and_heading():
    from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import (
        DrawnCamera,
        LudusControls,
    )
    from ludus_renderer.torch.ops import CAMERA_TYPE_BEV

    raw = (
        torch.arange(2 * 8 * 8 * 4, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(2, 8, 8, 4)
    )
    context = Mock(needs_vflip=False)
    context.render.return_value = raw
    renderer = LudusControls(
        context,
        21,
        [
            DrawnCamera(
                "front", None, (8, 8), 1.0, np.repeat(np.eye(4)[None], 2, axis=0)
            )
        ],
        start_us=0,
        fps=30,
        widths={5: 3.0},
        device="cpu",
    )
    rig = torch.eye(4).repeat(2, 1, 1)
    rig[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rig[1, :3, 3] = torch.tensor([2.0, 3.0, 0.0])
    image = renderer.render_bev(rig, torch.tensor([0, 66666]), width=8, height=8)
    assert torch.equal(image, raw[..., :3])
    call_args = context.render.call_args.kwargs
    assert call_args["scene_ids"].tolist() == [21, 21]
    assert call_args["camera_type_ids"].tolist() == [CAMERA_TYPE_BEV] * 2
    assert call_args["timestamps_us"].tolist() == [0, 33333]
    assert [renderer.scene_time_us(i) for i in (0, 1, 2, 100)] == [
        0,
        33333,
        33333,
        33333,
    ]
    center = call_args["camera_poses"] @ rig[:, :, 3:4]
    torch.testing.assert_close(
        center[:, :, 0], torch.tensor([[75.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    )
    forward = rig[:, :, 3] + 5 * rig[:, :, 0]
    projected = (call_args["camera_poses"] @ forward[:, :, None])[:, :, 0]
    torch.testing.assert_close(projected[:, 2], torch.full((2,), 5.0))
    context.upload_scene.assert_not_called()
    context.upload_color_palette.assert_not_called()
    context.upload_width_table.assert_called_once_with({5: 3.0})


def test_web_cockpit_uses_camera_video_and_native_bev_without_imgui(tmp_path):
    from interactive_drive.app import InteractiveDriveWebUILoop

    session = _session(tmp_path, CAMERAS, "--web-ui", "--present-camera", CAMERAS[0])
    assert isinstance(session.ui_loop, InteractiveDriveWebUILoop)
    assert session.model_loop.state.config.web_ui
    assert session.model_loop.state.config.no_ui
    assert session.model_loop.state.config.app.bev.enabled
    assert (session.session_desc.video_width, session.session_desc.video_height) == (
        832,
        480,
    )


def test_bev_uses_original_ego_pool_camera_and_reuses_topology():
    from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import (
        DrawnCamera,
        LudusControls,
    )
    from interactive_drive.config import BevConfig
    from interactive_drive.rasterizer import (
        _bev_sensor_to_rig,
        _build_bev_camera,
        _build_bev_ego_car_pool,
    )
    from ludus_renderer import PRIM_EGO_OBSTACLE, TimestampedScene

    context = Mock(needs_vflip=False)
    context.render.return_value = torch.zeros(2, 8, 8, 4, dtype=torch.uint8)
    renderer = LudusControls(
        context,
        21,
        [
            DrawnCamera(
                "front", None, (8, 8), 1.0, np.repeat(np.eye(4)[None], 2, axis=0)
            )
        ],
        start_us=0,
        fps=30,
        widths={},
        device="cpu",
        scene_data=TimestampedScene([], [], []),
    )
    rig = torch.eye(4).repeat(2, 1, 1)
    rig[1, :3, 3] = torch.tensor([2.0, 3.0, 0.0])
    times = torch.tensor([50_000, 83_333])
    dims = (4.8, 2.0, 1.6)
    renderer.render_bev(rig, times, width=8, height=8, ego_dimensions_lwh=dims)
    context.replace_scene.assert_called_once()
    replacement = context.replace_scene.call_args.args[1]
    pool = replacement.cube_pools[-1]
    original = _build_bev_ego_car_pool(
        rig_poses=rig, timestamps_us=times, dimensions_lwh=dims, track_capacity=2
    )
    assert pool.prim_type_id == PRIM_EGO_OBSTACLE
    torch.testing.assert_close(pool.translations, original.translations)
    torch.testing.assert_close(pool.colors, original.colors)
    query = context.render.call_args.kwargs
    assert query["timestamps_us"].tolist() == times.tolist()
    torch.testing.assert_close(
        query["camera_poses"],
        torch.linalg.inv(
            rig @ _bev_sensor_to_rig(height_m=75, device=torch.device("cpu"))
        ),
    )
    actual_camera = context.upload_cameras.call_args.args[0][0]
    original_camera = _build_bev_camera(
        BevConfig(width=8, height=8), torch.device("cpu")
    )
    for field in ("principal_point", "image_size", "pixels_per_meter", "depth_max"):
        torch.testing.assert_close(
            getattr(actual_camera, field), getattr(original_camera, field)
        )
    renderer.render_bev(
        rig[:1], times[:1] + 100_000, width=8, height=8, ego_dimensions_lwh=dims,
        rig_ground_clearance_m=0.84,
    )
    context.replace_scene.assert_called_once()
    context.update_cube_pool.assert_called_once()
    updated = context.update_cube_pool.call_args.args[2]
    assert updated.translations.shape[0] == 2
    torch.testing.assert_close(updated.translations[:, 2], torch.full((2,), -0.04))
    assert updated.timestamps_us[0] == times[0] + 100_000
    context.update_cube_pool.return_value = False
    with pytest.raises(RuntimeError, match="ego-car"):
        renderer.render_bev(rig, times, width=8, height=8, ego_dimensions_lwh=dims)
