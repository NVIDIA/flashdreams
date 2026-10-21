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

from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu
CAMERAS = ALL_CAMERAS[:7]


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
