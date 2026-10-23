# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerSnapshot,
    ControllerStatusEvent,
)
from handdreams import ui as ui_module
from handdreams.controllers.quest_pose import OPENXR_LOCAL_GRIP_POSE_CONVENTION
from handdreams.scenes import ScenePreview
from handdreams.ui import (
    HandDreamsPhase,
    HandDreamsUILoop,
    HandDreamsUIState,
    HandDreamsUIStatus,
)

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    TouchUserInputEvent,
    XRControllerUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _controller(
    *,
    tracked: bool = True,
    trigger: float = 0.0,
    primary_button: bool = False,
    secondary_button: bool = False,
    battery: float | None = 0.75,
) -> ControllerSnapshot:
    return ControllerSnapshot(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        axes=(),
        trigger=trigger,
        grip=0.0,
        tracked=tracked,
        primary_button=primary_button,
        secondary_button=secondary_button,
        battery=battery,
    )


def _event(
    *,
    left: ControllerSnapshot | None = None,
    right: ControllerSnapshot | None = None,
    disconnected: bool = False,
) -> ControllerFrameEvent:
    if disconnected:
        return ControllerFrameEvent(timestamp=np.uint64(1), action="disconnected")
    return ControllerFrameEvent(
        timestamp=np.uint64(1),
        action="state",
        left=left if left is not None else _controller(),
        right=right if right is not None else _controller(),
        frame_id=1,
        client_capture_ns=1_000,
        pose_convention=OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    )


def _events(*events) -> UserInputEvents:
    return UserInputEvents(list(events))


@pytest.mark.parametrize(
    "primary,secondary", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("phase", [HandDreamsPhase.SELECTING, HandDreamsPhase.LIVE])
def test_left_face_buttons_do_not_navigate_scenes(primary, secondary, phase) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=phase,
            current_scene="scene-a" if phase is HandDreamsPhase.LIVE else "",
            scene_catalog=("scene-a", "scene-b"),
        ),
        stage_width=64,
        stage_height=32,
    )
    loop.session_desc = SessionDesc(video_width=64, video_height=32)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    setattr(loop, "presented_model_frame", lambda channel_index=0: None)
    pressed = _event(
        left=_controller(
            primary_button=primary,
            secondary_button=secondary,
            trigger=0.75,
            battery=0.4,
        )
    )
    observations = (
        pressed,
        pressed,
        _event(),
        pressed,
        _event(left=_controller(tracked=False)),
        pressed,
        _event(disconnected=True),
        pressed,
    )
    for index, observation in enumerate(observations):
        loop.step(index, _events(observation))
        assert loop.flush_ui_loop_requests() is None
        assert loop.state.status.phase is phase
        assert loop.state.status.current_scene == (
            "scene-a" if phase is HandDreamsPhase.LIVE else ""
        )
        assert loop.state.controller_frame is observation
    assert loop.state.status.left_tracked
    assert loop.state.status.left_trigger == pytest.approx(0.75)
    assert not loop.state.status.left_open
    assert loop.state.status.left_battery == pytest.approx(0.4)
    assert loop.state.status.right_tracked and loop.state.status.right_open


@pytest.mark.parametrize("trigger,opened", [(0.0, True), (0.2, True), (0.20001, False)])
def test_pair_updates_both_claws_and_batteries(trigger: float, opened: bool) -> None:
    state = HandDreamsUIState(open_claw_threshold=0.2)
    event = _event(
        left=_controller(trigger=trigger, battery=0.1),
        right=_controller(trigger=trigger, battery=None),
    )
    state.observe_events(_events(event))
    assert state.status.left_tracked and state.status.right_tracked
    assert state.status.left_open is opened
    assert state.status.right_open is opened
    assert state.status.left_trigger == trigger
    assert state.status.right_trigger == trigger
    assert state.status.left_battery == pytest.approx(0.1)
    assert state.status.right_battery is None
    state.update_status(phase=HandDreamsPhase.LIVE, current_scene="", scene_catalog=())
    assert ui_module.status_message(state.status) == "Live - low controller battery"


def test_disconnect_clears_both_hands() -> None:
    state = HandDreamsUIState()
    state.observe_events(_events(_event(left=_controller(trigger=1.0))))
    state.update_status(
        phase=HandDreamsPhase.CALIBRATING,
        current_scene="",
        scene_catalog=(),
        calibration_progress=0.5,
    )
    state.observe_events(_events(_event(disconnected=True)))
    assert not state.status.left_tracked and not state.status.right_tracked
    assert not state.status.left_open and not state.status.right_open
    assert state.status.left_battery is None and state.status.right_battery is None
    assert state.status.left_trigger == 0.0 and state.status.right_trigger == 0.0
    assert state.status.calibration_progress == 0.0


def test_ui_ignores_generic_xr_events() -> None:
    state = HandDreamsUIState()
    event = XRControllerUserInputEvent(timestamp=np.uint64(1), handedness="left")
    state.observe_events(UserInputEvents([event]))
    assert state.status == HandDreamsUIStatus()


def test_connection_status_arrives_before_poses_and_in_timestamp_order() -> None:
    state = HandDreamsUIState()
    detected = ControllerConnectionStatus(headset="Detected")
    wake = ControllerConnectionStatus(headset="Press the power button")
    state.observe_events(
        _events(
            ControllerStatusEvent(timestamp=np.uint64(2), status=wake),
            ControllerStatusEvent(timestamp=np.uint64(1), status=detected),
        )
    )
    assert state.connection_status is wake
    assert state.controller_frame is None
    assert not state.status.left_tracked and not state.status.right_tracked


def test_connection_status_is_independent_of_tracking_model_and_reset() -> None:
    connection = ControllerConnectionStatus(
        headset="Press the power button", web_client="Ready", active_input="Waiting"
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState()
    loop.state.observe_events(
        _events(
            ControllerStatusEvent(timestamp=np.uint64(1), status=connection),
            _event(),
            _event(disconnected=True),
        )
    )
    loop.state.update_status(
        phase=HandDreamsPhase.LIVE,
        current_scene="scene-a",
        scene_catalog=("scene-a", "scene-b"),
    )
    assert loop.state.connection_status is connection
    assert not loop.state.status.left_tracked
    loop.session_desc = SessionDesc(metadata={})
    loop._ui_loop_requests = None
    loop._request_scene_name("scene-b")
    assert loop.state.connection_status is connection
    loop.reset()
    loop.reset()
    assert loop.state.connection_status is connection
    assert loop.state.controller_frame is None


@pytest.mark.parametrize(
    "phase,video_available,draw_hud",
    [
        (HandDreamsPhase.SELECTING, False, False),
        (HandDreamsPhase.LOADING, False, False),
        (HandDreamsPhase.WAITING, False, False),
        (HandDreamsPhase.LIVE, False, False),
        (HandDreamsPhase.LIVE, True, True),
    ],
)
def test_live_hud_does_not_cover_connection_guidance(
    monkeypatch, phase, video_available, draw_hud
) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(phase=phase),
        connection_status=ControllerConnectionStatus(
            headset="Press the power button", web_client_url="https://teleop.example/"
        ),
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=360)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    calls = []
    monkeypatch.setattr(
        loop,
        "presented_model_frame",
        lambda: torch.zeros((3, 360, 640), dtype=torch.uint8)
        if video_available
        else None,
    )
    monkeypatch.setattr(
        ui_module, "_draw_controller_hud", lambda *_: calls.append("hud")
    )
    loop.step(0, _events())
    assert calls == (["hud"] if draw_hud else [])
    requests = loop.flush_ui_loop_requests()
    if draw_hud:
        assert requests is None
    else:
        assert requests is not None
        assert requests.links is not None
        assert [link.url for link in requests.links] == ["https://teleop.example/"]


@pytest.mark.parametrize("with_scenes", [False, True])
def test_connection_links_follow_stage_bounds_and_clear_during_setup(
    monkeypatch, with_scenes
) -> None:
    connection = ControllerConnectionStatus(
        headset="Press the power button",
        web_client_url="https://teleop.example/?serverIP=10.0.0.1&port=48322",
        certificate_url="https://10.0.0.1:48322/",
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(phase=HandDreamsPhase.SELECTING),
        connection_status=connection,
        stage_width=640,
        stage_height=360,
        scene_previews=(ScenePreview("scene-a", "Scene A"),) if with_scenes else (),
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=480)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    monkeypatch.setattr(loop, "presented_model_frame", lambda: None)
    result = loop.step(0, _events())[0]
    requests = loop.flush_ui_loop_requests()
    assert requests is not None and requests.links is not None
    assert len(requests.links) == 1
    assert result.read_output().shape[-2] == (480 if with_scenes else 360)
    for actual, stage in zip(
        requests.links, ui_module._connection_links(connection, 640, 360), strict=True
    ):
        left, top, right, bottom = stage.bounds
        scale = 360 / result.read_output().shape[-2]
        assert actual.url == stage.url
        assert actual.bounds == pytest.approx(
            (left, top * scale, right, bottom * scale)
        )
        assert actual.bounds[3] < scale

    # Setup removes the old regions; returning to selection restores them.
    loop.state.status = replace(loop.state.status, phase=HandDreamsPhase.ALIGNING)
    loop.step(1, _events())
    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert requests.links == ()
    loop.state.status = replace(loop.state.status, phase=HandDreamsPhase.SELECTING)
    loop.step(2, _events())
    requests = loop.flush_ui_loop_requests()
    assert requests is not None and requests.links is not None
    assert len(requests.links) == 1
    loop.state.connection_status = replace(connection, web_client_url="")
    loop.step(3, _events())
    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert requests.links == ()
    loop.step(4, _events())
    assert loop.flush_ui_loop_requests() is None


@pytest.mark.parametrize(
    "phase",
    [
        HandDreamsPhase.SELECTING,
        HandDreamsPhase.LOADING,
        HandDreamsPhase.WAITING,
        HandDreamsPhase.LIVE,
    ],
)
@pytest.mark.parametrize("lost_side", ["left", "right"])
def test_connection_guidance_and_links_follow_controller_tracking(
    monkeypatch, phase, lost_side
) -> None:
    connection = ControllerConnectionStatus(
        headset="Press the headset power button",
        web_client_url="https://teleop.example/",
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(phase=phase), connection_status=connection
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=360)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    monkeypatch.setattr(loop, "presented_model_frame", lambda: None)
    messages = []
    original = ui_module.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui_module.ImageDraw.ImageDraw, "text", text)
    transitions = [
        (_events(), True),
        (_events(_event()), False),
        (
            _events(
                _event(**cast(dict[str, Any], {lost_side: _controller(tracked=False)}))
            ),
            True,
        ),
        (_events(_event()), False),
        (_events(_event(disconnected=True)), True),
        (_events(_event()), False),
    ]
    for step, (events, visible) in enumerate(transitions):
        messages.clear()
        loop.step(step, events)
        requests = loop.flush_ui_loop_requests()
        assert requests is not None
        assert requests.links is not None
        assert len(requests.links) == int(visible)
        assert any(message.startswith("Headset: ") for message in messages) is visible
        assert ("Isaac Teleop Web Client: " in messages) is visible
        assert ("Open Link" in messages) is visible
        assert loop.state.connection_status is connection
        if phase is HandDreamsPhase.SELECTING and not visible:
            assert "Left: Tracked" in messages
            assert "Right: Tracked" in messages
            assert "Battery: 75%" in messages

    loop.step(len(transitions), _events(_event()))
    assert loop.flush_ui_loop_requests() is None


def test_scene_request_wraps_and_preserves_session_metadata() -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            current_scene="scene-c",
            scene_catalog=("scene-a", "scene-b", "scene-c"),
        )
    )
    loop.session_desc = SessionDesc(metadata={"caller": "preserved"})
    loop._ui_loop_requests = None

    loop._highlight_scene(1)
    assert loop.state.selected_scene == "scene-a"
    assert loop.state.status.current_scene == "scene-c"
    assert loop.flush_ui_loop_requests() is None
    loop._request_scene_name(loop.state.selected_scene)
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert requests.new_session is not None
    assert requests.new_session.metadata == {
        "caller": "preserved",
        "sample": "scene-a",
        "scene_catalog": ("scene-a", "scene-b", "scene-c"),
    }


def test_delayed_model_status_does_not_replace_newer_controller_input() -> None:
    state = HandDreamsUIState()
    state.observe_events(_events(_event()))
    state.observe_events(
        _events(
            _event(
                left=_controller(tracked=False, battery=None),
                right=_controller(trigger=0.8, battery=0.1),
            ),
        )
    )

    state.update_status(
        phase=HandDreamsPhase.LIVE,
        current_scene="scene-a",
        scene_catalog=("scene-a", "scene-b"),
    )
    state.observe_events(_events())

    assert ui_module.status_message(state.status) == "Tracking lost"
    assert not state.status.left_tracked
    assert state.status.left_battery is None
    assert state.status.right_tracked
    assert not state.status.right_open
    assert state.status.right_battery == pytest.approx(0.1)
    assert state.status.current_scene == "scene-a"


@pytest.mark.parametrize("replay", [False, True])
def test_reset_preserves_scene_and_input_mode(replay) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            current_scene="scene-a",
            scene_catalog=("scene-a", "scene-b"),
            left_tracked=True,
            right_tracked=True,
            left_trigger=1.0,
            right_trigger=1.0,
            calibration_progress=0.9,
            replay=replay,
        ),
    )
    loop.reset()
    loop.reset()
    assert loop.state.status == HandDreamsUIStatus(
        phase=HandDreamsPhase.LIVE if replay else HandDreamsPhase.WAITING,
        current_scene="scene-a",
        scene_catalog=("scene-a", "scene-b"),
        replay=replay,
    )


def test_interactive_ui_stays_open() -> None:
    assert not HandDreamsUILoop().is_finished()


@pytest.mark.parametrize("tracked", [False, True])
def test_startup_stays_neutral_until_first_generated_frame(
    monkeypatch, tracked
) -> None:
    generated = torch.full((3, 32, 64), 200, dtype=torch.uint8)
    current = [None]
    messages = []

    def overlay(
        frame,
        status,
        *,
        waiting_for_video=False,
        controller_frame=None,
        tutorial_frame=None,
        connection_status=None,
    ):
        messages.append((ui_module.status_message(status), waiting_for_video))
        return frame

    monkeypatch.setattr(ui_module, "draw_status_overlay", overlay)
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.LIVE,
            left_tracked=tracked,
            right_tracked=tracked,
            left_open=True,
            right_open=True,
        ),
        stage_width=64,
        stage_height=32,
    )
    loop.session_desc = SessionDesc(video_width=64, video_height=32)
    loop.output_layout = VideoTensorLayout.tchw
    monkeypatch.setattr(loop, "presented_model_frame", lambda: current[0])

    starting_results = loop.step(0, _events())
    assert isinstance(starting_results, list)
    (starting,) = starting_results
    assert starting.frame_count == 1
    assert starting.output_layout is VideoTensorLayout.tchw
    assert not torch.any(starting.read_output())
    message = "Live" if tracked else "Tracking lost"
    assert messages == [(message, True)]
    assert loop.state.status.phase is HandDreamsPhase.LIVE

    current[0] = generated
    (live,) = loop.step(1, _events())
    assert torch.equal(live.read_output(), generated.unsqueeze(0))
    assert messages == [(message, True), (message, False)]


@pytest.mark.parametrize(
    "phase",
    [HandDreamsPhase.SELECTING, HandDreamsPhase.LOADING, HandDreamsPhase.WAITING],
)
def test_scene_setup_never_reads_or_displays_existing_video(monkeypatch, phase) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(phase=phase),
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=360)
    loop.output_layout = VideoTensorLayout.tchw

    def unexpected_video():
        raise AssertionError("Scene setup must not read presented video")

    monkeypatch.setattr(loop, "presented_model_frame", unexpected_video)
    (result,) = loop.step(0, _events())
    assert torch.equal(
        result.read_output()[0, :, :32],
        torch.tensor((14, 16, 20))[:, None, None].expand(3, 32, 640),
    )


def test_calibration_progress_changes_only_when_model_publishes() -> None:
    state = HandDreamsUIState()
    state.update_status(
        phase=HandDreamsPhase.CALIBRATING,
        current_scene="",
        scene_catalog=(),
        calibration_progress=0.4,
    )
    closed = _event(left=_controller(trigger=1.0), right=_controller(trigger=1.0))
    state.observe_events(_events(closed, closed))
    state.observe_events(_events())
    assert state.status.calibration_progress == 0.4
    state.update_status(
        phase=HandDreamsPhase.ALIGNING, current_scene="", scene_catalog=()
    )
    assert state.status.calibration_progress == 0.0


@pytest.mark.parametrize("tracked", [False, True])
def test_first_frame_wait_text_preserves_tracking_loss(monkeypatch, tracked) -> None:
    messages = []
    original = ui_module.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui_module.ImageDraw.ImageDraw, "text", text)
    status = HandDreamsUIStatus(
        phase=HandDreamsPhase.LIVE, left_tracked=tracked, right_tracked=tracked
    )
    ui_module.draw_status_overlay(
        torch.zeros((3, 32, 64), dtype=torch.uint8), status, waiting_for_video=True
    )
    assert messages[0] == ("Starting demo..." if tracked else "Tracking lost")


def test_scene_strip_is_separate_from_the_model_stage_and_cached() -> None:
    previews = tuple(
        ScenePreview(
            f"scene-{index}",
            f"Scene {index}",
            np.full((18, 32, 3), index * 20, dtype=np.uint8),
        )
        for index in range(5)
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.WAITING,
            current_scene="scene-0",
            scene_catalog=tuple(preview.name for preview in previews),
        ),
        scene_previews=previews,
        stage_width=640,
        stage_height=360,
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=468)
    loop.output_layout = VideoTensorLayout.tchw

    (first,) = loop.step(0, _events())
    cached = loop.state.scene_strip_cache
    (second,) = loop.step(1, _events())

    assert first.read_output().shape == (1, 3, 468, 640)
    assert torch.equal(
        first.read_output()[0, :, :32],
        torch.tensor((14, 16, 20))[:, None, None].expand(3, 32, 640),
    )
    assert loop.state.scene_strip_cache is cached
    assert second.read_output().shape == first.read_output().shape


@pytest.fixture
def scene_picker():
    previews = tuple(
        ScenePreview(f"scene-{index}", f"Scene {index}") for index in range(7)
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.SELECTING,
            scene_catalog=tuple(preview.name for preview in previews),
        ),
        scene_previews=previews,
        stage_width=640,
        stage_height=360,
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=468)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    setattr(loop, "presented_model_frame", lambda channel_index=0: None)
    return loop


def _keys(*keys: str) -> UserInputEvents:
    return _events(
        *(
            KeyboardUserInputEvent(
                timestamp=np.uint64(index),
                key=key,
                state=KeyboardInputState.PRESSED,
            )
            for index, key in enumerate(keys)
        )
    )


@pytest.mark.parametrize(
    "phase,video_available,title",
    [
        (HandDreamsPhase.SELECTING, False, "Choose a scene"),
        (HandDreamsPhase.LIVE, False, "Starting replay..."),
        (HandDreamsPhase.LIVE, True, None),
    ],
)
def test_replay_has_no_controller_ui(
    monkeypatch, scene_picker, phase, video_available, title
) -> None:
    loop = scene_picker
    loop.state.status = replace(loop.state.status, phase=phase, replay=True)
    messages = []
    original = ui_module.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    def unexpected_hud(*args):
        raise AssertionError("Replay must not draw a controller HUD")

    monkeypatch.setattr(ui_module.ImageDraw.ImageDraw, "text", text)
    monkeypatch.setattr(ui_module, "_draw_controller_hud", unexpected_hud)
    frame = torch.full((3, 360, 640), 200, dtype=torch.uint8)
    setattr(loop, "presented_model_frame", lambda: frame if video_available else None)
    loop.step(
        0,
        _events(
            _event(),
            ControllerStatusEvent(
                timestamp=np.uint64(1),
                status=ControllerConnectionStatus(
                    headset="Unavailable",
                    web_client_url="https://example.test/teleop",
                ),
            ),
        ),
    )
    assert loop.state.controller_frame is None
    assert loop.state.connection_status == ControllerConnectionStatus()
    assert not loop.state.status.left_tracked
    assert not loop.state.status.right_tracked
    assert loop.state.connection_links == ()
    assert loop.flush_ui_loop_requests() is None
    if title is not None:
        assert title in messages
    else:
        assert ui_module.status_message(loop.state.status) == "Replay"
        assert torch.equal(
            ui_module.draw_status_overlay(frame, loop.state.status), frame
        )
    assert not any(
        token in message
        for message in messages
        for token in ("controller", "Tracking", "tracked", "Headset", "Isaac Teleop")
    )


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("phase", [HandDreamsPhase.SELECTING, HandDreamsPhase.LIVE])
def test_arrow_keys_only_highlight_until_enter(scene_picker, phase, replay) -> None:
    loop = scene_picker
    original = replace(
        loop.state.status, phase=phase, current_scene="scene-0", replay=replay
    )
    loop.state.status = original
    loop.step(0, _keys("ArrowRight", "ArrowRight", "ArrowLeft"))

    assert loop.state.selected_scene == "scene-1"
    assert loop.state.status == original
    assert loop.flush_ui_loop_requests() is None

    loop.state.update_status(
        phase=phase,
        current_scene=original.current_scene,
        scene_catalog=original.scene_catalog,
    )
    assert loop.state.selected_scene == "scene-1"
    loop.step(1, _keys("Enter"))
    request = loop.flush_ui_loop_requests()
    assert request.new_session.metadata["sample"] == "scene-1"
    assert loop.state.status.phase is HandDreamsPhase.LOADING
    assert loop.state.highlighted_scene == ""

    loop.step(2, _keys("ArrowRight", "Enter"))
    assert loop.flush_ui_loop_requests() is None
    assert loop.state.selected_scene == "scene-1"


@pytest.mark.parametrize(
    "keys,scene",
    [
        (("Enter",), "scene-0"),
        (("ArrowRight", "ArrowRight", "Enter"), "scene-1"),
        (("ArrowLeft", "Enter"), "scene-6"),
        (("ArrowLeft", "ArrowRight", "Enter"), "scene-0"),
    ],
)
def test_scene_browsing_and_confirmation_in_one_batch(scene_picker, keys, scene):
    loop = scene_picker
    loop.step(0, _keys(*keys))
    request = loop.flush_ui_loop_requests()
    assert request.new_session.metadata["sample"] == scene
    assert loop.state.status.phase is HandDreamsPhase.LOADING


def test_highlight_updates_cached_page_and_hit_testing_without_loading(scene_picker):
    loop = scene_picker
    loop.step(0, _events())
    cached = loop.state.scene_strip_cache
    loop.step(1, _keys(*(["ArrowRight"] * 6)))
    assert loop.state.selected_scene == "scene-5"
    assert loop.state.scene_strip_cache is not cached
    assert loop.state.scene_strip_cache_key[0] == "scene-5"
    assert loop.flush_ui_loop_requests() is None
    assert loop.state.status.phase is HandDreamsPhase.SELECTING

    layout = ui_module._scene_card_layout(
        loop.state.scene_previews, loop.state.selected_scene, 640, 108
    )
    assert [preview.name for preview, _ in layout] == ["scene-5", "scene-6"]
    for preview, (left, top, right, bottom) in layout:
        assert (
            loop._scene_at((left + right) / 2 / 640, (360 + (top + bottom) / 2) / 468)
            == preview.name
        )


def test_scene_reset_clears_unconfirmed_keyboard_selection(scene_picker):
    loop = scene_picker
    loop.state.status = replace(loop.state.status, current_scene="scene-0")
    loop.step(0, _keys("ArrowRight"))
    assert loop.state.selected_scene == "scene-1"
    loop.reset()
    assert loop.state.highlighted_scene == ""
    assert loop.state.selected_scene == "scene-0"
    assert loop.flush_ui_loop_requests() is None


@pytest.mark.parametrize("key", ["ArrowLeft", "ArrowRight", "Enter"])
def test_scene_keys_ignore_release(scene_picker, key):
    loop = scene_picker
    loop.step(
        0,
        _events(
            KeyboardUserInputEvent(
                timestamp=np.uint64(1), key=key, state=KeyboardInputState.RELEASED
            )
        ),
    )
    assert loop.state.selected_scene == ""
    assert loop.flush_ui_loop_requests() is None


def test_empty_scene_catalog_ignores_navigation_and_confirmation(scene_picker):
    loop = scene_picker
    loop.state.status = replace(loop.state.status, scene_catalog=())
    loop.step(0, _keys("ArrowRight", "Enter", "ArrowLeft", "Enter"))
    assert loop.state.selected_scene == ""
    assert loop.flush_ui_loop_requests() is None


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("input_kind", ["mouse", "touch", "keyboard"])
def test_scene_selection_uses_v2_session_replacement(input_kind, replay) -> None:
    previews = tuple(
        ScenePreview(f"scene-{index}", f"Scene {index}") for index in range(3)
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.SELECTING,
            scene_catalog=tuple(preview.name for preview in previews),
            replay=replay,
        ),
        scene_previews=previews,
        stage_width=640,
        stage_height=360,
    )
    loop.session_desc = SessionDesc(
        video_width=640, video_height=468, metadata={"caller": "preserved"}
    )
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    _, (left, top, right, bottom) = ui_module._scene_card_layout(
        previews, "scene-0", 640, 108
    )[0]
    x = (left + right) / 2 / 640
    y = (360 + (top + bottom) / 2) / 468
    if input_kind == "mouse":
        events = _events(
            MouseUserInputEvent(
                timestamp=np.uint64(1),
                action="button",
                x=x,
                y=y,
                button=0,
                pressed=True,
            )
        )
    elif input_kind == "touch":
        events = _events(
            TouchUserInputEvent(
                timestamp=np.uint64(1),
                action="start",
                x=x,
                y=y,
                primary=True,
            )
        )
    else:
        events = _events(
            KeyboardUserInputEvent(
                timestamp=np.uint64(1),
                key="Enter",
                state=KeyboardInputState.PRESSED,
            )
        )

    (loading,) = loop.step(0, events)
    requests = loop.flush_ui_loop_requests()
    assert requests is not None and requests.new_session is not None
    assert requests.new_session.metadata["sample"] == "scene-0"
    assert requests.new_session.metadata["caller"] == "preserved"
    assert loop.state.status.phase is HandDreamsPhase.LOADING
    assert loop.state.status.replay is replay
    assert loading.read_output().shape == (1, 3, 468, 640)

    loop.step(
        1,
        _events(
            KeyboardUserInputEvent(
                timestamp=np.uint64(2),
                key="ArrowLeft",
                state=KeyboardInputState.PRESSED,
            )
        ),
    )
    assert loop.flush_ui_loop_requests() is None


@pytest.mark.parametrize("catalog", [("scene-a",), ("scene-a", "scene-b")])
@pytest.mark.parametrize("direction", [-1, 1])
def test_first_scene_navigation_uses_catalog_edges(catalog, direction) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.SELECTING, scene_catalog=catalog
        )
    )
    loop.session_desc = SessionDesc()
    loop._ui_loop_requests = None
    loop._highlight_scene(direction)
    assert loop.flush_ui_loop_requests() is None
    assert loop.state.selected_scene == catalog[0 if direction > 0 else -1]
    assert loop.state.status.phase is HandDreamsPhase.SELECTING
    assert loop.state.status.current_scene == ""


def test_default_scene_can_be_confirmed_once_and_loading_ignores_old_status() -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.SELECTING,
            current_scene="scene-a",
            scene_catalog=("scene-a", "scene-b"),
        )
    )
    loop.session_desc = SessionDesc(metadata={"sample": "scene-a"})
    loop._ui_loop_requests = None
    loop._request_scene_name("scene-a")
    request = loop.flush_ui_loop_requests()
    assert request is not None and request.new_session is not None
    assert request.new_session.metadata["sample"] == "scene-a"
    loop.state.update_status(
        phase=HandDreamsPhase.LIVE,
        current_scene="scene-b",
        scene_catalog=("scene-a", "scene-b"),
    )
    assert loop.state.status.phase is HandDreamsPhase.LOADING
    assert loop.state.status.current_scene == "scene-a"
    loop._request_scene_name("scene-b")
    assert loop.flush_ui_loop_requests() is None


@pytest.mark.parametrize("replay", [False, True])
def test_reset_before_selection_still_requires_selection(replay) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.SELECTING, scene_catalog=("scene-a",), replay=replay
        ),
    )
    loop.reset()
    assert loop.state.status.phase is HandDreamsPhase.SELECTING
    assert loop.state.status.current_scene == ""
    assert loop.state.status.replay is replay


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize(
    "phase",
    [
        HandDreamsPhase.WAITING,
        HandDreamsPhase.CALIBRATING,
        HandDreamsPhase.ALIGNING,
        HandDreamsPhase.LIVE,
    ],
)
def test_escape_returns_to_picker_before_same_batch_scene_navigation(
    phase, replay
) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=phase,
            current_scene="scene-b",
            scene_catalog=("scene-a", "scene-b"),
            calibration_progress=0.5,
            replay=replay,
        )
    )
    loop.session_desc = SessionDesc(
        video_width=640,
        video_height=360,
        metadata={"sample": "scene-b", "caller": "preserved"},
    )
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    loop.step(
        0,
        _events(
            _event(),
            KeyboardUserInputEvent(
                timestamp=np.uint64(1),
                key="ArrowRight",
                state=KeyboardInputState.PRESSED,
            ),
            KeyboardUserInputEvent(
                timestamp=np.uint64(1),
                key="Enter",
                state=KeyboardInputState.PRESSED,
            ),
            KeyboardUserInputEvent(
                timestamp=np.uint64(1),
                key="Escape",
                state=KeyboardInputState.PRESSED,
            ),
        ),
    )
    request = loop.flush_ui_loop_requests()
    assert request is not None and request.new_session is not None
    assert request.new_session.metadata == {
        "sample": "scene-b",
        "caller": "preserved",
        "scene_catalog": ("scene-a", "scene-b"),
        "select_scene": True,
    }
    assert loop.state.status.phase is HandDreamsPhase.LOADING
    assert loop.state.status.current_scene == ""
    assert loop.state.status.calibration_progress == 0.0
    assert loop.state.status.left_tracked is not replay
    assert loop.state.status.right_tracked is not replay
    assert not loop.is_finished()
    loop.state.update_status(
        phase=HandDreamsPhase.LIVE,
        current_scene="scene-b",
        scene_catalog=("scene-a", "scene-b"),
    )
    assert loop.state.status.phase is HandDreamsPhase.LOADING


@pytest.mark.parametrize(
    "phase,key_state",
    [
        (HandDreamsPhase.SELECTING, KeyboardInputState.PRESSED),
        (HandDreamsPhase.LOADING, KeyboardInputState.PRESSED),
        (HandDreamsPhase.WAITING, KeyboardInputState.RELEASED),
    ],
)
def test_escape_is_idempotent_and_ignores_key_release(phase, key_state) -> None:
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(status=HandDreamsUIStatus(phase=phase))
    loop.session_desc = SessionDesc(video_width=640, video_height=360)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    loop.step(
        0,
        _events(
            KeyboardUserInputEvent(
                timestamp=np.uint64(1), key="Escape", state=key_state
            )
        ),
    )
    assert loop.flush_ui_loop_requests() is None
    assert loop.state.status.phase is phase


def test_postprocess_button_updates_ui_and_model_without_replacing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ModelState:
        postprocess_enabled = True

        def set_postprocess_enabled(self, enabled: bool) -> None:
            self.postprocess_enabled = enabled

    model_state = _ModelState()
    monkeypatch.setattr(
        ui_module,
        "invoke_async",
        lambda _loop, operation: operation(model_state),
    )
    loop = HandDreamsUILoop()
    loop.state = HandDreamsUIState(
        status=HandDreamsUIStatus(
            phase=HandDreamsPhase.WAITING,
            current_scene="scene-0",
            scene_catalog=("scene-0",),
        ),
        scene_previews=(ScenePreview("scene-0", "Scene 0"),),
        stage_width=640,
        stage_height=360,
        show_postprocess_toggle=True,
        postprocess_enabled=True,
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=468)
    loop.output_layout = VideoTensorLayout.tchw
    setattr(loop, "_model_loop", object())
    loop._ui_loop_requests = None
    left, top, right, bottom = ui_module._postprocess_button_rect(640, 360)

    (result,) = loop.step(
        0,
        _events(
            MouseUserInputEvent(
                timestamp=np.uint64(1),
                action="button",
                x=(left + right) / 2 / 640,
                y=(top + bottom) / 2 / 468,
                button=0,
                pressed=True,
            )
        ),
    )

    assert result.read_output().shape == (1, 3, 468, 640)
    assert not loop.state.postprocess_enabled
    assert not model_state.postprocess_enabled
    assert loop.flush_ui_loop_requests() is None
