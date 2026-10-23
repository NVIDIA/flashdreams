# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for picker controller status and the live status strip."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerSnapshot,
)
from handdreams import ui
from handdreams.controllers.quest_pose import OPENXR_LOCAL_GRIP_POSE_CONVENTION
from handdreams.ui import (
    HandDreamsPhase,
    HandDreamsUIState,
    HandDreamsUIStatus,
    draw_status_overlay,
    status_message,
)

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_status_explains_waiting_auto_start_and_tracking_loss() -> None:
    waiting = HandDreamsUIStatus()
    assert status_message(waiting) == "Waiting for controllers"
    tracked = replace(waiting, left_tracked=True, right_tracked=True)
    assert status_message(tracked) == "Preparing scene"
    opened = replace(tracked, left_open=True, right_open=True)
    assert status_message(opened) == "Preparing scene"
    assert (
        status_message(replace(tracked, phase=HandDreamsPhase.ALIGNING))
        == "Open both claws"
    )
    assert status_message(replace(opened, phase=HandDreamsPhase.LIVE)) == "Live"
    assert status_message(replace(waiting, phase=HandDreamsPhase.LIVE)) == (
        "Tracking lost"
    )
    assert status_message(replace(waiting, phase=HandDreamsPhase.SELECTING)) == (
        "Choose a scene"
    )
    assert status_message(replace(waiting, phase=HandDreamsPhase.LOADING)) == (
        "Returning to scenes..."
    )
    assert status_message(
        replace(waiting, phase=HandDreamsPhase.LOADING, current_scene="scene-a")
    ) == ("Loading scene...")


@pytest.mark.parametrize(
    "emulated,progress,message",
    [
        (False, 0.5, "Close both claws for 1 second"),
        (True, 0.5, "Hold either trigger for 1 second"),
        (True, 1.0, "Release both triggers to start"),
    ],
)
def test_setup_instructions_follow_model_confirmed_controller_type(
    monkeypatch, emulated, progress, message
):
    status = HandDreamsUIStatus(
        phase=HandDreamsPhase.CALIBRATING,
        left_tracked=True,
        right_tracked=True,
        left_trigger=1.0,
        is_emulated=emulated,
        calibration_progress=progress,
    )
    assert status_message(status) == message
    messages = []
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    draw_status_overlay(frame, status)
    assert (message if emulated else "Hold down both triggers for 1 second") in messages


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_status_text_renders_without_mutating_video(dtype, channels) -> None:
    frame = torch.zeros((channels, 64, 640), dtype=dtype)
    rendered = draw_status_overlay(
        frame, HandDreamsUIStatus(phase=HandDreamsPhase.LIVE)
    )
    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype
    assert not torch.any(frame)
    assert torch.equal(rendered[:, 18:], frame[:, 18:])
    assert rendered[0, :18, :350].unique().numel() > 1
    if dtype.is_floating_point:
        assert rendered.min() >= -1 and rendered.max() <= 1
    if channels == 4:
        assert torch.equal(rendered[3], frame[3])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "phase",
    [HandDreamsPhase.SELECTING, HandDreamsPhase.LOADING, HandDreamsPhase.WAITING],
)
def test_waiting_stage_replaces_scene_pixels_and_preserves_contract(
    dtype, channels, phase
) -> None:
    source = torch.full((channels, 360, 640), 1, dtype=dtype)
    alternate = torch.full_like(source, 0)
    status = HandDreamsUIStatus(phase=phase)
    rendered = draw_status_overlay(source, status)
    other = draw_status_overlay(alternate, status)
    assert rendered.shape == source.shape
    assert rendered.dtype == source.dtype
    assert torch.all(source == 1)
    assert torch.equal(rendered[:3], other[:3])
    if dtype.is_floating_point:
        assert rendered.min() >= -1 and rendered.max() <= 1
    if channels == 4:
        assert torch.equal(rendered[3], source[3])


@pytest.mark.parametrize(
    "phase,scene,title,detail",
    [
        (
            HandDreamsPhase.SELECTING,
            "",
            "Choose a scene",
            "Click a scene, or use arrow keys then Enter.",
        ),
        (
            HandDreamsPhase.LOADING,
            "scene-a",
            "Loading scene...",
            "Preparing the selected scenario.",
        ),
        (
            HandDreamsPhase.LOADING,
            "",
            "Returning to scenes...",
            "Finishing the current scene.",
        ),
        (HandDreamsPhase.WAITING, "", "Preparing scene...", "Waiting for controllers"),
    ],
)
def test_waiting_stage_explains_next_action(
    monkeypatch, phase, scene, title, detail
) -> None:
    messages = []
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    draw_status_overlay(
        torch.zeros((3, 360, 640), dtype=torch.uint8),
        HandDreamsUIStatus(phase=phase, current_scene=scene),
    )
    assert messages == [title] + ([detail] if detail else []) + (
        ["Left: Not tracked", "Right: Not tracked"]
        if phase is HandDreamsPhase.SELECTING
        else []
    )


@pytest.mark.parametrize("width", [320, 640])
def test_picker_status_follows_missing_hands_disconnect_and_recovery(
    monkeypatch, width
) -> None:
    messages = []
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append((message, position, kwargs["fill"]))
        bounds = draw.textbbox(
            position, message, **{key: kwargs[key] for key in ("font", "anchor")}
        )
        assert 0 <= bounds[0] < bounds[2] <= width
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    state = HandDreamsUIState(
        status=HandDreamsUIStatus(phase=HandDreamsPhase.SELECTING)
    )
    tracked = ControllerSnapshot(
        position=(0, 0, 0), orientation=(0, 0, 0, 1), battery=0.8
    )
    missing = replace(tracked, tracked=False, battery=None)
    frame = torch.zeros((3, 360, width), dtype=torch.uint8)
    positions = None
    for hands, expected in (
        ((missing, missing), ["Left: Not tracked", "Right: Not tracked"]),
        ((tracked, missing), ["Left: Tracked", "Battery: 80%", "Right: Not tracked"]),
        (
            (missing, replace(tracked, battery=0.25)),
            ["Left: Not tracked", "Right: Tracked", "Battery: 25%"],
        ),
        (None, ["Left: Not tracked", "Right: Not tracked"]),
        (
            (replace(tracked, battery=None), replace(tracked, battery=0.0)),
            ["Left: Tracked", "Right: Tracked", "Battery: 0%"],
        ),
    ):
        event_options: dict[str, Any] = (
            dict(
                left=hands[0],
                right=hands[1],
                frame_id=1,
                client_capture_ns=1,
                pose_convention=OPENXR_LOCAL_GRIP_POSE_CONVENTION,
            )
            if hands
            else {}
        )
        event = ControllerFrameEvent(
            timestamp=np.uint64(1),
            action="state" if hands else "disconnected",
            **event_options,
        )
        state.observe_events(UserInputEvents([event]))
        messages.clear()
        draw_status_overlay(frame, state.status)
        assert [message for message, _, _ in messages[2:]] == expected
        labels = [
            position
            for message, position, _ in messages
            if message.startswith(("Left:", "Right:"))
        ]
        assert positions is None or labels == positions
        positions = labels
        for message, _, color in messages:
            if message in ("Battery: 25%", "Battery: 0%"):
                assert color == (235, 182, 65)
            elif message == "Battery: 80%":
                assert color == ui._MUTED


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_healthy_live_video_has_no_top_bar(dtype, channels) -> None:
    frame = torch.full(
        (channels, 64, 640), 0.25 if dtype.is_floating_point else 77, dtype=dtype
    )
    live = HandDreamsUIStatus(
        phase=HandDreamsPhase.LIVE, left_tracked=True, right_tracked=True
    )
    output = draw_status_overlay(frame, live)
    assert torch.equal(output, frame)
    assert output.data_ptr() != frame.data_ptr()


def test_low_battery_warning_remains_visible_during_live() -> None:
    frame = torch.zeros((3, 64, 640), dtype=torch.uint8)
    live = HandDreamsUIStatus(
        phase=HandDreamsPhase.LIVE,
        left_tracked=True,
        right_tracked=True,
        left_battery=0.05,
    )
    output = draw_status_overlay(frame, live)
    assert not torch.equal(output[:, :18], frame[:, :18])
    assert torch.equal(output[:, 18:], frame[:, 18:])


@pytest.mark.parametrize("width,height", [(320, 360), (640, 360), (1280, 720)])
@pytest.mark.parametrize(
    "phase",
    [HandDreamsPhase.SELECTING, HandDreamsPhase.LOADING, HandDreamsPhase.WAITING],
)
def test_connection_guidance_uses_short_link_labels_before_tracking(
    monkeypatch, width, height, phase
) -> None:
    connection = ControllerConnectionStatus(
        headset="Detected; press the power button",
        web_client="Ready",
        web_client_url=(
            "https://nvidia.github.io/IsaacTeleop/client/release-1.4.x/"
            "?oobEnable=1&serverIP=10.111.69.99&port=48322"
        ),
        certificate_url="https://10.111.69.99:48322/",
        active_input="Waiting for controllers",
    )
    messages = []
    text_bounds = {}
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        bounds = draw.textbbox(
            position,
            message,
            **{key: kwargs[key] for key in ("font", "anchor") if key in kwargs},
        )
        assert 0 <= bounds[0] < bounds[2] <= width
        assert 0 <= bounds[1] < bounds[3] <= height
        text_bounds[message] = bounds
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    frame = torch.zeros((3, height, width), dtype=torch.uint8)
    result = draw_status_overlay(
        frame, HandDreamsUIStatus(phase=phase), connection_status=connection
    )
    assert not torch.any(frame)
    assert torch.any(result)
    combined = "".join(messages)
    assert any(message.startswith("Headset: ") for message in messages)
    assert "Isaac Teleop Web Client: " in messages
    assert "Open Link" in messages
    assert "Trust server certificate" not in combined
    assert connection.web_client_url not in combined
    assert connection.certificate_url not in combined
    assert "Active input:" not in combined
    assert "XR ready" not in combined
    links = ui._connection_links(connection, width, height)
    (link,) = links
    assert link.url == connection.web_client_url
    assert link.label == "Open Isaac Teleop Web Client"
    left, top, right, bottom = text_bounds["Open Link"]
    assert link.bounds[0] * width <= left < right <= link.bounds[2] * width
    assert link.bounds[1] * height <= top < bottom <= link.bounds[3] * height


@pytest.mark.parametrize(
    "phase",
    [HandDreamsPhase.CALIBRATING, HandDreamsPhase.ALIGNING, HandDreamsPhase.LIVE],
)
def test_setup_and_live_do_not_repeat_connection_status(monkeypatch, phase):
    connection = ControllerConnectionStatus(active_input="Isaac Teleop web client")
    messages = []
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    ui._status_text.cache_clear()
    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    result = draw_status_overlay(
        frame, HandDreamsUIStatus(phase=phase), connection_status=connection
    )
    assert not any(
        "Input:" in text or connection.active_input in text for text in messages
    )
    if phase is HandDreamsPhase.LIVE:
        assert torch.equal(result[:, 18:], frame[:, 18:])


@pytest.mark.parametrize(
    "width,height", [(160, 160), (832, 160), (832, 240), (1280, 720)]
)
def test_only_teleop_link_is_clickable_and_fits_small_stages(width, height) -> None:
    status = ControllerConnectionStatus(
        web_client_url="https://teleop.example/",
        certificate_url="https://10.0.0.1:48322/",
    )
    (link,) = ui._connection_links(status, width, height)
    assert link.url == status.web_client_url
    assert 0 <= link.bounds[0] < link.bounds[2] <= 1
    assert 0 <= link.bounds[1] < link.bounds[3] <= 1


@pytest.mark.parametrize("web_client", ["Starting...", "Unavailable", ""])
def test_connection_rows_show_actual_status_without_a_link(
    monkeypatch, web_client
) -> None:
    connection = ControllerConnectionStatus(
        headset="Streaming over USB",
        web_client=web_client,
        certificate_url="https://10.0.0.1:48322/",
    )
    messages = []
    original = ui.ImageDraw.ImageDraw.text

    def text(draw, position, message, *args, **kwargs):
        messages.append(message)
        return original(draw, position, message, *args, **kwargs)

    monkeypatch.setattr(ui.ImageDraw.ImageDraw, "text", text)
    draw_status_overlay(
        torch.zeros((3, 360, 640), dtype=torch.uint8),
        HandDreamsUIStatus(phase=HandDreamsPhase.SELECTING),
        connection_status=connection,
    )
    assert "Headset: Streaming over USB" in messages
    assert f"Isaac Teleop Web Client: {web_client or 'Waiting...'}" in messages
    assert "Open Link" not in messages
    assert "Trust server certificate" not in messages
    assert ui._connection_links(connection, 640, 360) == ()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("battery", [None, 1.0, 0.26, 0.25, 0.1, 0.0])
def test_low_battery_warning_is_labeled_without_blocking_live(side, battery) -> None:
    live = HandDreamsUIStatus(
        phase=HandDreamsPhase.LIVE,
        left_tracked=True,
        right_tracked=True,
        **{f"{side}_battery": battery},
    )
    assert status_message(live) == (
        "Live - low controller battery"
        if battery is not None and battery <= 0.25
        else "Live"
    )


@pytest.mark.parametrize("shape", [(3, 0, 5), (3, 5, 0), (2, 5, 5), (5, 5)])
def test_invalid_status_frame_has_clear_error(shape) -> None:
    with pytest.raises(ValueError, match="frame must be"):
        draw_status_overlay(torch.zeros(shape), HandDreamsUIStatus())
