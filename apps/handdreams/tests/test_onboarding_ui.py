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

"""CPU checks for full-stage controller onboarding and tutorial playback."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from flashdreams_controller_input.events import ControllerFrameEvent, ControllerSnapshot
from flashdreams_controller_input.pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from handdreams import ui
from handdreams.alignment import AlignmentClaw, AlignmentPreview
from handdreams.tutorials import TutorialVideo
from PIL import Image, ImageDraw

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _status(phase=ui.HandDreamsPhase.ALIGNING):
    left = AlignmentClaw((-0.2, 0.18, 0.6), tuple(np.eye(3).flat))
    right = replace(left, position=(0.2, 0.18, 0.6))
    return ui.HandDreamsUIStatus(
        phase=phase,
        left_tracked=True,
        right_tracked=True,
        left_open=True,
        right_open=True,
        alignment=AlignmentPreview(left, right, left, right),
    )


@pytest.mark.parametrize(
    "phase,title,detail,footer",
    [
        (
            ui.HandDreamsPhase.CALIBRATING,
            "Try the robot grippers",
            "Squeeze each index trigger to close its gripper. Release to open.",
            "Hold down both triggers for 1 second",
        ),
        (
            ui.HandDreamsPhase.ALIGNING,
            "Match the outlines",
            "Release both triggers and move both hands into the gray guides.",
            "Hold still",
        ),
    ],
)
def test_setup_page_explains_the_current_task(
    monkeypatch, phase, title, detail, footer
):
    messages = []
    original = ImageDraw.ImageDraw.text

    def text(draw, xy, message, *args, **kwargs):
        messages.append(message)
        return original(draw, xy, message, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    frame = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    output = ui.draw_status_overlay(frame, _status(phase))
    assert {title, footer} <= set(messages)
    assert detail not in messages
    assert "schematic" not in " ".join(messages).lower()
    assert not torch.equal(output[:, 18:180], frame[:, 18:180])


@pytest.mark.parametrize(
    "phase", [ui.HandDreamsPhase.CALIBRATING, ui.HandDreamsPhase.ALIGNING]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_tutorial_inset_preserves_input_shape_dtype_and_alpha(phase, channels, dtype):
    frame = torch.full(
        (channels, 360, 640), 73 if dtype == torch.uint8 else 0.25, dtype=dtype
    )
    original = frame.clone()
    status = _status(phase)
    bare = ui.draw_status_overlay(frame, status)
    output = ui.draw_status_overlay(
        frame, status, tutorial_frame=Image.new("RGB", (100, 120), (210, 25, 180))
    )
    assert torch.equal(frame, original)
    assert output.shape == frame.shape and output.dtype == frame.dtype
    if channels == 4:
        assert torch.equal(output[3], original[3])
    if dtype.is_floating_point:
        assert output.min() >= -1 and output.max() <= 1
    changed = torch.any(output != bare, dim=0)
    ys, xs = changed.nonzero(as_tuple=True)
    assert len(xs) > 10_000
    assert xs.min() >= 524 and xs.max() <= 624
    assert ys.min() >= 79 and ys.max() <= 199


def test_alignment_display_compares_poses_relative_to_scene_midpoint():
    status = _status()
    preview = status.alignment
    assert preview is not None
    shifted = replace(
        preview,
        **{
            name: replace(
                getattr(preview, name),
                position=tuple(np.asarray(getattr(preview, name).position) + (4, 2, 8)),
            )
            for name in ("left", "right", "left_target", "right_target")
        },
    )
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    assert torch.equal(
        ui.draw_status_overlay(frame, status),
        ui.draw_status_overlay(frame, replace(status, alignment=shifted)),
    )


def test_alignment_progress_tracks_geometry_and_clears_when_input_cannot_align():
    status = _status()
    preview = status.alignment
    assert (
        preview is not None and preview.left is not None and preview.right is not None
    )
    halfway = replace(
        preview,
        left=replace(preview.left, position_error_m=0.18),
        right=replace(preview.right, position_error_m=0.18),
    )
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    yellow = torch.tensor((235, 182, 65), dtype=torch.uint8)[:, None]

    def progress_pixels(current):
        output = ui.draw_status_overlay(frame, current)
        return torch.all(output[:, 318, 192:449] == yellow, dim=0).sum().item()

    full = progress_pixels(status)
    half = progress_pixels(replace(status, alignment=halfway))
    assert full > 250 and 120 < half < 135
    for current in (
        replace(status, left_tracked=False),
        replace(status, right_tracked=False),
        replace(status, left_open=False),
        replace(status, right_open=False),
        replace(status, alignment=None),
        replace(status, alignment=replace(preview, left=None)),
    ):
        assert progress_pixels(current) == 0


def test_alignment_progress_bounds_missing_exact_partial_and_far_poses():
    preview = _status().alignment
    assert (
        preview is not None and preview.left is not None and preview.right is not None
    )
    assert preview.progress == 1.0
    assert replace(preview, left=None).progress == 0.0
    assert replace(preview, right=None).progress == 0.0
    partial = replace(preview.left, position_error_m=0.18, rotation_error_deg=20)
    assert 0 < replace(preview, left=partial, right=partial).progress < 1
    far = replace(preview.left, position_error_m=2.0)
    assert replace(preview, left=far).progress == 0.0
    backward = replace(preview.left, rotation_error_deg=180)
    assert replace(preview, right=backward).progress == 0.0


def test_alignment_claws_show_open_aligned_and_closed_states():
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    status = _status()
    opened = ui.draw_status_overlay(frame, status)[:, 80:248]
    closed = ui.draw_status_overlay(frame, replace(status, left_open=False))[:, 80:248]
    lost = ui.draw_status_overlay(
        frame, replace(status, left_open=False, left_tracked=False)
    )[:, 80:248]
    green = torch.tensor((118, 208, 110), dtype=torch.uint8)[:, None, None]
    red = torch.tensor((245, 65, 65), dtype=torch.uint8)[:, None, None]
    assert torch.all(opened == green, dim=0).any()
    assert torch.all(closed == green, dim=0).any()
    assert torch.all(closed == red, dim=0).any()
    assert not torch.all(opened == red, dim=0).any()
    assert not torch.all(lost == red, dim=0).any()


@pytest.mark.parametrize(
    "convention", [OPENXR_LOCAL_GRIP_POSE_CONVENTION, WRIST_PROXY_POSE_CONVENTION]
)
def test_trigger_tutorial_shows_relative_motion_without_changing_input(convention):
    event = ControllerFrameEvent(
        timestamp=np.uint64(1),
        action="state",
        frame_id=1,
        client_capture_ns=1000,
        pose_convention=convention,
        left=ControllerSnapshot(position=(-0.2, 1, -0.4), orientation=(0, 0, 0, 1)),
        right=ControllerSnapshot(position=(0.2, 1, -0.4), orientation=(0, 0, 0, 1)),
    )
    assert event.left is not None
    shifted = replace(
        event,
        left=replace(event.left, position=(-0.2, 1.5, -0.4)),
    )
    rotated = replace(event, left=replace(event.left, orientation=(0, 1, 0, 0)))
    state = ui.HandDreamsUIState(status=_status(ui.HandDreamsPhase.CALIBRATING))
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)

    def render(current):
        state.observe_events(UserInputEvents([current]))
        output = ui.draw_status_overlay(
            frame, state.status, controller_frame=state.controller_frame
        )
        assert state.controller_frame is current
        return output

    original = render(event)
    assert not torch.equal(original[:, 80:248], render(shifted)[:, 80:248])
    assert not torch.equal(original[:, 80:248], render(rotated)[:, 80:248])
    assert rotated.left is not None
    assert rotated.left.orientation == (0, 1, 0, 0)
    assert shifted.left is not None
    assert shifted.left.position == (-0.2, 1.5, -0.4)
    closed = replace(rotated, left=replace(rotated.left, trigger=1))
    assert not torch.equal(original[:, 80:248], render(closed)[:, 80:248])
    assert state.status.left_trigger == 1


@pytest.mark.parametrize("x", [-100, 100])
def test_alignment_claw_outside_view_disappears_instead_of_pinning_to_border(x):
    status = _status()
    preview = status.alignment
    moved = replace(preview.left, position=(x, 0.18, 0.6))
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    outside = ui.draw_status_overlay(
        frame, replace(status, alignment=replace(preview, left=moved))
    )
    absent = ui.draw_status_overlay(
        frame, replace(status, alignment=replace(preview, left=None))
    )
    assert torch.equal(outside[:, 83:245], absent[:, 83:245])


def test_ui_loop_selects_tutorial_and_releases_it_on_live_reset_and_close(monkeypatch):
    tutorial = Mock(spec=TutorialVideo)
    tutorial.frame.return_value = Image.new("RGB", (100, 120), (25, 60, 210))
    loop = ui.HandDreamsUILoop()
    loop.state = ui.HandDreamsUIState(
        status=_status(ui.HandDreamsPhase.CALIBRATING), tutorial=tutorial
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=360)
    loop.output_layout = VideoTensorLayout.tchw
    monkeypatch.setattr(loop, "presented_model_frame", lambda: None)
    monkeypatch.setattr(ui.time, "monotonic", lambda: 100.0)
    events = UserInputEvents([])

    (result,) = loop.step(0, events)
    assert result.read_output().shape == (1, 3, 360, 640)
    tutorial.frame.assert_called_once_with("press_triggers.mp4", 115, 144, 100.0)
    tutorial.close.assert_not_called()
    loop.state.status = replace(loop.state.status, phase=ui.HandDreamsPhase.ALIGNING)
    loop.step(1, events)
    tutorial.frame.assert_called_with("align_arms.mp4", 115, 144, 100.0)
    loop.state.status = replace(loop.state.status, phase=ui.HandDreamsPhase.LIVE)
    loop.step(2, events)
    assert tutorial.frame.call_count == 2
    tutorial.close.assert_called_once_with()
    loop.reset()
    loop.close()
    loop.close()
    assert tutorial.close.call_count == 4
    assert loop.state.status.phase is ui.HandDreamsPhase.WAITING
