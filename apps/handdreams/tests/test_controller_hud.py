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

"""CPU checks for the live controller HUD and its source event lifecycle."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from flashdreams_controller_input.events import ControllerFrameEvent, ControllerSnapshot
from flashdreams_controller_input.pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from handdreams import ui
from handdreams.controller_preview import (
    ControllerPreview,
    ControllerPreviewCalibration,
)
from handdreams.controllers.isaac_teleop_calibration import ParticipantPoseCalibrator
from handdreams.rotations import quat_xyzw_to_matrix
from handdreams.scenes import ScenePreview
from PIL import ImageDraw

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _event(*, convention=WRIST_PROXY_POSE_CONVENTION):
    return ControllerFrameEvent(
        timestamp=np.uint64(1),
        action="state",
        frame_id=1,
        client_capture_ns=1_000,
        pose_convention=convention,
        left=ControllerSnapshot(
            position=(-0.18, 1.0, -0.45), orientation=(0.0, 0.0, 0.0, 1.0)
        ),
        right=ControllerSnapshot(
            position=(0.18, 1.0, -0.45), orientation=(0.0, 0.0, 0.0, 1.0)
        ),
    )


def _state(event, phase=ui.HandDreamsPhase.LIVE):
    state = ui.HandDreamsUIState(status=ui.HandDreamsUIStatus(phase=phase))
    state.controller_preview = ControllerPreview(_calibration(_event()))
    state.observe_events(UserInputEvents([event]))
    return state


def _calibration(event, *, targets=None, gain=1.0, basis=None):
    frame = event.to_controller_frame()
    return ControllerPreviewCalibration.capture(
        frame,
        targets,
        participant_basis=basis,
        position_gain=gain,
    )


def _render(state):
    output = torch.zeros((3, 360, 640), dtype=torch.uint8)
    ui._draw_controller_hud(output, state)
    return output


@pytest.fixture
def claw_lines(monkeypatch):
    """Capture projected claw segments while retaining the real geometry code."""
    original = ui._draw_claw
    captured = []

    def draw_claw(draw, *args, **kwargs):
        lines = []
        recorder = SimpleNamespace(
            line=lambda xy, **kw: lines.append(np.asarray(xy).reshape(2, 2)),
            ellipse=lambda *args, **kw: None,
        )
        original(cast(Any, recorder), *args, **kwargs)
        captured.append(np.asarray(lines))

    monkeypatch.setattr(ui, "_draw_claw", draw_claw)

    def render(event):
        captured.clear()
        _render(_state(event))
        return np.asarray(captured)

    return render


def test_hud_keeps_live_input_across_delayed_model_status():
    """Retain the latest source pose when model status arrives later."""
    event = _event()
    state = _state(event, ui.HandDreamsPhase.WAITING)
    before = _render(state)
    updated = replace(event, left=replace(event.left, trigger=1.0), frame_id=2)
    state.observe_events(UserInputEvents([updated]))
    state.update_status(
        phase=ui.HandDreamsPhase.LIVE,
        current_scene="scene-a",
        scene_catalog=(),
        controller_calibration=state.controller_preview.calibration,
    )
    state.observe_events(UserInputEvents([]))

    assert state.controller_frame is updated
    assert state.status.left_trigger == 1.0
    assert not torch.equal(_render(state), before)


def test_disconnect_reset_and_recovery_remove_old_poses(claw_lines):
    """Clear stale controller geometry through disconnects and UI resets."""
    event = _event()
    disconnected = ControllerFrameEvent(timestamp=np.uint64(2), action="disconnected")
    state = _state(event)
    state.observe_events(UserInputEvents([disconnected]))
    assert state.controller_frame is disconnected
    assert not state.status.left_tracked and not state.status.right_tracked
    assert claw_lines(disconnected).size == 0

    recovered = replace(event, timestamp=np.uint64(3), frame_id=3)
    state.observe_events(UserInputEvents([event, disconnected, recovered]))
    assert state.controller_frame is recovered
    assert state.status.left_tracked and state.status.right_tracked
    assert len(claw_lines(recovered)) == 2

    loop = ui.HandDreamsUILoop()
    loop.state = state
    calibration = state.controller_preview.calibration
    loop.reset()
    assert state.controller_frame is None
    assert not torch.any(_render(state))
    state.observe_events(UserInputEvents([recovered]))
    assert state.controller_frame is recovered
    assert not torch.any(_render(state))
    state.update_status(
        phase=ui.HandDreamsPhase.LIVE,
        current_scene="scene-a",
        scene_catalog=(),
        controller_calibration=calibration,
    )
    assert torch.any(_render(state))


@pytest.mark.parametrize("side", ["left", "right"])
def test_untracked_hand_is_omitted_independently(claw_lines, side):
    """Keep the other hand visible without using an untracked placeholder pose."""
    event = _event()
    untracked = replace(getattr(event, side), tracked=False, position=(0, 0, 0))
    partial = replace(event, **{side: untracked})
    lines = claw_lines(partial)
    assert len(lines) == 1
    changed = replace(partial, **{side: replace(untracked, position=(99, 99, 99))})
    np.testing.assert_allclose(claw_lines(changed), lines)


def test_trigger_closure_and_quaternion_change_claw_geometry(claw_lines):
    """Close the jaw opening continuously and rotate the controller's forward ray."""
    event = _event()
    gaps = []
    for trigger in (0.0, 0.5, 1.0):
        lines = claw_lines(replace(event, left=replace(event.left, trigger=trigger)))
        gaps.append(lines[0, 6, 1, 0] - lines[0, 4, 1, 0])
    assert abs(gaps[0]) > abs(gaps[1]) > abs(gaps[2]) > 0

    neutral = claw_lines(event)
    assert neutral[0, 0, 1, 1] < neutral[0, 0, 0, 1]
    rotated = replace(event, left=replace(event.left, orientation=(0, 1, 0, 1)))
    turned = claw_lines(rotated)
    assert turned[0, 0, 1, 0] < turned[0, 0, 0, 0]
    assert not np.allclose(turned[0], neutral[0])
    np.testing.assert_allclose(turned[1], neutral[1])


@pytest.mark.parametrize("frame_width", [640, 1280])
@pytest.mark.parametrize("tracked", [True, False])
def test_hud_draws_thin_wireframes_without_hand_labels(
    monkeypatch, frame_width, tracked
):
    """Keep connectors visible between small joints and omit per-hand labels."""
    lines, joints, messages = [], [], []
    original_line = ImageDraw.ImageDraw.line
    original_ellipse = ImageDraw.ImageDraw.ellipse
    original_text = ImageDraw.ImageDraw.text

    def line(draw, xy, *args, **kwargs):
        lines.append((np.asarray(xy).reshape(2, 2), kwargs["width"]))
        return original_line(draw, xy, *args, **kwargs)

    def ellipse(draw, xy, *args, **kwargs):
        joints.append(np.asarray(xy))
        return original_ellipse(draw, xy, *args, **kwargs)

    def text(draw, xy, message, *args, **kwargs):
        messages.append(message)
        return original_text(draw, xy, message, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "line", line)
    monkeypatch.setattr(ImageDraw.ImageDraw, "ellipse", ellipse)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    event = _event()
    event = replace(
        event,
        left=replace(event.left, tracked=tracked),
        right=replace(event.right, tracked=tracked),
    )
    output = torch.zeros((3, 360, frame_width), dtype=torch.uint8)
    ui._draw_controller_hud(output, _state(event))

    assert messages == ["XR CONTROL"]
    assert len(lines) == len(joints) == (20 if tracked else 0)
    if tracked:
        assert all(width == 1 for _, width in lines)
        for bounds in joints:
            np.testing.assert_allclose(bounds[2:] - bounds[:2], (2, 2))
        assert max(np.linalg.norm(xy[1] - xy[0]) for xy, _ in lines) > 6


@pytest.mark.parametrize("position", [(-0.18, 1.1, -0.45), (-0.18, 1.0, -0.55)])
def test_upward_and_forward_motion_project_upward(claw_lines, position):
    """Project OpenXR positive Y and negative Z toward the top of the inset."""
    event = _event()
    before = claw_lines(event)
    after = claw_lines(replace(event, left=replace(event.left, position=position)))
    assert after[0, 0, 0, 1] < before[0, 0, 0, 1]


def test_both_hands_move_in_the_fixed_calibrated_view(claw_lines):
    event = _event()
    before = claw_lines(event)
    moved = replace(
        event,
        **{
            side: replace(
                hand, position=tuple(np.asarray(hand.position) + (0.04, 0.03, 0))
            )
            for side, hand in (("left", event.left), ("right", event.right))
        },
    )
    after = claw_lines(moved)
    assert np.all(after[:, 0, 0, 0] > before[:, 0, 0, 0])
    assert np.all(after[:, 0, 0, 1] < before[:, 0, 0, 1])
    np.testing.assert_allclose(after[0] - before[0], after[1] - before[1], atol=1e-5)


def test_tracking_loss_does_not_recenter_the_remaining_hand(claw_lines):
    event = _event()
    before = claw_lines(event)
    partial = replace(event, right=replace(event.right, tracked=False))
    np.testing.assert_allclose(claw_lines(partial)[0], before[0])


def test_hud_scale_does_not_change_when_hand_separation_changes(claw_lines):
    event = _event()
    before = claw_lines(event)
    wider = replace(event, left=replace(event.left, position=(-0.3, 1.0, -0.45)))
    after = claw_lines(wider)
    np.testing.assert_allclose(after[1], before[1])
    np.testing.assert_allclose(np.diff(after[0], axis=1), np.diff(before[0], axis=1))


def test_offscreen_projection_preserves_claw_shape(claw_lines):
    event = _event()
    before = claw_lines(event)
    moved = replace(event, left=replace(event.left, position=(-10, 20, -0.45)))
    after = claw_lines(moved)
    np.testing.assert_allclose(np.diff(after[0], axis=1), np.diff(before[0], axis=1))
    np.testing.assert_allclose(after[1], before[1])


@pytest.mark.parametrize(
    "position",
    [(-10, 1, -0.45), (10, 1, -0.45), (-0.18, 20, -0.45), (-0.18, -20, -0.45)],
)
def test_offscreen_hand_disappears_without_border_artifacts(position):
    event = _event()
    outside = replace(event, left=replace(event.left, position=position))
    absent = replace(event, left=replace(event.left, tracked=False))
    assert torch.equal(_render(_state(outside)), _render(_state(absent)))


@pytest.mark.parametrize(
    "position",
    [(-0.75, 1, -0.45), (0.75, 1, -0.45), (-0.18, 1.45, -0.45), (-0.18, 0.45, -0.45)],
)
def test_partially_visible_hand_is_cropped_inside_geometry_viewport(position):
    event = _event()
    partial = replace(
        event,
        left=replace(event.left, position=position),
        right=replace(event.right, tracked=False),
    )
    absent = replace(partial, left=replace(partial.left, tracked=False))
    changed = torch.any(_render(_state(partial)) != _render(_state(absent)), dim=0)
    ys, xs = changed.nonzero(as_tuple=True)
    assert len(xs) > 0
    # 135x81 inset at (497, 271), with a (9, 24, 126, 72) geometry viewport.
    assert torch.all((xs >= 506) & (xs < 623))
    assert torch.all((ys >= 295) & (ys < 343))


def test_preview_matches_calibrator_with_off_axis_basis_gain_and_scene_rotation():
    neutral = _event(convention=WRIST_PROXY_POSE_CONVENTION)
    neutral = replace(
        neutral,
        left=replace(neutral.left, orientation=(0.1, 0.2, 0.3, 0.9)),
        right=replace(neutral.right, orientation=(-0.2, 0.1, 0.0, 0.9)),
    )
    basis = quat_xyzw_to_matrix((0, 0.3, 0, 0.9)) @ np.diag([1, -1, -1])
    targets = {}
    for side, x in (("left", -0.15), ("right", 0.25)):
        pose = np.eye(4)
        pose[:3, 3] = (x, 0.1, 0.6)
        pose[:3, :3] = quat_xyzw_to_matrix((0.3, 0.2, 0.1, 0.8))
        targets[side] = pose
    reference = _calibration(neutral, targets=targets, basis=basis, gain=1.2)
    preview = ControllerPreview(reference)
    for claw, target in zip(preview.observe(neutral), targets.values(), strict=True):
        assert claw is not None
        np.testing.assert_allclose(claw.position, target[:3, 3], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(claw.rotation).reshape(3, 3), target[:3, :3], atol=1e-6
        )

    moved = replace(
        neutral,
        left=replace(
            neutral.left, position=(-0.1, 1.1, -0.5), orientation=(0.2, -0.1, 0.3, 0.8)
        ),
    )
    calibrator = ParticipantPoseCalibrator()
    anchor = calibrator.configure(
        neutral.to_controller_frame(), basis, position_gain=1.2
    )
    expected = calibrator.transform(moved.to_controller_frame())
    for side, claw in zip(("left", "right"), preview.observe(moved), strict=True):
        assert claw is not None
        np.testing.assert_allclose(
            claw.position,
            targets[side][:3, 3]
            + np.asarray(expected[side]["position"])
            - anchor[side]["position"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(claw.rotation).reshape(3, 3),
            quat_xyzw_to_matrix(expected[side]["orientation"]) @ targets[side][:3, :3],
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "convention", [OPENXR_LOCAL_GRIP_POSE_CONVENTION, OPENXR_STAGE_GRIP_POSE_CONVENTION]
)
def test_hud_rejects_raw_grip_poses(convention):
    raw = _event(convention=convention)
    with pytest.raises(ValueError, match="requires canonical wrist poses"):
        _render(_state(raw))
    with pytest.raises(ValueError, match="requires canonical wrist poses"):
        _calibration(raw)


def test_preview_snapshot_detaches_source_data_and_never_changes_input():
    event = _event(convention=WRIST_PROXY_POSE_CONVENTION)
    frame = event.to_controller_frame()
    targets = {side: np.eye(4) for side in ("left", "right")}
    basis = np.diag([1.0, -1.0, -1.0])
    reference = ControllerPreviewCalibration.capture(
        frame, targets, participant_basis=basis, position_gain=1.2
    )
    expected = ControllerPreview(reference).observe(event)
    frame["left"]["position"][0] = 999
    targets["left"][:] = 999
    basis[:] = 999
    assert ControllerPreview(reference).observe(event) == expected
    assert event.left.position == (-0.18, 1.0, -0.45)
    with pytest.raises(FrozenInstanceError):
        reference.position_gain = 2  # ty: ignore[invalid-assignment]  # Verify the read-only property.


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_hud_changes_only_bottom_right_stage_and_preserves_source(
    monkeypatch, dtype, channels
):
    """Composite within the video stage while preserving source pixels and alpha."""
    source = torch.full(
        (channels, 360, 640), 100 if dtype == torch.uint8 else 0.2, dtype=dtype
    )
    original = source.clone()
    event = _event()
    loop = ui.HandDreamsUILoop()
    loop.state = _state(event)
    loop.state.scene_previews = (ScenePreview("scene-a", "Scene A"),)
    loop.session_desc = SessionDesc(video_width=640, video_height=468)
    loop.output_layout = VideoTensorLayout.tchw
    monkeypatch.setattr(loop, "presented_model_frame", lambda: source)
    loop.state.controller_frame = None
    baseline = loop.step(0, UserInputEvents([]))[0].read_output()[0]
    rendered = loop.step(1, UserInputEvents([event]))[0].read_output()[0]

    assert torch.equal(source, original)
    assert rendered.dtype == dtype and rendered.shape == (channels, 468, 640)
    changed = torch.any(rendered != baseline, dim=0).nonzero()
    assert len(changed) > 0
    assert torch.all((changed[:, 0] >= 271) & (changed[:, 0] < 352))
    assert torch.all((changed[:, 1] >= 497) & (changed[:, 1] < 632))
    assert torch.equal(rendered[:, 360:], baseline[:, 360:])
    if channels == 4:
        assert torch.equal(rendered[3], baseline[3])
    if dtype.is_floating_point:
        assert rendered.min() >= -1 and rendered.max() <= 1


@pytest.mark.parametrize(
    "phase",
    [
        ui.HandDreamsPhase.CALIBRATING,
        ui.HandDreamsPhase.ALIGNING,
    ],
)
def test_hud_does_not_cover_setup_guides(phase):
    """Leave calibration and alignment presentation untouched."""
    assert not torch.any(_render(_state(_event(), phase)))


@pytest.mark.parametrize("shape", [(3, 199, 640), (3, 360, 319)])
def test_hud_skips_frames_too_small_for_the_inset(shape):
    """Leave undersized frames intact instead of drawing outside their bounds."""
    output = torch.zeros(shape, dtype=torch.uint8)
    ui._draw_controller_hud(output, _state(_event()))
    assert not torch.any(output)


@pytest.mark.parametrize("width", [320, 1280])
def test_short_stage_keeps_the_hud_below_status_and_inside_frame(width):
    """Fit the inset within both narrow and wide stages at minimum height."""
    output = torch.zeros((3, 200, width), dtype=torch.uint8)
    ui._draw_controller_hud(output, _state(_event()))
    changed = torch.any(output != 0, dim=0).nonzero()
    assert len(changed) > 0
    assert torch.all((changed[:, 0] >= 24) & (changed[:, 0] < 200))
    assert torch.all((changed[:, 1] >= 0) & (changed[:, 1] < width))
