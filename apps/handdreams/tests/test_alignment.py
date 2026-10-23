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

"""CPU checks for V2-owned setup geometry and onboarding feedback."""

import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from handdreams.alignment import AlignmentGuide
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
)
from handdreams.ui import (
    HandDreamsPhase,
    HandDreamsUIStatus,
    draw_status_overlay,
    status_message,
)

pytestmark = pytest.mark.ci_cpu


def _frame():
    return {
        "client_capture_ns": 1,
        "pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
        **{
            side: {
                "position": [x, 1.1, -0.4],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "tracked": True,
                "trigger": 0.0,
            }
            for side, x in (("left", -0.2), ("right", 0.2))
        },
    }


def _targets():
    targets = {}
    for side, x in (("left", -0.2), ("right", 0.2)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [x, 0.18, 0.6]
        targets[side] = pose
    return targets


def test_alignment_uses_unscaled_separation_and_current_midpoint() -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    first = guide.observe(frame)
    assert first.aligned
    assert first.left is not None and first.right is not None
    np.testing.assert_allclose(first.left.position, [-0.2, 0.18, 0.6], atol=1e-6)
    np.testing.assert_allclose(first.right.position, [0.2, 0.18, 0.6], atol=1e-6)

    frame["left"]["position"][0] += 0.26
    shifted = guide.observe(frame)
    assert not shifted.aligned
    assert shifted.left is not None and shifted.right is not None
    assert shifted.left.position_error_m == pytest.approx(0.13)
    assert shifted.right.position_error_m == pytest.approx(0.13)
    assert guide.observe(frame) == shifted


@pytest.mark.parametrize("degrees,aligned", [(34.9, True), (35.1, False)])
def test_alignment_checks_absolute_forward_direction(degrees, aligned) -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    angle = np.deg2rad(degrees)
    frame["right"]["orientation"] = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
    preview = guide.observe(frame)
    assert preview.right is not None
    assert preview.right.rotation_error_deg == pytest.approx(degrees, abs=1e-4)
    assert preview.aligned is aligned
    assert preview.right.position_error_m < 1e-6


@pytest.mark.parametrize("degrees", [90, 180])
def test_sideways_or_backward_initial_controllers_cannot_align(degrees) -> None:
    frame = _frame()
    angle = np.deg2rad(degrees)
    for side in ("left", "right"):
        frame[side]["orientation"] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    preview = AlignmentGuide(frame, _targets()).observe(frame)
    assert not preview.aligned
    for hand in (preview.left, preview.right):
        assert hand is not None
        assert hand.position_error_m < 1e-6
        assert hand.rotation_error_deg == pytest.approx(degrees, abs=1e-4)


def test_alignment_ignores_roll_around_correct_claw_direction() -> None:
    frame = _frame()
    frame["right"]["orientation"] = [0, 0, 1, 0]
    preview = AlignmentGuide(frame, _targets()).observe(frame)
    assert preview.aligned
    assert preview.right is not None
    assert preview.right.rotation_error_deg == pytest.approx(0)


def test_alignment_heading_ignores_pointing_and_stays_frozen() -> None:
    frame = _frame()
    original_basis = AlignmentGuide(frame, _targets()).participant_basis
    angle = np.deg2rad(45)
    for side in ("left", "right"):
        frame[side]["orientation"] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    guide = AlignmentGuide(frame, _targets())
    np.testing.assert_allclose(guide.participant_basis, original_basis)
    frame["right"]["position"][2] += 0.3
    guide.observe(frame)
    np.testing.assert_allclose(guide.participant_basis, original_basis)


def test_alignment_cancels_pair_translation_without_mutating_input() -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    first = guide.observe(frame)
    for side in ("left", "right"):
        frame[side]["position"] = (
            np.asarray(frame[side]["position"]) + [0.4, 0.2, -0.3]
        ).tolist()
    original = copy.deepcopy(frame)
    preview = guide.observe(frame)
    assert preview.aligned
    for side in ("left", "right"):
        np.testing.assert_allclose(
            getattr(preview, side).position, getattr(first, side).position, atol=1e-6
        )
    assert frame == original


@pytest.mark.parametrize("offset,aligned", [(0.119, True), (0.121, False)])
def test_alignment_tolerates_twelve_centimeters_per_hand(offset, aligned) -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    frame["left"]["position"][0] -= offset
    frame["right"]["position"][0] += offset
    preview = guide.observe(frame)
    assert preview.aligned is aligned
    assert preview.left is not None and preview.right is not None
    assert preview.left.position_error_m == pytest.approx(offset)
    assert preview.right.position_error_m == pytest.approx(offset)


@pytest.mark.parametrize("missing", [False, True])
def test_untracked_pair_hides_geometry_until_both_hands_recover(missing) -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    expected = guide.observe(frame)
    if missing:
        frame["left"] = None
    else:
        frame["left"]["tracked"] = False
    preview = guide.observe(frame)
    assert preview.left is None
    assert preview.right is None
    assert not preview.aligned
    assert guide.observe(_frame()) == expected


def test_alignment_snapshot_and_guide_do_not_alias_source_data() -> None:
    frame, targets = _frame(), _targets()
    guide = AlignmentGuide(frame, targets)
    preview = guide.observe(frame)
    targets["left"][:] = 0
    frame["left"]["position"][:] = [50, 50, 50]
    basis = guide.participant_basis
    basis[:] = 0
    assert guide.observe(_frame()) == preview
    assert preview.left is not None
    assert isinstance(preview.left.position, tuple)
    assert isinstance(preview.left.rotation, tuple)
    with pytest.raises(FrozenInstanceError):
        preview.left = None  # ty: ignore[invalid-assignment] - verify the frozen snapshot.


def test_alignment_rejects_missing_tracking_and_degenerate_basis() -> None:
    frame = _frame()
    frame["left"]["tracked"] = False
    with pytest.raises(ValueError, match="both tracked"):
        AlignmentGuide(frame, _targets())
    frame = _frame()
    frame["left"]["position"] = frame["right"]["position"].copy()
    with pytest.raises(ValueError, match="degenerate"):
        AlignmentGuide(frame, _targets())


@pytest.mark.parametrize(
    "convention", [None, OPENXR_LOCAL_GRIP_POSE_CONVENTION, "unknown"]
)
def test_alignment_never_applies_a_second_wrist_conversion(convention) -> None:
    frame = _frame()
    frame["pose_convention"] = convention
    with pytest.raises(ValueError, match="canonical wrist"):
        AlignmentGuide(frame, _targets())


def test_alignment_rejects_missing_or_nonrigid_scene_targets() -> None:
    targets = _targets()
    del targets["left"]
    with pytest.raises(ValueError, match="missing the left"):
        AlignmentGuide(_frame(), targets)
    targets = _targets()
    targets["left"][0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        AlignmentGuide(_frame(), targets)


def _status():
    return HandDreamsUIStatus(
        phase=HandDreamsPhase.ALIGNING,
        left_tracked=True,
        right_tracked=True,
        left_open=True,
        right_open=True,
        alignment=AlignmentGuide(_frame(), _targets()).observe(_frame()),
    )


def test_setup_status_names_missing_hands_and_alignment_progress() -> None:
    status = _status()
    assert status_message(status) == "Hold still"
    assert (
        status_message(replace(status, left_tracked=False))
        == "Waiting for left controller"
    )
    assert (
        status_message(replace(status, right_tracked=False))
        == "Waiting for right controller"
    )
    assert status_message(replace(status, left_open=False)) == "Open both claws"
    assert (
        status_message(replace(status, phase=HandDreamsPhase.CALIBRATING))
        == "Close both claws for 1 second"
    )
    assert status_message(replace(status, alignment=None)) == "Match claw outlines"


@pytest.mark.parametrize("left_open", [False, True])
@pytest.mark.parametrize("right_open", [False, True])
def test_trigger_calibration_instruction_precedes_open_claw_gate(
    left_open, right_open
) -> None:
    status = replace(
        _status(),
        phase=HandDreamsPhase.CALIBRATING,
        left_open=left_open,
        right_open=right_open,
        alignment=None,
    )
    assert status_message(status) == "Close both claws for 1 second"
    assert (
        status_message(replace(status, left_tracked=False))
        == "Waiting for left controller"
    )
    assert (
        status_message(replace(status, right_tracked=False))
        == "Waiting for right controller"
    )
    assert (
        status_message(replace(status, left_tracked=False, right_tracked=False))
        == "Waiting for controllers"
    )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_calibration_panel_displays_closing_claws_and_model_hold_progress(
    dtype,
) -> None:
    frame = torch.zeros((3, 360, 640), dtype=dtype)
    status = replace(_status(), phase=HandDreamsPhase.CALIBRATING, alignment=None)
    opened = draw_status_overlay(frame, status)
    closed = draw_status_overlay(
        frame, replace(status, left_trigger=1.0, right_trigger=1.0)
    )
    half = draw_status_overlay(
        frame,
        replace(status, left_trigger=1.0, right_trigger=1.0, calibration_progress=0.5),
    )
    assert not torch.equal(opened[:, 83:245], closed[:, 83:245])
    assert torch.equal(closed[:, 83:245], half[:, 83:245])
    assert not torch.equal(closed[:, 316:321], half[:, 316:321])
    assert torch.equal(closed[:, 316:321, 330:], half[:, 316:321, 330:])
    assert torch.any(opened[:, 18:180] != frame[:, 18:180])
    assert not torch.any(frame)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("interruption", ["tracking", "release"])
def test_calibration_hold_progress_is_hidden_on_tracking_loss_or_release(
    side, interruption
) -> None:
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    status = replace(
        _status(),
        phase=HandDreamsPhase.CALIBRATING,
        alignment=None,
        left_trigger=1.0,
        right_trigger=1.0,
        calibration_progress=1.0,
    )
    status = replace(
        status,
        **{f"{side}_tracked": False}
        if interruption == "tracking"
        else {f"{side}_trigger": 0.8},
    )
    lost = draw_status_overlay(frame, status)
    assert torch.equal(
        lost,
        draw_status_overlay(frame, replace(status, calibration_progress=0.0)),
    )


def test_small_frame_omits_calibration_panel_instead_of_overlapping_labels() -> None:
    frame = torch.zeros((3, 160, 640), dtype=torch.uint8)
    status = replace(_status(), phase=HandDreamsPhase.CALIBRATING, alignment=None)
    assert not torch.any(draw_status_overlay(frame, status)[:, 18:])


def test_calibration_panel_is_removed_after_setup() -> None:
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    status = replace(_status(), phase=HandDreamsPhase.CALIBRATING, alignment=None)
    assert torch.any(draw_status_overlay(frame, status)[:, 180:])
    assert torch.equal(
        draw_status_overlay(frame, replace(status, phase=HandDreamsPhase.LIVE))[:, 18:],
        frame[:, 18:],
    )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_closed_setup_claws_are_red_without_mutating_video(dtype) -> None:
    frame = torch.zeros((3, 360, 640), dtype=dtype)
    opened = draw_status_overlay(frame, _status())
    closed = draw_status_overlay(frame, replace(_status(), left_open=False))
    red = torch.tensor([245, 65, 65], dtype=dtype)
    if dtype.is_floating_point:
        red = red / 127.5 - 1.0
    assert torch.any(torch.all(closed == red[:, None, None], dim=0))
    assert not torch.any(torch.all(opened == red[:, None, None], dim=0))
    assert not torch.any(frame)


def test_preview_hides_untracked_claws_and_is_removed_in_live_phase() -> None:
    frame = torch.zeros((3, 360, 640), dtype=torch.uint8)
    status = replace(_status(), left_open=False, left_tracked=False)
    preview = draw_status_overlay(frame, status)
    red = torch.tensor([245, 65, 65], dtype=torch.uint8)[:, None, None]
    assert not torch.any(torch.all(preview == red, dim=0))
    live = draw_status_overlay(frame, replace(status, phase=HandDreamsPhase.LIVE))
    assert torch.equal(live[:, 18:], frame[:, 18:])


def test_extreme_tracked_positions_are_cropped_from_the_setup_workspace() -> None:
    frame = _frame()
    guide = AlignmentGuide(frame, _targets())
    frame["left"]["position"] = [1e20, -1e20, 1e20]
    preview = guide.observe(frame)
    assert not preview.aligned
    status = replace(_status(), alignment=preview, left_open=False)
    pixels = draw_status_overlay(torch.zeros((3, 360, 640), dtype=torch.uint8), status)
    red = torch.tensor([245, 65, 65], dtype=torch.uint8)[:, None, None]
    red_pixels = torch.all(pixels[:, :260] == red, dim=0)
    assert not red_pixels.any()
    assert torch.all(pixels[:, 260:] == red, dim=0).any()
