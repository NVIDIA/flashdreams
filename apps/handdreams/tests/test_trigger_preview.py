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

"""CPU checks for the display-only shared-forward trigger preview."""

from dataclasses import replace

import numpy as np
import pytest
from flashdreams_controller_input.events import ControllerFrameEvent, ControllerSnapshot
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION
from handdreams.controller_preview import trigger_preview
from handdreams.rotations import quat_xyzw_to_matrix

pytestmark = pytest.mark.ci_cpu


def _event():
    return ControllerFrameEvent(
        timestamp=np.uint64(1),
        action="state",
        frame_id=1,
        client_capture_ns=1000,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        left=ControllerSnapshot(position=(-0.2, 1, -0.4), orientation=(0, 0, 0, 1)),
        right=ControllerSnapshot(position=(0.2, 1, -0.4), orientation=(0, 0, 0, 1)),
    )


@pytest.mark.parametrize("yaw", [-2.7, -1.5, -0.6, 0.6, 1.5, 2.7])
def test_room_heading_and_origin_do_not_turn_the_trigger_preview(yaw):
    event = _event()
    quaternion = (0, np.sin(yaw / 2), 0, np.cos(yaw / 2))
    rotation = quat_xyzw_to_matrix(quaternion)
    turned = replace(
        event,
        **{
            side: replace(
                hand,
                position=tuple(rotation @ hand.position + (3, 0.5, -2)),
                orientation=quaternion,
            )
            for side, hand in (("left", event.left), ("right", event.right))
        },
    )
    for actual, expected in zip(
        trigger_preview(turned), trigger_preview(event), strict=True
    ):
        assert actual is not None
        assert expected is not None
        np.testing.assert_allclose(actual.position, expected.position, atol=1e-6)
        np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-6)


@pytest.mark.parametrize(
    "degrees", [-180, -120, -90, -61, 45, 60, 61, 89, 90, 91, 120, 180]
)
def test_shared_pointing_direction_faces_forward_without_mutating_input(degrees):
    event = _event()
    yaw = np.deg2rad(degrees)
    quaternion = (0, np.sin(yaw / 2), 0, np.cos(yaw / 2))
    turned = replace(
        event,
        left=replace(event.left, orientation=quaternion),
        right=replace(event.right, orientation=quaternion),
    )
    before = turned.to_controller_frame()
    for claw in trigger_preview(turned):
        assert claw is not None
        np.testing.assert_allclose(
            np.reshape(claw.rotation, (3, 3))[:, 2], (0, 0, 1), atol=1e-6
        )
    assert turned.to_controller_frame() == before


def test_relative_translation_pitch_and_roll_remain_visible():
    event = _event()
    neutral = trigger_preview(event)
    moved = replace(event, left=replace(event.left, position=(-0.3, 1.1, -0.5)))
    assert neutral[0] is not None
    moved_claw = trigger_preview(moved)[0]
    assert moved_claw is not None
    assert moved_claw.position != neutral[0].position
    for orientation in ((0.3, 0, 0, 0.95), (0, 0, 0.3, 0.95)):
        turned = replace(event, left=replace(event.left, orientation=orientation))
        turned_claw = trigger_preview(turned)[0]
        assert turned_claw is not None
        assert not np.allclose(turned_claw.rotation, neutral[0].rotation)


@pytest.mark.parametrize(
    "left,right",
    [
        ((0, 1, 0, 1), (0, -1, 0, 1)),
        ((1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 0, 1), (0, 0, 0, -1)),
    ],
)
def test_opposing_vertical_and_antipodal_poses_have_a_finite_display(left, right):
    event = _event()
    event = replace(
        event,
        left=replace(event.left, orientation=left),
        right=replace(event.right, orientation=right),
    )
    for claw in trigger_preview(event):
        assert claw is not None
        rotation = np.reshape(claw.rotation, (3, 3))
        assert np.isfinite(claw.position).all()
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(rotation) == pytest.approx(1)


def test_tracking_loss_and_coincident_hands_do_not_use_invalid_poses():
    event = _event()
    coincident = replace(
        event, right=replace(event.right, position=event.left.position)
    )
    assert all(claw is not None for claw in trigger_preview(coincident))
    missing = replace(
        event, right=replace(event.right, tracked=False, position=(999, 999, 999))
    )
    left, right = trigger_preview(missing)
    assert right is None
    assert left is not None
    np.testing.assert_allclose(left.position, (-0.18, 0, 0))
    missing = replace(missing, left=replace(missing.left, tracked=False))
    assert trigger_preview(missing) == (None, None)
    assert trigger_preview(None) == (None, None)
    assert trigger_preview(
        ControllerFrameEvent(timestamp=np.uint64(2), action="disconnected")
    ) == (None, None)
