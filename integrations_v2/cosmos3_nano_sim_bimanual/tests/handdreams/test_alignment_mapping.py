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

"""Physical alignment and participant-axis action encoding regressions."""

import copy
from typing import Any

import numpy as np
import pytest
from cosmos3_nano_sim_bimanual.apps.handdreams.actions import (
    controller_targets_to_raw_action,
)
from handdreams.alignment import AlignmentGuide
from handdreams.controllers.isaac_teleop_calibration import (
    ParticipantPoseCalibrator,
    rotation_to_quaternion,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
)
from handdreams.rotations import matrix_to_rot6d, quat_xyzw_to_matrix

pytestmark = pytest.mark.ci_cpu


def _rotation(axis, radians):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return quat_xyzw_to_matrix([*(axis * np.sin(radians / 2)), np.cos(radians / 2)])


@pytest.mark.parametrize("gain", [0.8, 1.2])
@pytest.mark.parametrize("axis", np.eye(3).tolist())
def test_accepted_alignment_preserves_live_participant_axes(gain, axis) -> None:
    yaw = _rotation([0, 1, 0], 0.6)
    basis = np.diag([1, -1, -1]) @ yaw.T
    controller_to_claw = np.diag([-1, 1, -1])
    neutral: dict[str, Any] = {
        "client_capture_ns": 1,
        "pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    }
    targets = {}
    for side, sign in (("left", -1), ("right", 1)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _rotation([1, 2, sign], sign * 0.3)
        pose[:3, 3] = [sign * 0.2, 0.12, 0.24]
        targets[side] = pose
        neutral[side] = {
            "position": (yaw @ [sign * 0.2, 1.1, -0.4]).tolist(),
            "orientation": rotation_to_quaternion(
                basis.T @ pose[:3, :3] @ controller_to_claw.T
            ).tolist(),
            "tracked": True,
            "trigger": 0.0,
            "gripper_open": 0.8,
        }
    held = copy.deepcopy(neutral)
    for side in ("left", "right"):
        held[side]["orientation"] = rotation_to_quaternion(
            yaw @ _rotation([0, 1, 0], np.pi / 2)
        ).tolist()
    guide = AlignmentGuide(held, targets)
    assert not guide.observe(held).aligned
    np.testing.assert_allclose(guide.participant_basis, basis, atol=1e-6)
    preview = guide.observe(neutral)
    assert preview.aligned
    for side in ("left", "right"):
        np.testing.assert_allclose(
            np.asarray(getattr(preview, side).rotation).reshape(3, 3),
            targets[side][:3, :3],
            atol=1e-6,
        )

    # Capture the accepted physical pose, not the earlier trigger-hold pose.
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(neutral, guide.participant_basis, position_gain=gain)
    moved = copy.deepcopy(neutral)
    moved["client_capture_ns"] = 2
    participant_rotation = _rotation(axis, 0.18)
    for side, sign in (("left", -1), ("right", 1)):
        moved[side]["position"] = (
            np.asarray(neutral[side]["position"])
            + basis.T @ [sign * 0.025, 0.015, -0.02]
        ).tolist()
        moved[side]["orientation"] = rotation_to_quaternion(
            basis.T
            @ participant_rotation
            @ basis
            @ quat_xyzw_to_matrix(neutral[side]["orientation"])
        ).tolist()
        moved[side]["gripper_open"] = 0.6

    calibrated_frames = [calibrator.transform(frame) for frame in (neutral, moved)]
    live_actions, _ = controller_targets_to_raw_action(
        calibrated_frames,
        wrist_rotations={side: pose[:3, :3] for side, pose in targets.items()},
    )
    for side, sign, start in (("left", -1, 34), ("right", 1, 9)):
        np.testing.assert_allclose(
            quat_xyzw_to_matrix(calibrated_frames[0][side]["orientation"]),
            np.eye(3),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            quat_xyzw_to_matrix(calibrated_frames[1][side]["orientation"]),
            participant_rotation,
            atol=1e-6,
        )
        scene_rotation = targets[side][:3, :3]
        np.testing.assert_allclose(
            live_actions[0, start : start + 3],
            scene_rotation.T @ (gain * np.array([sign * 0.025, 0.015, -0.02])),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            live_actions[0, start + 3 : start + 9],
            matrix_to_rot6d(scene_rotation.T @ participant_rotation @ scene_rotation),
            atol=1e-6,
        )
