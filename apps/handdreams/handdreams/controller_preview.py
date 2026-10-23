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

"""Display-only controller geometry for trigger practice and live input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from flashdreams_controller_input.events import ControllerFrameEvent, ControllerSnapshot
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION

from .alignment import AlignmentClaw
from .controllers.isaac_teleop_calibration import ParticipantPoseCalibrator
from .controllers.quest_controller import (
    WORLD_DOWN,
    both_controllers_tracked,
    controller_separation_basis,
    validate_controller_frame,
    validate_participant_basis,
)
from .rotations import quat_xyzw_to_matrix, validate_pose_track


def trigger_preview(
    event: ControllerFrameEvent | None,
) -> tuple[AlignmentClaw | None, AlignmentClaw | None]:
    """Show relative hand motion facing their shared horizontal pointing direction.

    Hand placement provides a fallback for opposing or vertical pointing rays.
    This moving display reference is never used for alignment or model input.
    """
    if event is None or event.action == "disconnected":
        return None, None
    hands = (event.left, event.right)
    tracked = [hand for hand in hands if hand is not None and hand.tracked]
    if not tracked:
        return None, None
    rotations = [quat_xyzw_to_matrix(hand.orientation) for hand in tracked]
    basis = np.diag([1.0, -1.0, -1.0])
    if len(tracked) == 2:
        try:
            basis = controller_separation_basis(event.to_controller_frame())
        except ValueError:
            pass
    pointing = -np.mean([rotation[:, 2] for rotation in rotations], axis=0)
    pointing[1] = 0
    length = np.linalg.norm(pointing)
    if length >= 1e-4:
        forward = pointing / length
        right = np.cross(WORLD_DOWN, forward)
        basis = np.stack([right, WORLD_DOWN, forward])

    midpoint = np.mean([hand.position for hand in tracked], axis=0)
    claws = []
    for index, hand in enumerate(hands):
        if hand is None or not hand.tracked:
            claws.append(None)
            continue
        position = (
            basis @ (np.asarray(hand.position) - midpoint)
            if len(tracked) == 2
            else np.array([-0.18 if index == 0 else 0.18, 0.0, 0.0])
        )
        rotation = basis @ quat_xyzw_to_matrix(hand.orientation) @ np.diag([-1, 1, -1])
        claws.append(AlignmentClaw(tuple(position), tuple(rotation.flat)))
    return claws[0], claws[1]


@dataclass(frozen=True, slots=True)
class ControllerPreviewCalibration:
    """Immutable model-to-UI reference for calibrated input, not shaped commands."""

    neutral: tuple[ControllerSnapshot, ControllerSnapshot]
    """Canonical wrist poses captured when calibration succeeds."""
    targets: tuple[AlignmentClaw, AlignmentClaw]
    """Scene wrist anchors, or participant-space anchors without scene metadata."""
    participant_basis: tuple[float, ...]
    """Row-major frozen participant basis used by the control pipeline."""
    position_gain: float
    """Gain applied to displacement from each neutral wrist."""

    @classmethod
    def capture(
        cls,
        neutral: dict[str, Any],
        wrist_poses: Mapping[str, Any] | None,
        *,
        participant_basis: Any | None,
        position_gain: float,
    ) -> ControllerPreviewCalibration:
        """Copy the accepted calibration and scene anchors without retaining arrays."""
        if neutral.get("pose_convention") != WRIST_PROXY_POSE_CONVENTION:
            raise ValueError("controller preview requires canonical wrist poses")
        frame = validate_controller_frame(neutral)
        if not both_controllers_tracked(frame):
            raise ValueError("controller preview calibration needs both tracked hands")
        basis = (
            controller_separation_basis(frame)
            if participant_basis is None
            else validate_participant_basis(participant_basis)
        )
        hands, targets = [], []
        for side in ("left", "right"):
            hand = frame[side]
            hands.append(
                ControllerSnapshot(
                    position=hand["position"], orientation=hand["orientation"]
                )
            )
            if wrist_poses is not None:
                pose = validate_pose_track(
                    np.asarray(wrist_poses[side])[None], side, minimum_frames=1
                )[0]
                position, rotation = pose[:3, 3], pose[:3, :3]
            else:
                position = basis @ hand["position"]
                rotation = (
                    basis
                    @ quat_xyzw_to_matrix(hand["orientation"])
                    @ np.diag([-1, 1, -1])
                )
            targets.append(
                AlignmentClaw(
                    tuple(float(v) for v in position),
                    tuple(float(v) for v in rotation.flat),
                )
            )
        return cls(
            neutral=(hands[0], hands[1]),
            targets=(targets[0], targets[1]),
            participant_basis=tuple(float(v) for v in basis.flat),
            position_gain=position_gain,
        )


class ControllerPreview:
    """Project fresh input using a UI-owned copy of the accepted calibration.

    This preview excludes stateful clutch and command shaping. It never calls
    the model's sidecar or alters the action pipeline.
    """

    def __init__(self, calibration: ControllerPreviewCalibration) -> None:
        self.calibration = calibration
        self._calibrator = ParticipantPoseCalibrator()
        neutral = {
            side: {"position": hand.position, "orientation": hand.orientation}
            for side, hand in zip(("left", "right"), calibration.neutral, strict=True)
        }
        self._anchor = self._calibrator.configure(
            neutral,
            np.asarray(calibration.participant_basis).reshape(3, 3),
            position_gain=calibration.position_gain,
        )

    def observe(
        self, event: ControllerFrameEvent
    ) -> tuple[AlignmentClaw | None, AlignmentClaw | None]:
        """Project canonical wrist input, omitting untracked hands independently.

        Raises:
            ValueError: The source has not supplied canonical wrist poses.
        """
        if event.action == "disconnected":
            return None, None
        if event.pose_convention != WRIST_PROXY_POSE_CONVENTION:
            raise ValueError("controller preview requires canonical wrist poses")
        frame = event.to_controller_frame()
        for side in ("left", "right"):
            hand = frame[side]
            if not hand["tracked"]:
                frame[side] = None
        calibrated = self._calibrator.transform(frame)
        claws = []
        for side, target in zip(
            ("left", "right"), self.calibration.targets, strict=True
        ):
            hand = calibrated[side]
            if hand is None:
                claws.append(None)
                continue
            position = (
                np.asarray(target.position)
                + np.asarray(hand["position"])
                - self._anchor[side]["position"]
            )
            rotation = quat_xyzw_to_matrix(hand["orientation"]) @ np.asarray(
                target.rotation
            ).reshape(3, 3)
            claws.append(
                AlignmentClaw(
                    tuple(float(v) for v in position),
                    tuple(float(v) for v in rotation.flat),
                )
            )
        return claws[0], claws[1]
