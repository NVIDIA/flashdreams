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

"""Frozen participant alignment geometry for the V2 setup preview."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .controllers.quest_controller import (
    both_controllers_tracked,
    controller_separation_basis,
    validate_controller_frame,
)
from .controllers.quest_pose import HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
from .rotations import quat_xyzw_to_matrix, validate_pose_track

CALIBRATION_TRIGGER_THRESHOLD = 0.9
"""Minimum claw-closing trigger value for the calibration hold and its UI."""

_POSITION_TOLERANCE_M = 0.12
"""Maximum midpoint-relative wrist displacement accepted during setup."""

_ROTATION_TOLERANCE_DEG = 35.0
"""Maximum claw pointing error; roll around the forward axis is unrestricted."""

_CONTROLLER_TO_CLAW = np.diag([-1.0, 1.0, -1.0])
"""Map the controller's local -Z pointing direction to the claw's local +Z."""


@dataclass(frozen=True, slots=True)
class AlignmentClaw:
    """Immutable wrist geometry shared with the UI thread."""

    position: tuple[float, ...]
    """Camera-coordinate position in meters."""
    rotation: tuple[float, ...]
    """Row-major camera-coordinate rotation matrix."""
    position_error_m: float = 0.0
    """Distance from the recorded scene wrist."""
    rotation_error_deg: float = 0.0
    """Angle from the recorded scene claw's forward direction."""

    @property
    def aligned(self) -> bool:
        """Return whether both wrist tolerances are satisfied."""
        return (
            self.position_error_m <= _POSITION_TOLERANCE_M
            and self.rotation_error_deg <= _ROTATION_TOLERANCE_DEG
        )


@dataclass(frozen=True, slots=True)
class AlignmentPreview:
    """Alignment snapshot with live geometry only for a tracked controller pair."""

    left: AlignmentClaw | None
    """Current left wrist, or ``None`` without a tracked pair."""
    right: AlignmentClaw | None
    """Current right wrist, or ``None`` without a tracked pair."""
    left_target: AlignmentClaw
    """Recorded left wrist at the start of the scene."""
    right_target: AlignmentClaw
    """Recorded right wrist at the start of the scene."""

    @property
    def aligned(self) -> bool:
        """Require both tracked wrists to satisfy the alignment tolerances."""
        return (
            self.left is not None
            and self.right is not None
            and self.left.aligned
            and self.right.aligned
        )

    @property
    def progress(self) -> float:
        """Return geometric closeness for the setup bar, not hold-time progress."""
        if self.left is None or self.right is None:
            return 0.0
        position_error = np.sqrt(
            (self.left.position_error_m**2 + self.right.position_error_m**2) / 2
        )
        rotation_error = max(
            self.left.rotation_error_deg, self.right.rotation_error_deg
        )
        return float(
            np.clip(
                min(
                    1 - position_error / (3 * _POSITION_TOLERANCE_M),
                    1 - rotation_error / (3 * _ROTATION_TOLERANCE_DEG),
                ),
                0.0,
                1.0,
            )
        )


class AlignmentGuide:
    """Map canonical wrists into a participant-relative setup guide.

    Hand separation freezes the participant heading without using controller
    pointing. The current midpoint keeps alignment location-independent, while
    absolute claw directions expose sideways or backward controllers. Movement
    gain belongs to live controls, not this physical-pose alignment check.
    """

    def __init__(
        self,
        frame: dict[str, Any],
        wrist_poses: Mapping[str, Any],
    ) -> None:
        frame = self._validate_frame(frame)
        if not both_controllers_tracked(frame):
            raise ValueError("alignment needs both tracked controllers")
        self._basis = controller_separation_basis(frame)
        self._targets = {}
        for side in ("left", "right"):
            if side not in wrist_poses:
                raise ValueError(f"alignment scene is missing the {side} wrist pose")
            pose = validate_pose_track(
                np.asarray(wrist_poses[side])[None], side, minimum_frames=1
            )[0].copy()
            self._targets[side] = pose
        self._scene_midpoint = np.mean(
            [pose[:3, 3] for pose in self._targets.values()], axis=0
        )

    @staticmethod
    def _validate_frame(frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("pose_convention") != HANDDREAMS_WRIST_PROXY_POSE_CONVENTION:
            raise ValueError("alignment requires canonical wrist-proxy poses")
        return validate_controller_frame(frame)

    @property
    def participant_basis(self) -> np.ndarray:
        """Return a copy of the frozen basis used for final calibration."""
        return self._basis.copy()

    def observe(self, frame: dict[str, Any]) -> AlignmentPreview:
        """Snapshot both wrists relative to their tracked pair's current midpoint."""
        frame = self._validate_frame(frame)
        midpoint = (
            np.mean([frame[side]["position"] for side in ("left", "right")], axis=0)
            if both_controllers_tracked(frame)
            else None
        )
        hands: dict[str, AlignmentClaw | None] = {}
        targets: dict[str, AlignmentClaw] = {}
        for side, target in self._targets.items():
            targets[side] = AlignmentClaw(
                tuple(float(v) for v in target[:3, 3]),
                tuple(float(v) for v in target[:3, :3].flat),
            )
            hand = frame.get(side)
            if midpoint is None:
                hands[side] = None
                continue
            assert hand is not None
            position = self._scene_midpoint + self._basis @ (
                np.asarray(hand["position"]) - midpoint
            )
            rotation = (
                self._basis
                @ quat_xyzw_to_matrix(hand["orientation"])
                @ _CONTROLLER_TO_CLAW
            )
            forward, target_forward = rotation[:, 2], target[:3, 2]
            cosine = np.clip(
                np.dot(forward, target_forward)
                / (np.linalg.norm(forward) * np.linalg.norm(target_forward)),
                -1.0,
                1.0,
            )
            hands[side] = AlignmentClaw(
                tuple(float(v) for v in position),
                tuple(float(v) for v in rotation.flat),
                float(np.linalg.norm(position - target[:3, 3])),
                float(np.degrees(np.arccos(cosine))),
            )
        return AlignmentPreview(
            left=hands["left"],
            right=hands["right"],
            left_target=targets["left"],
            right_target=targets["right"],
        )
