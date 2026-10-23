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

"""HandDreams controller validation and participant calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from flashdreams_controller_input.frames import (
    SIDES,
    SOURCE_CLIENT_CAPTURE_NS_KEY,
    ControllerFrame,
    ControllerPayload,
    as_mapping,
    controller_payload,
    controller_sides,
)
from flashdreams_controller_input.frames import (
    both_controllers_tracked as both_controllers_tracked,
)

from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
)
from handdreams.rotations import unit_vector

WORLD_DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float32)
"""Gravity direction in OpenXR world coordinates."""


def _controller_payload(raw: object) -> ControllerPayload | None:
    """Validate one controller and preserve HandDreams command fields."""
    payload = controller_payload(raw)
    if payload is None:
        return None
    value = as_mapping(raw)
    assert value is not None
    payload.pop("primary_button")
    payload.pop("secondary_button")
    if "gripper_open" in value:
        gripper_open = value["gripper_open"]
        if isinstance(gripper_open, (bool, np.bool_)) or not isinstance(
            gripper_open, (int, float, np.integer, np.floating)
        ):
            raise ValueError("controller gripper_open must be a number in [0, 1]")
        gripper_open = float(gripper_open)
        if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 1.0:
            raise ValueError("controller gripper_open must be a number in [0, 1]")
        payload["gripper_open"] = gripper_open
    return payload


def validate_controller_frame(raw: object) -> ControllerFrame:
    """Validate a controller observation at the Teleop pipeline boundary.

    Raises:
        ValueError: If the frame is not an object, lacks an integer
            ``client_capture_ns``, or carries a malformed participant basis or
            calibration epoch.
    """
    value = as_mapping(raw)
    if value is None:
        raise ValueError("controller frame must be an object")
    frame: ControllerFrame = {
        side: _controller_payload(value.get(side)) for side in SIDES
    }
    timestamp = value.get("client_capture_ns")
    if not isinstance(timestamp, int):
        raise ValueError("controller frame requires integer client_capture_ns")
    frame["client_capture_ns"] = timestamp
    if SOURCE_CLIENT_CAPTURE_NS_KEY in value:
        source_client_capture_ns = value[SOURCE_CLIENT_CAPTURE_NS_KEY]
        if (
            isinstance(source_client_capture_ns, bool)
            or not isinstance(source_client_capture_ns, int)
            or source_client_capture_ns < 0
        ):
            raise ValueError(
                f"controller frame {SOURCE_CLIENT_CAPTURE_NS_KEY} must be a "
                "non-negative integer"
            )
        frame[SOURCE_CLIENT_CAPTURE_NS_KEY] = source_client_capture_ns
    if "participant_basis" in value:
        frame["participant_basis"] = validate_participant_basis(
            value["participant_basis"]
        ).tolist()
    if CONTROLLER_CALIBRATION_EPOCH_KEY in value:
        epoch = value[CONTROLLER_CALIBRATION_EPOCH_KEY]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("controller calibration epoch must be an integer")
        if epoch < 1:
            raise ValueError("controller calibration epoch must be positive")
        frame[CONTROLLER_CALIBRATION_EPOCH_KEY] = epoch
    return frame


def validate_participant_basis(value: Any) -> np.ndarray:
    """Return a participant basis as ``float32`` after checking it is a rotation."""
    basis = np.asarray(value, dtype=np.float32)
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(basis), 1.0, atol=1e-4)
    ):
        raise ValueError(
            "controller participant_basis must be right-handed orthonormal"
        )
    return basis


def controller_separation_basis(frame: ControllerFrame) -> np.ndarray:
    """Derive a stable level frame from controller placement only.

    Controller pointing rays are deliberately excluded. When the controllers
    point toward one another those rays cancel and become numerically unstable,
    while the horizontal left-to-right hand axis remains well defined.
    """
    sides = controller_sides(frame)
    if not sides:
        raise ValueError("both Quest controllers are required for calibration")

    hand_right = np.asarray(sides["right"]["position"], dtype=np.float32) - np.asarray(
        sides["left"]["position"], dtype=np.float32
    )
    hand_right = hand_right - WORLD_DOWN * float(np.dot(hand_right, WORLD_DOWN))
    hand_right = unit_vector(hand_right, name="controller separation")
    hand_forward = unit_vector(np.cross(hand_right, WORLD_DOWN), name="hand forward")
    return np.stack([hand_right, WORLD_DOWN, hand_forward], axis=0).astype(np.float32)
