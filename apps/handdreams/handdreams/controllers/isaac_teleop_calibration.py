# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-frame calibration shared by the Isaac Teleop input paths.

The input to this module is already a HandDreams wrist proxy: the fixed
controller-local grip-to-wrist offset has been applied by the upstream Isaac
``Se3AbsRetargeter``. Calibration therefore contains no translational tool
offset of its own. It maps the tracking reference frame into the frozen
participant frame, applies movement gain around each neutral wrist position,
and cancels each controller's neutral orientation.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION = "handdreams-participant-wrist-v2"
"""Wrist proxy expressed in the calibrated HandDreams participant frame."""
POSITION_SCALING_CONVENTION = "neutral-relative"
"""Position gain is applied to displacement from the calibrated neutral pose."""
CONTROLLER_CALIBRATION_EPOCH_KEY = "controller_calibration_epoch"
"""Frame field binding a model window to one participant calibration."""

_SIDES = ("left", "right")
_ROTATION_ATOL = 2e-3


def _rotation(value: object, *, name: str) -> np.ndarray:
    """Validate and return one right-handed ``[3,3]`` rotation."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [3,3] rotation")
    if not np.allclose(
        result @ result.T,
        np.eye(3),
        atol=_ROTATION_ATOL,
        rtol=_ROTATION_ATOL,
    ) or not np.isclose(
        np.linalg.det(result),
        1.0,
        atol=_ROTATION_ATOL,
        rtol=_ROTATION_ATOL,
    ):
        raise ValueError(f"{name} must be right-handed and orthonormal")
    # Inputs normally originate as float32 quaternion/basis products. Project
    # their tolerated round-off back onto SO(3) so repeated multiplication does
    # not amplify it into a rotation that the output validator rejects.
    left, _, right = np.linalg.svd(result)
    projected = left @ right
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right
    return projected


def quaternion_to_matrix(value: object, *, name: str) -> np.ndarray:
    """Convert one finite nonzero XYZW quaternion to a rotation matrix."""
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"{name} must be a finite XYZW quaternion[4]")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError(f"{name} has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert an already-projected rotation to a float32 XYZW quaternion."""
    # Select the largest quaternion component to keep the conversion stable at
    # and around rotations of pi, where a trace-only formula loses precision.
    candidates = np.asarray(
        [
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
            1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
            1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
            1.0 + np.trace(matrix),
        ],
        dtype=np.float64,
    )
    index = int(np.argmax(candidates))
    root = 0.5 * np.sqrt(max(0.0, float(candidates[index])))
    if root < 1e-8:
        raise ValueError("calibrated controller orientation is degenerate")
    scale = 0.25 / root
    if index == 0:
        quaternion = np.asarray(
            [
                root,
                (matrix[0, 1] + matrix[1, 0]) * scale,
                (matrix[0, 2] + matrix[2, 0]) * scale,
                (matrix[2, 1] - matrix[1, 2]) * scale,
            ]
        )
    elif index == 1:
        quaternion = np.asarray(
            [
                (matrix[0, 1] + matrix[1, 0]) * scale,
                root,
                (matrix[1, 2] + matrix[2, 1]) * scale,
                (matrix[0, 2] - matrix[2, 0]) * scale,
            ]
        )
    elif index == 2:
        quaternion = np.asarray(
            [
                (matrix[0, 2] + matrix[2, 0]) * scale,
                (matrix[1, 2] + matrix[2, 1]) * scale,
                root,
                (matrix[1, 0] - matrix[0, 1]) * scale,
            ]
        )
    else:
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) * scale,
                (matrix[0, 2] - matrix[2, 0]) * scale,
                (matrix[1, 0] - matrix[0, 1]) * scale,
                root,
            ]
        )
    quaternion /= np.linalg.norm(quaternion)
    # q and -q describe the same rotation. A canonical sign makes serialized
    # calibration output deterministic without changing its pose.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def _matrix_to_quaternion(value: np.ndarray) -> np.ndarray:
    """Validate a proper rotation and return its normalized XYZW quaternion."""
    return rotation_to_quaternion(
        _rotation(value, name="calibrated controller orientation")
    )


def _pose(payload: object, *, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one controller position and rotation."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"participant calibration requires the {side} controller")
    controller = cast(Mapping[str, Any], payload)
    position = np.asarray(controller.get("position"), dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"{side} controller position must contain 3 finite values")
    orientation = quaternion_to_matrix(
        controller.get("orientation"), name=f"{side} controller orientation"
    )
    return position, orientation


def _position_gain(value: object) -> float:
    """Validate and return a finite positive controller movement gain."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("position_gain must be a finite positive number")
    gain = float(value)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("position_gain must be a finite positive number")
    return gain


@dataclass(frozen=True, slots=True)
class _Calibration:
    """Immutable state swapped atomically by :class:`ParticipantPoseCalibrator`."""

    basis: np.ndarray
    neutral_positions: dict[str, np.ndarray]
    orientation_offsets: dict[str, np.ndarray]
    position_gain: float


class ParticipantPoseCalibrator:
    """Map wrist proxies into a stable participant frame for both Isaac routes.

    ``configure`` captures a participant basis ``B`` and each hand's neutral
    wrist rotation ``R0``. Subsequent poses use exactly::

        c = B @ p0
        p_calibrated = c + g * (B @ p - c)
        R_calibrated = B @ R @ (B @ R0).T

    Thus both neutral rotations become identity while relative wrist motion is
    preserved in the participant axes. Configuration, reset, and snapshots are
    protected by a lock so a streaming thread can call :meth:`transform` while
    a control thread recalibrates. A transform always uses one complete state;
    it can never mix the basis from one calibration with offsets from another.

    The calibrator intentionally has no grip-to-wrist distance argument. Its
    input positions must already include the upstream fixed local offset, which
    prevents the HandDreams 5 cm correction from being applied twice.
    """

    pose_convention = HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calibration: _Calibration | None = None
        self._epoch = 0

    @property
    def active(self) -> bool:
        """Return whether a participant calibration is installed."""
        with self._lock:
            return self._calibration is not None

    @property
    def epoch(self) -> int:
        """Return the generation changed by every configure or reset call."""
        with self._lock:
            return self._epoch

    @property
    def position_gain(self) -> float | None:
        """Return the active neutral-relative gain, or ``None`` when reset."""
        with self._lock:
            state = self._calibration
            return None if state is None else state.position_gain

    def configure(
        self,
        neutral_frame: Mapping[str, Any],
        participant_basis: object,
        *,
        position_gain: object = 1.0,
    ) -> dict[str, Any]:
        """Atomically install a calibration and return its transformed anchor."""
        _, anchor = self.configure_with_epoch(
            neutral_frame, participant_basis, position_gain=position_gain
        )
        return anchor

    def configure_with_epoch(
        self,
        neutral_frame: Mapping[str, Any],
        participant_basis: object,
        *,
        position_gain: object = 1.0,
    ) -> tuple[int, dict[str, Any]]:
        """Install a calibration and return its exact epoch and anchor.

        Validation completes before the live state changes. A malformed request
        therefore leaves the previous calibration and epoch intact. Returning
        the epoch from the publication lock prevents a concurrent reconfigure
        from pairing this call's anchor with another call's generation.
        """
        if not isinstance(neutral_frame, Mapping):
            raise ValueError("participant calibration frame must be an object")
        basis = _rotation(participant_basis, name="participant basis").copy()
        gain = _position_gain(position_gain)
        poses = {side: _pose(neutral_frame.get(side), side=side) for side in _SIDES}
        state = _Calibration(
            basis=basis,
            neutral_positions={side: basis @ poses[side][0] for side in _SIDES},
            orientation_offsets={side: (basis @ poses[side][1]).T for side in _SIDES},
            position_gain=gain,
        )
        # Render the anchor before publishing state. Besides producing the
        # return value, this guarantees a bad request cannot install a
        # calibration and then fail while serializing its first pose.
        anchor = self._transform_with(neutral_frame, state)
        with self._lock:
            self._calibration = state
            self._epoch += 1
            epoch = self._epoch
        return epoch, anchor

    def reset(self) -> int:
        """Disable calibration and return the newly assigned epoch."""
        with self._lock:
            self._calibration = None
            self._epoch += 1
            return self._epoch

    def transform(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Transform a frame with one atomic state snapshot.

        When inactive this remains a copying pass-through, which lets callers
        use the same output path during setup and after a session reset.
        """
        return self.transform_many([frame])[0]

    def transform_many(
        self,
        frames: Sequence[Mapping[str, Any]],
        *,
        require_active: bool = False,
    ) -> list[dict[str, Any]]:
        """Transform one ordered window using a single calibration snapshot.

        This is the model-window boundary used by the live input adapters. A
        concurrent recenter or reset may happen before or after the batch, but
        can never split it across two participant frames and synthesize a false
        pose delta. ``require_active`` closes the live adapter's check/use race:
        reset cannot silently turn a calibrated window into raw pass-through.
        """
        _, transformed = self.transform_many_with_epoch(
            frames, require_active=require_active
        )
        return transformed

    def transform_many_with_epoch(
        self,
        frames: Sequence[Mapping[str, Any]],
        *,
        require_active: bool = False,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return one window and the exact calibration epoch used for it.

        Callers can carry the epoch through asynchronous upload and let the
        receiver reject a window transformed before a live recenter. The epoch
        and immutable state are captured under the same lock.
        """
        if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
            raise ValueError("controller frames must be a sequence")
        if any(not isinstance(frame, Mapping) for frame in frames):
            raise ValueError("each controller frame must be an object")
        with self._lock:
            state = self._calibration
            epoch = self._epoch
        if state is None:
            if require_active:
                raise RuntimeError("participant calibration is not active")
            return epoch, [deepcopy(dict(frame)) for frame in frames]
        return epoch, [self._transform_with(frame, state) for frame in frames]

    @staticmethod
    def _transform_with(
        frame: Mapping[str, Any], state: _Calibration
    ) -> dict[str, Any]:
        """Apply one immutable calibration while retaining all other fields."""
        result = deepcopy(dict(frame))
        for side in _SIDES:
            payload = frame.get(side)
            if payload is None:
                continue
            position, rotation = _pose(payload, side=side)
            adjusted = deepcopy(dict(payload))
            basis_position = state.basis @ position
            neutral_position = state.neutral_positions[side]
            adjusted["position"] = (
                (
                    neutral_position
                    + state.position_gain * (basis_position - neutral_position)
                )
                .astype(np.float32)
                .tolist()
            )
            adjusted["orientation"] = _matrix_to_quaternion(
                state.basis @ rotation @ state.orientation_offsets[side]
            ).tolist()
            result[side] = adjusted
        return result
