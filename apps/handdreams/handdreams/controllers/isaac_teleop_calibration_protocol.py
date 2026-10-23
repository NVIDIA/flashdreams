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

"""Calibration and control RPC messages for the isolated Isaac Teleop bridge."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

if __package__:
    from handdreams.controllers.isaac_teleop_calibration import (
        CONTROLLER_CALIBRATION_EPOCH_KEY,
        HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
        POSITION_SCALING_CONVENTION,
        ParticipantPoseCalibrator,
    )
    from handdreams.controllers.isaac_teleop_control import (
        HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
        ControllerControlCandidate,
        ControllerControlConfig,
        ControllerControlShaper,
    )
    from handdreams.controllers.quest_pose import (
        HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    )
else:
    # The bridge executes in an isolated environment without installing the
    # FlashDreams package. Its directory is on sys.path, so sibling imports are
    # the only common path between package and direct-script execution.
    from isaac_teleop_calibration import (  # ty: ignore[unresolved-import]
        CONTROLLER_CALIBRATION_EPOCH_KEY,
        HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
        POSITION_SCALING_CONVENTION,
        ParticipantPoseCalibrator,
    )
    from isaac_teleop_control import (  # ty: ignore[unresolved-import]
        HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
        ControllerControlCandidate,
        ControllerControlConfig,
        ControllerControlShaper,
    )
    from quest_pose import (  # ty: ignore[unresolved-import]
        HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    )

CALIBRATION_RPC_VERSION = 4
"""Protocol version for participant-calibration control messages."""

CALIBRATION_REQUEST_TYPES = frozenset(
    {
        "calibration_configure",
        "calibration_reset",
        "control_prepare_window",
        "control_commit_window",
        "control_abort_window",
    }
)
"""Request message types accepted by :class:`CalibrationRequestHandler`."""


def calibration_hello(mode: str) -> dict[str, Any]:
    """Build the participant-calibration protocol capability message.

    Args:
        mode: Nonempty identifier for the controller transport using the RPC.

    Returns:
        Calibration protocol and pose-space capabilities.

    Raises:
        ValueError: ``mode`` is not a nonempty string.
    """
    if not isinstance(mode, str) or not mode:
        raise ValueError("calibration hello mode must be a nonempty string")
    return {
        "type": "calibration_hello",
        "protocol": CALIBRATION_RPC_VERSION,
        "mode": mode,
        "input_pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
        "output_pose_convention": HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
        "output_control_convention": (HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION),
        "position_scaling": POSITION_SCALING_CONVENTION,
        "atomic_window_transform": True,
    }


class _ControlTransaction:
    """One prepared command retained until it is superseded safely."""

    __slots__ = (
        "candidate",
        "command_sequence",
        "epoch",
        "frames",
        "request_digest",
        "status",
    )

    def __init__(
        self,
        *,
        epoch: int,
        command_sequence: int,
        request_digest: str,
        candidate: ControllerControlCandidate,
        frames: list[dict[str, Any]],
    ) -> None:
        self.epoch = epoch
        self.command_sequence = command_sequence
        self.request_digest = request_digest
        self.candidate = candidate
        self.frames = frames
        self.status = "prepared"


def calibration_error(request: object, error: BaseException) -> dict[str, Any]:
    """Build a serializable error response for a calibration request.

    Args:
        request: Original request, used only to recover a valid request ID.
        error: Failure raised while validating or handling the request.

    Returns:
        Error message safe to serialize over the bridge connection.
    """
    request_id: int | None = None
    if isinstance(request, Mapping):
        candidate = cast(Mapping[str, Any], request).get("request_id")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            request_id = candidate
    return {
        "type": "error",
        "request_id": request_id,
        "error": f"{type(error).__name__}: {error}",
    }


class CalibrationRequestHandler:
    """Own calibration state and serialize ordered RPC requests."""

    def __init__(self) -> None:
        self.calibrator = ParticipantPoseCalibrator()
        self.control_shaper = ControllerControlShaper()
        self._last_request_id: int | None = None
        self._control_transaction: _ControlTransaction | None = None
        self._last_command_sequence = -1
        self._lock = threading.Lock()

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and execute one ordered calibration request.

        A syntactically valid request ID is consumed before dispatch, including
        when the operation itself fails. Retrying therefore always requires a
        newer ID and cannot ambiguously replay a state-changing request.

        Args:
            request: Decoded calibration request object.

        Returns:
            Response carrying the matching request ID and calibration epoch.

        Raises:
            TypeError: ``request`` is not an object.
            ValueError: The request ID, type, or operation payload is invalid.
            RuntimeError: A transform is requested without active calibration.
        """
        if not isinstance(request, Mapping):
            raise TypeError("calibration request must be an object")
        with self._lock:
            request_id = self._accept_request_id(request.get("request_id"))
            request_type = request.get("type")
            if request_type not in CALIBRATION_REQUEST_TYPES:
                raise ValueError(f"unknown calibration request type {request_type!r}")
            if request_type == "calibration_configure":
                return self._configure(request, request_id)
            if request_type == "calibration_reset":
                return self._reset(request_id)
            if request_type == "control_prepare_window":
                return self._prepare_window(request, request_id)
            if request_type == "control_commit_window":
                return self._finish_window(request, request_id, commit=True)
            return self._finish_window(request, request_id, commit=False)

    def _accept_request_id(self, value: object) -> int:
        """Consume one strictly increasing non-boolean integer request ID."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("calibration request_id must be a non-boolean integer")
        if self._last_request_id is not None and value <= self._last_request_id:
            raise ValueError("calibration request_id must increase strictly")
        self._last_request_id = value
        return value

    def _configure(self, request: Mapping[str, Any], request_id: int) -> dict[str, Any]:
        """Install the participant calibration carried by one request."""
        calibration = request.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("calibration_configure requires an object calibration")
        if "participant_basis" not in calibration:
            raise ValueError("calibration_configure requires participant_basis")
        if "position_gain" not in calibration:
            raise ValueError("calibration_configure requires position_gain")
        control_payload = request.get("control")
        control_config = (
            None
            if control_payload is None
            else ControllerControlConfig.from_payload(control_payload)
        )
        epoch, _ = self.calibrator.configure_with_epoch(
            calibration,
            calibration["participant_basis"],
            position_gain=calibration["position_gain"],
        )
        position_gain = self.calibrator.position_gain
        if position_gain is None:  # pragma: no cover - configure published it
            raise RuntimeError("participant calibration did not retain position_gain")
        if control_config is None:
            self.control_shaper.reset()
            control_space = None
            normalized_control = None
        else:
            self.control_shaper.configure(control_config)
            control_space = HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
            normalized_control = control_config.to_payload()
        self._control_transaction = None
        self._last_command_sequence = -1
        return {
            "type": "calibration_configured",
            "request_id": request_id,
            "pose_space": HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
            "epoch": epoch,
            "position_gain": position_gain,
            "control_space": control_space,
            "control": normalized_control,
        }

    def _reset(self, request_id: int) -> dict[str, Any]:
        """Clear participant calibration and return its new epoch."""
        epoch = self.calibrator.reset()
        self.control_shaper.reset()
        self._control_transaction = None
        self._last_command_sequence = -1
        return {
            "type": "calibration_reset_done",
            "request_id": request_id,
            "epoch": epoch,
            "control_space": None,
        }

    @staticmethod
    def _positive_epoch(value: object) -> int:
        """Validate an epoch carried by a model-bound transaction."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("control expected_epoch must be a positive integer")
        return value

    @staticmethod
    def _command_sequence(value: object) -> int:
        """Validate the caller's idempotence key for one complete window."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("control command_sequence must be a non-negative integer")
        return value

    @staticmethod
    def _window_digest(frames: list[Mapping[str, Any]]) -> str:
        """Return a canonical digest used to distinguish exact retries."""
        try:
            encoded = json.dumps(
                frames,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError) as error:
            raise ValueError(
                "control frames must contain finite JSON-compatible values"
            ) from error
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _prepared_response(
        transaction: _ControlTransaction, request_id: int
    ) -> dict[str, Any]:
        """Serialize a prepared transaction without exposing cached objects."""
        return {
            "type": "control_window_prepared",
            "request_id": request_id,
            "pose_space": HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
            "control_space": HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
            "epoch": transaction.epoch,
            "command_sequence": transaction.command_sequence,
            "token": transaction.candidate.token,
            "frames": deepcopy(transaction.frames),
        }

    def _prepare_window(
        self, request: Mapping[str, Any], request_id: int
    ) -> dict[str, Any]:
        """Prepare one stateful window without advancing live control state."""
        expected_epoch = self._positive_epoch(request.get("expected_epoch"))
        command_sequence = self._command_sequence(request.get("command_sequence"))
        frames = request.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(
                "control_prepare_window requires a nonempty list of frames"
            )
        if any(not isinstance(frame, Mapping) for frame in frames):
            raise ValueError("each control frame must be an object")
        typed_frames = cast(list[Mapping[str, Any]], frames)
        request_digest = self._window_digest(typed_frames)

        transaction = self._control_transaction
        if transaction is not None and command_sequence == transaction.command_sequence:
            if (
                expected_epoch != transaction.epoch
                or request_digest != transaction.request_digest
            ):
                raise ValueError(
                    "control command_sequence was reused with different input"
                )
            return self._prepared_response(transaction, request_id)
        if command_sequence <= self._last_command_sequence:
            raise ValueError("control command_sequence must increase strictly")
        if transaction is not None and transaction.status == "prepared":
            raise RuntimeError(
                "the previous control window must be committed or aborted first"
            )
        if not self.control_shaper.active:
            raise RuntimeError("controller control shaping is not configured")

        epoch, calibrated = self.calibrator.transform_many_with_epoch(
            typed_frames,
            require_active=True,
        )
        if epoch != expected_epoch:
            raise ValueError(
                f"calibration epoch mismatch: expected {expected_epoch}, active {epoch}"
            )
        for frame in calibrated:
            frame[CONTROLLER_CALIBRATION_EPOCH_KEY] = epoch
        candidate = self.control_shaper.prepare(typed_frames, calibrated)
        prepared_frames = candidate.frames
        transaction = _ControlTransaction(
            epoch=epoch,
            command_sequence=command_sequence,
            request_digest=request_digest,
            candidate=candidate,
            frames=prepared_frames,
        )
        self._control_transaction = transaction
        self._last_command_sequence = command_sequence
        return self._prepared_response(transaction, request_id)

    def _finish_window(
        self,
        request: Mapping[str, Any],
        request_id: int,
        *,
        commit: bool,
    ) -> dict[str, Any]:
        """Commit or abort one prepared window with idempotent terminal replies."""
        expected_epoch = self._positive_epoch(request.get("expected_epoch"))
        command_sequence = self._command_sequence(request.get("command_sequence"))
        token = request.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("control transaction token must be a nonempty string")
        transaction = self._control_transaction
        if transaction is None:
            raise ValueError("control transaction is not prepared")
        if command_sequence != transaction.command_sequence:
            raise ValueError("control command_sequence does not match prepared window")
        if expected_epoch != transaction.epoch:
            raise ValueError("control expected_epoch does not match prepared window")
        if expected_epoch != self.calibrator.epoch:
            raise ValueError("control transaction belongs to an inactive epoch")
        if token != transaction.candidate.token:
            raise ValueError("control transaction token does not match prepared window")

        desired_status = "committed" if commit else "aborted"
        if transaction.status != "prepared" and transaction.status != desired_status:
            raise ValueError(f"control transaction was already {transaction.status}")
        if transaction.status == "prepared":
            if commit:
                self.control_shaper.commit(transaction.candidate)
            else:
                self.control_shaper.abort(transaction.candidate)
            transaction.status = desired_status
        return {
            "type": f"control_window_{desired_status}",
            "request_id": request_id,
            "control_space": HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
            "epoch": transaction.epoch,
            "command_sequence": transaction.command_sequence,
            "token": transaction.candidate.token,
        }
