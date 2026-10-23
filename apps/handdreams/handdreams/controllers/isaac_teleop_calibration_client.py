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

"""Validated host client for the Isaac Teleop participant-calibration RPC."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from handdreams.controllers import quest_controller
from handdreams.controllers.controller_types import (
    LiveCalibrationState,
)
from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
    POSITION_SCALING_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_protocol import (
    CALIBRATION_RPC_VERSION,
)
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
    ControllerControlConfig,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
)

_MAX_MESSAGE_BYTES = 1024 * 1024
"""Maximum request or response size accepted from the private sidecar."""

RoundTrip = Callable[[dict[str, Any]], dict[str, Any]]
"""One serialized request/response exchange with the isolated sidecar."""


@dataclass(frozen=True, slots=True)
class PreparedControlWindow:
    """Opaque prepared sidecar transaction plus its model-facing frames."""

    frames: list[dict[str, Any]]
    epoch: int
    command_sequence: int
    token: str


class JsonSocketConnection:
    """Bounded newline-delimited JSON exchanges over one full-duplex socket."""

    def __init__(
        self,
        connection: socket.socket,
        *,
        timeout_s: float,
        peer_name: str,
    ) -> None:
        self._connection: socket.socket | None = connection
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._peer_name = str(peer_name)
        connection.settimeout(float(timeout_s))

    def set_timeout(self, timeout_s: float) -> None:
        """Set the timeout for subsequent request/response exchanges."""
        connection = self._connection
        if connection is None:
            raise RuntimeError(f"{self._peer_name} is closed")
        connection.settimeout(float(timeout_s))

    def receive(self) -> dict[str, Any]:
        """Receive and decode one bounded JSON object."""
        connection = self._connection
        if connection is None:
            raise RuntimeError(f"{self._peer_name} is closed")
        while b"\n" not in self._buffer:
            try:
                chunk = connection.recv(65536)
            except TimeoutError as error:
                raise TimeoutError(f"{self._peer_name} timed out") from error
            except OSError as error:
                raise RuntimeError(f"{self._peer_name} receive failed") from error
            if not chunk:
                raise RuntimeError(f"{self._peer_name} disconnected")
            self._buffer.extend(chunk)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise RuntimeError(f"{self._peer_name} response exceeds 1 MiB")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(f"{self._peer_name} returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"{self._peer_name} response must be an object")
        return payload

    def round_trip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and synchronously receive its response."""
        record = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        if len(record) > _MAX_MESSAGE_BYTES:
            raise ValueError(f"{self._peer_name} request exceeds 1 MiB")
        with self._lock:
            connection = self._connection
            if connection is None:
                raise RuntimeError(f"{self._peer_name} is closed")
            try:
                connection.sendall(record)
            except (BrokenPipeError, ConnectionError, OSError) as error:
                raise RuntimeError(f"{self._peer_name} disconnected") from error
            return self.receive()

    @property
    def ready(self) -> bool:
        """Return whether the socket remains owned by this client."""
        return self._connection is not None

    def close(self) -> None:
        """Close the socket and wake any blocked exchange."""
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()
        self._buffer.clear()


class CalibrationRpcClient:
    """Own calibration state while all pose math executes in the sidecar."""

    def __init__(
        self,
        *,
        hello: dict[str, Any],
        expected_mode: str,
        round_trip: RoundTrip,
    ) -> None:
        """Validate the sidecar contract and bind its request transport."""
        self._validate_hello(hello, expected_mode=expected_mode)
        self._round_trip = round_trip
        self._request_id = 0
        self._last_epoch = 0
        self._state: LiveCalibrationState | None = None
        self._failure: BaseException | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _validate_hello(hello: dict[str, Any], *, expected_mode: str) -> None:
        expected = {
            "type": "calibration_hello",
            "protocol": CALIBRATION_RPC_VERSION,
            "mode": expected_mode,
            "input_pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
            "output_pose_convention": HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
            "output_control_convention": (
                HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
            ),
            "position_scaling": POSITION_SCALING_CONVENTION,
            "atomic_window_transform": True,
        }
        for field, value in expected.items():
            if hello.get(field) != value:
                raise RuntimeError(
                    f"Isaac Teleop calibration has incompatible {field}: "
                    f"{hello.get(field)!r}; expected {value!r}"
                )

    @property
    def state(self) -> LiveCalibrationState | None:
        """Return the last calibration state acknowledged by the sidecar."""
        with self._lock:
            return self._state

    @property
    def ready(self) -> bool:
        """Return whether ordered calibration exchanges remain trustworthy."""
        with self._lock:
            return self._failure is None

    def _poison(self, error: BaseException) -> RuntimeError:
        """Invalidate client state after an ambiguous or malformed exchange."""
        self._failure = error
        self._state = None
        return RuntimeError(
            "Isaac Teleop calibration RPC is unusable after "
            f"{type(error).__name__}: {error}"
        )

    def _require_ready(self) -> None:
        """Reject requests after an exchange made ordering ambiguous."""
        failure = self._failure
        if failure is not None:
            raise RuntimeError(
                "Isaac Teleop calibration RPC is unusable after "
                f"{type(failure).__name__}: {failure}"
            ) from failure

    def _epoch(self, value: object, *, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            error = RuntimeError(f"{context} returned an invalid calibration epoch")
            raise self._poison(error) from error
        return value

    def _pose_space(self, response: dict[str, Any], *, context: str) -> None:
        actual = response.get("pose_space")
        if actual != HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION:
            error = RuntimeError(
                f"{context} returned incompatible pose space {actual!r}"
            )
            raise self._poison(error) from error

    def _control_space(
        self,
        response: dict[str, Any],
        *,
        expected: str | None,
        context: str,
    ) -> None:
        """Fail closed unless the response names the configured control space."""
        actual = response.get("control_space")
        if actual != expected:
            error = RuntimeError(
                f"{context} returned incompatible control space {actual!r}"
            )
            raise self._poison(error) from error

    @staticmethod
    def _control_payload(control: object) -> dict[str, Any] | None:
        """Validate and normalize configuration before changing sidecar state."""
        if control is None:
            return None
        return ControllerControlConfig.from_payload(control).to_payload()

    @staticmethod
    def _command_sequence(value: object) -> int:
        """Validate one model-bound window sequence before an exchange."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("control command_sequence must be a non-negative integer")
        return value

    def _response_sequence(self, value: object, *, expected: int, context: str) -> None:
        """Validate the transaction sequence echoed by the sidecar."""
        if value != expected or isinstance(value, bool):
            error = RuntimeError(f"{context} returned an unexpected command_sequence")
            raise self._poison(error) from error

    @staticmethod
    def _requested_position_gain(value: object) -> float:
        """Validate the gain before beginning a state-changing exchange."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("position_gain must be a finite positive number")
        gain = float(value)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("position_gain must be a finite positive number")
        return gain

    def _position_gain(
        self,
        response: dict[str, Any],
        *,
        expected: float,
        context: str,
    ) -> None:
        """Fail closed unless the sidecar echoes the exact installed gain."""
        value = response.get("position_gain")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            error = RuntimeError(f"{context} returned an invalid position_gain")
            raise self._poison(error) from error
        actual = float(value)
        if not math.isfinite(actual) or actual <= 0.0:
            error = RuntimeError(f"{context} returned an invalid position_gain")
            raise self._poison(error) from error
        if actual != expected:
            error = RuntimeError(
                f"{context} returned position_gain {actual!r}; expected {expected!r}"
            )
            raise self._poison(error) from error

    def _request(
        self,
        request_type: str,
        response_type: str,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_ready()
        request_id = self._request_id
        self._request_id += 1
        started_at = time.monotonic()
        try:
            response = self._round_trip(
                {"type": request_type, "request_id": request_id, **payload}
            )
        except TimeoutError as error:
            elapsed = time.monotonic() - started_at
            contextual = TimeoutError(
                f"{request_type} timed out after {elapsed:.3f}s: {error}"
            )
            raise self._poison(contextual) from error
        except (ConnectionError, OSError, RuntimeError) as error:
            elapsed = time.monotonic() - started_at
            contextual = RuntimeError(
                f"{request_type} transport failed after {elapsed:.3f}s: {error}"
            )
            raise self._poison(contextual) from error
        if not isinstance(response, dict):
            error = RuntimeError(
                "Isaac Teleop calibration returned a non-object response"
            )
            raise self._poison(error) from error
        if response.get("request_id") != request_id:
            error = RuntimeError(
                "Isaac Teleop calibration returned an out-of-order response"
            )
            raise self._poison(error) from error
        if response.get("type") == "error":
            raise RuntimeError(
                "Isaac Teleop calibration failed: "
                f"{response.get('error', 'unknown error')}"
            )
        if response.get("type") != response_type:
            error = RuntimeError(
                "Isaac Teleop calibration returned unexpected response "
                f"{response.get('type')!r}"
            )
            raise self._poison(error) from error
        return response

    def configure(
        self,
        calibration: dict[str, Any] | None,
        *,
        control: Mapping[str, Any] | None = None,
    ) -> LiveCalibrationState | None:
        """Install or reset calibration and optional sidecar control shaping."""
        if calibration is None and control is not None:
            raise ValueError("control configuration requires participant calibration")
        control_payload = self._control_payload(control)
        with self._lock:
            if calibration is None:
                response = self._request(
                    "calibration_reset",
                    "calibration_reset_done",
                )
                epoch = self._epoch(response.get("epoch"), context="calibration reset")
                self._control_space(
                    response,
                    expected=None,
                    context="calibration reset",
                )
                if epoch <= self._last_epoch:
                    error = RuntimeError(
                        "Isaac Teleop calibration reset did not advance its epoch"
                    )
                    raise self._poison(error) from error
                self._last_epoch = epoch
                self._state = None
                return None

            expected_position_gain = self._requested_position_gain(
                calibration.get("position_gain")
            )
            response = self._request(
                "calibration_configure",
                "calibration_configured",
                calibration=calibration,
                control=control_payload,
            )
            self._pose_space(response, context="calibration configure")
            expected_control_space = (
                HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
                if control_payload is not None
                else None
            )
            self._control_space(
                response,
                expected=expected_control_space,
                context="calibration configure",
            )
            if response.get("control") != control_payload:
                error = RuntimeError(
                    "calibration configure returned a different control configuration"
                )
                raise self._poison(error) from error
            self._position_gain(
                response,
                expected=expected_position_gain,
                context="calibration configure",
            )
            epoch = self._epoch(response.get("epoch"), context="calibration configure")
            if epoch <= self._last_epoch:
                error = RuntimeError(
                    "Isaac Teleop calibration configure did not advance its epoch"
                )
                raise self._poison(error) from error
            self._last_epoch = epoch
            state = LiveCalibrationState(
                pose_space=HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
                epoch=epoch,
                control_space=expected_control_space,
            )
            self._state = state
            return state

    def prepare_window(
        self,
        frames: list[dict[str, Any]],
        *,
        command_sequence: int,
    ) -> PreparedControlWindow:
        """Prepare one model-bound control window without advancing state."""
        sequence = self._command_sequence(command_sequence)
        if not isinstance(frames, list) or not frames:
            raise ValueError("control window must be a nonempty list")
        if any(not isinstance(frame, dict) for frame in frames):
            raise ValueError("each control frame must be an object")
        with self._lock:
            self._require_ready()
            state = self._state
            if state is None:
                raise RuntimeError("Isaac Teleop participant calibration is not active")
            if state.control_space != HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION:
                raise RuntimeError("Isaac Teleop control shaping is not active")
            response = self._request(
                "control_prepare_window",
                "control_window_prepared",
                expected_epoch=state.epoch,
                command_sequence=sequence,
                frames=frames,
            )
            self._pose_space(response, context="control prepare")
            self._control_space(
                response,
                expected=HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
                context="control prepare",
            )
            epoch = self._epoch(response.get("epoch"), context="control prepare")
            if epoch != state.epoch:
                error = RuntimeError(
                    "Isaac Teleop control prepare used an unexpected epoch"
                )
                raise self._poison(error) from error
            self._response_sequence(
                response.get("command_sequence"),
                expected=sequence,
                context="control prepare",
            )
            token = response.get("token")
            if not isinstance(token, str) or not token:
                error = RuntimeError(
                    "Isaac Teleop control prepare returned an invalid token"
                )
                raise self._poison(error) from error
            result = response.get("frames")
            if not isinstance(result, list) or len(result) != len(frames):
                error = RuntimeError(
                    "Isaac Teleop control prepare returned a malformed window"
                )
                raise self._poison(error) from error

            validated: list[dict[str, Any]] = []
            for source, transformed in zip(frames, result, strict=True):
                if not isinstance(transformed, dict):
                    error = RuntimeError(
                        "Isaac Teleop control prepare returned a non-object frame"
                    )
                    raise self._poison(error) from error
                transformed_frame = cast(dict[str, Any], transformed)
                try:
                    canonical = quest_controller.validate_controller_frame(
                        transformed_frame
                    )
                except (TypeError, ValueError) as error:
                    invalid = RuntimeError(
                        "Isaac Teleop control prepare returned an invalid frame"
                    )
                    raise self._poison(invalid) from error
                if transformed_frame.get("client_capture_ns") != source.get(
                    "client_capture_ns"
                ):
                    error = RuntimeError(
                        "Isaac Teleop control prepare changed a frame timestamp"
                    )
                    raise self._poison(error) from error
                if transformed_frame.get(CONTROLLER_CALIBRATION_EPOCH_KEY) != epoch:
                    error = RuntimeError(
                        "Isaac Teleop control prepare omitted the window epoch"
                    )
                    raise self._poison(error) from error
                if any(
                    not isinstance(canonical.get(side), dict)
                    or "gripper_open" not in cast(dict[str, Any], canonical[side])
                    for side in ("left", "right")
                ):
                    error = RuntimeError(
                        "Isaac Teleop control prepare omitted gripper_open"
                    )
                    raise self._poison(error) from error
                validated.append(deepcopy(transformed_frame))
            return PreparedControlWindow(
                frames=validated,
                epoch=epoch,
                command_sequence=sequence,
                token=token,
            )

    def commit_window(self, prepared: PreparedControlWindow) -> None:
        """Commit one accepted prepared window; exact retries are idempotent."""
        self._finish_window(prepared, commit=True)

    def abort_window(self, prepared: PreparedControlWindow) -> None:
        """Abort one rejected prepared window; exact retries are idempotent."""
        self._finish_window(prepared, commit=False)

    def _finish_window(self, prepared: PreparedControlWindow, *, commit: bool) -> None:
        """Validate and finish one opaque sidecar control transaction."""
        if not isinstance(prepared, PreparedControlWindow):
            raise TypeError("prepared control window has an incompatible type")
        with self._lock:
            self._require_ready()
            state = self._state
            if state is None or state.epoch != prepared.epoch:
                raise RuntimeError("prepared control window belongs to a stale epoch")
            response = self._request(
                f"control_{'commit' if commit else 'abort'}_window",
                f"control_window_{'committed' if commit else 'aborted'}",
                expected_epoch=prepared.epoch,
                command_sequence=prepared.command_sequence,
                token=prepared.token,
            )
            context = "control commit" if commit else "control abort"
            self._control_space(
                response,
                expected=HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
                context=context,
            )
            epoch = self._epoch(response.get("epoch"), context=context)
            if epoch != prepared.epoch:
                error = RuntimeError(f"{context} used an unexpected epoch")
                raise self._poison(error) from error
            self._response_sequence(
                response.get("command_sequence"),
                expected=prepared.command_sequence,
                context=context,
            )
            if response.get("token") != prepared.token:
                error = RuntimeError(f"{context} returned an unexpected token")
                raise self._poison(error) from error
