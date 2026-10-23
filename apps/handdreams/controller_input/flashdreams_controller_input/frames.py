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

"""Controller transport parsing, pose validation, and physical controls."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

SIDES = ("left", "right")

ControllerPayload = dict[str, Any]
"""One validated controller: pose, analog values, and named button state."""

ControllerFrame = dict[str, Any]
"""One observation with controller payloads and host/client capture clocks."""

SOURCE_CLIENT_CAPTURE_NS_KEY = "source_client_capture_ns"
"""Original producer monotonic capture timestamp normalized to nanoseconds."""


def as_mapping(value: object) -> dict[str, Any] | None:
    """Return ``value`` as a string-keyed mapping, or ``None`` if it is not one.

    Controller streams and bridge RPCs carry untrusted JSON, so every field
    access goes through this first.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _thumbstick_axes(value: dict[str, Any]) -> np.ndarray | None:
    """Read the thumbstick pair, or ``None`` when the source omits it.

    The ``xr-standard`` mapping puts the thumbstick at ``axes[2:4]``; two-axis
    controllers report it directly. Absence stays absent so bridge round trips
    preserve whether the source supplied axes.

    An empty array counts as absent. WebXR reports ``gamepad.axes: []`` for a
    controller it has not finished enumerating, which is a normal transient
    rather than a malformed frame.
    """
    if "axes" not in value:
        return None
    axes = np.asarray(value["axes"], dtype=np.float32).reshape(-1)
    if axes.size == 0:
        return None
    if axes.size == 2:
        stick = axes
    elif axes.size >= 4:
        stick = axes[2:4]
    else:
        raise ValueError(
            f"Quest controller axes must hold 2 or 4+ values, got {axes.size}"
        )
    if not np.isfinite(stick).all():
        raise ValueError("Quest controller axes must be finite")
    return np.clip(stick, -1.0, 1.0)


def controller_payload(raw: object) -> ControllerPayload | None:
    """Validate one controller into physical controls, or return ``None``."""
    value = as_mapping(raw)
    if value is None:
        return None
    position = np.asarray(value.get("position"), dtype=np.float32).reshape(-1)
    orientation = np.asarray(value.get("orientation"), dtype=np.float32).reshape(-1)
    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError("Quest controller requires position[3] and orientation[4]")
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        raise ValueError("Quest controller pose must be finite")
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-8:
        raise ValueError("Quest controller quaternion has zero norm")
    payload: dict[str, Any] = {
        "position": position.tolist(),
        "orientation": (orientation / norm).tolist(),
        "trigger": float(np.clip(float(value.get("trigger", 0.0)), 0.0, 1.0)),
        "grip": float(np.clip(float(value.get("grip", 0.0)), 0.0, 1.0)),
        "tracked": bool(value.get("tracked", True)),
        "primary_button": bool(value.get("primary_button", False)),
        "secondary_button": bool(value.get("secondary_button", False)),
    }
    axes = _thumbstick_axes(value)
    if axes is not None:
        payload["axes"] = axes.tolist()
    if "battery" in value and value["battery"] is not None:
        battery = float(value["battery"])
        if not np.isfinite(battery) or not 0.0 <= battery <= 1.0:
            raise ValueError("Quest controller battery must be between 0 and 1")
        payload["battery"] = battery
    return payload


def parse_xr_frame(
    raw: object, *, received_ns: int | None = None
) -> ControllerFrame | None:
    """Validate one relay ``xr_frame`` into physical controller state.

    Args:
        raw: Decoded JSON message from the relay.
        received_ns: Host receive time; defaults to the current perf counter.

    Returns:
        The validated frame, or ``None`` if the message is not an ``xr_frame``.

    Raises:
        ValueError: If the message claims to be an ``xr_frame`` but its
            controller payloads are malformed.
    """
    payload = as_mapping(raw)
    if payload is None or payload.get("type") != "xr_frame":
        return None
    controllers = as_mapping(payload.get("controllers"))
    if controllers is None:
        raise ValueError("xr_frame controllers must be an object")
    frame: ControllerFrame = {
        side: controller_payload(controllers.get(side)) for side in SIDES
    }
    is_emulated = payload.get("is_emulated", False)
    if type(is_emulated) is not bool:
        raise ValueError("xr_frame is_emulated must be a boolean")
    connection_id = payload.get("client_connection_id")
    if connection_id is not None and (
        not isinstance(connection_id, str) or not 1 <= len(connection_id) <= 512
    ):
        raise ValueError("xr_frame client_connection_id must be a nonempty string")
    frame["is_emulated"] = is_emulated
    frame["client_connection_id"] = connection_id
    frame["client_capture_ns"] = int(received_ns or time.perf_counter_ns())
    client_timestamp = payload.get("t_client")
    if client_timestamp is not None:
        if isinstance(client_timestamp, (bool, np.bool_)) or not isinstance(
            client_timestamp, (int, float, np.integer, np.floating)
        ):
            raise ValueError("xr_frame t_client must fit a uint64 nanosecond timestamp")
        try:
            capture_seconds = float(client_timestamp)
        except OverflowError as error:
            raise ValueError("xr_frame t_client exceeds uint64 nanoseconds") from error
        if not 0.0 <= capture_seconds <= (2**64 - 1) / 1_000_000_000:
            raise ValueError("xr_frame t_client must fit a uint64 nanosecond timestamp")
        capture_ns = int(round(capture_seconds * 1_000_000_000.0))
        if capture_ns > 2**64 - 1:
            raise ValueError("xr_frame t_client must fit a uint64 nanosecond timestamp")
        frame[SOURCE_CLIENT_CAPTURE_NS_KEY] = capture_ns
    return frame


def controller_sides(frame: ControllerFrame | None) -> dict[str, ControllerPayload]:
    """Return the present controller payloads, keyed by side.

    Callers use this instead of subscripting a possibly-``None`` frame, which
    keeps the "both present" check and the access in one place.
    """
    if not frame:
        return {}
    present = {
        side: payload for side in SIDES if isinstance(payload := frame.get(side), dict)
    }
    return present if len(present) == len(SIDES) else {}


def both_controllers_tracked(frame: ControllerFrame | None) -> bool:
    """Report whether both controllers are present and currently tracked."""
    sides = controller_sides(frame)
    return bool(sides) and all(
        bool(payload.get("tracked", True)) for payload in sides.values()
    )
