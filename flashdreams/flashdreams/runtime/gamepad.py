# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gamepad input validation and canonical driving conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime.canonical import DRIVER_COMMAND, DeviceConverterSchema
from flashdreams.runtime.inputs import UserInputCapability, UserInputs

GAMEPAD_STATE_EVENT = "gamepad_state"
"""Event type shared by local and browser gamepad sources."""

GAMEPAD_STATE_CAPABILITY = UserInputCapability(
    event_type=GAMEPAD_STATE_EVENT,
    input_modality="gamepad",
    payload_fields=frozenset(
        {"connected", "steer", "throttle", "brake", "reverse", "stop"}
    ),
    description="Normalized gamepad driving state.",
)


@dataclass(frozen=True, slots=True)
class GamepadState:
    """Normalized driving state reported by one gamepad source."""

    connected: bool
    """Whether the source still reports an attached gamepad."""

    steer: float
    """Steering in ``[-1, 1]`` with positive values turning left."""

    throttle: float
    """Throttle engagement in ``[0, 1]``."""

    brake: float
    """Brake engagement in ``[0, 1]``."""

    reverse: bool
    """Whether reverse driving is selected."""

    stop: bool
    """Whether immediate stopping is requested."""

    @property
    def active(self) -> bool:
        """Return whether the gamepad currently contributes driving input."""
        return self.connected and (
            self.throttle > 0.0
            or self.brake > 0.0
            or abs(self.steer) > 0.0
            or self.reverse
            or self.stop
        )


def parse_gamepad_state(payload: Mapping[str, Any]) -> GamepadState:
    """Validate and normalize one gamepad payload.

    Args:
        payload: JSON-like gamepad state.

    Returns:
        Validated gamepad state.

    Raises:
        TypeError: A required field has the wrong type.
        ValueError: A numeric field is non-finite or outside its range.
    """
    connected = payload["connected"]
    reverse = payload["reverse"]
    stop = payload["stop"]
    if not isinstance(connected, bool):
        raise TypeError("Gamepad field 'connected' must be boolean.")
    if not isinstance(reverse, bool):
        raise TypeError("Gamepad field 'reverse' must be boolean.")
    if not isinstance(stop, bool):
        raise TypeError("Gamepad field 'stop' must be boolean.")
    steer = _number(payload["steer"], name="steer", minimum=-1.0)
    throttle = _number(payload["throttle"], name="throttle", minimum=0.0)
    brake = _number(payload["brake"], name="brake", minimum=0.0)
    if not connected:
        return GamepadState(
            connected=False,
            steer=0.0,
            throttle=0.0,
            brake=0.0,
            reverse=False,
            stop=False,
        )
    return GamepadState(
        connected=True,
        steer=steer,
        throttle=throttle,
        brake=brake,
        reverse=reverse,
        stop=stop,
    )


def gamepad_state_payload(state: GamepadState) -> dict[str, bool | float]:
    """Return a JSON-compatible payload for ``state``.

    Args:
        state: Validated gamepad state.

    Returns:
        Complete payload accepted by :func:`parse_gamepad_state`.
    """
    return {
        "connected": state.connected,
        "steer": state.steer,
        "throttle": state.throttle,
        "brake": state.brake,
        "reverse": state.reverse,
        "stop": state.stop,
    }


class GamepadToDriverCommand:
    """Convert gamepad state snapshots into canonical driving commands."""

    def __init__(self, *, deadzone: float = 0.05, priority: int = 10) -> None:
        """Configure gamepad conversion.

        Args:
            deadzone: Inclusive magnitude treated as neutral.
            priority: Selection priority over converters producing
                :data:`DRIVER_COMMAND`.

        Raises:
            ValueError: ``deadzone`` is outside ``[0, 1)``.
        """
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1).")
        self._deadzone = deadzone
        self._state = GamepadState(False, 0.0, 0.0, 0.0, False, False)
        self._schema = DeviceConverterSchema(
            name="gamepad-to-driver-command",
            produces=DRIVER_COMMAND,
            consumes=(GAMEPAD_STATE_CAPABILITY,),
            device_kind="gamepad",
            priority=priority,
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        """Return the raw-input and canonical-output contract."""
        return self._schema

    def reset(self) -> None:
        """Clear held gamepad state."""
        self._state = GamepadState(False, 0.0, 0.0, 0.0, False, False)

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        """Convert gamepad snapshots inside ``window``.

        Args:
            user_inputs: Raw inputs containing complete gamepad snapshots.
            window: Half-open interval represented by the resulting command.

        Returns:
            Canonical driving state with piecewise segments, or ``None`` when
            the gamepad is neutral or disconnected.
        """
        segment_start = window.start_s
        command = self._level()
        segments: list[tuple[float, float, Mapping[str, Any]]] = []
        for event in user_inputs.events:
            if event.event_type != GAMEPAD_STATE_EVENT:
                continue
            event_time = min(
                max(float(event.timestamp_s), window.start_s),
                window.end_s,
            )
            if event_time > segment_start:
                segments.append((segment_start, event_time, command))
            self._state = parse_gamepad_state(event.payload)
            command = self._level()
            segment_start = event_time
        if not self._active():
            return None
        if window.end_s > segment_start or not segments:
            segments.append((segment_start, window.end_s, command))
        return DRIVER_COMMAND.value({**command, "segments": tuple(segments)})

    def _level(self) -> Mapping[str, Any]:
        """Return the current canonical level."""
        state = self._state
        steer = 0.0 if abs(state.steer) <= self._deadzone else state.steer
        throttle = 0.0 if state.throttle <= self._deadzone else state.throttle
        brake = 0.0 if state.brake <= self._deadzone else state.brake
        return {
            "throttle": throttle if state.connected else 0.0,
            "brake": brake if state.connected else 0.0,
            "steer": steer if state.connected else 0.0,
            "stop": state.stop if state.connected else False,
            "reverse": state.reverse if state.connected else False,
        }

    def _active(self) -> bool:
        """Return whether the current post-deadzone level is active."""
        level = self._level()
        return bool(
            level["throttle"]
            or level["brake"]
            or level["steer"]
            or level["stop"]
            or level["reverse"]
        )


def _number(
    value: object,
    *,
    name: str,
    minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Gamepad field {name!r} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Gamepad field {name!r} must be finite.")
    if not minimum <= parsed <= 1.0:
        raise ValueError(f"Gamepad field {name!r} must be in [{minimum:g}, 1].")
    return parsed


__all__ = [
    "GAMEPAD_STATE_CAPABILITY",
    "GAMEPAD_STATE_EVENT",
    "GamepadState",
    "GamepadToDriverCommand",
    "gamepad_state_payload",
    "parse_gamepad_state",
]
