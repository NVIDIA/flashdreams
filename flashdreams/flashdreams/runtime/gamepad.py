# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gamepad input validation and canonical driving conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime.canonical import (
    DRIVER_COMMAND,
    DeviceConverterSchema,
    KeyboardToDriverCommand,
)
from flashdreams.runtime.inputs import UserInputCapability, UserInputs

GAMEPAD_STATE_EVENT = "gamepad_state"
"""Event type shared by local and browser gamepad sources."""

_GAMEPAD_DEADZONE = 0.05
"""Deadzone shared by generation activation and canonical conversion."""

GAMEPAD_STATE_CAPABILITY = UserInputCapability(
    event_type=GAMEPAD_STATE_EVENT,
    input_modality="gamepad",
    payload_fields=frozenset({"connected", "steer", "throttle", "brake"}),
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

    def is_active(self) -> bool:
        """Return whether input exceeds the gamepad deadzone."""
        return self.connected and (
            self.throttle > _GAMEPAD_DEADZONE
            or self.brake > _GAMEPAD_DEADZONE
            or abs(self.steer) > _GAMEPAD_DEADZONE
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
    if not isinstance(connected, bool):
        raise TypeError("Gamepad field 'connected' must be boolean.")
    steer = _number(payload["steer"], name="steer", minimum=-1.0)
    throttle = _number(payload["throttle"], name="throttle", minimum=0.0)
    brake = _number(payload["brake"], name="brake", minimum=0.0)
    if not connected:
        return GamepadState(
            connected=False,
            steer=0.0,
            throttle=0.0,
            brake=0.0,
        )
    return GamepadState(
        connected=True,
        steer=steer,
        throttle=throttle,
        brake=brake,
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
    }


class DrivingInputConverter:
    """Merge keyboard and gamepad timelines into one driving command."""

    def __init__(self) -> None:
        """Configure live driving input conversion."""
        self._gamepad_state = GamepadState(False, 0.0, 0.0, 0.0)
        self._keyboard = KeyboardToDriverCommand()
        self._schema = DeviceConverterSchema(
            name="live-driving-input",
            produces=DRIVER_COMMAND,
            consumes=(
                UserInputCapability(
                    event_type="key_down",
                    payload_fields=frozenset({"key"}),
                ),
                UserInputCapability(
                    event_type="key_up",
                    payload_fields=frozenset({"key"}),
                ),
                GAMEPAD_STATE_CAPABILITY,
            ),
            device_kind="driving-input",
            accepted_keys=self._keyboard.schema.accepted_keys,
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        """Return the raw-input and canonical-output contract."""
        return self._schema

    def reset(self) -> None:
        """Clear held keyboard and gamepad state."""
        self._gamepad_state = GamepadState(False, 0.0, 0.0, 0.0)
        self._keyboard.reset()

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any]:
        """Merge keyboard and gamepad state at every timeline boundary.

        Args:
            user_inputs: Raw keyboard edges and gamepad snapshots.
            window: Half-open interval represented by the command.

        Returns:
            Canonical driving state with a complete merged segment timeline.
        """
        keyboard = self._keyboard.convert(UserInputs(), window)
        if keyboard is None:
            raise RuntimeError("Keyboard conversion did not produce a command.")

        segment_start = window.start_s
        level = self._gamepad_level() if self._gamepad_state.is_active() else keyboard
        segments: list[tuple[float, float, Mapping[str, Any]]] = []
        for event in user_inputs.events:
            if event.event_type not in {
                "key_down",
                "key_up",
                GAMEPAD_STATE_EVENT,
            }:
                continue
            event_time = min(
                max(float(event.timestamp_s), window.start_s),
                window.end_s,
            )
            if event_time > segment_start:
                segments.append((segment_start, event_time, level))
            if event.event_type == GAMEPAD_STATE_EVENT:
                self._gamepad_state = parse_gamepad_state(event.payload)
            else:
                keyboard = self._keyboard.convert(
                    UserInputs(events=(event,)),
                    window,
                )
                if keyboard is None:
                    raise RuntimeError("Keyboard conversion did not produce a command.")
            level = (
                self._gamepad_level() if self._gamepad_state.is_active() else keyboard
            )
            segment_start = event_time
        if window.end_s > segment_start or not segments:
            segments.append((segment_start, window.end_s, level))

        merged: list[tuple[float, float, Mapping[str, Any]]] = []
        for start, end, segment_level in segments:
            if merged and merged[-1][2] == segment_level:
                previous_start, _previous_end, previous_level = merged[-1]
                merged[-1] = (previous_start, end, previous_level)
            else:
                merged.append((start, end, segment_level))
        final_level = merged[-1][2]
        return DRIVER_COMMAND.value({**final_level, "segments": tuple(merged)})

    def _gamepad_level(self) -> Mapping[str, Any]:
        """Return the current post-deadzone gamepad level."""
        state = self._gamepad_state
        return {
            "throttle": (
                state.throttle
                if state.connected and state.throttle > _GAMEPAD_DEADZONE
                else 0.0
            ),
            "brake": (
                state.brake
                if state.connected and state.brake > _GAMEPAD_DEADZONE
                else 0.0
            ),
            "steer": (
                state.steer
                if state.connected and abs(state.steer) > _GAMEPAD_DEADZONE
                else 0.0
            ),
            "stop": False,
            "reverse": False,
        }


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
    "DrivingInputConverter",
    "GamepadState",
    "gamepad_state_payload",
    "parse_gamepad_state",
]
