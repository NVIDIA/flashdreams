# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical wheel-state converter for interactive driving."""

from __future__ import annotations

from collections.abc import Mapping

from flashdreams.runtime import (
    DRIVER_COMMAND,
    DeviceConverterSchema,
    TimeWindow,
    UserInputCapability,
    UserInputs,
)


class WheelToDriverCommand:
    """Convert absolute wheel/pedal state into normalized driving intent."""

    schema = DeviceConverterSchema(
        name="wheel-to-driver-command",
        produces=DRIVER_COMMAND,
        device_kind="wheel",
        priority=100,
        consumes=(
            UserInputCapability(
                event_type="wheel_state",
                payload_fields=frozenset(
                    {
                        "steer",
                        "throttle",
                        "brake",
                        "reverse",
                    }
                ),
            ),
        ),
    )

    def __init__(self) -> None:
        self._state: Mapping[str, object] | None = None

    def reset(self) -> None:
        self._state = None

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, object] | None:
        del window
        for event in user_inputs.events:
            if event.event_type == "wheel_state":
                self._state = event.payload
        if self._state is None:
            return None
        return DRIVER_COMMAND.value(
            {
                "steer": _number(self._state["steer"], name="steer"),
                "throttle": _number(self._state["throttle"], name="throttle"),
                "brake": _number(self._state["brake"], name="brake"),
                "reverse": bool(self._state["reverse"]),
                "stop": bool(self._state.get("stop", False)),
                "steer_is_direct": True,
                "manual_control": True,
            }
        )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"wheel_state.{name} must be numeric.")
    return float(value)


__all__ = ["WheelToDriverCommand"]
