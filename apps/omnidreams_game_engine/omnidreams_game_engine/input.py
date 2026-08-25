# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic V2 event reduction for model-thread driving."""

from __future__ import annotations

from dataclasses import dataclass

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.types import DriverCommand

_DRIVE_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
_TEXT_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
)


@dataclass(frozen=True, slots=True)
class DriverInputBatch:
    """Per-frame commands and optional submitted text for one model step."""

    commands: tuple[DriverCommand, ...]
    submitted_text: str | None
    text: str


class DriverInput:
    """Retain key levels while consuming V2 edge events exactly once."""

    def __init__(self, config: DriverInputConfig = DriverInputConfig()) -> None:
        self.config = config
        self._pressed: set[str] = set()
        self._steering = 0.0
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def reduce(
        self,
        events: UserInputEvents,
        *,
        frame_count: int,
        frame_interval_s: float,
        accepting_text: bool,
    ) -> DriverInputBatch:
        """Apply input edges and produce a command for every simulated frame."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        submitted: str | None = None
        for event in events.get_events():
            data = event.get_event_data()
            if isinstance(data, FocusUserInputEventData) and not data.focused:
                self._pressed.clear()
                continue
            if not isinstance(data, KeyboardUserInputEventData):
                continue
            raw_key = str(data.key)
            key = _normalize_key(raw_key)
            pressed = data.state is KeyboardInputState.PRESSED
            if accepting_text and pressed:
                consumed, value = self._consume_text_key(raw_key, key)
                if consumed:
                    submitted = value or submitted
                    continue
            if key in _DRIVE_KEYS:
                if pressed:
                    self._pressed.add(key)
                else:
                    self._pressed.discard(key)

        commands: list[DriverCommand] = []
        for _ in range(frame_count):
            target = self.config.steering_scale * (
                float(bool({"a", "left"} & self._pressed))
                - float(bool({"d", "right"} & self._pressed))
            )
            rate = (
                self.config.steering_rate_per_s
                if target
                else self.config.steering_return_rate_per_s
            )
            self._steering = _move_towards(
                self._steering,
                target,
                rate * frame_interval_s,
            )
            forward = bool({"w", "up"} & self._pressed)
            backward = bool({"s", "down"} & self._pressed)
            commands.append(
                DriverCommand(
                    throttle=1.0 if forward and not backward else 0.0,
                    brake=1.0 if backward or "space" in self._pressed else 0.0,
                    steer=self._steering,
                    handbrake="space" in self._pressed,
                    steer_is_direct=True,
                    manual_control=True,
                )
            )
        return DriverInputBatch(tuple(commands), submitted, self._text)

    def reset(self) -> None:
        """Clear every retained input value."""
        self._pressed.clear()
        self._steering = 0.0
        self._text = ""

    def _consume_text_key(self, raw_key: str, key: str) -> tuple[bool, str | None]:
        if key in {"enter", "return"}:
            return True, self._text if self._text.strip() else None
        if key == "backspace":
            self._text = self._text[:-1]
            return True, None
        if key == "escape":
            self._text = "DRIVER"
            return True, self._text
        if len(raw_key) == 1 and raw_key in _TEXT_CHARACTERS and len(self._text) < 12:
            self._text += raw_key
            return True, None
        return False, None


def _normalize_key(key: str) -> str:
    if key == " ":
        return "space"
    normalized = key.strip().lower()
    return {
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "spacebar": "space",
    }.get(normalized, normalized)


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    if current < target:
        return min(current + maximum_delta, target)
    return max(current - maximum_delta, target)
