# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic V2 event reduction for model-thread driving."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.types import DriverCommand

_DRIVE_KEYS = frozenset({"w", "a", "s", "d", "up", "down", "left", "right", "space"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DriveLevel:
    """Resolved digital driving state at one input transition."""

    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False
    handbrake: bool = False


@dataclass(frozen=True, slots=True)
class _TimedDriveLevel:
    """Resolved driving state paired with its V2 event timestamp."""

    timestamp_us: int
    level: _DriveLevel


@dataclass(frozen=True, slots=True)
class DriverInputBatch:
    """Per-frame driving commands for one model step."""

    commands: tuple[DriverCommand, ...]

    transition_timestamps_us: tuple[int | None, ...]
    """Input transition represented by each command; ``None`` for retained state."""

    transition_count: int
    """Number of distinct driving-state transitions received in this batch."""

    dropped_transition_count: int
    """Number of oldest transitions that could not fit in the model chunk."""

    ignored_event_count: int
    """Number of redundant drive events that did not change resolved controls."""


class DriverInput:
    """Retain key levels while consuming V2 edge events exactly once."""

    def __init__(self, config: DriverInputConfig = DriverInputConfig()) -> None:
        self.config = config
        self._pressed: set[str] = set()
        self._steering = 0.0

    def reduce(
        self,
        events: UserInputEvents,
        *,
        frame_count: int,
        frame_interval_s: float,
    ) -> DriverInputBatch:
        """Preserve input transitions and produce one command per simulated frame."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        transitions: list[_TimedDriveLevel] = []
        ignored_event_count = 0
        for event in events.get_events():
            data = event.get_event_data()
            previous = self._drive_level()
            if isinstance(data, FocusUserInputEventData) and not data.focused:
                self._pressed.clear()
            elif isinstance(data, KeyboardUserInputEventData):
                key = _normalize_key(str(data.key))
                if key not in _DRIVE_KEYS:
                    continue
                if data.state is KeyboardInputState.PRESSED:
                    self._pressed.add(key)
                else:
                    self._pressed.discard(key)
            else:
                continue

            current = self._drive_level()
            if current != previous:
                transitions.append(
                    _TimedDriveLevel(
                        timestamp_us=int(event.get_timestamp()),
                        level=current,
                    )
                )
            else:
                ignored_event_count += 1

        scheduled, dropped_transition_count = _schedule_drive_levels(
            transitions,
            retained=self._drive_level(),
            frame_count=frame_count,
            frame_interval_s=frame_interval_s,
        )
        if dropped_transition_count:
            _LOGGER.warning(
                "Dropped %d oldest input transitions that could not fit in a "
                "%d-frame model chunk",
                dropped_transition_count,
                frame_count,
            )

        commands: list[DriverCommand] = []
        transition_timestamps_us: list[int | None] = []
        for level, timestamp_us in scheduled:
            target = self.config.steering_scale * (
                float(level.left) - float(level.right)
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
            commands.append(
                DriverCommand(
                    throttle=1.0 if level.forward and not level.backward else 0.0,
                    brake=1.0 if level.backward or level.handbrake else 0.0,
                    steer=self._steering,
                    handbrake=level.handbrake,
                    steer_is_direct=True,
                    manual_control=True,
                )
            )
            transition_timestamps_us.append(timestamp_us)
        return DriverInputBatch(
            commands=tuple(commands),
            transition_timestamps_us=tuple(transition_timestamps_us),
            transition_count=len(transitions),
            dropped_transition_count=dropped_transition_count,
            ignored_event_count=ignored_event_count,
        )

    def reset(self) -> None:
        """Clear every retained input value."""
        self._pressed.clear()
        self._steering = 0.0

    def _drive_level(self) -> _DriveLevel:
        return _DriveLevel(
            forward=bool({"w", "up"} & self._pressed),
            backward=bool({"s", "down"} & self._pressed),
            left=bool({"a", "left"} & self._pressed),
            right=bool({"d", "right"} & self._pressed),
            handbrake="space" in self._pressed,
        )


def _schedule_drive_levels(
    transitions: list[_TimedDriveLevel],
    *,
    retained: _DriveLevel,
    frame_count: int,
    frame_interval_s: float,
) -> tuple[list[tuple[_DriveLevel, int | None]], int]:
    """Map timestamped transitions onto a fixed-size future model chunk."""
    if not transitions:
        return [(retained, None) for _ in range(frame_count)], 0

    scheduled: list[tuple[_DriveLevel, int, int]] = []
    safe_frame_interval_s = max(float(frame_interval_s), 1e-9)
    for index, transition in enumerate(transitions):
        if index + 1 < len(transitions):
            next_transition = transitions[index + 1]
            duration_s = max(
                0.0,
                (next_transition.timestamp_us - transition.timestamp_us) / 1_000_000.0,
            )
            repeat_count = max(0, round(duration_s / safe_frame_interval_s))
        else:
            repeat_count = max(1, frame_count - len(scheduled))
        scheduled.extend(
            (transition.level, transition.timestamp_us, index)
            for _ in range(repeat_count)
        )

    represented_before_capacity = {index for _, _, index in scheduled}
    if len(scheduled) > frame_count:
        scheduled = scheduled[-frame_count:]
    elif len(scheduled) < frame_count:
        latest = scheduled[-1]
        scheduled.extend(latest for _ in range(frame_count - len(scheduled)))

    retained_transition_indexes = {index for _, _, index in scheduled}
    dropped_transition_count = len(
        represented_before_capacity - retained_transition_indexes
    )
    return (
        [(level, timestamp_us) for level, timestamp_us, _ in scheduled],
        dropped_transition_count,
    )


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
