# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic V2 event reduction for model-thread driving."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from flashdreams.runtime.keyboard import DRIVING_SUPPORTED_KEYS
from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
from flashdreams.runtime_v2.keyboard_input import (
    KeyboardStateSegment,
    KeyboardStateTrack,
)
from flashdreams.runtime_v2.user_input_event import (
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.types import DriverCommand

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
class DriverInputBatch:
    """Per-frame driving commands for one model step."""

    commands: tuple[DriverCommand, ...]

    transition_timestamps_us: tuple[int | None, ...]
    """Input transition represented by each command; ``None`` for retained state."""

    transition_count: int
    """Number of driving-state transitions in the sampled input window."""

    dropped_transition_count: int
    """Number of sampled transitions not represented by an output command."""

    ignored_event_count: int
    """Number of keyboard events unsupported by the shared driving track."""


class DriverInput:
    """Sample shared V2 keyboard state into taxi driving commands."""

    def __init__(
        self,
        config: DriverInputConfig = DriverInputConfig(),
        *,
        samples_per_second: float = 30.0,
    ) -> None:
        self.config = config
        self._timeline = RealtimeInputTimeline(
            samples_per_second=samples_per_second,
        )
        self._keyboard = KeyboardStateTrack(
            supported_keys=DRIVING_SUPPORTED_KEYS,
        )
        self._last_level = _DriveLevel()
        self._steering = 0.0

    def reduce(
        self,
        events: UserInputEvents,
        *,
        frame_count: int,
    ) -> DriverInputBatch:
        """Produce one command for each shared-timeline keyboard sample."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        dispositions = self._keyboard.ingest(events)
        window = self._timeline.next_window(
            frame_count,
            input_times_s=(
                disposition.timestamp_s
                for disposition in dispositions
                if disposition.tracked
            ),
        )
        segments = self._keyboard.segments(window)
        scheduled, transition_count, dropped_transition_count = (
            self._sample_drive_levels(segments, window.sample_times_s)
        )
        ignored_event_count = sum(
            isinstance(disposition.event, KeyboardUserInputEvent)
            and not disposition.tracked
            for disposition in dispositions
        )
        if dropped_transition_count:
            _LOGGER.warning(
                "Dropped %d input transitions between model-frame samples in a "
                "%d-frame chunk",
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
                rate * self._timeline.sample_interval_s,
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
            transition_count=transition_count,
            dropped_transition_count=dropped_transition_count,
            ignored_event_count=ignored_event_count,
        )

    def reset(self) -> None:
        """Clear every retained input value."""
        self._timeline.reset()
        self._keyboard.reset()
        self._last_level = _DriveLevel()
        self._steering = 0.0

    def _sample_drive_levels(
        self,
        segments: list[KeyboardStateSegment],
        sample_times_s: tuple[float, ...],
    ) -> tuple[list[tuple[_DriveLevel, int | None]], int, int]:
        """Project shared held-key segments onto discrete driving frames."""
        resolved_segments: list[tuple[float, float, _DriveLevel, int | None]] = []
        previous_level = self._last_level
        transition_count = 0
        for start_s, end_s, keys in segments:
            level = _drive_level(keys)
            timestamp_us: int | None = None
            if level != previous_level:
                transition_count += 1
                timestamp_us = round(start_s * 1_000_000.0)
            resolved_segments.append((start_s, end_s, level, timestamp_us))
            previous_level = level

        scheduled: list[tuple[_DriveLevel, int | None]] = []
        represented_timestamps: set[int] = set()
        segment_index = 0
        current_timestamp_us: int | None = None
        for sample_s in sample_times_s:
            while (
                segment_index + 1 < len(resolved_segments)
                and sample_s > resolved_segments[segment_index][1]
            ):
                introduced_timestamp_us = resolved_segments[segment_index][3]
                if introduced_timestamp_us is not None:
                    current_timestamp_us = introduced_timestamp_us
                    represented_timestamps.add(introduced_timestamp_us)
                segment_index += 1
            _, _, level, introduced_timestamp_us = resolved_segments[segment_index]
            if introduced_timestamp_us is not None:
                current_timestamp_us = introduced_timestamp_us
                represented_timestamps.add(introduced_timestamp_us)
            scheduled.append((level, current_timestamp_us))

        self._last_level = previous_level
        dropped_transition_count = max(
            0,
            transition_count - len(represented_timestamps),
        )
        return scheduled, transition_count, dropped_transition_count


def _drive_level(keys: frozenset[str]) -> _DriveLevel:
    return _DriveLevel(
        forward=bool({"w", "up"} & keys),
        backward=bool({"s", "down"} & keys),
        left=bool({"a", "left"} & keys),
        right=bool({"d", "right"} & keys),
        handbrake="space" in keys,
    )


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    if current < target:
        return min(current + maximum_delta, target)
    return max(current - maximum_delta, target)
