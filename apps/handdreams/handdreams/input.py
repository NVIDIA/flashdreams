# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HandDreams controller payloads and overlapping Isaac Teleop pose windows."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from flashdreams_controller_input.events import ControllerFrameEvent

SIDES = ("left", "right")
ControllerFrame = dict[str, Any]


def controller_frame(event: ControllerFrameEvent) -> ControllerFrame:
    """Strip unbound face buttons before controller calibration."""
    frame = event.to_controller_frame()
    for side in SIDES:
        hand = frame[side]
        hand.pop("primary_button")
        hand.pop("secondary_button")
    return frame


@dataclass(slots=True)
class OverlapWindowBuffer:
    """Retain the endpoint of each ``N + 1`` pose window as the next seed."""

    _frames: deque[ControllerFrame] = field(default_factory=deque)

    def append(self, frame: Mapping[str, Any]) -> None:
        self._frames.append(dict(frame))

    def ready(self, action_rows: int) -> bool:
        return len(self._frames) >= action_rows + 1

    def peek(self, action_rows: int) -> list[ControllerFrame]:
        needed = action_rows + 1
        if len(self._frames) < needed:
            raise RuntimeError(
                f"{action_rows} actions require {needed} controller poses, "
                f"got {len(self._frames)}"
            )
        return list(self._frames)[:needed]

    def consume(self, action_rows: int) -> None:
        if len(self._frames) < action_rows + 1:
            raise RuntimeError("cannot consume an incomplete controller window")
        for _ in range(action_rows):
            self._frames.popleft()

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


def controller_status(
    frame: Mapping[str, Any] | None, *, threshold: float
) -> tuple[bool, bool, bool, bool]:
    """Return left/right tracking and open-claw flags for the UI."""
    if frame is None:
        return False, False, False, False
    values: list[bool] = []
    for side in SIDES:
        payload = frame.get(side)
        tracked = isinstance(payload, Mapping) and bool(payload.get("tracked", True))
        opened = (
            tracked
            and float(cast(Mapping[str, Any], payload).get("trigger", 1.0)) <= threshold
        )
        values.extend((tracked, opened))
    left_tracked, left_open, right_tracked, right_open = values
    return left_tracked, right_tracked, left_open, right_open
