# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map model-neutral Action2V snapshots to Waypoint controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

from action2v import ActionSnapshot

from waypoint import WaypointControl

_NAMED_KEY_CODES = {
    "ARROWDOWN": 0x28,
    "ARROWLEFT": 0x25,
    "ARROWRIGHT": 0x27,
    "ARROWUP": 0x26,
    "CONTROL": 0x11,
    "CTRL": 0x11,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "SPACE": 0x20,
    "SPACEBAR": 0x20,
    "TAB": 0x09,
}

# Browser/SlangPy order is left, middle, right. Waypoint follows the Windows
# virtual-key mouse IDs used by the official Biome client.
_MOUSE_BUTTON_CODES = {0: 0x01, 1: 0x04, 2: 0x02}


@dataclass(frozen=True, slots=True)
class WaypointActionMapper:
    """Map shared action snapshots to Waypoint's checkpoint vocabulary."""

    video_width: int
    """Width used to convert normalized horizontal motion to pixels."""

    video_height: int
    """Height used to convert normalized vertical motion to pixels."""

    mouse_sensitivity: float = 1.0
    """Finite non-negative multiplier applied after pixel conversion."""

    def __post_init__(self) -> None:
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("video dimensions must be positive")
        if not math.isfinite(self.mouse_sensitivity) or self.mouse_sensitivity < 0:
            raise ValueError("mouse_sensitivity must be finite and non-negative")

    def __call__(self, snapshot: ActionSnapshot) -> WaypointControl:
        """Convert one shared snapshot to a checkpoint-compatible action."""
        buttons = {
            button for key in snapshot.keys if (button := _key_code(key)) is not None
        }
        buttons.update(
            button
            for index in snapshot.mouse_buttons
            if (button := _MOUSE_BUTTON_CODES.get(index)) is not None
        )
        scroll_wheel = 1 if snapshot.wheel_y > 0 else -1 if snapshot.wheel_y < 0 else 0
        return WaypointControl(
            buttons=frozenset(buttons),
            mouse_dx=(snapshot.mouse_dx * self.video_width * self.mouse_sensitivity),
            mouse_dy=(snapshot.mouse_dy * self.video_height * self.mouse_sensitivity),
            scroll_wheel=scroll_wheel,
        )


def _key_code(key: str) -> int | None:
    if key == " ":
        return 0x20
    if len(key) == 1:
        upper = key.upper()
        if "A" <= upper <= "Z" or "0" <= upper <= "9":
            return ord(upper)
        return None
    normalized = key.replace("_", "").replace("-", "").replace(" ", "").upper()
    return _NAMED_KEY_CODES.get(normalized)


__all__ = ["WaypointActionMapper"]
