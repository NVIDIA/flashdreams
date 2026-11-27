# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

"""The Web cockpit's driving controls for SDL game_controller_node Joy messages."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import copysign


@dataclass(frozen=True)
class CockpitInput:
    """One controller sample. Steer is left-positive in vehicle coordinates."""

    steer: float
    throttle: float
    brake: float
    handbrake: bool
    gear: str
    reset: bool
    camera_delta: int


def _pedal(trigger: float) -> float:
    # SDL triggers rest at zero. Humble joy 3.3.0 reports a squeeze as negative.
    return max(0.0, (abs(trigger) - 0.02) / 0.98)


class CockpitJoy:
    """Track held controls and button edges as in CockpitApp.pollGamepad."""

    def __init__(self) -> None:
        self.gear = "D"
        self._held: set[int] = set()

    def release(self) -> None:
        """Release all buttons while retaining the selected gear."""
        self._held.clear()

    def update(self, axes: Sequence[float], buttons: Sequence[int]) -> CockpitInput:
        """Consume each /joy sample, including releases between model chunks."""
        if len(axes) < 6 or len(buttons) < 15:
            raise ValueError(
                "Expected game_controller_node layout (6 axes, 15+ buttons)"
            )
        if not all(-1.0 <= axis <= 1.0 for axis in axes):
            raise ValueError("Joy axes must be finite and within [-1, 1]")
        if not all(button in (0, 1) for button in buttons):
            raise ValueError("Joy buttons must be 0 or 1")

        held = {i for i, value in enumerate(buttons) if value}
        pressed = held - self._held
        self._held = held
        gears = ("P", "R", "N", "D")
        index = gears.index(self.gear)
        if 1 in pressed:  # B shifts up before X shifts down.
            index = min(3, index + 1)
        if 2 in pressed:
            index = max(0, index - 1)
        self.gear = gears[index]
        for button, gear in enumerate(("D", "R", "P", "N"), start=11):
            if button in held:
                self.gear = gear

        # CockpitApp.updateControls ignores shaped steering at or below 0.02.
        norm = max(0.0, (abs(axes[0]) - 0.025) / 0.975)
        shaped = 0.35 * norm + 0.65 * norm**2.2
        return CockpitInput(
            steer=copysign(shaped, axes[0]) if shaped > 0.02 else 0.0,
            throttle=0.0 if self.gear in ("P", "N") else _pedal(axes[5]),
            brake=1.0 if self.gear == "P" or 0 in held else _pedal(axes[4]),
            handbrake=0 in held,
            gear=self.gear,
            reset=3 in pressed,  # Y
            camera_delta=int(10 in pressed) - int(9 in pressed),  # RB / LB
        )
