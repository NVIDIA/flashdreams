# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

from interactive_drive.types import DriverCommand


@dataclass
class KeyboardDriveController:
    _pressed: set[str] = field(default_factory=set)

    def on_key(self, key_name: str, is_down: bool) -> None:
        if is_down:
            self._pressed.add(key_name)
        else:
            self._pressed.discard(key_name)

    def sample(self) -> DriverCommand:
        throttle = 1.0 if {"w", "up"} & self._pressed else 0.0
        brake = 1.0 if {"s", "down", "space"} & self._pressed else 0.0
        steer_left = 1.0 if {"a", "left"} & self._pressed else 0.0
        steer_right = 1.0 if {"d", "right"} & self._pressed else 0.0
        steer = steer_left - steer_right
        return DriverCommand(throttle=throttle, brake=brake, steer=steer)
