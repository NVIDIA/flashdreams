# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Crazy Robotaxi keyboard input and runtime telemetry."""

from __future__ import annotations

import time

from omnidreams_game_engine.input.keyboard import KeyboardState
from omnidreams_game_engine.types import DriverCommand, VehicleState

from crazy_robotaxi.game import TaxiGameSnapshot
from crazy_robotaxi.high_scores import (
    validate_player_name,
)
from crazy_robotaxi.live_edit.input_hooks import LiveEditRequests
from flashdreams.serving.realtime.input import normalize_key


def _move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


class CrazyRobotaxiKeyboardState(KeyboardState):
    """Keyboard state with progressive arcade steering and game telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self._game_state: TaxiGameSnapshot | None = None
        self._name_submission: str | None = None
        self._keyboard_steer = 0.0
        self._last_command_s = time.monotonic()
        # One-shot live-edit key requests (skin cycle / coins toggle) raised
        # by the presenters and drained by ``CrazyRobotaxiRuntime`` per tick.
        self.live_edit = LiveEditRequests()

    def submit_taxi_name(self, name: str) -> bool:
        """Validate and queue one high-score name submission."""
        try:
            normalized = validate_player_name(name)
        except ValueError:
            return False
        with self._lock:
            self._name_submission = normalized
        return True

    def consume_taxi_name_submission(self) -> str | None:
        """Return and clear the pending high-score name submission."""
        with self._lock:
            name = self._name_submission
            self._name_submission = None
            return name

    def update_runtime_state(
        self, state: VehicleState, game_state: TaxiGameSnapshot
    ) -> None:
        """Publish vehicle and game state as one coherent snapshot."""
        with self._lock:
            self._vehicle_state = state
            self._game_state = game_state

    def clear_telemetry(self) -> None:
        """Clear vehicle, game, and pending name state."""
        with self._lock:
            self._vehicle_state = None
            self._game_state = None
            self._name_submission = None

    @property
    def taxi_game_state(self) -> TaxiGameSnapshot | None:
        """Return the latest game snapshot."""
        with self._lock:
            return self._game_state

    @property
    def runtime_state(
        self,
    ) -> tuple[VehicleState | None, TaxiGameSnapshot | None]:
        """Return vehicle and game state from one publication lock."""
        with self._lock:
            return self._vehicle_state, self._game_state

    def command(self) -> DriverCommand:
        """Return a presenter-independent Crazy Robotaxi drive command."""
        now = time.monotonic()
        with self._lock:
            dt_s = max(0.0, min(0.1, now - self._last_command_s))
            self._last_command_s = now
            drive_command = next(
                (
                    self._drive_commands[source]
                    for source in ("keyboard", "browser", "default", "wheel")
                    if source in self._drive_commands
                ),
                None,
            )
            pressed = {normalize_key(key) for key in self._keyboard.snapshot()}
            game_state = self._game_state
        if game_state is not None and game_state.session_state != "playing":
            return DriverCommand()
        if drive_command is not None:
            if "space" not in pressed:
                return drive_command
            return DriverCommand(
                throttle=0.0,
                brake=drive_command.brake,
                steer=drive_command.steer,
                handbrake=True,
                reverse=drive_command.reverse,
                steer_is_direct=drive_command.steer_is_direct,
                manual_control=drive_command.manual_control,
            )

        target_steer = float(bool({"a", "left"} & pressed)) - float(
            bool({"d", "right"} & pressed)
        )
        steer_rate = 3.5 if abs(target_steer) > 0.0 else 5.0
        self._keyboard_steer = _move_towards(
            self._keyboard_steer, target_steer, steer_rate * dt_s
        )
        return DriverCommand(
            throttle=1.0 if {"w", "up"} & pressed else 0.0,
            brake=1.0 if {"s", "down"} & pressed else 0.0,
            steer=self._keyboard_steer,
            handbrake="space" in pressed,
        )
