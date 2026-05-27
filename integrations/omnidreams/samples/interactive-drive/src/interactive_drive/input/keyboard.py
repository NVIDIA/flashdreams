# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time

from interactive_drive.input.backend import InputBackend, SampledInput
from interactive_drive.types import ControlSnapshot, DriverCommand


class KeyboardState:
    """Owns live keyboard state plus the runtime UI affordances the loop reads.

    Implements :class:`~interactive_drive.runtime.runtime_controls.RuntimeControls`:
    ``view_mode`` is a property and reset is a rising-edge boolean consumed by
    the single loop reader. Pressed keys are still snapshotted via
    :meth:`snapshot` because iterating a shared set requires a defensive copy
    under the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._view_mode = "rgb"
        self._drive_command: DriverCommand | None = None
        self._reset_pending = False

    def set_key(self, name: str, down: bool) -> None:
        with self._lock:
            if down:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)

    def set_view_mode(self, mode: str) -> None:
        with self._lock:
            self._view_mode = mode

    def request_reset(self) -> None:
        with self._lock:
            self._reset_pending = True

    def set_drive_command(self, command: DriverCommand | None) -> None:
        with self._lock:
            self._drive_command = command

    def consume_reset_request(self) -> bool:
        with self._lock:
            pending = self._reset_pending
            self._reset_pending = False
            return pending

    @property
    def view_mode(self) -> str:
        with self._lock:
            return self._view_mode

    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            return ControlSnapshot(pressed=set(self._pressed), view_mode=self._view_mode)

    def command(self) -> DriverCommand:
        with self._lock:
            drive_command = self._drive_command
            pressed = set(self._pressed)
        if drive_command is not None:
            if "space" in pressed:
                return DriverCommand(
                    throttle=0.0,
                    brake=1.0,
                    steer=drive_command.steer,
                    stop=True,
                    reverse=drive_command.reverse,
                    steer_is_direct=drive_command.steer_is_direct,
                    manual_control=drive_command.manual_control,
                )
            return drive_command
        return command_from_snapshot(ControlSnapshot(pressed=pressed))


def command_from_snapshot(snapshot: ControlSnapshot) -> DriverCommand:
    throttle = 1.0 if {"w", "up"} & snapshot.pressed else 0.0
    brake = 1.0 if {"s", "down"} & snapshot.pressed else 0.0
    steer = 0.0
    if {"a", "left"} & snapshot.pressed:
        steer += 1.0
    if {"d", "right"} & snapshot.pressed:
        steer -= 1.0
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        stop="space" in snapshot.pressed,
    )


class KeyboardInputBackend(InputBackend):
    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def sample(self) -> SampledInput:
        sample_time = time.perf_counter()
        return SampledInput(command=self._keyboard.command(), sample_time=sample_time)
