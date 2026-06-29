# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time

from omnidreams.interactive_drive.input.backend import InputBackend, SampledInput
from omnidreams.interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    VehicleState,
)


class KeyboardState:
    """Owns live keyboard state plus the runtime UI affordances the loop reads.

    Implements :class:`~omnidreams.interactive_drive.runtime.runtime_controls.RuntimeControls`
    (``view_mode`` property, rising-edge reset consumed by the single loop
    reader). Also carries a one-slot telemetry channel
    (:meth:`update_telemetry` / :attr:`vehicle_state`): the loop pushes the
    latest :class:`VehicleState` each chunk so read-side observers (the
    presenter's ``/state`` endpoint) can publish speed/steer/position without
    referencing the per-scene simulation object.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._view_mode = "rgb"
        self._drive_command: DriverCommand | None = None
        self._reset_pending = False
        # Rising-edge "exit the current scene and return to the scene
        # selector" request, set by a wheel/controller's bound exit button
        # (the HUD's ``x`` key calls the presenter directly). The presenter
        # drains it on the main thread each ``process_events`` and converts
        # it into its own exit-to-selection signal, mirroring how
        # ``_reset_pending`` is drained by the runtime loop.
        self._exit_scene_pending = False
        # Output telemetry slot. ``None`` means the simulation hasn't
        # produced a chunk yet (warmup window) -- callers render an empty
        # speed readout in that case.
        self._vehicle_state: VehicleState | None = None

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

    def request_exit_scene(self) -> None:
        """Request a return to the scene selector from a bound device button."""
        with self._lock:
            self._exit_scene_pending = True

    def consume_exit_scene_request(self) -> bool:
        with self._lock:
            pending = self._exit_scene_pending
            self._exit_scene_pending = False
            return pending

    def set_drive_command(self, command: DriverCommand | None) -> None:
        with self._lock:
            self._drive_command = command

    def update_telemetry(self, state: VehicleState) -> None:
        """Publish the simulation's latest vehicle state for read-only consumers.

        Called once per chunk by :func:`run_main_loop` after the simulation
        advances. The MJPEG presenter's ``/state`` endpoint reads this on
        the HTTP handler thread, so the assignment runs under the same
        lock as the input mutators.
        """
        with self._lock:
            self._vehicle_state = state

    def clear_telemetry(self) -> None:
        """Drop the published vehicle state (back to the pre-first-chunk state).

        Called when a rollout is torn down (scene switch / exit to selector)
        so read-side speed readouts fall back to their empty state instead of
        lingering on the just-ended rollout's last reading.
        """
        with self._lock:
            self._vehicle_state = None

    @property
    def vehicle_state(self) -> VehicleState | None:
        """Most-recent simulation snapshot, or ``None`` before the first chunk."""
        with self._lock:
            return self._vehicle_state

    def consume_reset_request(self) -> bool:
        with self._lock:
            pending = self._reset_pending
            self._reset_pending = False
            return pending

    @property
    def view_mode(self) -> str:
        with self._lock:
            return self._view_mode

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
