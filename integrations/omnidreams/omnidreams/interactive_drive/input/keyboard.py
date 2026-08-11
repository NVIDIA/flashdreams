# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time

from omnidreams.interactive_drive.high_scores import validate_player_name
from omnidreams.interactive_drive.input.backend import InputBackend, SampledInput
from omnidreams.interactive_drive.taxi_game import TaxiGameSnapshot
from omnidreams.interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    VehicleState,
)

from flashdreams.serving.realtime.input import (
    DRIVING_SUPPORTED_KEYS,
    normalize_key,
)
from flashdreams.serving.realtime.input import (
    KeyboardState as RealtimeKeyboardState,
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
        self._keyboard = RealtimeKeyboardState(supported_keys=DRIVING_SUPPORTED_KEYS)
        self._view_mode = "rgb"
        self._drive_commands: dict[str, DriverCommand] = {}
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
        self._taxi_game_state: TaxiGameSnapshot | None = None
        self._taxi_name_submission: str | None = None
        self._taxi_controls_enabled = False

    def enable_taxi_controls(self) -> None:
        """Enable Taxi-only handbrake and brake-to-reverse input mapping."""
        with self._lock:
            self._taxi_controls_enabled = True

    def set_key(self, name: str, down: bool) -> None:
        with self._lock:
            self._keyboard.apply_event(
                event="keydown" if down else "keyup",
                key=name,
            )

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

    def submit_taxi_name(self, name: str) -> bool:
        """Validate and queue one high-score name submission.

        Args:
            name: Candidate player name from a presenter.

        Returns:
            ``True`` when the name was valid and queued.
        """
        try:
            normalized = validate_player_name(name)
        except ValueError:
            return False
        with self._lock:
            self._taxi_name_submission = normalized
        return True

    def consume_taxi_name_submission(self) -> str | None:
        """Return and clear the pending high-score name submission."""
        with self._lock:
            name = self._taxi_name_submission
            self._taxi_name_submission = None
            return name

    def consume_exit_scene_request(self) -> bool:
        with self._lock:
            pending = self._exit_scene_pending
            self._exit_scene_pending = False
            return pending

    def set_drive_command(
        self,
        command: DriverCommand | None,
        *,
        source: str | None = None,
    ) -> None:
        with self._lock:
            if source is None:
                if command is None:
                    self._drive_commands.clear()
                    return
                source = "default"
            if command is None:
                self._drive_commands.pop(source, None)
            else:
                self._drive_commands[source] = command

    def update_telemetry(self, state: VehicleState) -> None:
        """Publish the simulation's latest vehicle state for read-only consumers.

        Called once per chunk by :func:`run_main_loop` after the simulation
        advances. The MJPEG presenter's ``/state`` endpoint reads this on
        the HTTP handler thread, so the assignment runs under the same
        lock as the input mutators.
        """
        with self._lock:
            self._vehicle_state = state

    def update_runtime_state(
        self, state: VehicleState, taxi_game_state: TaxiGameSnapshot | None
    ) -> None:
        """Publish vehicle and taxi state as one coherent runtime snapshot."""
        with self._lock:
            self._vehicle_state = state
            self._taxi_game_state = taxi_game_state

    def clear_telemetry(self) -> None:
        """Drop the published vehicle state (back to the pre-first-chunk state).

        Called when a rollout is torn down (scene switch / exit to selector)
        so read-side speed readouts fall back to their empty state instead of
        lingering on the just-ended rollout's last reading.
        """
        with self._lock:
            self._vehicle_state = None
            self._taxi_game_state = None
            self._taxi_name_submission = None

    @property
    def vehicle_state(self) -> VehicleState | None:
        """Most-recent simulation snapshot, or ``None`` before the first chunk."""
        with self._lock:
            return self._vehicle_state

    @property
    def taxi_game_state(self) -> TaxiGameSnapshot | None:
        """Most-recent taxi-game snapshot, or ``None`` when the mode is inactive."""
        with self._lock:
            return self._taxi_game_state

    @property
    def runtime_state(
        self,
    ) -> tuple[VehicleState | None, TaxiGameSnapshot | None]:
        """Return vehicle and taxi state from the same publication lock."""
        with self._lock:
            return self._vehicle_state, self._taxi_game_state

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
            drive_command = next(
                (
                    self._drive_commands[source]
                    for source in ("keyboard", "browser", "default", "wheel")
                    if source in self._drive_commands
                ),
                None,
            )
            pressed = set(self._keyboard.snapshot())
            taxi_game_state = self._taxi_game_state
            taxi_controls_enabled = self._taxi_controls_enabled
        if taxi_game_state is not None and taxi_game_state.session_state != "playing":
            return DriverCommand()
        if drive_command is not None:
            if "space" in pressed:
                if taxi_controls_enabled:
                    return DriverCommand(
                        throttle=0.0,
                        brake=drive_command.brake,
                        steer=drive_command.steer,
                        handbrake=True,
                        reverse=drive_command.reverse,
                        steer_is_direct=drive_command.steer_is_direct,
                        manual_control=drive_command.manual_control,
                    )
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
        if taxi_controls_enabled:
            return taxi_command_from_snapshot(ControlSnapshot(pressed=pressed))
        return command_from_snapshot(ControlSnapshot(pressed=pressed))


def command_from_snapshot(snapshot: ControlSnapshot) -> DriverCommand:
    pressed = {normalize_key(key) for key in snapshot.pressed}
    forward = bool({"w", "up"} & pressed)
    reverse = bool({"s", "down"} & pressed)
    opposing_directions = forward and reverse
    throttle = 1.0 if forward != reverse else 0.0
    brake = 1.0 if opposing_directions else 0.0
    steer = 0.0
    if {"a", "left"} & pressed:
        steer += 1.0
    if {"d", "right"} & pressed:
        steer -= 1.0
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        stop="space" in pressed,
        reverse=reverse and not forward,
    )


def taxi_command_from_snapshot(snapshot: ControlSnapshot) -> DriverCommand:
    """Map raw keys to Taxi-only brake-to-reverse and handbrake controls."""
    pressed = {normalize_key(key) for key in snapshot.pressed}
    steer = float(bool({"a", "left"} & pressed)) - float(bool({"d", "right"} & pressed))
    return DriverCommand(
        throttle=1.0 if {"w", "up"} & pressed else 0.0,
        brake=1.0 if {"s", "down"} & pressed else 0.0,
        steer=steer,
        handbrake="space" in pressed,
    )


class KeyboardInputBackend(InputBackend):
    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def sample(self) -> SampledInput:
        sample_time = time.perf_counter()
        return SampledInput(command=self._keyboard.command(), sample_time=sample_time)
