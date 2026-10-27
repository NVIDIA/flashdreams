# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-clock gameplay, independent of model generation and presentation."""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from queue import Empty, SimpleQueue
from typing import Any, cast

from omnidreams_game_engine.engine import GameEngine
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.timeline import WorldSample, WorldTimeline

from crazy_robotaxi.controls import (
    BoundActionState,
    ControlsConfig,
    taxi_driver_command,
)
from crazy_robotaxi.debug_input import HeadlessInput
from flashdreams.runtime_v2.user_input_event import (
    UserInputEvent,
    FocusUserInputEvent,
    KeyboardUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class RealtimeGameplay:
    """Own physics/rules on one worker and publish immutable world observations."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], GameEngine],
        driver: DriverInput,
        controls: ControlsConfig,
        shutdown: threading.Event,
        debug: HeadlessInput | None = None,
        clock: Callable[[], float] = time.monotonic,
        capacity: int = 3600,
        rollout_epoch: int = 0,
    ) -> None:
        self.timeline = WorldTimeline(capacity=capacity)
        self._engine_factory = engine_factory
        self._driver = driver
        self._actions = BoundActionState(controls)
        self._shutdown = shutdown
        self._debug = debug
        self._clock = clock
        self._stop = threading.Event()
        self._inputs: SimpleQueue[tuple[UserInputEvents, float]] = SimpleQueue()
        self._requests: SimpleQueue[tuple[str, str]] = SimpleQueue()
        self._rollout_epoch = rollout_epoch
        self._thread = threading.Thread(
            target=self._run, name="robotaxi-gameplay", daemon=True
        )
        self._started_s = 0.0
        if self._debug is not None:
            self._debug.attach_gameplay(self.submit_input)
        self._thread.start()

    def submit_input(self, events: UserInputEvents) -> None:
        """Enqueue from the UI thread without waiting for physics or generation."""
        if events.get_events():
            self._inputs.put((events, self._clock()))

    def request(self, kind: str, value: str) -> None:
        """Queue a UI action or leaderboard text on the simulation owner."""
        self._requests.put((kind, value))

    def close(self) -> None:
        """Stop and join before destroying physics or starting a new epoch."""
        if self._debug is not None:
            self._debug.attach_gameplay(None)
        self._stop.set()
        self.timeline.close()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Gameplay worker did not stop")

    def _run(self) -> None:
        engine = None
        try:
            engine = self._engine_factory()
            live_edit = getattr(engine, "live_edit", None)
            if live_edit is not None and live_edit.obstacles is not None:
                live_edit.obstacles.advance_chunk_fraction = (
                    engine.physics_frame_interval_s / (8 / 30)
                )
            if self._debug is not None:
                self._debug.start()
            self._started_s = self._clock()
            pending: list[tuple[float, int, UserInputEvent]] = []
            sequence = 0
            input_origin: float | None = None
            previous = None
            tick = 0
            interval = engine.physics_frame_interval_s
            # ponytail: Python/GPU bindings can hold the GIL and delay this
            # worker. Fixed ticks catch up; hard deadlines require a native
            # runner or separate process, not a larger physics timestep.
            while not self._stop.is_set() and not self._shutdown.is_set():
                deadline = self._started_s + tick * interval
                if self._stop.wait(max(0, deadline - self._clock())):
                    break
                while True:
                    try:
                        events, received_s = self._inputs.get_nowait()
                    except Empty:
                        break
                    batch = [
                        event
                        for event in events.get_events()
                        if isinstance(
                            event,
                            (
                                FocusUserInputEvent,
                                KeyboardUserInputEvent,
                                GamepadUserInputEvent,
                                GameWheelUserInputEvent,
                            ),
                        )
                    ]
                    if not batch:
                        continue
                    if input_origin is None and (
                        self._debug is not None
                        or any(int(e.get_timestamp()) > 0 for e in batch)
                    ):
                        input_origin = (
                            self._debug.ready_ns / 1e9
                            if self._debug is not None
                            and self._debug.ready_ns is not None
                            else received_s
                            - max(int(e.get_timestamp()) for e in batch) / 1e6
                        )
                    for event in batch:
                        heapq.heappush(
                            pending,
                            (
                                received_s
                                if input_origin is None
                                else input_origin + int(event.get_timestamp()) / 1e6,
                                sequence,
                                event,
                            ),
                        )
                        sequence += 1
                due = []
                # The seed records time zero. Tick N applies input held at the
                # beginning of [N-1, N], never an input from the future endpoint.
                input_deadline = self._started_s + max(0, tick - 1) * interval
                while pending and pending[0][0] <= input_deadline + 1e-9:
                    due.append(heapq.heappop(pending)[2])
                events = UserInputEvents(due)
                transitions = self._driver.apply(events, record_transitions=False)
                actions = self._actions.apply(events)
                if self._debug is not None:
                    self._debug.consumed(events, gameplay_tick=tick)
                if live_edit is not None:
                    for action in ("style", "weather", "coins", "obstacle"):
                        if action in actions:
                            live_edit.request_action(action)
                while True:
                    try:
                        kind, value = self._requests.get_nowait()
                    except Empty:
                        break
                    if kind == "text":
                        engine.submit_text(value)
                        if previous is not None:
                            previous = replace(
                                previous,
                                step=replace(
                                    previous.step,
                                    game_frames=(engine.current_game_frame,),
                                ),
                            )
                    elif live_edit is not None:
                        live_edit.request_action(value)
                started = time.perf_counter()
                cpu_started = time.thread_time()
                if engine.is_running or previous is None:
                    command = taxi_driver_command(self._driver.command())
                    step = engine.advance((command,))
                    presentation = (
                        None
                        if live_edit is None
                        else live_edit.capture_frame(step.trajectory)
                    )
                    previous = WorldSample(step, presentation)
                else:
                    # Keep accepting text and actions on the authoritative owner
                    # without filling history while a leaderboard is displayed.
                    self.timeline.latest = previous
                    tick += 1
                    continue
                sample = replace(
                    previous,
                    transition_timestamp_us=None
                    if not transitions
                    else round(transitions[-1] * 1e6),
                )
                self.timeline.append(sample)
                if not engine.is_running:
                    self.timeline.finish()
                if self._debug is not None:
                    self._debug.messages.put(
                        (
                            None,
                            {
                                "status": "gameplay_tick",
                                "tick": tick,
                                "elapsed_us": self._debug.elapsed_us(),
                                "simulation_timestamp_us": int(
                                    step.trajectory.timestamps_us[0]
                                ),
                                "command": asdict(step.trajectory.applied_commands[0]),
                                "vehicle": asdict(step.trajectory.vehicle_states[0]),
                                "game": asdict(cast(Any, step.game_frames[0])),
                                "rollout_epoch": self._rollout_epoch,
                                "physics_debug": tuple(
                                    asdict(f)
                                    for f in step.trajectory.physics_debug_frames
                                ),
                                "transition_timestamp_us": sample.transition_timestamp_us,
                                "cpu_ms": (time.thread_time() - cpu_started) * 1000,
                                "wall_ms": (time.perf_counter() - started) * 1000,
                                "clock_lag_ms": (self._clock() - deadline) * 1000,
                                "native_thread_id": threading.get_native_id(),
                            },
                        )
                    )
                tick += 1
        except BaseException as error:
            self.timeline.close(error)
        finally:
            if engine is not None:
                engine.close()
