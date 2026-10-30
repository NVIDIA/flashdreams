# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-process gameplay with timestamped inputs and recorded world observations."""

from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty
from typing import Any

from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.timeline import WorldSample, WorldTimeline

from crazy_robotaxi.controls import (
    taxi_driver_command,
    BoundActionState,
    ControlsConfig,
)
from crazy_robotaxi.debug_input import HeadlessInput
from crazy_robotaxi.factory import build_taxi_engine
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
)


def _build_cpu_engine(**options):
    from ludus_renderer import prepare_physx

    prepare_physx()
    return build_taxi_engine(**options)


def _run_gameplay(
    engine_options,
    driver,
    inputs,
    outputs,
    stop,
    log_path,
    engine_factory=_build_cpu_engine,
    controls=ControlsConfig(),
) -> None:
    """Own CPU simulation in a spawned process, without a CUDA renderer."""
    engine = None
    log = None
    try:
        engine = engine_factory(**engine_options, render_conditions=False)
        live_edit = getattr(engine, "live_edit", None)
        actions = BoundActionState(controls)
        if live_edit is not None and live_edit.obstacles is not None:
            live_edit.obstacles.advance_chunk_fraction = engine.frame_interval_s / (
                8 / 30
            )
        if log_path is not None:
            engine.simulation.set_physx_debug_enabled(True)
            log = Path(log_path).open("x", encoding="utf-8")
        outputs.put(("ready",))
        while not stop.is_set():
            try:
                if not inputs.poll(0.1):
                    continue
                message = inputs.recv()
            except Empty:
                continue
            if message[0] == "start":
                _, origin, input_origin = message
                break
        else:
            return
        interval = engine.frame_interval_s
        pending = []
        sequence = 0
        tick = 0
        previous = None
        while not stop.is_set():
            deadline = origin + tick * interval
            if stop.wait(max(0, deadline - time.monotonic())):
                break
            while True:
                try:
                    if not inputs.poll():
                        break
                    message = inputs.recv()
                except Empty:
                    break
                kind = message[0]
                if kind == "input":
                    _, serial, events, received = message
                    if input_origin is None and any(
                        int(e.get_timestamp()) > 0 for e in events.get_events()
                    ):
                        input_origin = (
                            received
                            - max(int(e.get_timestamp()) for e in events.get_events())
                            / 1e6
                        )
                    # Retain timestamp order, including future events in a batch.
                    for event_index, event in enumerate(events.get_events()):
                        heapq.heappush(
                            pending,
                            (
                                received
                                if input_origin is None
                                else input_origin + int(event.get_timestamp()) / 1e6,
                                sequence,
                                event,
                                serial,
                                event_index,
                            ),
                        )
                        sequence += 1
                elif kind == "action" and live_edit is not None:
                    live_edit.request_action(message[1])
                elif kind == "text":
                    engine.submit_text(message[1])
                    if previous is not None:
                        previous = replace(
                            previous,
                            step=replace(
                                previous.step, game_frames=(engine.current_game_frame,)
                            ),
                        )
                        outputs.put_nowait(("latest", previous))
            due = []
            applied_batches: dict[int, list[int]] = {}
            # Input timestamps use the gameplay-ready wall clock. A slow physics
            # tick must not add its accumulated simulation deficit to input lag.
            input_now = time.monotonic()
            while pending and pending[0][0] <= input_now:
                _, _, event, serial, event_index = heapq.heappop(pending)
                due.append(event)
                applied_batches.setdefault(serial, []).append(event_index)
            if live_edit is not None:
                for action in actions.apply(UserInputEvents(due)):
                    if action in {"style", "weather", "coins", "obstacle"}:
                        live_edit.request_action(action)
            transitions = driver.apply(UserInputEvents(due), record_transitions=False)
            consumed_ns = time.monotonic_ns()
            for serial, event_indices in applied_batches.items():
                outputs.put_nowait(
                    ("consumed", serial, event_indices, consumed_ns, tick)
                )
            if engine.is_running or previous is None:
                started = time.monotonic_ns()
                cpu_started = time.thread_time_ns()
                step = engine.advance((taxi_driver_command(driver.command()),))
                completed = time.monotonic_ns()
                sample = WorldSample(
                    step,
                    presentation=None
                    if live_edit is None
                    else live_edit.capture_frame(step.trajectory),
                    transition_timestamp_us=round(transitions[-1] * 1e6)
                    if transitions
                    else None,
                )
                outputs.put_nowait(("sample", sample, not engine.is_running))
                previous = sample
                if log is not None:
                    log.write(
                        json.dumps(
                            {
                                "tick": tick,
                                "started_ns": started,
                                "completed_ns": completed,
                                "scheduled_ns": round(deadline * 1e9),
                                "cpu_ms": (time.thread_time_ns() - cpu_started) / 1e6,
                                "simulation_timestamp_us": int(
                                    step.trajectory.timestamps_us[0]
                                ),
                                "transition_timestamp_us": sample.transition_timestamp_us,
                                "command": asdict(step.trajectory.applied_commands[0]),
                                "vehicle": asdict(step.trajectory.vehicle_states[0]),
                                "game": asdict(step.game_frames[0]),
                            }
                        )
                        + "\n"
                    )
            tick += 1
    except BaseException:
        outputs.put(("error", traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close()
        if log is not None:
            log.close()
        outputs.put(("closed",))


class GameplayProcess:
    """Bridge ordinary UI input and a separately scheduled physics owner."""

    def __init__(
        self,
        engine_options: dict[str, Any],
        driver: DriverInput,
        debug: HeadlessInput | None,
        epoch: int,
        engine_factory=_build_cpu_engine,
        controls=ControlsConfig(),
    ) -> None:
        context = mp.get_context("spawn")
        self.timeline = WorldTimeline()
        self._debug = debug
        child_inputs, self._inputs = context.Pipe(duplex=False)
        self._send_lock = threading.Lock()
        self._outputs = context.Queue(maxsize=1024)
        self._stop = context.Event()
        self._ready = threading.Event()
        self._failure: str | None = None
        self._pending_inputs: dict[int, dict[int, UserInputEvent]] = {}
        self._input_lock = threading.Lock()
        self._serial = 0
        log_path = None
        if debug is not None and debug.capture is not None:
            log_path = str(debug.capture.directory / f"gameplay-{epoch}.jsonl")
        self._process = context.Process(
            target=_run_gameplay,
            args=(
                engine_options,
                driver,
                child_inputs,
                self._outputs,
                self._stop,
                log_path,
                engine_factory,
                controls,
            ),
            name="robotaxi-physics",
            daemon=True,
        )
        self._receiver = threading.Thread(
            target=self._receive, name="robotaxi-world-observations", daemon=True
        )
        self._process.start()
        child_inputs.close()
        self._receiver.start()

    def start(self) -> None:
        """Start wall-clock physics only after process initialization and warmup."""
        if not self._ready.wait(60):
            self.close()
            raise RuntimeError("Physics process did not initialize")
        self.check()
        if self._debug is not None:
            self._debug.start()
        now = time.monotonic()
        input_origin = (
            self._debug.ready_ns / 1e9
            if self._debug is not None and self._debug.ready_ns is not None
            else None
        )
        self._send(("start", now, input_origin))

    def _send(self, message) -> None:
        # Small control packets go directly to the pipe. A multiprocessing
        # Queue feeder would have to reacquire the model process's GIL first.
        with self._send_lock:
            self._inputs.send(message)

    def submit_input(self, events: UserInputEvents) -> None:
        """Forward events dispatched by the ordinary UI reader without waiting."""
        events = UserInputEvents(
            [
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
        )
        if not events.get_events() or self._stop.is_set():
            return
        self.check()
        with self._input_lock:
            serial = self._serial
            self._serial += 1
            self._pending_inputs[serial] = dict(enumerate(events.get_events()))
        self._send(("input", serial, events, time.monotonic()))

    def request_action(self, value: str) -> None:
        """Forward an optional live-edit action to the gameplay owner."""
        self._send(("action", value))

    def submit_text(self, value: str) -> None:
        """Forward leaderboard submission to the authoritative rules owner."""
        self._send(("text", value))

    def check(self) -> None:
        """Propagate a worker failure instead of presenting a frozen game."""
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if self._process.exitcode is not None and not self._stop.is_set():
            raise RuntimeError(f"Physics process exited: {self._process.exitcode}")

    def _receive(self) -> None:
        while True:
            try:
                message = self._outputs.get(timeout=0.1)
            except Empty:
                if self._process.exitcode is not None:
                    self.timeline.close(RuntimeError("Physics process exited"))
                    self._ready.set()
                    return
                continue
            kind = message[0]
            if kind == "closed":
                return
            if kind == "ready":
                self._ready.set()
            elif kind == "error":
                self._failure = message[1]
                self.timeline.close(RuntimeError(self._failure))
                self._ready.set()
            elif kind == "sample":
                try:
                    self.timeline.append(message[1])
                    if message[2]:
                        self.timeline.finish()
                except BaseException:
                    self._failure = traceback.format_exc()
                    self.timeline.close(RuntimeError(self._failure))
                    self._stop.set()
            elif kind == "latest":
                self.timeline.latest = message[1]
            elif kind == "consumed":
                with self._input_lock:
                    pending = self._pending_inputs[message[1]]
                    events = UserInputEvents(
                        [pending.pop(index) for index in message[2]]
                    )
                    if not pending:
                        del self._pending_inputs[message[1]]
                if self._debug is not None:
                    self._debug.consumed(
                        events,
                        step=message[4],
                        consumed_ns=message[3],
                        consumer="gameplay",
                    )

    def close(self) -> None:
        """Join the physics owner and observation reader before resetting the world."""
        self._stop.set()
        self.timeline.close()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            raise RuntimeError("Physics process did not shut down cleanly")
        self._receiver.join(timeout=5)
        self._inputs.close()
        self._outputs.close()
