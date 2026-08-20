# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent loops connecting one session to one client window."""

import logging
import queue
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)


class WhenFull(Enum):
    """What to do with a finished result when the result queue is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"


def _contains(events: UserInputEvents, event_type: type[UserInputEventData]) -> bool:
    """Return whether any event in ``events`` carries ``event_type`` data."""
    return any(
        isinstance(event.get_event_data(), event_type) for event in events.get_events()
    )


def _close_session(session: ISession, *, run_failed: bool) -> None:
    """Close a session without hiding an earlier failure.

    Args:
        session: Session to close.
        run_failed: Whether another failure already explains the run.

    Raises:
        Exception: Whatever :meth:`ISession.close` raises when the run succeeded.
    """
    try:
        session.close()
    except Exception:
        if not run_failed:
            raise
        _LOGGER.exception(
            "The session failed to close after the run had already failed."
        )


@dataclass
class SessionRunner:
    """State shared by the UI and generation loops for one session.

    The UI loop owns the window and runs on the I/O thread. The generation loop
    runs on the caller's thread. The event batch and pending-result queue are the
    two handoffs between them.

    Args:
        session: Uninitialized session to drive.
        window: Client window supplying input events and presenting results.
        steps: Maximum number of steps across resets, or ``None`` to run until
            the session finishes or the window closes.
        max_pending: Maximum number of results waiting to be presented.
        when_full: Behavior when ``max_pending`` results are already waiting.
    """

    session: ISession
    window: IClientWindow
    steps: int | None = None
    max_pending: int = 2
    when_full: WhenFull = WhenFull.BLOCK
    tick_seconds: float = field(init=False)
    pending_results: queue.Queue[tuple[int, StepResult]] = field(init=False)
    collected_events: list[UserInputEvent] = field(default_factory=list)
    input_events_lock: threading.Lock = field(default_factory=threading.Lock)
    ui_startup_complete: threading.Event = field(default_factory=threading.Event)
    shutdown_requested: threading.Event = field(default_factory=threading.Event)
    io_failure: list[Exception] = field(default_factory=list)
    generation: int = 0
    dropped_for_space: int = 0
    discarded_at_reset: int = 0

    def run_session(self) -> None:
        """Initialize and run the session until it reaches an end condition.

        Raises:
            ValueError: ``steps`` is negative, or ``max_pending`` is not positive.
        """
        if self.steps is not None and self.steps < 0:
            raise ValueError(f"steps must be >= 0 or None, got {self.steps}.")
        if self.max_pending <= 0:
            raise ValueError(f"max_pending must be > 0, got {self.max_pending}.")

        try:
            self.session.init()
        except Exception:
            _close_session(self.session, run_failed=True)
            raise

        self.tick_seconds = 1.0 / self.session.session_desc.frames_per_second_for_ui
        self.pending_results = queue.Queue(maxsize=self.max_pending)
        # The runner has two major loops: the UI loop owns the window on its
        # thread, while the calling thread runs the generation loop below.
        io_thread = threading.Thread(target=self._run_ui_loop, name="flashdreams-io")
        io_thread.start()
        try:
            self._run_generation_loop()
        finally:
            self.shutdown_requested.set()
            io_thread.join()
            run_failed = sys.exc_info()[0] is not None
            if self.io_failure and run_failed:
                _LOGGER.error(
                    "The window failed as well as the run, and this is that failure.",
                    exc_info=self.io_failure[0],
                )
            _close_session(
                self.session,
                run_failed=run_failed or bool(self.io_failure),
            )

        if self.dropped_for_space:
            _LOGGER.warning(
                "Dropped %d results the window could not keep up with.",
                self.dropped_for_space,
            )
        if self.discarded_at_reset:
            _LOGGER.info(
                "Discarded %d results generated before a reset.",
                self.discarded_at_reset,
            )
        if self.io_failure:
            raise self.io_failure[0]

    def _run_generation_loop(self) -> None:
        """Continuously generate results until the run reaches an end condition."""
        self.ui_startup_complete.wait()
        step_index = 0
        steps_run = 0
        while self.steps is None or steps_run < self.steps:
            if self.io_failure or self.shutdown_requested.is_set():
                return
            events, result_generation = self._take_events()
            if _contains(events, ResetUserInputEventData):
                self.session.reset()
                step_index = 0
            if self.session.is_finished():
                return
            result = self.session.step(step_index, events)
            self.dropped_for_space += self._queue_result(result_generation, result)
            step_index += 1
            steps_run += 1

    def _run_ui_loop(self) -> None:
        """Continuously collect input and present results on the window thread."""
        try:
            self.window.open(self.session.session_desc)
            # A first tick ensures the first generation step sees queued input.
            self._tick()
            self.ui_startup_complete.set()
            while not self.shutdown_requested.wait(self.tick_seconds):
                self._tick()
            self._present_results()
        except Exception as error:
            self.io_failure.append(error)
        finally:
            self.ui_startup_complete.set()
            try:
                self.window.close()
            except Exception as error:
                self.io_failure.append(error)

    def _tick(self) -> None:
        """Collect one input batch, update the UI, and present ready results."""
        events = self.window.get_user_input_events()
        with self.input_events_lock:
            self.collected_events.extend(events.get_events())
            # The UI loop observes reset first. Locking ties a result to either
            # the generation before the reset or the one after it, never both.
            if _contains(events, ResetUserInputEventData):
                self.generation += 1
        if _contains(events, CloseUserInputEventData):
            self.shutdown_requested.set()
        self.session.step_ui(events)
        self._present_results()

    def _take_events(self) -> tuple[UserInputEvents, int]:
        """Take all events collected since the previous generation step."""
        with self.input_events_lock:
            events = UserInputEvents(list(self.collected_events))
            self.collected_events.clear()
            return events, self.generation

    def _present_results(self) -> None:
        """Write every queued result that belongs to the current generation."""
        while True:
            try:
                result_generation, result = self.pending_results.get_nowait()
            except queue.Empty:
                return
            if result_generation != self.generation:
                self.discarded_at_reset += 1
                continue
            self.window.write(result)

    def _queue_result(self, result_generation: int, result: StepResult) -> int:
        """Queue a result, applying the configured backpressure policy.

        Args:
            result_generation: Generation that produced ``result``.
            result: Finished result to present.

        Returns:
            The number of queued results dropped to make room.
        """
        pending = (result_generation, result)
        if self.when_full is WhenFull.DROP_OLDEST:
            dropped = 0
            while True:
                try:
                    self.pending_results.put_nowait(pending)
                    return dropped
                except queue.Full:
                    try:
                        self.pending_results.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        continue

        while not (self.shutdown_requested.is_set() or self.io_failure):
            try:
                self.pending_results.put(pending, timeout=self.tick_seconds)
                break
            except queue.Full:
                continue
        return 0
