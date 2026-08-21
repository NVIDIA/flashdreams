# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step loop connecting one session to one client window."""

import logging
import queue
import sys
import threading
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
"""Logger for results a run could not present."""


class WhenFull(Enum):
    """What to do with a finished result when no room is left to hold it."""

    BLOCK = "block"
    """Hold generation back until the window catches up, presenting every result."""

    DROP_OLDEST = "drop_oldest"
    """Discard the oldest waiting result, skipping frames to keep latency down."""


def _contains(events: UserInputEvents, event_type: type[UserInputEventData]) -> bool:
    """Return whether any event in ``events`` carries ``event_type`` data."""
    return any(
        isinstance(event.get_event_data(), event_type) for event in events.get_events()
    )


def _close_session(session: ISession, *, run_failed: bool) -> None:
    """Close a session, keeping its close from hiding an earlier failure.

    Args:
        session: Session to close.
        run_failed: Whether something has already failed the run. When it has,
            a failing close is logged rather than raised over the top of it.

    Raises:
        Whatever the session raises, when nothing has failed yet.
    """
    try:
        session.close()
    except Exception:
        if not run_failed:
            raise
        _LOGGER.exception(
            "The session failed to close after the run had already failed."
        )


def run_session(
    session: ISession,
    window: IClientWindow,
    *,
    steps: int | None = None,
    max_pending: int = 2,
    when_full: WhenFull = WhenFull.BLOCK,
) -> None:
    """Drive one session against one client window.

    Runs on two threads. The calling thread initializes the session and calls
    ``step`` for each index, with the input collected since the previous step. A
    second thread owns the window: it opens it, ticks at
    ``frames_per_second_for_ui`` to read input, call ``step_ui`` and write
    whatever generation has finished, then closes it. A slow step therefore does
    not hold up input or output. Only the I/O thread touches the window, which is
    what a native window needs, and the window and session are always closed,
    including on failure.

    The window ends the run by reporting a :class:`CloseUserInputEventData`, and
    restarts it by reporting a :class:`ResetUserInputEventData`, which resets the
    session and takes the step index back to zero. The window stays open. Nothing
    from the abandoned generation is presented: each result carries the generation
    it was produced for, so results already waiting and a step that was still
    running when the reset arrived are both dropped rather than written.

    Input is not split at a reset: the batch carrying it reaches the first step
    afterwards whole, earlier events included. Events are edges, so a key held
    down when the client restarts is still held after, and dropping the edge that
    said so would lose that. A session that must not inherit what the abandoned
    generation was given has to ignore events older than its reset itself.

    Writing happens on the I/O thread, so a window slower than generation leaves
    results waiting. ``max_pending`` bounds how many wait, and ``when_full`` says
    what to do about the next one.

    A run whose output is a file works the same way, driven against
    :class:`~flashdreams.runtime_v2.mp4_client_window.Mp4ClientWindow`. That
    window reports no input, so it never reports a close, and such a run ends on
    ``steps`` or on the session saying it has finished.

    A session says so through :meth:`ISession.is_finished`, asked before every
    step, which is how a model that knows its own length ends its own run. The
    run ends at whichever comes first: that, ``steps``, or a close.

    A window that fails to close fails the run, because for a file that means the
    encode did not finish. A close that fails after something else already has is
    logged instead, though, since the earlier failure is what explains the run.

    Args:
        session: Uninitialized session to drive.
        window: Client window supplying input events and presenting results.
        steps: Most steps to run, counted across resets so a reset cannot extend
            the run. ``None`` runs until the session finishes or the window
            reports a close, which is what an interactive window does.
        max_pending: How many finished results may wait to be written.
        when_full: What to do with a result when ``max_pending`` are already
            waiting.

    Raises:
        ValueError: ``steps`` is negative, or ``max_pending`` is not positive.

    Note:
        ``step`` and ``step_ui`` run at the same time, so a session implementing
        both must guard what they share.
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be >= 0 or None, got {steps}.")
    if max_pending <= 0:
        raise ValueError(f"max_pending must be > 0, got {max_pending}.")

    # SessionDesc guarantees this is positive.
    tick_seconds = 1.0 / session.session_desc.frames_per_second_for_ui

    try:
        session.init()
    except Exception:
        # A partly initialized session still holds whatever it managed to load.
        _close_session(session, run_failed=True)
        raise

    # Backpressure is all here. Finished results wait here for the I/O thread to
    # write, and once max_pending of them are waiting, generation blocks in
    # add_pending_result, or drops the oldest result when asked to instead.
    # Each result carries the generation it was produced for, so a reset can tell
    # what belongs to the run the client abandoned from what belongs to the new one.
    pending_results: queue.Queue[tuple[int, StepResult]] = queue.Queue(
        maxsize=max_pending
    )
    generation = 0
    # Only the I/O thread may read the window, so input waits here for the next step.
    collected_events: list[UserInputEvent] = []
    collected_events_lock = threading.Lock()
    opened = threading.Event()
    stop = threading.Event()
    io_failure: list[Exception] = []
    # What never reached the window, reported once the run is over.
    dropped_for_space = 0
    discarded_at_reset = 0

    def present_pending_results() -> None:
        """Write every waiting result to the window, oldest first.

        Because each tick writes all of them, results only pile up when writing
        itself is slower than generation, not merely because the UI rate is lower.
        Results the client reset away from are dropped here rather than written,
        which is also what frees the room they were holding.
        """
        nonlocal discarded_at_reset
        while True:
            try:
                result_generation, result = pending_results.get_nowait()
            except queue.Empty:
                return
            if result_generation != generation:
                discarded_at_reset += 1
                continue
            window.write(result)

    def tick() -> None:
        nonlocal generation
        events = window.get_user_input_events()
        with collected_events_lock:
            collected_events.extend(events.get_events())
            # Move on to the next generation from here, since this thread sees the
            # reset first. Under the lock, so a step already picking up its input
            # either belongs to the generation being abandoned or to the new one,
            # never to neither.
            if _contains(events, ResetUserInputEventData):
                generation += 1
        # Stop from here rather than waiting for the step loop to notice, so a
        # slow step does not delay a client that has gone away.
        if _contains(events, CloseUserInputEventData):
            stop.set()
        session.step_ui(events)
        present_pending_results()

    def run_io() -> None:
        try:
            window.open(session.session_desc)
            # Collect once before generation starts, so the first step sees input
            # the window already has.
            tick()
            opened.set()
            # Stop as soon as the wait says to. Ticking once more would poll input
            # the stopped run can no longer act on, and a reset in that poll would
            # throw away the results it just finished.
            while not stop.wait(tick_seconds):
                tick()
            # Present anything the final step produced after the last tick.
            present_pending_results()
        except Exception as error:
            io_failure.append(error)
        finally:
            opened.set()
            try:
                window.close()
            except Exception as error:
                # Closing is where a sink finishes the writes it was holding, so
                # swallowing this would report a run as complete when the output
                # never landed. An open that raised part way through gets closed
                # here too, since it still holds whatever it had acquired.
                io_failure.append(error)

    def take_collected_events() -> tuple[UserInputEvents, int]:
        """Take the input waiting for the next step, and the generation it is for."""
        with collected_events_lock:
            events = UserInputEvents(list(collected_events))
            collected_events.clear()
            return events, generation

    def add_pending_result(result_generation: int, result: StepResult) -> int:
        """Hand one result to the I/O thread, applying ``when_full``.

        This is where backpressure reaches generation, since no room for a result
        is the only thing that ever slows this thread down.

        Args:
            result_generation: Generation this result was produced for.
            result: Finished result to hand over.

        Returns:
            How many waiting results were dropped to make room.
        """
        pending = (result_generation, result)
        if when_full is WhenFull.DROP_OLDEST:
            # Keep the newest and lose the stale, so a client behind a slow window
            # sees the present rather than catching up through the backlog.
            dropped = 0
            while True:
                try:
                    pending_results.put_nowait(pending)
                    return dropped
                except queue.Full:
                    try:
                        pending_results.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        continue
        # Otherwise let the window set the pace: generation waits here until there
        # is room, which is what an output that must keep every frame needs. Wait
        # interruptibly, though, because once the I/O thread is gone nothing writes
        # the waiting results and a plain put would never return.
        while not (stop.is_set() or io_failure):
            try:
                pending_results.put(pending, timeout=tick_seconds)
                break
            except queue.Full:
                continue
        return 0

    io_thread = threading.Thread(target=run_io, name="flashdreams-io")
    io_thread.start()
    try:
        opened.wait()
        step_index = 0
        # A caller passing steps gets exactly that many, which is what a test or a
        # fixed-length output needs. Bounding on step_index would break that, since
        # a reset takes it back to zero: each reset would extend the run, and
        # enough of them would stop it ever ending.
        steps_run = 0
        while steps is None or steps_run < steps:
            if io_failure or stop.is_set():
                break
            events, step_generation = take_collected_events()
            if _contains(events, ResetUserInputEventData):
                session.reset()
                step_index = 0
            # After the reset, so a session starting over is asked about the new
            # run rather than the one it just finished.
            if session.is_finished():
                break
            dropped_for_space += add_pending_result(
                step_generation, session.step(step_index, events)
            )
            step_index += 1
            steps_run += 1
    finally:
        stop.set()
        io_thread.join()
        # A failure here is what the run reports, since a window failure stops
        # generation rather than raising through it: both places this thread can
        # be sitting give up once io_failure is set, so a run that reports a
        # window failure got there without failing itself. The two are only ever
        # both set by failing independently, and then this is the one raised.
        run_failed = sys.exc_info()[0] is not None
        if io_failure and run_failed:
            _LOGGER.error(
                "The window failed as well as the run, and this is that failure.",
                exc_info=io_failure[0],
            )
        _close_session(session, run_failed=run_failed or bool(io_failure))

    # A log line is the only report of these: a caller cannot count them.
    if dropped_for_space:
        _LOGGER.warning(
            "Dropped %d results the window could not keep up with.", dropped_for_space
        )
    if discarded_at_reset:
        _LOGGER.info(
            "Discarded %d results generated before a reset.", discarded_at_reset
        )

    if io_failure:
        raise io_failure[0]
