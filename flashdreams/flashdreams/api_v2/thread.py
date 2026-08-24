# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model and UI threads for a session."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, final

from torch import Tensor

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEventData
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from flashdreams.runtime_v2.presentation_manager import PresentationManager

StateT = TypeVar("StateT")


@dataclass(slots=True)
class _Message(Generic[StateT]):
    operation: Callable[[StateT], None]
    """Operation to run before the thread's next step."""


class IThread(ABC, Generic[StateT]):
    """Base class for model and UI work."""

    def __init__(self, *, state: StateT, frequency: int) -> None:
        """Store the thread's state and rate.

        Args:
            state: State used by this thread.
            frequency: Maximum steps per second; zero disables pacing.

        Raises:
            TypeError: ``frequency`` is not an integer.
            ValueError: ``frequency`` is negative.
        """
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError("frequency must be an integer.")
        if frequency < 0:
            raise ValueError("frequency must be >= 0.")
        self.state = state
        self.frequency = frequency
        self._message_queue: queue.Queue[_Message[StateT]] = queue.Queue()
        self.user_events = UserInputEvents([])
        self.latest_result: StepResult | list[StepResult] | None = None
        self._step_index = 0
        self._generation = 0
        self._accepting_messages = True
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    @abstractmethod
    def step(
        self, step_index: int, events: UserInputEvents
    ) -> StepResult | list[StepResult] | None:
        """Run one step.

        Args:
            step_index: Zero-based index since the latest reset.
            events: Input events not seen by this thread before.

        Returns:
            Model channels, one UI frame, or ``None``.
        """
        ...

    def is_finished(self) -> bool:
        """Return whether this thread has completed its workload."""
        return False

    def reset(self) -> None:
        """Reset thread-owned state for a new session generation.

        Raises:
            NotImplementedError: This thread does not support reset.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset.")

    def close(self) -> None:
        """Release resources owned by this thread."""
        return

    @final
    def _invoke_async(self, operation: Callable[[StateT], None]) -> None:
        """Queue a state operation before the next :meth:`step` call.

        Args:
            operation: Callable receiving the thread-owned state.

        Raises:
            RuntimeError: The thread is shutting down.
        """
        with self._lifecycle_lock:
            if not self._accepting_messages:
                raise RuntimeError("Thread is shutting down.")
            self._message_queue.put(_Message(operation))

    @final
    def _run_loop(
        self,
        *,
        event_buffer: EventBuffer,
        reader_id: int,
        stop: threading.Event,
        failures: queue.Queue[BaseException],
        finished: threading.Event,
        publish: Callable[[int, list[StepResult]], None],
        max_steps: int | None = None,
    ) -> None:
        """Run model steps until stopped or finished.

        Args:
            event_buffer: Client input shared with the UI thread.
            reader_id: This thread's event reader ID.
            stop: Set when the session must stop.
            failures: Queue for uncaught errors.
            finished: Set after this method finishes.
            publish: Function called with each model result.
            max_steps: Maximum steps; ``None`` runs until stopped.
        """
        steps_run = 0
        last_run_started: float | None = None
        try:
            while not stop.is_set() and (max_steps is None or steps_run < max_steps):
                events, generation = event_buffer.read(reader_id)
                step_index = self._begin_run(events, generation, stop)
                if step_index is None:
                    break
                last_run_started = self._pace(last_run_started, stop)
                if stop.is_set():
                    break
                result = _model_results(self.step(step_index, events))
                self._finish_run(result)
                publish(generation, result)
                steps_run += 1
        except BaseException as error:
            failures.put(error)
            stop.set()
        finally:
            try:
                self._shutdown()
            except BaseException as error:
                failures.put(error)
                stop.set()

    @final
    def _begin_run(
        self,
        events: UserInputEvents,
        generation: int,
        stop: threading.Event,
    ) -> int | None:
        """Prepare one call to :meth:`step`."""
        self._run_message_batch()
        self.user_events = events
        if _contains_close(events):
            stop.set()
            return None
        if generation != self._generation:
            self.reset()
            self.latest_result = None
            self._step_index = 0
            self._generation = generation
        if self.is_finished():
            return None
        return self._step_index

    @final
    def _finish_run(self, result: StepResult | list[StepResult] | None) -> None:
        """Save one completed step."""
        self.latest_result = result
        self._step_index += 1

    @final
    def _shutdown(self) -> None:
        """Stop messages and close the thread once."""
        with self._lifecycle_lock:
            self._accepting_messages = False
            if self._closed:
                return
            self._closed = True
        try:
            self.close()
        finally:
            self._empty_message_queue()
            self.event_buffer.unregister(self.reader_id)
            self.finished.set()

    def _run_message_batch(self) -> None:
        batch: list[_Message[StateT]] = []
        with self._lifecycle_lock:
            while True:
                try:
                    batch.append(self._message_queue.get_nowait())
                except queue.Empty:
                    break
        for message in batch:
            result = message.operation(self.state)
            if result is not None:
                raise TypeError("Message operations must return None.")

    def _pace(self, last_run_started: float | None, stop: threading.Event) -> float:
        if self.frequency == 0 or last_run_started is None:
            return time.monotonic()
        earliest_start = last_run_started + 1.0 / self.frequency
        stop.wait(max(0.0, earliest_start - time.monotonic()))
        return time.monotonic()

    def _empty_message_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                return


class UIThread(IThread[StateT], ABC):
    """Thread whose output is sent to the client window."""

    def __init__(
        self,
        *,
        state: StateT,
        frequency: int,
        output_layout: VideoTensorLayout,
        presentation_manager: PresentationManager,
    ) -> None:
        super().__init__(state=state, frequency=frequency)
        self.output_layout = output_layout
        self._presentation_manager = presentation_manager

    @abstractmethod
    def step_ui(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Return the result to present for this UI iteration."""
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Return the result produced by :meth:`step_ui`."""
        return self.step_ui(step_index, events)

    @final
    def presented_model_frame(self, channel_index: int = 0) -> Tensor | None:
        """Return the current frame from one model-result channel."""
        return self._presentation_manager.presented_frame(channel_index)

    @final
    def presented_model_frames(self) -> tuple[Tensor, ...]:
        """Return the current frame from every model-result channel."""
        return self._presentation_manager.presented_frames()


class BlitModelOutputToScreenThread(UIThread[None]):
    """Draw every model channel into one UI frame."""

    def step_ui(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Draw the model channels in list order."""
        del events
        output = None
        for frame in self.presented_model_frames():
            output = self._presentation_manager.composite(output, frame)
        if output is None:
            return None
        return StepResult(
            step_index=step_index,
            output=_frame_to_layout(output, self.output_layout),
            frame_count=1,
            output_layout=self.output_layout,
        )

    def reset(self) -> None:
        return


def _contains_close(events: UserInputEvents) -> bool:
    return any(
        isinstance(event.get_event_data(), CloseUserInputEventData)
        for event in events.get_events()
    )


def _model_results(
    result: StepResult | list[StepResult] | None,
) -> list[StepResult]:
    if result is None or isinstance(result, StepResult):
        raise TypeError("A model-generation thread must return a list of StepResult.")
    return result


def _frame_to_layout(frame: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Add singleton time, batch, and view dimensions for ``layout``."""
    if layout is VideoTensorLayout.tchw:
        return frame.unsqueeze(0)
    if layout is VideoTensorLayout.btchw:
        return frame.unsqueeze(0).unsqueeze(0)
    if layout is VideoTensorLayout.bcthw:
        return frame.unsqueeze(0).unsqueeze(2)
    if layout is VideoTensorLayout.bvtchw:
        return frame.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported presentation layout: {layout}.")


def invoke_async(thread: IThread[StateT], operation: Callable[[StateT], None]) -> None:
    """Queue ``operation`` against ``thread`` state before its next step."""
    thread._invoke_async(operation)


__all__ = [
    "BlitModelOutputToScreenThread",
    "IThread",
    "UIThread",
    "invoke_async",
]
