# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model and UI loops for a session."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

from torch import Tensor

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    NewSessionUserInputEvent,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from flashdreams.runtime_v2.presentation_manager import PresentationManager
    from flashdreams.runtime_v2.session_desc import SessionDesc

StateT = TypeVar("StateT")


class ModelInferenceState(Enum):
    """Lifecycle state of model inference observed by the UI loop."""

    NOT_STARTED = "not_started"
    """Model inference has not started."""

    RUNNING = "running"
    """The model loop is running."""

    FINISHED = "finished"
    """The model loop has returned or failed."""


@dataclass(frozen=True, slots=True)
class _LoopRunResult:
    """Describe whether a loop should step or yield lifecycle control."""

    step_index: int | None = None
    """Index to step, or ``None`` when no step should run."""

    stop_requested: bool = False
    """Whether the session should stop without a replacement."""

    new_session_request: SessionDesc | None = None
    """Replacement session description, or ``None`` when none was requested."""


@dataclass(slots=True)
class _Message(Generic[StateT]):
    operation: Callable[[StateT], None]
    """Operation to run before the loop's next step."""


class ILoop(ABC, Generic[StateT]):
    """Shared state, messaging, and lifecycle for a session loop.

    A session registers one of each kind: an :class:`IModelLoop` that generates
    and an :class:`IUILoop` that presents. They run on separate threads and own
    their own ``state``, so the only supported way for one to change the other's
    is :func:`invoke_async`.
    """

    @final
    def register_session_loop_objects(
        self,
        *,
        state: StateT,
        frequency: int,
        shutdown_event: threading.Event,
        failure_queue: queue.Queue[BaseException],
    ) -> None:
        """Store objects supplied when this loop is registered with a session.

        Args:
            state: State used by this loop.
            frequency: Maximum steps per second; zero disables pacing.
            shutdown_event: Event to signal that the loop should shutdown.
            failure_queue: Queue that stores loop failures/exceptions.

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
        self._pending_user_events: list[UserInputEvent] = []
        self.latest_result: StepResult | list[StepResult] | None = None
        self._step_index = 0
        self._generation = 0
        self._accepting_messages = True
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._shutdown_event = shutdown_event
        self._failure_queue = failure_queue
        self._new_session_request: SessionDesc | None = None
        self._new_session_event_desc: SessionDesc | None = None
        self._initialize_loop_state()

    def _initialize_loop_state(self) -> None:
        """Initialize state specific to this kind of loop."""
        return

    @abstractmethod
    def step(
        self, step_index: int, events: UserInputEvents
    ) -> StepResult | list[StepResult] | None:
        """Run one step.

        The two kinds of loop return different things, and the runtime rejects
        the wrong one: a model loop must return ``list[StepResult]``, one entry
        per channel, and a UI loop must return one :class:`StepResult` or
        ``None`` to present nothing this step.

        Args:
            step_index: Zero-based index since the latest reset.
            events: Input events not seen by this loop before.

        Returns:
            Model channels from a model loop; one frame, or ``None``, from a UI
            loop.
        """
        ...

    def is_finished(self) -> bool:
        """Return whether this loop has completed its workload."""
        return False

    def reset(self) -> None:
        """Reset loop-owned state for a new session generation.

        Called when a client asks for a reset, before the next :meth:`step`.
        Overriding this is what makes a loop resettable, so implement it even
        when there is nothing to undo.

        Raises:
            NotImplementedError: This loop does not support reset.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset.")

    def close(self) -> None:
        """Release resources owned by this loop."""
        return

    @final
    def _invoke_async(self, operation: Callable[[StateT], None]) -> None:
        """Queue a state operation before the next :meth:`step` call.

        Args:
            operation: Callable receiving the loop-owned state.

        Raises:
            RuntimeError: The loop is shutting down.
        """
        with self._lifecycle_lock:
            if not self._accepting_messages:
                raise RuntimeError("Loop is shutting down.")
            self._message_queue.put(_Message(operation))

    @final
    def _begin_run(
        self,
        events: UserInputEvents,
        generation: int,
    ) -> _LoopRunResult:
        """Prepare one step or return a lifecycle request to the caller."""
        self._run_message_batch()
        self._pending_user_events.extend(events.get_events())
        transition = _lifecycle_transition(
            self._pending_user_events,
            self._new_session_request,
            self._new_session_event_desc,
        )
        if transition is not None:
            self._pending_user_events.clear()
            self._new_session_request = None
            return transition
        if self._shutdown_event.is_set():
            return _LoopRunResult(stop_requested=True)
        self.user_events = UserInputEvents(list(self._pending_user_events))
        if generation != self._generation:
            self.reset()
            self.latest_result = None
            self._step_index = 0
            self._generation = generation
        if self.is_finished():
            return _LoopRunResult()
        return _LoopRunResult(step_index=self._step_index)

    @final
    def _finish_run(self, result: StepResult | list[StepResult] | None) -> None:
        """Save one completed step."""
        self.latest_result = result
        self._step_index += 1
        self._pending_user_events.clear()

    @final
    def _shutdown(self) -> None:
        """Stop messages and close the loop once."""
        with self._lifecycle_lock:
            self._accepting_messages = False
            if self._closed:
                return
            self._closed = True
        try:
            self.close()
        finally:
            self._empty_message_queue()

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

    def _empty_message_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                return


class IModelLoop(ILoop[StateT], ABC):
    """Loop that generates model results on the model thread.

    :meth:`ILoop.step` must return ``list[StepResult]`` here, one entry per
    channel, with every channel reporting the same ``frame_count``. Returning a
    bare :class:`StepResult` or ``None`` raises :class:`TypeError`.
    """

    @final
    def _initialize_loop_state(self) -> None:
        """Initialize model-inference state."""
        self._inference_state = ModelInferenceState.NOT_STARTED
        self._inference_state_lock = threading.Lock()

    @property
    @final
    def inference_state(self) -> ModelInferenceState:
        """Return whether this model loop has started, is running, or finished."""
        with self._inference_state_lock:
            return self._inference_state

    def _set_inference_state(self, state: ModelInferenceState) -> None:
        """Set this model loop's inference state."""
        with self._inference_state_lock:
            self._inference_state = state

    @final
    def _run_model_loop(
        self,
        *,
        event_buffer: EventBuffer,
        reader_id: int,
        publish: Callable[[int, list[StepResult], float], None],
        max_steps: int | None = None,
    ) -> None:
        """Run model steps until shutdown or completion.

        Args:
            event_buffer: Client input shared by both loops.
            reader_id: This loop's event reader ID.
            publish: Function called with each model result and the elapsed
                seconds spent in :meth:`step`.
            max_steps: Maximum steps; ``None`` runs until stopped.
        """
        steps_run = 0
        last_run_started: float | None = None
        self._set_inference_state(ModelInferenceState.RUNNING)
        try:
            while not self._shutdown_event.is_set() and (
                max_steps is None or steps_run < max_steps
            ):
                events, generation = event_buffer.read(reader_id)
                run = self._begin_run(events, generation)
                if run.step_index is None:
                    break
                if self.frequency != 0 and last_run_started is not None:
                    earliest_start = last_run_started + 1.0 / self.frequency
                    self._shutdown_event.wait(
                        max(0.0, earliest_start - time.monotonic())
                    )
                last_run_started = time.monotonic()
                if self._shutdown_event.is_set():
                    break
                step_started_at = time.monotonic()
                raw_result = self.step(run.step_index, self.user_events)
                step_elapsed_s = time.monotonic() - step_started_at
                result = _model_results(raw_result)
                self._finish_run(result)
                publish(generation, result, step_elapsed_s)
                steps_run += 1
        except BaseException as error:
            self._failure_queue.put(error)
        finally:
            self._set_inference_state(ModelInferenceState.FINISHED)
            try:
                self._shutdown()
            except BaseException as error:
                self._failure_queue.put(error)


class IUILoop(ILoop[StateT], ABC):
    """Loop whose output is sent to the client window.

    :meth:`ILoop.step` must return one :class:`StepResult` or ``None`` here.
    Model frames to draw come from :meth:`presented_model_frame` and
    :meth:`presented_model_frames` rather than from the model loop directly.
    """

    @final
    def register_session_ui_loop_objects(
        self,
        *,
        session_desc: SessionDesc,
        presentation_manager: PresentationManager,
    ) -> None:
        """Store UI objects supplied when this loop is registered with a session.

        Args:
            session_desc: Description of the session this UI presents.
            presentation_manager: Buffer containing model frames.
        """
        self.session_desc = session_desc
        self.output_layout = session_desc.output_layout
        self._presentation_manager = presentation_manager
        self._new_session_event_desc = session_desc

    @final
    def _set_model_loop(self, model_loop: IModelLoop[Any]) -> None:
        """Store the model loop whose inference lifecycle this UI observes."""
        self._model_loop = model_loop

    @property
    @final
    def model_inference_state(self) -> ModelInferenceState:
        """Return whether model inference has started, is running, or finished."""
        return self._model_loop.inference_state

    @final
    def request_new_session(self, session_desc: SessionDesc) -> None:
        """Ask the runtime to replace this session after the current UI step.

        Args:
            session_desc: Fully resolved description for the replacement session.
        """
        self._new_session_request = session_desc

    @final
    def presented_model_frame(
        self,
        channel_index: int = 0,
    ) -> Tensor | None:
        """Return the current frame from one model-result channel.

        Args:
            channel_index: Channel to read, indexed as the model loop returned
                them.

        Returns:
            A ``[C, H, W]`` frame with one, three or four channels, or ``None``
            before the first model result has been presented.

        Raises:
            IndexError: The presented result has no such channel.
            ValueError: The presentation stream is on a different CUDA device
                from the presented result.
        """
        return self._presentation_manager.presented_frame(channel_index)

    @final
    def presented_model_frames(self) -> tuple[Tensor, ...]:
        """Return the current frame from every model-result channel.

        Returns:
            One ``[C, H, W]`` frame per channel, bottom channel first, or an
            empty tuple before the first model result has been presented.

        Raises:
            ValueError: The presentation stream is on a different CUDA device
                from a presented result.
        """
        return self._presentation_manager.presented_frames()


def _lifecycle_transition(
    events: list[UserInputEvent],
    new_session_request: SessionDesc | None,
    new_session_event_desc: SessionDesc | None,
) -> _LoopRunResult | None:
    """Return the latest terminal lifecycle request, if any."""
    transition = (
        None
        if new_session_request is None
        else _LoopRunResult(new_session_request=new_session_request)
    )
    for event in events:
        if isinstance(event, CloseUserInputEvent):
            transition = _LoopRunResult(stop_requested=True)
        elif isinstance(event, NewSessionUserInputEvent):
            transition = _LoopRunResult(new_session_request=new_session_event_desc)
    return transition


def _model_results(
    result: StepResult | list[StepResult] | None,
) -> list[StepResult]:
    if result is None or isinstance(result, StepResult):
        raise TypeError("A model loop must return a list of StepResult.")
    return result


def invoke_async(loop: ILoop[StateT], operation: Callable[[StateT], None]) -> None:
    """Queue ``operation`` against ``loop`` state before its next step.

    Returns immediately. ``loop`` snapshots its queue before its next
    :meth:`ILoop.step` and runs what was in it, on its own thread. Anything
    still queued at shutdown is dropped, so two loops cannot keep each other
    alive by messaging back and forth.

    Args:
        loop: Loop whose state the operation runs against.
        operation: Callable receiving that loop's state. Must return ``None``.

    Raises:
        RuntimeError: ``loop`` is shutting down.
    """
    loop._invoke_async(operation)


__all__ = [
    "ILoop",
    "IModelLoop",
    "IUILoop",
    "ModelInferenceState",
    "invoke_async",
]
