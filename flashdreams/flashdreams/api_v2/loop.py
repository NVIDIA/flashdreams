# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model and UI loops for a session."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, final

import torch
from torch import Tensor

from flashdreams.api_v2.client_window import ClientWindowLink
from flashdreams.api_v2.profiler import IProfiler
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.profiler import NullProfiler
from flashdreams.runtime_v2.profiler_utils import (
    bind_inference_profiler,
    unbind_inference_profiler,
)
from flashdreams.runtime_v2.selected_file import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILE_COUNT,
    DEFAULT_MAX_TOTAL_BYTES,
    FileSelectionRequest,
    normalize_selected_file_accept,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from flashdreams.runtime_v2.presentation_manager import PresentationManager
    from flashdreams.runtime_v2.session_desc import SessionDesc

StateT = TypeVar("StateT")


class _ModelStepControl(Protocol):
    """Private execution hook for step admission and synchronized input."""

    def ready(self, *, stopping: bool, failed: bool = False) -> bool: ...

    def inputs(
        self, events: UserInputEvents, generation: int
    ) -> tuple[UserInputEvents, int]: ...


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


@dataclass(slots=True)
class _Message(Generic[StateT]):
    operation: Callable[[StateT], None]
    """Operation to run before the loop's next step."""


@dataclass(slots=True)
class UILoopRequests:
    """Changes requested by a UI loop for the session runtime to apply.
    Note: `_LoopRunResult::stop_requested` is not part of `UILoopRequests` because
    it is a model & ui_loop request.
    """

    new_session: SessionDesc | None = None
    """Replacement session description, or ``None`` to keep this session."""

    hide_cursor: bool | None = None
    """Cursor visibility change, or ``None`` to leave it unchanged."""

    lock_cursor_to_window: bool | None = None
    """Cursor capture change, or ``None`` to leave it unchanged."""

    new_window_size: tuple[int, int] | None = None
    """Requested downstream window size, or ``None`` to leave it unchanged."""

    links: tuple[ClientWindowLink, ...] | None = None
    """Replacement clickable regions, or ``None`` to leave them unchanged."""

    file_selections: list[FileSelectionRequest] = field(default_factory=list)
    """File-selector requests in call order. The window answers each one."""


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
        profiler: IProfiler | None = None,
    ) -> None:
        """Store objects supplied when this loop is registered with a session.

        Args:
            state: State used by this loop.
            frequency: Maximum steps per second; zero disables pacing.
            shutdown_event: Event to signal that the loop should shutdown.
            failure_queue: Queue that stores loop failures/exceptions.
            profiler: Profiler the session runs with; records nothing by
                default.

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
        self.profiler: IProfiler = profiler if profiler is not None else NullProfiler()
        self._message_queue: queue.Queue[_Message[StateT]] = queue.Queue()
        self.user_events = UserInputEvents([])
        self._pending_user_events: list[UserInputEvent] = []
        self.latest_result: list[StepResult] | None = None
        self._step_index = 0
        self._generation = 0
        self._accepting_messages = True
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._shutdown_event = shutdown_event
        self._failure_queue = failure_queue
        self._initialize_loop_state()

    def _initialize_loop_state(self) -> None:
        """Initialize state specific to this kind of loop."""
        return

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Run one step.

        Both kinds of loop return ``list[StepResult]``. A model loop returns
        one entry per channel, or an empty list when the step produced no
        presentable output. A UI loop returns one frame to present, or an
        empty list to present nothing this step. A UI list longer than one
        raises :class:`TypeError`.

        Args:
            step_index: Zero-based index since the latest reset.
            events: Input events not seen by this loop before.

        Returns:
            Model channels from a model loop, possibly empty; zero or one
            frame from a UI loop.
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
        transition = _parse_lifecycle_events(self._pending_user_events)
        if transition is not None:
            self._pending_user_events.clear()
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
    def _finish_run(
        self,
        result: list[StepResult] | None,
        *,
        step_completed: bool,
    ) -> None:
        """Finish one prepared run, saving its completed step when present.

        Args:
            result: Value returned by the completed step, or ``None`` when no
                step ran.
            step_completed: Whether :meth:`step` returned successfully and
                ``result`` should replace :attr:`latest_result`.
        """
        if step_completed:
            self.latest_result = result
        self._step_index += 1
        # Input read by a run with no step is dropped, such as an ON_DEMAND UI
        # tick without a new frame.
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

    def _pace(self, last_run_started: float | None) -> float:
        # Recorded every step, so a short pace reads as no wait rather than
        # a missing range.
        with self.profiler.range("model.pace"):
            if self.frequency == 0 or last_run_started is None:
                return time.monotonic()
            earliest_start = last_run_started + 1.0 / self.frequency
            self._shutdown_event.wait(max(0.0, earliest_start - time.monotonic()))
            return time.monotonic()

    def _empty_message_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                return


class IModelLoop(ILoop[StateT], ABC):
    """Loop that generates model results on the model thread.

    :meth:`ILoop.step` must return ``list[StepResult]`` here, one entry per
    channel, with every channel reporting the same ``frame_count``. An empty
    list means the step produced no presentable output.

    Every admitted rank executes the same model step and ordered collectives.
    The integration owns TP/CP partitioning and must gather all CP shards needed
    for publication before the presenting rank returns non-empty results.
    Workers return ``[]`` only after all required collectives, including gathers.
    Gather before decoding when it needs the complete latent/view sequence;
    decoding may run only on the presenting rank when the model permits it.
    TP reductions that already replicate a complete value need no extra gather.

    A run whose steps all present nothing still ends when the model loop does.
    Returning a bare :class:`StepResult` or ``None`` raises :class:`TypeError`.
    """

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """We expect IModelLoop's to return a list of StepResults, one result per channel."""
        ...

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
        steps_run_out: list[int] | None = None,
        step_control: _ModelStepControl | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Run model steps until shutdown or completion.

        Args:
            event_buffer: Client input shared by both loops.
            reader_id: This loop's event reader ID.
            publish: Function called with each model result and the cumulative
                seconds spent in :meth:`step` since the previous result.
            max_steps: Maximum steps; ``None`` runs until stopped. Callers
                pass the remaining session cap here.
            steps_run_out: When provided, receives how many model-loop
                iterations this call ran.
            step_control: Admission and input synchronization; ``None`` runs locally.
            device: Mesh device to bind on this model thread before running hooks.
        """
        profiler_token = bind_inference_profiler(self.profiler)
        steps_run = 0
        last_run_started: float | None = None
        unpublished_step_elapsed_s = 0.0
        unpublished_generation: int | None = None
        # Input read by steps whose output is still buffered: (arrivals, step start).
        unpublished_inputs: list[tuple[list[int], int]] = []
        self._set_inference_state(ModelInferenceState.RUNNING)
        try:
            if device is not None and device.type == "cuda":
                torch.cuda.set_device(device)
            while True:
                stopping = self._shutdown_event.is_set() or (
                    max_steps is not None and steps_run >= max_steps
                )
                if step_control is not None:
                    if not step_control.ready(stopping=stopping):
                        break
                elif stopping:
                    break
                events, generation = event_buffer.read(reader_id)
                received_ns = event_buffer.last_read_received_ns(reader_id)
                if step_control is not None:
                    events, generation = step_control.inputs(events, generation)
                if generation != unpublished_generation:
                    unpublished_step_elapsed_s = 0.0
                    unpublished_generation = generation
                    # The reset discarded the buffered output this input went into.
                    if unpublished_inputs:
                        self.profiler.input_dropped(
                            [
                                ns
                                for arrivals, _ in unpublished_inputs
                                for ns in arrivals
                            ]
                        )
                        unpublished_inputs.clear()
                result: list[StepResult] | None = None
                step_completed = False
                try:
                    try:
                        run = self._begin_run(events, generation)
                        if run.step_index is not None:
                            last_run_started = self._pace(last_run_started)
                    except BaseException:
                        if step_control is not None:
                            try:
                                step_control.ready(stopping=True, failed=True)
                            except BaseException:
                                # A departed peer must not replace the original failure.
                                pass
                        raise
                    stopping = run.step_index is None or self._shutdown_event.is_set()
                    if step_control is not None:
                        if not step_control.ready(stopping=stopping):
                            break
                    elif stopping:
                        break
                    # Admission commits every rank to this step. A later UI stop is
                    # handled at the next boundary, never by skipping a collective.
                    assert run.step_index is not None
                    step_started_at = time.monotonic()
                    step_started_ns = time.monotonic_ns()
                    with self.profiler.range(f"model.step[{run.step_index}]"):
                        raw_result = self.step(run.step_index, self.user_events)
                    step_elapsed_s = time.monotonic() - step_started_at
                    result = _model_results(raw_result)
                    step_completed = True
                finally:
                    self._finish_run(result, step_completed=step_completed)
                unpublished_step_elapsed_s += step_elapsed_s
                if received_ns:
                    unpublished_inputs.append((received_ns, step_started_ns))

                # Carry timing across steps whose output remains buffered.
                if result:
                    # Keyed by the result's own step index, which is what the
                    # presenter sees, and recorded before the result can be shown.
                    step = (generation, result[0].step_index)
                    self.profiler.event(
                        "model.frame", count=result[0].frame_count, step=step
                    )
                    for arrivals, started_ns in unpublished_inputs:
                        self.profiler.input_consumed(step, arrivals, started_ns)
                    unpublished_inputs.clear()
                    # Read the rates before publishing: afterwards the result
                    # belongs to the presentation thread and must not be touched.
                    step_metrics = self.profiler.collect_fps()
                    step_metrics.update(self.profiler.collect_input_latency_ms())
                    if step_metrics:
                        for channel in result:
                            if channel.metrics is None:
                                # Profiling is on, so the result has measurements.
                                # Set the field on the frozen result the way its
                                # own __post_init__ does, keeping its output event.
                                object.__setattr__(
                                    channel, "metrics", dict(step_metrics)
                                )
                            else:
                                channel.metrics.update(step_metrics)
                    with self.profiler.range("model.publish"):
                        publish(generation, result, unpublished_step_elapsed_s)
                    unpublished_step_elapsed_s = 0.0
                steps_run += 1
        except BaseException as error:
            self._failure_queue.put(error)
        finally:
            if steps_run_out is not None:
                steps_run_out.append(steps_run)
            unbind_inference_profiler(profiler_token)
            self._set_inference_state(ModelInferenceState.FINISHED)
            try:
                self._shutdown()
            except BaseException as error:
                self._failure_queue.put(error)


class IUILoop(ILoop[StateT], ABC):
    """Loop whose output is sent to the client window.

    :meth:`ILoop.step` must return ``list[StepResult]`` here: one frame to
    present, or an empty list to present nothing this step. A list longer than
    one raises :class:`TypeError`. Model frames to draw come from
    :meth:`presented_model_frame` and :meth:`presented_model_frames` rather
    than from the model loop directly.
    :meth:`has_pending_model_frames` and :attr:`presented_model_frame_count`
    report whether more model frames are waiting and how many have been
    selected.
    """

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return zero or one :class:`StepResult` to present this tick."""
        ...

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

        self._ui_loop_requests: UILoopRequests | None = None

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
        self.get_or_create_ui_loop_requests().new_session = session_desc

    @final
    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Request that the client window show or hide its cursor."""
        if not isinstance(hide_cursor, bool):
            raise TypeError("hide_cursor must be a bool.")
        self.get_or_create_ui_loop_requests().hide_cursor = hide_cursor

    @final
    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Request that the client window release or capture pointer motion."""
        if not isinstance(lock_cursor_to_window, bool):
            raise TypeError("lock_cursor_to_window must be a bool.")
        self.get_or_create_ui_loop_requests().lock_cursor_to_window = (
            lock_cursor_to_window
        )

    @final
    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Request a positive downstream client-window width and height.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.

        Raises:
            TypeError: The size is not a pair of integers.
            ValueError: Either dimension is not positive.
        """
        if (
            not isinstance(new_window_size, tuple)
            or len(new_window_size) != 2
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                for dimension in new_window_size
            )
        ):
            raise TypeError("new_window_size must be a tuple of two integers.")
        if any(dimension <= 0 for dimension in new_window_size):
            raise ValueError("new_window_size dimensions must be > 0.")
        self.get_or_create_ui_loop_requests().new_window_size = new_window_size

    @final
    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Replace up to sixteen clickable regions of the output content.

        Args:
            links: Immutable regions; an empty tuple clears existing links.

        Raises:
            TypeError: The request is not a tuple of ``ClientWindowLink`` objects.
            ValueError: More than sixteen regions are supplied.
        """
        if not isinstance(links, tuple) or any(
            not isinstance(link, ClientWindowLink) for link in links
        ):
            raise TypeError("links must be a tuple of ClientWindowLink objects.")
        if len(links) > 16:
            raise ValueError("At most sixteen client-window links are supported.")
        self.get_or_create_ui_loop_requests().links = links

    @final
    def request_file_selector(
        self,
        request_id: str,
        initial_path: str | None = None,
        *,
        accept: Sequence[str] = (),
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_file_count: int = DEFAULT_MAX_FILE_COUNT,
    ) -> None:
        """Ask the client to show a file selector.

        The result arrives on a later tick, never during the step that asked,
        as one
        :class:`~flashdreams.runtime_v2.user_input_event.SelectedFilesUserInputEvent`
        carrying ``request_id``. One selector is open at a time; a call made
        while one is open is answered ``BUSY``. A selector stays open across a
        session reset; a replaced or ended session gets no result.

        Read ``status`` first; ``files`` is empty unless it is ``OK``:

        - ``OK``: each chosen file as its base ``name`` and ``data`` bytes.
        - ``CANCELLED``: the user dismissed the selector.
        - ``BUSY``: another selector was open.
        - ``DISALLOWED_TYPE``: a file did not match ``accept``.
        - ``TOO_LARGE``: a file was larger than ``max_file_bytes``, the files
          together were larger than ``max_total_bytes``, or more than
          ``max_file_count`` files were chosen.
        - ``UNAVAILABLE``: the window cannot show a selector, or it failed.

        ``accept`` checks filename suffixes only, so treat ``data`` as
        untrusted input.

        Example:

        .. code-block:: python

            from flashdreams.runtime_v2.selected_file import SelectedFilesStatus
            from flashdreams.runtime_v2.user_input_event import (
                SelectedFilesUserInputEvent,
            )

            if imgui.button("Open image"):
                self.request_file_selector("open-image", accept=(".png",))
            for event in self.user_events.get_events():
                if not isinstance(event, SelectedFilesUserInputEvent):
                    continue
                if event.request_id != "open-image":
                    continue
                if event.status is SelectedFilesStatus.OK:
                    image_bytes = event.files[0].data
                else:
                    self.state.message = f"No image: {event.status.value}"

        Args:
            request_id: Names the control that asked, so its result can be
                told apart from another control's. Echoed on the result.
            initial_path: Directory hint for the native selector; ``None``
                uses the host user's home directory. The browser ignores it.
            accept: Filename suffixes such as ``.png``. Empty allows any type.
            max_file_bytes: Largest single file in bytes. Must not exceed
                ``max_total_bytes``.
            max_total_bytes: Most bytes all chosen files may add up to. Has no
                effect when ``max_file_count`` is one.
            max_file_count: Most files the selector may return. Above one, the
                user may pick several.

        Raises:
            TypeError: ``request_id`` or ``initial_path`` is not a string, or
                ``accept`` or a limit has the wrong type.
            ValueError: ``request_id`` or ``initial_path`` is empty, ``accept``
                is invalid, a limit is not positive, or ``max_file_bytes``
                exceeds ``max_total_bytes``.
        """
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string.")
        if not request_id:
            raise ValueError("request_id must be a non-empty string.")
        if initial_path is None:
            resolved_initial_path = str(Path.home())
        else:
            if not isinstance(initial_path, str):
                raise TypeError("initial_path must be a string.")
            if not initial_path:
                raise ValueError("initial_path must be a non-empty string.")
            resolved_initial_path = initial_path
        suffixes = normalize_selected_file_accept(accept)
        for name, limit in (
            ("max_file_bytes", max_file_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_file_count", max_file_count),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"{name} must be an integer.")
            if limit <= 0:
                raise ValueError(f"{name} must be > 0.")
        if max_file_bytes > max_total_bytes:
            raise ValueError("max_file_bytes must not exceed max_total_bytes.")
        self.get_or_create_ui_loop_requests().file_selections.append(
            FileSelectionRequest(
                request_id=request_id,
                initial_path=resolved_initial_path,
                accept=suffixes,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                max_file_count=max_file_count,
            )
        )

    def get_or_create_ui_loop_requests(self) -> UILoopRequests:
        if self._ui_loop_requests is None:
            self._ui_loop_requests = UILoopRequests()
        return self._ui_loop_requests

    def flush_ui_loop_requests(self) -> UILoopRequests | None:
        """return UI loop requests and clear the internal state of requests."""
        requests = self._ui_loop_requests
        self._ui_loop_requests = None
        return requests

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

    @property
    @final
    def presented_model_frame_count(self) -> int:
        """Return how many model frames have been selected in this generation."""
        return self._presentation_manager.presented_frame_count

    @final
    def has_pending_model_frames(self) -> bool:
        """Return whether another model frame is ready to present."""
        return self._presentation_manager.has_pending_frames()


def _parse_lifecycle_events(
    events: list[UserInputEvent],
) -> _LoopRunResult | None:
    """Return the terminal lifecycle request, prioritizing close."""
    if any(isinstance(event, CloseUserInputEvent) for event in events):
        return _LoopRunResult(stop_requested=True)
    return None


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
    "FileSelectionRequest",
    "ILoop",
    "IModelLoop",
    "IUILoop",
    "ModelInferenceState",
    "UILoopRequests",
    "invoke_async",
]
