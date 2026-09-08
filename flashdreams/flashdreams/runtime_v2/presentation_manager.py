# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffer and present model frames."""

import logging
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.recent_frame_rate import RecentFrameRateTracker
from flashdreams.runtime_v2.session_desc import BackpressureMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_PRESENTATION_STREAM_PRIORITY = -1
"""Prefer short presentation work over queued model kernels."""

_MODEL_FPS_WINDOW_SECONDS = 2.0
"""Wall-time window used to estimate generated-frame throughput."""

_PRESENTATION_DRAIN_MARGIN = 0.9
"""Present slightly faster than recent model FPS when possible."""

_TRACE_LOGGER = logging.getLogger("flashdreams.runtime_v2.chunk_trace")
_TRACE_PREFIX = "[runtime-v2-chunk-trace]"


class _PresentationClock:
    """Schedule model-frame advances at recent model-step throughput."""

    def __init__(
        self,
        frames_per_second: int,
        maximum_frames_per_second: int | None = None,
    ) -> None:
        maximum_frames_per_second = maximum_frames_per_second or frames_per_second
        self._minimum_frame_interval = 1.0 / maximum_frames_per_second
        self._fallback_frame_interval = max(
            1.0 / frames_per_second,
            self._minimum_frame_interval,
        )
        self._frame_interval = self._fallback_frame_interval
        self._next_frame_at: float | None = None
        self._generation: int | None = None
        self._model_frame_rate = RecentFrameRateTracker(
            window_seconds=_MODEL_FPS_WINDOW_SECONDS
        )
        self._has_completion_baseline = False
        self._last_completion_at: float | None = None
        self._lock = threading.Lock()

    @property
    def frames_per_second(self) -> float:
        """Return the current model-frame presentation rate."""
        with self._lock:
            return 1.0 / self._frame_interval

    def observe_model_output(
        self,
        *,
        now: float,
        generation: int,
        frame_count: int,
        step_elapsed_s: float,
    ) -> None:
        """Add one completed model chunk to the rolling FPS estimate.

        Args:
            now: Monotonic completion time for the chunk.
            generation: Session generation that produced the chunk.
            frame_count: Number of generated frames in the chunk.
            step_elapsed_s: Time spent running the model step, excluding loop
                pacing and downstream publication backpressure.

        Raises:
            TypeError: ``frame_count`` is not an integer.
            ValueError: An observation value is invalid or ``now`` precedes
                the latest observation.
        """
        with self._lock:
            if self._generation is None or generation > self._generation:
                self._reset_generation(generation)
            elif generation < self._generation:
                return

            if self._last_completion_at is not None and now < self._last_completion_at:
                raise ValueError("now must not precede the latest observation.")
            frames_per_second = self._model_frame_rate.observe(
                completed_at=now,
                frame_count=frame_count,
                elapsed_s=step_elapsed_s,
            )
            self._last_completion_at = now
            if not self._has_completion_baseline:
                # Keep the configured cadence for the first chunk and avoid
                # letting one-time model warmup set the steady presentation rate.
                self._has_completion_baseline = True
                self._model_frame_rate.reset()
                return
            self._frame_interval = max(
                (1.0 / frames_per_second) * _PRESENTATION_DRAIN_MARGIN,
                self._minimum_frame_interval,
            )

    def is_due(self, now: float, generation: int, *, backlog: bool = False) -> bool:
        """Return whether the next model frame may be selected."""
        with self._lock:
            if generation != self._generation:
                self._reset_generation(generation)
            return backlog or self._next_frame_at is None or now >= self._next_frame_at

    def mark_advanced(self, now: float, *, backlog: bool = False) -> None:
        """Record one selected frame without catching up after a long stall."""
        frame_interval = (
            self._minimum_frame_interval if backlog else self._frame_interval
        )
        with self._lock:
            if backlog:
                self._next_frame_at = now + frame_interval
                return
            next_frame_at = self._next_frame_at
            if next_frame_at is None or now - next_frame_at >= frame_interval:
                self._next_frame_at = now + frame_interval
            else:
                self._next_frame_at = next_frame_at + frame_interval

    def _reset_generation(self, generation: int) -> None:
        self._generation = generation
        self._frame_interval = self._fallback_frame_interval
        self._next_frame_at = None
        self._model_frame_rate.reset()
        self._has_completion_baseline = False
        self._last_completion_at = None


class PresentationManager:
    """Buffer model output for a session's UI thread.

    The model thread publishes a chunk of channels per step into a bounded
    chunk queue; the UI thread calls :meth:`advance` once per tick to select
    the next frame. The queue holds at most one chunk; once the UI thread takes
    that chunk, the remaining frames stay in the active presented chunk and no
    longer count as backlog.

    :class:`BackpressureMode` decides what publishing does when the queue is
    full. Chunks that could not be kept are counted in
    :attr:`dropped_for_space` and :attr:`discarded_at_reset` rather than lost
    silently.

    :meth:`presentation_context` keeps the complete UI presentation path on one
    high-priority CUDA stream, separate from model inference.
    """

    def __init__(self, *, device: torch.device | None = None) -> None:
        """Create a frame manager and its CUDA presentation stream.

        Args:
            device: Presentation device. ``None`` uses the device of the first
                CUDA frame. A CPU device disables the CUDA stream.

        Raises:
            ValueError: ``device`` is neither a CPU nor CUDA device.
        """
        self._bufferedChunks: queue.Queue[tuple[int, list[StepResult]]] = queue.Queue(
            maxsize=1
        )
        self._presentation_clock = _PresentationClock(frames_per_second=30)
        self._backpressure_mode = BackpressureMode.BLOCK
        self._stop = threading.Event()
        self._put_timeout = 1.0 / 30.0
        self._counter_lock = threading.Lock()
        self._generation = 0
        self._presented_chunk: list[StepResult] | None = None
        self._frame_index = -1
        self._presented_frame_count = 0
        self._dropped_for_space = 0
        self._discarded_at_reset = 0
        self._stream_lock = threading.Lock()
        self._infer_stream_device = device is None
        self._trace_chunk_lifecycle = False
        self._presentation_stream: torch.cuda.Stream | None = None
        if device is not None:
            device = torch.device(device)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("Presentation requires a CPU or CUDA device.")
            if device.type == "cuda":
                device = resolve_cuda_device(device)
                with torch.cuda.device(device):
                    self._presentation_stream = torch.cuda.Stream(
                        device=device,
                        priority=_PRESENTATION_STREAM_PRIORITY,
                    )

    def configure(
        self,
        *,
        backpressure_mode: BackpressureMode,
        stop: threading.Event,
        put_timeout: float,
        trace_chunk_lifecycle: bool = False,
        frames_per_second: int = 30,
        maximum_frames_per_second: int | None = None,
    ) -> None:
        """Set presentation timing and backpressure mode.

        Called by ``run_session`` before either thread uses this.

        Args:
            backpressure_mode: What :meth:`publish` does when the queue is full.
            stop: Session shutdown event, so a blocked publish gives up.
            put_timeout: How long a blocked publish waits before rechecking
                ``stop``, in seconds.
            trace_chunk_lifecycle: Emit chunk lifecycle diagnostics.
            frames_per_second: Initial video presentation rate.
            maximum_frames_per_second: Upper bound for presentation cadence;
                ``None`` uses ``frames_per_second``.
        """
        self._reset_buffered_chunks()
        self._presentation_clock = _PresentationClock(
            frames_per_second=frames_per_second,
            maximum_frames_per_second=maximum_frames_per_second,
        )
        self._backpressure_mode = backpressure_mode
        self._stop = stop
        self._put_timeout = put_timeout
        self._trace_chunk_lifecycle = trace_chunk_lifecycle

    def publish(
        self,
        generation: int,
        chunk: list[StepResult],
        *,
        step_elapsed_s: float | None = None,
    ) -> None:
        """Add one completed model step to the presentation queue.

        Called on the model thread. ``BLOCK`` waits here when the
        queue is full, until there is room or the session stops;
        ``DROP_OLDEST`` evicts instead and returns.

        Args:
            generation: Reset generation the chunk was generated in. A chunk
                from an earlier one is discarded rather than presented.
            chunk: One :class:`StepResult` per model channel.
            step_elapsed_s: Time spent running the model step; ``None`` leaves
                the presentation cadence unchanged.

        Raises:
            ValueError: ``chunk`` is empty, or its channels disagree about
                ``frame_count``.
            TypeError: ``chunk`` holds something other than results.
        """
        if not chunk:
            raise ValueError("A presented chunk must contain at least one channel.")
        if any(not isinstance(result, StepResult) for result in chunk):
            raise TypeError("Every model channel must be a StepResult.")
        frame_count = chunk[0].frame_count
        if frame_count <= 0 or any(item.frame_count != frame_count for item in chunk):
            raise ValueError("Every channel in a chunk must have the same frame_count.")
        self._ensure_presentation_stream(
            next(
                (result._output.device for result in chunk if result._output.is_cuda),
                None,
            )
        )
        if step_elapsed_s is not None:
            self._presentation_clock.observe_model_output(
                now=time.monotonic(),
                generation=generation,
                frame_count=frame_count,
                step_elapsed_s=step_elapsed_s,
            )
        started_ns: int | None = None
        if self._trace_chunk_lifecycle:
            started_ns = time.monotonic_ns()
            self._trace(
                "publish_started",
                generation=generation,
                step=chunk[0].step_index,
                frames=frame_count,
                buffered_chunks=self.buffered_chunk_count,
                chunk_capacity=self.buffered_chunk_capacity,
            )
        pending = (generation, chunk)
        if self._backpressure_mode is BackpressureMode.DROP_OLDEST:
            self._publish_latest(pending)
            self._trace_publish_completed(pending, started_ns)
            return
        while not self._stop.is_set():
            try:
                self._bufferedChunks.put(pending, timeout=self._put_timeout)
                self._trace_publish_completed(pending, started_ns)
                return
            except queue.Full:
                continue
        if started_ns is not None:
            self._trace(
                "publish_stopped",
                generation=generation,
                step=chunk[0].step_index,
                wait_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
                buffered_chunks=self.buffered_chunk_count,
                chunk_capacity=self.buffered_chunk_capacity,
            )

    @contextmanager
    def presentation_context(self) -> Iterator[None]:
        """Make the manager-owned presentation stream current for a UI step.

        Managers without a CUDA presentation stream use the caller's current
        context.

        Yields:
            Control while the presentation stream is current.
        """
        stream = self._presentation_stream
        if stream is None:
            yield
            return
        device = resolve_cuda_device(stream.device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            yield

    def close(self) -> None:
        """Finish presentation work and release buffered output."""
        if self._presentation_stream is not None:
            self._presentation_stream.synchronize()
        self.clear()
        self._presentation_stream = None

    def advance(
        self,
        generation: int,
        *,
        now: float | None = None,
    ) -> tuple[bool, list[StepResult] | None]:
        """Move to the next model frame, if one is available.

        Called on the UI thread, once per tick. The manager advances only when
        the presentation clock is due, unless the chunk queue is full and needs
        to drain. A ``generation`` other than the last one seen drops what is
        being presented, so nothing generated before a reset survives it.

        Args:
            generation: Current reset generation, from the event buffer.
            now: Monotonic timestamp used for pacing; ``None`` reads the clock.

        Returns:
            Whether the frame changed, and the chunk for a newly selected chunk,
            which is ``None`` when the selected frame belongs to the same chunk
            as the previously presented frame.
        """
        if generation != self._generation:
            if self._presented_chunk is not None:
                self._trace_drop(
                    self._generation,
                    self._presented_chunk,
                    reason="generation_changed_active",
                )
            self._generation = generation
            self._presented_chunk = None
            self._frame_index = -1
            self._presented_frame_count = 0

        now = time.monotonic() if now is None else float(now)
        backlog = self.is_backlogged
        if not self._presentation_clock.is_due(now, generation, backlog=backlog):
            return False, None

        if (
            self._presented_chunk is not None
            and self._frame_index + 1 < self._presented_chunk[0].frame_count
        ):
            self._frame_index += 1
            self._presented_frame_count += 1
            self._trace_presented_frame(generation)
            self._presentation_clock.mark_advanced(now, backlog=backlog)
            return True, None

        chunk = self._take_buffered_chunk(
            generation,
            latest=self._backpressure_mode is BackpressureMode.DROP_OLDEST,
        )
        if chunk is None:
            return False, None
        self._presented_chunk = chunk
        self._frame_index = 0
        self._presented_frame_count += 1
        self._trace_presented_frame(generation)
        self._presentation_clock.mark_advanced(now, backlog=backlog)
        return True, chunk

    @property
    def presented_frame_count(self) -> int:
        """Return frames selected one-by-one in the current generation."""
        return self._presented_frame_count

    @property
    def dropped_for_space(self) -> int:
        """Return chunks dropped because presentation could not keep up."""
        with self._counter_lock:
            return self._dropped_for_space

    @property
    def discarded_at_reset(self) -> int:
        """Return chunks discarded because they predate a reset."""
        with self._counter_lock:
            return self._discarded_at_reset

    @property
    def buffered_chunk_count(self) -> int:
        """Return model chunks waiting in the bounded publish queue.

        The chunk currently being presented is intentionally excluded: only
        this queue depth controls whether :meth:`publish` blocks. As with
        :meth:`queue.Queue.qsize`, the value is a thread-safe point-in-time
        snapshot and may change immediately after it is returned.
        """
        return self._bufferedChunks.qsize()

    @property
    def buffer_capacity(self) -> int:
        """Return the maximum number of chunks that may wait to be presented."""
        return self._bufferedChunks.maxsize

    @property
    def buffered_chunk_capacity(self) -> int:
        """Return maximum chunks the publish queue is configured to hold."""
        return self._bufferedChunks.maxsize

    def presented_frame(
        self,
        channel_index: int,
    ) -> Tensor | None:
        """Return one frame ordered before the presentation stream.

        Args:
            channel_index: Model-result channel to read.

        Returns:
            The current ``[C, H, W]`` frame, or ``None`` before presentation
            starts.

        Raises:
            IndexError: The presented result has no such channel.
            ValueError: The presented result's layout or frame shape is
                unsupported.
        """
        if self._presented_chunk is None:
            return None
        try:
            result = self._presented_chunk[channel_index]
        except IndexError as error:
            raise IndexError(
                f"Presented chunk has {len(self._presented_chunk)} channels; "
                f"channel {channel_index} does not exist."
            ) from error
        with self.presentation_context():
            return _frame_at(result, self._frame_index)

    def presented_frames(self) -> tuple[Tensor, ...]:
        """Return all current frames ordered before the presentation stream."""
        if self._presented_chunk is None:
            return ()
        with self.presentation_context():
            return tuple(
                _frame_at(result, self._frame_index) for result in self._presented_chunk
            )

    def composite(self, bottom: Tensor | None, top: Tensor) -> Tensor:
        """Draw ``top`` over ``bottom``.

        Args:
            bottom: ``[C, H, W]`` frame to draw onto, or ``None`` to start from
                black.
            top: ``[C, H, W]`` frame to draw. Four channels is RGBA and blends;
                anything else replaces. Floating-point input is converted to
                ``bottom.dtype`` when needed.

        Returns:
            An RGB ``[3, H, W]`` frame.

        Raises:
            ValueError: The frames disagree about size or device, use
                incompatible dtypes, do not use the presentation-stream device,
                or are not presentable.
        """
        frames = (top,) if bottom is None else (bottom, top)
        self._ensure_presentation_stream(
            next((frame.device for frame in frames if frame.is_cuda), None)
        )
        stream = (
            self._presentation_stream
            if self._presentation_stream is not None
            and any(frame.is_cuda for frame in frames)
            else None
        )
        caller_stream: torch.cuda.Stream | None = None
        if stream is not None:
            device = resolve_cuda_device(stream.device)
            if any(
                not frame.is_cuda or resolve_cuda_device(frame.device) != device
                for frame in frames
            ):
                raise ValueError(
                    "Composited frames must use the presentation-stream device."
                )

            caller_stream = torch.cuda.current_stream(device)
            if caller_stream != stream:
                stream.wait_stream(caller_stream)
                for frame in frames:
                    frame.record_stream(stream)

        with self.presentation_context():
            if (
                bottom is not None
                and bottom.is_floating_point()
                and top.is_floating_point()
                and top.dtype != bottom.dtype
            ):
                top = top.to(dtype=bottom.dtype)
            output = _composite_frame(bottom, top)

        if stream is not None and caller_stream != stream:
            assert caller_stream is not None
            caller_stream.wait_stream(stream)
            output.record_stream(caller_stream)
        return output

    def _ensure_presentation_stream(self, device: torch.device | None) -> None:
        """Bind a default manager to the first CUDA device it presents."""
        if device is None or not self._infer_stream_device:
            return
        with self._stream_lock:
            if not self._infer_stream_device:
                return
            device = resolve_cuda_device(device)
            with torch.cuda.device(device):
                self._presentation_stream = torch.cuda.Stream(
                    device=device,
                    priority=_PRESENTATION_STREAM_PRIORITY,
                )
            self._infer_stream_device = False

    def has_pending_frames(self) -> bool:
        """Return whether another model frame is ready."""
        if (
            self._presented_chunk is not None
            and self._frame_index + 1 < self._presented_chunk[0].frame_count
        ):
            return True
        return not self._bufferedChunks.empty()

    @property
    def is_backlogged(self) -> bool:
        """Return whether a whole chunk is waiting behind the active one."""
        return self._bufferedChunks.full()

    def clear(self) -> None:
        """Discard buffered and currently presented model results."""
        self._presented_chunk = None
        self._frame_index = -1
        self._presented_frame_count = 0
        while True:
            try:
                self._bufferedChunks.get_nowait()
            except queue.Empty:
                return

    def _reset_buffered_chunks(self) -> None:
        self._bufferedChunks = queue.Queue(maxsize=1)

    def _publish_latest(self, pending: tuple[int, list[StepResult]]) -> None:
        while not self._stop.is_set():
            try:
                self._bufferedChunks.put_nowait(pending)
                return
            except queue.Full:
                try:
                    dropped_generation, dropped_chunk = (
                        self._bufferedChunks.get_nowait()
                    )
                    with self._counter_lock:
                        self._dropped_for_space += 1
                    self._trace_drop(
                        dropped_generation,
                        dropped_chunk,
                        reason="queue_full",
                        replacement=pending,
                    )
                except queue.Empty:
                    continue

    def _take_buffered_chunk(
        self, generation: int, *, latest: bool
    ) -> list[StepResult] | None:
        selected: list[StepResult] | None = None
        while True:
            try:
                chunk_generation, chunk = self._bufferedChunks.get_nowait()
            except queue.Empty:
                return selected
            if chunk_generation != generation:
                with self._counter_lock:
                    self._discarded_at_reset += 1
                self._trace_drop(
                    chunk_generation,
                    chunk,
                    reason="generation_mismatch",
                )
                continue
            if selected is not None:
                with self._counter_lock:
                    self._dropped_for_space += 1
                self._trace_drop(
                    generation,
                    selected,
                    reason="take_latest",
                    replacement=(chunk_generation, chunk),
                )
            selected = chunk
            if not latest:
                return selected

    def _trace_publish_completed(
        self,
        pending: tuple[int, list[StepResult]],
        started_ns: int | None,
    ) -> None:
        if started_ns is None:
            return
        generation, chunk = pending
        self._trace(
            "publish_completed",
            generation=generation,
            step=chunk[0].step_index,
            frames=chunk[0].frame_count,
            wait_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            buffered_chunks=self.buffered_chunk_count,
            chunk_capacity=self.buffered_chunk_capacity,
        )

    def _trace_presented_frame(self, generation: int) -> None:
        if not self._trace_chunk_lifecycle:
            return
        chunk = self._presented_chunk
        if chunk is None:
            return
        self._trace(
            "frame_presented",
            generation=generation,
            step=chunk[0].step_index,
            frame=self._frame_index,
            frames=chunk[0].frame_count,
            edge=(
                "both"
                if chunk[0].frame_count == 1
                else "first"
                if self._frame_index == 0
                else "last"
                if self._frame_index + 1 == chunk[0].frame_count
                else "middle"
            ),
            buffered_chunks=self.buffered_chunk_count,
            chunk_capacity=self.buffered_chunk_capacity,
        )

    def _trace_drop(
        self,
        generation: int,
        chunk: list[StepResult],
        *,
        reason: str,
        replacement: tuple[int, list[StepResult]] | None = None,
    ) -> None:
        if not self._trace_chunk_lifecycle:
            return
        fields: dict[str, object] = {
            "generation": generation,
            "step": chunk[0].step_index,
            "frames": chunk[0].frame_count,
            "reason": reason,
            "buffered_chunks": self.buffered_chunk_count,
        }
        if replacement is not None:
            replacement_generation, replacement_chunk = replacement
            fields["replacement_generation"] = replacement_generation
            fields["replacement_step"] = replacement_chunk[0].step_index
        self._trace("chunk_dropped", **fields)

    def _trace(self, phase: str, **fields: object) -> None:
        if not self._trace_chunk_lifecycle:
            return
        details = " ".join(f"{name}={value}" for name, value in fields.items())
        _TRACE_LOGGER.info(
            "%s phase=%s time_ns=%d %s",
            _TRACE_PREFIX,
            phase,
            time.monotonic_ns(),
            details,
        )


def _frame_at(result: StepResult, frame_index: int) -> Tensor:
    """Return one result frame as ``[C, H, W]``."""
    output = result.read_output()
    if result.output_layout is VideoTensorLayout.tchw:
        frame = output[frame_index]
    elif result.output_layout is VideoTensorLayout.btchw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("btchw presentation requires a batch size of one.")
        frame = output[0, frame_index]
    elif result.output_layout is VideoTensorLayout.bcthw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("bcthw presentation requires a batch size of one.")
        frame = output[0, :, frame_index]
    elif result.output_layout is VideoTensorLayout.bvtchw:
        if output.ndim != 6 or output.shape[:2] != (1, 1):
            raise ValueError("bvtchw presentation requires one batch and one view.")
        frame = output[0, 0, frame_index]
    else:
        raise ValueError(f"Unsupported presentation layout: {result.output_layout}.")
    _validate_frame(frame)
    return frame


def _validate_frame(frame: Tensor) -> None:
    if frame.ndim != 3 or frame.shape[0] not in (1, 3, 4):
        raise ValueError("A presented frame must have one, three, or four channels.")


def _composite_frame(bottom: Tensor | None, top: Tensor) -> Tensor:
    """Draw an RGB or RGBA frame over an RGB frame."""
    _validate_frame(top)
    color = top[:3]
    if color.shape[0] == 1:
        color = color.repeat(3, 1, 1)
    if bottom is not None:
        _validate_frame(bottom)
        bottom = bottom[:3]
        if bottom.shape[0] == 1:
            bottom = bottom.repeat(3, 1, 1)
        if color.shape[1:] != bottom.shape[1:]:
            raise ValueError("All composited frames must have the same dimensions.")
        if color.device != bottom.device or color.dtype != bottom.dtype:
            raise ValueError(
                "All composited frames must have the same device and dtype."
            )
    if top.shape[0] != 4:
        return color
    if not top.is_floating_point():
        raise ValueError("RGBA compositing requires a floating-point tensor.")
    if bottom is None:
        fill_value = -1.0 if color.is_floating_point() else 0
        bottom = torch.full_like(color, fill_value)
    alpha = top[3:4].to(device=bottom.device, dtype=torch.float32)
    alpha = alpha.clamp(0.0, 1.0).to(bottom.dtype)
    return torch.lerp(bottom, color, alpha)


__all__ = ["PresentationManager"]
