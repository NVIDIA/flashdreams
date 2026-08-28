# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffer and present model frames."""

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.session_desc import BackpressureMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_PRESENTATION_STREAM_PRIORITY = -1
"""Prefer short presentation work over queued model kernels."""


class PresentationManager:
    """Buffer model output for a session's UI thread.

    The model thread publishes a chunk of channels per step into a bounded
    queue; the UI thread calls :meth:`advance` once per tick to move to
    the next frame. A chunk holding several frames is walked frame by frame
    before another is taken, so a step that generated twelve frames is
    presented over twelve ticks rather than eleven being skipped.

    :class:`BackpressureMode` decides what publishing does when the queue is
    full. Chunks that could not be kept are counted in
    :attr:`dropped_for_space` and :attr:`discarded_at_reset` rather than lost
    silently, one count per chunk however many frames it held.

    :meth:`presentation_context` keeps the complete UI presentation path on one
    high-priority CUDA stream, separate from model inference.
    """

    def __init__(self, *, device: torch.device | None = None) -> None:
        """Create a frame manager and its CUDA presentation stream.

        Args:
            device: Presentation device; ``None`` or a CPU device does not
                create a CUDA stream.

        Raises:
            ValueError: ``device`` is neither a CPU nor CUDA device.
        """
        self._buffer: queue.Queue[tuple[int, list[StepResult]]] = queue.Queue(maxsize=2)
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
        max_pending: int,
        backpressure_mode: BackpressureMode,
        stop: threading.Event,
        put_timeout: float,
    ) -> None:
        """Set the queue size and backpressure mode.

        Called by ``run_session`` before either thread uses this.

        Args:
            max_pending: Model steps that may wait to be shown. Replaces the
                queue, so anything already buffered is discarded.
            backpressure_mode: What :meth:`publish` does when the queue is full.
            stop: Session shutdown event, so a blocked publish gives up.
            put_timeout: How long a blocked publish waits before rechecking
                ``stop``, in seconds.

        Raises:
            ValueError: ``max_pending`` is not positive.
        """
        if max_pending <= 0:
            raise ValueError(f"max_pending must be > 0, got {max_pending}.")
        self._buffer = queue.Queue(maxsize=max_pending)
        self._backpressure_mode = backpressure_mode
        self._stop = stop
        self._put_timeout = put_timeout

    def publish(
        self,
        generation: int,
        chunk: list[StepResult],
    ) -> None:
        """Add one completed model step to the presentation queue.

        Called on the model thread. ``BLOCK`` waits here when the
        queue is full, until there is room or the session stops;
        ``DROP_OLDEST`` evicts instead and returns.

        Args:
            generation: Reset generation the chunk was generated in. A chunk
                from an earlier one is discarded rather than presented.
            chunk: One :class:`StepResult` per model channel.

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
        pending = (generation, chunk)
        if self._backpressure_mode is BackpressureMode.DROP_OLDEST:
            self._publish_latest(pending)
            return
        while not self._stop.is_set():
            try:
                self._buffer.put(pending, timeout=self._put_timeout)
                return
            except queue.Full:
                continue

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

    def advance(self, generation: int) -> tuple[bool, list[StepResult] | None]:
        """Move to the next model frame, if one is available.

        Called on the UI thread, once per tick. A ``generation`` other than the
        last one seen drops what is being presented, so nothing generated before
        a reset survives it.

        Args:
            generation: Current reset generation, from the event buffer.

        Returns:
            Whether the frame changed, and the chunk newly taken off the queue,
            which is ``None`` when the change was to another frame of the chunk
            already being presented.
        """
        if generation != self._generation:
            self._generation = generation
            self._presented_chunk = None
            self._frame_index = -1
            self._presented_frame_count = 0

        if (
            self._presented_chunk is not None
            and self._frame_index + 1 < self._presented_chunk[0].frame_count
        ):
            self._frame_index += 1
            self._presented_frame_count += 1
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
        return self._buffer.qsize()

    @property
    def buffer_capacity(self) -> int:
        """Return the maximum number of chunks that may wait to be presented."""
        return self._buffer.maxsize

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
            ValueError: The presentation stream and output use different CUDA
                devices.
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
        result.wait_for_output(self._presentation_stream)
        return _frame_at(result, self._frame_index)

    def presented_frames(self) -> tuple[Tensor, ...]:
        """Return all current frames ordered before the presentation stream."""
        if self._presented_chunk is None:
            return ()
        for result in self._presented_chunk:
            result.wait_for_output(self._presentation_stream)
        return tuple(
            _frame_at(result, self._frame_index) for result in self._presented_chunk
        )

    def composite(self, bottom: Tensor | None, top: Tensor) -> Tensor:
        """Draw ``top`` over ``bottom``.

        Args:
            bottom: ``[C, H, W]`` frame to draw onto, or ``None`` to start from
                black.
            top: ``[C, H, W]`` frame to draw. Four channels is RGBA and blends;
                anything else replaces.

        Returns:
            An RGB ``[3, H, W]`` frame.

        Raises:
            ValueError: The frames disagree about size, device or dtype, do not
                use the presentation-stream device, or are not presentable.
        """
        stream = self._presentation_stream
        if stream is None:
            return _composite_frame(bottom, top)

        device = resolve_cuda_device(stream.device)
        frames = (top,) if bottom is None else (bottom, top)
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
            output = _composite_frame(bottom, top)
        if caller_stream != stream:
            caller_stream.wait_stream(stream)
            output.record_stream(caller_stream)
        return output

    def has_pending_frames(self) -> bool:
        """Return whether another model frame is ready."""
        if (
            self._presented_chunk is not None
            and self._frame_index + 1 < self._presented_chunk[0].frame_count
        ):
            return True
        return not self._buffer.empty()

    def clear(self) -> None:
        """Discard buffered and currently presented model results."""
        self._presented_chunk = None
        self._frame_index = -1
        self._presented_frame_count = 0
        while True:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                return

    def _publish_latest(self, pending: tuple[int, list[StepResult]]) -> None:
        while not self._stop.is_set():
            try:
                self._buffer.put_nowait(pending)
                return
            except queue.Full:
                try:
                    self._buffer.get_nowait()
                    with self._counter_lock:
                        self._dropped_for_space += 1
                except queue.Empty:
                    continue

    def _take_buffered_chunk(
        self, generation: int, *, latest: bool
    ) -> list[StepResult] | None:
        selected: list[StepResult] | None = None
        while True:
            try:
                chunk_generation, chunk = self._buffer.get_nowait()
            except queue.Empty:
                return selected
            if chunk_generation != generation:
                with self._counter_lock:
                    self._discarded_at_reset += 1
                continue
            if selected is not None:
                with self._counter_lock:
                    self._dropped_for_space += 1
            selected = chunk
            if not latest:
                return selected


def _frame_at(result: StepResult, frame_index: int) -> Tensor:
    """Return one result frame as ``[C, H, W]``."""
    output = result.output
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
    return color * alpha + bottom * (1.0 - alpha)


__all__ = ["PresentationManager"]
