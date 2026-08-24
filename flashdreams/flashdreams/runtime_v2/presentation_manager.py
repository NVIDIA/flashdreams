# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffer and present model frames."""

import queue
import threading

import torch
from torch import Tensor

from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class PresentationManager:
    """Buffer model output for a session's UI thread."""

    def __init__(self) -> None:
        self._buffer: queue.Queue[tuple[int, list[StepResult]]] = queue.Queue(maxsize=2)
        self._presentation_mode = PresentationMode.BLOCK
        self._stop = threading.Event()
        self._put_timeout = 1.0 / 30.0
        self._generation = 0
        self._presented_chunk: list[StepResult] | None = None
        self._frame_index = -1
        self.dropped_for_space = 0
        self.discarded_at_reset = 0

    def configure(
        self,
        *,
        max_pending: int,
        presentation_mode: PresentationMode,
        stop: threading.Event,
        put_timeout: float,
    ) -> None:
        """Set the queue size and presentation mode."""
        if max_pending <= 0:
            raise ValueError(f"max_pending must be > 0, got {max_pending}.")
        self._buffer = queue.Queue(maxsize=max_pending)
        self._presentation_mode = presentation_mode
        self._stop = stop
        self._put_timeout = put_timeout

    def publish(
        self,
        generation: int,
        chunk: list[StepResult],
    ) -> None:
        """Add one completed model step to the presentation queue.

        ``BLOCK`` and ``LOSSLESS`` wait here when the queue is full.
        """
        if not chunk:
            raise ValueError("A presented chunk must contain at least one channel.")
        if any(not isinstance(result, StepResult) for result in chunk):
            raise TypeError("Every model channel must be a StepResult.")
        frame_count = chunk[0].frame_count
        if frame_count <= 0 or any(item.frame_count != frame_count for item in chunk):
            raise ValueError("Every channel in a chunk must have the same frame_count.")
        pending = (generation, chunk)
        if self._presentation_mode is PresentationMode.DROP_OLDEST:
            self._publish_latest(pending)
            return
        while not self._stop.is_set():
            try:
                self._buffer.put(pending, timeout=self._put_timeout)
                return
            except queue.Full:
                continue

    def advance(self, generation: int) -> tuple[bool, list[StepResult] | None]:
        """Move to the next model frame, if one is available.

        Returns:
            Whether the frame changed, and the newly selected model step.
        """
        if generation != self._generation:
            self._generation = generation
            self._presented_chunk = None
            self._frame_index = -1

        if self._presentation_mode is PresentationMode.DROP_OLDEST:
            chunk = self._take_buffered_chunk(generation, latest=True)
            if chunk is not None:
                if (
                    self._presented_chunk is not None
                    and self._frame_index + 1 < self._presented_chunk[0].frame_count
                ):
                    self.dropped_for_space += 1
                self._presented_chunk = chunk
                self._frame_index = 0
                return True, chunk

        if (
            self._presented_chunk is not None
            and self._frame_index + 1 < self._presented_chunk[0].frame_count
        ):
            self._frame_index += 1
            return True, None

        chunk = self._take_buffered_chunk(generation, latest=False)
        if chunk is None:
            return False, None
        self._presented_chunk = chunk
        self._frame_index = 0
        return True, chunk

    def presented_frame(self, channel_index: int) -> Tensor | None:
        """Return the current ``[C, H, W]`` frame from one model channel."""
        if self._presented_chunk is None:
            return None
        try:
            result = self._presented_chunk[channel_index]
        except IndexError as error:
            raise IndexError(
                f"Presented chunk has {len(self._presented_chunk)} channels; "
                f"channel {channel_index} does not exist."
            ) from error
        return _frame_at(result, self._frame_index)

    def presented_frames(self) -> tuple[Tensor, ...]:
        """Return the current frames from bottom channel to top channel."""
        if self._presented_chunk is None:
            return ()
        return tuple(
            _frame_at(result, self._frame_index) for result in self._presented_chunk
        )

    def composite(self, bottom: Tensor | None, top: Tensor) -> Tensor:
        """Draw ``top`` over ``bottom``."""
        return _composite_frame(bottom, top)

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
                    self.dropped_for_space += 1
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
                self.discarded_at_reset += 1
                continue
            if selected is not None:
                self.dropped_for_space += 1
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
