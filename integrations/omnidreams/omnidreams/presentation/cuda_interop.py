# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA/Vulkan shared-buffer interop for presenting CUDA-resident RGB frames."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from omnidreams.presentation.base import Rect
from omnidreams.presentation.canvas import fit_rect

_BUFFER_COUNT = 3
"""Shared RGBA buffers in the ring; enough to keep the copy stream ahead of
the swapchain without letting a stalled present pin unbounded memory."""


class CudaRGBFrame:
    """A CUDA RGB tensor plus the producer event gating its contents."""

    def __init__(self, *, tensor: Any, source_event: Any | None, ready: bool) -> None:
        self.tensor = tensor
        self.source_event = source_event
        self.ready = ready


class SharedRGBABuffer:
    """One CUDA/Vulkan shared RGBA buffer and its in-flight bookkeeping."""

    def __init__(
        self,
        *,
        buffer: Any,
        row_pitch: int,
        size_bytes: int,
        rgba_tensor: Any | None,
        copy_done_event: Any | None,
        pending_submit_id: int | None,
    ) -> None:
        self.buffer = buffer
        self.row_pitch = row_pitch
        self.size_bytes = size_bytes
        self.rgba_tensor = rgba_tensor
        self.copy_done_event = copy_done_event
        self.pending_submit_id = pending_submit_id


class NonBlockingCudaStream:
    """Dedicated CUDA stream for presenter copies, kept off the default stream."""

    def __init__(self, torch_module: Any, device: Any) -> None:
        self._stream = torch_module.cuda.Stream(device=device)
        self._stream_ptr = int(self._stream.cuda_stream)

    @property
    def stream(self) -> Any:
        return self._stream

    @property
    def cuda_stream(self) -> int:
        return self._stream_ptr

    def close(self) -> None:
        if self._stream is None:
            return

        self._stream.synchronize()
        self._stream_ptr = 0
        self._stream = None


class CudaRGBInterop:
    """Ring of CUDA/Vulkan shared RGBA buffers fed from CUDA RGB frames.

    Lets a presenter upload a model frame that is still on the GPU without a
    host roundtrip: frames are scaled and composited on a private CUDA
    stream, then handed to the swapchain as a Vulkan-visible buffer.
    """

    def __init__(self, *, spy: Any, device: Any, width: int, height: int) -> None:
        import torch

        self._spy = spy
        self._device = device
        self._torch = torch
        self._width = int(width)
        self._height = int(height)
        self._row_pitch = self._width * 4
        self._size_bytes = self._row_pitch * self._height
        self._buffers = [
            SharedRGBABuffer(
                buffer=device.create_buffer(
                    size=self._size_bytes,
                    usage=spy.BufferUsage.shared | spy.BufferUsage.copy_source,
                    label=f"display_cuda_rgba_buffer_{index}",
                ),
                row_pitch=self._row_pitch,
                size_bytes=self._size_bytes,
                rgba_tensor=None,
                copy_done_event=None,
                pending_submit_id=None,
            )
            for index in range(_BUFFER_COUNT)
        ]
        for shared_buffer in self._buffers:
            shared_buffer.rgba_tensor = shared_buffer.buffer.to_torch(
                type=spy.DataType.uint8,
                shape=[self._height, self._width, 4],
            )
        self._next_buffer_index = 0
        first_tensor = self._buffers[0].rgba_tensor
        if first_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        self._cuda_device = first_tensor.device
        self._copy_stream = NonBlockingCudaStream(self._torch, self._cuda_device)
        self._device_mismatch_logged = False

    def as_cuda_rgb_frame(self, rgb_frame: object) -> CudaRGBFrame | None:
        """Resolve ``rgb_frame`` and require it to match the buffer dimensions."""
        cuda_frame = self.as_cuda_rgb_source(rgb_frame)
        if cuda_frame is None:
            return None
        if tuple(cuda_frame.tensor.shape) != (self._height, self._width, 3):
            return None
        return cuda_frame

    def as_cuda_rgb_source(self, rgb_frame: object) -> CudaRGBFrame | None:
        """Resolve ``rgb_frame`` to a CUDA RGB tensor of any size.

        Returns:
            ``None`` when the frame cannot take the interop path at all --
            not resolvable, not a uint8 CUDA tensor, wrong rank, or resident
            on a different device than the shared buffers.
        """
        to_cuda_tensor = getattr(rgb_frame, "to_cuda_tensor", None)
        try:
            tensor = to_cuda_tensor() if callable(to_cuda_tensor) else rgb_frame
        except RuntimeError:
            return None
        if not self._torch.is_tensor(tensor):
            return None
        if not tensor.is_cuda or tensor.dtype != self._torch.uint8:
            return None
        if self._cuda_device_index(tensor.device) != self._cuda_device_index(
            self._cuda_device
        ):
            if not self._device_mismatch_logged:
                logger.info(
                    "[presenter] cuda_interop skipped: model RGB tensor is on "
                    f"{tensor.device}, presenter shared buffer is on {self._cuda_device}",
                )
                self._device_mismatch_logged = True
            return None
        if tensor.ndim != 3 or tensor.shape[-1] < 3:
            return None
        to_cuda_event = getattr(rgb_frame, "to_cuda_event", None)
        source_event = to_cuda_event() if callable(to_cuda_event) else None
        return CudaRGBFrame(
            tensor=tensor[..., :3].detach(),
            source_event=source_event,
            ready=cuda_event_ready(source_event),
        )

    def enqueue_rgb_to_shared_rgba(self, rgb_frame: CudaRGBFrame) -> bool:
        """Copy a full-size RGB frame into the next free shared buffer.

        Returns:
            ``False`` when every buffer is still in flight, so the caller
            should retry on a later tick.
        """
        shared_buffer = self._acquire_buffer()
        if shared_buffer is None:
            return False
        rgba_tensor = shared_buffer.rgba_tensor
        if rgba_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        rgb_tensor = rgb_frame.tensor
        if rgb_frame.source_event is not None:
            self._copy_stream.stream.wait_event(rgb_frame.source_event)
        with self._torch.cuda.stream(self._copy_stream.stream):
            if not rgb_tensor.is_contiguous():
                rgb_tensor = rgb_tensor.contiguous()
            rgba_tensor[..., :3].copy_(rgb_tensor, non_blocking=True)
            rgba_tensor[..., 3].fill_(255)
            rgb_tensor.record_stream(self._copy_stream.stream)
            rgba_tensor.record_stream(self._copy_stream.stream)
            copy_done_event = self._torch.cuda.Event()
            copy_done_event.record(self._copy_stream.stream)
        shared_buffer.copy_done_event = copy_done_event
        return True

    def enqueue_camera_to_shared_rgba(
        self,
        rgb_frame: CudaRGBFrame,
        *,
        overlay_rgba: np.ndarray,
        camera_area: Rect,
        background: tuple[int, int, int],
    ) -> bool:
        """Composite a camera frame and an RGBA overlay into a shared buffer.

        The frame is letterboxed into ``camera_area`` over a ``background``
        fill, then ``overlay_rgba`` is alpha-composited on top -- all on the
        private copy stream, so chrome rendered on the CPU still reaches the
        swapchain without a host roundtrip for the camera pixels.

        Args:
            overlay_rgba: Full-canvas ``[H, W, 4]`` chrome with the camera
                region left transparent.
            camera_area: Where the camera image belongs on the canvas.
            background: Letterbox fill behind the camera image.

        Returns:
            ``False`` when no buffer is free or the frame is degenerate.

        Raises:
            ValueError: ``overlay_rgba`` does not match the buffer dimensions.
        """
        shared_buffer = self._acquire_buffer()
        if shared_buffer is None:
            return False
        rgba_tensor = shared_buffer.rgba_tensor
        if rgba_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")

        overlay = np.ascontiguousarray(overlay_rgba, dtype=np.uint8)
        if tuple(overlay.shape) != (self._height, self._width, 4):
            raise ValueError(
                "HUD overlay shape does not match shared display buffer: "
                f"{tuple(overlay.shape)} vs {(self._height, self._width, 4)}"
            )

        rgb_tensor = rgb_frame.tensor
        target = fit_rect(
            source_size=(int(rgb_tensor.shape[1]), int(rgb_tensor.shape[0])),
            area=camera_area,
        )
        if target is None:
            return False
        target_x, target_y, target_right, target_bottom = target
        target_w = target_right - target_x
        target_h = target_bottom - target_y

        if rgb_frame.source_event is not None:
            self._copy_stream.stream.wait_event(rgb_frame.source_event)

        with self._torch.cuda.stream(self._copy_stream.stream):
            rgba_tensor[..., 0].fill_(int(background[0]))
            rgba_tensor[..., 1].fill_(int(background[1]))
            rgba_tensor[..., 2].fill_(int(background[2]))
            rgba_tensor[..., 3].fill_(255)

            if not rgb_tensor.is_contiguous():
                rgb_tensor = rgb_tensor.contiguous()
            resized = self._resize_rgb_tensor(rgb_tensor, target_h, target_w)
            rgba_tensor[
                target_y : target_y + target_h,
                target_x : target_x + target_w,
                :3,
            ].copy_(resized, non_blocking=True)

            overlay_tensor = self._torch.from_numpy(overlay).to(
                device=self._cuda_device,
                non_blocking=True,
            )
            self._alpha_composite_rgba(rgba_tensor, overlay_tensor)

            rgb_tensor.record_stream(self._copy_stream.stream)
            resized.record_stream(self._copy_stream.stream)
            overlay_tensor.record_stream(self._copy_stream.stream)
            rgba_tensor.record_stream(self._copy_stream.stream)
            copy_done_event = self._torch.cuda.Event()
            copy_done_event.record(self._copy_stream.stream)
        shared_buffer.copy_done_event = copy_done_event
        return True

    def ready_rgba_buffer(self) -> tuple[SharedRGBABuffer, Any] | None:
        """Return the next completed buffer along with its CUDA stream handle."""
        for offset in range(len(self._buffers)):
            index = (self._next_buffer_index + offset) % len(self._buffers)
            shared_buffer = self._buffers[index]
            copy_done_event = shared_buffer.copy_done_event
            if copy_done_event is None or not cuda_event_ready(copy_done_event):
                continue
            stream = int(self._copy_stream.cuda_stream)
            cuda_stream = self._spy.NativeHandle(
                self._spy.NativeHandleType.CUstream, stream
            )
            return shared_buffer, cuda_stream
        return None

    def mark_submitted(self, shared_buffer: SharedRGBABuffer, submit_id: int) -> None:
        """Hand ``shared_buffer`` to the GPU queue until ``submit_id`` retires."""
        shared_buffer.copy_done_event = None
        shared_buffer.pending_submit_id = int(submit_id)

    def close(self) -> None:
        """Synchronize and release the private copy stream."""
        self._copy_stream.close()

    def _acquire_buffer(self) -> SharedRGBABuffer | None:
        for offset in range(len(self._buffers)):
            index = (self._next_buffer_index + offset) % len(self._buffers)
            shared_buffer = self._buffers[index]
            if shared_buffer.copy_done_event is not None:
                continue
            if shared_buffer.pending_submit_id is None:
                self._next_buffer_index = (index + 1) % len(self._buffers)
                return shared_buffer
            if self._device.is_submit_finished(shared_buffer.pending_submit_id):
                shared_buffer.pending_submit_id = None
                self._next_buffer_index = (index + 1) % len(self._buffers)
                return shared_buffer
        return None

    def _cuda_device_index(self, device: Any) -> int:
        index = device.index
        return 0 if index is None else int(index)

    def _resize_rgb_tensor(self, rgb_tensor: Any, target_h: int, target_w: int) -> Any:
        if tuple(rgb_tensor.shape[:2]) == (target_h, target_w):
            return rgb_tensor if rgb_tensor.is_contiguous() else rgb_tensor.contiguous()
        nchw = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self._torch.float32)
        resized = self._torch.nn.functional.interpolate(
            nchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return (
            resized[0]
            .permute(1, 2, 0)
            .round()
            .clamp_(0, 255)
            .to(self._torch.uint8)
            .contiguous()
        )

    def _alpha_composite_rgba(self, base_rgba: Any, overlay_rgba: Any) -> None:
        alpha = overlay_rgba[..., 3:4].to(self._torch.float32) * (1.0 / 255.0)
        blended = (
            overlay_rgba[..., :3].to(self._torch.float32) * alpha
            + base_rgba[..., :3].to(self._torch.float32) * (1.0 - alpha)
        ).round()
        base_rgba[..., :3].copy_(blended.to(self._torch.uint8), non_blocking=True)
        base_rgba[..., 3].fill_(255)


def cuda_event_ready(event: Any | None) -> bool:
    """Report whether a CUDA event has completed, treating unknowns as ready."""
    if event is None:
        return True
    query = getattr(event, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except RuntimeError:
        return False


__all__ = [
    "CudaRGBFrame",
    "CudaRGBInterop",
    "NonBlockingCudaStream",
    "SharedRGBABuffer",
    "cuda_event_ready",
]
