# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-to-host frame prefetch helpers for realtime presentation paths."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import numpy.typing as npt

_STREAMS_LOCK = threading.Lock()
_HOST_COPY_STREAMS: dict[int, Any] = {}


class CudaHostPrefetch:
    """Asynchronously stage one CUDA uint8 RGB frame into pinned host memory."""

    def __init__(self, tensor: Any, *, source_event: Any | None = None) -> None:
        self._tensor = tensor
        self._source_event = source_event
        self._host_tensor: Any | None = None
        self._done_event: Any | None = None
        self._started = False

    def start(self) -> bool:
        if self._started:
            return self._host_tensor is not None
        self._started = True

        try:
            import torch
        except ImportError:
            return False

        tensor = self._tensor
        if not torch.is_tensor(tensor) or not tensor.is_cuda:
            return False

        try:
            host_tensor = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            copy_stream = _host_copy_stream(torch, tensor.device)
            with torch.cuda.device(tensor.device):
                if self._source_event is not None:
                    copy_stream.wait_event(self._source_event)
                with torch.cuda.stream(copy_stream):
                    host_tensor.copy_(tensor, non_blocking=True)
                    tensor.record_stream(copy_stream)
                    done_event = torch.cuda.Event()
                    done_event.record(copy_stream)
        except Exception:
            self._host_tensor = None
            self._done_event = None
            return False

        self._host_tensor = host_tensor
        self._done_event = done_event
        return True

    def to_numpy(self) -> np.ndarray:
        host_tensor = self._host_tensor
        if host_tensor is None:
            raise RuntimeError("CUDA host prefetch was not started.")
        done_event = self._done_event
        if done_event is not None:
            done_event.synchronize()
        return np.ascontiguousarray(host_tensor.numpy(), dtype=np.uint8)


class LazyCudaFrame:
    """Expose one frame as CUDA first, with optional async host prefetch."""

    def __init__(
        self,
        frames_hwc_uint8: Any,
        frame_index: int,
        *,
        source_event: object | None = None,
        lost_source_message: str = "Lazy CUDA frame lost its source tensor before materialization.",
        already_materialized_message: str = "Lazy CUDA frame was already materialized on the host.",
        synchronize_source_event_on_host_copy: bool = False,
    ) -> None:
        self._frames_hwc_uint8: Any | None = frames_hwc_uint8
        self._frame_index = int(frame_index)
        self._source_event = source_event
        self._host: np.ndarray | None = None
        self._prefetch: CudaHostPrefetch | None = None
        self._lost_source_message = lost_source_message
        self._already_materialized_message = already_materialized_message
        self._synchronize_source_event_on_host_copy = (
            synchronize_source_event_on_host_copy
        )

    def prefetch_to_numpy(self) -> None:
        if (
            self._host is not None
            or self._prefetch is not None
            or self._frames_hwc_uint8 is None
        ):
            return
        frame = self._frames_hwc_uint8[self._frame_index].detach()
        prefetch = CudaHostPrefetch(frame, source_event=self._source_event)
        if prefetch.start():
            self._prefetch = prefetch

    def to_numpy(self) -> np.ndarray:
        if self._host is None:
            if self._prefetch is not None:
                prefetch = self._prefetch
                self._prefetch = None
                self._host = prefetch.to_numpy()
                self._frames_hwc_uint8 = None
                return self._host
            if self._frames_hwc_uint8 is None:
                raise RuntimeError(self._lost_source_message)
            if self._synchronize_source_event_on_host_copy:
                synchronize = getattr(self._source_event, "synchronize", None)
                if callable(synchronize):
                    synchronize()
            frame = self._frames_hwc_uint8[self._frame_index].detach().cpu().numpy()
            self._host = np.ascontiguousarray(frame, dtype=np.uint8)
            self._frames_hwc_uint8 = None
        return self._host

    def to_cuda_tensor(self) -> Any:
        if self._frames_hwc_uint8 is None:
            raise RuntimeError(self._already_materialized_message)
        return self._frames_hwc_uint8[self._frame_index]

    def to_cuda_event(self) -> object | None:
        if self._frames_hwc_uint8 is None:
            return None
        return self._source_event

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        array = self.to_numpy()
        if dtype is not None:
            target_dtype = np.dtype(dtype)
            if copy is False and target_dtype != array.dtype:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            array = array.astype(dtype, copy=False)
        if copy is True:
            return np.array(array, copy=True)
        return array


def prefetch_to_numpy(frame: object) -> None:
    """Start host materialization for frame-like objects that support it."""
    prefetch = getattr(frame, "prefetch_to_numpy", None)
    if callable(prefetch):
        prefetch()


def _host_copy_stream(torch: Any, device: Any) -> Any:
    key = _device_index(device)
    with _STREAMS_LOCK:
        stream = _HOST_COPY_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            _HOST_COPY_STREAMS[key] = stream
        return stream


def _device_index(device: Any) -> int:
    index = getattr(device, "index", None)
    return 0 if index is None else int(index)
