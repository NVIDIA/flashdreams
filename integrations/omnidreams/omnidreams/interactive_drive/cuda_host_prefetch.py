# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import threading
from typing import Any

import numpy as np

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
