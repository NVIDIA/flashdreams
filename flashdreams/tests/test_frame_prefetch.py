# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashdreams.infra.acceleration.frame_prefetch import (
    CudaHostPrefetch,
    LazyCudaFrame,
    prefetch_to_numpy,
)

pytestmark = pytest.mark.ci_cpu


class _Event:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _Prefetchable:
    def __init__(self) -> None:
        self.calls = 0

    def prefetch_to_numpy(self) -> None:
        self.calls += 1


class _CudaTensor:
    is_cuda = True
    device = torch.device("cuda:1")

    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def record_stream(self, stream: object) -> None:
        self.calls.append(("record", stream))


class _CudaStream:
    def __init__(
        self,
        calls: list[tuple[str, object]],
        device: torch.device = torch.device("cuda:1"),
    ) -> None:
        self.calls = calls
        self.device = device

    def wait_event(self, event: object) -> None:
        self.calls.append(("wait", event))


def test_cuda_host_prefetch_returns_false_for_cpu_tensor() -> None:
    prefetch = CudaHostPrefetch(torch.zeros((1, 2, 3), dtype=torch.uint8))

    assert not prefetch.start()
    assert not prefetch.start()
    with pytest.raises(RuntimeError, match="not started"):
        prefetch.to_numpy()


def test_lazy_cuda_frame_materializes_cpu_tensor_on_fallback_path() -> None:
    frames = torch.arange(18, dtype=torch.uint8).reshape(2, 1, 3, 3)
    frame = LazyCudaFrame(frames, 1)

    frame.prefetch_to_numpy()
    host = frame.to_numpy()

    assert host.flags.c_contiguous
    np.testing.assert_array_equal(host, frames[1].numpy())
    assert frame.to_cuda_event() is None
    with pytest.raises(RuntimeError, match="already materialized"):
        frame.to_cuda_tensor()


def test_lazy_cuda_frame_synchronizes_source_event_on_configured_host_copy() -> None:
    frames = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
    event = _Event()
    frame = LazyCudaFrame(
        frames,
        0,
        source_event=event,
        synchronize_source_event_on_host_copy=True,
    )

    frame.to_numpy()

    assert event.synchronize_calls == 1


def test_lazy_cuda_frame_keeps_source_event_until_materialized() -> None:
    frames = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
    event = object()
    frame = LazyCudaFrame(frames, 0, source_event=event)

    assert frame.to_cuda_event() is event
    assert torch.equal(frame.to_cuda_tensor(), frames[0])

    frame.to_numpy()

    assert frame.to_cuda_event() is None


def test_lazy_cuda_frame_orders_cuda_tensor_on_consumer_stream() -> None:
    calls: list[tuple[str, object]] = []
    tensor = _CudaTensor(calls)
    stream = _CudaStream(calls)
    event = object()
    frame = LazyCudaFrame([tensor], 0, source_event=event)

    assert frame.to_cuda_tensor(consumer_stream=stream) is tensor
    assert calls == [("wait", event), ("record", stream)]


def test_lazy_cuda_frame_defaults_to_current_device_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    tensor = _CudaTensor(calls)
    stream = _CudaStream(calls)
    event = object()
    frame = LazyCudaFrame([tensor], 0, source_event=event)
    requested_devices: list[object] = []

    def current_stream(device: object) -> _CudaStream:
        requested_devices.append(device)
        return stream

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    assert frame.to_cuda_tensor() is tensor
    assert requested_devices == [tensor.device]
    assert calls == [("wait", event), ("record", stream)]


def test_lazy_cuda_frame_rejects_consumer_stream_on_another_device() -> None:
    calls: list[tuple[str, object]] = []
    tensor = _CudaTensor(calls)
    stream = _CudaStream(calls, torch.device("cuda:0"))
    frame = LazyCudaFrame([tensor], 0, source_event=object())

    with pytest.raises(RuntimeError, match="must use the same device"):
        frame.to_cuda_tensor(consumer_stream=stream)

    assert calls == []


def test_lazy_cuda_frame_array_protocol_supports_dtype_and_copy() -> None:
    frames = torch.ones((1, 1, 1, 3), dtype=torch.uint8)
    frame = LazyCudaFrame(frames, 0)

    array = np.array(frame, dtype=np.float32, copy=True)

    assert array.dtype == np.float32
    assert array.flags.owndata
    np.testing.assert_array_equal(array, np.ones((1, 1, 3), dtype=np.float32))


def test_lazy_cuda_frame_array_protocol_rejects_unavoidable_copy() -> None:
    frames = torch.ones((1, 1, 1, 3), dtype=torch.uint8)
    frame = LazyCudaFrame(frames, 0)

    with pytest.raises(ValueError, match="Unable to avoid copy"):
        np.asarray(frame, dtype=np.float32, copy=False)


def test_lazy_cuda_frame_recovers_from_failed_prefetch() -> None:
    frames = torch.arange(3, dtype=torch.uint8).reshape(1, 1, 1, 3)
    frame = LazyCudaFrame(frames, 0)
    failed_prefetch = _FailedPrefetch()
    frame._prefetch = failed_prefetch  # ty:ignore[invalid-assignment]

    with pytest.raises(RuntimeError, match="copy failed"):
        frame.to_numpy()

    assert failed_prefetch.calls == 1
    host = frame.to_numpy()

    np.testing.assert_array_equal(host, frames[0].numpy())


def test_prefetch_to_numpy_dispatches_only_when_supported() -> None:
    prefetchable = _Prefetchable()

    prefetch_to_numpy(prefetchable)
    prefetch_to_numpy(object())

    assert prefetchable.calls == 1


class _FailedPrefetch:
    def __init__(self) -> None:
        self.calls = 0

    def to_numpy(self) -> np.ndarray:
        self.calls += 1
        raise RuntimeError("copy failed")
