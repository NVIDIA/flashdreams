# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from flashdreams.infra.acceleration.overlap import (
    CudaStreamOverlap,
    HostThreadOverlap,
    SynchronousOverlap,
)

pytestmark = pytest.mark.ci_cpu


def test_synchronous_overlap_runs_work_before_return() -> None:
    calls: list[str] = []
    overlap = SynchronousOverlap(name="sync")

    overlap.submit(lambda: calls.append("work"))

    assert calls == ["work"]
    assert not overlap.pending
    assert overlap.wait()


def test_host_thread_overlap_runs_work_until_wait() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    overlap = HostThreadOverlap(name="host-test")

    def work() -> None:
        started.set()
        release.wait(timeout=1.0)
        calls.append("done")

    overlap.submit(work)

    assert started.wait(timeout=1.0)
    assert overlap.pending
    assert not overlap.wait(timeout_s=0.001)

    release.set()

    assert overlap.wait(timeout_s=1.0)
    assert calls == ["done"]
    assert not overlap.pending


def test_host_thread_overlap_rejects_second_pending_task() -> None:
    release = threading.Event()
    overlap = HostThreadOverlap(name="host-test")

    overlap.submit(lambda: release.wait(timeout=1.0))

    with pytest.raises(RuntimeError, match="pending"):
        overlap.submit(lambda: None)

    release.set()
    assert overlap.wait(timeout_s=1.0)


def test_host_thread_overlap_can_report_worker_error_on_wait() -> None:
    overlap = HostThreadOverlap(name="host-test", reraise_worker_errors=False)

    def work() -> None:
        raise ValueError("boom")

    overlap.submit(work)

    assert overlap.wait(timeout_s=1.0)
    assert isinstance(overlap.last_error, ValueError)
    with pytest.raises(ValueError, match="boom"):
        overlap.wait(raise_error=True)


def test_cuda_stream_overlap_enqueues_work_and_waits_on_event() -> None:
    fake_torch = _FakeTorch()
    calls: list[str] = []
    overlap = CudaStreamOverlap(torch_module=fake_torch, device="cuda:0")

    overlap.submit(lambda: calls.append("work"))

    assert calls == ["work"]
    assert fake_torch.cuda.stream_contexts == [overlap._stream]
    assert fake_torch.cuda.device_contexts == ["cuda:0"]
    assert overlap.pending

    event = overlap._event
    assert event is not None
    event.complete()

    assert overlap.wait(timeout_s=1.0)
    assert not overlap.pending
    assert event.synchronize_calls == 0

    overlap.submit(lambda: calls.append("again"))
    assert calls == ["work", "again"]
    event = overlap._event
    assert event is not None
    event.complete()
    overlap.close()
    assert event.synchronize_calls == 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def device(self, value: str) -> str:
        return value


class _FakeCuda:
    def __init__(self) -> None:
        self.device_contexts: list[str] = []
        self.stream_contexts: list[object] = []

    def is_available(self) -> bool:
        return True

    def Stream(self, *, device: str | None = None) -> object:
        return _FakeStream(device)

    def Event(self) -> "_FakeEvent":
        return _FakeEvent()

    @contextmanager
    def device(self, device: str):
        self.device_contexts.append(device)
        yield

    @contextmanager
    def stream(self, stream: object):
        self.stream_contexts.append(stream)
        yield


class _FakeStream:
    def __init__(self, device: str | None) -> None:
        self.device = device


class _FakeEvent:
    def __init__(self) -> None:
        self._complete = False
        self.recorded_stream: object | None = None
        self.synchronize_calls = 0

    def record(self, stream: object) -> None:
        self.recorded_stream = stream

    def query(self) -> bool:
        return self._complete

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self._complete = True

    def complete(self) -> None:
        self._complete = True
