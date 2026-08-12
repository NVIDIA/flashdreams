# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading

import pytest

from flashdreams.runtime import ModelExecutionWorker, ThreadAffineRuntimeWorker

pytestmark = pytest.mark.ci_cpu


def test_model_execution_worker_keeps_legacy_worker_alias() -> None:
    assert ThreadAffineRuntimeWorker is ModelExecutionWorker


@pytest.mark.asyncio
async def test_worker_preserves_order_and_thread_affinity() -> None:
    worker = ThreadAffineRuntimeWorker(thread_name="test-runtime")
    calls: list[tuple[int, int]] = []

    def _record(value: int) -> int:
        calls.append((value, threading.get_ident()))
        return value * 2

    results = await asyncio.gather(*[worker.call(_record, value) for value in range(4)])
    await worker.close()

    assert results == [0, 2, 4, 6]
    assert [value for value, _thread_id in calls] == [0, 1, 2, 3]
    assert len({thread_id for _value, thread_id in calls}) == 1


@pytest.mark.asyncio
async def test_worker_propagates_exceptions_and_remains_usable() -> None:
    worker = ThreadAffineRuntimeWorker()

    def _raise() -> None:
        raise ValueError("bad runtime call")

    with pytest.raises(ValueError, match="bad runtime call"):
        await worker.call(_raise)

    assert await worker.call(lambda: 7) == 7
    await worker.close()


@pytest.mark.asyncio
async def test_cancelled_await_does_not_abandon_ordered_runtime_work() -> None:
    worker = ThreadAffineRuntimeWorker()
    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def _blocking_call() -> None:
        started.set()
        assert release.wait(timeout=2.0)
        completed.append("first")

    first = asyncio.create_task(worker.call(_blocking_call))
    assert await asyncio.to_thread(started.wait, 2.0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(worker.call(completed.append, "second"))
    release.set()
    await second
    await worker.close()

    assert completed == ["first", "second"]


@pytest.mark.asyncio
async def test_close_drains_work_and_rejects_new_calls() -> None:
    worker = ThreadAffineRuntimeWorker()
    assert await worker.call(lambda: "done") == "done"

    await worker.close()
    await worker.close()

    assert worker.closed
    with pytest.raises(RuntimeError, match="closed"):
        await worker.call(lambda: None)


@pytest.mark.asyncio
async def test_worker_sets_cuda_device_when_thread_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    monkeypatch.setattr("torch.cuda.set_device", seen.append)
    worker = ThreadAffineRuntimeWorker(device="cuda:3")

    await worker.call(lambda: None)
    await worker.close()

    assert [str(device) for device in seen] == ["cuda:3"]


def test_blocking_worker_call_is_not_reentrant() -> None:
    worker = ModelExecutionWorker()

    def _nested_dispatch() -> None:
        worker.call_blocking(lambda: None)

    try:
        with pytest.raises(RuntimeError, match="own thread"):
            worker.call_blocking(_nested_dispatch)
    finally:
        worker.close_blocking()


def test_async_worker_call_is_not_reentrant_from_worker_thread() -> None:
    worker = ModelExecutionWorker()

    def _nested_async_dispatch() -> None:
        async def _dispatch() -> None:
            await worker.call(lambda: None)

        asyncio.run(_dispatch())

    try:
        with pytest.raises(RuntimeError, match="own thread"):
            worker.call_blocking(_nested_async_dispatch)
    finally:
        worker.close_blocking()
