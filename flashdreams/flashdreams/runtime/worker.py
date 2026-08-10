# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-affine execution for stateful inference runtimes."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, cast

import torch

_T = TypeVar("_T")
_EXECUTOR_FUTURE_POLL_INTERVAL_S = 0.01


class ModelExecutionWorker:
    """Run ordered runtime lifecycle calls on one owned OS thread.

    CUDA graphs, Triton launchers, and some backend contexts are thread-local.
    A runtime should therefore submit initialization, reset, generation, and
    close operations through one worker instead of using ``asyncio.to_thread``.

    Cancelling an awaiting task does not cancel the submitted operation. The
    operation remains ordered on the worker, and later calls run only after it
    completes.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        thread_name: str = "flashdreams-runtime",
    ) -> None:
        self._device = None if device is None else torch.device(device)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name,
            initializer=self._initialize_thread,
        )
        self._state_lock = threading.Lock()
        self._accepting = True
        self._closed = False
        self._thread_id: int | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def worker_thread_id(self) -> int | None:
        return self._thread_id

    @property
    def is_worker_thread(self) -> bool:
        return self._thread_id == threading.get_ident()

    async def call(
        self,
        func: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one callable after all previously submitted worker calls."""
        self._require_not_worker_thread()
        self._require_accepting()
        future = self._submit(func, args, kwargs)
        try:
            return await _await_executor_future(future)
        except asyncio.CancelledError:
            future.add_done_callback(_consume_exception)
            raise

    def call_blocking(
        self,
        func: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one callable from synchronous code on the owned worker thread."""
        self._require_not_worker_thread()
        self._require_accepting()
        future = self._executor.submit(_invoke, func, args, kwargs)
        return cast(_T, future.result())

    async def close(self) -> None:
        """Drain submitted work and stop accepting lifecycle calls."""
        self._require_not_worker_thread()
        if not self._begin_close():
            return
        try:
            barrier = asyncio.wrap_future(self._executor.submit(_noop))
            await _await_executor_future(barrier)
        finally:
            self._finish_close()

    def close_blocking(self) -> None:
        """Synchronous close for non-async setup and teardown paths."""
        self._require_not_worker_thread()
        if not self._begin_close():
            return
        try:
            barrier = self._executor.submit(_noop)
            barrier.result()
        finally:
            self._finish_close()

    def _submit(
        self,
        func: Callable[..., _T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> asyncio.Future[_T]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, _invoke, func, args, kwargs)

    def _initialize_thread(self) -> None:
        self._thread_id = threading.get_ident()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)

    def _require_accepting(self) -> None:
        if not self._accepting:
            raise RuntimeError("runtime worker is closed")

    def _require_not_worker_thread(self) -> None:
        if self.is_worker_thread:
            raise RuntimeError(
                "Cannot dispatch to the model execution worker from its own "
                "thread; call the function directly."
            )

    def _begin_close(self) -> bool:
        with self._state_lock:
            if self._closed:
                return False
            self._accepting = False
            return True

    def _finish_close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._state_lock:
            self._closed = True


ThreadAffineRuntimeWorker = ModelExecutionWorker


def _invoke(
    func: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _T:
    return func(*args, **kwargs)


def _noop() -> None:
    return


async def _await_executor_future(future: asyncio.Future[_T]) -> _T:
    """Await an executor future without relying on a single cross-thread wakeup."""
    while not future.done():
        await asyncio.wait(
            {future},
            timeout=_EXECUTOR_FUTURE_POLL_INTERVAL_S,
        )
    return future.result()


def _consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["ModelExecutionWorker", "ThreadAffineRuntimeWorker"]
