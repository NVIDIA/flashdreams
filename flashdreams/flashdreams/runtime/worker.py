# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-affine execution for stateful inference runtimes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import torch

_T = TypeVar("_T")


class ThreadAffineRuntimeWorker:
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
        self._accepting = True
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def call(
        self,
        func: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one callable after all previously submitted worker calls."""
        if not self._accepting:
            raise RuntimeError("runtime worker is closed")
        future = self._submit(func, args, kwargs)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(_consume_exception)
            raise

    async def close(self) -> None:
        """Drain submitted work and stop accepting lifecycle calls."""
        async with self._close_lock:
            if self._closed:
                return
            self._accepting = False
            barrier = self._submit(_noop, (), {})
            await asyncio.shield(barrier)
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def _submit(
        self,
        func: Callable[..., _T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> asyncio.Future[_T]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, _invoke, func, args, kwargs)

    def _initialize_thread(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)


def _invoke(
    func: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _T:
    return func(*args, **kwargs)


def _noop() -> None:
    return


def _consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["ThreadAffineRuntimeWorker"]
