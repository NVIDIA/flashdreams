# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small overlap runners for deferring realtime post-work."""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


class SynchronousOverlap:
    """Run submitted work immediately while exposing the overlap-runner API."""

    def __init__(self, *, name: str = "sync-overlap") -> None:
        self.name = name
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return False

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def wait(
        self, *, timeout_s: float | None = None, raise_error: bool = False
    ) -> bool:
        del timeout_s
        if raise_error:
            self._raise_if_failed()
        return True

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        del name
        self._error = None
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
            raise

    def close(self, *, wait: bool = True) -> None:
        del wait

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


class HostThreadOverlap:
    """Run one overlapped task at a time on a daemon host thread."""

    def __init__(
        self,
        *,
        name: str = "host-overlap",
        daemon: bool = True,
        reraise_worker_errors: bool = True,
    ) -> None:
        self.name = name
        self._daemon = daemon
        self._reraise_worker_errors = reraise_worker_errors
        self._done = threading.Event()
        self._done.set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return not self._done.is_set()

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def wait(
        self, *, timeout_s: float | None = None, raise_error: bool = False
    ) -> bool:
        completed = self._done.wait(timeout=timeout_s)
        if completed and raise_error:
            self._raise_if_failed()
        return completed

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        with self._lock:
            if not self._done.is_set():
                raise RuntimeError(f"{self.name} already has pending overlap work.")
            self._error = None
            self._done.clear()
            thread = threading.Thread(
                target=self._run,
                args=(work,),
                name=name or self.name,
                daemon=self._daemon,
            )
            self._thread = thread
            thread.start()

    def close(self, *, wait: bool = True) -> None:
        thread = self._thread
        if wait and thread is not None:
            thread.join()

    def _run(self, work: Callable[[], Any]) -> None:
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
            if self._reraise_worker_errors:
                raise
        finally:
            self._done.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


class CudaStreamOverlap:
    """Run CUDA-enqueued work on a side stream and wait on its completion event."""

    def __init__(
        self,
        *,
        name: str = "cuda-stream-overlap",
        device: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.name = name
        self._torch = torch_module if torch_module is not None else _import_torch()
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA stream overlap requires torch.cuda availability.")
        self._device = self._torch.device(device) if device is not None else None
        self._stream = self._torch.cuda.Stream(device=self._device)
        self._event: Any | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._event is not None and not self._event.query()

    @property
    def last_error(self) -> BaseException | None:
        return self._error

    def wait(
        self, *, timeout_s: float | None = None, raise_error: bool = False
    ) -> bool:
        event = self._event
        if event is None:
            if raise_error:
                self._raise_if_failed()
            return True

        if timeout_s is None:
            event.synchronize()
            completed = True
        else:
            completed = _wait_for_cuda_event(event, timeout_s=timeout_s)

        if completed and raise_error:
            self._raise_if_failed()
        return completed

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        del name
        if not self.wait(timeout_s=0.0):
            raise RuntimeError(f"{self.name} already has pending overlap work.")
        self._error = None
        try:
            with _cuda_device_context(self._torch, self._device):
                with self._torch.cuda.stream(self._stream):
                    result = work()
                    del result
                    event = self._torch.cuda.Event()
                    event.record(self._stream)
        except BaseException as exc:
            self._error = exc
            raise
        self._event = event

    def close(self, *, wait: bool = True) -> None:
        if wait:
            self.wait()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _cuda_device_context(torch_module: Any, device: Any | None) -> Any:
    if device is None:
        return nullcontext()
    return torch_module.cuda.device(device)


def _wait_for_cuda_event(event: Any, *, timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for a CUDA event using a short host poll.

    PyTorch exposes blocking event synchronization and non-blocking ``query()``,
    but not a native timed CUDA-event wait. Use a 1 ms sleep between polls so
    callers can bound wait time without a tight CPU spin loop.
    """
    if timeout_s < 0.0:
        raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True
