# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal runtime host for the Phase 2 demo-session vertical slice."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession

_T = TypeVar("_T")


class RuntimeHost:
    """Thin synchronous host around an :class:`InferenceRuntime`.

    Phase 3 moves the thread-affine worker boundary here. Phase 2 keeps the
    dispatch direct so fake-model CPU tests can prove the session-driver shape
    without introducing worker behavior early.
    """

    def __init__(self, runtime: InferenceRuntime) -> None:
        self._runtime = runtime
        self._healthy = True

    @property
    def runtime(self) -> InferenceRuntime:
        """Return the hosted runtime."""
        return self._runtime

    @property
    def is_healthy(self) -> bool:
        """Return whether admission should continue accepting sessions."""
        return self._healthy

    def mark_unhealthy(self) -> None:
        """Latch the host as unhealthy."""
        self._healthy = False

    def call(self, func: Callable[..., _T], /, *args: object, **kwargs: object) -> _T:
        """Run one model-affine callable synchronously."""
        return func(*args, **kwargs)

    async def call_async(
        self,
        func: Callable[..., _T],
        /,
        *args: object,
        **kwargs: object,
    ) -> _T:
        """Async-compatible direct dispatch placeholder for Phase 3."""
        return self.call(func, *args, **kwargs)

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        """Start one inference session through the hosted runtime."""
        return self._runtime.start_session(inputs)

    def close(self) -> None:
        """Close the hosted runtime."""
        self._runtime.close()


__all__ = ["RuntimeHost"]
