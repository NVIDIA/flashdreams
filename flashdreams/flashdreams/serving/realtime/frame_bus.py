# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-slot latest-frame publishing for realtime transports."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition
from typing import Generic, TypeVar

FrameT = TypeVar("FrameT")


@dataclass(frozen=True, slots=True)
class PublishedFrame(Generic[FrameT]):
    """Frame payload plus its monotonically increasing publication count."""

    payload: FrameT
    count: int


class LatestFrameBus(Generic[FrameT]):
    """Thread-safe single-slot bus for latest-frame transports.

    The bus stores only the newest frame. Waiters can block until a frame
    newer than ``last_seen_count`` is published, or until ``close`` wakes them.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._latest: FrameT | None = None
        self._frame_count = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def frame_count(self) -> int:
        with self._condition:
            return self._frame_count

    def publish(self, frame: FrameT) -> int:
        """Publish ``frame`` and return its publication count."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed LatestFrameBus.")
            self._latest = frame
            self._frame_count += 1
            self._condition.notify_all()
            return self._frame_count

    def latest(self) -> PublishedFrame[FrameT] | None:
        """Return the latest frame without blocking, if one has been published."""
        with self._condition:
            if self._latest is None:
                return None
            return PublishedFrame(payload=self._latest, count=self._frame_count)

    def wait_for_frame(
        self,
        *,
        last_seen_count: int = 0,
        timeout_s: float | None = None,
    ) -> PublishedFrame[FrameT] | None:
        """Block until a newer frame is available or the bus closes.

        Returns ``None`` on timeout or when closed before a newer frame is
        published.
        """
        if last_seen_count < 0:
            raise ValueError("last_seen_count must be >= 0")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None or self._frame_count <= last_seen_count:
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return None
                self._condition.wait(timeout=remaining_s)
            return PublishedFrame(payload=self._latest, count=self._frame_count)

    def close(self) -> None:
        """Wake all waiters and reject future publications."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
