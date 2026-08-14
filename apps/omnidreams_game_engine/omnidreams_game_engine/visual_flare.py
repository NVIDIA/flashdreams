# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Collision-triggered full-screen visual feedback."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

COLLISION_VISUAL_FLARE_DURATION_S = 0.71


@dataclass
class CollisionVisualFlare:
    """Track a short, noticeable black fade-in/fade-out after an impact."""

    fade_in_s: float = 0.06
    hold_s: float = 0.10
    fade_out_s: float = 0.55
    peak_opacity: float = 0.72
    _started_at: float | None = None

    def trigger(self, now: float | None = None) -> None:
        """Start (or restart) the flare at ``now``."""
        self._started_at = time.monotonic() if now is None else float(now)

    def opacity(self, now: float | None = None) -> float:
        """Return the current black-overlay opacity in ``[0, peak_opacity]``."""
        if self._started_at is None:
            return 0.0
        elapsed = (time.monotonic() if now is None else float(now)) - self._started_at
        if elapsed < 0.0:
            return 0.0
        if elapsed < self.fade_in_s:
            return self.peak_opacity * elapsed / max(self.fade_in_s, 1.0e-9)
        elapsed -= self.fade_in_s
        if elapsed < self.hold_s:
            return self.peak_opacity
        elapsed -= self.hold_s
        if elapsed < self.fade_out_s:
            return self.peak_opacity * (1.0 - elapsed / max(self.fade_out_s, 1.0e-9))
        self._started_at = None
        return 0.0


@dataclass
class _QueuedVisualFlareEvent:
    chunk_index: int
    frame_index: int
    started_at: float | None = None


class VisualFlareEventQueue:
    """Activate collision flares when their generated frame is displayed."""

    def __init__(
        self, *, duration_s: float = COLLISION_VISUAL_FLARE_DURATION_S
    ) -> None:
        self._duration_s = float(duration_s)
        self._events: list[_QueuedVisualFlareEvent] = []

    def schedule(self, *, chunk_index: int, frame_index: int) -> None:
        self._events.append(
            _QueuedVisualFlareEvent(
                chunk_index=int(chunk_index),
                frame_index=int(frame_index),
            )
        )

    def update(
        self,
        trigger: Callable[[], None],
        *,
        displayed_position: tuple[int, int] | None = None,
        now: float | None = None,
    ) -> None:
        """Start due events and remove active events once their fade completes."""
        update_time = time.monotonic() if now is None else float(now)
        retained: list[_QueuedVisualFlareEvent] = []
        for event in self._events:
            started_at = event.started_at
            if started_at is None:
                target = (event.chunk_index, event.frame_index)
                if displayed_position is None or displayed_position < target:
                    retained.append(event)
                    continue
                trigger()
                started_at = update_time
                event.started_at = started_at
            if update_time - started_at < self._duration_s:
                retained.append(event)
        self._events = retained

    def __len__(self) -> int:
        return len(self._events)


def darken_rgb(rgb: np.ndarray, opacity: float) -> np.ndarray:
    """Return an RGB/RGBA copy darkened by black while preserving alpha."""
    source = np.asarray(rgb, dtype=np.uint8)
    opacity = min(1.0, max(0.0, float(opacity)))
    if opacity <= 0.0:
        return source
    darkened = source.copy()
    np.multiply(
        darkened[..., :3],
        1.0 - opacity,
        out=darkened[..., :3],
        casting="unsafe",
    )
    return darkened
