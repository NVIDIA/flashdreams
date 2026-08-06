# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Presentation-ready frame envelope and lazy RGB source helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flashdreams.infra.acceleration.frame_prefetch import prefetch_to_numpy


@dataclass(frozen=True, kw_only=True, slots=True)
class DisplayFrame:
    """One presentation-ready frame handed to a :class:`PresenterBackend`.

    Deliberately smaller than an integration's internal frame type: a
    presenter only needs the image to show, the overlay text to burn on top,
    and an opaque payload it forwards to the overlay untouched. Choosing
    *which* of several rendered sources becomes :attr:`image` is integration
    policy and happens before the frame reaches a presenter.
    """

    image: Any = None
    """``[H, W, 3]`` uint8 RGB source, or ``None`` before the first frame.

    May be a lazy handle exposing ``to_cuda_tensor`` / ``to_cuda_event`` /
    ``to_numpy`` (see :func:`as_rgb_host_uint8`) so a CUDA-resident frame
    reaches the GPU composite path without a host roundtrip.
    """

    timestamp_us: int = 0
    """Source timestamp of the frame, for presenters that pace or log."""

    status_message: str | None = None
    """Text to draw over the image; ``None`` presents the image alone."""

    allow_window_resize: bool = True
    """Whether this image's native resolution may drive a window resize.

    Set ``False`` for images already rendered at the window's own resolution,
    where growing the window to "fit" them would be circular. Only consulted
    when the presenter is configured to auto-resize at all.
    """

    overlay_data: Mapping[str, Any] = field(default_factory=dict)
    """Extra per-frame values forwarded verbatim to the overlay.

    Opaque to presenters. Integrations use it for chrome inputs that never
    become the presented image, such as OmniDreams' BEV minimap.
    """


def has_cuda_tensor(frame: object) -> bool:
    """Report whether ``frame`` can hand out a CUDA tensor without a host copy."""
    return callable(getattr(frame, "to_cuda_tensor", None))


def prefetch_frame(frame: object) -> None:
    """Start the asynchronous device-to-host copy for a lazy frame."""
    prefetch_to_numpy(frame)


def rgb_source_size(frame: object) -> tuple[int, int] | None:
    """Return an HWC RGB frame's ``(width, height)`` without a host copy.

    Returns:
        ``None`` when the frame is not yet resolvable or is not 3D, so
        callers can skip size-dependent work for this tick.
    """
    source = frame
    to_cuda_tensor = getattr(frame, "to_cuda_tensor", None)
    if callable(to_cuda_tensor):
        try:
            source = to_cuda_tensor()
        except RuntimeError:
            return None
    shape = getattr(source, "shape", None)
    if shape is None or len(shape) != 3:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def as_rgb_host_uint8(frame: object) -> np.ndarray:
    """Materialize ``frame`` as a contiguous ``[H, W, 3]`` uint8 host array."""
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])


__all__ = [
    "DisplayFrame",
    "as_rgb_host_uint8",
    "has_cuda_tensor",
    "prefetch_frame",
    "rgb_source_size",
]
