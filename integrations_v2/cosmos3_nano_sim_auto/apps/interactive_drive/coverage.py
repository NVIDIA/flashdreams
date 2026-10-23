# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-map coverage warnings for freely driven camera poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ludus_renderer import TimestampedScene

OUTSIDE_MAP_WARNING = (
    "Outside the mapped area. Driving responses may be unreliable. "
    "Press R or select Restart rollout to return to the start."
)
"""Recovery guidance shown while the vehicle has left the prepared map."""


class MapCoverage:
    """Buffered static geometry with a two-block exit debounce.

    Coverage is within 30 metres of a map segment or inside a map triangle.
    Re-entry uses 25 metres to prevent flicker at the edge. Moving actors do
    not extend coverage. This estimates map availability, not road legality
    or whether a particular camera can see enough conditioning.
    """

    def __init__(self, scene: TimestampedScene) -> None:
        segments = [
            pool.vertices.numpy().reshape(-1, 2, 3)[..., :2]
            for pool in scene.polyline_pools
        ]
        triangles = [
            pool.vertices.numpy().reshape(-1, 3, 3)[..., :2]
            for pool in scene.polygon_pools
        ]
        self._triangles = (
            np.concatenate(triangles) if triangles else np.empty((0, 3, 2))
        )
        for a, b in ((0, 1), (1, 2), (2, 0)):
            segments.append(self._triangles[:, [a, b]])
        self._segments = np.concatenate(segments).astype(np.float32)
        self._minimum = self._segments.min(axis=1)
        self._maximum = self._segments.max(axis=1)
        self._triangle_minimum = self._triangles.min(axis=1)
        self._triangle_maximum = self._triangles.max(axis=1)
        self._outside_blocks = 0

    @property
    def warning(self) -> str | None:
        """Return recovery guidance after sustained loss of map coverage."""
        return OUTSIDE_MAP_WARNING if self._outside_blocks >= 2 else None

    def update(self, xy: ArrayLike) -> str | None:
        """Check the last presented conditioning pose and advance the warning."""
        if not len(self._segments):
            # An empty map cannot establish where its coverage should be.
            self._outside_blocks = 0
        elif self.contains(xy, margin=25.0 if self.warning else 30.0):
            self._outside_blocks = 0
        else:
            self._outside_blocks = min(2, self._outside_blocks + 1)
        return self.warning

    def contains(self, xy: ArrayLike, *, margin: float = 30.0) -> bool:
        """Check nearby primitives, preserving gaps between mapped regions."""
        xy = np.asarray(xy, dtype=np.float64)
        nearby = (
            (self._minimum[:, 0] <= xy[0] + margin)
            & (self._maximum[:, 0] >= xy[0] - margin)
            & (self._minimum[:, 1] <= xy[1] + margin)
            & (self._maximum[:, 1] >= xy[1] - margin)
        )
        segments = self._segments[nearby]
        start = segments[:, 0]
        delta = segments[:, 1] - start
        length2 = np.sum(delta * delta, axis=1)
        t = np.divide(
            np.sum((xy - start) * delta, axis=1),
            length2,
            out=np.zeros(len(start)),
            where=length2 > 0,
        ).clip(0, 1)
        if np.any(np.sum((xy - start - t[:, None] * delta) ** 2, axis=1) <= margin**2):
            return True
        nearby = np.all(self._triangle_minimum <= xy, axis=1) & np.all(
            self._triangle_maximum >= xy, axis=1
        )
        triangles = self._triangles[nearby]
        edges = np.roll(triangles, -1, axis=1) - triangles
        offset = xy - triangles
        cross = edges[..., 0] * offset[..., 1] - edges[..., 1] * offset[..., 0]
        area = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        return bool(
            np.any(
                (np.abs(area) > 1e-8)
                & (np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1))
            )
        )
