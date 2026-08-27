# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Coin course: lane-aligned layout, FTheta projection, proximity pickup.

Ports the course/projection logic of
``integrations/omnidreams/scripts/composite_track_items.py`` from its fitted
pinhole camera onto the scene's exact
:class:`omnidreams_game_engine.camera.FThetaCameraModel` and the authoritative
per-frame ego pose (``PresentedFrame.rig_to_world`` / ``vehicle_state``).
Pure CPU/numpy; no GPU dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import VehicleState

from crazy_robotaxi.live_edit.config import LiveEditCoinsConfig
from crazy_robotaxi.navigation import NavigationLane

_GRID_CELL_M = 32.0
"""Spatial-hash cell edge for the coin course (see :class:`_CoinGrid`)."""


@dataclass(frozen=True)
class CoinSprite:
    """One coin projected into the camera image for compositing."""

    center_uv: tuple[float, float]
    """Coin-center pixel position in the model-resolution image."""

    height_px: float
    """On-screen coin diameter in pixels."""

    alpha: float
    """Composite opacity in ``[0, 1]`` (distance fade)."""

    distance_m: float
    """Horizontal camera-to-coin distance, used for far-to-near ordering."""

    spin_phase: float
    """Stable per-coin phase for the spin/squash animation."""

    sprite_key: str = "coin"
    """Compositor sprite-bank key (effect items carry their type here)."""

    spin: bool = True
    """Whether the spin/squash animation applies (items render static)."""


def build_coin_course(
    lanes: Sequence[NavigationLane],
    config: LiveEditCoinsConfig,
) -> npt.NDArray[np.float32]:
    """Lay out coin world positions along the driving-lane centerlines.

    Mirrors ``TaxiNavigationMap.sample_waypoints``' walk but emits lateral
    groups (rows of coins across the lane) at ``config.spacing_m`` intervals.

    Every directed car lane contributes, including lanes without a mapped
    roadside stopping edge (``allows_taxi_stops=False``): those are regular
    driving lanes, and restricting coins to curb-adjacent lanes left the
    course on road edges the ego never crosses within pickup radius.

    Args:
        lanes: Directed driving-lane centerlines.
        config: Coin layout parameters.

    Returns:
        Coin centers with shape ``[coins, 3]`` in world coordinates.

    Raises:
        ValueError: No lane yields a single coin.
    """
    coins: list[npt.NDArray[np.float32]] = []
    occupied_cells: set[tuple[int, int]] = set()
    for lane in lanes:
        points = np.asarray(lane.centerline_world, dtype=np.float32)
        if len(points) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = float(cumulative[-1])
        distance = config.spacing_m / 2.0
        while distance < total:
            index = int(np.searchsorted(cumulative, distance) - 1)
            index = max(0, min(index, len(points) - 2))
            span = float(cumulative[index + 1] - cumulative[index])
            fraction = 0.0 if span <= 0.0 else (distance - cumulative[index]) / span
            center = points[index] + fraction * (points[index + 1] - points[index])
            direction = points[index + 1, :2] - points[index, :2]
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-6:
                distance += config.spacing_m
                continue
            right = np.array(
                [direction[1] / norm, -direction[0] / norm], dtype=np.float32
            )
            cell = (round(center[0] * 0.5), round(center[1] * 0.5))
            if cell not in occupied_cells:
                occupied_cells.add(cell)
                for offset in config.group_offsets_m:
                    coin = center.copy()
                    coin[:2] += right * np.float32(offset)
                    coin[2] += np.float32(config.hover_height_m)
                    coins.append(coin)
            distance += config.spacing_m
    if not coins:
        raise ValueError("Coin course requires at least one drivable lane sample.")
    return np.stack(coins).astype(np.float32)


class _CoinGrid:
    """Static 2D spatial hash over coin XY positions.

    A course spans thousands of coins but only the ones within
    ``max_render_distance_m`` of the ego matter each frame; hashing the
    (immutable) coin layout into ``_GRID_CELL_M`` cells makes the per-frame
    candidate lookup O(nearby) instead of O(course). Query results are
    cached per (cell, radius) and only recomputed when the query point
    crosses a cell boundary, so the steady-state cost is one dict probe.
    """

    def __init__(
        self, coins_xy: npt.NDArray[np.float32], cell_m: float = _GRID_CELL_M
    ) -> None:
        self._cell_m = float(cell_m)
        cells = np.floor(coins_xy / self._cell_m).astype(np.int64)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        boundaries = np.flatnonzero(np.any(np.diff(sorted_cells, axis=0), axis=1)) + 1
        self._cells: dict[tuple[int, int], npt.NDArray[np.intp]] = {}
        for chunk in np.split(order, boundaries) if len(order) else ():
            key = (int(cells[chunk[0], 0]), int(cells[chunk[0], 1]))
            self._cells[key] = chunk
        self._window_cache: dict[tuple[int, int, float], npt.NDArray[np.intp]] = {}
        self._empty = np.empty(0, dtype=np.intp)

    def near(self, x: float, y: float, radius: float) -> npt.NDArray[np.intp]:
        """Indices of all coins possibly within ``radius`` of ``(x, y)``.

        Superset by construction (cell granularity); callers still apply the
        exact distance test. Cached per (query cell, radius): valid for any
        query point inside the cell because the reach adds one full cell.
        """
        cell_x = math.floor(x / self._cell_m)
        cell_y = math.floor(y / self._cell_m)
        key = (cell_x, cell_y, radius)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached
        reach = math.ceil(radius / self._cell_m) + 1
        chunks = [
            chunk
            for cx in range(cell_x - reach, cell_x + reach + 1)
            for cy in range(cell_y - reach, cell_y + reach + 1)
            if (chunk := self._cells.get((cx, cy))) is not None
        ]
        # Ascending index order keeps painter's-order ties deterministic and
        # identical to the pre-grid full scan.
        window = np.sort(np.concatenate(chunks)) if chunks else self._empty
        # Keep only the most recent windows; the ego revisits few cells.
        if len(self._window_cache) > 8:
            self._window_cache.clear()
        self._window_cache[key] = window
        return window


class CoinAbility:
    """Track coin collection and produce per-frame screen sprites."""

    def __init__(
        self,
        coins_world: npt.NDArray[np.float32],
        config: LiveEditCoinsConfig,
        *,
        sprite_keys: Sequence[str] | None = None,
        spin: bool = True,
    ) -> None:
        if coins_world.ndim != 2 or coins_world.shape[1] != 3:
            raise ValueError("coins_world must have shape [coins, 3]")
        if sprite_keys is not None and len(sprite_keys) != len(coins_world):
            raise ValueError("sprite_keys must match coins_world length")
        self._coins_world = coins_world.astype(np.float32)
        self._config = config
        self._sprite_keys = None if sprite_keys is None else tuple(sprite_keys)
        self._spin = spin
        self._collected = np.zeros(len(coins_world), dtype=bool)
        self._grid = _CoinGrid(self._coins_world[:, :2])
        self.enabled = True

    @classmethod
    def from_lanes(
        cls, lanes: Sequence[NavigationLane], config: LiveEditCoinsConfig
    ) -> CoinAbility:
        """Build the ability with a course laid out along ``lanes``."""
        return cls(build_coin_course(lanes, config), config)

    @property
    def collected_count(self) -> int:
        """Return the number of coins collected so far."""
        return int(self._collected.sum())

    @property
    def score(self) -> int:
        """Return the coin score contribution."""
        return self.collected_count * self._config.points_per_coin

    @property
    def remaining_count(self) -> int:
        """Return the number of uncollected coins."""
        return int((~self._collected).sum())

    def toggle(self) -> bool:
        """Flip rendering/collection on or off; return the new state."""
        self.enabled = not self.enabled
        return self.enabled

    def advance_frames(self, vehicle_states: Iterable[VehicleState]) -> int:
        """Collect coins within pickup radius of any pose; return new pickups."""
        return len(self.collect_near(vehicle_states))

    def collect_near(self, vehicle_states: Iterable[VehicleState]) -> tuple[int, ...]:
        """Collect coins within pickup radius; return their course indices.

        The indices let item-typed courses (:mod:`~.item_ability`) map each
        pickup back to its effect; plain coin callers only need the count.
        """
        if not self.enabled:
            return ()
        collected: list[int] = []
        radius = self._config.pickup_radius_m
        for state in vehicle_states:
            near = self._grid.near(state.x_m, state.y_m, radius)
            if len(near) == 0:
                continue
            candidates = near[~self._collected[near]]
            if len(candidates) == 0:
                continue
            deltas = self._coins_world[candidates, :2] - np.array(
                [state.x_m, state.y_m], dtype=np.float32
            )
            hits = np.linalg.norm(deltas, axis=1) <= radius
            if hits.any():
                self._collected[candidates[hits]] = True
                collected.extend(int(i) for i in candidates[hits])
        return tuple(collected)

    def visible_sprites(
        self,
        rig_to_world: npt.NDArray[np.float32],
        camera_model: FThetaCameraModel,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[CoinSprite, ...]:
        """Project uncollected coins near the camera into image pixels.

        The on-screen diameter comes from projecting each coin's vertical
        extent (center ± diameter/2), so fisheye distortion and camera pitch
        are handled exactly rather than via a pinhole ``fx/z`` approximation.

        Returns:
            Sprites sorted far-to-near, ready for painter's-algorithm
            compositing; at most ``config.max_visible_sprites`` (the
            nearest ones win when the course is dense).
        """
        if not self.enabled:
            return ()
        camera_xy = rig_to_world[:2, 3]
        window = self._grid.near(
            float(camera_xy[0]), float(camera_xy[1]), self._config.max_render_distance_m
        )
        if len(window) == 0:
            return ()
        remaining = window[~self._collected[window]]
        if len(remaining) == 0:
            return ()
        centers = self._coins_world[remaining]
        distances = np.linalg.norm(centers[:, :2] - camera_xy[None, :], axis=1)
        near = distances <= self._config.max_render_distance_m
        if not near.any():
            return ()
        centers = centers[near]
        distances = distances[near]
        indices = remaining[near]

        half = np.array(
            [0.0, 0.0, self._config.coin_diameter_m / 2.0], dtype=np.float32
        )
        points = np.concatenate((centers - half, centers + half), axis=0)
        uv, _depth, forward = camera_model.project_world(points, rig_to_world)
        count = len(centers)
        bottom_uv, top_uv = uv[:count], uv[count:]
        center_uv = (bottom_uv + top_uv) / 2.0
        heights_px = np.linalg.norm(top_uv - bottom_uv, axis=1)
        keep = (
            forward[:count]
            & forward[count:]
            & (center_uv[:, 0] >= 0.0)
            & (center_uv[:, 0] < image_width)
            & (center_uv[:, 1] >= 0.0)
            & (center_uv[:, 1] < image_height)
            & (heights_px >= 3.0)
        )
        if not keep.any():
            return ()
        kept = np.flatnonzero(keep)
        order = kept[np.argsort(-distances[kept], kind="stable")]
        cap = self._config.max_visible_sprites
        if cap > 0 and len(order) > cap:
            # Far-to-near order: dropping the head keeps the nearest coins.
            order = order[-cap:]
        alphas = self._fade_array(distances)
        keys = self._sprite_keys
        return tuple(
            CoinSprite(
                center_uv=(float(center_uv[i, 0]), float(center_uv[i, 1])),
                height_px=float(heights_px[i]),
                alpha=float(alphas[i]),
                distance_m=float(distances[i]),
                spin_phase=float(indices[i]) * 0.61,
                sprite_key="coin" if keys is None else keys[int(indices[i])],
                spin=self._spin,
            )
            for i in order
        )

    def _fade_array(
        self, distances_m: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Vectorized :meth:`_fade` over all candidate distances."""
        fade_start = self._config.fade_start_distance_m
        fade_end = self._config.max_render_distance_m
        if fade_end <= fade_start:
            return np.ones_like(distances_m)
        alphas = (fade_end - distances_m) / (fade_end - fade_start)
        return np.clip(alphas, 0.0, 1.0)

    def _fade(self, distance_m: float) -> float:
        fade_start = self._config.fade_start_distance_m
        fade_end = self._config.max_render_distance_m
        if distance_m <= fade_start:
            return 1.0
        if fade_end <= fade_start:
            return 1.0
        return max(0.0, (fade_end - distance_m) / (fade_end - fade_start))


def coin_squash(spin_phase: float, frame_index: int) -> float:
    """Return the horizontal squash of a spinning coin for one frame."""
    return max(0.3, abs(math.cos(frame_index * 2.0 * math.pi / 36.0 + spin_phase)))
