# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Effect-pickup items: sparse lane-course sprites that trigger abilities.

Extends the coin course concept: one item every ``spacing_m`` (much rarer
than coins), each with a sprite and an effect. Layout, FTheta projection,
spatial-hash culling, proximity pickup, and GPU compositing all reuse the
coin machinery (:class:`~.coin_ability.CoinAbility` with per-item sprite
keys); the effect dispatch (:class:`ItemEffects`) routes each pickup
through the existing ability state machines AT THE NEXT CHUNK BOUNDARY —
the same path the K/V key requests take, so the keys keep working alongside
pickups and both trigger paths share one set of state-machine rules:

- ``rain`` / ``snow`` items request that weather preset. Weather is
  base-world-only, so a pickup during an active skin is IGNORED with a HUD
  hint (chosen over queueing: a queued weather landing seconds later, with
  no visible cause, reads as a glitch; the ignore matches the V-key
  semantics exactly and keeps the state machine free of deferred intents).
  Re-picking the active weather refreshes its timed-weather timer.
- ``mystery`` boxes grant a random timed skin burst (seeded RNG knob picks
  which skin; ``mystery_burst_chunks`` overrides the global skin duration
  per activation, so the box grants a burst even in hold-forever mode). A
  burst during a key-held skin behaves like a K cycle: switch, fresh timer.
- ``nitro`` items are the exception to the boundary rule: the effect is a
  timed speed boost inside the app-authoritative physics tick
  (:class:`~.nitro_ability.NitroAbility`), physics-only with no world-model
  state to swap, so it activates immediately — the next sampled physics
  tick drives faster. It composes with every skin/weather/obstacle state
  and never touches the StyleAbility state machine. A second nitro while
  boosted resets the timer (no stacking).

Every pickup raises a short HUD flash ("RAIN!", "? PIXEL BURST!", ...)
drawn by the live-edit presenter next to the ability chips.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from loguru import logger
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import VehicleState

from crazy_robotaxi.live_edit.coin_ability import CoinAbility, CoinSprite
from crazy_robotaxi.live_edit.config import (
    ITEM_TYPES,
    LiveEditCoinsConfig,
    LiveEditItemsConfig,
)
from crazy_robotaxi.navigation import NavigationLane

_CANDIDATE_STEP_CAP_M = 25.0
"""Upper bound on the along-lane candidate step (matches the coin walk)."""


def build_item_course(
    lanes: Sequence[NavigationLane],
    config: LiveEditItemsConfig,
) -> tuple[npt.NDArray[np.float32], tuple[str, ...]]:
    """Lay out effect items with GLOBAL ``spacing_m`` sparsity over the map.

    Rarity is a property of the whole lane network, not of each polyline:
    real maps chop lanes into short segments (the shipped suburb map's
    segments are mostly shorter than the item spacing), so a per-lane walk
    at ``spacing_m`` intervals places almost nothing. Instead the walk
    samples candidates every ~``spacing_m / 4`` (capped at the coin step)
    and accepts one only when no already-accepted item WITH A SIMILAR
    HEADING lies within ``spacing_m`` (spacing-sized spatial hash, 3x3
    neighborhood check). The heading test keeps the two directions of a
    road independent — without it, whichever directed lane is walked first
    claims every spacing-disc along the whole road and drivers of the
    opposite lane never pass within pickup radius of an item. Deterministic
    for a given lane order; item types cycle through the configured
    ``item_types`` mix in acceptance order (equal rarity per kind).

    Returns:
        ``(centers [items, 3] world, types [items])``.

    Raises:
        ValueError: No lane yields a single item.
    """
    step = min(config.spacing_m / 4.0, _CANDIDATE_STEP_CAP_M)
    centers: list[npt.NDArray[np.float32]] = []
    headings: list[tuple[float, float]] = []
    cells: dict[tuple[int, int], list[int]] = {}
    cell_m = config.spacing_m
    for lane in lanes:
        points = np.asarray(lane.centerline_world, dtype=np.float32)
        if len(points) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = float(cumulative[-1])
        distance = step / 2.0
        while distance < total:
            index = int(np.searchsorted(cumulative, distance) - 1)
            index = max(0, min(index, len(points) - 2))
            span = float(cumulative[index + 1] - cumulative[index])
            fraction = 0.0 if span <= 0.0 else (distance - cumulative[index]) / span
            center = points[index] + fraction * (points[index + 1] - points[index])
            direction = points[index + 1, :2] - points[index, :2]
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-6:
                distance += step
                continue
            heading = (float(direction[0]) / norm, float(direction[1]) / norm)
            x, y = float(center[0]), float(center[1])
            cell_x, cell_y = math.floor(x / cell_m), math.floor(y / cell_m)
            too_close = any(
                math.hypot(x - float(centers[i][0]), y - float(centers[i][1]))
                < config.spacing_m
                and heading[0] * headings[i][0] + heading[1] * headings[i][1] > 0.5
                for nx in (cell_x - 1, cell_x, cell_x + 1)
                for ny in (cell_y - 1, cell_y, cell_y + 1)
                for i in cells.get((nx, ny), ())
            )
            if not too_close:
                item = center.copy()
                item[2] += np.float32(config.hover_height_m)
                cells.setdefault((cell_x, cell_y), []).append(len(centers))
                centers.append(item)
                headings.append(heading)
            distance += step
    if not centers:
        raise ValueError("Item course requires at least one drivable lane sample.")
    mix = config.item_types
    types = tuple(mix[i % len(mix)] for i in range(len(centers)))
    return np.stack(centers).astype(np.float32), types


class ItemAbility:
    """Track item pickups, produce screen sprites, hold the HUD flash.

    Wraps a :class:`~.coin_ability.CoinAbility` configured from the item
    knobs (per-item sprite keys, no spin) so projection/culling/pickup stay
    one implementation.
    """

    def __init__(
        self,
        items_world: npt.NDArray[np.float32],
        item_types: Sequence[str],
        config: LiveEditItemsConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(item_types) != len(items_world):
            raise ValueError("item_types must match items_world length")
        unknown = set(item_types) - set(ITEM_TYPES)
        if unknown:
            raise ValueError(f"unknown item types {sorted(unknown)}")
        self._config = config
        self._types = tuple(item_types)
        self._clock = clock
        self._flash: tuple[str, float] | None = None
        self._course = CoinAbility(
            items_world,
            _course_config(config),
            sprite_keys=self._types,
            spin=False,
        )

    @classmethod
    def from_lanes(
        cls,
        lanes: Sequence[NavigationLane],
        config: LiveEditItemsConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> ItemAbility:
        """Build the ability with a course laid out along ``lanes``."""
        centers, types = build_item_course(lanes, config)
        return cls(centers, types, config, clock=clock)

    @property
    def enabled(self) -> bool:
        """Whether items render and collect (rides the inner course flag)."""
        return self._course.enabled

    @property
    def remaining_count(self) -> int:
        """Return the number of uncollected items."""
        return self._course.remaining_count

    @property
    def collected_count(self) -> int:
        """Return the number of items picked up so far."""
        return self._course.collected_count

    def advance_frames(self, vehicle_states: Iterable[VehicleState]) -> tuple[str, ...]:
        """Collect items within pickup radius; return their types in order."""
        indices = self._course.collect_near(vehicle_states)
        return tuple(self._types[i] for i in indices)

    def visible_sprites(
        self,
        rig_to_world: npt.NDArray[np.float32],
        camera_model: FThetaCameraModel,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[CoinSprite, ...]:
        """Project uncollected items into image pixels (far-to-near)."""
        return self._course.visible_sprites(
            rig_to_world,
            camera_model,
            image_width=image_width,
            image_height=image_height,
        )

    def flash(self, label: str) -> None:
        """Raise the pickup HUD flash for ``config.flash_seconds``."""
        self._flash = (label, self._clock() + self._config.flash_seconds)

    @property
    def flash_label(self) -> str | None:
        """The active HUD flash label, or ``None`` once it has expired."""
        if self._flash is None:
            return None
        label, deadline = self._flash
        if self._clock() >= deadline:
            self._flash = None
            return None
        return label


def _course_config(config: LiveEditItemsConfig) -> LiveEditCoinsConfig:
    """Adapt the item knobs onto the coin-course machinery."""
    return LiveEditCoinsConfig(
        enabled=True,
        spacing_m=config.spacing_m,
        group_offsets_m=(0.0,),
        hover_height_m=config.hover_height_m,
        coin_diameter_m=config.item_diameter_m,
        pickup_radius_m=config.pickup_radius_m,
        points_per_coin=0,
        max_render_distance_m=config.max_render_distance_m,
        fade_start_distance_m=config.fade_start_distance_m,
        max_visible_sprites=16,
    )


class ItemEffects:
    """Route item pickups into the ability state machines.

    One instance per rollout (the mystery RNG re-seeds with the game), built
    by the runtime next to the abilities. ``apply`` never raises: an effect
    that cannot land (skin blocking weather, abilities not attached)
    degrades to a HUD hint so a pickup never crashes the frame loop.
    """

    def __init__(
        self,
        style_ability: object | None,
        config: LiveEditItemsConfig,
        *,
        nitro_ability: object | None = None,
    ) -> None:
        self._style = style_ability
        self._nitro = nitro_ability
        self._config = config
        self._rng = random.Random(config.mystery_seed)

    def apply(self, item_type: str) -> str:
        """Trigger one pickup's effect; return the HUD flash label."""
        if item_type in ("rain", "snow"):
            return self._apply_weather(item_type)
        if item_type == "mystery":
            return self._apply_mystery()
        if item_type == "nitro":
            return self._apply_nitro()
        logger.warning(f"[live-edit] unknown item pickup {item_type!r}")
        return f"{item_type.upper()}?"

    def _apply_nitro(self) -> str:
        # Physics-only: activates immediately (next sampled physics tick),
        # no chunk-boundary handshake and no StyleAbility coupling.
        activate = getattr(self._nitro, "activate", None)
        if activate is None:
            return "NITRO N/A"
        activate()
        return "NITRO!"

    def _apply_weather(self, name: str) -> str:
        style = self._style
        request = getattr(style, "request_weather", None)
        if request is None or not getattr(style, "weather_names", ()):
            return f"{name.upper()} N/A"
        if request(name):
            return f"{name.upper()}!"
        # Base-world-only rule: ignored (not queued) with a HUD hint, the
        # same rejection the V key gets while a skin is active.
        return f"{name.upper()} BLOCKED (SKIN ON)"

    def _apply_mystery(self) -> str:
        style = self._style
        names = getattr(style, "skin_names", ())
        request = getattr(style, "request_skin_burst", None)
        if request is None or not names:
            return "? NO SKINS"
        rolled = self._rng.choice(list(names))
        granted = request(rolled, self._config.mystery_burst_chunks)
        if granted is None:
            return "? NO SKINS"
        return f"? {granted.upper()} BURST!"
