# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Neutral gameplay-controller contract for optional physical actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ludus_renderer import BodyState, SceneObject


@dataclass(frozen=True)
class ActorControlDecision:
    """Native actuator state returned by a gameplay actor controller."""

    drive_enabled: bool
    detached_from_track: bool


@dataclass(frozen=True)
class ActorTrackTarget:
    """Logical route target for one gameplay-owned physical actor."""

    object_id: str
    """Stable object identifier shared with the physics graph."""

    timestamp_us: int
    """Logical timestamp to sample from the actor's source track."""

    velocity_scale: float = 1.0
    """Track velocity multiplier within ``[0, 1]``."""


class PhysicsActorController(Protocol):
    """Gameplay owner for a set of optional PhysX-driven scene objects.

    Controllers own actor intent and lifecycle. Rendering and physics remain
    downstream consumers of that state; the physics world only combines the
    active objects and routes observations back to their unique owner.
    """

    @property
    def objects(self) -> tuple[SceneObject, ...]: ...

    @property
    def active_objects(self) -> tuple[SceneObject, ...]: ...

    @property
    def active_object_ids(self) -> frozenset[str]: ...

    @property
    def active_timestamps_us(self) -> dict[str, int]: ...

    @property
    def object_ids(self) -> frozenset[str]: ...

    @property
    def max_drive_speeds_mps(self) -> dict[str, float]: ...

    def prepare_topology(self, ego: BodyState) -> None: ...

    def prepare_step(
        self, ego: BodyState, dt_s: float
    ) -> tuple[ActorTrackTarget, ...]: ...

    def observe_physics(
        self,
        object_id: str,
        *,
        struck: bool,
        body: BodyState,
        dt_s: float,
    ) -> ActorControlDecision | None: ...


__all__ = ["ActorControlDecision", "ActorTrackTarget", "PhysicsActorController"]
