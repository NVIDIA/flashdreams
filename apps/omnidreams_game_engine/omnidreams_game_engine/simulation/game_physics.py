# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive-drive adapter for the Ludus PhysX-first object graph."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger
from ludus_renderer import (
    BodyState,
    InvisibleBarrier,
    PhysicsObjectGraph,
    RigidBodyModel,
    SceneObject,
)

from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.game_map.vicinity import (
    GameMapVicinity,
    GameMapVicinityResolver,
)
from omnidreams_game_engine.simulation.actor_controller import (
    PhysicsActorController,
)
from omnidreams_game_engine.simulation.components import (
    BoxColliderComponent,
    GameEntity,
    RigidBodyComponent,
    TransformComponent,
    rigid_body_model_from_vehicle_config,
    suspension_for_object,
    vehicle_dynamics_for_object,
)
from omnidreams_game_engine.simulation.gameplay_physx import GameplayPhysXWorld
from omnidreams_game_engine.simulation.map_traffic import MapTrafficController
from omnidreams_game_engine.types import (
    DynamicActorTrajectory,
    PhysicsDebugFrame,
    SceneDefinition,
    VehicleState,
)

_BARRIER_LAYER_TOKENS = ("road_bound", "building", "house", "wall", "curb")
_PHYSX_RECENTER_DISTANCE_M = 32.0
_PHYSX_TOPOLOGY_REFRESH_INTERVAL_US = 2_000_000
"""Maximum time moving tracks can remain outside a stationary PhysX window."""

_PHYSX_BARRIER_SPACING_M = 2.0
_BARRIER_CONTACT_SLOP_M = 0.05
"""Extra proximity accepted when reinforcing a resolved barrier contact."""

_PHYSX_DEBUG_FORWARD_M = 125.0
_PHYSX_DEBUG_REAR_M = 15.0
_PHYSX_DEBUG_LATERAL_M = 100.0
_VISUAL_FLARE_MIN_SPEED_DELTA_MPS = 5.0 * 0.44704
_VISUAL_FLARE_COLLISION_WINDOW_US = 500_000
_NON_EGO_MAX_DRIVE_SPEED_MPS = 15.0 * 0.44704
_PHYSX_SIMULATION_RADIUS_M = 96.0
"""Collision horizon around the last recenter point.

The 32 m recenter threshold leaves at least 64 m of active topology around the
ego. Actors outside the horizon keep their recorded renderer trajectories until
they enter the PhysX window.
"""


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_state_from_vehicle(
    state: VehicleState, chassis_half_height_m: float
) -> BodyState:
    """Map the ground-anchored rig state to PhysX's chassis-center pose."""
    half_yaw = state.yaw_rad * 0.5
    return BodyState(
        position_m=np.asarray(
            [state.x_m, state.y_m, state.z_m + chassis_half_height_m],
            dtype=np.float32,
        ),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float32
        ),
        linear_velocity_mps=np.asarray(
            [
                state.velocity_x_mps
                if state.velocity_x_mps is not None
                else math.cos(state.yaw_rad) * state.speed_mps,
                state.velocity_y_mps
                if state.velocity_y_mps is not None
                else math.sin(state.yaw_rad) * state.speed_mps,
                0.0,
            ],
            dtype=np.float32,
        ),
        angular_velocity_radps=np.asarray(
            [0.0, 0.0, state.yaw_rate_radps], dtype=np.float32
        ),
    )


def _is_visual_flare_impact(
    collision_occurred: bool,
    before_velocity_mps: np.ndarray,
    after_velocity_mps: np.ndarray,
    driving_direction_xy: np.ndarray,
    impact_normal_xy: np.ndarray | None = None,
) -> bool:
    """Return whether a collision changed driving-axis speed by at least 5 mph."""
    speed_delta_mps = abs(
        abs(float(np.dot(after_velocity_mps[:2], driving_direction_xy)))
        - abs(float(np.dot(before_velocity_mps[:2], driving_direction_xy)))
    )
    if impact_normal_xy is not None:
        speed_delta_mps = max(
            speed_delta_mps,
            abs(
                float(
                    np.dot(
                        after_velocity_mps[:2] - before_velocity_mps[:2],
                        impact_normal_xy,
                    )
                )
            ),
        )
    meets_threshold = (
        speed_delta_mps >= _VISUAL_FLARE_MIN_SPEED_DELTA_MPS
        or math.isclose(
            speed_delta_mps,
            _VISUAL_FLARE_MIN_SPEED_DELTA_MPS,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    return collision_occurred and meets_threshold


def _ego_model(vehicle: VehicleConfig) -> RigidBodyModel:
    return rigid_body_model_from_vehicle_config(vehicle)


def _simplify_barrier_segments(segments_world: np.ndarray) -> tuple[np.ndarray, ...]:
    """Coalesce dense ordered map strokes into game-scale wall segments."""
    segments = np.asarray(segments_world, dtype=np.float32)
    if len(segments) == 0:
        return ()
    simplified: list[np.ndarray] = []
    start = segments[0, 0, :2].copy()
    end = segments[0, 1, :2].copy()
    for segment in segments[1:]:
        next_start = segment[0, :2]
        next_end = segment[1, :2]
        if float(np.linalg.norm(next_start - end)) > 0.25:
            if float(np.linalg.norm(end - start)) > 1e-4:
                simplified.append(np.stack([start, end]))
            start = next_start.copy()
        end = next_end.copy()
        if float(np.linalg.norm(end - start)) >= _PHYSX_BARRIER_SPACING_M:
            simplified.append(np.stack([start, end]))
            start = end.copy()
    if float(np.linalg.norm(end - start)) > 1e-4:
        simplified.append(np.stack([start, end]))
    return tuple(simplified)


@dataclass(frozen=True, slots=True)
class _BarrierReboundIndex:
    """Precomputed active-barrier geometry for vectorized contact detection."""

    starts_xy_m: np.ndarray
    """Start point for each active barrier segment."""

    vectors_xy_m: np.ndarray
    """End-minus-start vector for each active barrier segment."""

    length_squared_m2: np.ndarray
    """Squared segment lengths aligned with ``starts_xy_m``."""

    half_thickness_m: np.ndarray
    """Half collision thickness for each active barrier segment."""

    @classmethod
    def from_arrays(
        cls,
        segments_xy_m: np.ndarray,
        thicknesses_m: np.ndarray,
    ) -> _BarrierReboundIndex:
        """Build an index from active barrier arrays.

        Args:
            segments_xy_m: Barrier endpoints with shape ``[N, 2, 2]``.
            thicknesses_m: Barrier thicknesses with shape ``[N]``.

        Returns:
            Contiguous arrays reused for every frame until the physics window
            changes.
        """
        segments = np.ascontiguousarray(segments_xy_m, dtype=np.float32)
        thicknesses = np.ascontiguousarray(thicknesses_m, dtype=np.float32)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments_xy_m must have shape [N, 2, 2]")
        if thicknesses.shape != (len(segments),):
            raise ValueError("thicknesses_m must have shape [N]")
        starts = np.ascontiguousarray(segments[:, 0])
        vectors = np.ascontiguousarray(segments[:, 1] - starts)
        length_squared = np.einsum("ni,ni->n", vectors, vectors)
        return cls(
            starts_xy_m=starts,
            vectors_xy_m=vectors,
            length_squared_m2=np.ascontiguousarray(length_squared),
            half_thickness_m=np.ascontiguousarray(thicknesses * 0.5),
        )


@dataclass(frozen=True, slots=True)
class PhysicsBridgeTimings:
    """Measured Python adapter components surrounding one native PhysX step."""

    traffic_prepare_ms: float = 0.0
    """Tracked-traffic preparation before native simulation."""

    barrier_rebound_ms: float = 0.0
    """Static-barrier contact detection and velocity reinforcement."""

    traffic_update_ms: float = 0.0
    """Tracked-traffic observation and control publication."""

    state_materialize_ms: float = 0.0
    """Ego and actor state publication into engine-owned objects."""

    other_ms: float = 0.0
    """Remaining adapter work outside the named components."""


def _reinforce_static_barrier_rebound(
    barriers: _BarrierReboundIndex,
    requested_ego: BodyState,
    resolved_velocity_mps: np.ndarray,
    ego_model: RigidBodyModel,
    restitution: float,
) -> tuple[np.ndarray, bool]:
    """Raise outward barrier velocity after vectorized contact detection."""
    position = np.asarray(requested_ego.position_m[:2], dtype=np.float32)
    incoming_velocity = np.asarray(requested_ego.linear_velocity_mps, dtype=np.float32)
    reinforced = np.asarray(resolved_velocity_mps, dtype=np.float32).copy()
    if len(barriers.starts_xy_m) == 0:
        return reinforced, False

    yaw = _yaw_from_quaternion_xyzw(requested_ego.orientation_xyzw)
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    half_extents = ego_model.half_extents_m

    valid_segments = barriers.length_squared_m2 > 1.0e-8
    safe_length_squared = np.where(valid_segments, barriers.length_squared_m2, 1.0)
    relative_position = position[None, :] - barriers.starts_xy_m
    alpha = np.clip(
        np.einsum("ni,ni->n", relative_position, barriers.vectors_xy_m)
        / safe_length_squared,
        0.0,
        1.0,
    )
    offsets = position[None, :] - (
        barriers.starts_xy_m + barriers.vectors_xy_m * alpha[:, None]
    )
    distances = np.linalg.norm(offsets, axis=1)
    normals = np.empty_like(offsets)
    separated = distances > 1.0e-6
    normals[separated] = offsets[separated] / distances[separated, None]
    if bool(np.any(~separated)):
        speed = float(np.linalg.norm(incoming_velocity[:2]))
        fallback_normal = (
            -incoming_velocity[:2] / speed
            if speed > 1.0e-6
            else np.asarray([1.0, 0.0], dtype=np.float32)
        )
        normals[~separated] = fallback_normal

    supports = (
        np.abs(normals @ forward) * half_extents[0]
        + np.abs(normals @ left) * half_extents[1]
    )
    incoming_normal_speeds = normals @ incoming_velocity[:2]
    contact_indices = np.flatnonzero(
        valid_segments
        & (distances <= supports + barriers.half_thickness_m + _BARRIER_CONTACT_SLOP_M)
        & (incoming_normal_speeds < 0.0)
    )

    # Apply the small candidate set in authored order so corner contacts retain
    # the scalar path's response semantics.
    for index in contact_indices:
        normal = normals[index]
        incoming_normal_speed = float(incoming_normal_speeds[index])
        target_outward_speed = -restitution * incoming_normal_speed
        resolved_normal_speed = float(np.dot(reinforced[:2], normal))
        if resolved_normal_speed < target_outward_speed:
            reinforced[:2] += normal * (target_outward_speed - resolved_normal_speed)
    return reinforced, len(contact_indices) > 0


class GamePhysicsWorld:
    """Adapt a scene bundle to Ludus and delegate all simulation to PhysX."""

    def __init__(
        self,
        scene: SceneDefinition,
        vehicle: VehicleConfig,
        *,
        model_adapter: Callable[[RigidBodyModel], RigidBodyModel] | None = None,
        static_barrier_segments_world: np.ndarray | None = None,
        static_barrier_restitution: float | None = None,
        actor_controllers: tuple[PhysicsActorController, ...] = (),
    ) -> None:
        started_at = time.perf_counter()
        self._vehicle = vehicle
        if static_barrier_restitution is not None and (
            not math.isfinite(static_barrier_restitution)
            or not 0.0 <= static_barrier_restitution <= 1.0
        ):
            raise ValueError("static_barrier_restitution must be within [0, 1]")
        self._static_barrier_restitution = static_barrier_restitution
        adapt_model = model_adapter or (lambda model: model)

        game_map = getattr(scene, "game_map", None)
        self._map_traffic = MapTrafficController(
            () if game_map is None else game_map.traffic,
            vehicle,
        )
        self._actor_controllers: tuple[PhysicsActorController, ...] = (
            self._map_traffic,
            *actor_controllers,
        )
        if (
            vehicle.static_collision_enabled
            and static_barrier_segments_world is not None
        ):
            segments = np.asarray(static_barrier_segments_world, dtype=np.float32)
            if segments.ndim != 3 or segments.shape[1:] != (2, 3):
                raise ValueError(
                    "static_barrier_segments_world must have shape (N, 2, 3)"
                )
            barriers = tuple(
                InvisibleBarrier(
                    tuple(float(value) for value in segment[0]),
                    tuple(float(value) for value in segment[1]),
                    barrier_id=f"semantic-{index}",
                )
                for index, segment in enumerate(_simplify_barrier_segments(segments))
            )
        else:
            barriers = (
                self._build_barriers(scene) if vehicle.static_collision_enabled else ()
            )
        self.graph = PhysicsObjectGraph(objects=(), barriers=barriers)
        initial_transform = getattr(scene, "initial_rig_to_world", None)
        initial_xy = (
            np.asarray(initial_transform[:2, 3], dtype=np.float32)
            if initial_transform is not None
            else np.zeros(2, dtype=np.float32)
        )
        initial_timestamp_us = int(getattr(scene, "initial_timestamp_us", 0))
        self._vicinity_resolver = (
            None if game_map is None else GameMapVicinityResolver(game_map)
        )
        self._map_vicinity: GameMapVicinity | None = (
            None
            if self._vicinity_resolver is None
            else self._vicinity_resolver.resolve(
                float(initial_xy[0]), float(initial_xy[1])
            )
        )
        self._map_traffic.set_vicinity(self._map_vicinity)
        base_physics_graph = self.graph.copy_for_physx(
            initial_xy,
            _PHYSX_SIMULATION_RADIUS_M,
            timestamp_us=initial_timestamp_us,
        )
        self._physics_graph = self._with_active_controller_objects(base_physics_graph)
        self._synchronized_controller_ids = self._controller_active_ids()
        self._active_objects_by_id = {
            scene_object.object_id: scene_object
            for scene_object in self._physics_graph.objects
        }
        self._physics_center_xy = initial_xy.copy()
        self._physics_timestamp_us = initial_timestamp_us
        self._entities = [
            self._entity_from_object(obj) for obj in self._physics_graph.objects
        ]
        self._entities_by_id = {entity.entity_id: entity for entity in self._entities}
        self._detached_entity_ids: set[str] = set()
        self.last_step_timings = None
        self.last_step_bridge_timings = None
        self.last_step_actor_collision = False
        self.last_step_static_barrier_collision = False
        self.last_step_static_barrier_impact = False
        self._static_barrier_contact_active = False
        self._visual_flare_collision_velocity_mps: np.ndarray | None = None
        self._visual_flare_driving_direction_xy: np.ndarray | None = None
        self._visual_flare_impact_normal_xy: np.ndarray | None = None
        self._visual_flare_collision_deadline_us: int | None = None
        self._pending_struck_vehicle_ids: set[str] = set()
        self._ego_model = adapt_model(_ego_model(vehicle))
        self._world = GameplayPhysXWorld(
            base_physics_graph,
            self._ego_model,
            actor_collision_enabled=vehicle.actor_collision_enabled,
            max_actor_drive_speed_mps=_NON_EGO_MAX_DRIVE_SPEED_MPS,
            max_actor_drive_speeds_mps=self._controller_drive_speed_caps(),
        )
        self._world.synchronize(
            self._physics_graph,
            timestamp_us=initial_timestamp_us,
            initial_object_timestamps_us=self._controller_initial_timestamps(),
        )
        self._refresh_debug_barriers()
        logger.info(
            "[physics] PhysX graph ready in {:.1f} ms; objects={}/{} barriers={}/{} simulation_radius_m={:.0f}",
            (time.perf_counter() - started_at) * 1000.0,
            len(self._physics_graph.objects),
            len(self.graph.objects)
            + sum(len(controller.objects) for controller in self._actor_controllers),
            len(self._physics_graph.barriers),
            len(self.graph.barriers),
            _PHYSX_SIMULATION_RADIUS_M,
        )

    def _controller_owners(self) -> dict[str, PhysicsActorController]:
        """Return the unique gameplay owner of every controller actor."""
        owners: dict[str, PhysicsActorController] = {}
        scene_ids = {scene_object.object_id for scene_object in self.graph.objects}
        for controller in self._actor_controllers:
            for object_id in controller.object_ids:
                if object_id in scene_ids:
                    raise ValueError(
                        f"controller actor ID {object_id!r} conflicts with a scene actor"
                    )
                if object_id in owners:
                    raise ValueError(
                        f"controller actor ID {object_id!r} has multiple owners"
                    )
                owners[object_id] = controller
        return owners

    def _controller_active_ids(self) -> frozenset[str]:
        return frozenset(
            object_id
            for controller in self._actor_controllers
            for object_id in controller.active_object_ids
        )

    def _controller_initial_timestamps(self) -> dict[str, int]:
        timestamps: dict[str, int] = {}
        for controller in self._actor_controllers:
            for object_id, timestamp_us in controller.active_timestamps_us.items():
                if object_id in timestamps:
                    raise ValueError(
                        f"controller actor ID {object_id!r} has multiple timestamps"
                    )
                timestamps[object_id] = timestamp_us
        return timestamps

    def _controller_drive_speed_caps(self) -> dict[str, float]:
        speed_caps: dict[str, float] = {}
        for controller in self._actor_controllers:
            for object_id, speed_mps in controller.max_drive_speeds_mps.items():
                if object_id in speed_caps:
                    raise ValueError(
                        f"controller actor ID {object_id!r} has multiple speed caps"
                    )
                speed_caps[object_id] = speed_mps
        return speed_caps

    def _with_active_controller_objects(
        self, physics_graph: PhysicsObjectGraph
    ) -> PhysicsObjectGraph:
        """Add active gameplay-owned actors to a PhysX window."""
        incoming_ids = {
            scene_object.object_id for scene_object in physics_graph.objects
        }
        additions: list[SceneObject] = []
        for controller in self._actor_controllers:
            for scene_object in controller.active_objects:
                if scene_object.object_id in incoming_ids:
                    raise ValueError(
                        f"active actor ID {scene_object.object_id!r} is duplicated"
                    )
                incoming_ids.add(scene_object.object_id)
                additions.append(scene_object)
        if not additions:
            return physics_graph
        return PhysicsObjectGraph(
            objects=physics_graph.objects + tuple(additions),
            barriers=physics_graph.barriers,
        )

    @staticmethod
    def _entity_from_object(scene_object: SceneObject) -> GameEntity:
        dimensions = np.asarray(scene_object.model.half_extents_m) * 2.0
        return GameEntity(
            entity_id=scene_object.object_id,
            object_type=scene_object.object_type,
            transform=TransformComponent(
                scene_object.positions_m[0], scene_object.orientations_xyzw[0]
            ),
            rigid_body=RigidBodyComponent(
                mass_kg=scene_object.model.mass_kg,
                linear_velocity_mps=np.zeros(3, dtype=np.float32),
                angular_velocity_radps=np.zeros(3, dtype=np.float32),
                restitution=scene_object.model.restitution,
                friction=scene_object.model.friction,
            ),
            collider=BoxColliderComponent(scene_object.model.half_extents_m),
            vehicle=vehicle_dynamics_for_object(
                scene_object.object_type,
                dimensions,
                scene_object.model.mass_kg,
            ),
            suspension=suspension_for_object(scene_object.object_type),
        )

    @staticmethod
    def _build_barriers(scene: SceneDefinition) -> tuple[InvisibleBarrier, ...]:
        barriers: list[InvisibleBarrier] = []
        for layer_index, layer in enumerate(scene.line_layers):
            if any(
                token in layer.layer_name.lower() for token in _BARRIER_LAYER_TOKENS
            ):
                for segment_index, segment in enumerate(
                    _simplify_barrier_segments(layer.segments_world)
                ):
                    barriers.append(
                        InvisibleBarrier(
                            tuple(float(value) for value in segment[0]),
                            tuple(float(value) for value in segment[1]),
                            barrier_id=f"line-{layer_index}-{segment_index}",
                        )
                    )
        for layer_index, layer in enumerate(scene.polygon_layers):
            if not any(
                token in layer.layer_name.lower() for token in _BARRIER_LAYER_TOKENS
            ):
                continue
            for polygon_index, polygon in enumerate(layer.polygons_world):
                points = np.asarray(polygon, dtype=np.float32)
                for index in range(len(points)):
                    barriers.append(
                        InvisibleBarrier(
                            tuple(float(value) for value in points[index - 1, :2]),
                            tuple(float(value) for value in points[index, :2]),
                            barrier_id=(
                                f"polygon-{layer_index}-{polygon_index}-{index}"
                            ),
                        )
                    )
        return tuple(barriers)

    @property
    def entities(self) -> tuple[GameEntity, ...]:
        """Return actor entities synchronized from the Ludus object graph."""
        return tuple(self._entities)

    @property
    def _active_collider_ids(self) -> set[str]:
        """Expose the native collider set for invariant checks and diagnostics."""
        return set(self._world.active_collider_ids)

    def synchronize_window(
        self,
        center_xy_m: np.ndarray,
        timestamp_us: int | None = None,
        *,
        force_controller_refresh: bool = False,
    ) -> bool:
        """Incrementally recenter active PhysX topology when the ego moves."""
        center = np.asarray(center_xy_m, dtype=np.float32)
        if center.shape != (2,):
            raise ValueError("center_xy_m must have shape (2,)")
        traffic_topology_changed = False
        if self._vicinity_resolver is not None:
            self._map_vicinity = self._vicinity_resolver.resolve(
                float(center[0]),
                float(center[1]),
                previous=self._map_vicinity,
            )
            traffic_topology_changed = self._map_traffic.set_vicinity(
                self._map_vicinity
            )
        center_is_current = (
            float(np.linalg.norm(center - self._physics_center_xy))
            < _PHYSX_RECENTER_DISTANCE_M
        )
        timestamp_is_current = timestamp_us is None or (
            0
            <= timestamp_us - self._physics_timestamp_us
            < _PHYSX_TOPOLOGY_REFRESH_INTERVAL_US
        )
        if (
            center_is_current
            and timestamp_is_current
            and not traffic_topology_changed
            and not force_controller_refresh
        ):
            return False
        physics_graph = self.graph.copy_for_physx(
            center,
            _PHYSX_SIMULATION_RADIUS_M,
            timestamp_us=timestamp_us,
        )
        physics_graph = self._with_active_controller_objects(physics_graph)
        incoming_ids = {obj.object_id for obj in physics_graph.objects}
        controller_ids = frozenset(self._controller_owners())
        active_controller_ids = self._controller_active_ids()
        retained_detached = tuple(
            obj
            for obj in self._physics_graph.objects
            if obj.object_id in self._detached_entity_ids
            and (
                obj.object_id not in controller_ids
                or obj.object_id in active_controller_ids
            )
            and obj.object_id not in incoming_ids
        )
        if retained_detached:
            physics_graph = PhysicsObjectGraph(
                objects=physics_graph.objects + retained_detached,
                barriers=physics_graph.barriers,
            )
        self._world.set_actor_drive_speed_caps(self._controller_drive_speed_caps())
        self._world.synchronize(
            physics_graph,
            timestamp_us=timestamp_us,
            initial_object_timestamps_us=self._controller_initial_timestamps(),
        )
        existing_entities = {entity.entity_id: entity for entity in self._entities}
        self._entities = [
            existing_entities.get(scene_object.object_id)
            or self._entity_from_object(scene_object)
            for scene_object in physics_graph.objects
        ]
        self._entities_by_id = {entity.entity_id: entity for entity in self._entities}
        self._detached_entity_ids.intersection_update(self._entities_by_id)
        self._physics_graph = physics_graph
        self._synchronized_controller_ids = active_controller_ids
        self._active_objects_by_id = {
            scene_object.object_id: scene_object
            for scene_object in self._physics_graph.objects
        }
        self._physics_center_xy = center.copy()
        if timestamp_us is not None:
            self._physics_timestamp_us = timestamp_us
        self._refresh_debug_barriers()
        return True

    def _refresh_debug_barriers(self) -> None:
        if self._physics_graph.barriers:
            self._debug_barrier_ids = tuple(
                barrier.barrier_id or f"barrier-{index}"
                for index, barrier in enumerate(self._physics_graph.barriers)
            )
            self._debug_barrier_segments = np.asarray(
                [
                    [barrier.start_xy_m, barrier.end_xy_m]
                    for barrier in self._physics_graph.barriers
                ],
                dtype=np.float32,
            )
            self._debug_barrier_thicknesses = np.asarray(
                [barrier.thickness_m for barrier in self._physics_graph.barriers],
                dtype=np.float32,
            )
            self._debug_barrier_heights = np.asarray(
                [barrier.height_m for barrier in self._physics_graph.barriers],
                dtype=np.float32,
            )
        else:
            self._debug_barrier_ids = ()
            self._debug_barrier_segments = np.empty((0, 2, 2), dtype=np.float32)
            self._debug_barrier_thicknesses = np.empty((0,), dtype=np.float32)
            self._debug_barrier_heights = np.empty((0,), dtype=np.float32)
        self._barrier_rebound_index = _BarrierReboundIndex.from_arrays(
            self._debug_barrier_segments,
            self._debug_barrier_thicknesses,
        )

    def debug_frame(self, state: VehicleState) -> PhysicsDebugFrame:
        """Capture the active collider topology without rendering it."""
        half_yaw = state.yaw_rad * 0.5
        ego_xy = np.asarray([state.x_m, state.y_m], dtype=np.float32)
        forward = np.asarray(
            [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float32
        )
        left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
        (
            actor_ids,
            actor_positions,
            actor_orientations,
            actor_dimensions,
        ) = self._world.collider_state_arrays()
        if len(actor_positions):
            actor_delta = actor_positions[:, :2] - ego_xy
            actor_forward = actor_delta @ forward
            actor_lateral = actor_delta @ left
            quaternion = actor_orientations
            qx, qy, qz, qw = (quaternion[:, index] for index in range(4))
            # Project the full oriented collider onto the ego axes. Checking
            # only its center drops long vehicles whose body crosses the view
            # boundary even though the collider is still visible.
            local_axes_xy = (
                np.column_stack(
                    (
                        1.0 - 2.0 * (qy * qy + qz * qz),
                        2.0 * (qx * qy + qw * qz),
                    )
                ),
                np.column_stack(
                    (
                        2.0 * (qx * qy - qw * qz),
                        1.0 - 2.0 * (qx * qx + qz * qz),
                    )
                ),
                np.column_stack(
                    (
                        2.0 * (qx * qz + qw * qy),
                        2.0 * (qy * qz - qw * qx),
                    )
                ),
            )
            half_dimensions = actor_dimensions * 0.5
            actor_forward_radius = sum(
                half_dimensions[:, axis_index] * np.abs(local_axis @ forward)
                for axis_index, local_axis in enumerate(local_axes_xy)
            )
            actor_lateral_radius = sum(
                half_dimensions[:, axis_index] * np.abs(local_axis @ left)
                for axis_index, local_axis in enumerate(local_axes_xy)
            )
            actor_visible = (
                (actor_forward + actor_forward_radius >= -_PHYSX_DEBUG_REAR_M)
                & (actor_forward - actor_forward_radius <= _PHYSX_DEBUG_FORWARD_M)
                & (
                    np.abs(actor_lateral) - actor_lateral_radius
                    <= _PHYSX_DEBUG_LATERAL_M
                )
            )
            actor_positions = actor_positions[actor_visible]
            actor_orientations = actor_orientations[actor_visible]
            actor_dimensions = actor_dimensions[actor_visible]
            actor_ids = tuple(
                object_id
                for object_id, visible in zip(actor_ids, actor_visible, strict=True)
                if bool(visible)
            )
        else:
            actor_positions = np.empty((0, 3), dtype=np.float32)
            actor_orientations = np.empty((0, 4), dtype=np.float32)
            actor_dimensions = np.empty((0, 3), dtype=np.float32)
        if len(self._debug_barrier_segments):
            barrier_delta = self._debug_barrier_segments - ego_xy
            barrier_forward = barrier_delta @ forward
            barrier_lateral = barrier_delta @ left
            barrier_radius = self._debug_barrier_thicknesses * 0.5
            barrier_visible = (
                (
                    np.max(barrier_forward, axis=1) + barrier_radius
                    >= -_PHYSX_DEBUG_REAR_M
                )
                & (
                    np.min(barrier_forward, axis=1) - barrier_radius
                    <= _PHYSX_DEBUG_FORWARD_M
                )
                & (
                    np.min(barrier_lateral, axis=1) - barrier_radius
                    <= _PHYSX_DEBUG_LATERAL_M
                )
                & (
                    np.max(barrier_lateral, axis=1) + barrier_radius
                    >= -_PHYSX_DEBUG_LATERAL_M
                )
            )
            barrier_segments = self._debug_barrier_segments[barrier_visible]
            barrier_thicknesses = self._debug_barrier_thicknesses[barrier_visible]
            barrier_heights = self._debug_barrier_heights[barrier_visible]
            barrier_ids = tuple(
                barrier_id
                for barrier_id, visible in zip(
                    self._debug_barrier_ids, barrier_visible, strict=True
                )
                if bool(visible)
            )
        else:
            barrier_segments = self._debug_barrier_segments
            barrier_thicknesses = self._debug_barrier_thicknesses
            barrier_heights = self._debug_barrier_heights
            barrier_ids = self._debug_barrier_ids
        return PhysicsDebugFrame(
            ego_position_m=np.asarray(
                [
                    state.x_m,
                    state.y_m,
                    state.z_m + self._ego_model.half_extents_m[2],
                ],
                dtype=np.float32,
            ),
            ego_orientation_xyzw=np.asarray(
                [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
                dtype=np.float32,
            ),
            ego_dimensions_lwh=np.asarray(
                self._ego_model.half_extents_m, dtype=np.float32
            )
            * 2.0,
            actor_positions_m=actor_positions,
            actor_orientations_xyzw=actor_orientations,
            actor_dimensions_lwh=actor_dimensions,
            barrier_segments_xy_m=barrier_segments,
            barrier_thicknesses_m=barrier_thicknesses,
            barrier_heights_m=barrier_heights,
            actor_ids=actor_ids,
            barrier_ids=barrier_ids,
        )

    def step(
        self, state: VehicleState, timestamp_us: int, dt_s: float
    ) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
        """Advance the authoritative PhysX scene and return actor samples."""
        step_started_at = time.perf_counter()
        ego_before_step = _body_state_from_vehicle(
            state, self._ego_model.half_extents_m[2]
        )
        traffic_prepare_started_at = time.perf_counter()
        for controller in self._actor_controllers:
            controller.prepare_topology(ego_before_step)
        if self._controller_active_ids() != self._synchronized_controller_ids:
            self.synchronize_window(
                np.asarray(ego_before_step.position_m[:2], dtype=np.float32),
                timestamp_us,
                force_controller_refresh=True,
            )
        actor_targets = tuple(
            target
            for controller in self._actor_controllers
            for target in controller.prepare_step(ego_before_step, dt_s)
        )
        self._world.apply_actor_track_targets(actor_targets)
        traffic_prepare_ms = (time.perf_counter() - traffic_prepare_started_at) * 1000.0
        physics_step = self._world.step_compact(
            ego_before_step,
            timestamp_us,
            dt_s,
        )
        barrier_rebound_started_at = time.perf_counter()
        self.last_step_static_barrier_collision = False
        if self._static_barrier_restitution is not None:
            (
                reinforced_velocity,
                self.last_step_static_barrier_collision,
            ) = _reinforce_static_barrier_rebound(
                self._barrier_rebound_index,
                ego_before_step,
                physics_step.ego.linear_velocity_mps,
                self._ego_model,
                self._static_barrier_restitution,
            )
            physics_step = replace(
                physics_step,
                ego=replace(
                    physics_step.ego,
                    linear_velocity_mps=reinforced_velocity,
                ),
            )
        barrier_rebound_ms = (time.perf_counter() - barrier_rebound_started_at) * 1000.0
        traffic_update_started_at = time.perf_counter()
        self.last_step_static_barrier_impact = (
            self.last_step_static_barrier_collision
            and not self._static_barrier_contact_active
        )
        self._static_barrier_contact_active = self.last_step_static_barrier_collision
        if physics_step.struck_object_ids:
            self._pending_struck_vehicle_ids.update(physics_step.struck_object_ids)
            self._visual_flare_collision_velocity_mps = (
                ego_before_step.linear_velocity_mps.copy()
            )
            self._visual_flare_driving_direction_xy = np.asarray(
                [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float32
            )
            strongest_closing_speed_mps = _VISUAL_FLARE_MIN_SPEED_DELTA_MPS
            self._visual_flare_impact_normal_xy = None
            actor_bodies = {
                object_id: body for object_id, body, _ in physics_step.actor_samples
            }
            for object_id in physics_step.struck_object_ids:
                body = actor_bodies.get(object_id)
                scene_object = self._active_objects_by_id.get(object_id)
                if body is None or scene_object is None:
                    continue
                separation_xy = body.position_m[:2] - ego_before_step.position_m[:2]
                separation_m = float(np.linalg.norm(separation_xy))
                if separation_m <= 1e-6:
                    continue
                impact_normal_xy = separation_xy / separation_m
                traffic_state = self._map_traffic.state(object_id)
                actor_velocity_mps = body.linear_velocity_mps
                if traffic_state is not None:
                    _, _, actor_velocity_mps = scene_object.sample(
                        int(traffic_state.timestamp_us)
                    )
                relative_velocity_xy = (
                    actor_velocity_mps[:2] - ego_before_step.linear_velocity_mps[:2]
                )
                closing_speed_mps = -float(
                    np.dot(relative_velocity_xy, impact_normal_xy)
                )
                if closing_speed_mps >= strongest_closing_speed_mps:
                    strongest_closing_speed_mps = closing_speed_mps
                    self._visual_flare_impact_normal_xy = impact_normal_xy.copy()
            self._visual_flare_collision_deadline_us = (
                timestamp_us + _VISUAL_FLARE_COLLISION_WINDOW_US
            )
        collision_window_active = (
            self._visual_flare_collision_deadline_us is not None
            and timestamp_us <= self._visual_flare_collision_deadline_us
        )
        flare_baseline_velocity = self._visual_flare_collision_velocity_mps
        if flare_baseline_velocity is None:
            flare_baseline_velocity = ego_before_step.linear_velocity_mps
        flare_driving_direction = self._visual_flare_driving_direction_xy
        if flare_driving_direction is None:
            flare_driving_direction = np.asarray(
                [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float32
            )
        self.last_step_actor_collision = _is_visual_flare_impact(
            physics_step.impact or collision_window_active,
            flare_baseline_velocity,
            physics_step.ego.linear_velocity_mps,
            flare_driving_direction,
            self._visual_flare_impact_normal_xy,
        )
        collision_window_expired = (
            self._visual_flare_collision_deadline_us is not None
            and timestamp_us > self._visual_flare_collision_deadline_us
        )
        if self.last_step_actor_collision or collision_window_expired:
            self._visual_flare_collision_velocity_mps = None
            self._visual_flare_driving_direction_xy = None
            self._visual_flare_impact_normal_xy = None
            self._visual_flare_collision_deadline_us = None
            self._pending_struck_vehicle_ids.clear()
        actor_samples = []
        pending_controls = []
        controller_owners = self._controller_owners()
        actor_bodies: dict[str, BodyState] = {}
        for object_id, body, _native_detached in physics_step.actor_samples:
            controller = controller_owners.get(object_id)
            if controller is None:
                raise RuntimeError(f"PhysX returned unmanaged actor {object_id!r}")
            decision = controller.observe_physics(
                object_id,
                struck=object_id in physics_step.struck_object_ids,
                body=body,
                dt_s=dt_s,
            )
            if decision is None:
                raise RuntimeError(
                    f"actor controller rejected owned actor {object_id!r}"
                )
            detached = decision.detached_from_track
            pending_controls.append((object_id, decision.drive_enabled, detached))
            actor_bodies[object_id] = body
            actor_samples.append(
                (object_id, body.position_m, body.orientation_xyzw, detached)
            )
        self._world.apply_track_controls(tuple(pending_controls))
        traffic_update_ms = (time.perf_counter() - traffic_update_started_at) * 1000.0

        state_materialize_started_at = time.perf_counter()
        detached_ids = {sample[0] for sample in actor_samples if sample[3]}
        for object_id in self._detached_entity_ids - detached_ids:
            self._entities_by_id[object_id].detached_from_track = False
        for object_id, position, orientation, detached in actor_samples:
            entity = self._entities_by_id[object_id]
            entity.transform.position_m = position.copy()
            entity.transform.orientation_xyzw = orientation.copy()
            body = actor_bodies[object_id]
            entity.rigid_body.linear_velocity_mps = body.linear_velocity_mps.copy()
            entity.rigid_body.angular_velocity_radps = (
                body.angular_velocity_radps.copy()
            )
            entity.detached_from_track = detached
        self._detached_entity_ids = detached_ids

        ego = physics_step.ego
        yaw = _yaw_from_quaternion_xyzw(ego.orientation_xyzw)
        collision_response_active = (
            physics_step.impact
            or bool(physics_step.struck_object_ids)
            or state.ragdoll_active
        )
        yaw_rate_radps = float(ego.angular_velocity_radps[2])
        if collision_response_active:
            max_yaw_rate = self._vehicle.max_collision_yaw_rate_radps
            yaw_delta = math.atan2(
                math.sin(yaw - state.yaw_rad), math.cos(yaw - state.yaw_rad)
            )
            yaw_delta = float(
                np.clip(yaw_delta, -max_yaw_rate * dt_s, max_yaw_rate * dt_s)
            )
            yaw = state.yaw_rad + yaw_delta
            yaw_rate_radps = float(np.clip(yaw_rate_radps, -max_yaw_rate, max_yaw_rate))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)])
        ego_height_m = float(ego.position_m[2] - self._ego_model.half_extents_m[2])
        remains_unsettled = (
            abs(ego_height_m) > 0.10
            or abs(float(ego.linear_velocity_mps[2])) > 0.50
            or float(np.linalg.norm(ego.angular_velocity_radps[:2])) > 0.25
        )
        result_state = replace(
            state,
            x_m=float(ego.position_m[0]),
            y_m=float(ego.position_m[1]),
            z_m=ego_height_m,
            yaw_rad=yaw,
            speed_mps=float(np.dot(ego.linear_velocity_mps[:2], forward)),
            velocity_x_mps=float(ego.linear_velocity_mps[0]),
            velocity_y_mps=float(ego.linear_velocity_mps[1]),
            yaw_rate_radps=yaw_rate_radps,
            ragdoll_active=physics_step.impact or remains_unsettled,
        )
        samples = tuple(
            (
                object_id,
                position.copy(),
                orientation.copy(),
                detached,
            )
            for object_id, position, orientation, detached in actor_samples
        )
        state_materialize_ms = (
            time.perf_counter() - state_materialize_started_at
        ) * 1000.0
        step_total_ms = (time.perf_counter() - step_started_at) * 1000.0
        native_ms = (
            physics_step.timings.actor_update_ms
            + physics_step.timings.solver_ms
            + physics_step.timings.readback_ms
        )
        bridge_ms = max(0.0, step_total_ms - native_ms)
        self.last_step_timings = replace(
            physics_step.timings,
            total_ms=step_total_ms,
            bridge_ms=bridge_ms,
        )
        self.last_step_bridge_timings = PhysicsBridgeTimings(
            traffic_prepare_ms=traffic_prepare_ms,
            barrier_rebound_ms=barrier_rebound_ms,
            traffic_update_ms=traffic_update_ms,
            state_materialize_ms=state_materialize_ms,
            other_ms=max(
                0.0,
                bridge_ms
                - traffic_prepare_ms
                - barrier_rebound_ms
                - traffic_update_ms
                - state_materialize_ms,
            ),
        )
        return result_state, samples

    def synchronize_ego_state(self, state: VehicleState) -> None:
        """Publish an app-authoritative ego state to the owned PhysX scene.

        This adapter contains the native body identifier and state-array layout so
        application policies do not depend on Ludus implementation details.

        Args:
            state: Authoritative vehicle state to publish.
        """
        body = _body_state_from_vehicle(state, self._ego_model.half_extents_m[2])
        pose = np.concatenate((body.position_m, body.orientation_xyzw)).astype(
            np.float32, copy=False
        )
        self._world._scene.update_body(
            0,
            pose,
            np.asarray(body.linear_velocity_mps, dtype=np.float32),
            np.asarray(body.angular_velocity_radps, dtype=np.float32),
            False,
        )

    def close(self) -> None:
        """Release the Ludus PhysX world."""
        self._world.close()

    def build_trajectories(
        self,
        timestamps_us: np.ndarray,
        samples_by_frame: list[tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]],
    ) -> tuple[DynamicActorTrajectory, ...]:
        """Pack PhysX object samples for Ludus RGB and BEV HD-map rendering."""
        controller_objects = tuple(
            scene_object
            for controller in getattr(self, "_actor_controllers", ())
            for scene_object in controller.active_objects
        )
        if (not self.graph.objects and not controller_objects) or not samples_by_frame:
            return ()
        result: list[DynamicActorTrajectory] = []
        simulated_timestamps = np.asarray(timestamps_us, dtype=np.int64)
        samples_by_id = [
            {sample[0]: sample for sample in frame} for frame in samples_by_frame
        ]
        simulated_ids = {
            object_id for frame in samples_by_id for object_id in frame.keys()
        }
        render_objects = (*self.graph.objects, *controller_objects)
        for scene_object in render_objects:
            physically_simulated = scene_object.object_id in simulated_ids
            if physically_simulated:
                detached = any(
                    frame[scene_object.object_id][3]
                    for frame in samples_by_id
                    if scene_object.object_id in frame
                )
                frame_poses = []
                for timestamp, frame in zip(
                    simulated_timestamps, samples_by_id, strict=True
                ):
                    sample = frame.get(scene_object.object_id)
                    if sample is None:
                        position, orientation, _ = scene_object.sample(int(timestamp))
                    else:
                        position, orientation = sample[1], sample[2]
                    frame_poses.append((position, orientation))
                positions = np.stack([pose[0] for pose in frame_poses]).astype(
                    np.float32
                )
                orientations = np.stack([pose[1] for pose in frame_poses]).astype(
                    np.float32
                )
                trajectory_timestamps = simulated_timestamps
            else:
                continue
            result.append(
                DynamicActorTrajectory(
                    entity_id=scene_object.object_id,
                    object_type=scene_object.object_type,
                    timestamps_us=trajectory_timestamps,
                    translations_world=positions,
                    orientations_xyzw=orientations,
                    dimensions_lwh=np.asarray(
                        scene_object.model.half_extents_m, dtype=np.float32
                    )
                    * 2.0,
                    detached_from_track=detached,
                    is_simulated=True,
                )
            )
        return tuple(result)
