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
from dataclasses import replace

import numpy as np
from loguru import logger
from ludus_renderer import (
    BodyState,
    InvisibleBarrier,
    PhysicsObjectGraph,
    PhysXWorld,
    RigidBodyModel,
    SceneObject,
)

from interactive_drive.config import VehicleConfig
from interactive_drive.simulation.components import (
    BoxColliderComponent,
    GameEntity,
    RigidBodyComponent,
    TransformComponent,
    rigid_body_model_for_object,
    rigid_body_model_from_vehicle_config,
    suspension_for_object,
    vehicle_dynamics_for_object,
)
from interactive_drive.simulation.traffic_ai import TrafficDriverAI
from interactive_drive.types import (
    DynamicActorTrajectory,
    PhysicsDebugFrame,
    SceneBundle,
    VehicleState,
)

_BARRIER_LAYER_TOKENS = ("road_bound", "building", "house", "wall", "curb")
_PHYSX_RECENTER_DISTANCE_M = 32.0
_PHYSX_TOPOLOGY_REFRESH_INTERVAL_US = 2_000_000
"""Maximum time moving tracks can remain outside a stationary PhysX window."""

_PHYSX_BARRIER_SPACING_M = 2.0
_PHYSX_DEBUG_FORWARD_M = 125.0
_PHYSX_DEBUG_REAR_M = 15.0
_PHYSX_DEBUG_LATERAL_M = 100.0
_VISUAL_FLARE_MIN_SPEED_DELTA_MPS = 5.0 * 0.44704
_VISUAL_FLARE_COLLISION_WINDOW_US = 500_000
_NON_EGO_MAX_DRIVE_SPEED_MPS = 15.0 * 0.44704
_PERSISTENT_TRACK_TIMESTAMP_US = np.iinfo(np.int64).max // 4
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


def _recorded_actor_trajectory(scene_object: SceneObject) -> DynamicActorTrajectory:
    """Build one reusable renderer track that holds its final pose indefinitely."""
    timestamps = scene_object.timestamps_us
    positions = scene_object.positions_m
    orientations = scene_object.orientations_xyzw
    if int(timestamps[-1]) < _PERSISTENT_TRACK_TIMESTAMP_US:
        timestamps = np.concatenate(
            (timestamps, np.asarray([_PERSISTENT_TRACK_TIMESTAMP_US], dtype=np.int64))
        )
        positions = np.concatenate((positions, positions[-1:]), axis=0)
        orientations = np.concatenate((orientations, orientations[-1:]), axis=0)
    return DynamicActorTrajectory(
        entity_id=scene_object.object_id,
        object_type=scene_object.object_type,
        timestamps_us=timestamps,
        translations_world=positions,
        orientations_xyzw=orientations,
        dimensions_lwh=np.asarray(scene_object.model.half_extents_m, dtype=np.float32)
        * 2.0,
    )


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


class GamePhysicsWorld:
    """Adapt a scene bundle to Ludus and delegate all simulation to PhysX."""

    def __init__(self, scene: SceneBundle, vehicle: VehicleConfig) -> None:
        started_at = time.perf_counter()
        self._vehicle = vehicle
        objects = tuple(
            self._object_from_track(track, vehicle)
            for track in scene.vehicle_bbox_tracks
        )
        barriers = (
            self._build_barriers(scene) if vehicle.static_collision_enabled else ()
        )
        self.graph = PhysicsObjectGraph(objects=objects, barriers=barriers)
        self._recorded_trajectories_by_id = {
            obj.object_id: _recorded_actor_trajectory(obj) for obj in objects
        }
        initial_transform = getattr(scene, "initial_rig_to_world", None)
        initial_xy = (
            np.asarray(initial_transform[:2, 3], dtype=np.float32)
            if initial_transform is not None
            else np.zeros(2, dtype=np.float32)
        )
        initial_timestamp_us = int(getattr(scene, "initial_timestamp_us", 0))
        self._physics_graph = self.graph.copy_for_physx(
            initial_xy,
            _PHYSX_SIMULATION_RADIUS_M,
            timestamp_us=initial_timestamp_us,
        )
        self._physics_center_xy = initial_xy.copy()
        self._physics_timestamp_us = initial_timestamp_us
        self._entities = [
            self._entity_from_object(obj) for obj in self._physics_graph.objects
        ]
        self._entities_by_id = {entity.entity_id: entity for entity in self._entities}
        self._detached_entity_ids: set[str] = set()
        self.last_step_timings = None
        self.last_step_actor_collision = False
        self._visual_flare_collision_velocity_mps: np.ndarray | None = None
        self._visual_flare_driving_direction_xy: np.ndarray | None = None
        self._visual_flare_impact_normal_xy: np.ndarray | None = None
        self._visual_flare_collision_deadline_us: int | None = None
        self._pending_struck_vehicle_ids: set[str] = set()
        self._ego_model = _ego_model(vehicle)
        self._world = PhysXWorld(
            self._physics_graph,
            self._ego_model,
            actor_collision_enabled=vehicle.actor_collision_enabled,
            max_actor_drive_speed_mps=_NON_EGO_MAX_DRIVE_SPEED_MPS,
        )
        self._traffic_ai = TrafficDriverAI()
        self._traffic_ai.synchronize(self._physics_graph.objects)
        self._refresh_debug_barriers()
        logger.info(
            "[physics] PhysX graph ready in {:.1f} ms; objects={}/{} barriers={}/{} simulation_radius_m={:.0f}",
            (time.perf_counter() - started_at) * 1000.0,
            len(self._physics_graph.objects),
            len(self.graph.objects),
            len(self._physics_graph.barriers),
            len(self.graph.barriers),
            _PHYSX_SIMULATION_RADIUS_M,
        )

    @staticmethod
    def _object_from_track(track: object, vehicle: VehicleConfig) -> SceneObject:
        dimensions = np.asarray(track.dimensions_lwh[0], dtype=np.float32)
        return SceneObject(
            object_id=track.track_id,
            object_type=track.object_type,
            model=rigid_body_model_for_object(
                track.object_type,
                dimensions,
                restitution=vehicle.collision_restitution,
                friction=vehicle.collision_friction,
            ),
            timestamps_us=np.asarray(track.timestamps_us, dtype=np.int64),
            positions_m=np.asarray(track.centers_world, dtype=np.float32),
            orientations_xyzw=np.asarray(track.orientations_xyzw, dtype=np.float32),
            max_extrapolation_us=track.max_extrapolation_us,
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
    def _build_barriers(scene: SceneBundle) -> tuple[InvisibleBarrier, ...]:
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
        self, center_xy_m: np.ndarray, timestamp_us: int | None = None
    ) -> bool:
        """Incrementally recenter active PhysX topology when the ego moves."""
        center = np.asarray(center_xy_m, dtype=np.float32)
        if center.shape != (2,):
            raise ValueError("center_xy_m must have shape (2,)")
        center_is_current = (
            float(np.linalg.norm(center - self._physics_center_xy))
            < _PHYSX_RECENTER_DISTANCE_M
        )
        timestamp_is_current = timestamp_us is None or (
            0
            <= timestamp_us - self._physics_timestamp_us
            < _PHYSX_TOPOLOGY_REFRESH_INTERVAL_US
        )
        if center_is_current and timestamp_is_current:
            return False
        physics_graph = self.graph.copy_for_physx(
            center,
            _PHYSX_SIMULATION_RADIUS_M,
            timestamp_us=timestamp_us,
        )
        incoming_ids = {obj.object_id for obj in physics_graph.objects}
        retained_detached = tuple(
            obj
            for obj in self._physics_graph.objects
            if obj.object_id in self._detached_entity_ids
            and obj.object_id not in incoming_ids
        )
        if retained_detached:
            physics_graph = PhysicsObjectGraph(
                objects=physics_graph.objects + retained_detached,
                barriers=physics_graph.barriers,
            )
        self._world.synchronize(physics_graph, timestamp_us=timestamp_us)
        self._traffic_ai.synchronize(physics_graph.objects)
        existing_entities = {entity.entity_id: entity for entity in self._entities}
        self._entities = [
            existing_entities.get(scene_object.object_id)
            or self._entity_from_object(scene_object)
            for scene_object in physics_graph.objects
        ]
        self._entities_by_id = {entity.entity_id: entity for entity in self._entities}
        self._detached_entity_ids.intersection_update(self._entities_by_id)
        self._physics_graph = physics_graph
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
        physics_step = self._world.step_compact(
            ego_before_step,
            timestamp_us,
            dt_s,
        )
        active_objects = {
            scene_object.object_id: scene_object
            for scene_object in self._physics_graph.objects
        }
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
                scene_object = active_objects.get(object_id)
                if body is None or scene_object is None:
                    continue
                separation_xy = body.position_m[:2] - ego_before_step.position_m[:2]
                separation_m = float(np.linalg.norm(separation_xy))
                if separation_m <= 1e-6:
                    continue
                impact_normal_xy = separation_xy / separation_m
                _, _, track_velocity_mps = scene_object.sample(timestamp_us)
                relative_velocity_xy = (
                    track_velocity_mps[:2] - ego_before_step.linear_velocity_mps[:2]
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
        significant_struck_vehicle_ids = (
            self._pending_struck_vehicle_ids.copy()
            if self.last_step_actor_collision
            else set()
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
        native_track_samples = getattr(physics_step, "track_samples", ())
        for actor_index, (object_id, body, native_detached) in enumerate(
            physics_step.actor_samples
        ):
            scene_object = active_objects[object_id]
            if actor_index < len(native_track_samples):
                (
                    track_object_id,
                    track_position,
                    track_orientation,
                    track_velocity,
                ) = native_track_samples[actor_index]
                if track_object_id != object_id:
                    raise RuntimeError("native actor and track samples are misaligned")
            else:
                # Compatibility path for light-weight test doubles and older
                # native modules while their timestamp sampling remains Python-side.
                track_position, track_orientation, track_velocity = scene_object.sample(
                    timestamp_us
                )
            decision = self._traffic_ai.update(
                object_id,
                struck=object_id in significant_struck_vehicle_ids,
                body=body,
                track_position=track_position,
                track_orientation_xyzw=track_orientation,
                track_velocity_mps=track_velocity,
                dt_s=dt_s,
            )
            if decision is None:
                detached = native_detached
            else:
                detached = decision.detached_from_track
                # Do not let the track motor erase momentum while the visual
                # effect's short impact-measurement window is still open.
                drive_enabled = (
                    decision.drive_enabled
                    and object_id not in self._pending_struck_vehicle_ids
                )
                pending_controls.append((object_id, drive_enabled, detached))
            actor_samples.append(
                (object_id, body.position_m, body.orientation_xyzw, detached)
            )
        self._world.apply_track_controls(tuple(pending_controls))

        detached_ids = {sample[0] for sample in actor_samples if sample[3]}
        for object_id in self._detached_entity_ids - detached_ids:
            self._entities_by_id[object_id].detached_from_track = False
        for object_id, position, orientation, detached in actor_samples:
            entity = self._entities_by_id[object_id]
            entity.transform.position_m = position.copy()
            entity.transform.orientation_xyzw = orientation.copy()
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
        step_total_ms = (time.perf_counter() - step_started_at) * 1000.0
        native_ms = (
            physics_step.timings.actor_update_ms
            + physics_step.timings.solver_ms
            + physics_step.timings.readback_ms
        )
        self.last_step_timings = replace(
            physics_step.timings,
            total_ms=step_total_ms,
            bridge_ms=max(0.0, step_total_ms - native_ms),
        )
        return result_state, samples

    def close(self) -> None:
        """Release the Ludus PhysX world."""
        self._world.close()

    def build_trajectories(
        self,
        timestamps_us: np.ndarray,
        samples_by_frame: list[tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]],
    ) -> tuple[DynamicActorTrajectory, ...]:
        """Pack PhysX object samples for Ludus RGB and BEV HD-map rendering."""
        if not self.graph.objects or not samples_by_frame:
            return ()
        result: list[DynamicActorTrajectory] = []
        simulated_timestamps = np.asarray(timestamps_us, dtype=np.int64)
        samples_by_id = [
            {sample[0]: sample for sample in frame} for frame in samples_by_frame
        ]
        simulated_ids = {
            object_id for frame in samples_by_id for object_id in frame.keys()
        }
        for scene_object in self.graph.objects:
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
                result.append(self._recorded_trajectories_by_id[scene_object.object_id])
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
