# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-game policy adapter around the reusable model-thread PhysX world."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from loguru import logger
from ludus_renderer import PhysicsObjectGraph, RigidBodyModel
from omnidreams_game_engine.simulation.actor_controller import (
    PhysicsActorController,
)
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.types import (
    DriverCommand,
    PhysicsDebugFrame,
    SceneDefinition,
    VehicleState,
)

from crazy_robotaxi.dynamics import TaxiVehicleConfig

_CHASSIS_INSET_M = 0.16
_TAXI_PHYSX_RECENTER_DISTANCE_M = 32.0
"""Taxi spatial collision horizon recenter threshold."""


def inset_vehicle_chassis(model: RigidBodyModel) -> RigidBodyModel:
    """Inset Taxi vehicle boxes to approximate beveled corners app-side."""
    if model.vehicle is None:
        return model
    x_m, y_m, z_m = model.vehicle.chassis_half_extents_m
    vehicle = replace(
        model.vehicle,
        chassis_half_extents_m=(
            max(0.25, x_m - _CHASSIS_INSET_M),
            max(0.25, y_m - _CHASSIS_INSET_M),
            z_m,
        ),
    )
    return replace(model, vehicle=vehicle)


class TaxiPhysicsWorld(GamePhysicsWorld):
    """Apply Taxi policy around an otherwise unmodified generic PhysX world."""

    def __init__(
        self,
        scene: SceneDefinition,
        vehicle: TaxiVehicleConfig,
        *,
        curb_segments_world: np.ndarray | None = None,
        actor_controllers: tuple[PhysicsActorController, ...] = (),
    ) -> None:
        curb_segments = np.asarray(
            curb_segments_world
            if curb_segments_world is not None
            else np.empty((0, 2, 3), dtype=np.float32),
            dtype=np.float32,
        )
        if curb_segments.ndim != 3 or curb_segments.shape[1:] != (2, 3):
            raise ValueError("Taxi curb segments must have shape (N, 2, 3).")
        super().__init__(
            scene,
            vehicle,
            model_adapter=inset_vehicle_chassis,
            static_barrier_segments_world=(
                curb_segments
                if getattr(scene, "game_map", None) is not None or len(curb_segments)
                else None
            ),
            static_barrier_restitution=vehicle.curb_collision_restitution,
            actor_controllers=actor_controllers,
        )
        self._has_external_actor_controllers = bool(actor_controllers)
        self._taxi_vehicle = vehicle
        logger.info(
            "[crazy-robotaxi] Taxi physics active: app-authoritative heading, "
            "arcade handbrake, inset chassis, curb_segments={}",
            len(curb_segments),
        )
        self._last_contact_resolved_state: VehicleState | None = None

    def synchronize_window(
        self,
        center_xy_m: np.ndarray,
        timestamp_us: int | None = None,
        *,
        force_controller_refresh: bool = False,
    ) -> bool:
        """Refresh Taxi collision topology only when its spatial window changes.

        Crazy Robotaxi's base graph contains static semantic barriers and its
        moving actors are all owned by ``MapTrafficController``. The base
        implementation already detects map-traffic vicinity changes on every
        call, so its periodic timestamp-only rebuild merely re-filters the same
        thousands of static curb segments. Passing no timestamp retains spatial
        recentering and traffic additions/removals without that redundant scan.
        """
        if (
            getattr(self, "_has_external_actor_controllers", False)
            or force_controller_refresh
        ):
            return super().synchronize_window(
                center_xy_m,
                timestamp_us,
                force_controller_refresh=force_controller_refresh,
            )
        del timestamp_us
        center = np.asarray(center_xy_m, dtype=np.float32)
        if center.shape != (2,):
            raise ValueError("center_xy_m must have shape (2,)")
        if self.graph.objects or (
            float(np.linalg.norm(center - self._physics_center_xy))
            >= _TAXI_PHYSX_RECENTER_DISTANCE_M
        ):
            return super().synchronize_window(center, timestamp_us=None)

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
        if not traffic_topology_changed:
            return False

        incoming = self._map_traffic.active_objects
        incoming_ids = {scene_object.object_id for scene_object in incoming}
        retained_detached = tuple(
            scene_object
            for scene_object in self._physics_graph.objects
            if scene_object.object_id in self._detached_entity_ids
            and scene_object.object_id in self._map_traffic.active_object_ids
            and scene_object.object_id not in incoming_ids
        )
        objects = incoming + retained_detached
        # This graph is a desired-state message for PhysX, not a graph that is
        # spatially queried. Building it from ``objects=...`` would construct a
        # transient track index on every traffic-boundary crossing.
        physics_graph = PhysicsObjectGraph()
        physics_graph.objects = objects
        physics_graph.object_index = {
            scene_object.object_id: index for index, scene_object in enumerate(objects)
        }
        physics_graph.barriers = self._physics_graph.barriers
        self._world.synchronize(
            physics_graph,
            initial_object_timestamps_us=self._map_traffic.active_timestamps_us,
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
        return True

    def step_with_command(
        self,
        state: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
        """Resolve contacts while keeping Taxi drive intent authoritative."""
        resolved, samples = super().step(state, timestamp_us, dt_s)
        self._last_contact_resolved_state = resolved
        if command.handbrake and not resolved.ragdoll_active:
            velocity_x_mps = state.velocity_x_mps
            velocity_y_mps = state.velocity_y_mps
        else:
            velocity_x_mps = resolved.velocity_x_mps
            velocity_y_mps = resolved.velocity_y_mps
        forward = np.asarray(
            [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float32
        )
        velocity = np.asarray(
            [
                velocity_x_mps if velocity_x_mps is not None else 0.0,
                velocity_y_mps if velocity_y_mps is not None else 0.0,
            ],
            dtype=np.float32,
        )
        forward_speed_mps = float(np.dot(velocity, forward))
        if (
            getattr(self, "last_step_static_barrier_collision", False)
            and not command.handbrake
            and command.brake <= 0.01
            and not command.stop
            and state.speed_mps * forward_speed_mps > 0.0
        ):
            retained_speed_mps = (
                abs(state.speed_mps)
                * self._taxi_vehicle.curb_forward_momentum_retention
            )
            if abs(forward_speed_mps) < retained_speed_mps:
                forward_speed_mps = math.copysign(retained_speed_mps, state.speed_mps)
        resolved = replace(
            resolved,
            yaw_rad=state.yaw_rad,
            yaw_rate_radps=state.yaw_rate_radps,
            speed_mps=forward_speed_mps,
            velocity_x_mps=float(velocity[0]),
            velocity_y_mps=float(velocity[1]),
        )
        self.synchronize_ego_state(resolved)
        return resolved, samples

    def debug_frame(self, state: VehicleState) -> PhysicsDebugFrame:
        """Capture topology with the pre-policy PhysX contact pose for the ego."""
        debug = super().debug_frame(state)
        contact_state = getattr(self, "_last_contact_resolved_state", None)
        if contact_state is None:
            return debug
        half_yaw = contact_state.yaw_rad * 0.5
        return replace(
            debug,
            ego_position_m=np.asarray(
                [
                    contact_state.x_m,
                    contact_state.y_m,
                    contact_state.z_m + self._ego_model.half_extents_m[2],
                ],
                dtype=np.float32,
            ),
            ego_orientation_xyzw=np.asarray(
                [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
                dtype=np.float32,
            ),
        )

    def step(
        self,
        state: VehicleState,
        timestamp_us: int,
        dt_s: float,
    ) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
        """Resolve a commandless compatibility step with Taxi heading policy."""
        return self.step_with_command(
            state,
            DriverCommand(),
            timestamp_us,
            dt_s,
        )


def step_taxi_physics_world(
    physics_world: GamePhysicsWorld,
    state: VehicleState,
    command: DriverCommand,
    timestamp_us: int,
    dt_s: float,
) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
    """Advance one Taxi-only command-aware physics step."""
    if not isinstance(physics_world, TaxiPhysicsWorld):
        raise TypeError("Taxi physics step requires TaxiPhysicsWorld")
    return physics_world.step_with_command(state, command, timestamp_us, dt_s)
