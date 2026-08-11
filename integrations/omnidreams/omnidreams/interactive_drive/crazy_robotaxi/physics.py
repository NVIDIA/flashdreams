# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-game policy adapter around the reusable Interactive Drive PhysX world."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
from loguru import logger
from ludus_renderer import RigidBodyModel
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.simulation.components import canonical_object_type
from omnidreams.interactive_drive.simulation.game_physics import GamePhysicsWorld
from omnidreams.interactive_drive.types import DriverCommand, SceneBundle, VehicleState

_MOTOR_TRAFFIC_TYPES = frozenset({"car", "truck", "bus", "trailer"})
_CHASSIS_INSET_M = 0.16


def select_traffic_tracks(
    tracks: tuple[object, ...], density: float, scene_id: str
) -> tuple[object, ...]:
    """Select a stable Taxi-only fraction of motor traffic."""
    if not 0.0 < density <= 1.0:
        raise ValueError("traffic density must be greater than 0 and at most 1")
    if density >= 1.0:
        return tracks
    motor_tracks = tuple(
        track
        for track in tracks
        if canonical_object_type(str(track.object_type)) in _MOTOR_TRAFFIC_TYPES
    )
    if not motor_tracks:
        return tracks
    retained_count = max(1, math.ceil(len(motor_tracks) * density))

    def selection_key(track: object) -> bytes:
        identity = f"{scene_id}:{track.track_id}".encode()
        return hashlib.blake2b(identity, digest_size=8).digest()

    retained_ids = {
        str(track.track_id)
        for track in sorted(motor_tracks, key=selection_key)[:retained_count]
    }
    return tuple(
        track
        for track in tracks
        if canonical_object_type(str(track.object_type)) not in _MOTOR_TRAFFIC_TYPES
        or str(track.track_id) in retained_ids
    )


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
        scene: SceneBundle,
        vehicle: VehicleConfig,
        *,
        traffic_density: float,
    ) -> None:
        selected_tracks = select_traffic_tracks(
            tuple(scene.vehicle_bbox_tracks), traffic_density, scene.scene_id
        )
        taxi_scene = replace(scene, vehicle_bbox_tracks=selected_tracks)
        super().__init__(taxi_scene, vehicle, model_adapter=inset_vehicle_chassis)
        logger.info(
            "[crazy-robotaxi] Taxi physics active: app-authoritative heading, "
            "arcade handbrake, inset chassis, traffic_density={:.2f}",
            traffic_density,
        )

    def step_with_command(
        self,
        state: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
        """Resolve contacts while keeping Taxi drive intent authoritative."""
        resolved, samples = super().step(state, timestamp_us, dt_s)
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
        resolved = replace(
            resolved,
            yaw_rad=state.yaw_rad,
            yaw_rate_radps=state.yaw_rate_radps,
            speed_mps=float(np.dot(velocity, forward)),
            velocity_x_mps=float(velocity[0]),
            velocity_y_mps=float(velocity[1]),
        )
        self.synchronize_ego_state(resolved)
        return resolved, samples

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
