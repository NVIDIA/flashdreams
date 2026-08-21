# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime tracks and simple car-following controls for authored map traffic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ludus_renderer import BodyState, PhysXWorld, SceneObject

from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.game_map.types import GameMapTrafficVehicle
from omnidreams_game_engine.simulation.components import rigid_body_model_for_object

_OBJECT_ID_PREFIX = "map-traffic:"
_MIN_CLEARANCE_M = 2.0
_TIME_HEADWAY_S = 1.25
_BRAKING_MARGIN_M = 8.0
_LANE_CORRIDOR_M = 2.25
_MAX_HEADING_DELTA_RAD = math.radians(40.0)


@dataclass
class _RouteState:
    object_id: str
    scene_object: SceneObject
    timestamp_us: float
    duration_us: int
    max_speed_mps: float
    velocity_scale: float = 1.0


def _route_track(
    traffic: GameMapTrafficVehicle, vehicle: VehicleConfig
) -> tuple[SceneObject, int, int]:
    positions = np.asarray(traffic.centerline_world, dtype=np.float32).copy()
    dimensions = np.asarray(traffic.dimensions_lwh_m, dtype=np.float32)
    positions[:, 2] += dimensions[2] * 0.5
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    segment_speeds = np.maximum(
        np.minimum(traffic.speed_limits_mps[:-1], traffic.speed_limits_mps[1:]),
        np.float32(0.1),
    )
    durations_us = np.maximum(
        np.rint(segment_lengths / segment_speeds * 1_000_000.0).astype(np.int64),
        np.int64(1),
    )
    timestamps_us = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(durations_us, dtype=np.int64))
    )

    tangents = np.diff(positions[:, :2], axis=0)
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    yaw = np.concatenate((yaw, yaw[:1]))
    orientations = np.zeros((len(positions), 4), dtype=np.float32)
    orientations[:, 2] = np.sin(yaw * 0.5)
    orientations[:, 3] = np.cos(yaw * 0.5)

    cumulative_distance = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64))
    )
    start_timestamp_us = int(
        np.interp(
            traffic.start_distance_m,
            cumulative_distance,
            timestamps_us.astype(np.float64),
        )
    )
    object_id = f"{_OBJECT_ID_PREFIX}{traffic.vehicle_id}"
    scene_object = SceneObject(
        object_id=object_id,
        object_type=traffic.vehicle_type,
        model=rigid_body_model_for_object(
            traffic.vehicle_type,
            dimensions,
            restitution=vehicle.collision_restitution,
            friction=vehicle.collision_friction,
        ),
        timestamps_us=timestamps_us,
        positions_m=positions,
        orientations_xyzw=orientations,
    )
    return scene_object, start_timestamp_us, int(timestamps_us[-1])


class MapTrafficController:
    """Advance looping map tracks and maintain same-direction headway."""

    def __init__(
        self,
        traffic: tuple[GameMapTrafficVehicle, ...],
        vehicle: VehicleConfig,
    ) -> None:
        states: list[_RouteState] = []
        for definition in traffic:
            scene_object, start_timestamp_us, duration_us = _route_track(
                definition, vehicle
            )
            states.append(
                _RouteState(
                    object_id=scene_object.object_id,
                    scene_object=scene_object,
                    timestamp_us=float(start_timestamp_us),
                    duration_us=duration_us,
                    max_speed_mps=float(np.max(definition.speed_limits_mps)),
                )
            )
        self._states = tuple(states)
        self._states_by_id = {state.object_id: state for state in states}

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        """Return the procedural traffic objects owned by this controller."""
        return tuple(state.scene_object for state in self._states)

    @property
    def object_ids(self) -> frozenset[str]:
        """Return stable IDs used to retain procedural actors across windows."""
        return frozenset(self._states_by_id)

    @property
    def max_drive_speeds_mps(self) -> dict[str, float]:
        """Return per-object actuator caps derived from compiled route speeds."""
        return {state.object_id: state.max_speed_mps for state in self._states}

    def _target_velocity(self, state: _RouteState) -> np.ndarray:
        _, _, velocity = state.scene_object.sample(int(state.timestamp_us))
        return velocity

    def _headway_scale(
        self,
        state: _RouteState,
        world: PhysXWorld,
        ego: BodyState,
    ) -> float:
        body = world.body_state(state.object_id)
        velocity = self._target_velocity(state)
        speed_mps = float(np.linalg.norm(velocity[:2]))
        if speed_mps <= 1.0e-4:
            return 0.0
        forward = velocity[:2] / speed_mps
        own_half_length = state.scene_object.model.half_extents_m[0]
        best_clearance = math.inf
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = [
            (
                ego.position_m,
                ego.linear_velocity_mps,
                float(world.ego_model.half_extents_m[0]),
            )
        ]
        for other in self._states:
            if other.object_id == state.object_id:
                continue
            other_body = world.body_state(other.object_id)
            candidates.append(
                (
                    other_body.position_m,
                    self._target_velocity(other),
                    other.scene_object.model.half_extents_m[0],
                )
            )
        for position, other_velocity, other_half_length in candidates:
            delta = position[:2] - body.position_m[:2]
            longitudinal = float(np.dot(delta, forward))
            if longitudinal <= 0.0:
                continue
            lateral = abs(float(forward[0] * delta[1] - forward[1] * delta[0]))
            if lateral > _LANE_CORRIDOR_M:
                continue
            other_speed = float(np.linalg.norm(other_velocity[:2]))
            if other_speed > 1.0e-4:
                other_heading = other_velocity[:2] / other_speed
                angle = math.acos(
                    float(np.clip(np.dot(forward, other_heading), -1.0, 1.0))
                )
                if angle > _MAX_HEADING_DELTA_RAD:
                    continue
            clearance = longitudinal - own_half_length - other_half_length
            best_clearance = min(best_clearance, clearance)
        desired_clearance = _MIN_CLEARANCE_M + _TIME_HEADWAY_S * speed_mps
        if best_clearance <= desired_clearance:
            return 0.0
        return float(
            np.clip(
                (best_clearance - desired_clearance) / _BRAKING_MARGIN_M,
                0.0,
                1.0,
            )
        )

    def prepare_step(self, world: PhysXWorld, ego: BodyState, dt_s: float) -> None:
        """Advance route clocks, apply headway, and publish one native batch."""
        for state in self._states:
            state.timestamp_us = (
                state.timestamp_us + dt_s * 1_000_000.0 * state.velocity_scale
            ) % state.duration_us
        for state in self._states:
            state.velocity_scale = self._headway_scale(state, world, ego)
        world.apply_track_progress(
            tuple(
                (state.object_id, int(state.timestamp_us), state.velocity_scale)
                for state in self._states
            )
        )


__all__ = ["MapTrafficController"]
