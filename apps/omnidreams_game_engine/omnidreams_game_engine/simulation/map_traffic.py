# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime tracks and simple car-following controls for authored map traffic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from ludus_renderer import BodyState, SceneObject

from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.game_map.types import GameMapTrafficVehicle
from omnidreams_game_engine.game_map.vicinity import GameMapVicinity
from omnidreams_game_engine.simulation.actor_controller import (
    ActorControlDecision,
    ActorTrackTarget,
)
from omnidreams_game_engine.simulation.components import rigid_body_model_for_object

_OBJECT_ID_PREFIX = "map-traffic:"
_MIN_CLEARANCE_M = 2.0
_TIME_HEADWAY_S = 1.25
_BRAKING_MARGIN_M = 8.0
_LANE_CORRIDOR_M = 2.25
_MAX_HEADING_DELTA_RAD = math.radians(40.0)
_HEADWAY_GRID_CELL_M = 64.0
_RESTART_AFTER_STOPPED_S = 1.0
_MAX_COLLISION_SETTLING_S = 3.0
_STOPPED_LINEAR_SPEED_MPS = 0.10
_STOPPED_ANGULAR_SPEED_RADPS = 0.10
_RECOVERED_POSITION_ERROR_M = 0.60
_RECOVERED_HEADING_ERROR_RAD = math.radians(8.0)
_RECOVERED_VELOCITY_ERROR_MPS = 0.75
_TRACK_LOOKAHEAD_S = 0.35


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MapTrafficPhase(str, Enum):
    """Authoritative gameplay phase for a map traffic vehicle."""

    TRAVERSING = "traversing"
    COLLISION = "collision"
    RECOVERING = "recovering"


# Compatibility name for callers and tests written before external gameplay
# actor controllers were supported.
MapTrafficDecision = ActorControlDecision


@dataclass
class MapTrafficVehicleState:
    """Single gameplay-owned state for one map traffic vehicle."""

    object_id: str
    scene_object: SceneObject
    timestamp_us: float
    duration_us: int
    route_segment_index: int
    max_speed_mps: float
    route_element_ids: tuple[str, ...]
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity_mps: np.ndarray
    angular_velocity_radps: np.ndarray
    velocity_scale: float = 1.0
    phase: MapTrafficPhase = MapTrafficPhase.TRAVERSING
    stopped_duration_s: float = 0.0
    collision_duration_s: float = 0.0

    @property
    def decision(self) -> ActorControlDecision:
        """Return control outputs derived solely from the gameplay phase."""
        return ActorControlDecision(
            drive_enabled=self.phase is not MapTrafficPhase.COLLISION,
            detached_from_track=self.phase is MapTrafficPhase.COLLISION,
        )

    @property
    def element_id(self) -> str:
        """Return the semantic element occupied by the logical route pose."""
        segment = int(
            np.searchsorted(
                self.scene_object.timestamps_us, int(self.timestamp_us), side="right"
            )
            - 1
        )
        return self.route_element_ids[
            min(max(segment, 0), len(self.route_element_ids) - 1)
        ]


@dataclass(frozen=True)
class _TrafficObservation:
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    half_length_m: float


@dataclass(frozen=True)
class _RouteProjection:
    timestamp_us: float
    segment_index: int
    distance_sq: float
    progress_distance: float


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
    """Own route, collision, recovery, and physical snapshots for map traffic."""

    def __init__(
        self,
        traffic: tuple[GameMapTrafficVehicle, ...],
        vehicle: VehicleConfig,
    ) -> None:
        self._ego_half_length_m = vehicle.aabb_length_m * 0.5
        states: list[MapTrafficVehicleState] = []
        for definition in traffic:
            scene_object, start_timestamp_us, duration_us = _route_track(
                definition, vehicle
            )
            position, orientation, velocity = scene_object.sample(start_timestamp_us)
            states.append(
                MapTrafficVehicleState(
                    object_id=scene_object.object_id,
                    scene_object=scene_object,
                    timestamp_us=float(start_timestamp_us),
                    duration_us=duration_us,
                    route_segment_index=self._route_segment_index(
                        scene_object, start_timestamp_us
                    ),
                    max_speed_mps=float(np.max(definition.speed_limits_mps)),
                    route_element_ids=definition.route_element_ids,
                    position_m=position.copy(),
                    orientation_xyzw=orientation.copy(),
                    linear_velocity_mps=velocity.copy(),
                    angular_velocity_radps=np.zeros(3, dtype=np.float32),
                )
            )
        self._states = tuple(states)
        self._states_by_id = {state.object_id: state for state in states}
        self._active_ids: frozenset[str] = frozenset()

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        """Return every procedural traffic object owned by this controller."""
        return tuple(state.scene_object for state in self._states)

    @property
    def active_objects(self) -> tuple[SceneObject, ...]:
        """Return only traffic objects selected for the current map vicinity."""
        return tuple(
            state.scene_object
            for state in self._states
            if state.object_id in self._active_ids
        )

    @property
    def active_object_ids(self) -> frozenset[str]:
        """Return traffic IDs selected for PhysX and renderer conditioning."""
        return self._active_ids

    @property
    def active_timestamps_us(self) -> dict[str, int]:
        """Return logical track timestamps used to initialize newly active bodies."""
        return {
            object_id: int(self._states_by_id[object_id].timestamp_us)
            for object_id in self._active_ids
        }

    @property
    def object_ids(self) -> frozenset[str]:
        """Return stable IDs used to retain procedural actors across windows."""
        return frozenset(self._states_by_id)

    @property
    def max_drive_speeds_mps(self) -> dict[str, float]:
        """Return per-object actuator caps derived from compiled route speeds."""
        return {state.object_id: state.max_speed_mps for state in self._states}

    def state(self, object_id: str) -> MapTrafficVehicleState | None:
        """Return the authoritative state for a map NPC, if owned here."""
        return self._states_by_id.get(object_id)

    @staticmethod
    def _route_segment_index(scene_object: SceneObject, timestamp_us: float) -> int:
        segment_index = int(
            np.searchsorted(scene_object.timestamps_us, int(timestamp_us), side="right")
            - 1
        )
        return min(max(segment_index, 0), len(scene_object.timestamps_us) - 2)

    @staticmethod
    def _set_route_snapshot(state: MapTrafficVehicleState) -> None:
        state.route_segment_index = MapTrafficController._route_segment_index(
            state.scene_object, state.timestamp_us
        )
        position, orientation, velocity = state.scene_object.sample(
            int(state.timestamp_us)
        )
        state.position_m = position.copy()
        state.orientation_xyzw = orientation.copy()
        state.linear_velocity_mps = velocity.copy()
        state.angular_velocity_radps = np.zeros(3, dtype=np.float32)

    @classmethod
    def _reset_offscreen(cls, state: MapTrafficVehicleState) -> None:
        state.phase = MapTrafficPhase.TRAVERSING
        state.stopped_duration_s = 0.0
        state.collision_duration_s = 0.0
        state.velocity_scale = 1.0
        cls._set_route_snapshot(state)

    def set_vicinity(self, vicinity: GameMapVicinity | None) -> bool:
        """Select nearby cars and reset displaced cars once the player leaves."""
        visible_elements = (
            frozenset() if vicinity is None else vicinity.traffic_element_ids
        )
        for state in self._states:
            if (
                state.phase is not MapTrafficPhase.TRAVERSING
                and state.element_id not in visible_elements
            ):
                self._reset_offscreen(state)
        active_ids = frozenset(
            state.object_id
            for state in self._states
            if state.element_id in visible_elements
        )
        for object_id in active_ids - self._active_ids:
            self._set_route_snapshot(self._states_by_id[object_id])
        changed = active_ids != self._active_ids
        self._active_ids = active_ids
        return changed

    @staticmethod
    def _drive_target_timestamp_us(state: MapTrafficVehicleState) -> int:
        """Return a bounded actuator target derived from physical route progress."""
        if state.phase is not MapTrafficPhase.TRAVERSING:
            return int(state.timestamp_us)
        lookahead_us = _TRACK_LOOKAHEAD_S * 1_000_000.0 * state.velocity_scale
        return int((state.timestamp_us + lookahead_us) % state.duration_us)

    def _observation(self, state: MapTrafficVehicleState) -> _TrafficObservation:
        if state.object_id in self._active_ids:
            position = state.position_m[:2]
            _, _, velocity = state.scene_object.sample(int(state.timestamp_us))
        else:
            position, _, velocity = state.scene_object.sample(int(state.timestamp_us))
            position = position[:2]
        return _TrafficObservation(
            position_xy=np.asarray(position, dtype=np.float32),
            velocity_xy=np.asarray(velocity[:2], dtype=np.float32),
            half_length_m=float(state.scene_object.model.half_extents_m[0]),
        )

    @staticmethod
    def _cyclic_timestamp_distance(
        timestamp_us: float, reference_us: float, duration_us: int
    ) -> float:
        delta = abs(timestamp_us - reference_us) % duration_us
        return min(delta, duration_us - delta)

    @classmethod
    def _route_projection(
        cls,
        state: MapTrafficVehicleState,
        position_xy: np.ndarray,
        segment_index: int,
    ) -> _RouteProjection:
        positions = state.scene_object.positions_m[:, :2]
        timestamps = state.scene_object.timestamps_us
        start = positions[segment_index]
        segment = positions[segment_index + 1] - start
        length_sq = float(np.dot(segment, segment))
        alpha = 0.0
        if length_sq > 1.0e-12:
            alpha = float(np.dot(position_xy - start, segment) / length_sq)
            alpha = min(max(alpha, 0.0), 1.0)
        projection = start + alpha * segment
        offset = position_xy - projection
        timestamp_us = (
            float(
                timestamps[segment_index]
                + alpha * (timestamps[segment_index + 1] - timestamps[segment_index])
            )
            % state.duration_us
        )
        return _RouteProjection(
            timestamp_us=timestamp_us,
            segment_index=segment_index,
            distance_sq=float(np.dot(offset, offset)),
            progress_distance=cls._cyclic_timestamp_distance(
                timestamp_us, state.timestamp_us, state.duration_us
            ),
        )

    @staticmethod
    def _projection_is_better(
        candidate: _RouteProjection, current: _RouteProjection
    ) -> bool:
        return candidate.distance_sq < current.distance_sq - 1.0e-8 or (
            abs(candidate.distance_sq - current.distance_sq) <= 1.0e-8
            and candidate.progress_distance < current.progress_distance
        )

    @classmethod
    def _nearest_local_route_projection(
        cls, state: MapTrafficVehicleState, position_xy: np.ndarray
    ) -> _RouteProjection:
        """Walk from the route cursor to the nearest adjacent segment."""
        segment_count = len(state.scene_object.positions_m) - 1
        best = cls._route_projection(
            state, position_xy, state.route_segment_index % segment_count
        )
        visited = {best.segment_index}
        while len(visited) < segment_count:
            neighbor_indices = (
                (best.segment_index - 1) % segment_count,
                (best.segment_index + 1) % segment_count,
            )
            neighbors = tuple(
                cls._route_projection(state, position_xy, segment_index)
                for segment_index in neighbor_indices
                if segment_index not in visited
            )
            visited.update(projection.segment_index for projection in neighbors)
            better = tuple(
                projection
                for projection in neighbors
                if cls._projection_is_better(projection, best)
            )
            if not better:
                break
            best = min(
                better,
                key=lambda projection: (
                    projection.distance_sq,
                    projection.progress_distance,
                ),
            )
        return best

    @classmethod
    def _nearest_route_projection(
        cls, state: MapTrafficVehicleState, position_xy: np.ndarray
    ) -> _RouteProjection:
        """Search the full route when collision recovery needs reacquisition."""
        best = cls._route_projection(state, position_xy, 0)
        for segment_index in range(1, len(state.scene_object.positions_m) - 1):
            candidate = cls._route_projection(state, position_xy, segment_index)
            if cls._projection_is_better(candidate, best):
                best = candidate
        return best

    @staticmethod
    def _apply_route_projection(
        state: MapTrafficVehicleState, projection: _RouteProjection
    ) -> None:
        state.timestamp_us = projection.timestamp_us
        state.route_segment_index = projection.segment_index

    @staticmethod
    def _is_recovered(state: MapTrafficVehicleState, body: BodyState) -> bool:
        track_position, track_orientation, track_velocity = state.scene_object.sample(
            int(state.timestamp_us)
        )
        track_velocity = track_velocity * state.velocity_scale
        heading_error = math.atan2(
            math.sin(
                _yaw_from_quaternion_xyzw(track_orientation)
                - _yaw_from_quaternion_xyzw(body.orientation_xyzw)
            ),
            math.cos(
                _yaw_from_quaternion_xyzw(track_orientation)
                - _yaw_from_quaternion_xyzw(body.orientation_xyzw)
            ),
        )
        return (
            float(np.linalg.norm(track_position[:2] - body.position_m[:2]))
            <= _RECOVERED_POSITION_ERROR_M
            and abs(heading_error) <= _RECOVERED_HEADING_ERROR_RAD
            and float(np.linalg.norm(track_velocity[:2] - body.linear_velocity_mps[:2]))
            <= _RECOVERED_VELOCITY_ERROR_MPS
        )

    def observe_physics(
        self,
        object_id: str,
        *,
        struck: bool,
        body: BodyState,
        dt_s: float,
    ) -> MapTrafficDecision | None:
        """Update one NPC from PhysX and advance its collision state machine."""
        state = self._states_by_id.get(object_id)
        if state is None:
            return None
        state.position_m = body.position_m.copy()
        state.orientation_xyzw = body.orientation_xyzw.copy()
        state.linear_velocity_mps = body.linear_velocity_mps.copy()
        state.angular_velocity_radps = body.angular_velocity_radps.copy()

        if struck:
            state.phase = MapTrafficPhase.COLLISION
            state.stopped_duration_s = 0.0
            state.collision_duration_s = 0.0
            return state.decision

        if state.phase is MapTrafficPhase.COLLISION:
            state.collision_duration_s += dt_s
            linear_speed = float(np.linalg.norm(body.linear_velocity_mps[:2]))
            angular_speed = float(np.linalg.norm(body.angular_velocity_radps))
            stopped = (
                linear_speed <= _STOPPED_LINEAR_SPEED_MPS
                and angular_speed <= _STOPPED_ANGULAR_SPEED_RADPS
            )
            state.stopped_duration_s = (
                state.stopped_duration_s + dt_s if stopped else 0.0
            )
            if (
                state.stopped_duration_s >= _RESTART_AFTER_STOPPED_S
                or state.collision_duration_s >= _MAX_COLLISION_SETTLING_S
            ):
                self._apply_route_projection(
                    state,
                    self._nearest_route_projection(state, body.position_m[:2]),
                )
                state.phase = MapTrafficPhase.RECOVERING
                state.stopped_duration_s = 0.0
                state.collision_duration_s = 0.0
                state.velocity_scale = 1.0
        elif state.phase is MapTrafficPhase.RECOVERING and self._is_recovered(
            state, body
        ):
            state.phase = MapTrafficPhase.TRAVERSING

        return state.decision

    @staticmethod
    def _grid_cell(position_xy: np.ndarray) -> tuple[int, int]:
        return (
            math.floor(float(position_xy[0]) / _HEADWAY_GRID_CELL_M),
            math.floor(float(position_xy[1]) / _HEADWAY_GRID_CELL_M),
        )

    def _headway_scale(
        self,
        observation: _TrafficObservation,
        candidates: tuple[_TrafficObservation, ...],
    ) -> float:
        velocity = observation.velocity_xy
        speed_mps = float(np.linalg.norm(velocity[:2]))
        if speed_mps <= 1.0e-4:
            return 0.0
        forward = velocity[:2] / speed_mps
        best_clearance = math.inf
        for other in candidates:
            if other is observation:
                continue
            delta = other.position_xy - observation.position_xy
            longitudinal = float(np.dot(delta, forward))
            if longitudinal <= 0.0:
                continue
            lateral = abs(float(forward[0] * delta[1] - forward[1] * delta[0]))
            if lateral > _LANE_CORRIDOR_M:
                continue
            other_speed = float(np.linalg.norm(other.velocity_xy))
            if other_speed > 1.0e-4:
                other_heading = other.velocity_xy / other_speed
                angle = math.acos(
                    float(np.clip(np.dot(forward, other_heading), -1.0, 1.0))
                )
                if angle > _MAX_HEADING_DELTA_RAD:
                    continue
            clearance = longitudinal - observation.half_length_m - other.half_length_m
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

    def prepare_topology(self, ego: BodyState) -> None:
        """Keep the stable authored traffic set unchanged between steps."""
        del ego

    def prepare_step(
        self,
        ego: BodyState,
        dt_s: float,
    ) -> tuple[ActorTrackTarget, ...]:
        """Advance logical cars and return targets for active physical bodies."""
        for state in self._states:
            if state.phase is MapTrafficPhase.TRAVERSING:
                if state.object_id in self._active_ids:
                    self._apply_route_projection(
                        state,
                        self._nearest_local_route_projection(
                            state, state.position_m[:2]
                        ),
                    )
                else:
                    state.timestamp_us = (
                        state.timestamp_us + dt_s * 1_000_000.0 * state.velocity_scale
                    ) % state.duration_us
                    state.route_segment_index = self._route_segment_index(
                        state.scene_object, state.timestamp_us
                    )
        observations = {
            state.object_id: self._observation(state) for state in self._states
        }
        ego_observation = _TrafficObservation(
            position_xy=np.asarray(ego.position_m[:2], dtype=np.float32),
            velocity_xy=np.asarray(ego.linear_velocity_mps[:2], dtype=np.float32),
            half_length_m=self._ego_half_length_m,
        )
        buckets: dict[tuple[int, int], list[_TrafficObservation]] = {}
        for observation in (*observations.values(), ego_observation):
            buckets.setdefault(self._grid_cell(observation.position_xy), []).append(
                observation
            )
        for state in self._states:
            observation = observations[state.object_id]
            cell_x, cell_y = self._grid_cell(observation.position_xy)
            candidates = tuple(
                candidate
                for offset_x in (-1, 0, 1)
                for offset_y in (-1, 0, 1)
                for candidate in buckets.get((cell_x + offset_x, cell_y + offset_y), ())
            )
            state.velocity_scale = (
                0.0
                if state.phase is MapTrafficPhase.COLLISION
                else self._headway_scale(observation, candidates)
            )
        targets = tuple(
            ActorTrackTarget(
                object_id=state.object_id,
                timestamp_us=self._drive_target_timestamp_us(state),
                velocity_scale=state.velocity_scale,
            )
            for state in self._states
            if state.object_id in self._active_ids
        )
        return targets


__all__ = [
    "MapTrafficController",
    "MapTrafficDecision",
    "MapTrafficPhase",
    "MapTrafficVehicleState",
]
