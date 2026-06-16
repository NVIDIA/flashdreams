# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic waypoint replay for scene-editor drive trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.input.waypoint_trajectory import (
    Waypoint,
    WaypointTrajectory,
)
from omnidreams.interactive_drive.math3d import rig_pose_from_state
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.types import (
    DriverCommand,
    TrajectoryChunk,
    VehicleState,
)


@dataclass(frozen=True)
class _RouteSample:
    waypoint: Waypoint
    yaw_rad: float
    finished: bool


class InterpolatedWaypointTrajectorySimulation:
    """Simulation backend that samples ego poses directly from a waypoint route.

    This is intended for scene-editor recordings, where the HDMap raster should
    match the editor's frame slider preview exactly: distance along the route is
    ``speed_mps * frame / fps`` rather than the result of a closed-loop vehicle
    controller chasing lookahead targets.
    """

    def __init__(
        self,
        *,
        initial_state: VehicleState,
        trajectory: WaypointTrajectory,
        vehicle_config: VehicleConfig,
        ground_snapper: GroundSnapper | None,
        initial_timestamp_us: int,
        map_bounds: MapBounds | None = None,
        oob_margin_m: float = 50.0,
        oob_warning_zone_m: float = 100.0,
    ) -> None:
        self._trajectory = trajectory
        self._vehicle_config = vehicle_config
        self._ground_snapper = ground_snapper
        self._next_timestamp_us = int(initial_timestamp_us)
        self._map_bounds = map_bounds
        self._oob_margin_m = float(oob_margin_m)
        self._oob_warning_zone_m = float(oob_warning_zone_m)
        self._frame_index = 0
        self._path_length_m = _path_length(trajectory)
        self._finished = self._path_length_m <= 1e-6 and not trajectory.loop
        self._current_state = self._state_for_distance(
            0.0,
            fallback_pitch_rad=initial_state.pitch_rad,
            fallback_roll_rad=initial_state.roll_rad,
        )
        self._last_proximity = self._compute_proximity(self._current_state)

    @property
    def current_state(self) -> VehicleState:
        return self._current_state

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def last_proximity(self) -> float:
        return self._last_proximity

    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        del command
        if extrapolation_offset_s != 0.0:
            raise NotImplementedError(
                "Nonzero extrapolation_offset_s is not implemented for waypoint replay."
            )

        timestamps = np.array(
            [
                self._next_timestamp_us
                + frame_idx * _frame_interval_us(frame_interval_s)
                for frame_idx in range(chunk_size)
            ],
            dtype=np.int64,
        )
        poses = np.zeros((chunk_size, 4, 4), dtype=np.float32)

        state = self._current_state
        for frame_idx in range(chunk_size):
            absolute_frame = self._frame_index + frame_idx
            distance_m = self._trajectory.speed_mps * absolute_frame * frame_interval_s
            state = self._state_for_distance(
                distance_m,
                fallback_pitch_rad=state.pitch_rad,
                fallback_roll_rad=state.roll_rad,
            )
            poses[frame_idx] = rig_pose_from_state(
                x_m=state.x_m,
                y_m=state.y_m,
                z_m=state.z_m,
                yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad,
                roll_rad=state.roll_rad,
            )

        self._frame_index += chunk_size
        self._next_timestamp_us = int(
            timestamps[-1] + _frame_interval_us(frame_interval_s)
        )
        self._current_state = state
        self._last_proximity = self._compute_proximity(state)
        return TrajectoryChunk(
            timestamps_us=timestamps,
            rig_poses_world=poses,
            boundary_state_after_chunk=state,
        )

    def _state_for_distance(
        self,
        distance_m: float,
        *,
        fallback_pitch_rad: float,
        fallback_roll_rad: float,
    ) -> VehicleState:
        sample = _sample_route(self._trajectory, distance_m, self._path_length_m)
        self._finished = self._finished or sample.finished
        speed_mps = (
            0.0
            if sample.finished and self._trajectory.stop_at_end
            else self._trajectory.speed_mps
        )
        state = VehicleState(
            x_m=sample.waypoint.x_m,
            y_m=sample.waypoint.y_m,
            z_m=sample.waypoint.z_m,
            yaw_rad=sample.yaw_rad,
            speed_mps=speed_mps,
            steer_rad=0.0,
            pitch_rad=fallback_pitch_rad,
            roll_rad=fallback_roll_rad,
        )
        if self._ground_snapper is not None:
            state = self._ground_snapper.snap(state, self._vehicle_config)
        return state

    def _compute_proximity(self, state: VehicleState) -> float:
        if self._map_bounds is None:
            return 0.0
        return self._map_bounds.proximity(
            (state.x_m, state.y_m),
            margin_m=self._oob_margin_m,
            warning_zone_m=self._oob_warning_zone_m,
        )


def _frame_interval_us(frame_interval_s: float) -> int:
    return int(round(frame_interval_s * 1_000_000.0))


def _path_length(trajectory: WaypointTrajectory) -> float:
    return sum(_distance(start, end) for start, end in _segments(trajectory))


def _segments(trajectory: WaypointTrajectory) -> list[tuple[Waypoint, Waypoint]]:
    waypoints = trajectory.waypoints
    count = len(waypoints) if trajectory.loop else len(waypoints) - 1
    return [
        (waypoints[index % len(waypoints)], waypoints[(index + 1) % len(waypoints)])
        for index in range(count)
    ]


def _distance(a: Waypoint, b: Waypoint) -> float:
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def _interpolate(a: Waypoint, b: Waypoint, t: float) -> Waypoint:
    return Waypoint(
        x_m=a.x_m * (1.0 - t) + b.x_m * t,
        y_m=a.y_m * (1.0 - t) + b.y_m * t,
        z_m=a.z_m * (1.0 - t) + b.z_m * t,
    )


def _segment_yaw(a: Waypoint, b: Waypoint) -> float:
    return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)


def _sample_route(
    trajectory: WaypointTrajectory,
    distance_m: float,
    path_length_m: float,
) -> _RouteSample:
    segments = _segments(trajectory)
    if not segments:
        return _RouteSample(trajectory.waypoints[0], 0.0, finished=True)

    if path_length_m <= 1e-6:
        start, end = segments[0]
        return _RouteSample(start, _segment_yaw(start, end), finished=not trajectory.loop)

    finished = False
    route_distance_m = max(0.0, float(distance_m))
    if trajectory.loop:
        route_distance_m = route_distance_m % path_length_m
    elif route_distance_m >= path_length_m:
        route_distance_m = path_length_m
        finished = True

    walked = 0.0
    last_nonzero_segment = segments[0]
    for start, end in segments:
        length = _distance(start, end)
        if length <= 1e-6:
            continue
        last_nonzero_segment = (start, end)
        if route_distance_m <= walked + length:
            t = (route_distance_m - walked) / length
            return _RouteSample(
                _interpolate(start, end, max(0.0, min(1.0, t))),
                _segment_yaw(start, end),
                finished=finished,
            )
        walked += length

    start, end = last_nonzero_segment
    return _RouteSample(trajectory.waypoints[-1], _segment_yaw(start, end), finished=True)
