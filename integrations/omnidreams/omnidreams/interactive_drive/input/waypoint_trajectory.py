# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams.interactive_drive.input.backend import SampledInput
from omnidreams.interactive_drive.types import DriverCommand, VehicleState

TRAJECTORY_SCHEMA = "clipgt-waypoint-trajectory-v1"
_DIRECT_STEER_RAD = 0.4


@dataclass(frozen=True)
class Waypoint:
    x_m: float
    y_m: float
    z_m: float = 0.0


@dataclass(frozen=True)
class WaypointTrajectory:
    waypoints: tuple[Waypoint, ...]
    speed_mps: float = 4.0
    lookahead_m: float = 6.0
    waypoint_tolerance_m: float = 2.0
    stop_at_end: bool = True
    loop: bool = False
    name: str = "drive-trajectory"


@dataclass(frozen=True)
class _PathProjection:
    segment_index: int
    segment_t: float
    distance_m: float


def load_waypoint_trajectory(path: Path) -> WaypointTrajectory:
    """Load a scene-editor waypoint trajectory JSON file."""
    resolved = path.expanduser().resolve()
    doc = json.loads(resolved.read_text(encoding="utf-8"))
    schema = doc.get("schema")
    if schema not in (None, TRAJECTORY_SCHEMA):
        raise ValueError(
            f"Unsupported trajectory schema {schema!r}; expected {TRAJECTORY_SCHEMA!r}"
        )
    raw_waypoints = doc.get("waypoints") or []
    waypoints = tuple(
        _parse_waypoint(item, index) for index, item in enumerate(raw_waypoints)
    )
    if len(waypoints) < 2:
        raise ValueError("A drive trajectory requires at least two waypoints")

    speed_mps = _positive_float(doc.get("speed_mps", 4.0), "speed_mps")
    lookahead_m = _positive_float(doc.get("lookahead_m", 6.0), "lookahead_m")
    tolerance_m = _positive_float(
        doc.get("waypoint_tolerance_m", max(1.5, min(4.0, lookahead_m * 0.35))),
        "waypoint_tolerance_m",
    )
    return WaypointTrajectory(
        waypoints=waypoints,
        speed_mps=speed_mps,
        lookahead_m=lookahead_m,
        waypoint_tolerance_m=tolerance_m,
        stop_at_end=bool(doc.get("stop_at_end", True)),
        loop=bool(doc.get("loop", False)),
        name=str(doc.get("name") or "drive-trajectory"),
    )


class WaypointTrajectoryInputBackend:
    """Input backend that drives the ego vehicle along a waypoint polyline."""

    def __init__(
        self,
        trajectory: WaypointTrajectory,
        *,
        state_provider: Callable[[], VehicleState],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if len(trajectory.waypoints) < 2:
            raise ValueError("A drive trajectory requires at least two waypoints")
        self._trajectory = trajectory
        self._state_provider = state_provider
        self._clock = clock
        self._finished = False

    @property
    def trajectory(self) -> WaypointTrajectory:
        return self._trajectory

    @property
    def finished(self) -> bool:
        return self._finished

    def sample(self) -> SampledInput:
        state = self._state_provider()
        command = self._command_for_state(state)
        return SampledInput(command=command, sample_time=self._clock())

    def _command_for_state(self, state: VehicleState) -> DriverCommand:
        projection = _closest_projection(self._trajectory, state.x_m, state.y_m)
        remaining_m = _remaining_distance_to_end(self._trajectory, projection)
        if (
            not self._trajectory.loop
            and remaining_m <= self._trajectory.waypoint_tolerance_m
        ):
            self._finished = True

        target = (
            self._trajectory.waypoints[-1]
            if self._finished
            else _lookahead_point(self._trajectory, projection)
        )
        steer = _steer_to_target(state, target)
        target_speed = _target_speed(self._trajectory, remaining_m, self._finished)
        throttle, brake, stop = _pedal_command(state.speed_mps, target_speed)
        return DriverCommand(
            throttle=throttle,
            brake=brake,
            steer=steer,
            stop=stop,
            reverse=False,
            steer_is_direct=True,
            manual_control=False,
        )


class WaypointTrajectoryPlaybackInputBackend:
    """Neutral input backend used when the simulation replays poses directly."""

    def __init__(
        self,
        *,
        finished_provider: Callable[[], bool],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._finished_provider = finished_provider
        self._clock = clock

    @property
    def finished(self) -> bool:
        return bool(self._finished_provider())

    def sample(self) -> SampledInput:
        return SampledInput(command=DriverCommand(), sample_time=self._clock())


def _parse_waypoint(raw: Any, index: int) -> Waypoint:
    try:
        if isinstance(raw, dict):
            x_m = float(raw["x"])
            y_m = float(raw["y"])
            z_m = float(raw.get("z", 0.0))
        else:
            x_m = float(raw[0])
            y_m = float(raw[1])
            z_m = float(raw[2]) if len(raw) > 2 else 0.0
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Waypoint {index + 1} must contain numeric x/y values"
        ) from exc
    if not all(math.isfinite(value) for value in (x_m, y_m, z_m)):
        raise ValueError(f"Waypoint {index + 1} contains a non-finite value")
    return Waypoint(x_m=x_m, y_m=y_m, z_m=z_m)


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _segment_count(trajectory: WaypointTrajectory) -> int:
    return (
        len(trajectory.waypoints) if trajectory.loop else len(trajectory.waypoints) - 1
    )


def _segment(
    trajectory: WaypointTrajectory, segment_index: int
) -> tuple[Waypoint, Waypoint]:
    waypoints = trajectory.waypoints
    start = waypoints[segment_index % len(waypoints)]
    end = waypoints[(segment_index + 1) % len(waypoints)]
    return start, end


def _distance(a: Waypoint, b: Waypoint) -> float:
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def _interpolate(a: Waypoint, b: Waypoint, t: float) -> Waypoint:
    return Waypoint(
        x_m=a.x_m * (1.0 - t) + b.x_m * t,
        y_m=a.y_m * (1.0 - t) + b.y_m * t,
        z_m=a.z_m * (1.0 - t) + b.z_m * t,
    )


def _closest_projection(
    trajectory: WaypointTrajectory, x_m: float, y_m: float
) -> _PathProjection:
    best = _PathProjection(segment_index=0, segment_t=0.0, distance_m=math.inf)
    for segment_index in range(_segment_count(trajectory)):
        start, end = _segment(trajectory, segment_index)
        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-8:
            continue
        t = ((x_m - start.x_m) * dx + (y_m - start.y_m) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        projected_x = start.x_m + dx * t
        projected_y = start.y_m + dy * t
        distance_m = math.hypot(x_m - projected_x, y_m - projected_y)
        if distance_m < best.distance_m:
            best = _PathProjection(
                segment_index=segment_index,
                segment_t=t,
                distance_m=distance_m,
            )
    if math.isfinite(best.distance_m):
        return best
    return _PathProjection(segment_index=0, segment_t=0.0, distance_m=0.0)


def _remaining_distance_to_end(
    trajectory: WaypointTrajectory, projection: _PathProjection
) -> float:
    if trajectory.loop:
        return math.inf
    remaining = 0.0
    for segment_index in range(projection.segment_index, _segment_count(trajectory)):
        start, end = _segment(trajectory, segment_index)
        length = _distance(start, end)
        if segment_index == projection.segment_index:
            remaining += length * (1.0 - projection.segment_t)
        else:
            remaining += length
    return remaining


def _lookahead_point(
    trajectory: WaypointTrajectory, projection: _PathProjection
) -> Waypoint:
    remaining = trajectory.lookahead_m
    if trajectory.loop:
        length = _path_length(trajectory)
        if length > 1e-6:
            remaining = remaining % length

    segment_index = projection.segment_index
    t = projection.segment_t
    for _ in range(max(1, _segment_count(trajectory) + 1)):
        start, end = _segment(trajectory, segment_index)
        length = _distance(start, end)
        if length <= 1e-6:
            segment_index = (segment_index + 1) % max(1, _segment_count(trajectory))
            t = 0.0
            continue
        available = length * (1.0 - t)
        if remaining <= available:
            return _interpolate(start, end, t + remaining / length)
        remaining -= available
        if not trajectory.loop and segment_index >= _segment_count(trajectory) - 1:
            return trajectory.waypoints[-1]
        segment_index = (segment_index + 1) % _segment_count(trajectory)
        t = 0.0
    return trajectory.waypoints[-1]


def _path_length(trajectory: WaypointTrajectory) -> float:
    return sum(
        _distance(*_segment(trajectory, index))
        for index in range(_segment_count(trajectory))
    )


def _steer_to_target(state: VehicleState, target: Waypoint) -> float:
    target_angle = math.atan2(target.y_m - state.y_m, target.x_m - state.x_m)
    heading_error = _normalize_angle(target_angle - state.yaw_rad)
    return max(-1.0, min(1.0, heading_error / _DIRECT_STEER_RAD))


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _target_speed(
    trajectory: WaypointTrajectory, remaining_m: float, finished: bool
) -> float:
    if finished and trajectory.stop_at_end:
        return 0.0
    if trajectory.loop or not trajectory.stop_at_end:
        return trajectory.speed_mps
    slow_zone_m = max(trajectory.speed_mps * 2.0, trajectory.waypoint_tolerance_m * 3.0)
    if remaining_m >= slow_zone_m:
        return trajectory.speed_mps
    return min(trajectory.speed_mps, max(0.0, remaining_m / 2.0))


def _pedal_command(
    speed_mps: float, target_speed_mps: float
) -> tuple[float, float, bool]:
    if target_speed_mps <= 0.05:
        return 0.0, 1.0, abs(speed_mps) < 0.4

    speed_error = target_speed_mps - max(0.0, speed_mps)
    if speed_error >= 0.0:
        throttle = min(1.0, speed_error / max(1.0, target_speed_mps * 0.5))
        if throttle < 0.12 and speed_mps < target_speed_mps:
            throttle = 0.12
        return throttle, 0.0, False
    return 0.0, min(1.0, -speed_error / 3.0), False
