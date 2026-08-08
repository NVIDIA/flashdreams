# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-game state, waypoint generation, and HUD projection helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from omnidreams.interactive_drive.math3d import invert_transform, rig_pose_from_state
from omnidreams.interactive_drive.types import TrajectoryChunk, VehicleState

if TYPE_CHECKING:
    from omnidreams.interactive_drive.config import BevConfig

TaxiPhase = Literal["seeking_pickup", "to_dropoff"]
TaxiEvent = Literal["fare_complete", "time_expired"]


@dataclass(frozen=True)
class TaxiGameConfig:
    """Configuration for the overlay-only taxi game."""

    enabled: bool = False
    """Whether taxi-game state and HUD overlays are active."""

    seed: int = 0
    """User-controlled seed mixed with the stable scene identifier."""

    waypoint_spacing_m: float = 10.0
    """Arc-length spacing between candidates sampled from the reference route."""

    pickup_min_distance_m: float = 20.0
    """Minimum straight-line distance from the ego to a newly selected pickup."""

    pickup_radius_m: float = 5.0
    """Distance at which the ego collects a pickup."""

    dropoff_radius_m: float = 6.0
    """Distance at which the ego completes a dropoff."""

    fare_min_route_distance_m: float = 40.0
    """Preferred minimum reference-route distance between fare endpoints."""

    fare_max_route_distance_m: float = 120.0
    """Preferred maximum reference-route distance between fare endpoints."""

    target_speed_mps: float = 10.0
    """Nominal travel speed used to derive the fare deadline."""

    grace_s: float = 8.0
    """Fixed time added to the distance-derived fare deadline."""

    min_time_s: float = 12.0
    """Minimum fare deadline."""

    max_time_s: float = 45.0
    """Maximum fare deadline."""

    base_fare_points: int = 100
    """Points awarded for every successful fare."""

    bonus_points_per_second: int = 10
    """Additional points awarded per whole second remaining."""

    event_banner_s: float = 2.0
    """Simulation-time duration of completion and failure banners."""


@dataclass(frozen=True)
class TaxiGameSnapshot:
    """Immutable taxi-game state published to HUD consumers."""

    phase: TaxiPhase
    """Current pickup or dropoff phase."""

    target_xyz_m: tuple[float, float, float]
    """Active target position in scene world coordinates."""

    distance_m: float
    """Straight-line XY distance from the ego to the active target."""

    relative_bearing_rad: float
    """Target bearing relative to ego heading; positive angles point left."""

    remaining_time_s: float | None
    """Dropoff time remaining, or ``None`` while seeking a pickup."""

    score: int
    """Total points earned during the current rollout."""

    event: TaxiEvent | None = None
    """Most recent fare result while its banner remains visible."""

    awarded_points: int = 0
    """Points awarded by the visible completion event."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the snapshot."""
        return {
            "phase": self.phase,
            "target_xyz_m": list(self.target_xyz_m),
            "distance_m": self.distance_m,
            "relative_bearing_rad": self.relative_bearing_rad,
            "remaining_time_s": self.remaining_time_s,
            "score": self.score,
            "event": self.event,
            "awarded_points": self.awarded_points,
        }


@dataclass(frozen=True)
class _Waypoint:
    xyz_m: npt.NDArray[np.float32]
    """World-space waypoint position."""

    route_distance_m: float
    """Arc distance from the start of the reference route."""


def _stable_seed(scene_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{scene_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resample_route(
    route_world: npt.NDArray[np.float32], spacing_m: float, offset_m: float
) -> tuple[_Waypoint, ...]:
    route = np.asarray(route_world, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] != 3:
        raise ValueError(
            "Taxi reference route must have shape [N, 3], "
            f"got {route.shape}."
        )
    if spacing_m <= 0.0:
        raise ValueError("Taxi waypoint spacing must be positive.")
    if len(route) < 2:
        raise ValueError("Taxi mode requires at least two reference-route poses.")

    segment_lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-4))
    route = route[keep]
    if len(route) < 2:
        raise ValueError("Taxi reference route has no usable travel distance.")

    segment_lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(
        np.float32
    )
    total_distance = float(cumulative[-1])
    sample_distances = np.arange(offset_m, total_distance + 1e-6, spacing_m)
    if len(sample_distances) < 2:
        sample_distances = np.array([0.0, total_distance], dtype=np.float32)

    waypoints: list[_Waypoint] = []
    for distance in sample_distances:
        right = int(np.searchsorted(cumulative, distance, side="right"))
        right = min(max(1, right), len(route) - 1)
        left = right - 1
        span = float(cumulative[right] - cumulative[left])
        alpha = 0.0 if span <= 1e-6 else (float(distance) - cumulative[left]) / span
        xyz = ((1.0 - alpha) * route[left] + alpha * route[right]).astype(np.float32)
        waypoints.append(_Waypoint(xyz_m=xyz, route_distance_m=float(distance)))
    if len(waypoints) < 2:
        raise ValueError("Taxi mode requires at least two distinct route waypoints.")
    return tuple(waypoints)


def normalize_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)``."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def relative_target_bearing_rad(
    ego_x_m: float,
    ego_y_m: float,
    ego_yaw_rad: float,
    target_x_m: float,
    target_y_m: float,
) -> float:
    """Return the target bearing relative to ego heading."""
    world_bearing = math.atan2(target_y_m - ego_y_m, target_x_m - ego_x_m)
    return normalize_angle_rad(world_bearing - ego_yaw_rad)


def project_target_to_bev(
    target_xyz_m: tuple[float, float, float],
    vehicle_state: VehicleState,
    bev: BevConfig,
) -> tuple[float, float, bool]:
    """Project a world target into normalized BEV image coordinates.

    Returns:
        Horizontal coordinate, vertical coordinate, and whether the point is
        inside the BEV camera frustum.
    """
    theta = math.radians(float(bev.tilt_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    sensor_to_rig = np.array(
        [
            [sin_t, 0.0, cos_t, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-cos_t, 0.0, sin_t, float(bev.height_m)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rig_to_world = rig_pose_from_state(
        x_m=vehicle_state.x_m,
        y_m=vehicle_state.y_m,
        z_m=vehicle_state.z_m,
        yaw_rad=vehicle_state.yaw_rad,
        pitch_rad=vehicle_state.pitch_rad,
        roll_rad=vehicle_state.roll_rad,
    )
    world_to_sensor = invert_transform(rig_to_world @ sensor_to_rig)
    target_h = np.array([*target_xyz_m, 1.0], dtype=np.float32)
    target_sensor_flu = (world_to_sensor @ target_h)[:3]
    depth = float(target_sensor_flu[0])
    if depth <= 1e-5:
        return 0.5, 0.5, False

    focal = (float(bev.height) / 2.0) / math.tan(
        math.radians(float(bev.fov_deg)) / 2.0
    )
    u_px = float(bev.width) / 2.0 - focal * float(target_sensor_flu[1]) / depth
    v_px = float(bev.height) / 2.0 - focal * float(target_sensor_flu[2]) / depth
    u = u_px / float(bev.width)
    v = v_px / float(bev.height)
    return u, v, 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


class TaxiGameController:
    """Advance deterministic taxi fares over a scene reference route."""

    def __init__(
        self,
        *,
        scene_id: str,
        reference_route_world: npt.NDArray[np.float32],
        initial_state: VehicleState,
        config: TaxiGameConfig,
    ) -> None:
        self._config = config
        self._rng = np.random.default_rng(_stable_seed(scene_id, config.seed))
        offset = float(self._rng.uniform(0.0, config.waypoint_spacing_m))
        self._waypoints = _resample_route(
            reference_route_world, config.waypoint_spacing_m, offset
        )
        self._phase: TaxiPhase = "seeking_pickup"
        self._score = 0
        self._remaining_time_s: float | None = None
        self._event: TaxiEvent | None = None
        self._event_remaining_s = 0.0
        self._awarded_points = 0
        self._pickup_index: int | None = None
        self._dropoff_index: int | None = None
        self._target_index = self._select_pickup(
            initial_state.x_m, initial_state.y_m, excluded=frozenset()
        )

    @property
    def config(self) -> TaxiGameConfig:
        """Return the immutable game configuration."""
        return self._config

    def advance(self, trajectory: TrajectoryChunk, frame_interval_s: float) -> None:
        """Advance game state over every simulated pose in a chunk.

        Args:
            trajectory: Authoritative simulated poses for the requested chunk.
            frame_interval_s: Simulation duration represented by each pose.
        """
        if frame_interval_s < 0.0:
            raise ValueError("Taxi frame interval must be non-negative.")
        for pose in trajectory.rig_poses_world:
            self._advance_banner(frame_interval_s)
            x_m = float(pose[0, 3])
            y_m = float(pose[1, 3])
            target = self._waypoints[self._target_index]
            distance = math.hypot(
                float(target.xyz_m[0]) - x_m, float(target.xyz_m[1]) - y_m
            )
            if self._phase == "seeking_pickup":
                if distance <= self._config.pickup_radius_m:
                    self._start_fare(x_m, y_m)
                continue

            if distance <= self._config.dropoff_radius_m:
                self._complete_fare(x_m, y_m)
                continue
            assert self._remaining_time_s is not None
            self._remaining_time_s = max(
                0.0, self._remaining_time_s - frame_interval_s
            )
            if self._remaining_time_s <= 0.0:
                self._expire_fare(x_m, y_m)

    def snapshot(self, vehicle_state: VehicleState) -> TaxiGameSnapshot:
        """Return the HUD snapshot relative to the supplied ego state."""
        target = self._waypoints[self._target_index].xyz_m
        distance = math.hypot(
            float(target[0]) - vehicle_state.x_m,
            float(target[1]) - vehicle_state.y_m,
        )
        bearing = relative_target_bearing_rad(
            vehicle_state.x_m,
            vehicle_state.y_m,
            vehicle_state.yaw_rad,
            float(target[0]),
            float(target[1]),
        )
        return TaxiGameSnapshot(
            phase=self._phase,
            target_xyz_m=(float(target[0]), float(target[1]), float(target[2])),
            distance_m=distance,
            relative_bearing_rad=bearing,
            remaining_time_s=self._remaining_time_s,
            score=self._score,
            event=self._event if self._event_remaining_s > 0.0 else None,
            awarded_points=(
                self._awarded_points if self._event_remaining_s > 0.0 else 0
            ),
        )

    def _select_pickup(
        self, x_m: float, y_m: float, *, excluded: frozenset[int]
    ) -> int:
        distances = [
            math.hypot(float(point.xyz_m[0]) - x_m, float(point.xyz_m[1]) - y_m)
            for point in self._waypoints
        ]
        eligible = [
            index
            for index, distance in enumerate(distances)
            if index not in excluded
            and distance >= self._config.pickup_min_distance_m
        ]
        if eligible:
            return min(eligible, key=distances.__getitem__)
        fallback = [index for index in range(len(self._waypoints)) if index not in excluded]
        if not fallback:
            fallback = list(range(len(self._waypoints)))
        return max(fallback, key=distances.__getitem__)

    def _select_dropoff(self, pickup_index: int) -> tuple[int, float]:
        pickup_distance = self._waypoints[pickup_index].route_distance_m
        route_distances = [
            abs(point.route_distance_m - pickup_distance) for point in self._waypoints
        ]
        eligible = [
            index
            for index, distance in enumerate(route_distances)
            if index != pickup_index
            and self._config.fare_min_route_distance_m
            <= distance
            <= self._config.fare_max_route_distance_m
        ]
        if eligible:
            index = int(self._rng.choice(eligible))
        else:
            index = max(
                (i for i in range(len(self._waypoints)) if i != pickup_index),
                key=route_distances.__getitem__,
            )
        return index, route_distances[index]

    def _start_fare(self, x_m: float, y_m: float) -> None:
        del x_m, y_m
        self._pickup_index = self._target_index
        self._dropoff_index, route_distance = self._select_dropoff(
            self._pickup_index
        )
        self._target_index = self._dropoff_index
        self._phase = "to_dropoff"
        raw_time = route_distance / max(self._config.target_speed_mps, 1e-6)
        raw_time += self._config.grace_s
        self._remaining_time_s = float(
            np.clip(raw_time, self._config.min_time_s, self._config.max_time_s)
        )

    def _complete_fare(self, x_m: float, y_m: float) -> None:
        assert self._remaining_time_s is not None
        awarded = self._config.base_fare_points + (
            math.floor(self._remaining_time_s) * self._config.bonus_points_per_second
        )
        self._score += awarded
        self._set_event("fare_complete", awarded)
        self._activate_next_pickup(x_m, y_m)

    def _expire_fare(self, x_m: float, y_m: float) -> None:
        self._set_event("time_expired", 0)
        self._activate_next_pickup(x_m, y_m)

    def _activate_next_pickup(self, x_m: float, y_m: float) -> None:
        excluded = frozenset(
            index
            for index in (self._pickup_index, self._dropoff_index)
            if index is not None
        )
        self._target_index = self._select_pickup(x_m, y_m, excluded=excluded)
        self._phase = "seeking_pickup"
        self._remaining_time_s = None

    def _set_event(self, event: TaxiEvent, awarded_points: int) -> None:
        self._event = event
        self._awarded_points = awarded_points
        self._event_remaining_s = self._config.event_banner_s

    def _advance_banner(self, frame_interval_s: float) -> None:
        self._event_remaining_s = max(
            0.0, self._event_remaining_s - frame_interval_s
        )

