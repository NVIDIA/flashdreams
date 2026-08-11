# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-game state, waypoint generation, and HUD projection helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from omnidreams.interactive_drive.camera import FThetaCameraModel
from omnidreams.interactive_drive.high_scores import (
    HighScoreEntry,
    HighScoreStore,
    default_high_scores_path,
)
from omnidreams.interactive_drive.math3d import (
    invert_transform,
    level_rig_pose_from_vehicle_state,
    rig_pose_from_vehicle_state,
)
from omnidreams.interactive_drive.types import (
    CameraCalibration,
    TrajectoryChunk,
    VehicleState,
)

if TYPE_CHECKING:
    from omnidreams.interactive_drive.config import BevConfig

TaxiPhase = Literal["seeking_pickup", "to_dropoff"]
TaxiEvent = Literal["pickup_complete", "fare_complete", "time_expired"]
TaxiSessionState = Literal["playing", "awaiting_name", "leaderboard"]


@dataclass(frozen=True)
class TaxiGameConfig:
    """Configuration for the overlay-only taxi game."""

    enabled: bool = False
    """Whether taxi-game state and HUD overlays are active."""

    seed: int | None = None
    """Debug seed mixed with the scene ID; ``None`` uses fresh entropy."""

    waypoint_spacing_m: float = 10.0
    """Arc-length spacing between candidates sampled from each navigation route."""

    pickup_min_distance_m: float = 20.0
    """Minimum straight-line distance from the ego to a newly selected pickup."""

    pickup_radius_m: float = 5.0
    """Distance at which the ego collects a pickup."""

    dropoff_radius_m: float = 6.0
    """Distance at which the ego completes a dropoff."""

    fare_min_route_distance_m: float = 40.0
    """Preferred minimum straight-line distance between fare endpoints."""

    fare_max_route_distance_m: float = 250.0
    """Preferred maximum straight-line distance between fare endpoints."""

    target_speed_mps: float = 10.0
    """Nominal travel speed used to derive the fare deadline."""

    grace_s: float = 8.0
    """Fixed time added to the distance-derived fare deadline."""

    min_time_s: float = 12.0
    """Minimum fare deadline."""

    max_time_s: float = 45.0
    """Maximum fare deadline."""

    trip_time_multiplier: float = 2.0
    """Multiplier applied after deriving and clamping the fare deadline."""

    base_fare_points: int = 500
    """Points awarded for every successful fare."""

    bonus_points_per_second: int = 100
    """Additional points awarded per whole second remaining."""

    event_banner_s: float = 2.0
    """Simulation-time duration of completion and failure banners."""

    global_time_s: float = 60.0
    """Simulation-time duration of a new game."""

    dropoff_time_bonus_s: float = 30.0
    """Global time added after each successful dropoff."""

    high_scores_path: Path = field(default_factory=default_high_scores_path)
    """CSV path used to persist the global top-ten leaderboard."""


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

    target_radius_m: float
    """World-space radius that activates the current target."""

    remaining_time_s: float | None
    """Dropoff time remaining, or ``None`` while seeking a pickup."""

    score: int
    """Total points earned during the current rollout."""

    high_score: int | None = None
    """Best persisted score, or ``None`` when the leaderboard is empty."""

    global_remaining_time_s: float = 0.0
    """Simulation time remaining before the game ends."""

    session_state: TaxiSessionState = "playing"
    """Current play, name-entry, or leaderboard state."""

    leaderboard: tuple[HighScoreEntry, ...] = ()
    """Current top-ten entries after the game ends."""

    high_score_rank: int | None = None
    """Prospective or recorded rank for the finished score."""

    event: TaxiEvent | None = None
    """Most recent fare result while its banner remains visible."""

    awarded_points: int = 0
    """Points awarded by the visible completion event."""

    awarded_global_time_s: float = 0.0
    """Global time awarded by the visible completion event."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the snapshot."""
        return {
            "phase": self.phase,
            "target_xyz_m": list(self.target_xyz_m),
            "distance_m": self.distance_m,
            "relative_bearing_rad": self.relative_bearing_rad,
            "target_radius_m": self.target_radius_m,
            "remaining_time_s": self.remaining_time_s,
            "score": self.score,
            "high_score": self.high_score,
            "global_remaining_time_s": self.global_remaining_time_s,
            "session_state": self.session_state,
            "leaderboard": [entry.as_dict() for entry in self.leaderboard],
            "high_score_rank": self.high_score_rank,
            "event": self.event,
            "awarded_points": self.awarded_points,
            "awarded_global_time_s": self.awarded_global_time_s,
        }


@dataclass(frozen=True)
class _Waypoint:
    xyz_m: npt.NDArray[np.float32]
    """World-space waypoint position."""


@dataclass(frozen=True)
class TaxiCameraMarkerProjection:
    """Projected world-marker geometry in camera image pixels."""

    anchor_uv: tuple[float, float]
    """Exact image location of the active waypoint."""

    beacon_top_uv: tuple[float, float] | None
    """Projected top of the vertical beacon, when visible."""

    ring_edges_uv: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    """Visible line segments forming the target's activation-radius ring."""


def _stable_seed(scene_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{scene_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resample_route(
    route_world: npt.NDArray[np.float32], spacing_m: float, offset_m: float
) -> tuple[_Waypoint, ...]:
    route = np.asarray(route_world, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] != 3:
        raise ValueError(
            f"Taxi reference route must have shape [N, 3], got {route.shape}."
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
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)
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
        waypoints.append(_Waypoint(xyz_m=xyz))
    if len(waypoints) < 2:
        raise ValueError("Taxi mode requires at least two distinct route waypoints.")
    return tuple(waypoints)


def _resample_navigation_routes(
    routes_world: tuple[npt.NDArray[np.float32], ...],
    spacing_m: float,
    offset_m: float,
) -> tuple[_Waypoint, ...]:
    """Resample and spatially deduplicate navigation routes."""
    sampled: list[_Waypoint] = []
    for route in routes_world:
        try:
            sampled.extend(_resample_route(route, spacing_m, offset_m))
        except ValueError:
            continue
    if not sampled:
        if not routes_world:
            raise ValueError("Taxi mode requires navigation geometry.")
        return _resample_route(routes_world[0], spacing_m, offset_m)

    deduplicated: list[_Waypoint] = []
    occupied_cells: set[tuple[int, int]] = set()
    for waypoint in sampled:
        cell = (
            int(round(float(waypoint.xyz_m[0]) * 2.0)),
            int(round(float(waypoint.xyz_m[1]) * 2.0)),
        )
        if cell in occupied_cells:
            continue
        occupied_cells.add(cell)
        deduplicated.append(waypoint)
    if len(deduplicated) < 2:
        raise ValueError("Taxi mode requires at least two distinct road waypoints.")
    return tuple(deduplicated)


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
    rig_to_world = level_rig_pose_from_vehicle_state(vehicle_state)
    world_to_sensor = invert_transform(rig_to_world @ sensor_to_rig)
    target_h = np.array([*target_xyz_m, 1.0], dtype=np.float32)
    target_sensor_flu = (world_to_sensor @ target_h)[:3]
    depth = float(target_sensor_flu[0])
    if depth <= 1e-5:
        return 0.5, 0.5, False

    focal = (float(bev.height) / 2.0) / math.tan(math.radians(float(bev.fov_deg)) / 2.0)
    u_px = float(bev.width) / 2.0 - focal * float(target_sensor_flu[1]) / depth
    v_px = float(bev.height) / 2.0 - focal * float(target_sensor_flu[2]) / depth
    u = u_px / float(bev.width)
    v = v_px / float(bev.height)
    return u, v, 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def project_taxi_marker_to_camera(
    snapshot: TaxiGameSnapshot,
    rig_to_world: npt.NDArray[np.float32],
    camera_model: FThetaCameraModel,
    *,
    image_width: int,
    image_height: int,
    ring_samples: int = 32,
    beacon_height_m: float = 3.5,
) -> TaxiCameraMarkerProjection | None:
    """Project the active taxi target into a camera image.

    Return ``None`` when the target anchor is behind the camera or outside the
    image. This deliberately does not clamp off-screen targets to an edge; the
    always-visible direction arrow already covers that case.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Taxi camera image dimensions must be positive.")
    if ring_samples < 3:
        raise ValueError("Taxi target ring requires at least three samples.")

    target = np.asarray(snapshot.target_xyz_m, dtype=np.float32)
    angles = np.linspace(
        0.0, 2.0 * math.pi, ring_samples, endpoint=False, dtype=np.float32
    )
    ring = np.repeat(target[None, :], ring_samples, axis=0)
    ring[:, 0] += np.float32(snapshot.target_radius_m) * np.cos(angles)
    ring[:, 1] += np.float32(snapshot.target_radius_m) * np.sin(angles)
    points = np.concatenate(
        (
            target[None, :],
            (target + np.array([0.0, 0.0, beacon_height_m], dtype=np.float32))[None, :],
            ring,
        ),
        axis=0,
    )
    uv, _depth, forward = camera_model.project_world(points, rig_to_world)
    inside = (
        forward
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
    )
    if not bool(inside[0]):
        return None

    ring_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index in range(ring_samples):
        left = 2 + index
        right = 2 + ((index + 1) % ring_samples)
        if bool(inside[left] and inside[right]):
            ring_edges.append(
                (
                    (float(uv[left, 0]), float(uv[left, 1])),
                    (float(uv[right, 0]), float(uv[right, 1])),
                )
            )

    return TaxiCameraMarkerProjection(
        anchor_uv=(float(uv[0, 0]), float(uv[0, 1])),
        beacon_top_uv=((float(uv[1, 0]), float(uv[1, 1])) if bool(inside[1]) else None),
        ring_edges_uv=tuple(ring_edges),
    )


class TaxiGameController:
    """Advance taxi fares over scene navigation routes."""

    def __init__(
        self,
        *,
        scene_id: str,
        reference_route_world: npt.NDArray[np.float32],
        navigation_routes_world: tuple[npt.NDArray[np.float32], ...] = (),
        initial_state: VehicleState,
        config: TaxiGameConfig,
        initial_camera: CameraCalibration | None = None,
        high_score_store: HighScoreStore | None = None,
    ) -> None:
        self._config = config
        rng_seed = None if config.seed is None else _stable_seed(scene_id, config.seed)
        self._rng = np.random.default_rng(rng_seed)
        offset = float(self._rng.uniform(0.0, config.waypoint_spacing_m))
        routes_world = navigation_routes_world or (reference_route_world,)
        self._waypoints = _resample_navigation_routes(
            routes_world, config.waypoint_spacing_m, offset
        )
        self._phase: TaxiPhase = "seeking_pickup"
        self._session_state: TaxiSessionState = "playing"
        self._score = 0
        self._global_remaining_time_s = config.global_time_s
        self._remaining_time_s: float | None = None
        self._event: TaxiEvent | None = None
        self._event_remaining_s = 0.0
        self._awarded_points = 0
        self._awarded_global_time_s = 0.0
        self._pickup_index: int | None = None
        self._dropoff_index: int | None = None
        self._high_score_store = high_score_store or HighScoreStore(
            config.high_scores_path
        )
        existing_scores = self._high_score_store.read()
        self._high_score = existing_scores[0].score if existing_scores else None
        self._leaderboard: tuple[HighScoreEntry, ...] = ()
        self._high_score_rank: int | None = None
        self._target_index = self._select_initial_pickup(
            initial_state,
            initial_camera,
        )

    @property
    def config(self) -> TaxiGameConfig:
        """Return the immutable game configuration."""
        return self._config

    @property
    def is_playing(self) -> bool:
        """Return whether driving and simulation should continue."""
        return self._session_state == "playing"

    def submit_high_score_name(self, name: str) -> None:
        """Persist the finished score and transition to the leaderboard.

        Args:
            name: Valid player name supplied by the active presenter.

        Raises:
            RuntimeError: The game is not waiting for a player name.
            ValueError: ``name`` does not satisfy leaderboard validation.
        """
        if self._session_state != "awaiting_name":
            raise RuntimeError("Taxi game is not waiting for a high-score name.")
        inserted, self._leaderboard = self._high_score_store.record(name, self._score)
        self._high_score = (
            self._leaderboard[0].score if self._leaderboard else self._high_score
        )
        self._high_score_rank = (
            next(
                index
                for index, entry in enumerate(self._leaderboard, start=1)
                if entry is inserted
            )
            if inserted is not None
            else None
        )
        self._session_state = "leaderboard"

    def advance(self, trajectory: TrajectoryChunk, frame_interval_s: float) -> None:
        """Advance game state over every simulated pose in a chunk.

        Args:
            trajectory: Authoritative simulated poses for the requested chunk.
            frame_interval_s: Simulation duration represented by each pose.
        """
        self.advance_frames(trajectory, frame_interval_s)

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> tuple[TaxiGameSnapshot, ...]:
        """Advance the game and return state synchronized to every pose."""
        if frame_interval_s < 0.0:
            raise ValueError("Taxi frame interval must be non-negative.")
        snapshots: list[TaxiGameSnapshot] = []
        for vehicle_state in trajectory.vehicle_states:
            x_m = vehicle_state.x_m
            y_m = vehicle_state.y_m
            yaw_rad = vehicle_state.yaw_rad
            if self._session_state != "playing":
                snapshots.append(self._snapshot_for_pose(x_m, y_m, yaw_rad))
                continue
            self._advance_banner(frame_interval_s)
            target = self._waypoints[self._target_index]
            distance = math.hypot(
                float(target.xyz_m[0]) - x_m, float(target.xyz_m[1]) - y_m
            )
            if self._phase == "seeking_pickup":
                if distance <= self._config.pickup_radius_m:
                    self._start_fare(x_m, y_m)
            elif distance <= self._config.dropoff_radius_m:
                self._complete_fare(x_m, y_m)
            else:
                assert self._remaining_time_s is not None
                self._remaining_time_s = max(
                    0.0, self._remaining_time_s - frame_interval_s
                )
                if self._remaining_time_s <= 0.0:
                    self._expire_fare(x_m, y_m)

            self._global_remaining_time_s = max(
                0.0, self._global_remaining_time_s - frame_interval_s
            )
            if self._global_remaining_time_s <= 0.0:
                self._end_game()

            snapshots.append(self._snapshot_for_pose(x_m, y_m, yaw_rad))
        return tuple(snapshots)

    def snapshot(self, vehicle_state: VehicleState) -> TaxiGameSnapshot:
        """Return the HUD snapshot relative to the supplied ego state."""
        return self._snapshot_for_pose(
            vehicle_state.x_m, vehicle_state.y_m, vehicle_state.yaw_rad
        )

    def _snapshot_for_pose(
        self, x_m: float, y_m: float, yaw_rad: float
    ) -> TaxiGameSnapshot:
        target = self._waypoints[self._target_index].xyz_m
        distance = math.hypot(
            float(target[0]) - x_m,
            float(target[1]) - y_m,
        )
        bearing = relative_target_bearing_rad(
            x_m,
            y_m,
            yaw_rad,
            float(target[0]),
            float(target[1]),
        )
        return TaxiGameSnapshot(
            phase=self._phase,
            target_xyz_m=(float(target[0]), float(target[1]), float(target[2])),
            distance_m=distance,
            relative_bearing_rad=bearing,
            target_radius_m=(
                self._config.pickup_radius_m
                if self._phase == "seeking_pickup"
                else self._config.dropoff_radius_m
            ),
            remaining_time_s=self._remaining_time_s,
            score=self._score,
            high_score=self._high_score,
            global_remaining_time_s=self._global_remaining_time_s,
            session_state=self._session_state,
            leaderboard=self._leaderboard,
            high_score_rank=self._high_score_rank,
            event=self._event if self._event_remaining_s > 0.0 else None,
            awarded_points=(
                self._awarded_points if self._event_remaining_s > 0.0 else 0
            ),
            awarded_global_time_s=(
                self._awarded_global_time_s if self._event_remaining_s > 0.0 else 0.0
            ),
        )

    def _select_pickup(
        self,
        x_m: float,
        y_m: float,
        *,
        excluded: frozenset[int],
    ) -> int:
        distances, eligible = self._pickup_candidates(x_m, y_m, excluded=excluded)
        if eligible:
            return int(self._rng.choice(eligible))
        fallback = [
            index for index in range(len(self._waypoints)) if index not in excluded
        ]
        if not fallback:
            fallback = list(range(len(self._waypoints)))
        return max(fallback, key=distances.__getitem__)

    def _select_initial_pickup(
        self,
        initial_state: VehicleState,
        initial_camera: CameraCalibration | None,
    ) -> int:
        """Select the only pickup constrained by the player's initial view."""
        x_m = initial_state.x_m
        y_m = initial_state.y_m
        distances, eligible = self._pickup_candidates(x_m, y_m, excluded=frozenset())

        if initial_camera is not None:
            camera_model = FThetaCameraModel(initial_camera)
            points = np.stack([point.xyz_m for point in self._waypoints])
            uv, _depth, forward = camera_model.project_world(
                points,
                rig_pose_from_vehicle_state(initial_state),
            )
            visible = [
                index
                for index in range(len(self._waypoints))
                if bool(forward[index])
                and 0.0 <= float(uv[index, 0]) < float(initial_camera.width)
                and 0.0 <= float(uv[index, 1]) < float(initial_camera.height)
            ]
        else:
            visible = [
                index
                for index, point in enumerate(self._waypoints)
                if abs(
                    relative_target_bearing_rad(
                        x_m,
                        y_m,
                        initial_state.yaw_rad,
                        float(point.xyz_m[0]),
                        float(point.xyz_m[1]),
                    )
                )
                < math.pi * 0.5
            ]

        eligible_set = frozenset(eligible)
        eligible_visible = [index for index in visible if index in eligible_set]
        if eligible_visible:
            if initial_camera is None:
                return min(eligible_visible, key=distances.__getitem__)
            return int(self._rng.choice(eligible_visible))
        if visible:
            return max(visible, key=distances.__getitem__)
        return self._select_pickup(x_m, y_m, excluded=frozenset())

    def _pickup_candidates(
        self,
        x_m: float,
        y_m: float,
        *,
        excluded: frozenset[int],
    ) -> tuple[list[float], list[int]]:
        """Return distances and valid pickup indices for a vehicle position."""
        distances = [
            math.hypot(float(point.xyz_m[0]) - x_m, float(point.xyz_m[1]) - y_m)
            for point in self._waypoints
        ]
        eligible = [
            index
            for index, distance in enumerate(distances)
            if index not in excluded and distance >= self._config.pickup_min_distance_m
        ]
        return distances, eligible

    def _select_dropoff(self, pickup_index: int) -> tuple[int, float]:
        pickup = self._waypoints[pickup_index].xyz_m
        fare_distances = [
            math.hypot(
                float(point.xyz_m[0] - pickup[0]),
                float(point.xyz_m[1] - pickup[1]),
            )
            for point in self._waypoints
        ]
        eligible = [
            index
            for index, distance in enumerate(fare_distances)
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
                key=fare_distances.__getitem__,
            )
        return index, fare_distances[index]

    def _start_fare(self, x_m: float, y_m: float) -> None:
        del x_m, y_m
        self._pickup_index = self._target_index
        self._dropoff_index, route_distance = self._select_dropoff(self._pickup_index)
        self._target_index = self._dropoff_index
        self._phase = "to_dropoff"
        raw_time = route_distance / max(self._config.target_speed_mps, 1e-6)
        raw_time += self._config.grace_s
        clamped_time = float(
            np.clip(raw_time, self._config.min_time_s, self._config.max_time_s)
        )
        self._remaining_time_s = clamped_time * self._config.trip_time_multiplier
        self._set_event("pickup_complete", 0)

    def _complete_fare(self, x_m: float, y_m: float) -> None:
        assert self._remaining_time_s is not None
        awarded = self._config.base_fare_points + (
            math.floor(self._remaining_time_s) * self._config.bonus_points_per_second
        )
        self._score += awarded
        self._global_remaining_time_s += self._config.dropoff_time_bonus_s
        self._set_event(
            "fare_complete",
            awarded,
            awarded_global_time_s=self._config.dropoff_time_bonus_s,
        )
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

    def _set_event(
        self,
        event: TaxiEvent,
        awarded_points: int,
        *,
        awarded_global_time_s: float = 0.0,
    ) -> None:
        self._event = event
        self._awarded_points = awarded_points
        self._awarded_global_time_s = awarded_global_time_s
        self._event_remaining_s = self._config.event_banner_s

    def _advance_banner(self, frame_interval_s: float) -> None:
        self._event_remaining_s = max(0.0, self._event_remaining_s - frame_interval_s)

    def _end_game(self) -> None:
        self._global_remaining_time_s = 0.0
        self._leaderboard = self._high_score_store.read()
        self._high_score = (
            self._leaderboard[0].score if self._leaderboard else self._high_score
        )
        self._high_score_rank = self._high_score_store.qualifying_rank(self._score)
        self._session_state = (
            "awaiting_name" if self._high_score_rank is not None else "leaderboard"
        )
