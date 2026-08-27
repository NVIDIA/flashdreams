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
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.contracts import GameUpdate
from omnidreams_game_engine.game_map.vicinity import (
    GameMapVicinity,
    GameMapVicinityResolver,
)
from omnidreams_game_engine.math3d import (
    extract_yaw_from_transform,
    invert_transform,
    level_rig_pose_from_vehicle_state,
    rig_pose_from_state,
    rig_pose_from_vehicle_state,
)
from omnidreams_game_engine.types import (
    CameraCalibration,
    TrajectoryChunk,
    VehicleState,
)

from crazy_robotaxi.dynamics import (
    TaxiVehicleConfig,
)
from crazy_robotaxi.high_scores import (
    HighScoreEntry,
    HighScoreStore,
    default_high_scores_path,
)
from crazy_robotaxi.navigation import (
    LanePosition,
    NavigationFareRegion,
    NavigationLane,
    NavigationWaypoint,
    RoutePlan,
    TaxiNavigationMap,
)

if TYPE_CHECKING:
    from omnidreams_game_engine.config import BevConfig

TaxiPhase = Literal["seeking_pickup", "to_dropoff"]
TaxiEvent = Literal["pickup_complete", "fare_complete", "time_expired"]
TaxiSessionState = Literal["playing", "awaiting_name", "leaderboard"]


@dataclass(frozen=True)
class TaxiGameConfig:
    """Configuration for Crazy Robotaxi rules and presentation."""

    vehicle: TaxiVehicleConfig = TaxiVehicleConfig()
    """Taxi-only control and vehicle-dynamics configuration."""

    seed: int | None = None
    """Debug seed mixed with the scene ID; ``None`` uses fresh entropy."""

    waypoint_spacing_m: float = 10.0
    """Arc-length spacing between candidates sampled from each navigation route."""

    pickup_grid_spacing_m: float = 60.0
    """Grid spacing used to distribute simultaneous pickup points across the map."""

    pickup_min_distance_m: float = 20.0
    """Minimum straight-line distance from the ego to a newly selected pickup."""

    initial_pickup_max_distance_m: float = 200.0
    """Maximum preferred distance to the camera-visible initial pickup."""

    pickup_radius_m: float = 5.0
    """Distance at which the ego collects a pickup."""

    dropoff_radius_m: float = 6.0
    """Distance at which the ego completes a dropoff."""

    fare_min_route_distance_m: float = 200.0
    """Preferred minimum routed distance between fare endpoints."""

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

    ground_snap_max_absolute_rotation_deg: float = 10.0
    """Maximum ground rotation accepted by the taxi ground snapper."""

    ground_snap_settle_fraction: float = 0.25
    """Fraction of stale ground attitude removed after an invalid sample."""

    def __post_init__(self) -> None:
        """Validate Taxi-only values at configuration time."""
        if self.pickup_grid_spacing_m <= 0.0:
            raise ValueError("pickup_grid_spacing_m must be positive")


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

    pickup_targets_xyz_m: tuple[tuple[float, float, float], ...] = ()
    """All pickup positions available during the pickup phase."""

    pickup_passengers_xyz_m: tuple[tuple[float, float, float], ...] = ()
    """Waiting-passenger ground positions aligned with the pickup targets."""

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
            "pickup_targets_xyz_m": [
                list(target) for target in self.pickup_targets_xyz_m
            ],
            "pickup_passengers_xyz_m": [
                list(target) for target in self.pickup_passengers_xyz_m
            ],
        }


@dataclass(frozen=True)
class TaxiCameraMarkerProjection:
    """Projected world-marker geometry in camera image pixels."""

    anchor_uv: tuple[float, float]
    """Exact image location of the active waypoint."""

    beacon_top_uv: tuple[float, float] | None
    """Projected top of the vertical beacon, when visible."""

    ring_edges_uv: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    """Visible line segments forming the target's activation-radius ring."""

    distance_m: float
    """Horizontal distance from the displayed camera pose to the target."""


def _stable_seed(scene_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{scene_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


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
    rig_to_world = level_rig_pose_from_vehicle_state(vehicle_state)
    return project_target_pose_to_bev(target_xyz_m, rig_to_world, bev)


def project_target_pose_to_bev(
    target_xyz_m: tuple[float, float, float],
    rig_to_world: npt.NDArray[np.float32],
    bev: BevConfig,
) -> tuple[float, float, bool]:
    """Project a target using the exact rig pose that produced a BEV image."""
    world_to_sensor = _bev_world_to_sensor(rig_to_world, bev)
    target_h = np.array([*target_xyz_m, 1.0], dtype=np.float32)
    target_sensor_flu = (world_to_sensor @ target_h)[:3]
    projected = _project_bev_sensor_point(target_sensor_flu, bev)
    if projected is None:
        return 0.5, 0.5, False
    u, v = projected
    return u, v, 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def project_target_pose_to_bev_edge(
    target_xyz_m: tuple[float, float, float],
    rig_to_world: npt.NDArray[np.float32],
    bev: BevConfig,
) -> tuple[float, float] | None:
    """Project a target direction to the edge of a pose-aligned BEV."""
    target_u, target_v, _visible = project_target_pose_to_bev(
        target_xyz_m, rig_to_world, bev
    )
    delta_u = target_u - 0.5
    delta_v = target_v - 0.5
    extent = max(abs(delta_u), abs(delta_v))
    if not math.isfinite(extent) or extent <= 1.0e-9:
        return None
    scale = 0.5 / extent
    return 0.5 + delta_u * scale, 0.5 + delta_v * scale


def project_segment_pose_to_bev(
    segment_world: npt.NDArray[np.float32],
    rig_to_world: npt.NDArray[np.float32],
    bev: BevConfig,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Project and viewport-clip one world-space enclosure segment."""
    segment = np.asarray(segment_world, dtype=np.float32)
    if segment.shape != (2, 3) or not np.isfinite(segment).all():
        raise ValueError("BEV segment must have finite shape (2, 3).")
    world_to_sensor = _bev_world_to_sensor(rig_to_world, bev)
    homogeneous = np.concatenate((segment, np.ones((2, 1), dtype=np.float32)), axis=1)
    sensor_points = (world_to_sensor @ homogeneous.T).T[:, :3]
    near_depth = 1.0e-5
    depths = sensor_points[:, 0]
    if bool(np.all(depths <= near_depth)):
        return None
    if bool(np.any(depths <= near_depth)):
        behind = int(np.argmin(depths))
        ahead = 1 - behind
        span = float(depths[ahead] - depths[behind])
        if span <= 0.0:
            return None
        alpha = (near_depth - float(depths[behind])) / span
        sensor_points[behind] = sensor_points[behind] + alpha * (
            sensor_points[ahead] - sensor_points[behind]
        )
    projected = tuple(_project_bev_sensor_point(point, bev) for point in sensor_points)
    if projected[0] is None or projected[1] is None:
        return None
    return _clip_normalized_segment(projected[0], projected[1])


def _bev_world_to_sensor(
    rig_to_world: npt.NDArray[np.float32], bev: BevConfig
) -> npt.NDArray[np.float32]:
    leveled_rig_to_world = rig_pose_from_state(
        float(rig_to_world[0, 3]),
        float(rig_to_world[1, 3]),
        float(rig_to_world[2, 3]),
        extract_yaw_from_transform(rig_to_world),
    )
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
    return invert_transform(leveled_rig_to_world @ sensor_to_rig)


def _project_bev_sensor_point(
    point_sensor_flu: npt.NDArray[np.float32], bev: BevConfig
) -> tuple[float, float] | None:
    depth = float(point_sensor_flu[0])
    if depth <= 1e-5:
        return None

    focal = (float(bev.height) / 2.0) / math.tan(math.radians(float(bev.fov_deg)) / 2.0)
    u_px = float(bev.width) / 2.0 - focal * float(point_sensor_flu[1]) / depth
    v_px = float(bev.height) / 2.0 - focal * float(point_sensor_flu[2]) / depth
    return u_px / float(bev.width), v_px / float(bev.height)


def _clip_normalized_segment(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a 2D segment to the unit square with Liang-Barsky."""
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, 1.0 - x0), (-dy, y0), (dy, 1.0 - y0)):
        if abs(p) <= 1.0e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return (
        (x0 + lower * dx, y0 + lower * dy),
        (x0 + upper * dx, y0 + upper * dy),
    )


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
    projections = _project_taxi_targets_to_camera(
        (snapshot.target_xyz_m,),
        target_radius_m=snapshot.target_radius_m,
        rig_to_world=rig_to_world,
        camera_model=camera_model,
        image_width=image_width,
        image_height=image_height,
        ring_samples=ring_samples,
        beacon_height_m=beacon_height_m,
    )
    return projections[0] if projections else None


def _project_taxi_targets_to_camera(
    targets_xyz_m: tuple[tuple[float, float, float], ...],
    *,
    target_radius_m: float,
    rig_to_world: npt.NDArray[np.float32],
    camera_model: FThetaCameraModel,
    image_width: int,
    image_height: int,
    ring_samples: int = 32,
    beacon_height_m: float = 3.5,
) -> tuple[TaxiCameraMarkerProjection, ...]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Taxi camera image dimensions must be positive.")
    if ring_samples < 3:
        raise ValueError("Taxi target ring requires at least three samples.")
    if not targets_xyz_m:
        return ()

    targets = np.asarray(targets_xyz_m, dtype=np.float32)
    angles = np.linspace(
        0.0, 2.0 * math.pi, ring_samples, endpoint=False, dtype=np.float32
    )
    rings = np.repeat(targets[:, None, :], ring_samples, axis=1)
    rings[:, :, 0] += np.float32(target_radius_m) * np.cos(angles)
    rings[:, :, 1] += np.float32(target_radius_m) * np.sin(angles)
    beacons = targets + np.asarray([0.0, 0.0, beacon_height_m], dtype=np.float32)
    points = np.concatenate(
        (targets[:, None, :], beacons[:, None, :], rings),
        axis=1,
    )
    uv, _depth, forward = camera_model.project_world(
        points.reshape(-1, 3),
        rig_to_world,
    )
    uv = uv.reshape(len(targets), ring_samples + 2, 2)
    forward = forward.reshape(len(targets), ring_samples + 2)
    inside = (
        forward
        & (uv[:, :, 0] >= 0.0)
        & (uv[:, :, 0] < float(image_width))
        & (uv[:, :, 1] >= 0.0)
        & (uv[:, :, 1] < float(image_height))
    )

    projections = []
    ego_x = float(rig_to_world[0, 3])
    ego_y = float(rig_to_world[1, 3])
    for target_index, target in enumerate(targets):
        if not bool(inside[target_index, 0]):
            continue
        target_uv = uv[target_index]
        target_inside = inside[target_index]
        ring_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for ring_index in range(ring_samples):
            left = 2 + ring_index
            right = 2 + ((ring_index + 1) % ring_samples)
            if bool(target_inside[left] and target_inside[right]):
                ring_edges.append(
                    (
                        (float(target_uv[left, 0]), float(target_uv[left, 1])),
                        (float(target_uv[right, 0]), float(target_uv[right, 1])),
                    )
                )
        projections.append(
            TaxiCameraMarkerProjection(
                anchor_uv=(float(target_uv[0, 0]), float(target_uv[0, 1])),
                beacon_top_uv=(
                    (float(target_uv[1, 0]), float(target_uv[1, 1]))
                    if bool(target_inside[1])
                    else None
                ),
                ring_edges_uv=tuple(ring_edges),
                distance_m=math.hypot(
                    float(target[0]) - ego_x,
                    float(target[1]) - ego_y,
                ),
            )
        )
    return tuple(projections)


def project_taxi_markers_to_camera(
    snapshot: TaxiGameSnapshot,
    rig_to_world: npt.NDArray[np.float32],
    camera_model: FThetaCameraModel,
    *,
    image_width: int,
    image_height: int,
) -> tuple[TaxiCameraMarkerProjection, ...]:
    """Project the nearest three visible pickups or the active dropoff."""
    if snapshot.phase == "seeking_pickup" and snapshot.pickup_targets_xyz_m:
        targets = _nearest_visible_pickup_targets(
            snapshot.pickup_targets_xyz_m,
            rig_to_world,
            camera_model,
            image_width=image_width,
            image_height=image_height,
        )
    else:
        targets = (snapshot.target_xyz_m,)
    return _project_taxi_targets_to_camera(
        targets,
        target_radius_m=snapshot.target_radius_m,
        rig_to_world=rig_to_world,
        camera_model=camera_model,
        image_width=image_width,
        image_height=image_height,
    )


def _nearest_visible_pickup_targets(
    targets_xyz_m: tuple[tuple[float, float, float], ...],
    rig_to_world: npt.NDArray[np.float32],
    camera_model: FThetaCameraModel,
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float], ...]:
    targets = np.asarray(targets_xyz_m, dtype=np.float32)
    uv, _depth, forward = camera_model.project_world(targets, rig_to_world)
    inside = (
        forward
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
    )
    visible_indices = np.flatnonzero(inside)
    if visible_indices.size == 0:
        return ()
    ego_xy = np.asarray(rig_to_world[:2, 3], dtype=np.float32)
    offsets_xy = targets[:, :2] - ego_xy
    distance_squared = np.einsum("ni,ni->n", offsets_xy, offsets_xy)
    nearest_order = np.argsort(
        distance_squared[visible_indices],
        kind="stable",
    )[:3]
    return tuple(targets_xyz_m[int(index)] for index in visible_indices[nearest_order])


def _xyz_tuple(point: npt.NDArray[np.float32]) -> tuple[float, float, float]:
    return float(point[0]), float(point[1]), float(point[2])


def _passenger_xyz_tuple(
    waypoint: NavigationWaypoint,
) -> tuple[float, float, float]:
    point = (
        waypoint.passenger_xyz_m
        if waypoint.passenger_xyz_m is not None
        else waypoint.xyz_m
    )
    return _xyz_tuple(point)


class TaxiGameController:
    """Advance taxi fares over scene navigation routes."""

    def __init__(
        self,
        *,
        scene_id: str,
        reference_route_world: npt.NDArray[np.float32],
        navigation_routes_world: tuple[npt.NDArray[np.float32], ...] = (),
        navigation_lanes: tuple[NavigationLane, ...] = (),
        fare_regions: tuple[NavigationFareRegion, ...] = (),
        initial_state: VehicleState,
        config: TaxiGameConfig,
        initial_camera: CameraCalibration | None = None,
        high_score_store: HighScoreStore | None = None,
        vicinity_resolver: GameMapVicinityResolver | None = None,
    ) -> None:
        self._config = config
        rng_seed = None if config.seed is None else _stable_seed(scene_id, config.seed)
        self._rng = np.random.default_rng(rng_seed)
        self._vicinity_resolver = vicinity_resolver
        self._vicinity: GameMapVicinity | None = None
        offset = float(self._rng.uniform(0.0, config.waypoint_spacing_m))
        if navigation_lanes:
            self._navigation = TaxiNavigationMap(navigation_lanes)
        else:
            routes_world = navigation_routes_world or (reference_route_world,)
            self._navigation = TaxiNavigationMap.from_polylines(
                routes_world,
                bidirectional=True,
            )
        self._waypoints = self._navigation.sample_waypoints(
            config.waypoint_spacing_m, offset
        ) + self._navigation.sample_fare_regions(
            fare_regions, config.waypoint_spacing_m, self._rng
        )
        self._eligible_waypoint_indices = tuple(range(len(self._waypoints)))
        self._pickup_point_indices = self._sample_pickup_point_indices()
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
        self._target_index, _initial_route = self._select_initial_pickup(
            initial_state,
            initial_camera,
        )
        self._available_pickup_indices = self._pickup_indices(
            initial_state,
            excluded=frozenset(),
        )
        if self._target_index not in self._available_pickup_indices:
            self._available_pickup_indices += (self._target_index,)

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
            name: Valid player name supplied by the V2 UI thread.

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
            if self._phase == "seeking_pickup":
                pickup_index = self._collected_pickup_index(x_m, y_m)
                if pickup_index is not None:
                    self._start_fare(pickup_index, vehicle_state)
            else:
                target = self._waypoints[self._target_index]
                distance = math.hypot(
                    float(target.xyz_m[0]) - x_m,
                    float(target.xyz_m[1]) - y_m,
                )
                if distance <= self._config.dropoff_radius_m:
                    self._complete_fare(vehicle_state)
                else:
                    assert self._remaining_time_s is not None
                    self._remaining_time_s = max(
                        0.0, self._remaining_time_s - frame_interval_s
                    )
                    if self._remaining_time_s <= 0.0:
                        self._expire_fare(vehicle_state)

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
        if self._vicinity_resolver is not None:
            self._vicinity = self._vicinity_resolver.resolve(
                x_m,
                y_m,
                previous=self._vicinity,
            )
        target_index = (
            min(
                self._available_pickup_indices,
                key=lambda index: (
                    math.hypot(
                        float(self._waypoints[index].xyz_m[0]) - x_m,
                        float(self._waypoints[index].xyz_m[1]) - y_m,
                    ),
                    index,
                ),
            )
            if self._phase == "seeking_pickup" and self._available_pickup_indices
            else self._target_index
        )
        target = self._waypoints[target_index].xyz_m
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
        vicinity = self._vicinity
        passenger_indices = tuple(
            index
            for index in self._available_pickup_indices
            if self._waypoints[index].element_id is None
            or (
                vicinity is not None
                and self._waypoints[index].element_id in vicinity.pedestrian_element_ids
            )
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
            pickup_targets_xyz_m=(
                tuple(
                    _xyz_tuple(self._waypoints[index].xyz_m)
                    for index in self._available_pickup_indices
                )
                if self._phase == "seeking_pickup"
                else ()
            ),
            pickup_passengers_xyz_m=(
                tuple(
                    _passenger_xyz_tuple(self._waypoints[index])
                    for index in passenger_indices
                )
                if self._phase == "seeking_pickup"
                else ()
            ),
        )

    def _pickup_indices(
        self,
        vehicle_state: VehicleState,
        *,
        excluded: frozenset[int],
    ) -> tuple[int, ...]:
        """Return every pickup that is available from the current position."""
        _distances, eligible = self._pickup_candidates(
            vehicle_state.x_m,
            vehicle_state.y_m,
            excluded=excluded,
        )
        if eligible:
            return tuple(eligible)
        return tuple(
            index for index in self._pickup_point_indices if index not in excluded
        )

    def _sample_pickup_point_indices(self) -> tuple[int, ...]:
        """Choose one stable pickup point per world-space grid cell."""
        cell_size = self._config.pickup_grid_spacing_m
        candidates_by_cell: dict[tuple[int, int], list[int]] = {}
        for index in self._eligible_waypoint_indices:
            waypoint = self._waypoints[index]
            point = waypoint.xyz_m
            cell = (
                math.floor(float(point[0]) / cell_size),
                math.floor(float(point[1]) / cell_size),
            )
            candidates_by_cell.setdefault(cell, []).append(index)
        selected = tuple(
            sorted(
                min(
                    candidates,
                    key=lambda index: (
                        (
                            float(self._waypoints[index].xyz_m[0])
                            - (cell[0] + 0.5) * cell_size
                        )
                        ** 2
                        + (
                            float(self._waypoints[index].xyz_m[1])
                            - (cell[1] + 0.5) * cell_size
                        )
                        ** 2,
                        index,
                    ),
                )
                for cell, candidates in candidates_by_cell.items()
            )
        )
        if len(selected) >= 2:
            return selected
        return self._eligible_waypoint_indices[:2]

    def _collected_pickup_index(self, x_m: float, y_m: float) -> int | None:
        """Return the closest available pickup inside its activation radius."""
        candidates = (
            (
                math.hypot(
                    float(self._waypoints[index].xyz_m[0]) - x_m,
                    float(self._waypoints[index].xyz_m[1]) - y_m,
                ),
                index,
            )
            for index in self._available_pickup_indices
        )
        distance, index = min(candidates, default=(math.inf, -1))
        return index if distance <= self._config.pickup_radius_m else None

    def _select_pickup(
        self,
        vehicle_state: VehicleState,
        *,
        excluded: frozenset[int],
    ) -> tuple[int, RoutePlan | None]:
        """Choose a reachable pickup and its shortest legal route."""
        if len(excluded) >= len(self._waypoints):
            excluded = frozenset()
        distances, eligible = self._pickup_candidates(
            vehicle_state.x_m,
            vehicle_state.y_m,
            excluded=excluded,
        )
        for source in self._route_sources(vehicle_state):
            route_distances = self._navigation.route_distances(source, self._waypoints)
            pickup_indices = frozenset(self._pickup_point_indices)
            reachable = [
                index
                for index, route_distance in enumerate(route_distances)
                if index in pickup_indices
                and index not in excluded
                and math.isfinite(route_distance)
                and distances[index] > 1.0
            ]
            preferred_candidates = [
                index for index in eligible if index in frozenset(reachable)
            ]
            candidates = preferred_candidates or reachable
            if not candidates:
                continue
            pickup_index = (
                int(self._rng.choice(candidates))
                if preferred_candidates
                else max(candidates, key=distances.__getitem__)
            )
            plan = self._navigation.route(source, self._waypoints[pickup_index])
            if plan is not None:
                return pickup_index, plan
        fallback = [
            index
            for index in self._pickup_point_indices
            if index not in excluded and distances[index] > 1.0
        ]
        if not fallback:
            fallback = [
                index for index in self._pickup_point_indices if distances[index] > 1.0
            ]
        if not fallback:
            fallback = list(self._pickup_point_indices)
        return min(fallback, key=distances.__getitem__), None

    def _select_initial_pickup(
        self,
        initial_state: VehicleState,
        initial_camera: CameraCalibration | None,
    ) -> tuple[int, RoutePlan | None]:
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
                for index in self._pickup_point_indices
                if bool(forward[index])
                and 0.0 <= float(uv[index, 0]) < float(initial_camera.width)
                and 0.0 <= float(uv[index, 1]) < float(initial_camera.height)
            ]
        else:
            visible = [
                index
                for index in self._pickup_point_indices
                for point in (self._waypoints[index],)
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
        ideal_distance_m = self._config.initial_pickup_max_distance_m
        for source in self._route_sources(initial_state):
            route_distances = self._navigation.route_distances(source, self._waypoints)
            reachable = frozenset(
                index
                for index, route_distance in enumerate(route_distances)
                if math.isfinite(route_distance) and distances[index] > 1.0
            )
            candidate_groups = (
                [
                    index
                    for index in visible
                    if index in eligible_set and index in reachable
                ],
                [index for index in visible if index in reachable],
            )
            for candidates in candidate_groups:
                if not candidates:
                    continue
                pickup_index = min(
                    candidates,
                    key=lambda index: (
                        abs(distances[index] - ideal_distance_m),
                        distances[index],
                        index,
                    ),
                )
                plan = self._navigation.route(source, self._waypoints[pickup_index])
                if plan is not None:
                    return pickup_index, plan
        return self._select_pickup(initial_state, excluded=frozenset())

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
            for index in self._pickup_point_indices
            if index not in excluded
            and distances[index] >= self._config.pickup_min_distance_m
        ]
        return distances, eligible

    def _select_dropoff(
        self, pickup_index: int, vehicle_state: VehicleState
    ) -> tuple[int, RoutePlan]:
        """Choose a reachable dropoff and its shortest legal route."""
        sources = self._navigation.nearest_lane_positions(
            vehicle_state.x_m,
            vehicle_state.y_m,
            vehicle_state.yaw_rad,
        )
        pickup = self._waypoints[pickup_index]
        fallback_sources = pickup.departure_anchors or (
            LanePosition(
                lane_index=pickup.lane_index,
                distance_along_lane_m=pickup.distance_along_lane_m,
                lateral_distance_m=0.0,
                heading_error_rad=0.0,
            ),
        )
        source_candidates = (
            fallback_sources
            if pickup.departure_anchors
            else tuple(
                source for source in sources if source.lateral_distance_m <= 12.0
            )
            or fallback_sources
        )

        for source in source_candidates:
            route_distances = self._navigation.route_distances(source, self._waypoints)
            reachable = [
                index
                for index, distance in enumerate(route_distances)
                if index in self._eligible_waypoint_indices
                and index != pickup_index
                and math.isfinite(distance)
                and distance > 1.0
            ]
            if not reachable:
                continue
            preferred = [
                index
                for index in reachable
                if self._config.fare_min_route_distance_m
                <= route_distances[index]
                <= self._config.fare_max_route_distance_m
            ]
            far_enough = [
                index
                for index in reachable
                if route_distances[index] >= self._config.fare_min_route_distance_m
            ]
            dropoff_index = int(self._rng.choice(preferred or far_enough or reachable))
            plan = self._navigation.route(source, self._waypoints[dropoff_index])
            if plan is not None:
                return dropoff_index, plan
        raise RuntimeError("Taxi pickup has no reachable dropoff destination.")

    def _route_sources(self, vehicle_state: VehicleState) -> tuple[LanePosition, ...]:
        """Return nearby heading-compatible route origins."""
        matches = self._navigation.nearest_lane_positions(
            vehicle_state.x_m,
            vehicle_state.y_m,
            vehicle_state.yaw_rad,
        )
        nearby = tuple(
            source for source in matches if source.lateral_distance_m <= 12.0
        )
        return nearby or matches

    def _start_fare(self, pickup_index: int, vehicle_state: VehicleState) -> None:
        self._pickup_index = pickup_index
        self._dropoff_index, route_plan = self._select_dropoff(
            self._pickup_index, vehicle_state
        )
        self._target_index = self._dropoff_index
        self._phase = "to_dropoff"
        self._available_pickup_indices = ()
        raw_time = route_plan.distance_m / max(self._config.target_speed_mps, 1e-6)
        raw_time += self._config.grace_s
        clamped_time = float(
            np.clip(raw_time, self._config.min_time_s, self._config.max_time_s)
        )
        self._remaining_time_s = clamped_time * self._config.trip_time_multiplier
        self._set_event("pickup_complete", 0)

    def _complete_fare(self, vehicle_state: VehicleState) -> None:
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
        self._activate_next_pickup(vehicle_state)

    def _expire_fare(self, vehicle_state: VehicleState) -> None:
        self._set_event("time_expired", 0)
        self._activate_next_pickup(vehicle_state)

    def _activate_next_pickup(self, vehicle_state: VehicleState) -> None:
        excluded = frozenset(
            index
            for index in (self._pickup_index, self._dropoff_index)
            if index is not None
        )
        self._target_index, _pickup_route = self._select_pickup(
            vehicle_state,
            excluded=excluded,
        )
        self._available_pickup_indices = self._pickup_indices(
            vehicle_state,
            excluded=excluded,
        )
        if self._target_index not in self._available_pickup_indices:
            self._available_pickup_indices += (self._target_index,)
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


class TaxiGameRules:
    """Game-engine rules adapter for fares, passengers, and high scores."""

    def __init__(self, controller: TaxiGameController) -> None:
        self.controller = controller

    @property
    def is_running(self) -> bool:
        return self.controller.is_playing

    def snapshot(self, vehicle_state: VehicleState) -> TaxiGameSnapshot:
        return self.controller.snapshot(vehicle_state)

    def advance_frames(
        self,
        trajectory: TrajectoryChunk,
        frame_interval_s: float,
    ) -> GameUpdate:
        from crazy_robotaxi.passengers import build_pickup_passenger_trajectories

        frames = self.controller.advance_frames(trajectory, frame_interval_s)
        passengers = build_pickup_passenger_trajectories(
            frames,
            trajectory.timestamps_us,
        )
        return GameUpdate(frames=frames, dynamic_actors=passengers)

    def submit_text(
        self,
        value: str,
        vehicle_state: VehicleState,
    ) -> TaxiGameSnapshot:
        self.controller.submit_high_score_name(value)
        return self.controller.snapshot(vehicle_state)
