# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Track-backed, map-capable live-edit obstacle events.

Obstacle gameplay is deliberately separate from routed NPC traffic. An event
owns its archetype, placement, scripted motion, lifetime, and (optionally)
physical body. Rendering and PhysX consume that state downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from loguru import logger
from ludus_renderer import BodyState, SceneObject
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.game_map.types import GameMapLane, ResolvedGameMap
from omnidreams_game_engine.simulation.actor_controller import (
    ActorControlDecision,
    ActorTrackTarget,
)
from omnidreams_game_engine.simulation.components import rigid_body_model_for_object
from omnidreams_game_engine.types import (
    DynamicActorTrajectory,
    TrajectoryChunk,
    VehicleState,
)

from crazy_robotaxi.live_edit.config import LiveEditObstacleConfig
from crazy_robotaxi.live_edit.obstacle_templates import (
    ObstacleTemplate,
    ObstacleTemplateCatalog,
    load_obstacle_template_catalog,
)

OBSTACLE_ENTITY_PREFIX = "live-edit-obstacle"
_STATIC_PERSIST_US = 10**13
_MAX_OBSTACLE_DRIVE_SPEED_MPS = 15.0 * 0.44704
"""Upper bound shared with the game engine's non-ego actor controller."""


class ObstaclePhase(str, Enum):
    """Authoritative lifecycle phase for one obstacle event."""

    SCRIPTED = "scripted"
    DETACHED = "detached"
    EXPIRED = "expired"


@dataclass
class ObstacleEvent:
    """One gameplay-owned obstacle and its current scripted/physical state."""

    entity_id: str
    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    translations_world: npt.NDArray[np.float32]
    orientations_xyzw: npt.NDArray[np.float32]
    dimensions_lwh: npt.NDArray[np.float32]
    template_index: int
    """Stable catalog index of the cloned source track."""

    drive_speed_mps: float
    """Per-object PhysX drive cap derived from the sampled trajectory."""

    static: bool = False
    phase: ObstaclePhase = ObstaclePhase.SCRIPTED
    chunks: int = 0
    hit_logged: bool = False
    scene_object: SceneObject | None = None
    logical_timestamp_us: float = 0.0
    physical_position_m: npt.NDArray[np.float32] | None = None
    physical_orientation_xyzw: npt.NDArray[np.float32] | None = None

    def actor(self) -> DynamicActorTrajectory:
        """Return the complete renderer trajectory for a visual-only event."""
        return DynamicActorTrajectory(
            entity_id=self.entity_id,
            object_type=self.object_type,
            timestamps_us=self.timestamps_us,
            translations_world=self.translations_world,
            orientations_xyzw=self.orientations_xyzw,
            dimensions_lwh=self.dimensions_lwh,
            detached_from_track=self.phase is ObstaclePhase.DETACHED,
            is_simulated=True,
        )

    def center_at(self, timestamp_us: int) -> npt.NDArray[np.float32] | None:
        """Return the current physical center or interpolate scripted motion."""
        if self.phase is ObstaclePhase.EXPIRED:
            return None
        if self.physical_position_m is not None:
            return self.physical_position_m.copy()
        if timestamp_us < int(self.timestamps_us[0]) or timestamp_us > int(
            self.timestamps_us[-1]
        ):
            return None
        return np.asarray(
            [
                np.interp(
                    float(timestamp_us),
                    self.timestamps_us,
                    self.translations_world[:, i],
                )
                for i in range(3)
            ],
            dtype=np.float32,
        )

    def orientation_at(self, timestamp_us: int) -> npt.NDArray[np.float32] | None:
        """Return the current physical or nearest scripted orientation."""
        if self.phase is ObstaclePhase.EXPIRED:
            return None
        if self.physical_orientation_xyzw is not None:
            return self.physical_orientation_xyzw.copy()
        if timestamp_us < int(self.timestamps_us[0]) or timestamp_us > int(
            self.timestamps_us[-1]
        ):
            return None
        sample = int(np.argmin(np.abs(self.timestamps_us - np.int64(timestamp_us))))
        return self.orientations_xyzw[sample].copy()


def local_ground_z(
    vertices: npt.NDArray[np.floating] | None,
    xy: npt.NDArray[np.floating],
    radius_m: float = 3.0,
) -> float | None:
    """Return the median nearby ground height, when ground samples exist."""
    if vertices is None:
        return None
    points = np.asarray(vertices)
    near = np.linalg.norm(points[:, :2] - np.asarray(xy)[None, :], axis=1) < radius_m
    if not near.any():
        return None
    return float(np.median(points[near, 2]))


def _yaw_quaternion(yaw_rad: float) -> npt.NDArray[np.float32]:
    return np.asarray(
        [0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)],
        dtype=np.float32,
    )


def _quaternion_yaw(quaternion_xyzw: npt.NDArray[np.floating]) -> float:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quaternion_multiply(
    first_xyzw: npt.NDArray[np.floating],
    second_xyzw: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    ax, ay, az, aw = (float(value) for value in first_xyzw)
    bx, by, bz, bw = (float(value) for value in second_xyzw)
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _angle_delta(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _polyline_lengths(points: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    return np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).astype(np.float64)


def _sample_polyline(
    points: npt.NDArray[np.floating], distance_m: float
) -> tuple[npt.NDArray[np.float32], float]:
    lengths = _polyline_lengths(points)
    total = float(np.sum(lengths))
    distance = min(max(float(distance_m), 0.0), total)
    cumulative = 0.0
    for index, length in enumerate(lengths):
        if cumulative + float(length) >= distance or index == len(lengths) - 1:
            alpha = 0.0 if length <= 1.0e-8 else (distance - cumulative) / float(length)
            position = points[index] + alpha * (points[index + 1] - points[index])
            tangent = points[index + 1, :2] - points[index, :2]
            return np.asarray(position, dtype=np.float32), math.atan2(
                float(tangent[1]), float(tangent[0])
            )
        cumulative += float(length)
    raise AssertionError("non-empty lane polyline was not sampled")


def _nearest_lane_progress(
    game_map: ResolvedGameMap,
    ego_state: VehicleState,
) -> tuple[GameMapLane, float] | None:
    ego_xy = np.asarray([ego_state.x_m, ego_state.y_m], dtype=np.float64)
    best_compatible: tuple[float, str, GameMapLane, float] | None = None
    best_fallback: tuple[float, str, GameMapLane, float] | None = None
    for lane in game_map.lanes:
        points = np.asarray(lane.centerline_world, dtype=np.float64)
        lengths = _polyline_lengths(points)
        cumulative = 0.0
        for index, length in enumerate(lengths):
            segment = points[index + 1, :2] - points[index, :2]
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1.0e-10:
                cumulative += float(length)
                continue
            alpha = float(
                np.clip(
                    np.dot(ego_xy - points[index, :2], segment) / length_sq, 0.0, 1.0
                )
            )
            projected = points[index, :2] + alpha * segment
            distance = float(np.linalg.norm(ego_xy - projected))
            heading = math.atan2(float(segment[1]), float(segment[0]))
            heading_error = abs(_angle_delta(heading, ego_state.yaw_rad))
            candidate = (
                distance,
                lane.lane_id,
                lane,
                cumulative + alpha * float(length),
            )
            if best_fallback is None or candidate[:2] < best_fallback[:2]:
                best_fallback = candidate
            if heading_error <= math.pi * 0.5 and (
                best_compatible is None or candidate[:2] < best_compatible[:2]
            ):
                best_compatible = candidate
            cumulative += float(length)
    best = best_compatible or best_fallback
    return None if best is None else (best[2], best[3])


def _straightest_successor(
    lane: GameMapLane,
    lanes_by_id: dict[str, GameMapLane],
) -> GameMapLane | None:
    if not lane.successor_ids:
        return None
    _, outgoing_heading = _sample_polyline(
        lane.centerline_world, float(np.sum(_polyline_lengths(lane.centerline_world)))
    )
    candidates: list[tuple[float, str, GameMapLane]] = []
    for successor_id in lane.successor_ids:
        successor = lanes_by_id.get(successor_id)
        if successor is None:
            continue
        _, heading = _sample_polyline(successor.centerline_world, 0.0)
        candidates.append(
            (abs(_angle_delta(heading, outgoing_heading)), successor.lane_id, successor)
        )
    return None if not candidates else min(candidates, key=lambda item: item[:2])[2]


def road_ahead_pose(
    game_map: ResolvedGameMap,
    ego_state: VehicleState,
    ahead_m: float,
) -> tuple[npt.NDArray[np.float32], float] | None:
    """Walk directed lanes ahead, choosing the straightest legal successor."""
    nearest = _nearest_lane_progress(game_map, ego_state)
    if nearest is None:
        return None
    lane, distance = nearest
    lanes_by_id = {candidate.lane_id: candidate for candidate in game_map.lanes}
    remaining = float(ahead_m)
    visited = 0
    while visited <= len(lanes_by_id):
        total = float(np.sum(_polyline_lengths(lane.centerline_world)))
        available = max(0.0, total - distance)
        if remaining <= available:
            return _sample_polyline(lane.centerline_world, distance + remaining)
        remaining -= available
        successor = _straightest_successor(lane, lanes_by_id)
        if successor is None:
            return _sample_polyline(lane.centerline_world, total)
        lane = successor
        distance = 0.0
        visited += 1
    return _sample_polyline(lane.centerline_world, distance)


def _placement_pose(
    *,
    ego_state: VehicleState,
    ahead_m: float,
    lateral_m: float,
    placement: str,
    game_map: ResolvedGameMap | None,
) -> tuple[npt.NDArray[np.float32], float]:
    pose = (
        road_ahead_pose(game_map, ego_state, ahead_m)
        if placement == "road-ahead" and game_map is not None
        else None
    )
    if pose is None:
        heading = ego_state.yaw_rad
        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32)
        position = np.asarray(
            [ego_state.x_m, ego_state.y_m, ego_state.z_m], dtype=np.float32
        )
        position[:2] += ahead_m * forward
    else:
        position, heading = pose
    left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float32)
    position = position.copy()
    position[:2] += lateral_m * left
    return position, heading


def build_obstacle_event(
    template: ObstacleTemplate,
    *,
    ego_state: VehicleState,
    spawn_timestamp_us: int,
    config: LiveEditObstacleConfig,
    entity_id: str,
    slot: int = 0,
    static: bool = False,
    game_map: ResolvedGameMap | None = None,
    ground_vertices: npt.NDArray[np.floating] | None = None,
) -> ObstacleEvent:
    """Retime, place, and optionally rotate one source vehicle trajectory."""
    ahead_m = (
        config.static_ahead_m if static else config.spawn_ahead_m
    ) + slot * config.spacing_m
    lateral_m = (
        config.static_lateral_m * (1 if slot % 2 else -1)
        if static
        else config.lateral_m
    )
    target, road_heading = _placement_pose(
        ego_state=ego_state,
        ahead_m=ahead_m,
        lateral_m=lateral_m,
        placement=config.placement,
        game_map=game_map,
    )

    yaw_delta = 0.0
    if static:
        yaw_delta = road_heading - _quaternion_yaw(template.orientations_xyzw[0])
    elif config.placement == "road-ahead" and game_map is not None:
        crossing_heading = road_heading + (
            math.pi * 0.5 if slot % 2 == 0 else -math.pi * 0.5
        )
        yaw_delta = crossing_heading - template.motion_heading_rad

    translations = template.translations_local_m.copy()
    orientations = template.orientations_xyzw.copy()
    if abs(yaw_delta) > 1.0e-8:
        cos_yaw = math.cos(yaw_delta)
        sin_yaw = math.sin(yaw_delta)
        rotation = np.asarray(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32
        )
        translations[:, :2] = translations[:, :2] @ rotation.T
        yaw_quaternion = _yaw_quaternion(yaw_delta)
        orientations = np.stack(
            [
                _quaternion_multiply(yaw_quaternion, quaternion)
                for quaternion in orientations
            ]
        )
        orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)

    ground_z = local_ground_z(ground_vertices, target[:2])
    first_center_z = (
        float(target[2]) if ground_z is None else ground_z
    ) + template.source_ground_offset_m
    translations[:, 0] += target[0]
    translations[:, 1] += target[1]
    translations[:, 2] += np.float32(first_center_z)
    timestamps = template.timestamps_us + np.int64(spawn_timestamp_us)
    if static:
        timestamps = np.concatenate(
            [timestamps, np.asarray([_STATIC_PERSIST_US], dtype=np.int64)]
        )
        translations = np.concatenate([translations, translations[-1:]], axis=0)
        orientations = np.concatenate([orientations, orientations[-1:]], axis=0)

    return ObstacleEvent(
        entity_id=entity_id,
        object_type=template.object_type,
        timestamps_us=timestamps,
        translations_world=translations.astype(np.float32),
        orientations_xyzw=orientations.astype(np.float32),
        dimensions_lwh=template.dimensions_lwh.copy(),
        template_index=template.template_index,
        drive_speed_mps=(
            0.0
            if static
            else min(template.sampled_speed_mps, _MAX_OBSTACLE_DRIVE_SPEED_MPS)
        ),
        static=static,
    )


class ObstacleAbility:
    """Own track-backed obstacle lifecycle, rendering, and PhysX control."""

    def __init__(
        self,
        config: LiveEditObstacleConfig,
        *,
        catalog: ObstacleTemplateCatalog | None = None,
        templates: tuple[ObstacleTemplate, ...] | None = None,
        parked_templates: tuple[ObstacleTemplate, ...] | None = None,
        game_map: ResolvedGameMap | None = None,
        ground_vertices: npt.NDArray[np.floating] | None = None,
        vehicle: VehicleConfig | None = None,
    ) -> None:
        self._config = config
        self._game_map = game_map
        self._ground_vertices = ground_vertices
        self._vehicle = vehicle or VehicleConfig()
        resolved_catalog = catalog
        if resolved_catalog is None and (
            templates is None or (parked_templates is None and config.static_count > 0)
        ):
            resolved_catalog = load_obstacle_template_catalog()
        if templates is None:
            assert resolved_catalog is not None
            self._templates = resolved_catalog.moving(
                min_drift_m=config.min_drift_m,
                min_coverage_s=config.min_coverage_s,
                length_range_m=config.length_range_m,
            )
        else:
            self._templates = templates
        if parked_templates is not None:
            self._parked_templates = parked_templates
        elif config.static_count == 0:
            self._parked_templates = ()
        else:
            assert resolved_catalog is not None
            self._parked_templates = resolved_catalog.parked(
                length_range_m=config.length_range_m
            )
        self._pending: list[tuple[int, int]] = []
        self._events: list[ObstacleEvent] = []
        self._chunk_index = 0
        self._event_count = 0
        self._burst_count = 0
        self._hit_count = 0
        self._static_initialized = False
        self._owned_ids: set[str] = set()
        self._drive_speeds_mps: dict[str, float] = {}
        logger.info(
            "[live-edit] obstacle templates: {} moving, {} parked (of {})",
            len(self._templates),
            len(self._parked_templates),
            len(resolved_catalog.templates) if resolved_catalog is not None else 0,
        )

    @property
    def active(self) -> bool:
        return bool(self.events)

    @property
    def event(self) -> ObstacleEvent | None:
        return self.events[0] if self.events else None

    @property
    def events(self) -> tuple[ObstacleEvent, ...]:
        return tuple(
            event for event in self._events if event.phase is not ObstaclePhase.EXPIRED
        )

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        return tuple(
            event.scene_object
            for event in self.events
            if event.scene_object is not None
        )

    @property
    def active_objects(self) -> tuple[SceneObject, ...]:
        return self.objects if self._config.physics else ()

    @property
    def active_object_ids(self) -> frozenset[str]:
        return frozenset(scene_object.object_id for scene_object in self.active_objects)

    @property
    def active_timestamps_us(self) -> dict[str, int]:
        return {scene_object.object_id: 0 for scene_object in self.active_objects}

    @property
    def object_ids(self) -> frozenset[str]:
        return frozenset(self._owned_ids)

    @property
    def max_drive_speeds_mps(self) -> dict[str, float]:
        return dict(self._drive_speeds_mps)

    def request_spawn(self) -> None:
        """Queue one configured crossing burst; only one burst may be active."""
        if not self._templates:
            logger.warning("[live-edit] obstacle spawn requested but no templates")
            return
        if self._pending or any(not event.static for event in self.events):
            return
        base = self._chunk_index
        self._pending = [
            (slot, base + slot * self._config.stagger_chunks)
            for slot in range(self._config.count)
        ]
        self._burst_count += 1

    def reset(self) -> None:
        self._pending.clear()
        self._events.clear()
        self._chunk_index = 0
        self._static_initialized = False
        self._owned_ids.clear()
        self._drive_speeds_mps.clear()

    def _make_event(
        self,
        ego_state: VehicleState,
        spawn_timestamp_us: int,
        slot: int,
        *,
        static: bool,
    ) -> ObstacleEvent:
        entity_id = (
            f"{OBSTACLE_ENTITY_PREFIX}-static-{slot}"
            if static
            else f"{OBSTACLE_ENTITY_PREFIX}-{self._event_count}"
        )
        if not static:
            self._event_count += 1
        if static:
            if not self._parked_templates:
                raise RuntimeError("No parked obstacle templates are available")
            template = self._parked_templates[slot % len(self._parked_templates)]
        else:
            ahead_m = self._config.spawn_ahead_m + slot * self._config.spacing_m
            _, reference_heading = _placement_pose(
                ego_state=ego_state,
                ahead_m=ahead_m,
                lateral_m=self._config.lateral_m,
                placement=self._config.placement,
                game_map=self._game_map,
            )
            template = self._templates[self._select_template_index(reference_heading)]
        event = build_obstacle_event(
            template,
            ego_state=ego_state,
            spawn_timestamp_us=spawn_timestamp_us,
            config=self._config,
            entity_id=entity_id,
            slot=slot,
            static=static,
            game_map=self._game_map,
            ground_vertices=self._ground_vertices,
        )
        self._owned_ids.add(entity_id)
        self._drive_speeds_mps[entity_id] = event.drive_speed_mps
        if self._config.physics:
            relative_timestamps = event.timestamps_us - np.int64(spawn_timestamp_us)
            event.scene_object = SceneObject(
                object_id=event.entity_id,
                object_type=event.object_type,
                model=rigid_body_model_for_object(
                    event.object_type,
                    event.dimensions_lwh,
                    restitution=self._vehicle.collision_restitution,
                    friction=self._vehicle.collision_friction,
                ),
                timestamps_us=relative_timestamps,
                positions_m=event.translations_world.copy(),
                orientations_xyzw=event.orientations_xyzw.copy(),
            )
        self._events.append(event)
        start = event.translations_world[0]
        logger.info(
            "[live-edit] obstacle spawned {} template={} mode={} placement={} "
            "at ({:.1f}, {:.1f}, {:.1f})",
            event.entity_id,
            event.template_index,
            "physical" if self._config.physics else "visual",
            self._config.placement,
            start[0],
            start[1],
            start[2],
        )
        return event

    def _select_template_index(self, reference_heading_rad: float) -> int:
        """Pick a distinct source track that crosses the reference heading."""
        forward = np.asarray(
            [math.cos(reference_heading_rad), math.sin(reference_heading_rad)],
            dtype=np.float64,
        )

        def crossness(index: int) -> float:
            motion = (
                self._templates[index].translations_local_m[-1, :2].astype(np.float64)
            )
            return abs(float(motion @ forward / (np.linalg.norm(motion) or 1.0)))

        ranked = sorted(range(len(self._templates)), key=crossness)
        pool_size = max(1, min(max(4, self._config.count), len(ranked)))
        top = ranked[:pool_size]
        offset = (self._burst_count - 1) % len(top)
        rotated = top[offset:] + top[:offset]
        in_use_template_ids = {
            event.template_index for event in self.events if not event.static
        }
        unused = [
            index
            for index in rotated
            if self._templates[index].template_index not in in_use_template_ids
        ]
        return (unused or rotated)[0]

    def _ego_state_from_body(self, ego: BodyState) -> VehicleState:
        x, y, z, w = (float(value) for value in ego.orientation_xyzw)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return VehicleState(
            x_m=float(ego.position_m[0]),
            y_m=float(ego.position_m[1]),
            z_m=float(ego.position_m[2] - self._vehicle.aabb_height_m * 0.5),
            yaw_rad=yaw,
            speed_mps=float(np.linalg.norm(ego.linear_velocity_mps[:2])),
            steer_rad=0.0,
        )

    def _spawn_due(self, ego_state: VehicleState, timestamp_us: int) -> None:
        if not self._static_initialized:
            self._static_initialized = True
            if self._config.static_count and not self._parked_templates:
                logger.warning(
                    "[live-edit] static roadblock requested but no parked templates"
                )
            else:
                for slot in range(self._config.static_count):
                    self._make_event(ego_state, timestamp_us, slot, static=True)
        due = [entry for entry in self._pending if entry[1] <= self._chunk_index]
        self._pending = [
            entry for entry in self._pending if entry[1] > self._chunk_index
        ]
        for slot, _ in due:
            self._make_event(ego_state, timestamp_us, slot, static=False)

    def prepare_topology(self, ego: BodyState) -> None:
        """Materialize due physical events before the native-world sync."""
        if not self._config.physics:
            return
        self._spawn_due(self._ego_state_from_body(ego), 0)

    def prepare_step(self, ego: BodyState, dt_s: float) -> tuple[ActorTrackTarget, ...]:
        """Advance scripted motion and return targets for physical obstacles."""
        del ego
        if not self._config.physics:
            return ()
        targets: list[ActorTrackTarget] = []
        for event in self.events:
            if event.phase is not ObstaclePhase.SCRIPTED or event.scene_object is None:
                continue
            event.logical_timestamp_us = min(
                event.logical_timestamp_us + dt_s * 1_000_000.0,
                float(event.scene_object.timestamps_us[-1]),
            )
            targets.append(
                ActorTrackTarget(
                    object_id=event.entity_id,
                    timestamp_us=int(event.logical_timestamp_us),
                )
            )
        return tuple(targets)

    def observe_physics(
        self,
        object_id: str,
        *,
        struck: bool,
        body: BodyState,
        dt_s: float,
    ) -> ActorControlDecision | None:
        del dt_s
        event = next(
            (item for item in self.events if item.entity_id == object_id), None
        )
        if event is None:
            return None
        event.physical_position_m = body.position_m.copy()
        event.physical_orientation_xyzw = body.orientation_xyzw.copy()
        if struck and event.phase is ObstaclePhase.SCRIPTED:
            event.phase = ObstaclePhase.DETACHED
            if not event.hit_logged:
                event.hit_logged = True
                self._hit_count += 1
                logger.info("[live-edit] obstacle HIT {}", event.entity_id)
        return ActorControlDecision(
            drive_enabled=event.phase is ObstaclePhase.SCRIPTED,
            detached_from_track=event.phase is ObstaclePhase.DETACHED,
        )

    def _check_visual_collision(
        self, event: ObstacleEvent, trajectory: TrajectoryChunk
    ) -> None:
        if event.hit_logged:
            return
        for timestamp_us, state in zip(
            trajectory.timestamps_us, trajectory.vehicle_states, strict=True
        ):
            center = event.center_at(int(timestamp_us))
            if (
                center is not None
                and math.hypot(
                    float(center[0]) - state.x_m, float(center[1]) - state.y_m
                )
                <= self._config.collision_radius_m
            ):
                event.hit_logged = True
                self._hit_count += 1
                logger.info("[live-edit] obstacle HIT {}", event.entity_id)
                return

    def advance_frames(
        self, trajectory: TrajectoryChunk
    ) -> tuple[DynamicActorTrajectory, ...]:
        """Advance chunk lifetimes and return visual-only renderer actors."""
        if not self._config.physics:
            self._spawn_due(
                trajectory.vehicle_states[0], int(trajectory.timestamps_us[0])
            )
        actors: list[DynamicActorTrajectory] = []
        last_timestamp_us = int(trajectory.timestamps_us[-1])
        for event in self.events:
            event.chunks += 1
            if not self._config.physics:
                self._check_visual_collision(event, trajectory)
                actors.append(event.actor())
            track_exhausted = (
                event.logical_timestamp_us
                >= float(event.scene_object.timestamps_us[-1])
                if event.scene_object is not None
                else last_timestamp_us >= int(event.timestamps_us[-1])
            )
            if not event.static and (
                event.chunks >= self._config.active_chunks or track_exhausted
            ):
                event.phase = ObstaclePhase.EXPIRED
                reason = "track exhausted" if track_exhausted else "duration reached"
                logger.info(
                    "[live-edit] obstacle despawned {} ({})", event.entity_id, reason
                )
        self._chunk_index += 1
        return tuple(actors)


__all__ = [
    "OBSTACLE_ENTITY_PREFIX",
    "ObstacleAbility",
    "ObstacleEvent",
    "ObstaclePhase",
    "ObstacleTemplate",
    "ObstacleTemplateCatalog",
    "build_obstacle_event",
    "local_ground_z",
    "road_ahead_pose",
]
