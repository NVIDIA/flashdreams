# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Race-mode progression and leaderboard state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.game_map import GameMapRaceCourse, ResolvedGameMap
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState
from shapely.geometry import LineString, Point, Polygon

from crazy_robotaxi.game import relative_target_bearing_rad
from crazy_robotaxi.high_scores import RaceTimeEntry, RaceTimeStore

RaceSessionState = Literal["awaiting_start", "racing", "awaiting_name", "leaderboard"]
RaceTargetKind = Literal["start", "checkpoint"]
RaceEvent = Literal["race_started", "checkpoint", "lap_complete", "race_complete"]


@dataclass(frozen=True)
class RaceGameSnapshot:
    """Immutable race state published to native and browser HUDs."""

    map_id: str
    """Stable ID of the containing map."""

    course_id: str
    """Selected course ID within the map."""

    session_state: RaceSessionState
    """Current pre-race, racing, name-entry, or leaderboard state."""

    target_kind: RaceTargetKind
    """Kind of gate the player must enter next."""

    target_element_id: str
    """Map element whose surface is the active gate."""

    target_xyz_m: tuple[float, float, float]
    """Fixed midpoint of the active gate in world coordinates."""

    gate_start_xyz_m: tuple[float, float, float]
    """First fixed endpoint of the active gate line."""

    gate_end_xyz_m: tuple[float, float, float]
    """Second fixed endpoint of the active gate line."""

    checkpoint_markers: bool
    """Whether presenters display camera-view gate markers."""

    distance_m: float
    """Shortest XY distance from the ego to the active gate."""

    relative_bearing_rad: float
    """Bearing from ego heading toward the active gate."""

    checkpoint_index: int
    """Zero-based active checkpoint index."""

    checkpoint_count: int
    """Number of ordered checkpoints in the course."""

    completed_laps: int
    """Number of laps closed by returning to the start gate."""

    lap_count: int
    """Required laps, or zero for a point-to-point race."""

    elapsed_time_us: int
    """Current total race time in integer microseconds."""

    best_time_us: int | None
    """Fastest persisted total time for this map/course pair."""

    final_time_us: int | None = None
    """Frozen finished time, or ``None`` while the race is active."""

    leaderboard: tuple[RaceTimeEntry, ...] = ()
    """Map- and course-specific top times."""

    high_score_rank: int | None = None
    """Prospective or recorded rank for the finished race."""

    event: RaceEvent | None = None
    """Progress event emitted by the latest processed pose."""

    game_mode: Literal["race"] = "race"
    """Mode discriminator consumed by presenter clients."""

    @property
    def phase(self) -> Literal["race"]:
        """Return the compatibility phase used by shared target projection."""
        return "race"

    @property
    def target_radius_m(self) -> float:
        """Return the display radius for the race target marker."""
        return 4.0

    @property
    def pickup_targets_xyz_m(self) -> tuple[tuple[float, float, float], ...]:
        """Return no alternate targets for the ordered race course."""
        return ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the snapshot."""
        return {
            "game_mode": self.game_mode,
            "map_id": self.map_id,
            "course_id": self.course_id,
            "session_state": self.session_state,
            "target_kind": self.target_kind,
            "target_element_id": self.target_element_id,
            "target_xyz_m": list(self.target_xyz_m),
            "gate_start_xyz_m": list(self.gate_start_xyz_m),
            "gate_end_xyz_m": list(self.gate_end_xyz_m),
            "checkpoint_markers": self.checkpoint_markers,
            "distance_m": self.distance_m,
            "relative_bearing_rad": self.relative_bearing_rad,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_count": self.checkpoint_count,
            "completed_laps": self.completed_laps,
            "lap_count": self.lap_count,
            "elapsed_time_us": self.elapsed_time_us,
            "elapsed_time_s": self.elapsed_time_us / 1_000_000.0,
            "best_time_us": self.best_time_us,
            "best_time_s": (
                None if self.best_time_us is None else self.best_time_us / 1_000_000.0
            ),
            "final_time_us": self.final_time_us,
            "final_time_s": (
                None if self.final_time_us is None else self.final_time_us / 1_000_000.0
            ),
            "leaderboard": [entry.as_dict() for entry in self.leaderboard],
            "high_score_rank": self.high_score_rank,
            "event": self.event,
        }


class RaceController:
    """Advance one ordered map course using swept gate-line activation."""

    def __init__(
        self,
        game_map: ResolvedGameMap,
        course: GameMapRaceCourse,
        initial_state: VehicleState,
        time_store: RaceTimeStore,
    ) -> None:
        self._map_id = game_map.map_id
        self._course = course
        self._time_store = time_store
        surfaces = {
            element.element_id: Polygon(element.surface_world[:, :2])
            for element in game_map.elements
        }
        element_ids = (course.start_element_id, *course.checkpoint_element_ids)
        self._surfaces = {
            element_id: surfaces[element_id] for element_id in element_ids
        }
        self._surface_z = {
            element.element_id: float(element.surface_world[:, 2].mean())
            for element in game_map.elements
            if element.element_id in self._surfaces
        }
        self._gates = self._build_gates()
        self._session_state: RaceSessionState = "awaiting_start"
        self._target_kind: RaceTargetKind = "start"
        self._checkpoint_index = 0
        self._completed_laps = 0
        self._start_timestamp_us: int | None = None
        self._elapsed_time_us = 0
        self._final_time_us: int | None = None
        self._event: RaceEvent | None = None
        self._previous_xy = (initial_state.x_m, initial_state.y_m)
        self._leaderboard = time_store.read(game_map.map_id, course.course_id)
        self._best_time_us = (
            self._leaderboard[0].elapsed_time_us if self._leaderboard else None
        )
        self._high_score_rank: int | None = None

    @property
    def is_playing(self) -> bool:
        """Return whether simulation should continue advancing."""
        return self._session_state in {"awaiting_start", "racing"}

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> tuple[RaceGameSnapshot, ...]:
        """Advance checkpoints using each authoritative timestamped pose."""
        if frame_interval_s < 0.0:
            raise ValueError("Race frame interval must be non-negative.")
        snapshots: list[RaceGameSnapshot] = []
        for state, timestamp_us in zip(
            trajectory.vehicle_states, trajectory.timestamps_us, strict=True
        ):
            if self.is_playing:
                self._advance_pose(state, int(timestamp_us))
            self._previous_xy = (state.x_m, state.y_m)
            snapshots.append(self.snapshot(state))
        return tuple(snapshots)

    def snapshot(self, state: VehicleState) -> RaceGameSnapshot:
        """Return race state relative to the supplied ego pose."""
        target_id = self._target_element_id
        gate = self._gates[target_id]
        ego = Point(state.x_m, state.y_m)
        start_xy, end_xy = gate.coords[0], gate.coords[-1]
        target_xyz = (
            (float(start_xy[0]) + float(end_xy[0])) / 2.0,
            (float(start_xy[1]) + float(end_xy[1])) / 2.0,
            self._surface_z[target_id],
        )
        gate_start_xyz = (
            float(start_xy[0]),
            float(start_xy[1]),
            self._surface_z[target_id],
        )
        gate_end_xyz = (
            float(end_xy[0]),
            float(end_xy[1]),
            self._surface_z[target_id],
        )
        distance = float(ego.distance(gate))
        return RaceGameSnapshot(
            map_id=self._map_id,
            course_id=self._course.course_id,
            session_state=self._session_state,
            target_kind=self._target_kind,
            target_element_id=target_id,
            target_xyz_m=target_xyz,
            gate_start_xyz_m=gate_start_xyz,
            gate_end_xyz_m=gate_end_xyz,
            checkpoint_markers=self._course.checkpoint_markers,
            distance_m=distance,
            relative_bearing_rad=relative_target_bearing_rad(
                state.x_m,
                state.y_m,
                state.yaw_rad,
                target_xyz[0],
                target_xyz[1],
            ),
            checkpoint_index=self._checkpoint_index,
            checkpoint_count=len(self._course.checkpoint_element_ids),
            completed_laps=self._completed_laps,
            lap_count=self._course.lap_count,
            elapsed_time_us=self._elapsed_time_us,
            best_time_us=self._best_time_us,
            final_time_us=self._final_time_us,
            leaderboard=self._leaderboard,
            high_score_rank=self._high_score_rank,
            event=self._event,
        )

    def submit_high_score_name(self, name: str) -> None:
        """Persist the finished total race time under a validated player name."""
        if self._session_state != "awaiting_name" or self._final_time_us is None:
            raise RuntimeError("Race is not waiting for a leaderboard name.")
        inserted, self._leaderboard = self._time_store.record(
            self._map_id,
            self._course.course_id,
            name,
            self._final_time_us,
        )
        self._best_time_us = (
            self._leaderboard[0].elapsed_time_us if self._leaderboard else None
        )
        self._high_score_rank = (
            None if inserted is None else 1 + self._leaderboard.index(inserted)
        )
        self._session_state = "leaderboard"

    @property
    def _target_element_id(self) -> str:
        if self._target_kind == "start":
            return self._course.start_element_id
        return self._course.checkpoint_element_ids[self._checkpoint_index]

    def _advance_pose(self, state: VehicleState, timestamp_us: int) -> None:
        self._event = None
        if self._start_timestamp_us is not None:
            self._elapsed_time_us = max(0, timestamp_us - self._start_timestamp_us)
        if not self._target_hit(state.x_m, state.y_m):
            return
        if self._session_state == "awaiting_start":
            self._start_timestamp_us = timestamp_us
            self._elapsed_time_us = 0
            self._session_state = "racing"
            self._target_kind = "checkpoint"
            self._event = "race_started"
            return
        if self._target_kind == "start":
            self._completed_laps += 1
            if self._completed_laps >= self._course.lap_count:
                self._finish(timestamp_us)
            else:
                self._target_kind = "checkpoint"
                self._checkpoint_index = 0
                self._event = "lap_complete"
            return
        if self._checkpoint_index + 1 < len(self._course.checkpoint_element_ids):
            self._checkpoint_index += 1
            self._event = "checkpoint"
            return
        if self._course.lap_count == 0:
            self._finish(timestamp_us)
        else:
            self._target_kind = "start"
            self._event = "checkpoint"

    def _target_hit(self, x_m: float, y_m: float) -> bool:
        gate = self._gates[self._target_element_id]
        current = Point(x_m, y_m)
        if gate.covers(current):
            return True
        if self._previous_xy == (x_m, y_m):
            return False
        return LineString((self._previous_xy, (x_m, y_m))).intersects(gate)

    def _build_gates(self) -> dict[str, LineString]:
        gates = {
            self._course.start_element_id: _cross_course_gate(
                self._surfaces[self._course.start_element_id],
                self._surfaces[self._course.checkpoint_element_ids[0]],
            )
        }
        previous_id = self._course.start_element_id
        for checkpoint_id in self._course.checkpoint_element_ids:
            gates[checkpoint_id] = _cross_course_gate(
                self._surfaces[checkpoint_id],
                self._surfaces[previous_id],
            )
            previous_id = checkpoint_id
        return gates

    def _finish(self, timestamp_us: int) -> None:
        assert self._start_timestamp_us is not None
        self._elapsed_time_us = max(1, timestamp_us - self._start_timestamp_us)
        self._final_time_us = self._elapsed_time_us
        self._leaderboard = self._time_store.read(self._map_id, self._course.course_id)
        self._high_score_rank = self._time_store.qualifying_rank(
            self._map_id, self._course.course_id, self._final_time_us
        )
        self._session_state = (
            "awaiting_name" if self._high_score_rank is not None else "leaderboard"
        )
        self._event = "race_complete"


def _cross_course_gate(surface: Polygon, adjacent_surface: Polygon) -> LineString:
    """Build a fixed line across the side facing an adjacent course element.

    Args:
        surface: Surface on which to place the gate.
        adjacent_surface: Adjacent course surface used to select the relevant
            entrance or exit and infer travel direction.

    Returns:
        A line spanning the target surface perpendicular to travel.
    """
    target = surface.representative_point()
    approach = adjacent_surface.representative_point()
    approach_xy = np.asarray([approach.x, approach.y], dtype=np.float64)
    direction = np.asarray(
        [target.x - approach.x, target.y - approach.y], dtype=np.float64
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        direction /= norm
    span = max(
        surface.bounds[2] - surface.bounds[0],
        surface.bounds[3] - surface.bounds[1],
    )
    span = max(10.0, span * 4.0)
    anchor = np.asarray([target.x, target.y], dtype=np.float64)
    ray = LineString(
        (
            (approach.x, approach.y),
            tuple(anchor + direction * span),
        )
    )
    path_parts = _line_parts(ray.intersection(surface))
    if path_parts:
        path = min(path_parts, key=lambda part: part.distance(approach))
        endpoints = (np.asarray(path.coords[0]), np.asarray(path.coords[-1]))
        entry = min(
            endpoints,
            key=lambda point: np.linalg.norm(point - approach_xy),
        )
        anchor = entry + direction * min(1.0, path.length * 0.15)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    cross_line = LineString(
        (tuple(anchor - perpendicular * span), tuple(anchor + perpendicular * span))
    )
    cross_parts = _line_parts(cross_line.intersection(surface))
    if not cross_parts:
        return LineString(
            (tuple(anchor - perpendicular), tuple(anchor + perpendicular))
        )
    return max(cross_parts, key=lambda part: part.length)


def _line_parts(geometry: object) -> tuple[LineString, ...]:
    if isinstance(geometry, LineString):
        return (geometry,) if geometry.length > 1.0e-6 else ()
    return tuple(
        part
        for part in getattr(geometry, "geoms", ())
        if isinstance(part, LineString) and part.length > 1.0e-6
    )


def project_race_gate_to_camera(
    snapshot: RaceGameSnapshot,
    rig_to_world: npt.NDArray[np.float32],
    camera_model: FThetaCameraModel,
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Project and clip the active race gate into camera pixels.

    Args:
        snapshot: Current race state and fixed gate endpoints.
        rig_to_world: Camera-rig pose in world coordinates.
        camera_model: Camera projection model.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.

    Returns:
        Clipped pixel endpoints, or ``None`` when camera markers are disabled or
        the gate is not visible.
    """
    if not snapshot.checkpoint_markers:
        return None
    points = np.asarray(
        (snapshot.gate_start_xyz_m, snapshot.gate_end_xyz_m), dtype=np.float32
    )
    points[:, 2] += np.float32(0.08)
    uv, _depth, forward = camera_model.project_world(points, rig_to_world)
    if not bool(forward.all()):
        return None
    clipped = _clip_unit_segment(
        (float(uv[0, 0]) / image_width, float(uv[0, 1]) / image_height),
        (float(uv[1, 0]) / image_width, float(uv[1, 1]) / image_height),
    )
    if clipped is None:
        return None
    return (
        (clipped[0][0] * image_width, clipped[0][1] * image_height),
        (clipped[1][0] * image_width, clipped[1][1] * image_height),
    )


def _clip_unit_segment(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
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
