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

"""CPU tests for race courses, progression, and scoped top times."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from crazy_robotaxi.high_scores import RaceTimeStore
from crazy_robotaxi.race import RaceController, RaceGameSnapshot
from omnidreams_game_engine.game_map import (
    GameMapError,
    GameMapRaceCourse,
    load_game_map,
)
from omnidreams_game_engine.game_map.types import ResolvedGameMap
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState
from shapely.geometry import Polygon

pytestmark = pytest.mark.ci_cpu

_MAP = Path(__file__).parent / "maps" / "race_course.robotaxi.yaml"
_BOULEVARD_MAP = (
    Path(__file__).parents[1]
    / "crazy_robotaxi"
    / "maps"
    / "boulevard_district.robotaxi.yaml"
)
_RACEWAY_MAP = (
    Path(__file__).parents[1]
    / "crazy_robotaxi"
    / "maps"
    / "flashdreams_raceway.robotaxi.yaml"
)


def _state(x_m: float, y_m: float) -> VehicleState:
    return VehicleState(x_m, y_m, 0.0, 0.0, 0.0, 0.0)


def _trajectory(
    points: list[tuple[float, float]], timestamps_us: list[int]
) -> TrajectoryChunk:
    states = tuple(_state(*point) for point in points)
    return TrajectoryChunk(
        timestamps_us=np.asarray(timestamps_us, dtype=np.int64),
        rig_poses_world=np.stack(
            [rig_pose_from_vehicle_state(state) for state in states]
        ),
        vehicle_states=states,
        boundary_state_after_chunk=states[-1],
    )


def _point(game_map: ResolvedGameMap, element_id: str) -> tuple[float, float]:
    element = next(
        element for element in game_map.elements if element.element_id == element_id
    )
    point = Polygon(element.surface_world[:, :2]).representative_point()
    return float(point.x), float(point.y)


def _active_gate_midpoint(controller: RaceController) -> tuple[float, float]:
    snapshot = controller.snapshot(_state(0.0, 0.0))
    return snapshot.target_xyz_m[0], snapshot.target_xyz_m[1]


def _cross_active_gate(
    controller: RaceController,
    timestamp_us: int,
    *,
    forward: bool = True,
) -> RaceGameSnapshot:
    target_id = controller._target_element_id
    gate = controller._gates[target_id]
    direction = controller._gate_directions[target_id]
    midpoint = np.asarray(gate.interpolate(0.5, normalized=True).coords[0])
    before = midpoint - direction * 5.0
    after = midpoint + direction * 5.0
    if not forward:
        before, after = after, before
    controller._previous_xy = float(before[0]), float(before[1])
    return controller.advance_frames(
        _trajectory([(float(after[0]), float(after[1]))], [timestamp_us]), 1.0
    )[-1]


def test_loop_finishes_only_after_final_return_to_start(
    tmp_path: Path,
) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    start = _point(game_map, course.start_element_id)
    controller = RaceController(
        game_map,
        course,
        _state(start[0] + 100.0, start[1] - 100.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    awaiting_start = controller.snapshot(_state(*start))
    assert awaiting_start.target_label == "START"
    assert awaiting_start.as_dict()["target_label"] == "START"

    timestamp = 1_000_000
    started = _cross_active_gate(controller, timestamp)
    assert started.target_label == "CHECKPOINT"
    for lap in range(course.lap_count):
        for _checkpoint in course.checkpoint_element_ids:
            timestamp += 1_000_000
            _cross_active_gate(controller, timestamp)
        assert controller.is_playing
        assert controller.snapshot(_state(*start)).target_label == "FINISH"
        timestamp += 1_000_000
        snapshot = _cross_active_gate(controller, timestamp)
        assert snapshot.completed_laps == lap + 1

    assert not controller.is_playing
    assert snapshot.session_state == "awaiting_name"
    assert snapshot.final_time_us == timestamp - 1_000_000
    controller.submit_high_score_name("Racer")
    leaderboard = controller.snapshot(_state(*start))
    assert leaderboard.session_state == "leaderboard"
    assert leaderboard.leaderboard[0].name == "Racer"
    assert leaderboard.high_score_rank == 1


def test_point_to_point_finishes_at_last_checkpoint_and_rejects_skips(
    tmp_path: Path,
) -> None:
    game_map = load_game_map(_MAP)
    authored = game_map.race_courses[0]
    course = GameMapRaceCourse(
        course_id="sprint",
        start_element_id=authored.start_element_id,
        checkpoint_element_ids=(
            authored.checkpoint_element_ids[0],
            authored.checkpoint_element_ids[2],
        ),
        lap_count=0,
    )
    start = _point(game_map, course.start_element_id)
    last = _point(game_map, course.checkpoint_element_ids[1])
    controller = RaceController(
        game_map,
        course,
        _state(start[0] + 100, start[1] - 100),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    _cross_active_gate(controller, 1_000_000)
    skipped = controller.advance_frames(_trajectory([last], [2_000_000]), 1.0)[-1]
    assert skipped.checkpoint_index == 0
    assert controller.is_playing
    final_gate = _cross_active_gate(controller, 3_000_000)
    assert final_gate.target_label == "FINISH"
    finished = _cross_active_gate(controller, 4_000_000)

    assert finished.final_time_us == 3_000_000
    assert not controller.is_playing


def test_start_gate_detects_a_swept_crossing(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    probe = RaceController(
        game_map,
        course,
        _state(0.0, 0.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    gate = probe.snapshot(_state(0.0, 0.0))
    start = np.asarray(gate.gate_start_xyz_m[:2])
    end = np.asarray(gate.gate_end_xyz_m[:2])
    midpoint = (start + end) / 2.0
    normal = np.asarray([-(end - start)[1], (end - start)[0]])
    normal /= np.linalg.norm(normal)
    before = midpoint - normal * 5.0
    after = midpoint + normal * 5.0
    controller = RaceController(
        game_map,
        course,
        _state(float(before[0]), float(before[1])),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    snapshot = controller.advance_frames(
        _trajectory([(float(after[0]), float(after[1]))], [7_000_000]), 1.0
    )[-1]

    assert snapshot.session_state == "racing"
    assert snapshot.elapsed_time_us == 0


@pytest.mark.parametrize(
    ("element_id", "expected_direction"),
    [
        ("south_west", (1.0, 0.0)),
        ("southeast", (1.0, 0.0)),
        ("east_north", (0.0, 1.0)),
        ("north", (-1.0, 0.0)),
        ("west_south", (0.0, -1.0)),
    ],
)
def test_gate_directions_follow_authored_course(
    tmp_path: Path,
    element_id: str,
    expected_direction: tuple[float, float],
) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    assert (
        np.dot(controller._gate_directions[element_id], np.asarray(expected_direction))
        > 0.99
    )


def test_backward_start_crossing_does_not_start(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    snapshot = _cross_active_gate(controller, 1_000_000, forward=False)

    assert snapshot.session_state == "awaiting_start"
    assert snapshot.target_kind == "start"
    assert snapshot.event is None
    assert snapshot.elapsed_time_us == 0


def test_backward_checkpoint_crossing_does_not_advance(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    _cross_active_gate(controller, 1_000_000)
    checkpoint_id = controller._target_element_id

    snapshot = _cross_active_gate(controller, 2_000_000, forward=False)

    assert snapshot.session_state == "racing"
    assert snapshot.checkpoint_index == 0
    assert snapshot.target_element_id == checkpoint_id
    assert snapshot.event is None


def test_backward_finish_crossing_does_not_complete_race(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    authored = game_map.race_courses[0]
    course = GameMapRaceCourse(
        course_id="one-lap",
        start_element_id=authored.start_element_id,
        checkpoint_element_ids=authored.checkpoint_element_ids,
        lap_count=1,
    )
    store = RaceTimeStore(tmp_path / "times.csv")
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        store,
    )
    timestamp_us = 1_000_000
    _cross_active_gate(controller, timestamp_us)
    for _checkpoint in course.checkpoint_element_ids:
        timestamp_us += 1_000_000
        _cross_active_gate(controller, timestamp_us)
    assert controller._target_element_id == course.start_element_id

    snapshot = _cross_active_gate(controller, timestamp_us + 1_000_000, forward=False)

    assert snapshot.session_state == "racing"
    assert snapshot.completed_laps == 0
    assert snapshot.target_label == "FINISH"
    assert snapshot.final_time_us is None
    assert snapshot.event is None
    assert store.read(game_map.map_id, course.course_id) == ()


@pytest.mark.parametrize("gate_fraction", [0.1, 0.9])
def test_swept_crossing_advances_through_adjacent_gates(
    tmp_path: Path, gate_fraction: float
) -> None:
    game_map = load_game_map(_MAP)
    authored = game_map.race_courses[0]
    course = GameMapRaceCourse(
        course_id="adjacent-gates",
        start_element_id=authored.start_element_id,
        checkpoint_element_ids=("southeast", "east_south", "north"),
        lap_count=0,
    )
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    _cross_active_gate(controller, 1_000_000)
    first_gate = controller._gates[course.checkpoint_element_ids[0]]
    second_gate = controller._gates[course.checkpoint_element_ids[1]]
    first_crossing = np.asarray(
        first_gate.interpolate(gate_fraction, normalized=True).coords[0]
    )
    second_crossing = np.asarray(
        second_gate.interpolate(gate_fraction, normalized=True).coords[0]
    )
    direction = second_crossing - first_crossing
    direction /= np.linalg.norm(direction)
    before = first_crossing - direction
    after = second_crossing + direction
    controller._previous_xy = (float(before[0]), float(before[1]))

    snapshot = controller.advance_frames(
        _trajectory([(float(after[0]), float(after[1]))], [2_000_000]), 1.0
    )[-1]

    assert snapshot.checkpoint_index == 2
    assert snapshot.target_element_id == "north"


def test_start_gate_is_near_course_exit_instead_of_element_midpoint(
    tmp_path: Path,
) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    start_center = np.asarray(_point(game_map, course.start_element_id))
    first_checkpoint = np.asarray(_point(game_map, course.checkpoint_element_ids[0]))
    gate_center = np.asarray(_active_gate_midpoint(controller))

    assert np.linalg.norm(gate_center - first_checkpoint) < np.linalg.norm(
        start_center - first_checkpoint
    )


def test_course_gates_span_the_full_road_surface(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    for element_id in (course.start_element_id, *course.checkpoint_element_ids):
        assert controller._gates[element_id].length == pytest.approx(8.4, abs=0.15)


def test_boulevard_intersection_gates_span_the_full_arterial(tmp_path: Path) -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    for element_id in (course.start_element_id, *course.checkpoint_element_ids):
        assert controller._gates[element_id].length >= 15.5


def test_grand_prix_hairpin_gate_is_at_the_course_entry(tmp_path: Path) -> None:
    game_map = load_game_map(_RACEWAY_MAP)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    checkpoint_index = 4
    checkpoint_id = course.checkpoint_element_ids[checkpoint_index]
    previous_id = course.checkpoint_element_ids[checkpoint_index - 1]
    following_id = course.checkpoint_element_ids[checkpoint_index + 1]

    assert checkpoint_id == "infield_hairpin"
    assert controller._gates[checkpoint_id].distance(
        controller._surfaces[previous_id]
    ) < controller._gates[checkpoint_id].distance(controller._surfaces[following_id])


def test_race_times_are_isolated_by_map_and_course(tmp_path: Path) -> None:
    store = RaceTimeStore(tmp_path / "times.csv", limit=2)
    store.record("map-a", "course-a", "Slow", 4_000_000)
    store.record("map-a", "course-a", "Fast", 2_000_000)
    store.record("map-a", "course-b", "Other", 1_000_000)
    store.record("map-b", "course-a", "Elsewhere", 500_000)

    assert [entry.name for entry in store.read("map-a", "course-a")] == [
        "Fast",
        "Slow",
    ]
    assert [entry.name for entry in store.read("map-a", "course-b")] == ["Other"]
    assert [entry.name for entry in store.read("map-b", "course-a")] == ["Elsewhere"]
    assert store.qualifying_rank("map-a", "course-a", 3_000_000) == 2
    assert store.qualifying_rank("map-a", "course-a", 5_000_000) is None


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"start": "missing"}, "unknown node or road"),
        ({"checkpoints": []}, "at least one checkpoint"),
        ({"lap_count": -1}, "nonnegative integer"),
        ({"checkpoints": ["south_west"]}, "may not reuse start"),
        ({"checkpoint_markers": "yes"}, "must be a boolean"),
    ],
)
def test_invalid_race_course_schema_is_rejected(
    tmp_path: Path, update: dict[str, object], message: str
) -> None:
    document = yaml.safe_load(_MAP.read_text(encoding="utf-8"))
    document["race_courses"][0].update(update)
    path = tmp_path / "invalid.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(GameMapError, match=message):
        load_game_map(path)


def test_checkpoint_markers_can_be_disabled_without_disabling_gates(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(_MAP.read_text(encoding="utf-8"))
    document["race_courses"][0]["checkpoint_markers"] = False
    path = tmp_path / "hidden-markers.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    game_map = load_game_map(path)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map,
        course,
        _state(-300.0, -300.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )

    before = controller.snapshot(_state(-300.0, -300.0))
    after = _cross_active_gate(controller, 1_000_000)

    assert before.checkpoint_markers is False
    assert after.session_state == "racing"
