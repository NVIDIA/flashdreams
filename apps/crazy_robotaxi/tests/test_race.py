# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for race courses, progression, and scoped top times."""

from __future__ import annotations

from dataclasses import replace
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
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState

pytestmark = pytest.mark.ci_cpu

_MAPS = Path(__file__).parent / "maps"
_MAP = _MAPS / "race_course.robotaxi.yaml"


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


def _drive(
    controller: RaceController, points: list[tuple[float, float]], first_second: int
) -> tuple[RaceGameSnapshot, ...]:
    return controller.advance_frames(
        _trajectory(
            points,
            [
                second * 1_000_000
                for second in range(first_second, first_second + len(points))
            ],
        ),
        1.0,
    )


def test_loop_finishes_only_after_final_return_to_start(tmp_path: Path) -> None:
    document = yaml.safe_load(_MAP.read_text(encoding="utf-8"))
    document["race_courses"][0]["checkpoint_markers"] = False
    path = tmp_path / "hidden-markers.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    game_map = load_game_map(path)
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map, course, _state(-20.0, -100.0), RaceTimeStore(tmp_path / "times.csv")
    )
    before = controller.snapshot(_state(-20.0, -100.0))
    assert before.target_label == "START"
    assert before.checkpoint_markers is False

    # The authored start road ends at (0, -100). Neither endpoint touches its gate.
    started = _drive(controller, [(20.0, -100.0)], 1)[-1]
    assert started.session_state == "racing"
    assert started.elapsed_time_us == 0
    assert started.target_label == "CHECKPOINT"

    # Follow the test map counterclockwise: east, north, west, south, then east.
    # These coordinates come from the authored roads, never controller gate output.
    lap_points = [
        (130.0, -100.0),
        (150.0, -100.0),
        (150.0, -20.0),
        (150.0, 20.0),
        (150.0, 80.0),
        (150.0, 100.0),
        (20.0, 100.0),
        (-20.0, 100.0),
        (-130.0, 100.0),
        (-150.0, 100.0),
        (-150.0, 20.0),
        (-150.0, -20.0),
        (-150.0, -80.0),
        (-150.0, -100.0),
        (-20.0, -100.0),
        (20.0, -100.0),
    ]
    for lap in range(course.lap_count):
        snapshots = _drive(controller, lap_points, 2 + lap * len(lap_points))
        assert [snapshots[index].target_element_id for index in (1, 3, 7, 11)] == [
            "east_north",
            "north",
            "west_south",
            "south_west",
        ]
        assert snapshots[-2].target_label == "FINISH"
        assert snapshots[-2].session_state == "racing"
        assert snapshots[-1].completed_laps == lap + 1

    assert not controller.is_playing
    assert snapshots[-1].session_state == "awaiting_name"
    assert snapshots[-1].final_time_us == 48_000_000
    controller.submit_high_score_name("Racer")
    leaderboard = controller.snapshot(_state(20.0, -100.0))
    assert leaderboard.session_state == "leaderboard"
    assert leaderboard.leaderboard[0].name == "Racer"
    assert leaderboard.high_score_rank == 1


def test_point_to_point_finishes_at_last_checkpoint_and_rejects_skips(
    tmp_path: Path,
) -> None:
    game_map = load_game_map(_MAP)
    course = replace(
        game_map.race_courses[0],
        course_id="sprint",
        checkpoint_element_ids=("southeast", "north"),
        lap_count=0,
    )
    controller = RaceController(
        game_map, course, _state(-20.0, -100.0), RaceTimeStore(tmp_path / "times.csv")
    )
    _drive(controller, [(20.0, -100.0)], 1)
    # Cross north westbound while bypassing southeast, the active checkpoint.
    skipped = _drive(controller, [(20.0, 100.0), (-20.0, 100.0)], 2)[-1]
    assert skipped.checkpoint_index == 0
    assert controller.is_playing
    snapshots = _drive(
        controller,
        [
            (20.0, 100.0),
            (20.0, -100.0),
            (130.0, -100.0),
            (150.0, -100.0),
            (150.0, 100.0),
            (-20.0, 100.0),
        ],
        4,
    )
    assert snapshots[3].target_label == "FINISH"
    assert snapshots[-1].final_time_us == 8_000_000
    assert not controller.is_playing


def test_backward_start_crossing_does_not_start(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    controller = RaceController(
        game_map,
        game_map.race_courses[0],
        _state(20.0, -100.0),
        RaceTimeStore(tmp_path / "times.csv"),
    )
    snapshot = _drive(controller, [(-20.0, -100.0)], 1)[-1]
    assert snapshot.session_state == "awaiting_start"
    assert snapshot.target_kind == "start"
    assert snapshot.event is None
    assert snapshot.elapsed_time_us == 0


def test_swept_crossing_advances_through_adjacent_gates(tmp_path: Path) -> None:
    game_map = load_game_map(_MAP)
    course = replace(
        game_map.race_courses[0],
        course_id="adjacent-gates",
        checkpoint_element_ids=("southeast", "east_south", "north"),
        lap_count=0,
    )
    controller = RaceController(
        game_map, course, _state(-20.0, -100.0), RaceTimeStore(tmp_path / "times.csv")
    )
    _drive(controller, [(20.0, -100.0), (144.0, -102.0)], 1)
    # One diagonal segment crosses the eastbound corner entrance and northbound exit.
    snapshot = _drive(controller, [(152.0, -94.0)], 3)[-1]
    assert snapshot.checkpoint_index == 2
    assert snapshot.target_element_id == "north"


def test_intersection_gate_spans_the_full_arterial(tmp_path: Path) -> None:
    game_map = load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    course = GameMapRaceCourse(
        course_id="wide-junction",
        spawn_id="start",
        start_element_id="west_road",
        checkpoint_element_ids=("center", "east_road"),
        lap_count=0,
    )
    controller = RaceController(
        game_map, course, _state(-30.0, 0.0), RaceTimeStore(tmp_path / "times.csv")
    )
    # The skewed junction must span at least four 3.6 m lanes plus two 0.6 m curbs.
    assert controller._gates["center"].length >= 15.6 - 0.15


def test_hairpin_gate_is_at_the_course_entry(tmp_path: Path) -> None:
    game_map = load_game_map(_MAPS / "hairpin.robotaxi.yaml")
    course = game_map.race_courses[0]
    controller = RaceController(
        game_map, course, _state(200.0, 20.0), RaceTimeStore(tmp_path / "times.csv")
    )
    gate = controller._gates["hairpin"]
    assert gate.distance(controller._surfaces["approach"]) < gate.distance(
        controller._surfaces["exit"]
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
        ({"spawn": "missing"}, "unknown spawn"),
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


def test_race_course_spawn_is_required(tmp_path: Path) -> None:
    document = yaml.safe_load(_MAP.read_text(encoding="utf-8"))
    del document["race_courses"][0]["spawn"]
    path = tmp_path / "missing-course-spawn.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(GameMapError, match="requires.*spawn"):
        load_game_map(path)
