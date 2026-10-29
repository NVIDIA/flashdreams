# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for capture overview evidence and missing observations."""

import json
from xml.etree import ElementTree

import pytest
from crazy_robotaxi.debug_overview import write_capture_overview
from crazy_robotaxi.debug_report import write_timing_report

pytestmark = pytest.mark.ci_cpu


def _write_capture(directory, records):
    (directory / "manifest.json").write_text(
        json.dumps({"capture_complete": False, "termination_reason": "test <stop>"})
    )
    (directory / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )


def test_overview_preserves_physical_route_epochs_and_native_completion(tmp_path):
    records = []
    for epoch in (0, 1):
        for index in range(3):
            records.append(
                {
                    "rollout_epoch": epoch,
                    "frame": index,
                    "simulation_timestamp_us": 100_000_000 + index * 1_000_000,
                    "physics_vehicle": {
                        "x_m": epoch * 100 + index * 3,
                        "y_m": index * 4,
                        "z_m": index * 2,
                        "speed_mps": 5,
                    },
                    "vehicle": {"x_m": 999, "y_m": 999},
                    "actor_collision": index == 1,
                    "off_road": index == 2,
                    "game": {
                        "checkpoint_index": index,
                        "completed_laps": int(index == 2),
                        "event": "race_complete" if epoch == 0 and index == 2 else None,
                        "session_state": "awaiting_name" if index == 2 else "racing",
                        "final_time_us": 2_000_000
                        if epoch == 0 and index == 2
                        else None,
                    },
                }
            )
    _write_capture(tmp_path, records)
    report = write_capture_overview(tmp_path)
    first, second = report["epochs"]
    assert [e["sampled_distance_m"] for e in report["epochs"]] == [10, 10]
    assert first["simulation_elapsed_s"] == second["simulation_elapsed_s"] == 2
    assert first["race_completed"] and first["completion_time_s"] == 2
    assert first["final_time_us"] == 2_000_000
    assert first["speed_range_mps"] == [5, 5]
    assert first["elevation_range_m"] == [0, 4]
    assert not second["race_completed"]  # Name-entry state alone isn't completion.
    assert first["collision_observations"] == first["off_road_observations"] == 1
    assert first["race_events"] == [{"time_s": 2, "event": "race_complete"}]
    assert not report["capture_complete"]
    assert report["generation_adherence"] == "not_evaluated"
    assert json.loads((tmp_path / "overview.json").read_text()) == report
    root = ElementTree.parse(tmp_path / "overview.svg").getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "test <stop>" in "".join(root.itertext())


def test_overview_does_not_bridge_missing_positions_or_invent_offroad(tmp_path):
    records = [
        {
            "rollout_epoch": 0,
            "frame": i,
            "simulation_timestamp_us": i * 1000,
            "vehicle": vehicle,
        }
        for i, vehicle in enumerate(
            [{"x_m": 0, "y_m": 0}, {}, {"x_m": 100, "y_m": 100}]
        )
    ]
    _write_capture(tmp_path, records)
    report = write_capture_overview(tmp_path)
    assert report["epochs"][0]["sampled_distance_m"] == 0
    assert report["epochs"][0]["off_road_observations"] is None
    assert report["epochs"][0]["final_time_us"] is None
    assert not report["epochs"][0]["race_completed"]


def test_timing_report_emits_empty_overview_for_early_failure(tmp_path):
    _write_capture(tmp_path, [])
    write_timing_report(tmp_path)
    report = json.loads((tmp_path / "overview.json").read_text())
    assert report["epochs"] == []
    assert "No observations recorded" in (tmp_path / "overview.svg").read_text()


def test_overview_prefers_tick_path_per_epoch_and_reports_missing_data(tmp_path):
    _write_capture(
        tmp_path,
        [
            {
                "rollout_epoch": epoch,
                "frame": i,
                "simulation_timestamp_us": i * 100_000,
                "vehicle": {"x_m": i * 2, "y_m": 0, "speed_mps": 1},
                "actor_collision": False,
                "static_barrier_collision": False,
            }
            for epoch in (1, 2)
            for i in (0, 1)
        ],
    )
    ticks = [
        {
            "tick": i,
            "simulation_timestamp_us": i * 10_000,
            "vehicle": {"x_m": x, "y_m": y, "z_m": i, "speed_mps": 2},
            "game": {"session_state": "playing"},
        }
        for i, x, y in [(0, 0, 0), (1, 1, 1), (3, 2, 0)]
    ]
    (tmp_path / "gameplay-1.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in ticks)
    )
    (tmp_path / "gameplay-2.jsonl").write_text(
        ""
    )  # Failed worker retains telemetry fallback.
    report = write_capture_overview(tmp_path)
    first, second = report["epochs"]
    assert first["source"] == "gameplay-1.jsonl"
    assert first["sample_kind"] == "physics_tick"
    assert first["sampled_distance_m"] == pytest.approx(2 * 2**0.5)
    assert first["samples"] == 3 and first["speed_range_mps"] == [2, 2]
    assert first["median_simulation_interval_s"] == pytest.approx(0.015)
    assert first["missing_sequence_observations"] == 1
    assert first["collision_observations"] is None
    assert first["unknown_collision_observations"] == 3
    assert first["off_road_observations"] is None
    assert second["source"] == "telemetry.jsonl"
    assert second["sampled_distance_m"] == 2
    assert second["collision_observations"] == 0
    svg = (tmp_path / "overview.svg").read_text()
    assert "gameplay-1.jsonl" in svg and "Missing IDs 1" in svg
    assert "Native checkpoint index" not in svg
    assert "Game state" in svg and "playing" in svg


def test_overview_taxi_score_and_phase_are_not_race_results(tmp_path):
    _write_capture(
        tmp_path,
        [
            {
                "rollout_epoch": 0,
                "frame": i,
                "simulation_timestamp_us": i * 1_000_000,
                "vehicle": {"x_m": i, "y_m": 0},
                "game": {
                    "phase": phase,
                    "score": score,
                    "session_state": "playing",
                    "event": event,
                },
            }
            for i, phase, score, event in [
                (0, "seeking_pickup", 0, None),
                (1, "driving_to_dropoff", 0, None),
                (2, "seeking_pickup", 500, "fare_complete"),
            ]
        ],
    )
    report = write_capture_overview(tmp_path)
    epoch = report["epochs"][0]
    assert not epoch["race_completed"] and epoch["race_events"] == []
    assert epoch["game_events"] == [{"time_s": 2, "event": "fare_complete"}]
    assert [p["phase"] for p in epoch["phase_changes"]] == [
        "seeking_pickup",
        "driving_to_dropoff",
        "seeking_pickup",
    ]
    svg = (tmp_path / "overview.svg").read_text()
    assert "Taxi score" in svg and "Score 500" in svg
    assert "Native checkpoint index" not in svg


def test_overview_bounds_chart_size_but_keeps_all_tick_statistics(tmp_path):
    _write_capture(tmp_path, [])
    ticks = [
        {
            "tick": i,
            "simulation_timestamp_us": i * 1000,
            "vehicle": {"x_m": i, "y_m": 0, "speed_mps": 50 if i == 3456 else 1},
        }
        for i in range(5001)
    ]
    (tmp_path / "gameplay-3.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in ticks)
    )
    report = write_capture_overview(tmp_path)
    assert report["epochs"][0]["samples"] == 5001
    assert report["epochs"][0]["sampled_distance_m"] == 5000
    assert report["epochs"][0]["speed_range_mps"] == [1, 50]
    root = ElementTree.parse(tmp_path / "overview.svg").getroot()
    curves = [
        e.attrib["d"]
        for e in root.iter("{http://www.w3.org/2000/svg}path")
        if e.attrib.get("stroke-width") == "2.2"
    ]
    assert curves and all(d.count("L ") + d.count("M ") <= 2000 for d in curves)
