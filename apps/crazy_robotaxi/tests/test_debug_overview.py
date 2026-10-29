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
