# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU isolation checks for paired straight-line first-frame comparisons."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML
from omnidreams_game_engine.game_map import load_game_map
from apps.crazy_robotaxi.tests.run_first_frame_comparison import (
    MAP_NAMES,
    capture_command,
    check_matching_records,
    check_straight_motion,
    prepare_pair,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("name", MAP_NAMES)
def test_pair_preserves_geometry_prompts_and_spawn(name: str, tmp_path: Path) -> None:
    maps = Path(__file__).parents[1] / "crazy_robotaxi/maps"
    source = maps / f"{name}.robotaxi.yaml"
    reference = (
        Path(__file__).parents[2]
        / "omnidreams_game_engine/omnidreams_game_engine/screenshot.jpg"
    )
    before = source.read_bytes()
    paths = prepare_pair(source, reference, tmp_path / "inputs")
    yaml = YAML(typ="safe")
    original = yaml.load(before)
    first, second = (
        yaml.load(paths[label].read_text()) for label in ("reference", "candidate")
    )
    assert Path(first["spawns"][0]["image"]).read_bytes() == reference.read_bytes()
    assert (
        Path(second["spawns"][0]["image"]).read_bytes()
        == (maps / original["spawns"][0]["image"]).read_bytes()
    )
    assert first["spawns"][0].pop("image") != second["spawns"][0].pop("image")
    assert first == second
    original["traffic_count"] = 0
    original.pop("traffic", None)
    original["spawns"][0].pop("image")
    assert first == original
    assert source.read_bytes() == before
    assert not load_game_map(paths["reference"]).traffic
    assert not load_game_map(paths["candidate"]).traffic
    with pytest.raises(FileExistsError):
        prepare_pair(source, reference, tmp_path / "inputs")


def test_commands_use_isolated_settings_and_only_neutral_or_forward_controls(
    tmp_path: Path,
) -> None:
    command = capture_command(
        "test-model", tmp_path / "reference.robotaxi.yaml", tmp_path / "capture", 7, 600
    )
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--game-mode") + 1] == "free-roam"
    assert command[command.index("--config") + 1] == str(tmp_path / "settings.yaml")
    assert "--no-live-edit-dynamic-prompts" in command
    assert "--no-live-edit-weather" in command
    script = YAML(typ="safe").load(
        (Path(__file__).parent / "scenarios/first_frame_acceleration.yaml").read_text()
    )
    assert script == {
        "steps": [{"frames": 15}, {"frames": 90, "throttle": 1, "steer": 0}]
    }


def _records() -> list[dict]:
    records = []
    yaw = math.pi / 3
    for index in range(105):
        time = max(0, index - 14) / 30
        vehicle = {
            "x_m": 30 + 2 * time**2 * math.cos(yaw),
            "y_m": -3 + 2 * time**2 * math.sin(yaw),
            "speed_mps": 4 * time,
            "yaw_rad": yaw,
            "ragdoll_active": False,
        }
        pose = np.eye(4)
        pose[:2, 3] = [vehicle["x_m"], vehicle["y_m"]]
        records.append(
            {
                "frame": index,
                "simulation_timestamp_us": index * 33333,
                "command": {
                    "throttle": 0 if index < 15 else 1,
                    "steer": 0,
                    "brake": 0,
                    "handbrake": False,
                    "reverse": False,
                    "stop": False,
                },
                "vehicle": vehicle,
                "physics_vehicle": vehicle.copy(),
                "rig_pose_world": pose.tolist(),
                "actor_collision": False,
                "static_barrier_collision": False,
            }
        )
    return records


def test_accepts_acceleration_along_authored_heading() -> None:
    records = _records()
    summary = check_straight_motion(records)
    assert summary["distance_forward_m"] == pytest.approx(18)
    assert summary["final_speed_mps"] == 12
    check_matching_records(records, copy.deepcopy(records))


@pytest.mark.parametrize(
    "failure",
    [
        "steering",
        "curb",
        "actor",
        "ragdoll",
        "yaw",
        "lateral",
        "frozen",
        "nan",
        "truncated",
    ],
)
def test_rejects_confounding_motion(failure: str) -> None:
    records = _records()
    if failure == "steering":
        records[50]["command"]["steer"] = 0.1
    elif failure in ("curb", "actor"):
        records[50][
            "static_barrier_collision" if failure == "curb" else "actor_collision"
        ] = True
    elif failure == "ragdoll":
        records[50]["physics_vehicle"]["ragdoll_active"] = True
    elif failure == "yaw":
        records[50]["vehicle"]["yaw_rad"] += 0.1
    elif failure == "lateral":
        records[50]["vehicle"]["y_m"] += 1
    elif failure == "frozen":
        for record in records:
            record["vehicle"] = records[0]["vehicle"].copy()
    elif failure == "nan":
        records[50]["vehicle"]["x_m"] = float("nan")
    else:
        records.pop()
    with pytest.raises(AssertionError):
        check_straight_motion(records)


@pytest.mark.parametrize(
    "field",
    [
        "frame",
        "simulation_timestamp_us",
        "command",
        "rig_pose_world",
        "physics_vehicle",
    ],
)
def test_pair_rejects_different_inputs_or_trajectory(field: str) -> None:
    reference = _records()
    candidate = copy.deepcopy(reference)
    if field in ("frame", "simulation_timestamp_us"):
        candidate[50][field] += 1
    elif field == "command":
        candidate[50][field]["throttle"] = 0.5
    elif field == "rig_pose_world":
        candidate[50][field][0][3] += 0.1
    else:
        candidate[50][field]["x_m"] += 0.1
    with pytest.raises(AssertionError):
        check_matching_records(reference, candidate)
