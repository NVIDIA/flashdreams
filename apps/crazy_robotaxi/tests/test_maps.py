# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU validation for shipped semantic maps."""

import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from omnidreams_game_engine.game_map import load_game_map, load_game_map_header
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.scene import SceneRequest, load_scene
from PIL import Image

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "filename",
    ["boulevard_district.robotaxi.yaml", "flashdreams_raceway.robotaxi.yaml"],
)
def test_shipped_map_is_valid(filename: str) -> None:
    path = Path(__file__).parents[1] / "crazy_robotaxi" / "maps" / filename
    game_map = load_game_map(path)

    assert game_map.map_id.startswith("crazy-robotaxi-")
    assert game_map.spawns
    assert game_map.lanes


def test_compiled_map_uses_canonical_spawn_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(__file__).parent / "maps" / "race_course.robotaxi.yaml"
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    game_map = load_game_map(path)

    scene = load_scene(
        SceneRequest(map_path=path),
        RasterConfig(width=64, height=32, compute_device="automatic"),
    )

    assert scene.prompt == game_map.default_spawn.prompt
    assert scene.initial_rgb.shape == (32, 64, 3)
    with zipfile.ZipFile(scene.scene_path) as archive:
        names = set(archive.namelist())
    assert {"prompt.txt", "first_image.png"} <= names
    assert not any(name.startswith(("prompt_", "first_image_")) for name in names)


def test_map_menu_thumbnail_falls_back_to_first_authored_spawn_image(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(
        (Path(__file__).parent / "maps" / "race_course.robotaxi.yaml").read_text(
            encoding="utf-8"
        )
    )
    first_spawn = document["spawns"][0]
    first_spawn.pop("image")
    second_spawn = dict(first_spawn)
    second_spawn.update(
        {
            "id": "second",
            "distance_m": 35,
            "image": "package://omnidreams_game_engine/screenshot.jpg",
        }
    )
    document["spawns"].append(second_spawn)
    document["race_courses"][0]["spawn"] = "second"
    path = tmp_path / "thumbnail.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    fallback_header = load_game_map_header(path)

    assert fallback_header.menu_thumbnail_path is not None
    assert fallback_header.menu_thumbnail_path.name == "screenshot.jpg"
    assert fallback_header.race_courses[0].spawn_id == "second"
    assert (
        fallback_header.race_courses[0].spawn_image_path
        == fallback_header.menu_thumbnail_path
    )

    thumbnail_path = tmp_path / "menu.png"
    Image.new("RGB", (16, 9), "blue").save(thumbnail_path)
    document["menu_thumbnail"] = "menu.png"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert load_game_map_header(path).menu_thumbnail_path == thumbnail_path.resolve()


def test_scene_request_selects_nondefault_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).parent / "maps" / "race_course.robotaxi.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    second_spawn = dict(document["spawns"][0])
    second_spawn.update(
        {
            "id": "course-start",
            "distance_m": 35,
            "prompt": "A distinct second race spawn.",
        }
    )
    document["spawns"].append(second_spawn)
    document["race_courses"][0]["spawn"] = "course-start"
    path = tmp_path / "second-spawn.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path / "cache"))

    default_scene = load_scene(
        SceneRequest(map_path=path),
        RasterConfig(width=64, height=32, compute_device="automatic"),
    )
    course_scene = load_scene(
        SceneRequest(map_path=path, spawn_id="course-start"),
        RasterConfig(width=64, height=32, compute_device="automatic"),
    )

    assert default_scene.prompt != course_scene.prompt
    assert course_scene.prompt == "A distinct second race spawn."
    assert not np.array_equal(
        default_scene.initial_rig_to_world,
        course_scene.initial_rig_to_world,
    )
    assert course_scene.game_map is not None
    assert course_scene.game_map.race_courses[0].spawn_id == "course-start"


def test_boulevard_traffic_turns_are_continuous_and_physically_limited() -> None:
    path = (
        Path(__file__).parents[1]
        / "crazy_robotaxi"
        / "maps"
        / "boulevard_district.robotaxi.yaml"
    )
    game_map = load_game_map(path)
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    node_types = {node.node_id: node.node_type for node in game_map.topology.nodes}

    connectors = (
        lane
        for lane in game_map.lanes
        if ":connector:" in lane.lane_id and node_types[lane.element_id] != "cul_de_sac"
    )
    for connector in connectors:
        sources = [
            lane for lane in game_map.lanes if connector.lane_id in lane.successor_ids
        ]
        assert len(sources) == 1
        target = lanes[connector.successor_ids[0]]
        tangent_pairs = (
            (
                sources[0].centerline_world[-1, :2]
                - sources[0].centerline_world[-2, :2],
                connector.centerline_world[1, :2] - connector.centerline_world[0, :2],
            ),
            (
                connector.centerline_world[-1, :2] - connector.centerline_world[-2, :2],
                target.centerline_world[1, :2] - target.centerline_world[0, :2],
            ),
        )
        for first, second in tangent_pairs:
            cosine = float(
                np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
            )
            assert cosine >= 0.97, connector.lane_id

    cul_de_sacs = {
        node.node_id
        for node in game_map.topology.nodes
        if node.node_type == "cul_de_sac"
    }
    for vehicle in game_map.traffic:
        segments = np.diff(vehicle.centerline_world[:, :2], axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        assert np.all(lengths >= 0.25 - 1.0e-5), vehicle.vehicle_id

        headings = np.arctan2(segments[:, 1], segments[:, 0])
        heading_changes = np.abs(
            (headings - np.roll(headings, 1) + np.pi) % (2.0 * np.pi) - np.pi
        )
        previous_lengths = np.roll(lengths, 1)
        previous_speeds = np.roll(vehicle.speed_limits_mps[:-1], 1)
        segment_speeds = np.maximum(
            np.minimum(previous_speeds, vehicle.speed_limits_mps[:-1]), 0.1
        )
        yaw_rates = heading_changes * segment_speeds / previous_lengths
        assert np.max(yaw_rates) <= 1.201, vehicle.vehicle_id

        for index, heading_change in enumerate(heading_changes):
            previous = (index - 1) % len(vehicle.route_element_ids)
            current = index % len(vehicle.route_element_ids)
            if {
                vehicle.route_element_ids[previous],
                vehicle.route_element_ids[current],
            }.isdisjoint(cul_de_sacs):
                assert heading_change <= math.radians(30.0), (
                    vehicle.vehicle_id,
                    index,
                )
