# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for elevation in the original node-graph YAML compiler."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import (
    GameMapError,
    compile_game_map,
    load_game_map,
)
from omnidreams_game_engine.game_map.compiler import _boundary_chunks
from omnidreams_game_engine.game_map.types import game_map_from_dict, game_map_to_dict
from omnidreams_game_engine.scene import load_scene_bundle
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper

pytestmark = pytest.mark.ci_cpu

_MAPS = Path(__file__).parent / "maps"


@pytest.fixture(scope="module")
def hill():
    game_map = load_game_map(_MAPS / "debug_hill.robotaxi.yaml")
    return game_map, GroundSnapper(game_map.ground_vertices, game_map.ground_faces)


def test_legacy_road_height_matches_ground_paint_and_lane_seams(hill) -> None:
    game_map, ground = hill
    assert game_map.schema_version == 1
    assert "source_format" not in game_map.compiler_settings
    assert max(node.z_m for node in game_map.topology.nodes) == 32
    lane = next(lane for lane in game_map.lanes if lane.lane_id == "climb:lane:2")
    for t, expected in ((0, 0), (0.25, 3.3125), (0.5, 16), (0.75, 28.6875), (1, 32)):
        point = lane.centerline_world[0] * (1 - t) + lane.centerline_world[-1] * t
        assert ground.ground_height_at(*point) == pytest.approx(expected, abs=0.003)
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    for lane in lanes.values():
        for successor in lane.successor_ids:
            assert lane.centerline_world[-1, 2] == pytest.approx(
                lanes[successor].centerline_world[0, 2], abs=1e-5
            )
    paths = [divider.polyline_world for divider in game_map.lane_dividers]
    paths += [
        item.polyline_world
        for element in game_map.elements
        for item in (*element.curbs, *element.road_boundaries)
    ]
    for points in paths:
        np.testing.assert_allclose(ground._raycast(points), points[:, 2], atol=1e-4)
    for points in _boundary_chunks(game_map):
        np.testing.assert_allclose(ground._raycast(points), points[:, 2], atol=1e-4)
    triangles = game_map.ground_vertices[game_map.ground_faces]
    assert np.all(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[
            :, 2
        ]
        > 0
    )


def test_elevated_metadata_round_trips_and_flat_defaults_remain_flat(hill) -> None:
    game_map, _ = hill
    data = game_map_to_dict(game_map)
    assert game_map_to_dict(game_map_from_dict(data)) == data
    flat = load_game_map(_MAPS / "debug_flat.robotaxi.yaml")
    assert all(node.z_m == 0 for node in flat.topology.nodes)
    assert flat.ground_vertices.shape == (4, 3)
    assert flat.ground_faces.shape == (2, 3)
    assert np.all(flat.ground_vertices[:, 2] == 0)
    assert all(np.all(lane.centerline_world[:, 2] == 0) for lane in flat.lanes)


@pytest.mark.parametrize("filename", ["parking_driveway", "intersection_geometry"])
def test_level_elevated_nodes_keep_connectors_and_parking_at_authored_height(
    tmp_path: Path, filename: str
) -> None:
    document = yaml.safe_load((_MAPS / f"{filename}.robotaxi.yaml").read_text())
    for node in document["nodes"]:
        if "pose" in node:
            node["pose"]["z_m"] = 20
    source = tmp_path / "raised.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document, sort_keys=False))
    game_map = load_game_map(source)
    assert all(node.z_m == 20 for node in game_map.topology.nodes)
    for element in game_map.elements:
        np.testing.assert_allclose(element.surface_world[:, 2], 20, atol=1e-4)
    for lane in game_map.lanes:
        np.testing.assert_allclose(lane.centerline_world[:, 2], 20, atol=1e-4)
    assert game_map.default_spawn.position_world[2] == pytest.approx(20)


def test_grade_spawn_compiles_directly_to_clipgt_without_opendrive(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load((_MAPS / "debug_hill.robotaxi.yaml").read_text())
    document["spawns"][0].update(road="climb", distance_m=60)
    source = tmp_path / "grade.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document, sort_keys=False))
    compiled = compile_game_map(source, cache_root=tmp_path / "cache")
    with zipfile.ZipFile(compiled.archive_path) as archive:
        assert "map.xodr" not in archive.namelist()
    scene = load_scene_bundle(
        compiled.archive_path,
        "camera_front_wide_120fov",
        RasterConfig(width=64, height=32),
    )
    assert scene.game_map is not None
    assert scene.ground_mesh_vertices is not None
    assert scene.ground_mesh_faces is not None
    spawn = scene.game_map.default_spawn
    assert spawn.position_world[2] > 0 and spawn.pitch_rad < 0
    np.testing.assert_allclose(scene.initial_rig_to_world[:3, 3], spawn.position_world)
    assert scene.initial_rig_to_world[2, 0] > 0
    ground = GroundSnapper(scene.ground_mesh_vertices, scene.ground_mesh_faces)
    assert ground.ground_height_at(*spawn.position_world) == pytest.approx(
        spawn.position_world[2], abs=0.003
    )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "uphill"])
def test_invalid_node_heights_fail_validation(tmp_path: Path, value) -> None:
    document = yaml.safe_load((_MAPS / "debug_flat.robotaxi.yaml").read_text())
    document["nodes"][0]["pose"]["z_m"] = value
    source = tmp_path / "invalid.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(GameMapError, match="z_m"):
        load_game_map(source)
