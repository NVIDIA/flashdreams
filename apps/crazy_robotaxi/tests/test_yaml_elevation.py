# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for schema-v2 elevation and schema-v1 map compatibility."""

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
    load_game_map_header,
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


def test_v2_road_height_matches_ground_paint_and_lane_seams(hill) -> None:
    game_map, ground = hill
    assert game_map.schema_version == 2
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
    assert flat.schema_version == 1
    assert all(node.z_m == 0 for node in flat.topology.nodes)
    assert flat.ground_vertices.shape == (4, 3)
    assert flat.ground_faces.shape == (2, 3)
    assert np.all(flat.ground_vertices[:, 2] == 0)
    assert all(np.all(lane.centerline_world[:, 2] == 0) for lane in flat.lanes)


def test_flat_map_upgrade_preserves_geometry(tmp_path: Path) -> None:
    document = yaml.safe_load((_MAPS / "debug_flat.robotaxi.yaml").read_text())
    source = tmp_path / "flat.robotaxi.yaml"
    resolved = []
    for version in (1, 2):
        document["schema_version"] = version
        source.write_text(yaml.safe_dump(document))
        assert load_game_map_header(source).map_id == document["id"]
        game_map = load_game_map(source)
        data = game_map_to_dict(game_map)
        assert game_map_to_dict(game_map_from_dict(data)) == data
        assert data.pop("schema_version") == version
        resolved.append(data)
    assert resolved[0] == resolved[1]


@pytest.mark.parametrize("height", [0, 20])
def test_v1_rejects_node_height_with_upgrade_hint(tmp_path: Path, height: int) -> None:
    document = yaml.safe_load((_MAPS / "debug_flat.robotaxi.yaml").read_text())
    document["nodes"][0]["pose"]["z_m"] = height
    source = tmp_path / "invalid.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(GameMapError, match="pose.z_m requires schema_version: 2"):
        load_game_map(source)


@pytest.mark.parametrize("version", [0, 3, True, 1.0, "2", None])
def test_invalid_schema_versions_fail_header_and_map_loading(
    tmp_path: Path, version: object
) -> None:
    document = yaml.safe_load((_MAPS / "debug_flat.robotaxi.yaml").read_text())
    document["schema_version"] = version
    source = tmp_path / "invalid.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document))
    for load in (load_game_map_header, load_game_map):
        with pytest.raises(GameMapError, match="Unsupported schema_version"):
            load(source)


@pytest.mark.parametrize("filename", ["parking_driveway", "intersection_geometry"])
def test_level_elevated_nodes_keep_connectors_and_parking_at_authored_height(
    tmp_path: Path, filename: str
) -> None:
    document = yaml.safe_load((_MAPS / f"{filename}.robotaxi.yaml").read_text())
    document["schema_version"] = 2
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


@pytest.mark.parametrize("sample_spacing_m", [1.5, 20.0])
def test_grade_spawn_compiles_directly_to_clipgt_without_opendrive(
    tmp_path: Path,
    sample_spacing_m: float,
) -> None:
    document = yaml.safe_load((_MAPS / "debug_hill.robotaxi.yaml").read_text())
    document["compiler"]["sample_spacing_m"] = sample_spacing_m
    document["traffic_count"] = 2
    document["spawns"][0].update(road="climb", distance_m=73)
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
    assert scene.game_map.schema_version == 2
    assert scene.ground_mesh_vertices is not None
    assert scene.ground_mesh_faces is not None
    spawn = scene.game_map.default_spawn
    assert spawn.position_world[2] > 0 and spawn.pitch_rad < 0
    np.testing.assert_allclose(scene.initial_rig_to_world[:3, 3], spawn.position_world)
    assert scene.initial_rig_to_world[2, 0] > 0
    ground = GroundSnapper(scene.ground_mesh_vertices, scene.ground_mesh_faces)
    assert ground.ground_height_at(*spawn.position_world) == pytest.approx(
        spawn.position_world[2], abs=1e-5
    )
    surface = ground.ground_surface_at(*spawn.position_world)
    assert surface is not None
    _, normal = surface
    assert float(scene.initial_rig_to_world[:3, 0] @ normal) == pytest.approx(
        0, abs=1e-6
    )
    paths = [
        points
        for lane in scene.game_map.lanes
        for points in (
            lane.centerline_world,
            lane.left_edge_world,
            lane.right_edge_world,
            lane.roadside_edge_world,
        )
    ]
    paths += [divider.polyline_world for divider in scene.game_map.lane_dividers]
    paths += [
        item.polyline_world
        for element in scene.game_map.elements
        for item in (*element.curbs, *element.road_boundaries)
    ]
    assert len(scene.game_map.traffic) == 2
    paths += [traffic.centerline_world for traffic in scene.game_map.traffic]
    for points in paths:
        for fraction in (0.25, 0.5, 0.75):
            between = points[:-1] * (1 - fraction) + points[1:] * fraction
            np.testing.assert_allclose(
                ground._raycast(between), between[:, 2], atol=0.003
            )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "uphill"])
def test_invalid_node_heights_fail_validation(tmp_path: Path, value) -> None:
    document = yaml.safe_load((_MAPS / "debug_flat.robotaxi.yaml").read_text())
    document["schema_version"] = 2
    document["nodes"][0]["pose"]["z_m"] = value
    source = tmp_path / "invalid.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(GameMapError, match="z_m"):
        load_game_map(source)
