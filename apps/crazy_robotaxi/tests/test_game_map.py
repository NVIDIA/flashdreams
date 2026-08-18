# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for Crazy Robotaxi node-graph maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from crazy_robotaxi import cli
from crazy_robotaxi.game import TaxiGameConfig, TaxiGameController
from crazy_robotaxi.navigation import NavigationLane, TaxiNavigationMap
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import (
    GameMapError,
    compile_game_map,
    load_game_map,
    write_game_map_preview,
)
from omnidreams_game_engine.game_map import compiler as game_map_compiler
from omnidreams_game_engine.game_map.types import (
    ResolvedGameMap,
    game_map_from_dict,
    game_map_to_dict,
)
from omnidreams_game_engine.scene_loader import load_scene_bundle
from omnidreams_game_engine.simulation.map_bounds import MapBounds
from omnidreams_game_engine.types import VehicleState

pytestmark = pytest.mark.ci_cpu

_MAPS = Path(__file__).parents[1] / "crazy_robotaxi" / "maps"
_STARTER_MAP = _MAPS / "minimal_loop.robotaxi.yaml"
_BOULEVARD_MAP = _MAPS / "boulevard_district.robotaxi.yaml"


def _write_map(tmp_path: Path, source: dict[str, object], name: str = "map") -> Path:
    path = tmp_path / f"{name}.robotaxi.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def _reachable(game_map: ResolvedGameMap, start_lane_id: str) -> set[str]:
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    pending = [start_lane_id]
    reached = {start_lane_id}
    while pending:
        for successor_id in lanes[pending.pop()].successor_ids:
            if successor_id not in reached:
                reached.add(successor_id)
                pending.append(successor_id)
    return reached


def test_bundled_maps_use_schema_version_1() -> None:
    starter = load_game_map(_STARTER_MAP)
    boulevard = load_game_map(_BOULEVARD_MAP)

    assert starter.schema_version == boulevard.schema_version == 1
    assert {node.node_type for node in starter.topology.nodes} == {
        "intersection",
        "cul_de_sac",
        "driveway",
        "parking_lot",
    }
    assert len(starter.topology.roads) == 2
    assert len(starter.topology.roads[0].bezier_spans_world) == 4
    assert len(boulevard.topology.nodes) == 44
    assert len(boulevard.topology.roads) == 41
    assert len(boulevard.topology.direct_links) == 10
    assert boulevard.default_spawn.lane_id == "central_boulevard:lane:2"


def test_unknown_root_fields_are_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["unexpected"] = []

    with pytest.raises(GameMapError, match="Map must contain exactly"):
        load_game_map(_write_map(tmp_path, source))


def test_topology_round_trip_is_lossless() -> None:
    original = load_game_map(_STARTER_MAP)
    restored = game_map_from_dict(game_map_to_dict(original))

    assert restored.topology == original.topology
    assert restored.topology.adjacency == original.topology.adjacency
    assert restored.default_spawn.lane_id == original.default_spawn.lane_id


def test_node_rotation_does_not_change_road_path(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    hub = next(node for node in source["nodes"] if node["id"] == "hub")
    baseline = load_game_map(_STARTER_MAP)
    hub["pose"]["rotation_deg"] = 45
    rotated = load_game_map(_write_map(tmp_path, source))

    baseline_road = next(
        lane for lane in baseline.lanes if lane.lane_id == "dead_end_road:lane:1"
    )
    rotated_road = next(
        lane for lane in rotated.lanes if lane.lane_id == "dead_end_road:lane:1"
    )
    baseline_direction = (
        baseline_road.centerline_world[-1, :2] - baseline_road.centerline_world[0, :2]
    )
    rotated_direction = (
        rotated_road.centerline_world[-1, :2] - rotated_road.centerline_world[0, :2]
    )
    baseline_direction /= np.linalg.norm(baseline_direction)
    rotated_direction /= np.linalg.norm(rotated_direction)
    np.testing.assert_allclose(rotated_direction, baseline_direction, atol=1.0e-5)


def test_askew_intersection_road_angles_are_geometry_driven() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    nodes = {node.node_id: node for node in game_map.topology.nodes}
    road = next(
        item
        for item in game_map.topology.roads
        if item.road_id == "eastern_north_approach"
    )
    start, end = nodes[road.from_node_id], nodes[road.to_node_id]
    bearing = np.degrees(np.arctan2(end.y_m - start.y_m, end.x_m - start.x_m)) % 360

    assert start.node_id == "eastern_gateway"
    assert start.rotation_deg == pytest.approx(0)
    assert bearing == pytest.approx(75.1, abs=0.2)


def test_multi_span_curve_is_one_topological_road() -> None:
    game_map = load_game_map(_STARTER_MAP)
    road = next(
        item for item in game_map.topology.roads if item.road_id == "neighborhood_loop"
    )
    road_lanes = [lane for lane in game_map.lanes if lane.element_id == road.road_id]

    assert road.from_node_id == road.to_node_id == "hub"
    assert len(road.bezier_spans_world) == 4
    assert len(road_lanes) == 2
    assert (
        max(
            np.max(
                np.linalg.norm(np.diff(lane.centerline_world[:, :2], axis=0), axis=1)
            )
            for lane in road_lanes
        )
        < 4.0
    )
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    assert any(
        "neighborhood_loop:lane:1" in lanes[successor].successor_ids
        for successor in lanes["neighborhood_loop:lane:1"].successor_ids
    )


def test_malformed_curve_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    loop = next(road for road in source["roads"] if road["id"] == "neighborhood_loop")
    loop["path"][-1]["end"] = {"x_m": 1, "y_m": 0}

    with pytest.raises(GameMapError, match="final path endpoint"):
        load_game_map(_write_map(tmp_path, source))


def test_cul_de_sac_must_terminate_exactly_one_road(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["roads"].append(
        {
            "id": "invalid_second_road",
            "from": "hub",
            "to": "dead_end",
            "profile": "neighborhood",
        }
    )

    with pytest.raises(GameMapError, match="terminate exactly one road"):
        load_game_map(_write_map(tmp_path, source))


def test_direct_driveway_links_are_not_authored_roads() -> None:
    game_map = load_game_map(_STARTER_MAP)
    road_ids = {road.road_id for road in game_map.topology.roads}
    link_ids = {link.link_id for link in game_map.topology.direct_links}
    implicit = {
        element.element_id
        for element in game_map.elements
        if element.element_type == "implicit_driveway"
    }

    assert "hub_to_lot_driveway" not in road_ids
    assert link_ids == {"hub_to_lot_driveway", "lot_driveway_to_lot"}
    assert implicit == link_ids
    width = np.ptp(
        next(
            element.surface_world[:, 0]
            for element in game_map.elements
            if element.element_id == "hub_to_lot_driveway"
        )
    )
    assert width == pytest.approx(6.4, abs=0.2)


def test_inline_driveway_does_not_split_road(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["links"] = [
        link for link in source["links"] if link["id"] != "hub_to_lot_driveway"
    ]
    driveway = next(node for node in source["nodes"] if node["id"] == "lot_driveway")
    driveway["pose"] = {"x_m": 4.2, "y_m": 15, "rotation_deg": 0}
    lot = next(node for node in source["nodes"] if node["id"] == "neighborhood_lot")
    lot["pose"] = {"x_m": 25, "y_m": 15, "rotation_deg": 0}
    source["road_attachments"] = [{"driveway": "lot_driveway", "road": "dead_end_road"}]
    game_map = load_game_map(_write_map(tmp_path, source))

    road = next(
        item for item in game_map.topology.roads if item.road_id == "dead_end_road"
    )
    attachment = game_map.topology.road_attachments[0]
    assert road.from_node_id == "hub"
    assert road.to_node_id == "dead_end"
    assert attachment.road_id == road.road_id
    assert sum(item.road_id == road.road_id for item in game_map.topology.roads) == 1
    assert any(
        successor.startswith("lot_driveway_to_lot")
        for lane in game_map.lanes
        if lane.element_id == road.road_id
        for successor in lane.successor_ids
    )


def test_inline_driveway_pose_and_outward_rotation_are_validated(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["links"] = [source["links"][1]]
    driveway = next(node for node in source["nodes"] if node["id"] == "lot_driveway")
    driveway["pose"] = {"x_m": 4.2, "y_m": 15, "rotation_deg": 180}
    source["road_attachments"] = [{"driveway": "lot_driveway", "road": "dead_end_road"}]

    with pytest.raises(GameMapError, match="rotation must point outward"):
        load_game_map(_write_map(tmp_path, source))


def test_parking_lot_accepts_multiple_driveways(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    hub = next(node for node in source["nodes"] if node["id"] == "hub")
    hub["geometry"] = {"width_m": 24, "depth_m": 24}
    first_driveway = next(
        node for node in source["nodes"] if node["id"] == "lot_driveway"
    )
    first_driveway["pose"] = {"x_m": 0, "y_m": -20, "rotation_deg": 270}
    lot = next(node for node in source["nodes"] if node["id"] == "neighborhood_lot")
    lot["pose"] = {"x_m": 6, "y_m": -45, "rotation_deg": 0}
    source["nodes"].append(
        {
            "id": "second_driveway",
            "type": "driveway",
            "profile": "parking_access",
            "pose": {"x_m": 12, "y_m": -20, "rotation_deg": 270},
            "geometry": {"width_m": 6.4},
        }
    )
    source["links"].extend(
        [
            {"id": "hub_to_second", "a": "hub", "b": "second_driveway"},
            {"id": "second_to_lot", "a": "second_driveway", "b": "neighborhood_lot"},
        ]
    )
    game_map = load_game_map(_write_map(tmp_path, source))

    lot_neighbors = [
        link
        for link in game_map.topology.direct_links
        if "neighborhood_lot" in {link.node_a_id, link.node_b_id}
    ]
    assert len(lot_neighbors) == 2
    assert any(lane.element_id == "neighborhood_lot:aisle:1" for lane in game_map.lanes)


def test_lane_graph_routes_to_and_from_parking_lot() -> None:
    game_map = load_game_map(_STARTER_MAP)
    outbound = _reachable(game_map, game_map.default_spawn.lane_id)
    returning = _reachable(game_map, "neighborhood_lot:lane:0")

    assert "neighborhood_lot:lane:1" in outbound
    assert "neighborhood_loop:lane:0" in returning


def test_parking_lots_compile_as_green_roadnet_masks() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    rows = game_map_compiler._road_marking_rows(game_map)
    lots = [node for node in game_map.topology.nodes if node.node_type == "parking_lot"]

    assert len(rows) == len(lots) == 5
    assert all(
        cast(dict[str, Any], row["road_marking"])["category"]
        == "ROI_POLYGON_ROADNET_MASK"
        for row in rows
    )
    assert not game_map.line_markings


def test_compiler_settings_remain_map_local(tmp_path: Path) -> None:
    baseline = load_game_map(_STARTER_MAP)
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["compiler"]["ground_margin_m"] = 35.0
    modified = load_game_map(_write_map(tmp_path, source))

    assert baseline.compiler_settings["ground_margin_m"] == pytest.approx(20)
    assert np.ptp(modified.ground_vertices[:, 0]) == pytest.approx(
        np.ptp(baseline.ground_vertices[:, 0]) + 30
    )


def test_lane_graph_initializes_navigation_and_gameplay() -> None:
    game_map = load_game_map(_STARTER_MAP)
    spawn = game_map.default_spawn
    lanes = tuple(
        NavigationLane(
            centerline_world=lane.centerline_world,
            road_edge_world=lane.roadside_edge_world
            if lane.allows_taxi_stops
            else None,
            allows_taxi_stops=lane.allows_taxi_stops,
            lane_id=lane.lane_id,
            successor_ids=lane.successor_ids,
        )
        for lane in game_map.lanes
    )
    navigation = TaxiNavigationMap(lanes, endpoint_snap_tolerance_m=1.0e-6)
    spawn_lane = next(lane for lane in lanes if lane.lane_id == spawn.lane_id)
    state = VehicleState(
        x_m=float(spawn.position_world[0]),
        y_m=float(spawn.position_world[1]),
        z_m=float(spawn.position_world[2]),
        yaw_rad=spawn.yaw_rad,
        speed_mps=0,
        steer_rad=0,
    )
    controller = TaxiGameController(
        scene_id=game_map.map_id,
        reference_route_world=spawn_lane.centerline_world,
        navigation_lanes=lanes,
        initial_state=state,
        config=TaxiGameConfig(enabled=True, seed=17),
    )

    assert len(navigation.sample_waypoints(spacing_m=20, offset_m=0)) > 2
    assert controller.snapshot(state).pickup_targets_xyz_m


def test_compile_preview_and_scene_discovery(tmp_path: Path) -> None:
    first = compile_game_map(_STARTER_MAP, cache_root=tmp_path / "cache")
    second = compile_game_map(_STARTER_MAP, cache_root=tmp_path / "cache")
    preview = write_game_map_preview(_STARTER_MAP, tmp_path / "preview.svg")
    options = cli._discover_scene_options(_MAPS, _STARTER_MAP)

    assert not first.cache_hit and second.cache_hit
    assert first.archive_path == second.archive_path
    assert preview.read_text(encoding="utf-8").startswith("<svg")
    assert (
        next(item for item in options if item.path == _STARTER_MAP.resolve()).label
        == "Minimal Loop and Parking Lot"
    )
    scene = load_scene_bundle(
        first.archive_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(),
    )
    assert scene.game_map is not None
    assert scene.game_map.topology == first.game_map.topology
    assert MapBounds.from_scene(scene) is not None
