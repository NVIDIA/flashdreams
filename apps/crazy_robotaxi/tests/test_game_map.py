# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for Crazy Robotaxi node-graph maps."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from crazy_robotaxi import cli
from crazy_robotaxi.game import TaxiGameConfig, TaxiGameController
from crazy_robotaxi.navigation import NavigationLane, TaxiNavigationMap
from crazy_robotaxi.scene import load_scene_data
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
from omnidreams_game_engine.types import VehicleState
from shapely.geometry import LineString, Point, Polygon

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


def _surface(game_map: ResolvedGameMap, element_id: str) -> Polygon:
    element = next(item for item in game_map.elements if item.element_id == element_id)
    return Polygon(element.surface_world[:, :2])


def _curb_lines(game_map: ResolvedGameMap) -> list[LineString]:
    return [
        LineString(curb.polyline_world[:, :2])
        for element in game_map.elements
        for curb in element.curbs
    ]


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
    for game_map in (starter, boulevard):
        for node in game_map.topology.nodes:
            if node.node_type != "intersection":
                continue
            assert set(node.geometry) in (
                {"intersection_arm_length_m"},
                {
                    "intersection_width_m",
                    "intersection_depth_m",
                    "intersection_arm_length_m",
                },
            ), node.node_id


def test_intersection_geometry_is_never_inferred(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    hub = next(node for node in source["nodes"] if node["id"] == "hub")
    del hub["intersection_arm_length_m"]

    with pytest.raises(GameMapError, match="missing attributes"):
        load_game_map(_write_map(tmp_path, source))


def test_unknown_root_fields_are_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["unexpected"] = []

    with pytest.raises(GameMapError, match="Map has unknown fields"):
        load_game_map(_write_map(tmp_path, source))


def test_boolean_is_not_accepted_as_a_numeric_setting(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["compiler"]["sample_spacing_m"] = True

    with pytest.raises(GameMapError, match="sample_spacing_m must be a number"):
        load_game_map(_write_map(tmp_path, source))


def test_element_ids_are_unique_across_kinds(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["roads"][1]["id"] = "hub"

    with pytest.raises(GameMapError, match="shared by a node and road"):
        load_game_map(_write_map(tmp_path, source))


def test_profiles_without_curbs_do_not_emit_collision_segments(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    for profile in source["profiles"].values():
        profile["curb"] = False
    for node in source["nodes"]:
        if "curb" in node:
            node["curb"] = False

    game_map = load_game_map(_write_map(tmp_path, source))

    assert not any(element.curbs for element in game_map.elements)


def test_profile_is_optional_when_attributes_are_direct(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    profile = source["profiles"]["neighborhood"]
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    del road["profile"]
    road.update(profile)

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved = next(
        item for item in game_map.topology.roads if item.road_id == road["id"]
    )

    assert resolved.profile_id is None
    assert resolved.attributes.lane_width_m == pytest.approx(3.6)


def test_direct_attributes_override_partial_profile_defaults(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    profile = dict(source["profiles"]["neighborhood"])
    del profile["lane_width_m"]
    source["profiles"]["partial"] = profile
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["profile"] = "partial"
    road["lane_width_m"] = 4.1

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved = next(
        item for item in game_map.topology.roads if item.road_id == road["id"]
    )

    assert resolved.profile_id == "partial"
    assert resolved.attributes.lane_width_m == pytest.approx(4.1)
    assert resolved.attributes.speed_limit_mps == pytest.approx(13.4)


def test_direct_attributes_override_values_present_in_profile(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["lane_width_m"] = 4.1
    hub = next(item for item in source["nodes"] if item["id"] == "hub")
    source["profiles"]["intersection_defaults"] = {
        "intersection_arm_length_m": 20,
        "curb": False,
    }
    hub["profile"] = "intersection_defaults"

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved_road = next(
        item for item in game_map.topology.roads if item.road_id == road["id"]
    )
    resolved_hub = next(
        item for item in game_map.topology.nodes if item.node_id == hub["id"]
    )

    assert resolved_road.attributes.lane_width_m == pytest.approx(4.1)
    assert resolved_hub.geometry["intersection_arm_length_m"] == pytest.approx(6.93)
    assert resolved_hub.attributes.curb is True


def test_profiles_root_is_optional_when_all_attributes_are_direct(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    profiles = source.pop("profiles")
    for road in source["roads"]:
        road.update(profiles[road.pop("profile")])
    for node in source["nodes"]:
        if "profile" in node:
            node.update(profiles[node.pop("profile")])

    game_map = load_game_map(_write_map(tmp_path, source))

    assert all(road.profile_id is None for road in game_map.topology.roads)
    assert all(node.profile_id is None for node in game_map.topology.nodes)


def test_profile_may_contain_attributes_irrelevant_to_consumer(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["profiles"]["neighborhood"]["culdesac_radius_m"] = 50

    game_map = load_game_map(_write_map(tmp_path, source))

    road = next(
        item for item in game_map.topology.roads if item.profile_id == "neighborhood"
    )
    assert road.attributes.lane_width_m == pytest.approx(3.6)


def test_missing_effective_attribute_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    del source["profiles"]["neighborhood"]["speed_limit_mps"]

    with pytest.raises(GameMapError, match="missing attributes.*speed_limit_mps"):
        load_game_map(_write_map(tmp_path, source))


def test_topology_round_trip_is_lossless() -> None:
    original = load_game_map(_STARTER_MAP)
    restored = game_map_from_dict(game_map_to_dict(original))

    assert restored.topology == original.topology
    assert restored.topology.adjacency == original.topology.adjacency
    assert len(restored.lane_dividers) == len(original.lane_dividers)
    for restored_divider, original_divider in zip(
        restored.lane_dividers, original.lane_dividers, strict=True
    ):
        assert restored_divider.divider_id == original_divider.divider_id
        assert restored_divider.lane_edges == original_divider.lane_edges
        np.testing.assert_array_equal(
            restored_divider.polyline_world, original_divider.polyline_world
        )
    assert restored.default_spawn.lane_id == original.default_spawn.lane_id
    assert [element.attributes for element in restored.elements] == [
        element.attributes for element in original.elements
    ]
    for restored_element, original_element in zip(
        restored.elements, original.elements, strict=True
    ):
        np.testing.assert_array_equal(
            restored_element.surface_world, original_element.surface_world
        )
        assert [curb.curb_id for curb in restored_element.curbs] == [
            curb.curb_id for curb in original_element.curbs
        ]
        for restored_curb, original_curb in zip(
            restored_element.curbs, original_element.curbs, strict=True
        ):
            np.testing.assert_array_equal(
                restored_curb.polyline_world, original_curb.polyline_world
            )


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


def test_roads_cannot_cross_without_a_connection_node(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["path"] = [
        {
            "control_points": [{"x_m": 45, "y_m": 0}, {"x_m": 45, "y_m": 30}],
            "end": {"x_m": 0, "y_m": 30},
        }
    ]

    with pytest.raises(GameMapError, match="Unrelated elements.*overlap"):
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


def test_intersection_surface_follows_incident_road_edges() -> None:
    game_map = load_game_map(_STARTER_MAP)
    surface = _surface(game_map, "hub")

    assert surface.convex_hull.area - surface.area > 10.0
    assert not surface.contains(Point(6.0, 6.0))


def test_cul_de_sac_has_full_width_flat_road_connection() -> None:
    game_map = load_game_map(_STARTER_MAP)
    cul_de_sac = _surface(game_map, "dead_end")
    road = _surface(game_map, "dead_end_road")
    boundary = np.asarray(cul_de_sac.exterior.coords)
    segment_lengths = np.linalg.norm(np.diff(boundary, axis=0), axis=1)

    assert cul_de_sac.distance(road) < 1.0e-3
    assert float(segment_lengths.max()) == pytest.approx(8.4, abs=0.05)


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
    assert width == pytest.approx(7.2, abs=0.2)
    first = _surface(game_map, "hub_to_lot_driveway")
    second = _surface(game_map, "lot_driveway_to_lot")
    driveway = _surface(game_map, "lot_driveway")
    assert first.intersection(second).area == pytest.approx(0.0, abs=1.0e-6)
    assert first.distance(driveway) == pytest.approx(0.0, abs=1.0e-6)
    assert second.distance(driveway) == pytest.approx(0.0, abs=1.0e-6)
    assert any(element.element_type == "driveway" for element in game_map.elements)


def test_inline_driveway_does_not_split_road(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["links"] = [
        link for link in source["links"] if link["id"] != "hub_to_lot_driveway"
    ]
    driveway = next(node for node in source["nodes"] if node["id"] == "lot_driveway")
    driveway["pose"] = {"x_m": 4.2, "y_m": 15, "rotation_deg": 0}
    lot = next(node for node in source["nodes"] if node["id"] == "neighborhood_lot")
    lot["pose"] = {"x_m": 15, "y_m": 15, "rotation_deg": 0}
    lot["parking_lot_width_m"] = 10
    lot["parking_lot_depth_m"] = 10
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
    barriers = _curb_lines(game_map)
    assert min(barrier.distance(Point(-4.2, 15.0)) for barrier in barriers) < 0.05
    assert min(barrier.distance(Point(4.2, 15.0)) for barrier in barriers) > 2.5


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
    hub["intersection_width_m"] = 24
    hub["intersection_depth_m"] = 24
    hub["intersection_arm_length_m"] = 13.2
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


def test_routing_connectors_are_not_world_model_conditioning() -> None:
    game_map = load_game_map(_STARTER_MAP)
    connectors = [lane for lane in game_map.lanes if ":connector:" in lane.lane_id]
    compiled_lane_ids = {
        cast(dict[str, str], row["key"])["label_class_id"]
        for row in game_map_compiler._lane_rows(game_map)
    }

    assert connectors
    assert all(not lane.conditioning_visible for lane in connectors)
    assert compiled_lane_ids.isdisjoint(lane.lane_id for lane in connectors)


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_every_authored_join_has_exact_non_overlapping_surfaces(
    source_path: Path,
) -> None:
    game_map = load_game_map(source_path)
    surfaces = {
        element.element_id: Polygon(element.surface_world[:, :2]).buffer(0)
        for element in game_map.elements
    }
    pairs: list[tuple[str, str]] = []
    for road in game_map.topology.roads:
        for node_id in (road.from_node_id, road.to_node_id):
            pairs.append((road.road_id, node_id))
    for link in game_map.topology.direct_links:
        for node_id in (link.node_a_id, link.node_b_id):
            pairs.append((link.link_id, node_id))
    pairs.extend(
        (attachment.road_id, attachment.driveway_node_id)
        for attachment in game_map.topology.road_attachments
    )
    for first_id, second_id in pairs:
        first, second = surfaces[first_id], surfaces[second_id]
        assert first.intersection(second).area <= 1.0e-4, (first_id, second_id)
        assert first.distance(second) <= 1.0e-4, (first_id, second_id)


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_compiled_curbs_belong_to_their_elements(source_path: Path) -> None:
    game_map = load_game_map(source_path)
    for element in game_map.elements:
        surface_boundary = Polygon(element.surface_world[:, :2]).boundary
        for curb in element.curbs:
            line = LineString(curb.polyline_world[:, :2])
            assert line.difference(surface_boundary.buffer(1.0e-4)).length < 1.0e-4
        if element.attributes.curb:
            assert element.curbs
            assert sum(curb.polyline_world.shape[0] for curb in element.curbs) > 2
        else:
            assert not element.curbs


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_every_authored_road_emits_every_profile_divider(source_path: Path) -> None:
    document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    game_map = load_game_map(source_path)
    actual: dict[str, int] = {}
    lane_elements = {lane.lane_id: lane.element_id for lane in game_map.lanes}
    for divider in game_map.lane_dividers:
        element_ids = {lane_elements[lane_id] for lane_id, _side in divider.lane_edges}
        assert len(element_ids) == 1
        element_id = element_ids.pop()
        actual[element_id] = actual.get(element_id, 0) + 1

    for road in document["roads"]:
        markings = document["profiles"][road["profile"]]["divider_markings"]
        expected = sum(marking["style"].upper() != "VIRTUAL" for marking in markings)
        assert actual.get(road["id"], 0) == expected, road["id"]


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_final_clipgt_archive_contains_all_authored_map_geometry(
    source_path: Path, tmp_path: Path
) -> None:
    document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    compiled = compile_game_map(source_path, cache_root=tmp_path / "cache")

    with zipfile.ZipFile(compiled.archive_path) as archive:
        lane_lines = pq.read_table(
            io.BytesIO(archive.read("clipgt/lane_line.parquet"))
        ).to_pylist()
        boundaries = pq.read_table(
            io.BytesIO(archive.read("clipgt/road_boundary.parquet"))
        )
        intersections = pq.read_table(
            io.BytesIO(archive.read("clipgt/intersection_area.parquet"))
        )

    labels = [row["key"]["label_class_id"] for row in lane_lines]
    for road in document["roads"]:
        markings = document["profiles"][road["profile"]]["divider_markings"]
        expected = sum(marking["style"].upper() != "VIRTUAL" for marking in markings)
        actual = sum(
            label.startswith(f"lane_line:{road['id']}:lane:") for label in labels
        )
        assert actual == expected, road["id"]
    assert boundaries.num_rows == sum(
        len(element.curbs) for element in compiled.game_map.elements
    )
    assert intersections.num_rows == sum(
        node.node_type == "intersection" for node in compiled.game_map.topology.nodes
    )


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
    preview_text = preview.read_text(encoding="utf-8")
    assert preview_text.startswith("<svg")
    assert "#ff453a" not in preview_text
    assert "#5f6673" in preview_text
    assert preview_text.count("<circle") == len(first.game_map.topology.nodes)
    assert preview_text.count("<text") == (
        len(first.game_map.topology.nodes)
        + len(first.game_map.topology.roads)
        + len(first.game_map.topology.direct_links)
    )
    for node in first.game_map.topology.nodes:
        assert f"{node.node_id} [node:{node.node_type}]" in preview_text
    for road in first.game_map.topology.roads:
        assert f"{road.road_id} [road:{road.profile_id};" in preview_text
    for link in first.game_map.topology.direct_links:
        assert f"{link.link_id} [access;" in preview_text
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
    taxi_scene = load_scene_data(scene)
    assert taxi_scene.navigation_lanes
    assert len(taxi_scene.curb_segments_world) == sum(
        len(curb.polyline_world) - 1
        for element in first.game_map.elements
        for curb in element.curbs
    )
