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


def _road_joint_map(*, curved_approach: bool = False) -> dict[str, object]:
    """Build a compact two-road map around one road joint."""
    approach: dict[str, object] = {
        "id": "approach",
        "from": "west_end",
        "to": "bend",
        "profile": "street",
    }
    if curved_approach:
        approach["path"] = [{"x_m": -22, "y_m": -7}]
    return {
        "schema_version": 1,
        "id": "road-joint-test",
        "name": "Road Joint Test",
        "compiler": {
            "sample_spacing_m": 0.5,
            "ground_margin_m": 5,
            "intersection_connector_samples": 8,
        },
        "profiles": {
            "street": {
                "lane_width_m": 3.6,
                "curb_offset_m": 0.6,
                "lanes": ["backward", "forward"],
                "speed_limit_mps": 13.4,
                "lane_marking": {"style": "SOLID_GROUP", "color": "YELLOW"},
                "divider_markings": [{"style": "SOLID_GROUP", "color": "YELLOW"}],
            }
        },
        "nodes": [
            {
                "id": "west_end",
                "type": "cul_de_sac",
                "pose": {"x_m": -45, "y_m": 0},
                "culdesac_radius_m": 8,
            },
            {
                "id": "bend",
                "type": "road_joint",
                "pose": {"x_m": 0, "y_m": 0},
            },
            {
                "id": "east_end",
                "type": "cul_de_sac",
                "pose": {"x_m": 35, "y_m": 35},
                "culdesac_radius_m": 8,
            },
        ],
        "roads": [
            approach,
            {
                "id": "exit",
                "from": "bend",
                "to": "east_end",
                "profile": "street",
                "speed_limit_mps": 8,
            },
        ],
        "spawns": [
            {
                "id": "taxi_start",
                "road": "approach",
                "lane": 1,
                "distance_m": 5,
                "variants": {
                    "default": {
                        "image": "package://omnidreams_game_engine/screenshot.jpg",
                        "prompt": "A road bending through a quiet neighborhood.",
                    }
                },
            }
        ],
    }


def _intersection_transition_map(
    transition_length_m: float = 20,
) -> dict[str, object]:
    """Build a four-way intersection with one narrower through arm."""
    common = {
        "curb_offset_m": 0.6,
        "speed_limit_mps": 12,
        "lane_marking": {"style": "DASHED_SINGLE", "color": "WHITE"},
    }
    return {
        "schema_version": 1,
        "id": "intersection-transition-test",
        "name": "Intersection Transition Test",
        "compiler": {
            "sample_spacing_m": 1,
            "ground_margin_m": 10,
            "intersection_connector_samples": 8,
        },
        "profiles": {
            "wide": {
                **common,
                "lane_width_m": 3.6,
                "lanes": ["backward", "backward", "forward", "forward"],
                "divider_markings": [
                    {"style": "DASHED_SINGLE", "color": "WHITE"},
                    {"style": "SOLID_GROUP", "color": "YELLOW"},
                    {"style": "DASHED_SINGLE", "color": "WHITE"},
                ],
            },
            "narrow": {
                **common,
                "lane_width_m": 3.2,
                "curb_offset_m": 0.5,
                "curb": False,
                "lanes": ["backward", "forward"],
                "divider_markings": [{"style": "SOLID_GROUP", "color": "YELLOW"}],
            },
        },
        "nodes": [
            {
                "id": "center",
                "type": "intersection",
                "pose": {"x_m": 0, "y_m": 0},
                "lane_transition_length_m": transition_length_m,
            },
            *(
                {
                    "id": node_id,
                    "type": "cul_de_sac",
                    "pose": {"x_m": x_m, "y_m": y_m},
                    "culdesac_radius_m": 12,
                }
                for node_id, x_m, y_m in (
                    ("north", 0, 100),
                    ("south", 0, -100),
                    ("east", 100, 0),
                    ("west", -100, 0),
                )
            ),
        ],
        "roads": [
            {
                "id": "north_road",
                "from": "center",
                "to": "north",
                "profile": "narrow",
            },
            {
                "id": "south_road",
                "from": "center",
                "to": "south",
                "profile": "wide",
            },
            {
                "id": "east_road",
                "from": "center",
                "to": "east",
                "profile": "narrow",
            },
            {
                "id": "west_road",
                "from": "center",
                "to": "west",
                "profile": "narrow",
            },
        ],
        "spawns": [
            {
                "id": "start",
                "road": "south_road",
                "lane": 2,
                "distance_m": 10,
                "variants": {
                    "default": {
                        "image": "package://omnidreams_game_engine/screenshot.jpg",
                        "prompt": "A road widening at an intersection.",
                    }
                },
            }
        ],
    }


def _self_loop_map() -> dict[str, object]:
    """Restore a compact self-loop fixture independent of the bundled demo."""
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["nodes"] = [node for node in source["nodes"] if node["type"] != "road_joint"]
    source["roads"] = [
        road for road in source["roads"] if road["id"] == "dead_end_road"
    ]
    source["roads"].insert(
        0,
        {
            "id": "neighborhood_loop",
            "from": "hub",
            "to": "hub",
            "profile": "neighborhood",
            "path": [
                {"x_m": 45, "y_m": 15},
                {"x_m": 45, "y_m": 50},
                {"x_m": -45, "y_m": 50},
                {"x_m": -45, "y_m": 15},
            ],
        },
    )
    source["spawns"][0]["road"] = "neighborhood_loop"
    return cast(dict[str, object], source)


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
        "road_joint",
        "cul_de_sac",
        "parking_lot",
    }
    assert len(starter.topology.roads) == 6
    assert len(boulevard.topology.nodes) == 75
    assert len(boulevard.topology.roads) == 81
    assert len(boulevard.topology.parking_accesses) == 8
    assert boulevard.default_spawn.lane_id == "spawn_arterial:lane:2"
    for game_map in (starter, boulevard):
        for node in game_map.topology.nodes:
            if node.node_type != "intersection":
                continue
            assert set(node.geometry) == {"lane_transition_length_m"}, node.node_id


@pytest.mark.parametrize(
    "field",
    [
        "intersection_arm_length_m",
        "intersection_width_m",
        "intersection_depth_m",
    ],
)
def test_authored_intersection_geometry_is_rejected(tmp_path: Path, field: str) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    hub = next(node for node in source["nodes"] if node["id"] == "hub")
    hub[field] = 24

    with pytest.raises(GameMapError, match="unknown attributes"):
        load_game_map(_write_map(tmp_path, source))


def test_lane_transition_length_is_rejected_on_cul_de_sac(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    dead_end = next(node for node in source["nodes"] if node["id"] == "dead_end")
    dead_end["lane_transition_length_m"] = 10

    with pytest.raises(GameMapError, match="unknown attributes"):
        load_game_map(_write_map(tmp_path, source))


def test_lane_transition_length_must_be_nonnegative(tmp_path: Path) -> None:
    with pytest.raises(GameMapError, match="must be nonnegative"):
        load_game_map(_write_map(tmp_path, _intersection_transition_map(-1)))


def test_unknown_root_fields_are_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["unexpected"] = []

    with pytest.raises(GameMapError, match="Map has unknown fields"):
        load_game_map(_write_map(tmp_path, source))


def test_parking_accesses_are_not_a_top_level_authoring_field(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["parking_accesses"] = []

    with pytest.raises(GameMapError, match="unknown fields.*parking_accesses"):
        load_game_map(_write_map(tmp_path, source))


def test_legacy_link_topology_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    source["links"] = []

    with pytest.raises(GameMapError, match="unknown fields.*links"):
        load_game_map(_write_map(tmp_path, source))


def test_parking_lot_vertices_must_be_clockwise(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    lot = next(node for node in source["nodes"] if node["type"] == "parking_lot")
    lot["vertices"].reverse()

    with pytest.raises(GameMapError, match="must be clockwise"):
        load_game_map(_write_map(tmp_path, source))


def test_parking_opening_vertex_must_select_a_polygon_edge(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    lot = next(node for node in source["nodes"] if node["type"] == "parking_lot")
    lot["opening_vertex"] = len(lot["vertices"]) + 1

    with pytest.raises(GameMapError, match="opening_vertex must be between"):
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
        if node["type"] in {"intersection", "cul_de_sac"}:
            node["curb"] = False

    source_path = _write_map(tmp_path, source)
    game_map = load_game_map(source_path)

    assert all(
        not element.curbs
        for element in game_map.elements
        if element.element_type not in {"parking_lot", "parking_access"}
    )
    assert all(
        element.curbs
        for element in game_map.elements
        if element.element_type in {"parking_lot", "parking_access"}
    )
    assert any(element.road_boundaries for element in game_map.elements)
    rows = game_map_compiler._boundary_rows(game_map)
    assert len(rows) == sum(
        len(element.road_boundaries) for element in game_map.elements
    )
    assert all(row["road_boundary"]["category"] == "road_boundary" for row in rows)
    compiled = compile_game_map(source_path, cache_root=tmp_path / "cache")
    with zipfile.ZipFile(compiled.archive_path) as archive:
        archived_rows = pq.read_table(
            io.BytesIO(archive.read("clipgt/road_boundary.parquet"))
        ).to_pylist()
    assert len(archived_rows) == len(rows)
    assert all(
        row["road_boundary"]["category"] == "road_boundary" for row in archived_rows
    )


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
        "curb": False,
    }
    hub["profile"] = "intersection_defaults"
    hub["curb"] = True

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved_road = next(
        item for item in game_map.topology.roads if item.road_id == road["id"]
    )
    resolved_hub = next(
        item for item in game_map.topology.nodes if item.node_id == hub["id"]
    )

    assert resolved_road.attributes.lane_width_m == pytest.approx(4.1)
    assert resolved_hub.geometry == {"lane_transition_length_m": 0}
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
        assert [
            boundary.boundary_id for boundary in restored_element.road_boundaries
        ] == [boundary.boundary_id for boundary in original_element.road_boundaries]
        for restored_boundary, original_boundary in zip(
            restored_element.road_boundaries,
            original_element.road_boundaries,
            strict=True,
        ):
            np.testing.assert_array_equal(
                restored_boundary.polyline_world, original_boundary.polyline_world
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


def test_legacy_serialized_curbs_supply_missing_road_boundaries() -> None:
    serialized = game_map_to_dict(load_game_map(_STARTER_MAP))
    for element in serialized["elements"]:
        element.pop("road_boundaries")

    restored = game_map_from_dict(serialized)

    for element in restored.elements:
        assert len(element.road_boundaries) == len(element.curbs)
        for boundary, curb in zip(element.road_boundaries, element.curbs, strict=True):
            np.testing.assert_array_equal(boundary.polyline_world, curb.polyline_world)


def test_serialized_road_boundaries_do_not_require_physical_curbs() -> None:
    serialized = game_map_to_dict(load_game_map(_STARTER_MAP))
    for element in serialized["elements"]:
        element.pop("curbs")

    restored = game_map_from_dict(serialized)

    assert any(element.road_boundaries for element in restored.elements)
    assert not any(element.curbs for element in restored.elements)


def test_node_rotation_is_not_an_authored_pose_field(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    hub = next(node for node in source["nodes"] if node["id"] == "hub")
    hub["pose"]["rotation_deg"] = 45

    with pytest.raises(GameMapError, match="pose requires x_m and y_m"):
        load_game_map(_write_map(tmp_path, source))


def test_curb_defaults_to_true_for_roads_and_boundary_nodes() -> None:
    game_map = load_game_map(_STARTER_MAP)

    assert all(road.attributes.curb for road in game_map.topology.roads)
    assert all(
        node.attributes.curb
        for node in game_map.topology.nodes
        if node.node_type in {"intersection", "cul_de_sac"}
    )
    serialized = game_map_to_dict(game_map)
    assert all("rotation_deg" not in node for node in serialized["topology"]["nodes"])


def test_askew_intersection_road_angles_are_geometry_driven() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    nodes = {node.node_id: node for node in game_map.topology.nodes}
    road = next(
        item
        for item in game_map.topology.roads
        if item.road_id == "diagonal_north_lower"
    )
    start, end = nodes[road.from_node_id], nodes[road.to_node_id]
    bearing = np.degrees(np.arctan2(end.y_m - start.y_m, end.x_m - start.x_m)) % 360

    assert start.node_id == "diagonal_arterial_crossing"
    assert bearing == pytest.approx(81.0, abs=0.2)


def test_road_joint_trims_roads_and_emits_visible_curve(tmp_path: Path) -> None:
    source_path = _write_map(tmp_path, _road_joint_map())
    game_map = load_game_map(source_path)
    node = next(node for node in game_map.topology.nodes if node.node_id == "bend")
    joint_lanes = [lane for lane in game_map.lanes if lane.element_id == "bend"]
    lanes = {lane.lane_id: lane for lane in game_map.lanes}

    assert node.node_type == "road_joint"
    assert node.geometry == {"lane_transition_length_m": 0}
    assert node.attributes.speed_limit_mps == pytest.approx(8)
    assert len(joint_lanes) == 2
    assert all(lane.conditioning_visible for lane in joint_lanes)
    assert all(not lane.allows_taxi_stops for lane in joint_lanes)
    assert lanes["approach:lane:1"].successor_ids == ("bend:lane:1",)
    assert lanes["bend:lane:1"].successor_ids == ("exit:lane:1",)
    assert lanes["bend:lane:1"].speed_limit_mps == pytest.approx(13.4)
    assert lanes["bend:lane:0"].speed_limit_mps == pytest.approx(8)
    assert lanes["approach:lane:1"].centerline_world[-1, 0] < -1

    joint = _surface(game_map, "bend")
    approach = _surface(game_map, "approach")
    exit_road = _surface(game_map, "exit")
    assert joint.is_valid
    assert joint.intersection(approach).area < 1.0e-4
    assert joint.intersection(exit_road).area < 1.0e-4
    assert joint.boundary.intersection(approach.boundary).length > 7
    assert joint.boundary.intersection(exit_road.boundary).length > 7
    element = next(item for item in game_map.elements if item.element_id == "bend")
    assert element.road_boundaries
    assert element.curbs
    assert max(len(curb.polyline_world) for curb in element.curbs) > 3
    curved_rail = max(element.curbs, key=lambda curb: len(curb.polyline_world))
    rail_xy = curved_rail.polyline_world[:, :2]
    chord = LineString((rail_xy[0], rail_xy[-1]))
    assert max(chord.distance(Point(point)) for point in rail_xy[1:-1]) > 0.1
    assert any(
        divider.divider_id.startswith("bend:") for divider in game_map.lane_dividers
    )

    restored = game_map_from_dict(game_map_to_dict(game_map))
    restored_node = next(
        node for node in restored.topology.nodes if node.node_id == "bend"
    )
    assert restored_node.attributes == node.attributes
    assert restored_node.geometry == node.geometry

    preview_path = write_game_map_preview(
        source_path, tmp_path / "road-joint-preview.svg"
    )
    assert "bend [node:road_joint]" in preview_path.read_text(encoding="utf-8")
    compiled = compile_game_map(source_path, cache_root=tmp_path / "cache")
    with zipfile.ZipFile(compiled.archive_path) as archive:
        lane_rows = pq.read_table(io.BytesIO(archive.read("clipgt/lane.parquet")))
        divider_rows = pq.read_table(
            io.BytesIO(archive.read("clipgt/lane_line.parquet"))
        )
    assert lane_rows.num_rows == sum(
        lane.conditioning_visible for lane in game_map.lanes
    )
    assert divider_rows.num_rows == len(game_map.lane_dividers)


def test_road_joint_rejects_removed_curve_length(tmp_path: Path) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    bend = next(node for node in nodes if node["id"] == "bend")
    bend["curve_length_m"] = 8

    with pytest.raises(GameMapError, match="unknown attributes.*curve_length_m"):
        load_game_map(_write_map(tmp_path, source))


def test_road_joint_rounds_ninety_degree_outside_boundary(tmp_path: Path) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    east = next(node for node in nodes if node["id"] == "east_end")
    east["pose"] = {"x_m": 0, "y_m": 35}

    game_map = load_game_map(_write_map(tmp_path, source))
    joint = next(
        element for element in game_map.elements if element.element_id == "bend"
    )
    outside = max(joint.curbs, key=lambda curb: len(curb.polyline_world))
    points = outside.polyline_world[:, :2]
    chord = LineString((points[0], points[-1]))

    assert len(points) > 3
    assert max(chord.distance(Point(point)) for point in points[1:-1]) > 0.5


def test_straight_road_joint_remains_straight(tmp_path: Path) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    east = next(node for node in nodes if node["id"] == "east_end")
    east["pose"] = {"x_m": 35, "y_m": 0}

    game_map = load_game_map(_write_map(tmp_path, source))
    joint_lane = next(lane for lane in game_map.lanes if lane.lane_id == "bend:lane:1")

    assert np.ptp(joint_lane.centerline_world[:, 1]) < 1.0e-4


def test_road_joint_without_curbs_keeps_semantic_boundaries(tmp_path: Path) -> None:
    source = _road_joint_map()
    profiles = cast(dict[str, dict[str, Any]], source["profiles"])
    profiles["street"]["curb"] = False

    game_map = load_game_map(_write_map(tmp_path, source))
    joint = next(
        element for element in game_map.elements if element.element_id == "bend"
    )

    assert joint.road_boundaries
    assert not joint.curbs


def test_road_joint_trims_curved_approach_tangentially(tmp_path: Path) -> None:
    game_map = load_game_map(
        _write_map(tmp_path, _road_joint_map(curved_approach=True))
    )
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    approach = lanes["approach:lane:1"].centerline_world
    joint = lanes["bend:lane:1"].centerline_world
    approach_tangent = next(
        approach[-1, :2] - point[:2]
        for point in approach[-2::-1]
        if np.linalg.norm(approach[-1, :2] - point[:2]) > 0.05
    )
    joint_tangent = joint[1, :2] - joint[0, :2]
    approach_tangent /= np.linalg.norm(approach_tangent)
    joint_tangent /= np.linalg.norm(joint_tangent)

    np.testing.assert_allclose(joint_tangent, approach_tangent, atol=0.08)


def test_road_joint_infers_independent_trims_for_unequal_widths(
    tmp_path: Path,
) -> None:
    source = _road_joint_map()
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[0]["curb_offset_m"] = 0.2
    roads[1]["curb_offset_m"] = 1.2

    game_map = load_game_map(_write_map(tmp_path, source))
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    approach_cut = np.linalg.norm(lanes["approach:lane:1"].centerline_world[-1, :2])
    exit_cut = np.linalg.norm(lanes["exit:lane:1"].centerline_world[0, :2])

    assert approach_cut != pytest.approx(exit_cut, abs=0.1)


def test_road_joint_normalizes_reversed_authored_road_direction(tmp_path: Path) -> None:
    source = _road_joint_map()
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[1]["from"] = "east_end"
    roads[1]["to"] = "bend"

    game_map = load_game_map(_write_map(tmp_path, source))
    lanes = {lane.lane_id: lane for lane in game_map.lanes}

    assert lanes["approach:lane:1"].successor_ids == ("bend:lane:1",)
    assert lanes["bend:lane:1"].successor_ids == ("exit:lane:0",)


def test_road_joint_builds_visible_lane_count_and_width_transition(
    tmp_path: Path,
) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    bend = next(node for node in nodes if node["id"] == "bend")
    bend["lane_transition_length_m"] = 16
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[0]["lanes"] = ["backward", "backward", "forward", "forward"]
    roads[0]["divider_markings"] = [
        {"style": "DASHED_SINGLE", "color": "WHITE"},
        {"style": "SOLID_GROUP", "color": "YELLOW"},
        {"style": "DASHED_SINGLE", "color": "WHITE"},
    ]
    roads[1]["lane_width_m"] = 3.2

    game_map = load_game_map(_write_map(tmp_path, source))
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    transitions = [
        lane
        for lane in game_map.lanes
        if lane.lane_id.startswith("bend:transition:exit:")
    ]

    assert len(transitions) == 4
    assert all(lane.conditioning_visible for lane in transitions)
    assert lanes["exit:lane:0"].successor_ids == (
        "bend:transition:exit:lane:0",
        "bend:transition:exit:lane:1",
    )
    assert lanes["bend:transition:exit:lane:2"].successor_ids == ("exit:lane:1",)
    assert lanes["bend:transition:exit:lane:3"].successor_ids == ("exit:lane:1",)
    joint = _surface(game_map, "bend")
    assert joint.is_valid
    assert joint.bounds[2] > 15


def test_road_joint_lane_change_requires_transition_length(tmp_path: Path) -> None:
    source = _road_joint_map()
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[1]["lane_width_m"] = 4.2

    with pytest.raises(GameMapError, match="positive lane_transition_length_m"):
        load_game_map(_write_map(tmp_path, source))


def test_road_joint_builds_lane_width_only_transition(tmp_path: Path) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    bend = next(node for node in nodes if node["id"] == "bend")
    bend["lane_transition_length_m"] = 12
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[1]["lane_width_m"] = 4.2

    game_map = load_game_map(_write_map(tmp_path, source))
    transitions = [
        lane
        for lane in game_map.lanes
        if lane.lane_id.startswith("bend:transition:approach:")
    ]

    assert len(transitions) == 2
    for lane in transitions:
        widths = np.linalg.norm(
            lane.left_edge_world[:, :2] - lane.right_edge_world[:, :2],
            axis=1,
        )
        assert {round(float(widths[0]), 1), round(float(widths[-1]), 1)} == {
            3.6,
            4.2,
        }


def test_road_joint_transition_preserves_each_curb_offset(
    tmp_path: Path,
) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    bend = next(node for node in nodes if node["id"] == "bend")
    bend["lane_transition_length_m"] = 12
    roads = cast(list[dict[str, Any]], source["roads"])
    roads[0]["curb_offset_m"] = 0.4
    roads[1]["lane_width_m"] = 4.2
    roads[1]["curb_offset_m"] = 1.0

    game_map = load_game_map(_write_map(tmp_path, source))
    transition = next(
        lane
        for lane in game_map.lanes
        if lane.lane_id == "bend:transition:approach:lane:1"
    )
    curb_offsets = np.linalg.norm(
        transition.right_edge_world[:, :2] - transition.roadside_edge_world[:, :2],
        axis=1,
    )

    np.testing.assert_allclose(curb_offsets, 0.4, atol=0.02)


def test_intersection_builds_transition_only_on_mismatched_through_arm(
    tmp_path: Path,
) -> None:
    game_map = load_game_map(_write_map(tmp_path, _intersection_transition_map()))
    lanes = {lane.lane_id: lane for lane in game_map.lanes}
    transition_ids = {
        lane.lane_id for lane in game_map.lanes if ":transition:" in lane.lane_id
    }

    assert transition_ids == {
        f"center:transition:north_road:lane:{index}" for index in range(4)
    }
    assert lanes["north_road:lane:0"].successor_ids == (
        "center:transition:north_road:lane:0",
        "center:transition:north_road:lane:1",
    )
    assert lanes["center:transition:north_road:lane:2"].successor_ids == (
        "north_road:lane:1",
    )
    assert lanes["center:transition:north_road:lane:3"].successor_ids == (
        "north_road:lane:1",
    )
    assert not any("east_road" in lane_id for lane_id in transition_ids)
    assert not any("west_road" in lane_id for lane_id in transition_ids)
    center = _surface(game_map, "center")
    assert center.bounds[3] == pytest.approx(23.71, abs=0.1)
    assert center.bounds[0] == pytest.approx(-7.8, abs=0.1)
    assert center.bounds[2] == pytest.approx(7.8, abs=0.1)
    north_transition = lanes["center:transition:north_road:lane:3"]
    curb_offsets = np.linalg.norm(
        north_transition.right_edge_world[:, :2]
        - north_transition.roadside_edge_world[:, :2],
        axis=1,
    )
    np.testing.assert_allclose(curb_offsets, 0.5, atol=0.02)
    center_element = next(
        element for element in game_map.elements if element.element_id == "center"
    )
    assert max(
        float(np.max(boundary.polyline_world[:, 1]))
        for boundary in center_element.road_boundaries
    ) == pytest.approx(23.71, abs=0.1)
    assert (
        max(float(np.max(curb.polyline_world[:, 1])) for curb in center_element.curbs)
        < 15
    )


def test_intersection_lane_change_requires_transition_length(tmp_path: Path) -> None:
    with pytest.raises(GameMapError, match="positive lane_transition_length_m"):
        load_game_map(_write_map(tmp_path, _intersection_transition_map(0)))


def test_intersection_rejects_transition_that_consumes_road_arm(
    tmp_path: Path,
) -> None:
    with pytest.raises(GameMapError, match="consumes its road arm"):
        load_game_map(_write_map(tmp_path, _intersection_transition_map(95)))


def test_road_joint_requires_exactly_two_roads(tmp_path: Path) -> None:
    source = _road_joint_map()
    roads = cast(list[dict[str, Any]], source["roads"])
    roads.pop()

    with pytest.raises(GameMapError, match="must connect exactly two distinct roads"):
        load_game_map(_write_map(tmp_path, source))


def test_road_rejects_combined_trims_from_both_endpoint_joints(
    tmp_path: Path,
) -> None:
    source = _road_joint_map()
    nodes = cast(list[dict[str, Any]], source["nodes"])
    east = next(node for node in nodes if node["id"] == "east_end")
    east["type"] = "road_joint"
    east["pose"] = {"x_m": 2, "y_m": 2}
    east.pop("culdesac_radius_m")
    nodes.append(
        {
            "id": "far_end",
            "type": "cul_de_sac",
            "pose": {"x_m": 2, "y_m": 35},
            "culdesac_radius_m": 8,
        }
    )
    roads = cast(list[dict[str, Any]], source["roads"])
    roads.append(
        {
            "id": "tail",
            "from": "east_end",
            "to": "far_end",
            "profile": "street",
        }
    )

    with pytest.raises(GameMapError, match="too short for road-joint trims"):
        load_game_map(_write_map(tmp_path, source))


def test_path_road_is_one_topological_road(tmp_path: Path) -> None:
    game_map = load_game_map(_write_map(tmp_path, _self_loop_map(), "self-loop"))
    road = next(
        item for item in game_map.topology.roads if item.road_id == "neighborhood_loop"
    )
    road_lanes = [lane for lane in game_map.lanes if lane.element_id == road.road_id]

    assert road.from_node_id == road.to_node_id == "hub"
    assert len(road.bezier_spans_world) == 5
    np.testing.assert_allclose(
        road.bezier_spans_world[0][3],
        np.asarray([45, 15, 0], dtype=np.float32),
    )
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


def test_malformed_path_points_are_rejected(tmp_path: Path) -> None:
    source = _self_loop_map()
    loop = next(road for road in source["roads"] if road["id"] == "neighborhood_loop")
    loop["path"][0] = {"x_m": 0, "y_m": 0}

    with pytest.raises(GameMapError, match="degenerate segment"):
        load_game_map(_write_map(tmp_path, source))


def test_explicit_bezier_is_supported(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["bezier"] = [
        {
            "control_points": [{"x_m": 0, "y_m": 10}, {"x_m": 0, "y_m": 20}],
            "end": {"x_m": 0, "y_m": 30},
        }
    ]

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved = next(
        item for item in game_map.topology.roads if item.road_id == "dead_end_road"
    )

    assert len(resolved.bezier_spans_world) == 1
    np.testing.assert_allclose(
        resolved.bezier_spans_world[0],
        np.asarray([[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0]], dtype=np.float32),
    )


@pytest.mark.parametrize("field", ["path", "bezier"])
def test_road_geometry_fields_must_not_be_empty(tmp_path: Path, field: str) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road[field] = []

    with pytest.raises(GameMapError, match=rf"\.{field} must not be empty"):
        load_game_map(_write_map(tmp_path, source))


def test_path_rejects_bezier_spans(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["path"] = [
        {
            "control_points": [{"x_m": 0, "y_m": 15}, {"x_m": 0, "y_m": 20}],
            "end": {"x_m": 0, "y_m": 30},
        }
    ]

    with pytest.raises(GameMapError, match="put explicit spans under bezier"):
        load_game_map(_write_map(tmp_path, source))


def test_bezier_rejects_path_points(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["bezier"] = [{"x_m": 0, "y_m": 15}]

    with pytest.raises(GameMapError, match="Bezier spans require"):
        load_game_map(_write_map(tmp_path, source))


def test_bezier_takes_precedence_over_path(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["path"] = [{"x_m": 10, "y_m": 15}]
    road["bezier"] = [
        {
            "control_points": [{"x_m": 0, "y_m": 10}, {"x_m": 0, "y_m": 20}],
            "end": {"x_m": 0, "y_m": 30},
        }
    ]

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved = next(
        item for item in game_map.topology.roads if item.road_id == "dead_end_road"
    )

    np.testing.assert_allclose(
        resolved.bezier_spans_world[0],
        np.asarray([[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0]], dtype=np.float32),
    )


def test_path_is_validated_when_bezier_is_present(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["path"] = [{"x_m": 10}]
    road["bezier"] = [
        {
            "control_points": [{"x_m": 0, "y_m": 10}, {"x_m": 0, "y_m": 20}],
            "end": {"x_m": 0, "y_m": 30},
        }
    ]

    with pytest.raises(GameMapError, match="requires exactly x_m and y_m"):
        load_game_map(_write_map(tmp_path, source))


def test_bezier_final_endpoint_must_match_to_node(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["bezier"] = [
        {
            "control_points": [{"x_m": 0, "y_m": 10}, {"x_m": 0, "y_m": 20}],
            "end": {"x_m": 1, "y_m": 30},
        }
    ]

    with pytest.raises(GameMapError, match="final Bezier endpoint"):
        load_game_map(_write_map(tmp_path, source))


def test_self_loop_supports_explicit_bezier(tmp_path: Path) -> None:
    source = _self_loop_map()
    loop = next(road for road in source["roads"] if road["id"] == "neighborhood_loop")
    del loop["path"]
    loop["bezier"] = [
        {
            "control_points": [{"x_m": 30, "y_m": 0}, {"x_m": 45, "y_m": 0}],
            "end": {"x_m": 45, "y_m": 15},
        },
        {
            "control_points": [{"x_m": 45, "y_m": 35}, {"x_m": 45, "y_m": 50}],
            "end": {"x_m": 30, "y_m": 50},
        },
        {
            "control_points": [{"x_m": -30, "y_m": 50}, {"x_m": -45, "y_m": 35}],
            "end": {"x_m": -45, "y_m": 15},
        },
        {
            "control_points": [{"x_m": -45, "y_m": 0}, {"x_m": -30, "y_m": 0}],
            "end": {"x_m": 0, "y_m": 0},
        },
    ]

    game_map = load_game_map(_write_map(tmp_path, source))
    resolved = next(
        item for item in game_map.topology.roads if item.road_id == "neighborhood_loop"
    )

    assert len(resolved.bezier_spans_world) == 4


def test_self_loop_requires_path_or_bezier(tmp_path: Path) -> None:
    source = _self_loop_map()
    loop = next(road for road in source["roads"] if road["id"] == "neighborhood_loop")
    del loop["path"]

    with pytest.raises(GameMapError, match="requires path or bezier"):
        load_game_map(_write_map(tmp_path, source))


def test_roads_cannot_cross_without_a_connection_node(tmp_path: Path) -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    road = next(item for item in source["roads"] if item["id"] == "dead_end_road")
    road["path"] = [
        {"x_m": 45, "y_m": 0},
        {"x_m": 45, "y_m": 30},
    ]

    with pytest.raises(
        GameMapError,
        match=(
            "Unrelated elements.*overlap|completely contained by its endpoint "
            "footprints|invalid boundary ribbon"
        ),
    ):
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


def test_intersection_surface_is_inferred_from_incident_road_edges() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    crossing = _surface(game_map, "central_arterial_crossing")
    merge = _surface(game_map, "arterial_merge_crossing")

    assert np.ptp(np.asarray(crossing.exterior.coords)[:, 0]) == pytest.approx(
        8.4, abs=0.05
    )
    assert np.ptp(np.asarray(crossing.exterior.coords)[:, 1]) == pytest.approx(
        15.6, abs=0.05
    )
    assert crossing.area == pytest.approx(8.4 * 15.6, abs=0.5)
    assert np.ptp(np.asarray(merge.exterior.coords)[:, 0]) < 30.0
    assert np.ptp(np.asarray(merge.exterior.coords)[:, 1]) < 28.0
    assert merge.convex_hull.area - merge.area < 5.0


def test_boulevard_intersection_curbs_have_no_cap_fragments() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    elements = {element.element_id: element for element in game_map.elements}

    assert not elements["central_arterial_crossing"].curbs
    merge_lengths = [
        LineString(curb.polyline_world[:, :2]).length
        for curb in elements["arterial_merge_crossing"].curbs
    ]
    assert merge_lengths
    assert min(merge_lengths) > 1.0


def test_south_parking_lot_uses_its_complete_authored_opening() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    elements = {element.element_id: element for element in game_map.elements}
    opening = LineString(((145.0, -150.0), (157.0, -150.0)))
    lot = elements["south_parking_lot"]
    access = _surface(game_map, "south_parking_lot:access")

    assert access.boundary.intersection(opening).length == pytest.approx(12.0)
    assert (
        sum(
            LineString(curb.polyline_world[:, :2]).intersection(opening).length
            for curb in lot.curbs
        )
        <= 1.0e-4
    )


def test_cul_de_sac_has_full_width_flat_road_connection() -> None:
    game_map = load_game_map(_STARTER_MAP)
    cul_de_sac = _surface(game_map, "dead_end")
    road = _surface(game_map, "dead_end_road")
    boundary = np.asarray(cul_de_sac.exterior.coords)
    segment_lengths = np.linalg.norm(np.diff(boundary, axis=0), axis=1)

    assert cul_de_sac.distance(road) < 1.0e-3
    assert float(segment_lengths.max()) == pytest.approx(8.4, abs=0.05)


def test_parking_access_is_inferred_from_lot_node_and_not_authored_as_a_road() -> None:
    source = yaml.safe_load(_STARTER_MAP.read_text(encoding="utf-8"))
    lot = next(node for node in source["nodes"] if node["type"] == "parking_lot")
    game_map = load_game_map(_STARTER_MAP)
    road_ids = {road.road_id for road in game_map.topology.roads}
    access_ids = {access.access_id for access in game_map.topology.parking_accesses}
    inferred = {
        element.element_id
        for element in game_map.elements
        if element.element_type == "parking_access"
    }

    assert lot["connected_to"] == "hub"
    assert lot["opening_vertex"] == 2
    assert "neighborhood_lot:access" not in road_ids
    assert access_ids == {"neighborhood_lot:access"}
    assert inferred == access_ids
    width = np.ptp(
        next(
            element.surface_world[:, 0]
            for element in game_map.elements
            if element.element_id == "neighborhood_lot:access"
        )
    )
    assert width == pytest.approx(6.0, abs=0.2)
    lanes = [
        lane for lane in game_map.lanes if lane.element_id == "neighborhood_lot:access"
    ]
    assert len(lanes) == 2
    assert all(lane.speed_limit_mps == pytest.approx(5.5) for lane in lanes)
    assert all(lane.marking_style == "VIRTUAL" for lane in lanes)
    assert all(not lane.allows_taxi_stops for lane in lanes)


def test_degree_two_parking_sources_compile_as_driveway_nodes() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    nodes = {node.node_id: node for node in game_map.topology.nodes}

    assert nodes["southwest_lot_driveway"].node_type == "driveway"
    assert nodes["south_lot_driveway"].node_type == "driveway"
    assert nodes["east_corner_lot_driveway"].node_type == "driveway"
    assert nodes["south_upper_lot_driveway"].node_type == "driveway"
    assert nodes["arterial_crossing_895"].node_type == "intersection"
    assert not any(
        element.element_type == "intersection"
        for element in game_map.elements
        if element.element_id == "southwest_lot_driveway"
    )


def test_lane_graph_routes_to_and_from_parking_lot() -> None:
    game_map = load_game_map(_STARTER_MAP)
    outbound = _reachable(game_map, game_map.default_spawn.lane_id)
    returning = _reachable(game_map, "neighborhood_lot:access:lane:1")

    assert "neighborhood_lot:access:lane:0" in outbound
    assert "neighborhood_loop_southwest:lane:0" in returning
    assert not any(lane.element_id == "neighborhood_lot" for lane in game_map.lanes)


def test_parking_lots_compile_as_green_roadnet_masks() -> None:
    game_map = load_game_map(_BOULEVARD_MAP)
    rows = game_map_compiler._road_marking_rows(game_map)
    lots = [node for node in game_map.topology.nodes if node.node_type == "parking_lot"]

    assert len(rows) == len(lots) == 8
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
    for access in game_map.topology.parking_accesses:
        for node_id in (access.source_node_id, access.parking_lot_node_id):
            pairs.append((access.access_id, node_id))
    for first_id, second_id in pairs:
        first, second = surfaces[first_id], surfaces[second_id]
        assert first.intersection(second).area <= 1.0e-4, (first_id, second_id)
        assert first.distance(second) <= 1.0e-4, (first_id, second_id)


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_connected_surface_seams_do_not_emit_boundaries(source_path: Path) -> None:
    game_map = load_game_map(source_path)
    elements = {element.element_id: element for element in game_map.elements}
    surfaces = {
        element.element_id: Polygon(element.surface_world[:, :2]).buffer(0)
        for element in game_map.elements
    }
    pairs: list[tuple[str, str]] = []
    for road in game_map.topology.roads:
        for node_id in (road.from_node_id, road.to_node_id):
            pairs.append((road.road_id, node_id))
    for access in game_map.topology.parking_accesses:
        for node_id in (access.source_node_id, access.parking_lot_node_id):
            pairs.append((access.access_id, node_id))

    for first_id, second_id in pairs:
        seam = surfaces[first_id].boundary.intersection(surfaces[second_id].boundary)
        seam_boundaries = sum(
            LineString(boundary.polyline_world[:, :2]).intersection(seam).length
            for element_id in (first_id, second_id)
            for boundary in elements[element_id].road_boundaries
        )
        assert seam_boundaries <= 1.0e-4, (first_id, second_id, seam_boundaries)


@pytest.mark.parametrize("source_path", [_STARTER_MAP, _BOULEVARD_MAP])
def test_compiled_boundaries_belong_to_their_elements(source_path: Path) -> None:
    game_map = load_game_map(source_path)
    for element in game_map.elements:
        surface_boundary = Polygon(element.surface_world[:, :2]).boundary
        for boundary in element.road_boundaries:
            line = LineString(boundary.polyline_world[:, :2])
            assert line.difference(surface_boundary.buffer(1.0e-4)).length < 1.0e-4
        for curb in element.curbs:
            line = LineString(curb.polyline_world[:, :2])
            assert line.difference(surface_boundary.buffer(1.0e-4)).length < 1.0e-4
        if element.attributes.curb:
            assert all(curb.polyline_world.shape[0] >= 2 for curb in element.curbs)
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
        len(element.road_boundaries) for element in compiled.game_map.elements
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
        + len(first.game_map.topology.parking_accesses)
    )
    for node in first.game_map.topology.nodes:
        assert f"{node.node_id} [node:{node.node_type}]" in preview_text
    for road in first.game_map.topology.roads:
        assert f"{road.road_id} [road:{road.profile_id};" in preview_text
    for access in first.game_map.topology.parking_accesses:
        assert f"{access.access_id} [parking access;" in preview_text
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
    assert any(region.kind == "area" for region in taxi_scene.fare_regions)
    assert any(region.kind == "boundary" for region in taxi_scene.fare_regions)
    assert len(taxi_scene.curb_segments_world) == sum(
        len(curb.polyline_world) - 1
        for element in first.game_map.elements
        for curb in element.curbs
    )
