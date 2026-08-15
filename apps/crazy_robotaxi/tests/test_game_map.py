# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for semantic Crazy Robotaxi maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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
from omnidreams_game_engine.scene_loader import load_scene_bundle
from omnidreams_game_engine.simulation.map_bounds import MapBounds
from omnidreams_game_engine.types import VehicleState

pytestmark = pytest.mark.ci_cpu

_STARTER_MAP = (
    Path(__file__).parents[1] / "crazy_robotaxi" / "maps" / "minimal_loop.robotaxi.yaml"
)


def test_starter_map_resolves_loop_intersection_and_dead_end() -> None:
    game_map = load_game_map(_STARTER_MAP)

    assert game_map.schema_version == 1
    assert len(game_map.elements) == 11
    assert {element.element_type for element in game_map.elements} == {
        "road_segment",
        "intersection",
    }
    east_road = next(
        element for element in game_map.elements if element.element_id == "east_road"
    )
    assert float(np.ptp(east_road.surface_world[:, 1])) == pytest.approx(8.4)
    for lane in (lane for lane in game_map.lanes if lane.element_id == "east_road"):
        rail_widths = np.linalg.norm(
            lane.left_edge_world[:, :2] - lane.right_edge_world[:, :2], axis=1
        )
        np.testing.assert_allclose(rail_widths, 3.6)
        roadside_offsets = np.linalg.norm(
            lane.roadside_edge_world[:, :2] - lane.right_edge_world[:, :2], axis=1
        )
        np.testing.assert_allclose(roadside_offsets, 0.6, atol=1.0e-6)
    dead_end = next(
        element for element in game_map.elements if element.element_id == "dead_end"
    )
    ports = {port[0]: port for port in dead_end.ports}
    assert ports["start"][4] is True
    assert ports["end"][4] is False
    end_xy = np.asarray(ports["end"][1:3], dtype=np.float32)
    assert any(
        np.allclose(segment[:, :2].mean(axis=0), end_xy, atol=1.0e-3)
        for segment in game_map.collision_segments_world
    )


def test_semantic_lane_successors_drive_navigation_without_endpoint_inference() -> None:
    game_map = load_game_map(_STARTER_MAP)
    lanes = tuple(
        NavigationLane(
            centerline_world=lane.centerline_world,
            road_edge_world=(
                lane.roadside_edge_world if lane.allows_taxi_stops else None
            ),
            allows_taxi_stops=lane.allows_taxi_stops,
            lane_id=lane.lane_id,
            successor_ids=lane.successor_ids,
        )
        for lane in game_map.lanes
    )

    navigation = TaxiNavigationMap(lanes, endpoint_snap_tolerance_m=1.0e-6)

    spawn_lane = next(
        lane for lane in navigation.lanes if lane.lane_id == "east_road:lane:1"
    )
    assert spawn_lane.successor_ids == ("northeast_curve:lane:1",)
    assert len(navigation.sample_waypoints(spacing_m=20.0, offset_m=0.0)) > 2


def test_compiler_emits_shared_divider_and_separate_road_boundaries() -> None:
    game_map = load_game_map(_STARTER_MAP)
    lane_rows = {
        row["key"]["label_class_id"]: row["lane"]
        for row in game_map_compiler._lane_rows(game_map)
    }
    line_rows = game_map_compiler._lane_line_rows(game_map)
    east_lines = [
        row for row in line_rows if "east_road:lane:0" in row["key"]["label_class_id"]
    ]

    assert len(line_rows) == 10
    assert len(east_lines) == 1
    divider = east_lines[0]["lane_line"]
    assert all(point["y"] == pytest.approx(0.0) for point in divider["line_rail"])
    assert divider["styles"] == ["SOLID_GROUP"]
    assert divider["colors"] == ["YELLOW"]
    assert lane_rows["east_road:lane:0"]["left_edge_styles"] == ["SOLID_GROUP"]
    assert lane_rows["east_road:lane:0"]["left_edge_colors"] == ["YELLOW"]
    assert lane_rows["east_road:lane:0"]["right_edge_styles"] == ["VIRTUAL"]
    assert lane_rows["east_road:lane:1"]["left_edge_styles"] == ["SOLID_GROUP"]
    assert lane_rows["east_road:lane:1"]["left_edge_colors"] == ["YELLOW"]
    assert lane_rows["east_road:lane:1"]["right_edge_styles"] == ["VIRTUAL"]


def test_starter_map_can_initialize_gameplay() -> None:
    game_map = load_game_map(_STARTER_MAP)
    spawn = game_map.default_spawn
    lanes = tuple(
        NavigationLane(
            centerline_world=lane.centerline_world,
            road_edge_world=(
                lane.roadside_edge_world if lane.allows_taxi_stops else None
            ),
            allows_taxi_stops=lane.allows_taxi_stops,
            lane_id=lane.lane_id,
            successor_ids=lane.successor_ids,
        )
        for lane in game_map.lanes
    )
    spawn_lane = next(lane for lane in lanes if lane.lane_id == spawn.lane_id)
    state = VehicleState(
        x_m=float(spawn.position_world[0]),
        y_m=float(spawn.position_world[1]),
        z_m=float(spawn.position_world[2]),
        yaw_rad=spawn.yaw_rad,
        speed_mps=0.0,
        steer_rad=0.0,
    )

    controller = TaxiGameController(
        scene_id=game_map.map_id,
        reference_route_world=spawn_lane.centerline_world,
        navigation_lanes=lanes,
        initial_state=state,
        config=TaxiGameConfig(enabled=True, seed=17),
    )

    assert controller.snapshot(state).pickup_targets_xyz_m


def test_compiler_round_trip_embeds_semantic_map_and_reuses_cache(
    tmp_path: Path,
) -> None:
    first = compile_game_map(_STARTER_MAP, cache_root=tmp_path / "cache")
    second = compile_game_map(_STARTER_MAP, cache_root=tmp_path / "cache")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.archive_path == second.archive_path
    scene = load_scene_bundle(
        first.archive_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(),
    )
    assert scene.game_map is not None
    assert scene.game_map.map_id == "crazy-robotaxi-minimal-loop"
    assert "full-width two-lane asphalt public street" in scene.prompt
    assert "double solid yellow centerline" in scene.prompt
    assert scene.initial_speed_mps == pytest.approx(0.0)
    np.testing.assert_allclose(
        scene.initial_rig_to_world[:2, 3],
        scene.game_map.default_spawn.position_world[:2],
    )
    bounds = MapBounds.from_scene(scene)
    assert bounds is not None
    assert bounds.width_m < float(np.ptp(scene.ground_mesh_vertices[:, 0]))


def test_preview_and_scene_discovery_use_semantic_yaml(tmp_path: Path) -> None:
    preview = write_game_map_preview(_STARTER_MAP, tmp_path / "preview.svg")
    options = cli._discover_scene_options(_STARTER_MAP.parent, _STARTER_MAP)

    assert preview.read_text(encoding="utf-8").startswith("<svg")
    option = next(item for item in options if item.path == _STARTER_MAP.resolve())
    assert option.label == "Minimal Loop and Dead End"
    assert option.variants == ("default",)
    assert option.thumbnail is not None


def test_loop_closure_validation_reports_gap(tmp_path: Path) -> None:
    source = _STARTER_MAP.read_text(encoding="utf-8")
    broken = tmp_path / "broken.robotaxi.yaml"
    broken.write_text(
        source.replace("length_m: 76.8", "length_m: 72"), encoding="utf-8"
    )

    with pytest.raises(GameMapError, match="does not close: gap="):
        load_game_map(broken)
