# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for semantic Crazy Robotaxi maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi import cli
from crazy_robotaxi.navigation import NavigationLane, TaxiNavigationMap
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import (
    GameMapError,
    compile_game_map,
    load_game_map,
    write_game_map_preview,
)
from omnidreams_game_engine.scene_loader import load_scene_bundle
from omnidreams_game_engine.simulation.map_bounds import MapBounds

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
            road_edge_world=(lane.right_edge_world if lane.allows_taxi_stops else None),
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
        source.replace("length_m: 74.4", "length_m: 70"), encoding="utf-8"
    )

    with pytest.raises(GameMapError, match="does not close: gap="):
        load_game_map(broken)
