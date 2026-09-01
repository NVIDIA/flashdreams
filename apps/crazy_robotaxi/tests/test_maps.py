# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU validation for shipped semantic maps."""

import math
from pathlib import Path

import numpy as np
import pytest
from omnidreams_game_engine.game_map import (
    GameMapError,
    load_game_map,
    load_game_map_header,
    render_spawn_first_frame,
    render_spawn_first_frame_with_road_mask,
)
from omnidreams_game_engine.game_map.types import game_map_from_dict, game_map_to_dict

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
    variant = game_map.default_spawn.variants[0]
    assert variant.time_of_day == "day"
    assert variant.image is not None
    assert (path.parent / variant.image).is_file()
    restored = game_map_from_dict(game_map_to_dict(game_map))
    assert restored.default_spawn.variants[0].time_of_day == "day"

    semantic, road_mask = render_spawn_first_frame_with_road_mask(
        game_map, game_map.default_spawn, resolution_wh=(160, 96)
    )
    assert semantic.shape == (96, 160, 3)
    assert semantic.dtype == np.uint8
    assert road_mask.shape == (96, 160)
    assert road_mask.dtype == np.bool_
    assert road_mask.any()
    assert not road_mask.all()
    assert np.array_equal(
        semantic,
        render_spawn_first_frame(
            game_map, game_map.default_spawn, resolution_wh=(160, 96)
        ),
    )


def test_map_rejects_unknown_spawn_time_of_day(tmp_path: Path) -> None:
    path = tmp_path / "invalid.robotaxi.yaml"
    path.write_text(
        """\
schema_version: 1
id: invalid
name: Invalid
compiler: {}
nodes: []
roads: []
spawns:
  - id: start
    road: road
    lane: 0
    distance_m: 0
    variants:
      default:
        time_of_day: noon
        prompt: A road at noon.
""",
        encoding="utf-8",
    )

    with pytest.raises(GameMapError, match="time_of_day must be one of"):
        load_game_map_header(path)


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
