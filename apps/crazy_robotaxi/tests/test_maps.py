# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU validation for shipped semantic maps."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from omnidreams_game_engine.game_map import (
    GameMapError,
    compile_game_map,
    load_game_map,
)
from omnidreams_game_engine.game_map.types import (
    game_map_from_dict,
    game_map_to_dict,
)

pytestmark = pytest.mark.ci_cpu

_MAP_FIXTURES = Path(__file__).parent / "maps"


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
    assert variant.prompt_context
    assert variant.prompt_context != variant.prompt


def test_prompt_context_is_trimmed_and_round_trips(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (_MAP_FIXTURES / "intersection_geometry.robotaxi.yaml").read_text()
    )
    source["nodes"][0]["prompt_context"] = "  A neighborhood landmark.  "
    source["roads"][0]["prompt_context"] = "  Detached homes line the road.  "
    source["spawns"][0]["variants"]["default"]["prompt_context"] = (
        "  A forward-facing road view.  "
    )
    path = tmp_path / "prompt-context.robotaxi.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))

    original = load_game_map(path)
    restored = game_map_from_dict(game_map_to_dict(original))

    assert original.topology.nodes[0].prompt_context == "A neighborhood landmark."
    assert original.topology.roads[0].prompt_context == (
        "Detached homes line the road."
    )
    assert original.default_spawn.variants[0].prompt_context == (
        "A forward-facing road view."
    )
    assert restored.default_spawn.variants == original.default_spawn.variants
    assert restored.topology == original.topology


@pytest.mark.parametrize("value", ["", "   ", 42, ["not", "text"]])
def test_prompt_context_requires_nonempty_text(tmp_path: Path, value: object) -> None:
    source = yaml.safe_load(
        (_MAP_FIXTURES / "intersection_geometry.robotaxi.yaml").read_text()
    )
    source["roads"][0]["prompt_context"] = value
    path = tmp_path / "invalid-prompt-context.robotaxi.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))

    with pytest.raises(GameMapError, match="prompt_context must be a nonempty string"):
        load_game_map(path)


@pytest.mark.parametrize("value", ["", "   ", 42, ["not", "text"]])
def test_spawn_prompt_context_requires_nonempty_text(
    tmp_path: Path, value: object
) -> None:
    source = yaml.safe_load(
        (_MAP_FIXTURES / "intersection_geometry.robotaxi.yaml").read_text()
    )
    source["spawns"][0]["variants"]["default"]["prompt_context"] = value
    path = tmp_path / "invalid-spawn-prompt-context.robotaxi.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))

    with pytest.raises(GameMapError, match="prompt_context must be a nonempty string"):
        load_game_map(path)


def test_prompt_context_change_invalidates_compiler_cache(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (_MAP_FIXTURES / "intersection_geometry.robotaxi.yaml").read_text()
    )
    path = tmp_path / "cached-prompt-context.robotaxi.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))
    cache_root = tmp_path / "cache"

    first = compile_game_map(path, cache_root=cache_root)
    assert compile_game_map(path, cache_root=cache_root).cache_hit
    source["roads"][0]["prompt_context"] = "A changed roadside setting."
    path.write_text(yaml.safe_dump(source, sort_keys=False))
    changed = compile_game_map(path, cache_root=cache_root)

    assert first.cache_hit is False
    assert changed.cache_hit is False
    assert changed.archive_path != first.archive_path


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
