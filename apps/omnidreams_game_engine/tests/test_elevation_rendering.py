# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual CUDA rendering checks for elevated pavement and the flat seed view."""

import math
from pathlib import Path

import numpy as np
import omnidreams_game_engine.conditioning as conditioning
import pytest
import torch
import yaml
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.game_map import compile_game_map
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.scene import load_scene_bundle
from omnidreams_game_engine.types import SceneDefinition, TrajectoryChunk, VehicleState

pytestmark = pytest.mark.manual

_SOURCE = (
    Path(__file__).parents[2]
    / "crazy_robotaxi/crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml"
)
_RASTER = RasterConfig(width=320, height=160)


@pytest.fixture(scope="module")
def raceway(tmp_path_factory: pytest.TempPathFactory) -> SceneDefinition:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for semantic rendering")
    compiled = compile_game_map(_SOURCE, cache_root=tmp_path_factory.mktemp("raceway"))
    return load_scene_bundle(compiled.archive_path, "camera_front_wide_120fov", _RASTER)


def _trajectory(states: tuple[VehicleState, ...]) -> TrajectoryChunk:
    return TrajectoryChunk(
        np.arange(len(states), dtype=np.int64),
        np.stack([rig_pose_from_vehicle_state(state) for state in states]),
        states,
        states[-1],
    )


def test_sloped_bev_pavement_does_not_hide_lane_paint(
    raceway: SceneDefinition, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert raceway.game_map is not None
    states = []
    for lane_id in (
        "infield_hairpin_climb:lane:2",
        "back_straight:lane:2",
        "west_straight:lane:2",
    ):
        lane = next(lane for lane in raceway.game_map.lanes if lane.lane_id == lane_id)
        points = lane.centerline_world
        index = len(points) // 3
        tangent = points[index + 1] - points[index]
        states.append(
            VehicleState(
                float(points[index, 0]),
                float(points[index, 1]),
                float(points[index, 2]),
                math.atan2(float(tangent[1]), float(tangent[0])),
                0,
                0,
                pitch_rad=-math.atan2(
                    float(tangent[2]), float(np.linalg.norm(tangent[:2]))
                ),
            )
        )
    trajectory = _trajectory(tuple(states))
    counts = []
    for paved in (True, False):
        if not paved:
            monkeypatch.setattr(
                conditioning, "_build_bev_road_surface_pool", lambda *_, **__: None
            )
        renderer = conditioning.LudusConditionRenderer(
            _RASTER, BevConfig(width=256, height=256)
        )
        try:
            renderer.load_scene(raceway)
            bev = renderer.render(trajectory).bev_tchw
            assert bev is not None
            frames = bev[:, :3].cpu().numpy()
            counts.append(np.count_nonzero(np.all(frames > 220, axis=1), axis=(1, 2)))
        finally:
            renderer.close()
    assert np.all(counts[1] > 50)
    assert np.all(counts[0] >= 0.9 * counts[1]), counts


def test_starting_conditioning_matches_the_flat_map(
    raceway: SceneDefinition, tmp_path: Path
) -> None:
    document = yaml.safe_load(_SOURCE.read_text())
    for node in document["nodes"]:
        node["pose"]["z_m"] = 0
    source = tmp_path / "flat.robotaxi.yaml"
    source.write_text(yaml.safe_dump(document, sort_keys=False))
    compiled = compile_game_map(source, cache_root=tmp_path / "cache")
    flat = load_scene_bundle(compiled.archive_path, "camera_front_wide_120fov", _RASTER)
    assert raceway.game_map is not None
    spawn = raceway.game_map.default_spawn
    trajectory = _trajectory(
        (
            VehicleState(
                float(spawn.position_world[0]),
                float(spawn.position_world[1]),
                float(spawn.position_world[2]),
                spawn.yaw_rad,
                0,
                0,
            ),
        )
    )
    frames = []
    for scene in (raceway, flat):
        renderer = conditioning.LudusConditionRenderer(
            _RASTER, BevConfig(enabled=False)
        )
        try:
            renderer.load_scene(scene)
            frames.append(renderer.render(trajectory).hdmap_bvtchw.cpu())
        finally:
            renderer.close()
    # Float tessellation can change a few edge pixels even on the same flat road.
    changed = torch.any(frames[0] != frames[1], dim=3)
    assert int(torch.count_nonzero(changed)) <= 4
    np.testing.assert_array_equal(raceway.initial_rgb, flat.initial_rgb)
