# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU seed rendering checks for slopes, occlusion, and flat-map compatibility."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from omnidreams_game_engine.game_map.spawn_render import (
    _camera_ground_intersections,
    _camera_mesh_distances,
    render_spawn_first_frame,
)
from omnidreams_game_engine.game_map.types import GameMapSpawn, ResolvedGameMap

pytestmark = pytest.mark.ci_cpu


def test_camera_rays_hit_analytic_slope_across_forward_plane() -> None:
    vertices = np.asarray([[-20, -20, -6], [40, -20, 6], [40, 20, 6], [-20, 20, -6]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    rays = np.asarray([[1, 0, -0.1], [1, 0, 0], [1, 0, 0.1], [1, 0, 0.3]], dtype=float)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    distances = _camera_mesh_distances(rays, vertices, faces)
    # The plane is z = 0.2*x - 2. Its upper pixels rise above the old flat horizon.
    np.testing.assert_allclose(distances[:3], 2 / (0.2 * rays[:3, 0] - rays[:3, 2]))
    assert np.isinf(distances[3])
    assert np.all(np.isinf(_camera_mesh_distances(rays, vertices - [100, 0, 0], faces)))


def test_nearest_crest_occludes_the_far_road() -> None:
    vertices = np.asarray(
        [
            [x, y, z]
            for x, z in ((-10, -2), (10, -2), (20, 4), (40, -2), (80, -2))
            for y in (-10, 10)
        ]
    )
    faces = np.asarray(
        [[2 * i, 2 * i + 2, 2 * i + 3] for i in range(4)]
        + [[2 * i, 2 * i + 3, 2 * i + 1] for i in range(4)]
    )
    ray = np.asarray([[1.0, 0, -0.03]])
    ray /= np.linalg.norm(ray, axis=1)[:, None]
    distance = _camera_mesh_distances(ray, vertices, faces)[0]
    # This ray also reaches the far flat road at x=66.67, behind the first crest.
    assert distance * ray[0, 0] == pytest.approx(8 / 0.63)


def test_flat_seed_pixels_remain_unchanged_and_camera_uses_spawn_pitch() -> None:
    vertices = np.asarray(
        [[-50, -50, 0], [600, -50, 0], [600, 50, 0], [-50, 50, 0]], dtype=np.float32
    )
    spawn = GameMapSpawn(
        "start", "road:lane:0", 0, np.zeros(3, dtype=np.float32), 0, None, "Daylight"
    )
    game_map = cast(
        ResolvedGameMap,
        SimpleNamespace(
            ground_vertices=vertices,
            ground_faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
            elements=(
                SimpleNamespace(
                    element_type="road",
                    surface_world=vertices,
                    road_boundaries=(),
                    curbs=(),
                ),
            ),
            road_marking_polygons_world=(),
            lane_dividers=(),
            line_markings=(),
        ),
    )
    image = render_spawn_first_frame(game_map, spawn, resolution_wh=(160, 88))
    # Recorded from the original plane renderer before elevation support.
    assert (
        hashlib.sha256(image.tobytes()).hexdigest()
        == "6f41336e5e16601effe863f63d87384b14f32263781e5ad52c03599697d76b0b"
    )
    level = _camera_ground_intersections(spawn, 160, 88, game_map)
    pitched = _camera_ground_intersections(
        replace(spawn, pitch_rad=-0.2), 160, 88, game_map
    )
    assert np.count_nonzero(pitched[-1]) < np.count_nonzero(level[-1])
