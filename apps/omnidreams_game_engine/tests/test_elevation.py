# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for bounded sampling of elevated pavement."""

import numpy as np
import pytest
import shapely
from omnidreams_game_engine.game_map.elevation import _pavement_samples, drape_polyline
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from shapely.geometry import LineString

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("profile", ["flat", "grade", "hill"])
def test_draping_retains_only_required_height_samples(profile: str) -> None:
    x = np.arange(101, dtype=np.float32)
    z = 0.2 * x if profile == "grade" else np.zeros_like(x)
    if profile == "hill":
        z = 5 * np.sin(np.pi * x / 100) ** 2
    vertices = np.concatenate(
        [np.column_stack((x, np.full_like(x, y), z)) for y in (-5, 5)]
    )
    faces = np.asarray(
        [(i, i + 1, i + 101) for i in range(100)]
        + [(i + 1, i + 102, i + 101) for i in range(100)],
        dtype=np.int32,
    )
    ground = GroundSnapper(vertices, faces)
    # Keep the original bend even though its height is linear on flat/graded ground.
    source = np.asarray([[0, 0, 0], [50, 2, 0], [100, 0, 0]], dtype=np.float32)
    actual = drape_polyline(source, ground, 0.5)
    for corner in source:
        assert np.any(np.all(actual[:, :2] == corner[:2], axis=1))
    if profile != "hill":
        np.testing.assert_array_equal(actual[:, :2], source[:, :2])
    else:
        assert len(source) < len(actual) < 202
    probes = np.column_stack((x, np.interp(x, source[:, 0], source[:, 1]), x * 0))
    np.testing.assert_allclose(
        np.interp(x, actual[:, 0], actual[:, 2]), ground._raycast(probes), atol=0.003
    )


def test_diagonal_pavement_tiles_preserve_the_shared_lattice(monkeypatch) -> None:
    region = LineString([(-30, -30), (30, 30)]).buffer(0.3)
    spacing = 0.1
    x0, y0, x1, y1 = region.bounds
    x, y = np.meshgrid(
        np.arange(np.floor(x0 / spacing), np.ceil(x1 / spacing) + 1) * spacing,
        np.arange(np.floor(y0 / spacing), np.ceil(y1 / spacing) + 1) * spacing,
    )
    expected = np.column_stack((x.ravel(), y.ravel()))
    expected = expected[shapely.intersects_xy(region, expected[:, 0], expected[:, 1])]
    meshgrid = np.meshgrid
    allocations = []

    def bounded_meshgrid(xs, ys):
        allocations.append(len(xs) * len(ys))
        assert allocations[-1] <= 256 * 256
        return meshgrid(xs, ys)

    monkeypatch.setattr(np, "meshgrid", bounded_meshgrid)
    actual = np.concatenate(list(_pavement_samples(region, spacing)))
    assert len(allocations) > 1
    assert len(actual) == len(expected)
    np.testing.assert_array_equal(
        actual[np.lexsort((actual[:, 1], actual[:, 0]))],
        expected[np.lexsort((expected[:, 1], expected[:, 0]))],
    )
