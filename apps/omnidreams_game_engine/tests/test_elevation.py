# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for bounded sampling of elevated pavement."""

import numpy as np
import pytest
import shapely
from omnidreams_game_engine.game_map.elevation import _pavement_samples
from shapely.geometry import LineString

pytestmark = pytest.mark.ci_cpu


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
