# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for model versus UI conditioning tensor contracts."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import shapely
import torch
from ludus_renderer import PRIM_BEV_ROAD_SURFACE
from omnidreams_game_engine.conditioning import (
    _bev_presentation_frames,
    _build_bev_road_surface_pool,
)
from omnidreams_game_engine.game_map.types import GameMapElement
from shapely.geometry import LineString, Polygon

pytestmark = pytest.mark.ci_cpu


def test_bev_presentation_preserves_renderer_bytes_in_tchw_layout() -> None:
    source = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 4, 4)

    result = _bev_presentation_frames(source)

    assert result.shape == (2, 4, 3, 4)
    assert result.dtype is torch.uint8
    assert result.is_contiguous()
    assert torch.equal(result.permute(0, 2, 3, 1), source)


@pytest.mark.parametrize(
    "source",
    [
        torch.zeros(1, 3, 4, 3, dtype=torch.float32),
        torch.zeros(1, 3, 4, 3, dtype=torch.uint8),
        torch.zeros(3, 4, 3, dtype=torch.uint8),
    ],
)
def test_bev_presentation_rejects_non_renderer_contract(source: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="uint8 THWC RGBA"):
        _bev_presentation_frames(source)


def test_bev_road_surface_pool_triangulates_concave_pavement() -> None:
    surface = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    pool = _build_bev_road_surface_pool(
        (cast(GameMapElement, SimpleNamespace(surface_world=surface)),),
        torch.device("cpu"),
    )

    assert pool is not None
    assert pool.prim_type_id == PRIM_BEV_ROAD_SURFACE
    assert pool.timestamped_varrays_prefix_sum.tolist() == [1]
    assert pool.varrays_prefix_sum.tolist() == [6]
    assert pool.triangle_prefix_sum.tolist() == [4]
    assert torch.allclose(pool.vertices[:, 2], torch.full((6,), -0.01))
    points = pool.vertices[:, :2]
    triangle_points = points[pool.triangles.to(torch.int64)]
    doubled_areas = torch.abs(
        (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
        - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
        * (triangle_points[:, 2, 0] - triangle_points[:, 0, 0])
    )
    assert torch.isclose(doubled_areas.sum() / 2.0, torch.tensor(7.0))


@pytest.mark.parametrize(
    "surfaces",
    [
        [Polygon([(1, 1), (9, 8), (8.5, 9), (0.5, 2)])],
        [LineString([(1, 2), (4, 6), (8, 8)]).buffer(0.4)],
        [
            Polygon([(1, 1), (8, 1), (8, 2), (2, 2), (2, 8), (1, 8)]),
            Polygon([(3, 3), (8, 3), (8, 8), (3, 8)]),
        ],
    ],
)
def test_elevated_bev_clips_to_pavement_and_preserves_ground_height(
    surfaces: list[Polygon],
) -> None:
    ground = np.asarray([[0, 0, 0], [10, 0, 2], [10, 10, 7], [0, 10, 3]], np.float32)
    elements = tuple(
        cast(
            GameMapElement,
            SimpleNamespace(
                surface_world=np.column_stack(
                    (
                        np.asarray(surface.exterior.coords),
                        np.zeros(len(surface.exterior.coords)),
                    )
                ).astype(np.float32)
            ),
        )
        for surface in surfaces
    )
    pool = _build_bev_road_surface_pool(
        elements,
        torch.device("cpu"),
        ground_vertices=ground,
        ground_faces=np.asarray([[0, 1, 2], [0, 2, 3]], np.int32),
    )
    assert pool is not None
    xyz = pool.vertices.numpy()
    vertex_starts = np.r_[0, pool.varrays_prefix_sum.numpy()[:-1]]
    triangle_counts = np.diff(np.r_[0, pool.triangle_prefix_sum.numpy()])
    indices = (
        pool.triangles.numpy() + np.repeat(vertex_starts, triangle_counts)[:, None]
    )
    rendered = shapely.union_all(shapely.polygons(xyz[indices, :2]))
    expected = shapely.union_all(surfaces)
    assert rendered.symmetric_difference(expected).area < 1e-5
    x, y = xyz[:, :2].T
    height = np.where(x >= y, 0.2 * x + 0.5 * y, 0.4 * x + 0.3 * y)
    np.testing.assert_allclose(xyz[:, 2], height - 0.01, atol=1e-6)


def test_elevated_bev_keeps_pavement_in_local_culling_bounds() -> None:
    x, y = np.meshgrid(np.arange(9), np.arange(9))
    ground = np.column_stack((x.ravel(), y.ravel(), x.ravel() * 0.1)).astype(np.float32)
    cells = np.asarray([j * 9 + i for j in range(8) for i in range(8)])
    faces = np.concatenate(
        (cells[:, None] + [0, 1, 10], cells[:, None] + [0, 10, 9])
    ).astype(np.int32)
    surface = LineString([(0, 1), (4, 4), (8, 7)]).buffer(0.4)
    xy = np.asarray(surface.exterior.coords)
    element = cast(
        GameMapElement,
        SimpleNamespace(surface_world=np.column_stack((xy, np.zeros(len(xy))))),
    )
    pool = _build_bev_road_surface_pool(
        (element,), torch.device("cpu"), ground_vertices=ground, ground_faces=faces
    )
    assert pool is not None
    pieces = np.split(pool.vertices.numpy(), pool.varrays_prefix_sum.numpy()[:-1])
    # Distant pavement must be independently culled before CUDA tessellation.
    assert all(np.all(np.ptp(piece[:, :2], axis=0) <= 1.00001) for piece in pieces)
