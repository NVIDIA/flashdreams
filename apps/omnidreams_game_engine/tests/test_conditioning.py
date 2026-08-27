# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for model versus UI conditioning tensor contracts."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from ludus_renderer import PRIM_BEV_ROAD_SURFACE
from omnidreams_game_engine.conditioning import (
    _bev_presentation_frames,
    _build_bev_road_surface_pool,
)
from omnidreams_game_engine.game_map.types import GameMapElement

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
