# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler checks for complete bounded solid markings and unchanged dash paths."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from omnidreams_game_engine.game_map.compiler import _lane_line_rows
from omnidreams_game_engine.game_map.types import GameMapLineMarking, ResolvedGameMap

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("count", [2, 256, 257, 509, 510, 690])
@pytest.mark.parametrize("style", ["SOLID_SINGLE", "SOLID_GROUP", "DASHED_SINGLE"])
def test_compiled_markings_preserve_every_segment(count: int, style: str) -> None:
    t = np.linspace(0, 2 * np.pi, count)
    points = np.column_stack((100 * np.cos(t), 100 * np.sin(t), 4 * np.sin(t)))
    marking = GameMapLineMarking("curve", points, style, "YELLOW")
    game_map = cast(
        ResolvedGameMap,
        SimpleNamespace(
            map_id="debug-curved-paint",
            schema_version=2,
            lane_dividers=(),
            line_markings=(marking,),
        ),
    )
    rows = _lane_line_rows(game_map)
    rails = [
        np.asarray(
            [
                [point[axis] for axis in ("x", "y", "z")]
                for point in row["lane_line"]["line_rail"]
            ]
        )
        for row in rows
    ]
    if style == "DASHED_SINGLE":
        assert len(rails) == 1  # Keep distance/phase continuous for dash expansion.
    else:
        assert all(len(rail) <= 256 for rail in rails)
    assert len({row["key"]["label_class_id"] for row in rows}) == len(rows)
    restored = np.concatenate([rails[0]] + [rail[3:] for rail in rails[1:]])
    np.testing.assert_array_equal(restored, points)
    for left, right in zip(rails[:-1], rails[1:], strict=True):
        np.testing.assert_array_equal(left[-3:], right[:3])
    assert rows[0]["lane_line"]["is_first_point_physical_end"] == "true"
    assert rows[-1]["lane_line"]["is_last_point_physical_end"] == "true"
    assert all(
        row["lane_line"]["is_first_point_physical_end"] == "false" for row in rows[1:]
    )
    assert all(
        row["lane_line"]["is_last_point_physical_end"] == "false" for row in rows[:-1]
    )
