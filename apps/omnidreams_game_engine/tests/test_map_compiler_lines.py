# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler checks for complete bounded solid markings and unchanged dash paths."""

from types import SimpleNamespace
from typing import Any, cast

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
    rows = cast(list[dict[str, Any]], _lane_line_rows(game_map))
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


@pytest.mark.parametrize("style", ["SOLID_DASHED", "DASHED_SOLID"])
@pytest.mark.parametrize("simplified", [True, False])
@pytest.mark.parametrize("color", ["YELLOW", "WHITE"])
def test_long_mixed_stripes_preserve_endpoint_and_dash_phase(
    tmp_path, style: str, simplified: bool, color: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ludus_renderer.clipgt import _create_dual_line, read_lane_line

    t = np.linspace(0, 2, 550)
    points = np.column_stack((100 * np.cos(t), 100 * np.sin(t), 4 * t)).astype(
        np.float32
    )
    marking = GameMapLineMarking("curve", points, style, color)
    m = cast(
        ResolvedGameMap,
        SimpleNamespace(
            map_id="debug", schema_version=2, lane_dividers=(), line_markings=(marking,)
        ),
    )
    path = tmp_path / "lane_line.parquet"
    pq.write_table(pa.Table.from_pylist(_lane_line_rows(m)), path)
    paint = read_lane_line(path, simplify_dual_lines=simplified)
    solid_stripes = paint.yellow_solid if color == "YELLOW" else paint.white_solid
    dashed_stripes = paint.yellow_dashed if color == "YELLOW" else paint.white_dashed
    assert all(len(rail) <= 256 for rail in solid_stripes)
    solid = np.concatenate(
        [solid_stripes[0], *(rail[3:] for rail in solid_stripes[1:])]
    )
    if simplified:
        np.testing.assert_array_equal(solid, points)
        assert not dashed_stripes
    else:
        expected_solid, expected_dashes = _create_dual_line(
            points, left_dashed=style == "SOLID_DASHED"
        )
        np.testing.assert_array_equal(solid, expected_solid)
        for actual, expected in zip(dashed_stripes, expected_dashes, strict=True):
            np.testing.assert_array_equal(actual, expected)
