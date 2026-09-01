# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for semantic spawn rendering."""

import numpy as np
import pytest
from omnidreams_game_engine.game_map.spawn_render import _styled_line_geometry
from shapely.geometry import Point

pytestmark = pytest.mark.ci_cpu


def test_dashed_lane_marking_has_world_space_gaps() -> None:
    line = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=np.float32)

    dashed = _styled_line_geometry([(line, "DASHED_SINGLE")], 0.12)
    solid = _styled_line_geometry([(line, "SOLID_GROUP")], 0.12)

    assert dashed.intersects(Point(1.5, 0.0))
    assert not dashed.intersects(Point(5.0, 0.0))
    assert dashed.intersects(Point(10.0, 0.0))
    assert solid.intersects(Point(5.0, 0.0))
