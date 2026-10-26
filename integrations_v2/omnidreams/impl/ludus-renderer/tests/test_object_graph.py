# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for mutable collision topology."""

import pytest
from ludus_renderer import InvisibleBarrier, PhysicsObjectGraph

pytestmark = pytest.mark.ci_cpu


def test_adding_and_replacing_barriers_preserves_endpoint_heights() -> None:
    graph = PhysicsObjectGraph()
    for start_z, end_z in ((8.0, 12.0), (-4.0, -2.0)):
        graph.upsert_barrier(
            "wall",
            InvisibleBarrier((-5, 0), (5, 0), start_z_m=start_z, end_z_m=end_z),
        )
        active = graph.copy_for_physx((0, 0), 20)
        assert len(active.barriers) == 1
        wall = active.barriers[0]
        assert wall.barrier_id == "wall"
        assert (wall.start_z_m, wall.end_z_m) == (start_z, end_z)
