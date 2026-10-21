# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for map gaps, presentation timing, and coverage recovery."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
    MultiViewRenderBackend,
    MultiViewSettings,
)
from cosmos3_nano_sim_auto.apps.interactive_drive.coverage import (
    OUTSIDE_MAP_WARNING,
    MapCoverage,
)
from cosmos3_nano_sim_auto.apps.interactive_drive.poses import SuppliedPoses
from interactive_drive.config import ChunkConfig, RasterConfig

pytestmark = pytest.mark.ci_cpu


def _map(lines=(), triangles=()):
    def pools(vertices):
        return (
            [SimpleNamespace(vertices=torch.tensor(vertices).reshape(-1, 3))]
            if len(vertices)
            else []
        )

    return SimpleNamespace(
        polyline_pools=pools(lines),
        polygon_pools=pools(triangles),
        # Actors outside the static map must not expand its coverage.
        cube_pools=[SimpleNamespace(translations=torch.tensor([[1000, 1000, 0]]))],
    )


def test_coverage_preserves_gaps_and_checks_segment_interiors():
    coverage = MapCoverage(
        _map(lines=[[[0, 0, 0], [0, 1000, 0]], [[200, 0, 0], [200, 1000, 0]]])
    )
    assert coverage.contains([10, 500])
    assert not coverage.contains([100, 500])  # Inside the overall bounding box.
    assert not coverage.contains([1000, 1000])  # An actor is not a map.
    assert not coverage.contains([0, 1100])  # Past the segment's endpoint.


@pytest.mark.parametrize("reverse", [False, True])
def test_large_triangle_interiors_are_covered_in_either_winding(reverse):
    triangle = [[0, 0, 0], [1000, 0, 0], [0, 1000, 0]]
    coverage = MapCoverage(_map(triangles=[triangle[::-1] if reverse else triangle]))
    assert coverage.contains([200, 200])
    assert not coverage.contains([900, 900])  # In its box but outside its surface.


def test_empty_and_degenerate_geometry_do_not_invent_coverage():
    empty = MapCoverage(_map())
    assert not empty.contains([0, 0])
    assert all(empty.update([0, 0]) is None for _ in range(4))
    point = MapCoverage(_map(triangles=[[[0, 0, 0]] * 3]))
    assert point.contains([0, 0])
    assert not point.contains([100, 100])


def test_warning_debounces_exit_and_clears_on_reentry():
    coverage = MapCoverage(_map(lines=[[[0, -100, 0], [0, 100, 0]]]))
    assert coverage.update([40, 0]) is None
    assert coverage.update([0, 0]) is None  # A brief excursion does not latch.
    assert coverage.update([40, 0]) is None
    assert coverage.update([40, 0]) == OUTSIDE_MAP_WARNING
    assert coverage.update([28, 0]) == OUTSIDE_MAP_WARNING  # Hysteresis.
    assert coverage.update([20, 0]) is None
    assert coverage.update([40, 0]) is None  # Exit debounce starts afresh.


@pytest.mark.parametrize("delay", [0, 3])
def test_warning_follows_presented_conditioning_poses_and_resets(delay):
    backend = MultiViewRenderBackend(
        ChunkConfig(), RasterConfig(), settings=MultiViewSettings(device="cpu")
    )
    backend._coverage = MapCoverage(_map(lines=[[[0, -100, 0], [0, 100, 0]]]))
    poses = SuppliedPoses(
        np.eye(4),
        opening_rig_to_world=np.eye(4),
        speed_mps=0,
        seconds_per_frame=1,
        delay_frames=delay,
        track_positions=True,
    )
    backend._poses = poses
    path = np.repeat(np.eye(4)[None], 5, axis=0)
    path[:, 0, 3] = [0, 100, 100, 100, 0]
    poses.supply(path)
    # All controls, including the return, are rendered before presentation.
    poses.poses(0, len(path) + delay)
    assert backend._coverage_warning(1 + delay) is None
    assert backend._coverage_warning(1) is None
    assert backend._coverage_warning(1) == OUTSIDE_MAP_WARNING
    assert backend._coverage_warning(1) == OUTSIDE_MAP_WARNING
    assert backend._coverage_warning(1) is None
    assert poses._drawn_positions == []  # Consumed history is released.

    backend._coverage.update([100, 0])
    backend._coverage.update([100, 0])
    backend.reset_scene_conditioning()
    assert backend._coverage_warning(1) is None
    assert backend._poses is None
