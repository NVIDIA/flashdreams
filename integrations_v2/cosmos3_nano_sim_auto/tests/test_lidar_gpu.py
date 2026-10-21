# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU smoke checks for joint LiDAR generation and reusable presentation."""

import pytest
import torch

from flashdreams.core.visualization.lidar import LidarView, LidarViewport

from .test_lidar import make_rollout

pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_joint_lidar_cuda_rollout_and_bounded_history():
    """Exercise real MoT CUDA kernels with tiny random weights across cache eviction."""
    rollout = make_rollout(
        conditioned=True, lidar_conditioned=True, device="cuda", dtype=torch.bfloat16
    )
    ranges = []
    while not rollout.is_finished:
        _, rgb = rollout.step()
        ranges.append(rollout.last_lidar_range)
        assert rgb.is_cuda and rollout.last_lidar.is_cuda
        assert torch.isfinite(rgb).all() and torch.isfinite(rollout.last_lidar).all()
        assert len(rollout.history) <= 1
    assert ranges == [(0, 1), (1, 5), (5, 9), (9, 13)]
    fields, _ = rollout._memory()
    assert set(fields.causal_step_id.tolist()) == {0, 3}
    assert not fields.is_noisy.any()
    torch.cuda.synchronize()


@pytest.mark.parametrize("view", list(LidarView))
def test_lidar_cuda_view_matches_cpu(view):
    """Compare all GPU display modes to the CPU renderer on the same sweep."""
    ranges = torch.tensor(
        [
            [[10.0, 20.0, 30.0, 0.0], [15.0, 25.0, 35.0, 45.0]],
            [[0.0, 0.3, 0.6, 0.9], [0.2, 0.4, 0.6, 0.8]],
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    rays = torch.tensor([1.0, 0.0, 0.0]).expand(2, 4, 3)
    viewport = LidarViewport(width=320, height=240)
    expected = viewport.render_sweep(ranges, rays, view=view)
    actual = viewport.render_sweep(ranges.cuda(), rays.cuda(), view=view)
    assert actual.is_cuda and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)
