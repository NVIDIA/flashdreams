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

"""CPU regression tests for the interactive-driving BEV projection."""

import pytest
import torch
from interactive_drive.rasterizer import (
    _build_bev_ego_car_pool,
    _level_rig_poses_for_bev,
)
from ludus_renderer import PRIM_EGO_OBSTACLE

pytestmark = pytest.mark.ci_cpu


def test_bev_pose_discards_driving_pitch_and_roll() -> None:
    yaw = torch.tensor(0.65)
    pitch = torch.tensor(-0.24)
    roll = torch.tensor(0.18)

    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    rotate_yaw = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotate_pitch = torch.tensor(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ]
    )
    rotate_roll = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ]
    )
    rig_pose = torch.eye(4).unsqueeze(0)
    rig_pose[0, :3, :3] = rotate_yaw @ rotate_pitch @ rotate_roll
    rig_pose[0, :3, 3] = torch.tensor([12.0, -4.0, 1.5])

    level_pose = _level_rig_poses_for_bev(rig_pose)[0]

    torch.testing.assert_close(level_pose[:3, :3], rotate_yaw)
    torch.testing.assert_close(level_pose[:3, 3], rig_pose[0, :3, 3])


def test_bev_ego_car_pool_uses_runtime_pose_and_vehicle_dimensions() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[0, :3, 3] = torch.tensor([1.0, 2.0, 0.0])
    poses[1, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poses[1, :3, 3] = torch.tensor([3.0, 4.0, 0.5])

    pool = _build_bev_ego_car_pool(
        rig_poses=poses,
        timestamps_us=torch.tensor([100, 200]),
        dimensions_lwh=(4.8, 2.0, 1.6),
        track_capacity=4,
    )

    torch.testing.assert_close(pool.scales, torch.tensor([[4.8, 2.0, 1.6]]))
    torch.testing.assert_close(
        pool.translations,
        torch.tensor(
            [
                [1.0, 2.0, 0.8],
                [3.0, 4.0, 1.3],
                [3.0, 4.0, 1.3],
                [3.0, 4.0, 1.3],
            ]
        ),
    )
    torch.testing.assert_close(
        pool.quaternions[1],
        torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]),
    )
    assert pool.track_timestamps_us.tolist() == [100, 200, 300, 400]
    assert pool.prim_type_id == PRIM_EGO_OBSTACLE
    assert pool.colors[0, 1].item() == pytest.approx(185.0 / 255.0)
