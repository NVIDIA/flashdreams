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

from __future__ import annotations

import pytest
import torch
from ludus_renderer import CUBE_FLAG_WIREFRAME, PRIM_OBSTACLE
from ludus_renderer.clipgt import OBSTACLE_COLORS_V3
from omnidreams.grpc.protos import common_pb2, video_model_pb2
from omnidreams.grpc.utils import (
    dynamic_state_to_ludus_cube_pool,
    proto_to_dict,
)

pytestmark = pytest.mark.ci_cpu


def _pose_at_time(timestamp_us: int, x: float) -> common_pb2.PoseAtTime:
    return common_pb2.PoseAtTime(
        timestamp_us=timestamp_us,
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=1.0, z=2.0),
            quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )


def _actor(*poses: common_pb2.PoseAtTime) -> video_model_pb2.DynamicActor:
    return video_model_pb2.DynamicActor(
        class_id=video_model_pb2.ActorClassId.CAR,
        bbox_dims=common_pb2.AABB(size_x=4.0, size_y=2.0, size_z=1.5),
        trajectory=common_pb2.Trajectory(poses=list(poses)),
    )


def test_empty_dynamic_state_produces_no_cube_pool() -> None:
    pool = dynamic_state_to_ludus_cube_pool(
        {"actors": []}, frame_timestamps_us=[1_000_000], device="cpu"
    )

    assert pool is None


def test_dynamic_actor_proto_converts_to_ludus_cube_pool() -> None:
    state = video_model_pb2.DynamicWorldState(
        actors=[_actor(_pose_at_time(0, 0.0), _pose_at_time(10, 10.0))]
    )

    pool = dynamic_state_to_ludus_cube_pool(
        proto_to_dict(state),
        frame_timestamps_us=[0, 5, 10, 15],
        device=torch.device("cpu"),
    )

    assert pool is not None
    assert int(pool.prim_type_id) == PRIM_OBSTACLE
    assert int(pool.render_flags) == CUBE_FLAG_WIREFRAME
    torch.testing.assert_close(
        pool.cube_ts_prefix_sum, torch.tensor([3], dtype=torch.int32)
    )
    torch.testing.assert_close(pool.timestamps_us, torch.tensor([0, 5, 10]))
    torch.testing.assert_close(pool.track_timestamps_us, torch.tensor([0, 5, 10]))
    torch.testing.assert_close(pool.scales, torch.tensor([[4.0, 2.0, 1.5]]))
    torch.testing.assert_close(
        pool.translations,
        torch.tensor([[0.0, 1.0, 2.0], [5.0, 1.0, 2.0], [10.0, 1.0, 2.0]]),
    )
    torch.testing.assert_close(
        pool.quaternions,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(3, -1),
    )
    torch.testing.assert_close(
        pool.colors,
        torch.tensor([[*OBSTACLE_COLORS_V3["Car"][0], *OBSTACLE_COLORS_V3["Car"][1]]]),
    )


def test_dynamic_actor_trajectory_is_sorted_and_range_limited() -> None:
    state = video_model_pb2.DynamicWorldState(
        actors=[_actor(_pose_at_time(10, 10.0), _pose_at_time(0, 0.0))]
    )

    pool = dynamic_state_to_ludus_cube_pool(
        proto_to_dict(state),
        frame_timestamps_us=[5, 15],
        device="cpu",
    )

    assert pool is not None
    torch.testing.assert_close(pool.track_timestamps_us, torch.tensor([5]))
    torch.testing.assert_close(pool.translations, torch.tensor([[5.0, 1.0, 2.0]]))


def test_dynamic_actor_with_nonpositive_bbox_is_rejected() -> None:
    state = {
        "actors": [
            {
                "class_id": "CAR",
                "bbox_dims": {"size_x": 0.0, "size_y": 2.0, "size_z": 1.5},
                "trajectory": {
                    "poses": [
                        {
                            "timestamp_us": 0,
                            "pose": {
                                "vec": {"x": 0.0, "y": 0.0, "z": 0.0},
                                "quat": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                            },
                        }
                    ]
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="nonpositive bbox"):
        dynamic_state_to_ludus_cube_pool(state, frame_timestamps_us=[0], device="cpu")
