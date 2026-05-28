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

import numpy as np
import pytest
from omnidreams.grpc.protos import common_pb2
from omnidreams.grpc.utils import (
    compute_camera_poses_from_rig,
    parse_rig_to_camera_transforms,
)

pytestmark = pytest.mark.ci_cpu


def _pose(x: float, y: float, z: float) -> common_pb2.Pose:
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=z),
        quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def test_session_rig_to_camera_transforms_are_mapped_by_camera_order() -> None:
    transforms = parse_rig_to_camera_transforms(
        [_pose(1.0, 2.0, 3.0), _pose(4.0, 5.0, 6.0)],
        ["front", "rear"],
    )

    front = np.eye(4, dtype=np.float32)
    front[:3, 3] = [1.0, 2.0, 3.0]
    rear = np.eye(4, dtype=np.float32)
    rear[:3, 3] = [4.0, 5.0, 6.0]

    assert list(transforms) == ["front", "rear"]
    np.testing.assert_allclose(transforms["front"], front)
    np.testing.assert_allclose(transforms["rear"], rear)


def test_session_rig_to_camera_count_must_match_camera_specs() -> None:
    with pytest.raises(ValueError, match="exactly one Pose per camera_spec"):
        parse_rig_to_camera_transforms([_pose(1.0, 0.0, 0.0)], ["front", "rear"])


def test_camera_pose_uses_session_rig_to_camera_transform() -> None:
    rig_poses = np.eye(4, dtype=np.float32)[None, ...]
    transforms = parse_rig_to_camera_transforms(
        [_pose(1.0, 2.0, 3.0)],
        ["front"],
    )

    camera_poses = compute_camera_poses_from_rig(rig_poses, transforms["front"])

    expected = np.eye(4, dtype=np.float32)[None, ...]
    expected[0, :3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(camera_poses, expected)
