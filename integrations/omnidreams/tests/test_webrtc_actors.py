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

"""CPU-only unit tests for user-spawned WebRTC actors."""

from __future__ import annotations

import numpy as np
import pytest
from omnidreams.webrtc.actors import (
    ACTOR_PRESETS,
    RIG_HEIGHT_M,
    actors_to_cube_pool,
    spawn_actor_ahead,
)
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.ci_cpu


def _ego_pose(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", np.deg2rad(yaw_deg)).as_matrix()
    pose[:3, 3] = [x, y, 0.0]
    return pose


def test_spawn_ahead_places_actor_along_heading():
    actor = spawn_actor_ahead(
        preset="car",
        ego_pose=_ego_pose(x=5.0, y=2.0, yaw_deg=90.0),
        spawn_timestamp_us=1_000_000,
        distance_m=10.0,
        lateral_m=1.0,
    )
    # Heading +90deg: forward is +y, left is -x.
    np.testing.assert_allclose(actor.translation[0], 4.0, atol=1e-5)
    np.testing.assert_allclose(actor.translation[1], 12.0, atol=1e-5)
    # Bbox center sits half its height above the road plane (the ego pose is
    # the rig origin, RIG_HEIGHT_M above the road).
    np.testing.assert_allclose(
        actor.translation[2],
        ACTOR_PRESETS["car"][1][2] / 2.0 - RIG_HEIGHT_M,
        atol=1e-6,
    )
    np.testing.assert_allclose(actor.velocity, np.zeros(3), atol=1e-6)


def test_spawn_with_speed_moves_along_heading():
    actor = spawn_actor_ahead(
        preset="truck",
        ego_pose=_ego_pose(),
        spawn_timestamp_us=0,
        distance_m=20.0,
        speed_mps=5.0,
    )
    later = actor.translation_at(2_000_000)  # +2 s
    np.testing.assert_allclose(later[0] - actor.translation[0], 10.0, atol=1e-4)
    np.testing.assert_allclose(later[1], actor.translation[1], atol=1e-6)


def test_spawn_heading_ignores_camera_pitch():
    pose = _ego_pose()
    pose[:3, :3] = Rotation.from_euler("y", np.deg2rad(-20.0)).as_matrix()
    actor = spawn_actor_ahead(
        preset="cone", ego_pose=pose, spawn_timestamp_us=0, distance_m=8.0
    )
    # Forward projected to the ground plane: full 8 m in x, none in z beyond
    # the half-height-minus-rig offset.
    np.testing.assert_allclose(actor.translation[0], 8.0, atol=1e-5)
    np.testing.assert_allclose(
        actor.translation[2],
        ACTOR_PRESETS["cone"][1][2] / 2.0 - RIG_HEIGHT_M,
        atol=1e-6,
    )


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        spawn_actor_ahead(preset="dragon", ego_pose=_ego_pose(), spawn_timestamp_us=0)


def test_actors_to_cube_pool_respects_spawn_time():
    frame_ts = [0, 33_333, 66_666, 99_999]
    early = spawn_actor_ahead(
        preset="car", ego_pose=_ego_pose(), spawn_timestamp_us=0, distance_m=10.0
    )
    late = spawn_actor_ahead(
        preset="cone",
        ego_pose=_ego_pose(),
        spawn_timestamp_us=66_666,
        distance_m=5.0,
    )
    pool = actors_to_cube_pool([early, late], frame_ts, device="cpu")
    assert pool is not None
    # Track lengths: early actor has all 4 frames, late actor only the last 2.
    lengths = np.diff(np.concatenate([[0], pool.cube_ts_prefix_sum.cpu().numpy()]))
    assert lengths.tolist() == [4, 2]
    assert pool.scales.shape[0] == 2

    # Not-yet-spawned actors produce no pool at all.
    future = spawn_actor_ahead(
        preset="car", ego_pose=_ego_pose(), spawn_timestamp_us=10_000_000
    )
    assert actors_to_cube_pool([future], frame_ts, device="cpu") is None


def test_pool_positions_track_constant_velocity():
    frame_ts = [0, 1_000_000]
    actor = spawn_actor_ahead(
        preset="car",
        ego_pose=_ego_pose(),
        spawn_timestamp_us=0,
        distance_m=10.0,
        speed_mps=3.0,
    )
    pool = actors_to_cube_pool([actor], frame_ts, device="cpu")
    assert pool is not None
    translations = pool.translations.cpu().numpy()
    np.testing.assert_allclose(translations[0][0], 10.0, atol=1e-4)
    np.testing.assert_allclose(translations[1][0], 13.0, atol=1e-4)
