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
import torch
from ludus_renderer import CubePool
from omnidreams.webrtc.actors import (
    ACTOR_PRESETS,
    RIG_HEIGHT_M,
    actors_to_cube_pool,
    clone_template_pool,
    extract_parked_templates,
    find_empty_gap,
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


def _scene_pool(tracks: list[dict]) -> CubePool:
    """Synthetic CubePool from per-track specs (xy, n, t0_s, length, drift)."""
    track_ts, translations, quaternions, scales, colors, lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for track in tracks:
        n = track.get("n", 12)
        t0 = track.get("t0_s", 0.0)
        ts = torch.tensor(
            [int((t0 + 0.5 * i) * 1e6) for i in range(n)], dtype=torch.int64
        )
        xy = np.asarray(track["xy"], dtype=np.float64)
        drift = track.get("drift", 0.0)
        pos = torch.tensor(
            [[xy[0] + drift * i / max(n - 1, 1), xy[1], 0.7] for i in range(n)],
            dtype=torch.float64,
        )
        # Deterministic sub-centimeter jitter, like real perception tracks.
        pos[:, 0] += 0.01 * torch.sin(torch.arange(n, dtype=torch.float64))
        length = track.get("length", 4.5)
        track_ts.append(ts)
        translations.append(pos)
        quaternions.append(
            torch.tensor([[0.0, 0.0, 0.2, 0.98]]).repeat(n, 1).to(torch.float64)
        )
        scales.append(torch.tensor([[length, 1.9, 1.5]], dtype=torch.float64))
        colors.append(torch.rand(1, 6, dtype=torch.float64))
        lengths.append(n)
    all_ts = torch.cat(track_ts)
    return CubePool(
        timestamps_us=torch.unique(all_ts).sort()[0],
        cube_ts_prefix_sum=torch.cumsum(
            torch.tensor(lengths, dtype=torch.int32), dim=0
        ).to(torch.int32),
        track_timestamps_us=all_ts,
        translations=torch.cat(translations),
        quaternions=torch.cat(quaternions),
        scales=torch.cat(scales),
        colors=torch.cat(colors),
    )


def test_extract_templates_filters_and_sorts_by_distance():
    pool = _scene_pool(
        [
            {"xy": (40.0, -7.0)},  # good, farther
            {"xy": (15.0, -7.0)},  # good, nearest -> first
            {"xy": (20.0, -7.0), "drift": 4.0},  # moving: rejected
            {"xy": (25.0, -7.0), "n": 4},  # short coverage: rejected
            {"xy": (30.0, -7.0), "length": 8.0},  # truck-sized: rejected
            {"xy": (35.0, -7.0), "t0_s": 2.0},  # starts too late: rejected
        ]
    )
    templates = extract_parked_templates([pool], ego_pose=_ego_pose(), t0_us=0)
    assert [t.source_fwd_m for t in templates] == pytest.approx([15.0, 40.0], abs=0.05)
    assert templates[0].source_lateral_m == pytest.approx(-7.0, abs=0.05)
    assert templates[0].translations.shape == (12, 3)


def test_find_empty_gap_targets_largest_free_span():
    pool = _scene_pool([{"xy": (25.0, -7.0)}, {"xy": (40.0, -7.2)}])
    center, width = find_empty_gap(
        [pool], ego_pose=_ego_pose(), lateral_m=-7.0, fwd_range=(20.0, 65.0)
    )
    # Occupied: 25 and 40, each +-(4.5/2 + 1.5) -> largest gap is (43.75, 65).
    assert center == pytest.approx((43.75 + 65.0) / 2, abs=0.05)
    assert width == pytest.approx(65.0 - 43.75, abs=0.05)

    # Actors on other lateral lines do not shrink the gap.
    _, full_width = find_empty_gap(
        [pool], ego_pose=_ego_pose(), lateral_m=7.0, fwd_range=(20.0, 65.0)
    )
    assert full_width == pytest.approx(45.0, abs=1e-6)


def test_clone_template_pool_moves_rigidly_and_preserves_track():
    ego = _ego_pose(x=3.0, y=-2.0, yaw_deg=90.0)  # forward +y, left -x
    source = _scene_pool([{"xy": (10.0, 30.0)}])
    (template,) = extract_parked_templates([source], ego_pose=ego, t0_us=0)

    pool = clone_template_pool([(template, 30.0, -7.0)], ego_pose=ego)
    first = pool.translations[0].cpu().numpy()
    np.testing.assert_allclose(first[:2], [3.0 + 7.0, -2.0 + 30.0], atol=1e-6)
    # Rigid shift: per-frame jitter, z, orientation, size, colors all survive.
    np.testing.assert_allclose(
        (pool.translations - pool.translations[0]).cpu().numpy(),
        (template.translations - template.translations[0]).cpu().numpy(),
        atol=1e-9,
    )
    assert torch.equal(pool.quaternions, template.quaternions)
    assert torch.equal(pool.scales, template.scale)
    assert torch.equal(pool.colors, template.colors)

    two = clone_template_pool(
        [(template, 30.0, -7.0), (template, 40.0, -7.0)], ego_pose=ego
    )
    assert two.cube_ts_prefix_sum.cpu().tolist() == [12, 24]
    assert two.scales.shape[0] == 2
