# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The host application's path, as our control render reads it."""

from __future__ import annotations

import numpy as np
import pytest
from cosmos3_nano_sim_auto.apps.interactive_drive.poses import (
    SuppliedPoses,
    as_recorded,
    rig_tracks,
)
from cosmos3_nano_sim_auto.impl.conditioning.driving import PoseSource
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera import (
    camera_pose,
)

pytestmark = pytest.mark.ci_cpu


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    """A rig pose at ``x`` metres along the road, turned by ``yaw``."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose[0, 3] = x
    return pose


def _sensor_to_rig(forward: float, left: float) -> np.ndarray:
    """Where a camera is bolted, in the host's forward-left-up rig frame."""
    offset = np.eye(4, dtype=np.float32)
    offset[0, 3] = forward
    offset[1, 3] = left
    return offset


def _supplied(
    anchor: np.ndarray | None = None, *, speed: float = 0.0, fps: float = 1.0
) -> SuppliedPoses:
    """Poses opening at the origin, standing still unless a speed is given.

    A frame a second, so a test's extrapolated distances read as its speed.
    """
    return SuppliedPoses(
        np.eye(4, dtype=np.float32) if anchor is None else anchor,
        opening_rig_to_world=_pose(0.0),
        speed_mps=speed,
        seconds_per_frame=1.0 / fps,
    )


def test_supplied_poses_satisfies_the_pose_source_a_render_asks_for() -> None:
    assert isinstance(_supplied(), PoseSource)


def test_presented_position_tracks_shifted_prefetch_and_consumes_partial_spans():
    supplied = SuppliedPoses(
        np.eye(4),
        opening_rig_to_world=_pose(0),
        speed_mps=2,
        seconds_per_frame=1,
        delay_frames=3,
        track_positions=True,
    )
    supplied.supply(np.stack([_pose(0), _pose(10), _pose(20)]))
    drawn = supplied.poses(0, 6)
    assert drawn[:, 0, 3].tolist() == [2, 4, 6, 6, 16, 26]
    assert supplied.take_position(2).tolist() == [4, 0]
    assert supplied.take_position(3).tolist() == [16, 0]
    assert supplied.take_position(1).tolist() == [26, 0]
    assert supplied._drawn_positions == []
    with pytest.raises(ValueError, match="tracked, drawn frames"):
        supplied.take_position(1)


def test_untracked_workers_do_not_retain_presented_positions():
    supplied = _supplied()
    supplied.poses(0, 100)
    assert supplied._drawn_positions is None
    with pytest.raises(ValueError, match="tracked, drawn frames"):
        supplied.take_position(1)


def test_a_drawn_rig_lands_where_the_host_rasterizer_puts_it() -> None:
    """The whole chain agrees with the app's own convention, which is the point.

    Interactive Drive hands ludus ``inv(rig @ sensor_to_rig)``. Our render path
    instead composes the rig's shape out of camera tracks and passes the result
    through ``camera_pose``, which carries an axis change the app does not. If
    the two disagree the map is drawn from a camera pointing somewhere else, and
    nothing raises, so this is the assertion that catches it.
    """
    rig = _sensor_to_rig(1.5, 0.0), _sensor_to_rig(0.0, -0.8)
    tracks = rig_tracks(rig)
    supplied = _supplied(rig[0])
    rig_poses = np.stack([_pose(0.0), _pose(2.0, yaw=0.1)])

    supplied.supply(rig_poses)
    anchor = supplied.poses(0, 2)

    for view, sensor_to_rig in enumerate(rig):
        bolted = np.linalg.inv(tracks[0][0]) @ tracks[view][0]
        ours = camera_pose(anchor @ bolted)
        theirs = np.linalg.inv(rig_poses @ sensor_to_rig)
        assert np.allclose(ours, theirs, atol=1e-5), f"camera {view}"


def test_poses_are_read_back_a_span_at_a_time_in_order() -> None:
    supplied = _supplied()
    supplied.supply(np.stack([_pose(float(frame)) for frame in range(5)]))

    assert supplied.next_frame == 0
    first = supplied.poses(0, 2)
    assert [float(pose[0, 3]) for pose in first] == [0.0, 1.0]
    assert supplied.next_frame == 2
    assert [float(pose[0, 3]) for pose in supplied.poses(2, 5)] == [2.0, 3.0, 4.0]


def test_a_span_may_cross_the_chunks_it_was_supplied_in() -> None:
    supplied = _supplied()
    supplied.supply(np.stack([_pose(0.0), _pose(1.0)]))
    supplied.supply(np.stack([_pose(2.0), _pose(3.0)]))

    assert [float(pose[0, 3]) for pose in supplied.poses(0, 4)] == [0.0, 1.0, 2.0, 3.0]


def test_poses_refuse_a_span_that_does_not_carry_on() -> None:
    supplied = _supplied()
    supplied.supply(np.stack([_pose(0.0), _pose(1.0)]))
    supplied.poses(0, 1)

    with pytest.raises(ValueError, match="carry on from 1"):
        supplied.poses(0, 1)


def test_the_path_carries_straight_on_past_what_the_host_supplied() -> None:
    """The opening, where the rollout wants a map before the app has driven.

    The control window is rendered while the rollout is built, so these frames
    are asked for before the app has sampled a trajectory at all. Refusing them
    would mean no drive could start.
    """
    supplied = _supplied(speed=2.0)

    span = supplied.poses(0, 4)

    assert [float(pose[0, 3]) for pose in span] == [2.0, 4.0, 6.0, 8.0]


def test_carrying_on_resumes_from_the_last_pose_the_host_did_supply() -> None:
    supplied = _supplied(speed=2.0)
    supplied.supply(np.stack([_pose(10.0), _pose(11.0)]))

    span = supplied.poses(0, 4)

    assert [float(pose[0, 3]) for pose in span] == [10.0, 11.0, 13.0, 15.0]


def test_poses_for_ground_already_drawn_are_dropped() -> None:
    """The app's opening commands arrive after their map is drawn.

    It supplies a chunk for frames it has just been shown, and those were drawn
    from the extrapolation. Queueing them would put the app a window behind for
    the rest of the drive rather than only at the start.
    """
    supplied = _supplied(speed=1.0)
    supplied.poses(0, 4)
    supplied.supply(np.stack([_pose(float(frame)) for frame in range(4)]))
    supplied.supply(np.stack([_pose(100.0), _pose(101.0)]))

    span = supplied.poses(4, 6)

    assert [float(pose[0, 3]) for pose in span] == [100.0, 101.0]


def test_an_empty_chunk_of_poses_is_refused() -> None:
    supplied = _supplied()

    with pytest.raises(ValueError, match="cannot be empty"):
        supplied.supply(np.zeros((0, 4, 4), dtype=np.float32))


def test_as_recorded_refuses_anything_that_is_not_a_pose() -> None:
    with pytest.raises(ValueError, match=r"\[4, 4\] or \[T, 4, 4\]"):
        as_recorded(np.zeros((4, 3), dtype=np.float32))


def test_a_rig_needs_a_camera_to_take_its_shape_from() -> None:
    with pytest.raises(ValueError, match="at least one camera"):
        rig_tracks([])
