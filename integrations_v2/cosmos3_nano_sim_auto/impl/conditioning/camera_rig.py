# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical camera identities used by the model and scene bundles."""

from collections.abc import Sequence

CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)
"""Canonical rig order; each camera's index is its learned view identity.

Physical camera indices in source clips can use a different ordering.
"""


def view_ids(cameras: Sequence[str]) -> tuple[int, ...]:
    """The view id of each name, refusing a rig that is not in rig order.

    A view id is an index into :data:`CAMERAS`, and it is the only thing that
    says which seat on the car a view was filmed from. The drawing takes its
    intrinsics and pose from that camera, and final Sim-Auto checkpoints add
    the learned embedding for the same canonical ID. So a rig listed in the wrong
    order is not an error anywhere downstream. It draws every camera from
    somebody else's seat, at the right resolution, through a plausible lens,
    and the run comes out looking like a checkpoint having a bad day.

    Ascending rather than contiguous, because a run may take any subset: four
    views is the first four, and ``--camera`` picks one out to write on its
    own. What is refused is two names whose order disagrees with the rig's.

    Raises:
        ValueError: a name is not this rig's, the list repeats one, or the
            names are not in :data:`CAMERAS` order.
    """
    unknown = [name for name in cameras if name not in CAMERAS]
    if unknown:
        raise ValueError(f"not cameras of this rig: {', '.join(unknown)}")

    ids = tuple(CAMERAS.index(name) for name in cameras)
    if len(set(ids)) != len(ids):
        raise ValueError(f"a camera is named twice: {', '.join(cameras)}")
    if list(ids) != sorted(ids):
        raise ValueError(
            "cameras are not in rig order, so a view would be drawn from the "
            f"wrong seat: got {', '.join(cameras)}, which is rig positions "
            f"{', '.join(str(one) for one in ids)}."
        )
    return ids
