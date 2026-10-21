# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for what a recorded drive says about itself.

What is being asserted here is mostly direction: that forward is where the
recording went rather than where a mount happens to point, and that the speed
taken over is the one the footage at that frame actually had. Both fail
quietly. A car driving into the road surface still produces a map to draw, and
the model renders whatever it is shown, so the result reads as the checkpoint
having a bad afternoon rather than as a basis being wrong.

The keyboard and the bicycle model these used to feed went with the batch
command's ``--drive``; steering is the Interactive Drive application's now, and
is tested against its own vehicle model.
"""

from __future__ import annotations

import numpy as np
import pytest
from cosmos3_nano_sim_auto.impl.conditioning.driving import (
    body_basis,
    recorded_speed,
)

pytestmark = pytest.mark.ci_cpu

FPS = 30.0


def straight(frames: int = 64, *, metres: float = 0.5) -> np.ndarray:
    """A track driving along the rig's own forward axis, which is ``+z``.

    Identity rotation, so a camera's axes are the world's and a direction that
    comes back out of the basis can be read directly.
    """
    track = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    track[:, 2, 3] = np.arange(frames, dtype=np.float32) * metres
    return track


def test_forward_is_where_the_recording_went_not_where_the_lens_points() -> None:
    """The basis comes off the track, so a pitched mount does not drive uphill.

    Here the drive went along ``+x`` while the lens looks along ``+z``. Taking
    the lens for the heading would drive the car ninety degrees away from the
    road it is being drawn on.
    """
    track = np.tile(np.eye(4, dtype=np.float32), (32, 1, 1))
    track[:, 0, 3] = np.arange(32, dtype=np.float32) * 0.4

    basis = body_basis(track)

    assert np.allclose(basis[:, 0], (1.0, 0.0, 0.0), atol=1e-6)
    assert np.isclose(np.linalg.det(basis), 1.0, atol=1e-6)


def test_a_clip_recorded_at_a_standstill_falls_back_to_the_lens() -> None:
    """No motion names no heading, and jitter is worse than the mount's guess."""
    basis = body_basis(np.tile(np.eye(4, dtype=np.float32), (8, 1, 1)))

    assert np.allclose(basis[:, 0], (0.0, 0.0, 1.0), atol=1e-6)


def test_the_speed_read_off_a_clip_is_the_one_it_was_going() -> None:
    """A standstill under moving opening footage would stop the car dead."""
    assert recorded_speed(straight(metres=0.4), FPS) == pytest.approx(12.0, rel=1e-3)


def test_the_speed_taken_over_is_the_one_at_that_frame() -> None:
    """A run mirroring another clip takes over partway in.

    Measuring at frame 0 there starts the car at a speed the footage under it
    never had, and a clip stopped at a light at one end can be doing 50 at the
    other.
    """
    track = np.tile(np.eye(4, dtype=np.float32), (80, 1, 1))
    metres = np.where(np.arange(79) < 40, 0.1, 0.4).astype(np.float32)
    track[1:, 2, 3] = np.cumsum(metres)

    assert recorded_speed(track, FPS, first=0) == pytest.approx(3.0, rel=1e-3)
    assert recorded_speed(track, FPS, first=45) == pytest.approx(12.0, rel=1e-3)


def test_taking_over_at_the_last_pose_still_names_a_speed() -> None:
    """Nothing steps out of it, so the step into it is the measurement."""
    track = straight(metres=0.4)

    speed = recorded_speed(track, FPS, first=len(track) - 1)

    assert speed == pytest.approx(12.0, rel=1e-3)


def test_a_frame_outside_the_track_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the track's 64 poses"):
        recorded_speed(straight(), FPS, first=64)


def test_a_track_too_short_to_name_a_direction_is_refused() -> None:
    one_pose = np.eye(4, dtype=np.float32)[None]

    with pytest.raises(ValueError, match=r"\[T>=2, 4, 4\]"):
        body_basis(one_pose)
    with pytest.raises(ValueError, match=r"\[T>=2, 4, 4\]"):
        recorded_speed(one_pose, FPS)
