# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recorded drive's geometry, and the contract for steering away from it.

``ludus_controls`` draws a chunk at whatever poses it is handed, which is what
makes a steered run possible at all: the recorded track is one source of poses
and a host application's vehicle model is another. What lives here is the part
that belongs to neither of them, being read off the recording itself: which way
it was actually going and how fast it was going there, together with the one
method a pose source has to answer.

The vehicle model that consumes this is the Interactive Drive application's. A
bicycle model and a keyboard reader lived here while the batch command was also
a way to drive, and went when it stopped being one.

Which way is forward is read off the clip rather than assumed. Poses are
recorded in the rig's own camera axes -- right, down, forward -- so a camera's
optical axis is roughly where the car was pointing, but the mount carries pitch
and yaw of its own, and driving along a mount's idea of forward would climb or
sink away from the road over a minute of it. The recorded drive knows better:
where the car actually went, expressed in those axes, is the forward the map
was surveyed around.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "PoseSource",
    "body_basis",
    "recorded_speed",
]


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` at unit length, or unchanged if it has none to speak of."""
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else vector


def body_basis(track: np.ndarray, *, still: float = 1e-3) -> np.ndarray:
    """``[3, 3]`` whose columns are forward, left and up in a camera's axes.

    Forward is where the recorded drive actually went, averaged over the frames
    it was moving, which takes the mount's pitch and yaw out without having to
    know what they were. Up starts as the rig's own -- its axes are right, down
    and forward, so up is negative y -- and is then squared against forward,
    because a car climbing a hill has the two at slightly less than a right
    angle and a basis has to be a basis.

    Args:
        track: ``[T, 4, 4]`` camera-to-world, as the rig recorded it.
        still: metres a frame below which the car counts as stopped, and the
            step's direction as noise rather than heading.

    Raises:
        ValueError: fewer than two poses, which names no direction at all.
    """
    if track.ndim != 3 or track.shape[0] < 2 or track.shape[1:] != (4, 4):
        raise ValueError(f"expected [T>=2, 4, 4] poses, got {tuple(track.shape)}.")
    steps = np.diff(track[:, :3, 3].astype(np.float64), axis=0)
    turned = track[:-1, :3, :3].astype(np.float64).transpose(0, 2, 1)
    local = np.einsum("tij,tj->ti", turned, steps)
    moving = local[np.linalg.norm(local, axis=1) > still]
    # A clip recorded at a red light names no forward, so the mount's own is
    # all there is. Rare, and better than a basis built out of jitter.
    forward = _unit(moving.mean(axis=0)) if len(moving) else np.array([0.0, 0.0, 1.0])
    rigs_up = np.array([0.0, -1.0, 0.0])
    up = _unit(rigs_up - forward * float(np.dot(rigs_up, forward)))
    return np.stack([forward, np.cross(up, forward), up], axis=1)


def recorded_speed(
    track: np.ndarray, fps: float, *, first: int = 0, over: int = 30
) -> float:
    """How fast the car was going where the drive is taken over, in m/s.

    A steered run starts at the recording's speed rather than at rest. The
    opening frames are real footage of a moving car, and a standstill under
    them would ask the model to stop dead in one chunk.

    Args:
        track: ``[T>=2, 4, 4]`` camera-to-world, as recorded.
        fps: frames a second the track was recorded at.
        first: the frame the drive takes over at, which is where the measuring
            starts rather than the track's own beginning. A run mirroring
            another clip takes over partway in, and a clip can be stopped at a
            light at one end and doing 50 at the other. Taking over at the last
            pose measures the step into it, there being none after it.
        over: frames to average across, a single step being mostly noise.
    """
    if track.ndim != 3 or track.shape[0] < 2:
        raise ValueError(f"expected [T>=2, 4, 4] poses, got {tuple(track.shape)}.")
    if not 0 <= first < len(track):
        raise ValueError(f"frame {first} is outside the track's {len(track)} poses.")
    start = min(first, len(track) - 2)
    steps = np.diff(track[start : start + over + 1, :3, 3].astype(np.float64), axis=0)
    return float(np.linalg.norm(steps, axis=1).mean() * fps)


@runtime_checkable
class PoseSource(Protocol):
    """Where the anchor camera's poses come from, a span at a time.

    A host application owning a vehicle model hands over the poses it has
    already computed, which is the one thing
    :class:`~cosmos3_nano_sim_auto.impl.conditioning.ludus_controls.LudusControls` needs: it asks for
    this rather than for a driver, so a clip left without one stays on the
    track it recorded.
    """

    def poses(self, first: int, last: int) -> np.ndarray:
        """``[T, 4, 4]`` camera-to-world for frames ``[first, last)``.

        Asked for in order and once each, because whoever answers has state
        that moves with the car and no way back to an earlier frame.
        """
        ...
