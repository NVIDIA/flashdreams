# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The host application's path, in the shape the rollout pulls it.

Interactive Drive works out where the car goes: it reads one command a chunk,
integrates its own vehicle model across the whole chunk, and hands the backend
the rig's poses. Our rollout instead pulls poses for a frame span whenever it
needs a chunk's map drawn. So this holds what the app has handed over and
answers the pull, which is the only thing
:class:`~cosmos3_nano_sim_auto.impl.conditioning.driving.PoseSource` asks of anyone.

The two also disagree about a camera's own axes. The app inverts a
sensor-to-world pose and gives ludus that, where
:func:`~cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera.camera_pose` inverts and then turns the axes
round, because the rig records them the other way up from the renderer.
:func:`as_recorded` is that difference, applied once, so a pose that goes
through our render path lands exactly where the app's own rasterizer puts it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera import (
    RDF_TO_FLU,
)

__all__ = [
    "SuppliedPoses",
    "as_recorded",
    "rig_tracks",
]


def as_recorded(sensor_to_world: np.ndarray) -> np.ndarray:
    """A host application's camera pose as the rig would have recorded it.

    ``camera_pose`` is applied to everything our renderer draws, and it carries
    an axis change the host does not. Undoing that here rather than branching
    inside the render keeps one pose convention through the drawing, and keeps
    the correction in the one place a reader can check it.

    Args:
        sensor_to_world: a ``[4, 4]`` pose, or a ``[T, 4, 4]`` stack of them, as
            the host holds it.
    """
    pose = np.asarray(sensor_to_world, dtype=np.float32)
    if pose.shape[-2:] != (4, 4) or pose.ndim not in (2, 3):
        raise ValueError(
            f"expected [4, 4] or [T, 4, 4] poses, got {tuple(pose.shape)}."
        )
    return pose @ RDF_TO_FLU


class SuppliedPoses:
    """Poses handed over a chunk at a time, read back a span at a time.

    The two parties are not in step, and cannot be. The rollout renders its
    whole control window while it is being built, which is before the host has
    sampled any trajectory at all, and afterwards it draws a chunk's map at the
    moment it generates that chunk, by which time the host is a window ahead.
    So a span can fall either side of what the host has supplied.

    Past the supply, the path carries straight on at the speed it last had,
    which is what the demo's driver does before the first key is pressed. The
    opening is therefore not steerable: by the time the host's first command
    arrives, the map for those frames is already drawn. Frames the host supplies
    for ground already drawn are dropped rather than queued, since the picture
    conditioned on the extrapolation is the one the viewer has seen.
    """

    def __init__(
        self,
        anchor_sensor_to_rig: np.ndarray,
        *,
        opening_rig_to_world: np.ndarray,
        speed_mps: float,
        seconds_per_frame: float,
        delay_frames: int = 0,
        track_positions: bool = False,
    ) -> None:
        """
        Args:
            anchor_sensor_to_rig: where the rig's first camera sits on the car.
                The rig's poses are the car's, and ludus draws the whole rig
                from the first camera's, so this is the one conversion between
                them.
            opening_rig_to_world: where the car starts, in the host's own
                convention, to carry on from until the host has driven.
            speed_mps: how fast it is going when the drive opens.
            seconds_per_frame: the host's frame rate, for how far that gets per
                frame of extrapolation.
            delay_frames: draw the host's frame ``h`` as frame ``h + delay``, so
                a span is known that many frames before the host reaches it,
                which is what drawing the next chunk during this one needs.
                The first ``delay`` frames carry straight on from the opening,
                and the host's path is moved forward along the opening heading
                by as far as they go, so it carries on from them without a
                jump: a rigid copy of the host's path, ``delay`` frames late.
            track_positions: retain drawn XY positions until presentation consumes
                them, so coverage warnings follow the conditioning's path.
        """
        self._anchor = as_recorded(np.asarray(anchor_sensor_to_rig, dtype=np.float32))
        self._latest = np.asarray(opening_rig_to_world, dtype=np.float32)
        if self._latest.shape != (4, 4):
            raise ValueError(
                f"expected a [4, 4] opening pose, got {tuple(self._latest.shape)}."
            )
        self._per_frame = float(speed_mps) * float(seconds_per_frame)
        self._held: list[np.ndarray] = []
        self._drawn_positions: list[np.ndarray] | None = [] if track_positions else None
        self._presented = 0
        self._read = 0
        self._given = 0
        self._shift = np.eye(4, dtype=np.float32)
        if delay_frames:
            opening = _carried_on(self._latest, delay_frames, self._per_frame)
            self._held.append(opening)
            self._given = delay_frames
            self._shift = opening[-1] @ np.linalg.inv(self._latest)

    @property
    def next_frame(self) -> int:
        """The frame the next span has to start at."""
        return self._read

    @property
    def supplied(self) -> int:
        """Frames whose pose is known, read, dropped or waiting: the host's and any delay's."""
        return self._given

    def supply(self, rig_poses_world: np.ndarray) -> None:
        """Take one chunk of the host's rig poses.

        Args:
            rig_poses_world: ``[T, 4, 4]`` rig-to-world, as the host's
                trajectory carries them.
        """
        poses = np.asarray(rig_poses_world, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"expected [T, 4, 4] rig poses, got {tuple(poses.shape)}.")
        if not len(poses):
            raise ValueError("a chunk of poses cannot be empty.")
        poses = self._shift @ poses
        opening = self._given
        self._given += len(poses)
        unread = poses[max(self._read - opening, 0) :]
        if len(unread):
            self._held.append(unread)

    def poses(self, first: int, last: int) -> np.ndarray:
        """``[T, 4, 4]`` camera-to-world for frames ``[first, last)``.

        Raises:
            ValueError: the span is empty or does not carry on from the last
                one.
        """
        if first < 0 or first >= last:
            raise ValueError(f"[{first}, {last}) is not a frame span.")
        if first != self._read:
            raise ValueError(
                f"Poses must carry on from {self._read}, got a span starting at "
                f"{first}. The host supplies its path in order and this keeps no "
                "frame it has already drawn."
            )
        rig = self._rig_span(last - first)
        self._read = last
        self._latest = rig[-1]
        if self._drawn_positions is not None:
            self._drawn_positions.append(rig[:, :2, 3].copy())
        return rig @ self._anchor

    def take_position(self, frames: int) -> np.ndarray:
        """Consume presented poses and return the last one's world XY position."""
        if (
            self._drawn_positions is None
            or not 0 < frames <= self._read - self._presented
        ):
            raise ValueError("Presented positions require enough tracked, drawn frames")
        self._presented += frames
        return _taken(self._drawn_positions, frames)[-1]

    def _rig_span(self, frames: int) -> np.ndarray:
        """``frames`` rig poses, from the host as far as it goes and then on."""
        waiting = max(self._given - self._read, 0)
        span: list[np.ndarray] = []
        if waiting:
            span.append(_taken(self._held, min(waiting, frames)))
            frames -= len(span[0])
        if frames:
            span.append(
                _carried_on(
                    span[-1][-1] if span else self._latest, frames, self._per_frame
                )
            )
        return np.concatenate(span) if len(span) > 1 else span[0]


def _taken(held: list[np.ndarray], frames: int) -> np.ndarray:
    """Pull ``frames`` off the front of the held chunks, dropping what is spent.

    Chunks are consumed whole where they can be, since the host's chunk and the
    span asked for are usually the same frames; a span that stops mid-chunk
    leaves the remainder at the front for the next one.
    """
    taken: list[np.ndarray] = []
    wanted = frames
    while wanted:
        chunk = held[0]
        if len(chunk) <= wanted:
            taken.append(held.pop(0))
            wanted -= len(chunk)
        else:
            taken.append(chunk[:wanted])
            held[0] = chunk[wanted:]
            wanted = 0
    return np.concatenate(taken) if len(taken) > 1 else taken[0]


def _carried_on(latest: np.ndarray, frames: int, per_frame: float) -> np.ndarray:
    """``frames`` poses carrying on from ``latest`` in the direction it faces.

    Straight, because a heading is all a pose states and guessing a turn rate
    from one would be inventing the very thing the host decides. Column zero is
    the car's forward axis, the same one the scene's opening yaw is read off.
    """
    carried = np.repeat(latest[None], frames, axis=0)
    ahead = latest[:3, 0] * per_frame
    carried[:, :3, 3] += ahead * np.arange(1, frames + 1, dtype=np.float32)[:, None]
    return carried


def rig_tracks(sensor_to_rig: Sequence[np.ndarray]) -> list[np.ndarray]:
    """A one-frame track per camera, in rig space, for ``DrawnCamera.track``.

    ``LudusControls`` reads the rig's shape off its cameras' tracks: where each
    one sits relative to the first, taken at the frame the run opens on. A host
    application states that directly as calibration, so a track of one frame
    says everything a recorded one would and nothing it cannot know.
    """
    if not sensor_to_rig:
        raise ValueError("a rig needs at least one camera.")
    return [
        as_recorded(np.asarray(one, dtype=np.float32))[None] for one in sensor_to_rig
    ]
