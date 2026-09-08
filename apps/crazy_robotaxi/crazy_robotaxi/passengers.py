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

"""Pedestrian conditioning tracks for Crazy Robotaxi pickup targets."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.types import DynamicActorTrajectory

from crazy_robotaxi.rules import TaxiGameSnapshot

_PASSENGER_DIMENSIONS_LWH_M = np.array([0.6, 0.6, 1.8], dtype=np.float32)
"""Full dimensions of one pedestrian conditioning box in metres."""

_PASSENGER_CENTER_HEIGHT_M = 0.9
"""Height of a grounded passenger box center above its pickup target."""


def _target_key(target_xyz_m: tuple[float, float, float]) -> bytes:
    return struct.pack("<3f", *target_xyz_m)


def _passenger_track(
    target_xyz_m: tuple[float, float, float],
    timestamps_us: npt.NDArray[np.int64],
) -> DynamicActorTrajectory:
    target = np.asarray(target_xyz_m, dtype=np.float32)
    center = target + np.array([0.0, 0.0, _PASSENGER_CENTER_HEIGHT_M], dtype=np.float32)
    track_length = len(timestamps_us)
    coordinate_digest = hashlib.sha256(_target_key(target_xyz_m)).hexdigest()[:16]
    return DynamicActorTrajectory(
        entity_id=f"taxi-passenger-{coordinate_digest}",
        object_type="Pedestrian",
        timestamps_us=timestamps_us.copy(),
        translations_world=np.repeat(center[None, :], track_length, axis=0),
        orientations_xyzw=np.repeat(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            track_length,
            axis=0,
        ),
        dimensions_lwh=_PASSENGER_DIMENSIONS_LWH_M.copy(),
        is_simulated=True,
    )


def build_pickup_passenger_trajectories(
    snapshots: Sequence[TaxiGameSnapshot],
    timestamps_us: npt.NDArray[np.int64],
) -> tuple[DynamicActorTrajectory, ...]:
    """Build stationary pedestrian tracks for visible pickup targets.

    Full-chunk visibility uses one stationary track. Partial visibility uses
    one-sample tracks because Ludus extrapolates multi-sample object tracks
    beyond their endpoints.

    Args:
        snapshots: Taxi state synchronized to each generated frame.
        timestamps_us: Timestamps for the same frames.

    Returns:
        Contiguous passenger visibility tracks in first-visible order.

    Raises:
        ValueError: ``snapshots`` and ``timestamps_us`` have different lengths.
    """
    if len(snapshots) != len(timestamps_us):
        raise ValueError(
            "snapshots must match timestamps_us; got "
            f"{len(snapshots)} snapshots for {len(timestamps_us)} timestamps"
        )

    open_tracks: dict[bytes, tuple[int, tuple[float, float, float]]] = {}
    completed_tracks: list[tuple[int, int, tuple[float, float, float]]] = []
    for frame_index, snapshot in enumerate(snapshots):
        visible_targets = (
            (snapshot.pickup_passengers_xyz_m or snapshot.pickup_targets_xyz_m)
            if snapshot.session_state == "playing"
            else ()
        )
        visible_by_key = {
            _target_key(target_xyz_m): target_xyz_m for target_xyz_m in visible_targets
        }

        for key in open_tracks.keys() - visible_by_key.keys():
            start_index, target_xyz_m = open_tracks.pop(key)
            completed_tracks.append((start_index, frame_index, target_xyz_m))
        for key, target_xyz_m in visible_by_key.items():
            open_tracks.setdefault(key, (frame_index, target_xyz_m))

    for start_index, target_xyz_m in open_tracks.values():
        completed_tracks.append((start_index, len(snapshots), target_xyz_m))

    completed_tracks.sort(key=lambda track: (track[0], _target_key(track[2])))
    passenger_tracks: list[DynamicActorTrajectory] = []
    for start_index, end_index, target_xyz_m in completed_tracks:
        if start_index == 0 and end_index == len(snapshots):
            passenger_tracks.append(_passenger_track(target_xyz_m, timestamps_us))
            continue
        passenger_tracks.extend(
            _passenger_track(target_xyz_m, timestamps_us[index : index + 1])
            for index in range(start_index, end_index)
        )
    return tuple(passenger_tracks)
