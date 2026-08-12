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

"""User-spawned dynamic actors for the Omnidreams WebRTC drive.

The model's control branch was trained to materialize objects at rendered
HDMap bboxes, so "add an object mid-drive" is expressed as a wireframe cube
in the Ludus conditioning stream: spawn a box, the model paints an object
there (the prompt names its appearance). Actors follow a constant-velocity
world-frame motion model — enough for parked obstacles and lead vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ludus_renderer import CubePool
from omnidreams.grpc.utils import dynamic_state_to_ludus_cube_pool
from scipy.spatial.transform import Rotation

## Spawn presets

RIG_HEIGHT_M = 1.5
"""Ego rig-origin height above the road plane; spawn z-correction."""

ACTOR_PRESETS: dict[str, tuple[str, tuple[float, float, float]]] = {
    # preset -> (actor class, FLU bbox size (length, width, height) in meters)
    "car": ("CAR", (4.6, 2.0, 1.6)),
    "truck": ("TRUCK", (8.0, 2.6, 3.2)),
    "pedestrian": ("PEDESTRIAN", (0.6, 0.6, 1.8)),
    "cyclist": ("CYCLIST", (1.8, 0.7, 1.7)),
    "cone": ("OTHER", (0.4, 0.4, 0.8)),
    # True-scale cones render too faintly (below the model's salience
    # threshold for "Other" boxes) — oversized variants for obstacles.
    "cone_big": ("OTHER", (1.0, 1.0, 1.2)),
    "barrier": ("OTHER", (2.4, 0.6, 1.0)),
}


@dataclass
class SpawnedActor:
    """One user-spawned actor with a constant-velocity world trajectory."""

    class_id: str
    """Actor class (drives the obstacle color), e.g. ``"CAR"``."""

    size_xyz: tuple[float, float, float]
    """FLU bbox dimensions in meters."""

    spawn_timestamp_us: int
    """First frame timestamp at which the actor exists."""

    translation: np.ndarray
    """``[3]`` world-frame bbox-center position at spawn time."""

    quat_xyzw: np.ndarray
    """``[4]`` world-frame orientation quaternion."""

    velocity: np.ndarray
    """``[3]`` world-frame velocity in m/s (zeros = parked)."""

    def translation_at(self, timestamp_us: int) -> np.ndarray:
        dt_s = (timestamp_us - self.spawn_timestamp_us) * 1e-6
        return self.translation + self.velocity * dt_s


def spawn_actor_ahead(
    *,
    preset: str,
    ego_pose: np.ndarray,
    spawn_timestamp_us: int,
    distance_m: float = 12.0,
    speed_mps: float = 0.0,
    lateral_m: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> SpawnedActor:
    """Place a preset actor relative to the ego vehicle.

    Args:
        preset: Key into :data:`ACTOR_PRESETS`.
        ego_pose: ``[4, 4]`` world-from-ego FLU pose (x forward, y left,
            z up) to spawn relative to.
        spawn_timestamp_us: Timestamp of the first frame the actor exists.
        distance_m: Meters ahead of the ego along its heading.
        speed_mps: Actor speed along the ego heading (0 = parked).
        lateral_m: Meters to the left (+) / right (-) of the ego heading.
        yaw_offset_deg: Box heading relative to the ego heading (0 = same
            direction, 180 = oncoming). The rendered box's front/back face
            colors encode heading, which the model reads as travel
            direction.

    Raises:
        KeyError: Unknown preset.
    """
    class_id, size_xyz = ACTOR_PRESETS[preset]

    ego_pose = np.asarray(ego_pose, dtype=np.float64)
    rotation = ego_pose[:3, :3]
    # Ground-plane heading: project the ego forward axis onto XY so tilted
    # camera poses don't pitch the spawned box into the road.
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    forward_xy = np.array([forward[0], forward[1], 0.0])
    norm = float(np.linalg.norm(forward_xy))
    if norm < 1e-6:
        forward_xy = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    forward_xy /= norm
    left_xy = np.array([-forward_xy[1], forward_xy[0], 0.0])

    center = (
        ego_pose[:3, 3]
        + distance_m * forward_xy
        + lateral_m * left_xy
        # Bbox center sits half a height above the road. The ego pose is the
        # RIG origin (~camera height above ground, empirically ~1.5 m on the
        # HDMap scenes — verified against the scene's own actor boxes);
        # without the correction spawned boxes float at eye level and the
        # model under-renders them.
        + np.array([0.0, 0.0, size_xyz[2] / 2.0 - RIG_HEIGHT_M])
    )
    yaw = float(np.arctan2(forward_xy[1], forward_xy[0])) + float(
        np.deg2rad(yaw_offset_deg)
    )
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat().astype(np.float32)

    return SpawnedActor(
        class_id=class_id,
        size_xyz=size_xyz,
        spawn_timestamp_us=int(spawn_timestamp_us),
        translation=center.astype(np.float32),
        quat_xyzw=quat_xyzw,
        velocity=(speed_mps * forward_xy).astype(np.float32),
    )


def actors_to_cube_pool(
    actors: list[SpawnedActor],
    frame_timestamps_us: list[int],
    device: torch.device | str,
) -> CubePool | None:
    """Sample the actors at the chunk's frame timestamps as a Ludus pool.

    Reuses the gRPC ``DynamicWorldState`` conversion path (colors,
    interpolation, category mapping) by building the equivalent actor dicts
    with one exact pose per frame timestamp. Actors spawned mid-chunk simply
    have no poses for the earlier frames.
    """
    actor_dicts: list[dict] = []
    for actor in actors:
        poses = []
        for ts in frame_timestamps_us:
            ts = int(ts)
            if ts < actor.spawn_timestamp_us:
                continue
            x, y, z = (float(v) for v in actor.translation_at(ts))
            qx, qy, qz, qw = (float(v) for v in actor.quat_xyzw)
            poses.append(
                {
                    "timestamp_us": ts,
                    "pose": {
                        "vec": {"x": x, "y": y, "z": z},
                        "quat": {"x": qx, "y": qy, "z": qz, "w": qw},
                    },
                }
            )
        if not poses:
            continue
        size_x, size_y, size_z = actor.size_xyz
        actor_dicts.append(
            {
                "class_id": actor.class_id,
                "bbox_dims": {"size_x": size_x, "size_y": size_y, "size_z": size_z},
                "trajectory": {"poses": poses},
            }
        )
    if not actor_dicts:
        return None
    return dynamic_state_to_ludus_cube_pool(
        {"actors": actor_dicts}, frame_timestamps_us, device
    )
