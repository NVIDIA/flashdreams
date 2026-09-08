# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ludus CUDA scene construction for the first-person PhysX debug view."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from ludus_renderer import CUBE_FLAG_WIREFRAME, CubePool, TimestampedScene

from interactive_drive.types import PhysicsDebugFrame, PresentedFrame

_ACTOR_FRONT_RGB = (1.0, 190.0 / 255.0, 58.0 / 255.0)
_ACTOR_BACK_RGB = (0.64, 0.28, 0.04)
_BARRIER_FRONT_RGB = (0.0, 225.0 / 255.0, 1.0)
_BARRIER_BACK_RGB = (0.0, 0.34, 0.48)
_HIDDEN_TRANSLATION = (1_000_000.0, 1_000_000.0, -1_000_000.0)


class PhysxDebugSceneContext(Protocol):
    """Ludus scene operations used by the persistent debug buffer."""

    def upload_scene(self, scene: TimestampedScene) -> int: ...

    def update_cube_pool(
        self, scene_id: int, prim_type_id: int, pool: CubePool
    ) -> bool: ...

    def replace_scene(self, scene_id: int, scene: TimestampedScene) -> int: ...


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _stable_collider_ids(
    values: tuple[str, ...], count: int, prefix: str
) -> tuple[str, ...]:
    result = values or tuple(f"{prefix}-{index}" for index in range(count))
    if len(result) != count:
        raise ValueError(f"{prefix}_ids must match the collider count")
    if len(set(result)) != count:
        raise ValueError(f"{prefix}_ids must be unique within each frame")
    return result


def build_physx_debug_cube_pool(
    snapshots: Sequence[PhysicsDebugFrame],
    timestamps_us: np.ndarray,
    *,
    device: torch.device,
    capacity: int | None = None,
) -> CubePool:
    """Pack exact per-frame collider snapshots into one timestamped CUDA pool.

    Each visible collider owns one track whose non-owning timestamps are placed
    far outside the renderable world. This preserves exact per-frame topology
    even when PhysX activates or deactivates objects between adjacent frames.
    Capacity is rounded up and may be supplied by the persistent scene buffer
    so normal count fluctuations do not reallocate renderer storage.
    """
    timestamps = np.ascontiguousarray(timestamps_us, dtype=np.int64)
    if len(snapshots) != len(timestamps):
        raise ValueError("PhysX debug snapshots must match the timestamp count")
    if not len(timestamps):
        raise ValueError("PhysX debug rendering requires at least one timestamp")

    actor_ids_by_frame = [
        _stable_collider_ids(
            snapshot.actor_ids, len(snapshot.actor_positions_m), "actor"
        )
        for snapshot in snapshots
    ]
    barrier_ids_by_frame = [
        _stable_collider_ids(
            snapshot.barrier_ids, len(snapshot.barrier_segments_xy_m), "barrier"
        )
        for snapshot in snapshots
    ]
    track_indices: dict[tuple[str, str], int] = {}
    for actor_ids, barrier_ids in zip(
        actor_ids_by_frame, barrier_ids_by_frame, strict=True
    ):
        for collider_type, collider_ids in (
            ("actor", actor_ids),
            ("barrier", barrier_ids),
        ):
            for collider_id in collider_ids:
                track_indices.setdefault(
                    (collider_type, collider_id), len(track_indices)
                )
    required = len(track_indices)
    pool_capacity = max(1, int(capacity or _next_power_of_two(max(1, required))))
    if pool_capacity < required:
        raise ValueError(
            f"PhysX debug pool capacity {pool_capacity} is smaller than {required}"
        )

    frame_count = len(timestamps)
    translations = np.full(
        (pool_capacity, frame_count, 3),
        _HIDDEN_TRANSLATION,
        dtype=np.float32,
    )
    quaternions = np.zeros((pool_capacity, frame_count, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    scales = np.full((pool_capacity, 3), 1e-3, dtype=np.float32)
    colors = np.zeros((pool_capacity, 6), dtype=np.float32)

    for frame_index, snapshot in enumerate(snapshots):
        actor_ids = actor_ids_by_frame[frame_index]
        actor_count = len(snapshot.actor_positions_m)
        if actor_count:
            actor_indices = np.fromiter(
                (track_indices[("actor", actor_id)] for actor_id in actor_ids),
                dtype=np.intp,
                count=actor_count,
            )
            translations[actor_indices, frame_index] = np.asarray(
                snapshot.actor_positions_m, dtype=np.float32
            )
            quaternions[actor_indices, frame_index] = np.asarray(
                snapshot.actor_orientations_xyzw, dtype=np.float32
            )
            scales[actor_indices] = np.asarray(
                snapshot.actor_dimensions_lwh, dtype=np.float32
            )
            colors[actor_indices, :3] = _ACTOR_FRONT_RGB
            colors[actor_indices, 3:] = _ACTOR_BACK_RGB

        segments = np.asarray(snapshot.barrier_segments_xy_m, dtype=np.float32)
        barrier_count = len(segments)
        if not barrier_count:
            continue
        delta = segments[:, 1] - segments[:, 0]
        lengths = np.linalg.norm(delta, axis=1)
        valid = lengths > 1e-6
        if not bool(np.any(valid)):
            continue
        segments = segments[valid]
        delta = delta[valid]
        lengths = lengths[valid]
        barrier_ids = tuple(
            barrier_id
            for barrier_id, keep in zip(
                barrier_ids_by_frame[frame_index], valid, strict=True
            )
            if bool(keep)
        )
        thicknesses = np.asarray(snapshot.barrier_thicknesses_m, dtype=np.float32)[
            valid
        ]
        heights = np.asarray(snapshot.barrier_heights_m, dtype=np.float32)[valid]
        barrier_count = len(segments)
        barrier_indices = np.fromiter(
            (track_indices[("barrier", barrier_id)] for barrier_id in barrier_ids),
            dtype=np.intp,
            count=barrier_count,
        )
        centers = (segments[:, 0] + segments[:, 1]) * 0.5
        ground_z = float(
            snapshot.ego_position_m[2] - snapshot.ego_dimensions_lwh[2] * 0.5
        )
        translations[barrier_indices, frame_index, :2] = centers
        translations[barrier_indices, frame_index, 2] = ground_z + heights * 0.5
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        quaternions[barrier_indices, frame_index, 2] = np.sin(yaw * 0.5)
        quaternions[barrier_indices, frame_index, 3] = np.cos(yaw * 0.5)
        scales[barrier_indices] = np.column_stack((lengths, thicknesses, heights))
        colors[barrier_indices, :3] = _BARRIER_FRONT_RGB
        colors[barrier_indices, 3:] = _BARRIER_BACK_RGB

    # Each collider owns one stable track across the whole chunk. Missing
    # frames remain at the hidden sentinel instead of allocating a new track
    # for every collider/frame pair.
    track_lengths = torch.full(
        (pool_capacity,), frame_count, dtype=torch.int32, device=device
    )
    timestamps_tensor = torch.as_tensor(timestamps, dtype=torch.int64, device=device)
    track_timestamps = timestamps_tensor.repeat(pool_capacity)
    return CubePool(
        timestamps_us=timestamps_tensor,
        cube_ts_prefix_sum=torch.cumsum(track_lengths, dim=0),
        track_timestamps_us=track_timestamps,
        translations=torch.as_tensor(
            translations.reshape(-1, 3), dtype=torch.float32, device=device
        ),
        quaternions=torch.as_tensor(
            quaternions.reshape(-1, 4), dtype=torch.float32, device=device
        ),
        scales=torch.as_tensor(scales, dtype=torch.float32, device=device),
        colors=torch.as_tensor(colors, dtype=torch.float32, device=device),
        render_flags=CUBE_FLAG_WIREFRAME,
    )


class LudusPhysxDebugSceneBuffer:
    """Persist and update the CUDA scene slot used by the PhysX debug view."""

    def __init__(
        self, context: PhysxDebugSceneContext, *, device: torch.device
    ) -> None:
        self._context = context
        self._device = device
        self._scene_id: int | None = None
        self._capacity = 0
        self._frame_count = 0

    def reset(self) -> None:
        """Forget the scene slot after the parent context clears its scenes."""
        self._scene_id = None
        self._capacity = 0
        self._frame_count = 0

    def update(
        self,
        snapshots: Sequence[PhysicsDebugFrame],
        timestamps_us: np.ndarray,
    ) -> int:
        """Update collider tracks and return the stable Ludus scene ID."""
        collider_keys = {
            (collider_type, collider_id)
            for snapshot in snapshots
            for collider_type, collider_ids in (
                (
                    "actor",
                    _stable_collider_ids(
                        snapshot.actor_ids,
                        len(snapshot.actor_positions_m),
                        "actor",
                    ),
                ),
                (
                    "barrier",
                    _stable_collider_ids(
                        snapshot.barrier_ids,
                        len(snapshot.barrier_segments_xy_m),
                        "barrier",
                    ),
                ),
            )
            for collider_id in collider_ids
        }
        required = len(collider_keys)
        capacity = max(self._capacity, _next_power_of_two(max(1, required)))
        pool = build_physx_debug_cube_pool(
            snapshots,
            timestamps_us,
            device=self._device,
            capacity=capacity,
        )
        scene = TimestampedScene(polyline_pools=[], polygon_pools=[], cube_pools=[pool])
        frame_count = len(timestamps_us)
        if self._scene_id is None:
            self._scene_id = self._context.upload_scene(scene)
        elif (
            capacity != self._capacity
            or frame_count != self._frame_count
            or not self._context.update_cube_pool(
                self._scene_id, pool.prim_type_id, pool
            )
        ):
            self._scene_id = self._context.replace_scene(self._scene_id, scene)
        self._capacity = capacity
        self._frame_count = frame_count
        return self._scene_id


def select_presented_rgb(
    frame: PresentedFrame, view_mode: str, *, width: int, height: int
) -> object:
    """Select the already-rendered lazy CUDA source for a presenter view."""
    del width, height
    if view_mode == "physx":
        if frame.physx_debug is None:
            raise RuntimeError("PhysX view requires a physics-debug snapshot")
        if frame.physx_rgb_host_uint8 is None:
            raise RuntimeError("PhysX view requires a Ludus-rendered lazy debug frame")
        return frame.physx_rgb_host_uint8
    if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
        return frame.model_rgb_host_uint8
    return frame.rgb_host_uint8


__all__ = [
    "LudusPhysxDebugSceneBuffer",
    "build_physx_debug_cube_pool",
    "select_presented_rgb",
]
