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

"""CUDA HD-map box construction for simulated object-graph trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from ludus_renderer._ops.primitives import (
    CUBE_FLAG_WIREFRAME,
    PRIM_OBSTACLE,
    CubePool,
    TimestampedScene,
)

from omnidreams_game_engine.colors import BBOX_V3_COLORS


def _get_obstacle_color(
    object_type: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the canonical dark/light semantic colors for an actor type."""
    return BBOX_V3_COLORS.get(object_type, BBOX_V3_COLORS["Others"])


class ObjectTrajectory(Protocol):
    """Structural input accepted from simulation clients."""

    entity_id: str
    object_type: str
    timestamps_us: Any
    translations_world: Any
    orientations_xyzw: Any
    dimensions_lwh: Any
    is_simulated: bool


class MutableSceneContext(Protocol):
    """Rendering operations required by :class:`MutableObjectSceneBuffer`."""

    def update_cube_pool(
        self, scene_id: int, prim_type_id: int, pool: CubePool
    ) -> bool: ...

    def update_cube_pool_at_index(
        self, scene_id: int, pool_index: int, pool: CubePool
    ) -> bool: ...

    def replace_scene(self, scene_id: int, scene: TimestampedScene) -> int: ...


class MutableObjectSceneBuffer:
    """Own dynamic-object topology and reuse policy for one CUDA scene slot."""

    def __init__(
        self,
        context: MutableSceneContext,
        scene_id: int,
        base_scene: TimestampedScene,
        *,
        device: torch.device,
    ) -> None:
        self._context = context
        self._scene_id = scene_id
        self._base_scene = base_scene
        self._device = device
        self._initialized = False
        self._actor_partition: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._dynamic_pool_index: int | None = None

    @property
    def scene_id(self) -> int:
        """Return the stable renderer scene slot."""
        return self._scene_id

    def update(self, actors: Sequence[ObjectTrajectory]) -> None:
        """Update simulated tracks while retaining immutable tracks on CUDA."""
        if not actors:
            if self._initialized:
                static_pools = [
                    pool
                    for pool in (self._base_scene.cube_pools or [])
                    if pool.prim_type_id != PRIM_OBSTACLE
                ]
                replacement = TimestampedScene(
                    polyline_pools=self._base_scene.polyline_pools,
                    polygon_pools=self._base_scene.polygon_pools,
                    cube_pools=static_pools,
                )
                self._scene_id = self._context.replace_scene(
                    self._scene_id, replacement
                )
                self._initialized = False
                self._actor_partition = None
                self._dynamic_pool_index = None
            return

        static_actors = tuple(
            actor for actor in actors if not getattr(actor, "is_simulated", False)
        )
        dynamic_actors = tuple(
            actor for actor in actors if getattr(actor, "is_simulated", False)
        )
        partition = (
            tuple(actor.entity_id for actor in static_actors),
            tuple(actor.entity_id for actor in dynamic_actors),
        )
        if (
            self._initialized
            and partition == self._actor_partition
            and self._dynamic_pool_index is None
        ):
            return
        if (
            self._initialized
            and partition == self._actor_partition
            and self._dynamic_pool_index is not None
        ):
            dynamic_pool = build_hdmap_object_pool(dynamic_actors, device=self._device)
            if self._context.update_cube_pool_at_index(
                self._scene_id, self._dynamic_pool_index, dynamic_pool
            ):
                return

        static_pools = [
            pool
            for pool in (self._base_scene.cube_pools or [])
            if pool.prim_type_id != PRIM_OBSTACLE
        ]
        actor_pools = []
        if static_actors:
            actor_pools.append(
                build_hdmap_object_pool(static_actors, device=self._device)
            )
        dynamic_pool_index = None
        if dynamic_actors:
            dynamic_pool_index = len(static_pools) + len(actor_pools)
            actor_pools.append(
                build_hdmap_object_pool(dynamic_actors, device=self._device)
            )
        replacement = TimestampedScene(
            polyline_pools=self._base_scene.polyline_pools,
            polygon_pools=self._base_scene.polygon_pools,
            cube_pools=[*static_pools, *actor_pools],
        )
        self._scene_id = self._context.replace_scene(self._scene_id, replacement)
        self._initialized = True
        self._actor_partition = partition
        self._dynamic_pool_index = dynamic_pool_index


def build_hdmap_object_pool(
    actors: Sequence[ObjectTrajectory], *, device: torch.device
) -> CubePool:
    """Build the CUDA box pool consumed by RGB and BEV model inputs.

    Simulation clients normally provide NumPy arrays. Stage those arrays in
    contiguous batches so an update performs one host-to-device transfer per
    field instead of several small transfers per object.
    """
    if not actors:
        raise ValueError("actors must not be empty")

    def _host_array(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    track_lengths_host = np.fromiter(
        (len(actor.timestamps_us) for actor in actors),
        dtype=np.int32,
        count=len(actors),
    )
    track_timestamps_host = np.concatenate(
        [_host_array(actor.timestamps_us, np.dtype(np.int64)) for actor in actors]
    )
    translations_host = np.concatenate(
        [
            _host_array(actor.translations_world, np.dtype(np.float32))
            for actor in actors
        ]
    )
    quaternions_host = np.concatenate(
        [_host_array(actor.orientations_xyzw, np.dtype(np.float32)) for actor in actors]
    )
    scales_host = np.stack(
        [_host_array(actor.dimensions_lwh, np.dtype(np.float32)) for actor in actors]
    )
    colors_host = np.asarray(
        [
            np.asarray(_get_obstacle_color(actor.object_type)).reshape(-1)
            for actor in actors
        ],
        dtype=np.float32,
    )

    track_lengths = torch.as_tensor(track_lengths_host, device=device)
    track_timestamps = torch.as_tensor(track_timestamps_host, device=device)
    translations = torch.as_tensor(translations_host, device=device)
    quaternions = torch.as_tensor(quaternions_host, device=device)
    scales = torch.as_tensor(scales_host, device=device)
    colors = torch.as_tensor(colors_host, device=device)
    return CubePool(
        timestamps_us=torch.unique(track_timestamps).sort()[0],
        cube_ts_prefix_sum=torch.cumsum(track_lengths, dim=0, dtype=torch.int32),
        track_timestamps_us=track_timestamps,
        translations=translations,
        quaternions=quaternions,
        scales=scales,
        colors=colors,
        prim_type_id=PRIM_OBSTACLE,
        render_flags=CUBE_FLAG_WIREFRAME,
    )


__all__ = [
    "MutableObjectSceneBuffer",
    "MutableSceneContext",
    "ObjectTrajectory",
    "build_hdmap_object_pool",
]
