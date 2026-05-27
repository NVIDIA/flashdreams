# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from interactive_drive.backends.base import RenderBackend
from interactive_drive.config import BevConfig, ChunkConfig, RasterConfig
from interactive_drive.rasterizer import LudusConditionRasterizer
from interactive_drive.types import FrameChunk, SceneBundle, TrajectoryChunk


class RasterRenderBackend(RenderBackend):
    def __init__(
        self,
        chunk: ChunkConfig,
        raster: RasterConfig,
        bev: BevConfig | None = None,
    ) -> None:
        super().__init__(chunk=chunk, raster=raster)
        self._rasterizer = LudusConditionRasterizer(raster, bev=bev)
        self._scene: SceneBundle | None = None

    def warmup(self, scene: SceneBundle) -> None:
        self._scene = scene
        self._rasterizer.load_scene(scene)

    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render_chunk(trajectory)

    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render_chunk(trajectory)

    def _render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        raster_chunk = self._rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
        )
        return FrameChunk(
            frames=raster_chunk.frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="raster",
        )

    def close(self) -> None:
        self._rasterizer.cleanup()
