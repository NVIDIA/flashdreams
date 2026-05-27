# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from interactive_drive.backends.base import RenderBackend
from interactive_drive.types import FrameChunk, SceneBundle, TrajectoryChunk


class LocalVideoModelAdapter:
    """Adapter for existing in-process render backends.

    Keeps the local path zero encode/decode and returns FrameChunk directly.
    Implements :class:`~interactive_drive.video_model.chunk_pipeline.VideoModelBackend`:
    construction is cheap; ``warmup`` does the actual model/scene loading and
    is called by :class:`ChunkPipeline` on its worker thread.
    """

    def __init__(self, backend: RenderBackend) -> None:
        self._backend = backend
        self._is_first_chunk = True

    def warmup(self, scene: SceneBundle) -> None:
        self._backend.warmup(scene)

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        if self._is_first_chunk:
            self._is_first_chunk = False
            return self._backend.render_first_chunk(trajectory)
        return self._backend.render_next_chunk(trajectory)

    def reset(self) -> None:
        self._is_first_chunk = True
