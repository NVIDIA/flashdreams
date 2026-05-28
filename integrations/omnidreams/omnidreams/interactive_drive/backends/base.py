# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod

from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig
from omnidreams.interactive_drive.types import FrameChunk, SceneBundle, TrajectoryChunk


class RenderBackend(ABC):
    def __init__(self, chunk: ChunkConfig, raster: RasterConfig) -> None:
        self._chunk = chunk
        self._raster = raster

    @property
    def fps(self) -> int:
        return self._chunk.fps

    @property
    def initial_chunk_frames(self) -> int:
        return self._chunk.initial_chunk_frames

    @property
    def chunk_frames(self) -> int:
        return self._chunk.chunk_frames

    @abstractmethod
    def warmup(self, scene: SceneBundle) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        raise NotImplementedError

    @abstractmethod
    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        raise NotImplementedError

    def close(self) -> None:
        return
