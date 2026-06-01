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

    @property
    def can_prewarm(self) -> bool:
        """Whether :meth:`warmup_model` does its heavy build without a scene.

        ``True`` lets the demo start loading the model immediately at
        launch, overlapping warmup with the scene-selection wait. ``False``
        means the build is deferred until the first :meth:`load_scene`
        (e.g. the world model under ``--offload-text-encoder``, which must
        precompute per-scene embeddings and free the one-shot encoders
        before allocating the diffusion pipeline to keep peak VRAM low).
        """
        return True

    @abstractmethod
    def warmup_model(self) -> None:
        """Load/compile the scene-independent model. Called once per process."""
        raise NotImplementedError

    @abstractmethod
    def load_scene(self, scene: SceneBundle) -> None:
        """Bind one scene's geometry / conditioning. Called once per scene."""
        raise NotImplementedError

    def warmup(self, scene: SceneBundle) -> None:
        """Load the model and a single scene in one call.

        Convenience for callers that never switch scenes (the bare
        ``--no-hud`` path and unit tests). The scene-switching pipeline
        instead calls :meth:`warmup_model` once and :meth:`load_scene` per
        scene so the model is not rebuilt on each scene change.
        """
        self.warmup_model()
        self.load_scene(scene)

    @abstractmethod
    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        raise NotImplementedError

    @abstractmethod
    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        raise NotImplementedError

    def close(self) -> None:
        return
