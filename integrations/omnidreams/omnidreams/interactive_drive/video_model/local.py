# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.types import FrameChunk, SceneBundle, TrajectoryChunk


class LocalVideoModelAdapter:
    """Adapter for existing in-process render backends.

    Keeps the local path zero encode/decode and returns FrameChunk directly.
    Implements :class:`~omnidreams.interactive_drive.video_model.chunk_pipeline.VideoModelBackend`:
    construction is cheap; ``warmup_model`` loads the model once and
    ``load_scene`` binds each scene, both called by :class:`ChunkPipeline`
    on its worker thread.
    """

    def __init__(self, backend: RenderBackend) -> None:
        self._backend = backend
        self._is_first_chunk = True

    @property
    def can_prewarm(self) -> bool:
        return self._backend.can_prewarm

    def warmup_model(self) -> None:
        self._backend.warmup_model()

    def load_scene(self, scene: SceneBundle) -> None:
        # Scene/variant switches must not carry over rollout cache, text/image
        # embeddings, or first-chunk state from the previous selection.
        self._backend.reset_scene_conditioning()
        self._backend.load_scene(scene)
        # A scene (re)load restarts the rollout: the next chunk must be a
        # first chunk so the world-model session re-initialises its cache
        # from the new scene's initial frame and prompt.
        self._is_first_chunk = True

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        if self._is_first_chunk:
            self._is_first_chunk = False
            return self._backend.render_first_chunk(trajectory)
        return self._backend.render_next_chunk(trajectory)

    def reset(self) -> None:
        self._backend.reset()
        self._is_first_chunk = True
