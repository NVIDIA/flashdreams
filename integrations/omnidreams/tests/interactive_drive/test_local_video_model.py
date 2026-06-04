# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import numpy as np
from omnidreams.interactive_drive._pipeline_fakes import make_trajectory, minimal_scene
from omnidreams.interactive_drive.types import FrameChunk, PresentedFrame, SceneBundle
from omnidreams.interactive_drive.video_model.local import LocalVideoModelAdapter


class _Backend:
    def __init__(self) -> None:
        self.load_scene_calls = 0
        self.reset_calls = 0
        self.reset_scene_conditioning_calls = 0
        self.first_chunk_calls = 0
        self.next_chunk_calls = 0

    @property
    def can_prewarm(self) -> bool:
        return True

    def warmup_model(self) -> None:
        return

    def load_scene(self, scene: SceneBundle) -> None:
        del scene
        self.load_scene_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1

    def reset_scene_conditioning(self) -> None:
        self.reset_scene_conditioning_calls += 1

    def render_first_chunk(self, trajectory: object) -> FrameChunk:
        self.first_chunk_calls += 1
        return _frame_chunk(trajectory)

    def render_next_chunk(self, trajectory: object) -> FrameChunk:
        self.next_chunk_calls += 1
        return _frame_chunk(trajectory)


def _frame_chunk(trajectory: object) -> FrameChunk:
    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
    )
    return FrameChunk(
        frames=(frame,),
        boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
        source_name="fake",
    )


def test_scene_load_resets_scene_conditioning_before_binding_scene() -> None:
    backend = _Backend()
    adapter = LocalVideoModelAdapter(backend)

    adapter.load_scene(minimal_scene())

    assert backend.reset_scene_conditioning_calls == 1
    assert backend.load_scene_calls == 1
    assert backend.reset_calls == 0


def test_scene_load_and_manual_reset_restart_first_chunk() -> None:
    backend = _Backend()
    adapter = LocalVideoModelAdapter(backend)
    trajectory = make_trajectory(1)

    adapter.load_scene(minimal_scene())
    adapter.render_chunk(trajectory)
    adapter.render_chunk(trajectory)
    adapter.reset()
    adapter.render_chunk(trajectory)
    adapter.load_scene(minimal_scene())
    adapter.render_chunk(trajectory)

    assert backend.first_chunk_calls == 3
    assert backend.next_chunk_calls == 1
    assert backend.reset_calls == 1
    assert backend.reset_scene_conditioning_calls == 2
