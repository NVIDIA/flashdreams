# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omnidreams.interactive_drive.rasterizer import (
    _LoadedSceneData,
    _LudusConditionRasterizerImpl,
    _RenderedCameraFrames,
)


class _Event:
    def __init__(self) -> None:
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


def _impl_for_render_chunk(*, use_cuda_frames: bool) -> _LudusConditionRasterizerImpl:
    impl = _LudusConditionRasterizerImpl.__new__(_LudusConditionRasterizerImpl)
    impl._scene_data = _LoadedSceneData(clipgt_scene=object(), scene_adapter=object())
    impl._scene_id = 7
    impl._selected_camera_name = "front"
    impl._all_camera_map = {"front": 0}
    impl._sensor_to_rig = {"front": torch.eye(4)}
    impl._bev = None
    impl._bev_camera_id = None
    impl._bev_sensor_to_rig = None
    impl._temp_dir = None
    impl._device = torch.device("cpu")
    impl._raster = SimpleNamespace(height=2, width=3)
    impl._use_cuda_frames = use_cuda_frames
    impl._to_ludus_camera_pose = lambda poses: poses

    def fake_render_one_camera(**kwargs):
        n_frames = int(kwargs["timestamps_batch"].shape[0])
        frames = torch.arange(n_frames * 2 * 3 * 3, dtype=torch.uint8).reshape(
            n_frames, 2, 3, 3
        )
        return _RenderedCameraFrames(frames_hwc_uint8=frames, ready_event=_Event())

    impl._render_one_camera = fake_render_one_camera
    return impl


def test_raster_chunk_uses_cuda_backed_frames_by_default() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)

    chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
    )

    assert callable(getattr(chunk.frames[0].rgb_host_uint8, "to_cuda_tensor", None))


def test_raster_chunk_can_disable_cuda_backed_frames() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=False)

    chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
    )

    first = chunk.frames[0].rgb_host_uint8
    assert isinstance(first, np.ndarray)
    assert not callable(getattr(first, "to_cuda_tensor", None))
    assert np.array_equal(first, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
