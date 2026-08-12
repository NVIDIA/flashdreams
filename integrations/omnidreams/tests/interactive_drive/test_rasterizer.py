# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import omnidreams.interactive_drive.rasterizer as rasterizer_module
import pytest
import torch
from ludus_renderer._ops import context as context_module
from ludus_renderer._ops.context import LudusCudaTimestampedContext
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.interactive_drive.rasterizer import (
    LudusConditionRasterizer,
    _LoadedSceneData,
    _LudusConditionRasterizerImpl,
    _RenderedCameraFrames,
)

pytestmark = pytest.mark.ci_cpu


class _Event:
    def __init__(self) -> None:
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


def test_rasterizer_keeps_adaptive_cube_tessellation_enabled(monkeypatch) -> None:
    """PhysX boxes need curved F-theta faces at the image boundary."""
    configured_cube_levels: list[int] = []

    class _Context:
        def __init__(self, *, device) -> None:
            self.device = device

        def set_depth_scaling(self, enabled: bool) -> None:
            pass

        def set_msaa_samples(self, samples: int) -> None:
            pass

        def set_max_tessellation_levels(self, *, cube: int) -> None:
            configured_cube_levels.append(cube)

        def set_line_widths(self, **kwargs) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rasterizer_module, "LudusCudaTimestampedContext", _Context)

    _LudusConditionRasterizerImpl(RasterConfig(), None)

    assert configured_cube_levels == [3]


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
    impl._physx_debug_scene = SimpleNamespace(update=lambda snapshots, timestamps: 8)

    def fake_render_one_camera(**kwargs):
        n_frames = len(kwargs["timestamps_us"])
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


def test_rgb_render_uploads_attached_actor_tracks_from_first_chunk() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    uploaded: list[tuple[object, ...]] = []
    impl._replace_dynamic_actor_scene = lambda actors: uploaded.append(actors)
    actor = SimpleNamespace(detached_from_track=False)

    impl.render_rgb_frames(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
        dynamic_actors=(actor,),  # type: ignore[arg-type]
    )

    assert uploaded == [(actor,)]


def test_single_camera_render_uses_uniform_scalar_dispatch() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    timestamps = np.array([11, 22, 33], dtype=np.int64)
    calls: list[dict[str, object]] = []

    class _Context:
        needs_vflip = False

        def render_uniform(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros((3, 2, 3, 4), dtype=torch.uint8)

        def render(self, *args, **kwargs):
            raise AssertionError("generic regrouping path should not be used")

    impl.ctx = _Context()
    _LudusConditionRasterizerImpl._render_one_camera(
        impl,
        rig_poses=torch.eye(4).repeat(3, 1, 1),
        timestamps_us=timestamps,
        scene_id=7,
        camera_id=2,
        sensor_to_rig=torch.eye(4),
        camera_type=1,
        resolution=(2, 3),
    )

    assert len(calls) == 1
    assert calls[0]["scene_id"] == 7
    assert calls[0]["camera_id"] == 2
    assert calls[0]["timestamps_us"] is timestamps
    assert calls[0]["camera_type_id"] == 1


def test_bev_stays_level_and_centered_when_ego_points_upward() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    del impl._render_one_camera
    impl._bev = BevConfig(enabled=True, width=3, height=2, height_m=75.0)
    impl._bev_camera_id = 1
    impl._bev_sensor_to_rig = rasterizer_module._bev_sensor_to_rig(
        height_m=impl._bev.height_m,
        tilt_deg=impl._bev.tilt_deg,
        device=torch.device("cpu"),
    )
    captured_camera_poses: list[torch.Tensor] = []

    class _Context:
        needs_vflip = False

        def render_uniform(self, **kwargs):
            captured_camera_poses.append(kwargs["camera_poses"])
            return torch.zeros((1, 2, 3, 4), dtype=torch.uint8)

    impl.ctx = _Context()
    yaw = 0.6
    pitch = torch.pi / 2
    cos_yaw, sin_yaw = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    cos_pitch = torch.cos(torch.tensor(pitch))
    sin_pitch = torch.sin(torch.tensor(pitch))
    rig_pose = torch.eye(4).unsqueeze(0)
    rig_pose[0, :3, :3] = torch.tensor(
        [
            [cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch],
            [sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch],
            [-sin_pitch, 0.0, cos_pitch],
        ]
    )
    rig_pose[0, :3, 3] = torch.tensor([12.0, -4.0, 3.0])

    impl.render_bev_frames(rig_poses_torch=rig_pose, timestamps_us=np.array([1]))

    camera_pose = captured_camera_poses[0][0]
    torch.testing.assert_close(
        camera_pose[:3, 0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(camera_pose[:3, 3], torch.tensor([12.0, -4.0, 78.0]))


def test_uniform_dispatch_reuses_expanded_camera_intrinsics(monkeypatch) -> None:
    context = LudusCudaTimestampedContext.__new__(LudusCudaTimestampedContext)
    context._camera_intrinsics = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    context._uniform_intrinsics_cache = {}
    context._cameras = [object()]
    context._scenes = [
        {
            "timestamps": torch.empty(0),
            "int32": torch.empty(0),
            "vertices": torch.empty(0),
            "triangles": torch.empty(0),
            "floats": torch.empty(0),
            "polyline_pools": torch.empty(0),
            "polygon_pools": torch.empty(0),
            "cube_pools": torch.empty(0),
            "cube_pool_counts": [],
            "total_cubes": 0,
            "max_varrays_per_ts_polyline": 0,
            "max_varrays_per_ts_polygon": 0,
        }
    ]
    context.cpp_wrapper = object()
    context._max_extrapolation_us = 1
    context._tessellation_threshold = 1.0
    intrinsic_buffers: list[torch.Tensor] = []
    cube_pool_counts: list[list[int]] = []

    class _Plugin:
        def ludus_render_fwd_cuda_timestamped(self, *args):
            cube_pool_counts.append(args[10])
            intrinsic_buffers.append(args[-4])
            return torch.zeros((4, 1, 1, 4), dtype=torch.uint8)

    monkeypatch.setattr(context_module, "_get_plugin", lambda: _Plugin())
    poses = torch.eye(4).repeat(4, 1, 1)
    for _ in range(2):
        context.render_uniform(
            scene_id=0,
            camera_id=0,
            timestamps_us=np.arange(4, dtype=np.int64),
            camera_type_id=0,
            camera_poses=poses,
            resolution=(1, 1),
        )

    assert len(context._uniform_intrinsics_cache) == 1
    assert cube_pool_counts == [[], []]
    assert intrinsic_buffers[0] is intrinsic_buffers[1]


def test_physx_debug_chunk_is_ludus_rendered_and_stays_lazy() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)

    chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
        physics_debug_frames=(object(), object()),  # type: ignore[arg-type]
    )

    debug = chunk.frames[0].physx_rgb_host_uint8
    assert callable(getattr(debug, "to_cuda_tensor", None))
    assert torch.equal(
        debug.to_cuda_tensor(), torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3)
    )


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


def test_lagged_bev_poll_does_not_wait_for_in_flight_render() -> None:
    rasterizer = LudusConditionRasterizer.__new__(LudusConditionRasterizer)
    pending: concurrent.futures.Future[_RenderedCameraFrames | None] = (
        concurrent.futures.Future()
    )
    latest = _RenderedCameraFrames(
        frames_hwc_uint8=torch.zeros((1, 1, 1, 3), dtype=torch.uint8),
        ready_event=None,
    )
    rasterizer._pending_bev = pending
    rasterizer._latest_bev = latest

    assert rasterizer._poll_ready_bev() is latest
    assert rasterizer._pending_bev is pending


def test_lagged_bev_poll_promotes_completed_render() -> None:
    rasterizer = LudusConditionRasterizer.__new__(LudusConditionRasterizer)
    pending: concurrent.futures.Future[_RenderedCameraFrames | None] = (
        concurrent.futures.Future()
    )
    rendered = _RenderedCameraFrames(
        frames_hwc_uint8=torch.ones((1, 1, 1, 3), dtype=torch.uint8),
        ready_event=None,
    )
    pending.set_result(rendered)
    rasterizer._pending_bev = pending
    rasterizer._latest_bev = None

    assert rasterizer._poll_ready_bev() is rendered
    assert rasterizer._pending_bev is None


def test_build_chunk_resamples_lagged_bev_with_different_frame_count() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    rgb_frames = _RenderedCameraFrames(
        frames_hwc_uint8=torch.zeros((7, 1, 1, 3), dtype=torch.uint8),
        ready_event=None,
    )
    bev_frames = _RenderedCameraFrames(
        frames_hwc_uint8=torch.arange(5, dtype=torch.uint8).reshape(5, 1, 1, 1),
        ready_event=None,
    )

    chunk = impl.build_chunk(
        timestamps_us=np.arange(7, dtype=np.int64),
        rgb_frames=rgb_frames,
        bev_frames=bev_frames,
    )

    bev_values = [
        int(frame.bev_host_uint8.to_cuda_tensor()[0, 0, 0]) for frame in chunk.frames
    ]
    assert bev_values == [0, 1, 1, 2, 3, 3, 4]
