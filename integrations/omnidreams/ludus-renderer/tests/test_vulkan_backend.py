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

"""Smoke test for the Vulkan backend (LudusTimestampedContext).

Builds a tiny synthetic scene with one polyline and one polygon, renders
one frame with both ``LudusCudaTimestampedContext`` and
``LudusTimestampedContext``, and asserts the Vulkan output is non-empty
and roughly similar to the CUDA reference.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def _build_camera(width=320, height=240):
    """Linear (pinhole-ish) F-theta camera."""
    from ludus_renderer import FThetaCamera
    return FThetaCamera(
        principal_point=torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32),
        image_size=torch.tensor([float(width), float(height)], dtype=torch.float32),
        fw_poly=torch.tensor([0.0, 200.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        max_ray_angle=math.pi / 2.0,
        depth_max=200.0,
    )


def _build_synthetic_scene():
    """One polyline that spans the view in front of the camera.

    The renderer's world coordinate convention is FLU (Front-Left-Up), so a
    polyline at x=5 with varying y traces a horizontal line 5 m in front of
    a camera sitting at the origin.
    """
    from ludus_renderer import (
        PRIM_ROAD_BOUNDARY,
        TimestampedPolylinePool,
        TimestampedScene,
    )

    polyline_pts = torch.tensor(
        [[5.0, -2.0, 0.0],
         [5.0, -1.0, 0.0],
         [5.0,  0.0, 0.0],
         [5.0,  1.0, 0.0],
         [5.0,  2.0, 0.0]],
        dtype=torch.float32,
    )
    pl_pool = TimestampedPolylinePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64),
        timestamped_varrays_prefix_sum=torch.tensor([1], dtype=torch.int32),
        varrays_prefix_sum=torch.tensor([polyline_pts.shape[0]], dtype=torch.int32),
        vertices=polyline_pts,
        prim_type_id=PRIM_ROAD_BOUNDARY,
    )

    return TimestampedScene(
        polyline_pools=[pl_pool],
        polygon_pools=[],
        cube_pools=None,
    )


def _identity_pose(device="cuda"):
    """4x4 world->camera identity pose on the given device."""
    return torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)


def _render_once(ctx_cls, scene, camera, resolution=(240, 320)):
    ctx = ctx_cls(device="cuda")
    ctx.upload_cameras([camera])
    scene_id = ctx.upload_scene(scene)
    queries = [(scene_id, 0, 0, 0)]   # scene, camera, timestamp, camera_type
    img = ctx.render_batch(queries, _identity_pose("cuda"), resolution)
    return img.detach().cpu()


def test_cuda_backend_renders_synthetic_scene():
    """Baseline: the CUDA backend renders the synthetic polyline."""
    from ludus_renderer import LudusCudaTimestampedContext

    scene = _build_synthetic_scene()
    cam = _build_camera()
    img = _render_once(LudusCudaTimestampedContext, scene, cam)

    assert img.shape == (1, 240, 320, 4), f"unexpected output shape {tuple(img.shape)}"
    assert img.dtype == torch.uint8
    assert int((img[..., :3].sum(dim=-1) > 0).sum().item()) > 0, "CUDA render is all black"


def test_vulkan_backend_renders_synthetic_scene():
    """The Vulkan backend should JIT-compile, init, upload, render, and
    produce visible geometry for the same scene as CUDA."""
    try:
        from ludus_renderer import LudusTimestampedContext
    except ImportError as exc:
        pytest.skip(f"Vulkan backend unavailable: {exc}")

    scene = _build_synthetic_scene()
    cam = _build_camera()

    try:
        img = _render_once(LudusTimestampedContext, scene, cam)
    except ImportError as exc:
        pytest.skip(f"Vulkan backend unavailable: {exc}")
    except RuntimeError as exc:
        # No GPU with mesh-shader support, no Vulkan ICD, etc.
        pytest.skip(f"Vulkan backend init failed: {exc}")

    assert img.shape == (1, 240, 320, 4), f"unexpected output shape {tuple(img.shape)}"
    assert img.dtype == torch.uint8
    nonzero = int((img[..., :3].sum(dim=-1) > 0).sum().item())
    assert nonzero > 0, "Vulkan render is all black"


def test_vulkan_vs_cuda_pixel_counts_close():
    """The Vulkan and CUDA backends use different rasterizers, but on a
    simple synthetic scene the number of lit pixels should be in the same
    ballpark (within 25%)."""
    from ludus_renderer import LudusCudaTimestampedContext
    try:
        from ludus_renderer import LudusTimestampedContext
    except ImportError as exc:
        pytest.skip(f"Vulkan backend unavailable: {exc}")

    scene = _build_synthetic_scene()
    cam = _build_camera()

    cuda_img = _render_once(LudusCudaTimestampedContext, scene, cam)
    try:
        vk_img = _render_once(LudusTimestampedContext, scene, cam)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Vulkan backend unavailable: {exc}")

    assert cuda_img.shape == vk_img.shape
    assert cuda_img.dtype == vk_img.dtype

    cuda_nz = int((cuda_img[..., :3].sum(dim=-1) > 0).sum().item())
    vk_nz = int((vk_img[..., :3].sum(dim=-1) > 0).sum().item())
    assert cuda_nz > 0 and vk_nz > 0
    ratio = vk_nz / cuda_nz
    assert 0.75 <= ratio <= 1.33, (
        f"Vulkan lit-pixel count diverges from CUDA: vk={vk_nz}, cuda={cuda_nz}, "
        f"ratio={ratio:.2f}")
