#!/usr/bin/env python3
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

"""Render the same HDMap frame with both the CUDA and Vulkan backends and
save the outputs side-by-side for visual comparison.

Defaults to ``example_data/test_hdmap`` (the bundled clipgt sample) but
will also fall back to a small synthetic scene with ``--synthetic`` so
this script runs even when no scene data is present.

Outputs (in ``--out-dir``, default ``./_vk_compare``):
  cuda.png         CUDA backend render
  vulkan.png       Vulkan backend render
  diff_10x.png     |cuda - vulkan| * 10
  side_by_side.png CUDA | Vulkan | diff in one strip

Usage:
    uv run python examples/compare_vulkan_vs_cuda.py
    uv run python examples/compare_vulkan_vs_cuda.py --frame 30 \
        --width 1280 --height 720
    uv run python examples/compare_vulkan_vs_cuda.py --synthetic
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Ensure we import the local ``ludus_renderer`` (this project) even when an
# editable install of a sibling project is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

DEFAULT_SCENE_PATH = str(_PROJECT_ROOT / "example_data" / "test_hdmap")
DEFAULT_CAMERA = "camera:front:wide:120fov"


# ---------------------------------------------------------------------------
# Synthetic fallback scene (no external data required)
# ---------------------------------------------------------------------------

def build_synthetic_scene():
    from ludus_renderer import (
        PRIM_CROSSWALK,
        PRIM_OBSTACLE,
        PRIM_ROAD_BOUNDARY,
        CubePool,
        TimestampedPolygonPool,
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
    polyline = TimestampedPolylinePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64),
        timestamped_varrays_prefix_sum=torch.tensor([1], dtype=torch.int32),
        varrays_prefix_sum=torch.tensor([polyline_pts.shape[0]], dtype=torch.int32),
        vertices=polyline_pts,
        prim_type_id=PRIM_ROAD_BOUNDARY,
    )

    polygon_pts = torch.tensor(
        [[8.0, -1.5, 0.0],
         [12.0, -1.5, 0.0],
         [12.0,  1.5, 0.0],
         [8.0,   1.5, 0.0]],
        dtype=torch.float32,
    )
    polygon_tris = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    polygon = TimestampedPolygonPool(
        timestamps_us=torch.tensor([0], dtype=torch.int64),
        timestamped_varrays_prefix_sum=torch.tensor([1], dtype=torch.int32),
        varrays_prefix_sum=torch.tensor([polygon_pts.shape[0]], dtype=torch.int32),
        triangle_prefix_sum=torch.tensor([polygon_tris.shape[0]], dtype=torch.int32),
        vertices=polygon_pts,
        triangles=polygon_tris,
        prim_type_id=PRIM_CROSSWALK,
    )

    cube = CubePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64),
        cube_ts_prefix_sum=torch.tensor([1], dtype=torch.int32),
        track_timestamps_us=torch.tensor([0], dtype=torch.int64),
        translations=torch.tensor([[7.0, -2.5, 0.5]], dtype=torch.float32),
        quaternions=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        scales=torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32),
        colors=torch.tensor([[1.0, 0.4, 0.2, 0.8, 0.2, 0.2]], dtype=torch.float32),
        prim_type_id=PRIM_OBSTACLE,
    )

    return TimestampedScene(
        polyline_pools=[polyline],
        polygon_pools=[polygon],
        cube_pools=[cube],
    )


def render_synthetic(ctx_cls, width: int, height: int) -> np.ndarray:
    from ludus_renderer import FThetaCamera
    cam = FThetaCamera(
        principal_point=torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32),
        image_size=torch.tensor([float(width), float(height)], dtype=torch.float32),
        fw_poly=torch.tensor([0.0, 200.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        max_ray_angle=math.pi / 2.0,
        depth_max=200.0,
    )
    ctx = ctx_cls(device="cuda")
    ctx.upload_cameras([cam])
    scene_id = ctx.upload_scene(build_synthetic_scene())
    poses = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0)
    img = ctx.render_batch([(scene_id, 0, 0, 0)], poses, (height, width))
    return img.detach().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# HDMap scene rendering (mirrors examples/render_hdmap_scene.py)
# ---------------------------------------------------------------------------

def render_hdmap(ctx_cls, scene_path: str, frame: int, width: int, height: int,
                 camera_name: str, msaa: int) -> np.ndarray:
    from ludus_renderer.render_utils import (
        create_camera,
        load_scene_adapted as load_scene,
        render_frame,
    )
    from ludus_renderer.util import resample_timestamps

    device = torch.device("cuda")
    scene = load_scene(scene_path, device, include_ego_obstacle=False,
                       include_ego_trajectory=False, use_gpu_decoder=True)
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    if frame >= len(timestamps):
        raise SystemExit(
            f"frame {frame} out of range (scene has {len(timestamps)} frames)")

    ctx = ctx_cls(device=device)
    ctx.set_depth_scaling(True)
    if msaa > 0:
        ctx.set_msaa_samples(msaa)

    cam = create_camera(width, height, device, bev=False, bev_height=80.0,
                        bev_fov=60.0, scene=scene, camera_name=camera_name)
    ctx.upload_cameras([cam])
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    img = render_frame(ctx, scene, scene_id, timestamps, frame,
                       width, height, device, bev_height=None,
                       camera_name=camera_name)
    return np.array(img)  # PIL.Image -> (H, W, 3 or 4) uint8


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def _to_rgba(img: np.ndarray) -> np.ndarray:
    """Coerce to (H, W, 4) uint8."""
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=-1)
    if img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img.astype(np.uint8)


def composite(cuda_img: np.ndarray, vk_img: np.ndarray):
    cuda_img = _to_rgba(cuda_img)
    vk_img = _to_rgba(vk_img)
    diff = np.abs(cuda_img.astype(np.int16) - vk_img.astype(np.int16))
    diff_amp = np.clip(diff * 10, 0, 255).astype(np.uint8)
    diff_amp[..., 3] = 255

    h = cuda_img.shape[0]
    sep_w = 4
    sep = np.zeros((h, sep_w, 4), dtype=np.uint8)
    sep[..., 3] = 255
    side = np.concatenate([cuda_img, sep, vk_img, sep, diff_amp], axis=1)
    return cuda_img, vk_img, diff_amp, side


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE_PATH,
                        help="Path to a clipgt scene directory")
    parser.add_argument("--frame", type=int, default=12,
                        help="Frame index to render (default: 12)")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="Scene camera name")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--msaa", type=int, default=0, choices=[0, 4],
                        help="MSAA sample count for the CUDA backend (0 or 4)")
    parser.add_argument("--out-dir", default="_vk_compare",
                        help="Output directory (default: ./_vk_compare)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use the small synthetic fallback scene instead "
                             "of the clipgt HDMap scene.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from ludus_renderer import LudusCudaTimestampedContext
    try:
        from ludus_renderer import LudusTimestampedContext
    except ImportError as exc:
        print(f"Vulkan backend unavailable: {exc}", file=sys.stderr)
        return 1

    use_synthetic = args.synthetic or not Path(args.scene).is_dir()
    if not args.synthetic and not Path(args.scene).is_dir():
        print(f"scene path {args.scene!r} not found; falling back to synthetic")

    if use_synthetic:
        label = "synthetic"
        print(f"Rendering synthetic scene at {args.width}x{args.height}...")
        cuda_img = render_synthetic(LudusCudaTimestampedContext, args.width, args.height)
        print(f"  CUDA   lit pixels: {int((cuda_img[..., :3].sum(-1) > 0).sum())}")
        vk_img = render_synthetic(LudusTimestampedContext, args.width, args.height)
        print(f"  Vulkan lit pixels: {int((vk_img[..., :3].sum(-1) > 0).sum())}")
    else:
        label = f"hdmap frame {args.frame} ({args.camera})"
        print(f"Rendering {label} at {args.width}x{args.height}...")
        print(f"  CUDA backend...")
        cuda_img = render_hdmap(LudusCudaTimestampedContext, args.scene,
                                args.frame, args.width, args.height,
                                args.camera, args.msaa)
        print(f"    lit pixels: {int((cuda_img[..., :3].sum(-1) > 0).sum())}")
        print(f"  Vulkan backend...")
        vk_img = render_hdmap(LudusTimestampedContext, args.scene,
                              args.frame, args.width, args.height,
                              args.camera, args.msaa)
        print(f"    lit pixels: {int((vk_img[..., :3].sum(-1) > 0).sum())}")

    cuda_img, vk_img, diff_amp, side = composite(cuda_img, vk_img)

    Image.fromarray(cuda_img, mode="RGBA").save(out_dir / "cuda.png")
    Image.fromarray(vk_img,   mode="RGBA").save(out_dir / "vulkan.png")
    Image.fromarray(diff_amp, mode="RGBA").save(out_dir / "diff_10x.png")
    Image.fromarray(side,     mode="RGBA").save(out_dir / "side_by_side.png")

    max_diff = int(np.abs(cuda_img.astype(np.int16) - vk_img.astype(np.int16))[..., :3].max())
    mean_diff = float(np.abs(cuda_img.astype(np.int16) - vk_img.astype(np.int16))[..., :3].mean())
    differing = int((np.abs(cuda_img.astype(np.int16) - vk_img.astype(np.int16))[..., :3].sum(-1) > 0).sum())
    total = cuda_img.shape[0] * cuda_img.shape[1]

    print(f"\nOutputs written to: {out_dir}")
    for name in ("cuda.png", "vulkan.png", "diff_10x.png", "side_by_side.png"):
        print(f"  {name:<18} {os.path.getsize(out_dir / name):>9} bytes")
    print(f"\nPixel comparison (RGB only):")
    print(f"  Max per-channel difference: {max_diff} / 255")
    print(f"  Mean per-channel difference: {mean_diff:.3f} / 255")
    print(f"  Differing pixels: {differing} / {total} ({100.0 * differing / total:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
