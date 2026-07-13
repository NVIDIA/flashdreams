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

``--scene`` accepts either a clipgt scene directory or a clipgt ``.usdz``
archive (e.g. ``~/.cache/flashdreams/omnidreams-scenes/clipgt-<uuid>.usdz``),
whose ``clipgt/*.parquet`` payload is extracted on demand. When ``--scene`` is
left at the default and the bundled sample is absent, a cached USDZ scene under
``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes`` is used automatically; only if none
is found does it fall back to a small synthetic scene (also forced via
``--synthetic``).

Outputs (in ``--out-dir``, default ``./_vk_compare``):
  cuda.png         CUDA backend render
  vulkan.png       Vulkan backend render
  diff_10x.png     |cuda - vulkan| * 10
  side_by_side.png CUDA | Vulkan | diff in one strip

Usage:
    uv run python examples/compare_vulkan_vs_cuda.py
    uv run python examples/compare_vulkan_vs_cuda.py --frame 30 \
        --width 1280 --height 720
    uv run python examples/compare_vulkan_vs_cuda.py \
        --scene ~/.cache/flashdreams/omnidreams-scenes/clipgt-<uuid>.usdz
    uv run python examples/compare_vulkan_vs_cuda.py --synthetic
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure we import the local ``ludus_renderer`` (this project) even when an
# editable install of a sibling project is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_SCENE_PATH = str(_PROJECT_ROOT / "example_data" / "test_hdmap")
DEFAULT_CAMERA = "camera:front:wide:120fov"

# Shared cache used by the omnidreams demos. Scene archives live at
# ``<root>/clipgt-<uuid>[-<variant>].usdz`` and bundle the HDMap vector data as
# ``clipgt/<element>.parquet`` (alongside large mesh/checkpoint payloads that
# rasterization does not need).
DEFAULT_SCENE_CACHE = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "omnidreams-scenes"
)


# ---------------------------------------------------------------------------
# Scene resolution: clipgt directory or USDZ archive
# ---------------------------------------------------------------------------


def extract_clipgt_from_usdz(usdz_path: Path) -> Path:
    """Extract the ``clipgt/*.parquet`` HDMap payload from a USDZ archive.

    A clipgt USDZ is a plain zip; ``load_clipgt_scene`` only needs the parquet
    files (not the bundled meshes/checkpoint). They are extracted (flattened)
    into a sibling ``<stem>.clipgt`` directory and reused on later runs.
    """
    usdz_path = Path(usdz_path)
    dest = usdz_path.parent / f"{usdz_path.stem}.clipgt"
    with zipfile.ZipFile(usdz_path) as zf:
        members = [
            m
            for m in zf.namelist()
            if m.startswith("clipgt/") and m.endswith(".parquet")
        ]
        if not members:
            raise SystemExit(
                f"{usdz_path} contains no clipgt/*.parquet payload; "
                "is this a clipgt scene archive?"
            )
        already = len(list(dest.glob("*.parquet"))) if dest.is_dir() else 0
        if already < len(members):
            dest.mkdir(parents=True, exist_ok=True)
            print(
                f"Extracting {len(members)} clipgt parquet files from "
                f"{usdz_path.name} -> {dest}"
            )
            for m in members:
                (dest / Path(m).name).write_bytes(zf.read(m))
    return dest


def discover_cached_usdz() -> Path | None:
    """Return a cached clipgt USDZ to render by default, preferring the base
    (non-weather) variant, or ``None`` when the scenes cache is empty."""
    if not DEFAULT_SCENE_CACHE.is_dir():
        return None
    candidates = sorted(DEFAULT_SCENE_CACHE.glob("clipgt-*.usdz"))
    if not candidates:
        return None
    # The base archive (``clipgt-<uuid>.usdz``) has no ``-rain``/``-snow``
    # suffix, so it is the shortest filename.
    return min(candidates, key=lambda p: len(p.name))


def resolve_clipgt_dir(scene_arg: str) -> Path | None:
    """Resolve ``--scene`` to a clipgt parquet directory.

    Accepts a clipgt directory or a ``.usdz`` archive (extracted on demand);
    returns ``None`` when no usable scene data is present.
    """
    scene_path = Path(scene_arg)
    if scene_path.suffix == ".usdz" and scene_path.is_file():
        return extract_clipgt_from_usdz(scene_path)
    if scene_path.is_dir():
        return scene_path
    return None


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
        [
            [5.0, -2.0, 0.0],
            [5.0, -1.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 2.0, 0.0],
        ],
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
        [[8.0, -1.5, 0.0], [12.0, -1.5, 0.0], [12.0, 1.5, 0.0], [8.0, 1.5, 0.0]],
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


def render_hdmap(
    ctx_cls,
    scene_path: str,
    frame: int,
    width: int,
    height: int,
    camera_name: str,
    msaa: int,
) -> np.ndarray:
    from ludus_renderer.render_utils import (
        create_camera,
        get_available_cameras,
        render_frame,
    )
    from ludus_renderer.render_utils import (
        load_scene_adapted as load_scene,
    )
    from ludus_renderer.util import resample_timestamps

    device = torch.device("cuda")
    scene = load_scene(
        scene_path,
        device,
        include_ego_obstacle=False,
        include_ego_trajectory=False,
        use_gpu_decoder=True,
    )

    # Fall back to an available camera if the requested one isn't in the scene
    # (camera names vary between captures), so the example still renders.
    available = get_available_cameras(scene)
    if camera_name not in available and available:
        print(
            f"    camera {camera_name!r} not in scene; using {available[0]!r} "
            f"(available: {available})"
        )
        camera_name = available[0]

    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    if frame >= len(timestamps):
        raise SystemExit(
            f"frame {frame} out of range (scene has {len(timestamps)} frames)"
        )

    ctx = ctx_cls(device=device)
    ctx.set_depth_scaling(True)
    if msaa > 0:
        ctx.set_msaa_samples(msaa)

    cam = create_camera(
        width,
        height,
        device,
        bev=False,
        bev_height=80.0,
        bev_fov=60.0,
        scene=scene,
        camera_name=camera_name,
    )
    ctx.upload_cameras([cam])
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    img = render_frame(
        ctx,
        scene,
        scene_id,
        timestamps,
        frame,
        width,
        height,
        device,
        bev_height=None,
        camera_name=camera_name,
    )
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
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE_PATH,
        help="clipgt scene directory or a clipgt .usdz archive "
        "(default: bundled sample, else a cached USDZ under "
        "$FLASHDREAMS_CACHE_DIR/omnidreams-scenes)",
    )
    parser.add_argument(
        "--frame", type=int, default=12, help="Frame index to render (default: 12)"
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Scene camera name")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--msaa",
        type=int,
        default=0,
        choices=[0, 4],
        help="MSAA sample count for the CUDA backend (0 or 4)",
    )
    parser.add_argument(
        "--out-dir",
        default="_vk_compare",
        help="Output directory (default: ./_vk_compare)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the small synthetic fallback scene instead "
        "of the clipgt HDMap scene.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from ludus_renderer import LudusCudaTimestampedContext

    try:
        from ludus_renderer import LudusTimestampedContext
    except ImportError as exc:
        print(f"Vulkan backend unavailable: {exc}", file=sys.stderr)
        return 1

    # Resolve a real scene: explicit clipgt dir / USDZ, else auto-discover a
    # cached USDZ, else fall back to the synthetic scene.
    scene_dir = None
    if not args.synthetic:
        scene_dir = resolve_clipgt_dir(args.scene)
        if scene_dir is None and args.scene == DEFAULT_SCENE_PATH:
            usdz = discover_cached_usdz()
            if usdz is not None:
                print(
                    f"default scene {args.scene!r} not found; using cached "
                    f"USDZ scene {usdz.name}"
                )
                scene_dir = extract_clipgt_from_usdz(usdz)

    use_synthetic = args.synthetic or scene_dir is None
    if not args.synthetic and scene_dir is None:
        print(
            f"no clipgt scene found for {args.scene!r} (and none cached under "
            f"{DEFAULT_SCENE_CACHE}); falling back to synthetic"
        )

    if use_synthetic:
        label = "synthetic"
        print(f"Rendering synthetic scene at {args.width}x{args.height}...")
        cuda_img = render_synthetic(
            LudusCudaTimestampedContext, args.width, args.height
        )
        print(f"  CUDA   lit pixels: {int((cuda_img[..., :3].sum(-1) > 0).sum())}")
        vk_img = render_synthetic(LudusTimestampedContext, args.width, args.height)
        print(f"  Vulkan lit pixels: {int((vk_img[..., :3].sum(-1) > 0).sum())}")
    else:
        label = f"hdmap frame {args.frame} ({args.camera})"
        print(f"Rendering {label} from {scene_dir} at {args.width}x{args.height}...")
        print("  CUDA backend...")
        cuda_img = render_hdmap(
            LudusCudaTimestampedContext,
            str(scene_dir),
            args.frame,
            args.width,
            args.height,
            args.camera,
            args.msaa,
        )
        print(f"    lit pixels: {int((cuda_img[..., :3].sum(-1) > 0).sum())}")
        print("  Vulkan backend...")
        vk_img = render_hdmap(
            LudusTimestampedContext,
            str(scene_dir),
            args.frame,
            args.width,
            args.height,
            args.camera,
            args.msaa,
        )
        print(f"    lit pixels: {int((vk_img[..., :3].sum(-1) > 0).sum())}")

    cuda_img, vk_img, diff_amp, side = composite(cuda_img, vk_img)

    Image.fromarray(cuda_img, mode="RGBA").save(out_dir / "cuda.png")
    Image.fromarray(vk_img, mode="RGBA").save(out_dir / "vulkan.png")
    Image.fromarray(diff_amp, mode="RGBA").save(out_dir / "diff_10x.png")
    Image.fromarray(side, mode="RGBA").save(out_dir / "side_by_side.png")

    rgb_diff = np.abs(
        cuda_img[..., :3].astype(np.int16) - vk_img[..., :3].astype(np.int16)
    )
    max_diff = int(rgb_diff.max())
    mean_diff = float(rgb_diff.mean())
    differing = int((rgb_diff.sum(-1) > 0).sum())
    total = cuda_img.shape[0] * cuda_img.shape[1]

    print(f"\nOutputs written to: {out_dir}")
    for name in ("cuda.png", "vulkan.png", "diff_10x.png", "side_by_side.png"):
        print(f"  {name:<18} {os.path.getsize(out_dir / name):>9} bytes")
    print("\nPixel comparison (RGB only):")
    print(f"  Max per-channel difference: {max_diff} / 255")
    print(f"  Mean per-channel difference: {mean_diff:.3f} / 255")
    print(
        f"  Differing pixels: {differing} / {total} ({100.0 * differing / total:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
