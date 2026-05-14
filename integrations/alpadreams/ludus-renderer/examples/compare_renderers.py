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
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare wm_render and ludus_renderer outputs.

Renders the same scene with both renderers at specified FPS, computes diffs,
and optionally creates side-by-side comparison videos.

Setup:
    # Install wm-render for comparison (one-time)
    uv sync --group compare

Usage:
    # Run full comparison with both renderers
    uv run --group compare python examples/compare_renderers.py \
        --scene example_data/0300edb0-9310-4829-89f0-66743cbb8fa5 \
        --fps 10 --video

    # Render at native frame rate (all frames, no subsampling)
    uv run --group compare python examples/compare_renderers.py \
        --scene example_data/0300edb0-9310-4829-89f0-66743cbb8fa5 \
        --native --video

    # Skip rendering, just create videos from existing images
    uv run --group compare python examples/compare_renderers.py \
        --scene example_data/0300edb0-9310-4829-89f0-66743cbb8fa5 \
        --fps 10 --skip-wm --skip-ludus --skip-diff --video

Output structure:
    _images/comparison_<fps>hz/
        wm_render/          # wm_render output images per camera
        ludus_render/       # ludus_renderer output images per camera
        diffs/              # Amplified diff images (5x) per camera
        videos/             # Side-by-side comparison videos per camera
        metadata.txt        # Frame selection and timing info
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


def get_native_timestamps(scene_path: str) -> List[int]:
    """Read native timestamps from clipgt egomotion parquet."""
    import pandas as pd
    
    clipgt_path = Path(scene_path)
    
    # Find clip_id
    for f in clipgt_path.glob("*.egomotion_estimate.parquet"):
        clip_id = f.stem.replace(".egomotion_estimate", "")
        break
    else:
        raise ValueError(f"No egomotion parquet found in {scene_path}")
    
    ego_path = clipgt_path / f"{clip_id}.egomotion_estimate.parquet"
    df = pd.read_parquet(ego_path)
    
    timestamps = []
    for _, row in df.iterrows():
        key = row.get("key", {})
        if isinstance(key, dict) and "timestamp_micros" in key:
            timestamps.append(key["timestamp_micros"])
    
    timestamps.sort()
    return timestamps


def subsample_to_fps(timestamps: List[int], target_fps: float) -> List[int]:
    """Subsample timestamps to achieve target FPS."""
    if len(timestamps) < 2:
        return list(range(len(timestamps)))
    
    target_interval_us = 1_000_000 / target_fps
    
    selected_indices = [0]
    last_ts = timestamps[0]
    
    for i, ts in enumerate(timestamps[1:], 1):
        if ts - last_ts >= target_interval_us * 0.9:  # Allow 10% tolerance
            selected_indices.append(i)
            last_ts = ts
    
    return selected_indices


def render_wm(scene_path: str, output_dir: Path, frame_indices: List[int],
              width: int, height: int, cameras: Optional[List[str]] = None):
    """Render with wm_render."""
    import wm_render  # ty:ignore[unresolved-import]
    
    print(f"\n{'='*60}")
    print("WM_RENDER")
    print(f"{'='*60}")
    
    renderer = wm_render.Renderer.from_clipgt(
        clipgt_dir=scene_path,
        width=width,
        height=height,
        use_gpu=True,
        cameras=cameras,
    )
    
    camera_names = renderer.camera_names
    print(f"Cameras: {camera_names}")
    print(f"Rendering {len(frame_indices)} frames...")
    
    # Create output dirs
    output_dirs = {}
    for cam_name in camera_names:
        cam_dir = output_dir / cam_name.replace(':', '_').replace('/', '_')
        cam_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[cam_name] = cam_dir
    
    t0 = time.time()
    for out_idx, frame_idx in enumerate(frame_indices):
        images = renderer.render_frame(frame_idx)
        
        for cam_idx, cam_name in enumerate(camera_names):
            img = Image.fromarray(images[cam_idx])
            img.save(output_dirs[cam_name] / f"frame_{out_idx:04d}.png")
        
        if (out_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            fps = (out_idx + 1) / elapsed
            print(f"  Progress: {out_idx + 1}/{len(frame_indices)} ({fps:.1f} FPS)")
    
    total_time = time.time() - t0
    total_images = len(frame_indices) * len(camera_names)
    print(f"Done: {total_images} images in {total_time:.2f}s ({total_images/total_time:.1f} FPS)")
    
    return camera_names


def render_ludus(scene_path: str, output_dir: Path, frame_indices: List[int],
                 width: int, height: int, cameras: Optional[List[str]] = None,
                 msaa_samples: int = 0):
    """Render with ludus_renderer."""
    import torch
    from ludus_renderer import load_clipgt_scene
    from ludus_renderer.render_utils import SceneAdapter, compute_camera_poses
    from ludus_renderer.torch import LudusTimestampedContext
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR
    
    print(f"\n{'='*60}")
    print("LUDUS_RENDERER")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    
    # Load scene WITHOUT ego trajectory
    raw_scene = load_clipgt_scene(
        scene_path, device=device,
        include_ego_trajectory=False,
        include_ego_obstacle=False,
    )
    scene = SceneAdapter(raw_scene)
    
    # Get timestamps
    all_timestamps = scene.ego_tracks.timestamps
    selected_timestamps = all_timestamps[frame_indices]
    
    # Get camera names
    available_cameras = list(raw_scene.camera_name_to_id.keys())
    camera_names = cameras if cameras else available_cameras
    print(f"Cameras: {camera_names}")
    print(f"Rendering {len(frame_indices)} frames...")
    
    # Create context
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    if msaa_samples > 0:
        ctx.set_msaa_samples(msaa_samples)
    
    # Create and upload cameras
    from ludus_renderer.torch import FThetaCamera
    all_cams = []
    for cam_name in camera_names:
        cam_id = raw_scene.camera_name_to_id[cam_name]
        cam = raw_scene.cameras[cam_id]
        
        # Scale to target resolution
        native_w = cam.image_size[0].item()
        native_h = cam.image_size[1].item()
        scale_x = width / native_w
        scale_y = height / native_h
        scale = (scale_x + scale_y) / 2.0
        
        scaled_cam = FThetaCamera(
            principal_point=torch.tensor([cam.principal_point[0].item() * scale_x,
                                          cam.principal_point[1].item() * scale_y], device=device),
            image_size=torch.tensor([float(width), float(height)], device=device),
            fw_poly=cam.fw_poly * scale,
            max_ray_angle=cam.max_ray_angle,
            linear_distortion=cam.linear_distortion,
            depth_max=cam.depth_max,
        )
        all_cams.append(scaled_cam)
    ctx.upload_cameras(all_cams)
    
    # Upload scene
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    
    # Compute poses for all cameras
    all_poses = []
    for cam_name in camera_names:
        poses, _ = compute_camera_poses(scene, selected_timestamps, device, camera_name=cam_name)
        all_poses.append(poses)
    
    # Create output dirs
    output_dirs = {}
    for cam_name in camera_names:
        cam_dir = output_dir / cam_name.replace(':', '_').replace('/', '_')
        cam_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[cam_name] = cam_dir
    
    camera_type_id = CAMERA_TYPE_REGULAR
    n_frames = len(frame_indices)
    
    t0 = time.time()
    for cam_idx, cam_name in enumerate(camera_names):
        camera_poses = all_poses[cam_idx]
        
        for out_idx in range(n_frames):
            scene_ids = torch.tensor([scene_id], dtype=torch.int32, device=device)
            camera_ids = torch.tensor([cam_idx], dtype=torch.int32, device=device)
            timestamps_batch = selected_timestamps[out_idx:out_idx+1].to(torch.int64)
            camera_type_ids = torch.tensor([camera_type_id], dtype=torch.int32, device=device)
            poses_batch = camera_poses[out_idx:out_idx+1]
            
            images = ctx.render(
                scene_ids, camera_ids, timestamps_batch, camera_type_ids,
                poses_batch, resolution=(height, width)
            )
            
            img_np = images[0, :, :, :3].flip(0).cpu().numpy()
            img = Image.fromarray(img_np)
            img.save(output_dirs[cam_name] / f"frame_{out_idx:04d}.png")
        
        elapsed = time.time() - t0
        fps = ((cam_idx + 1) * n_frames) / elapsed
        print(f"  [{cam_idx+1}/{len(camera_names)}] {cam_name} ({fps:.1f} FPS)")
    
    total_time = time.time() - t0
    total_images = n_frames * len(camera_names)
    print(f"Done: {total_images} images in {total_time:.2f}s ({total_images/total_time:.1f} FPS)")
    
    return camera_names


def compute_diffs(wm_dir: Path, ludus_dir: Path, diff_dir: Path,
                  camera_names_wm: List[str], camera_names_ludus: List[str]):
    """Compute image diffs between renderers."""
    print(f"\n{'='*60}")
    print("COMPUTING DIFFS")
    print(f"{'='*60}")
    
    # Map wm camera names to ludus camera names (underscore vs colon)
    # wm: camera_front_wide_120fov -> ludus: camera:front:wide:120fov
    wm_to_ludus = {}
    for wm_name in camera_names_wm:
        # Convert wm name to ludus format for matching
        ludus_name = wm_name.replace('_', ':')
        if ludus_name in camera_names_ludus:
            wm_to_ludus[wm_name] = ludus_name
        else:
            # Try direct match
            if wm_name in camera_names_ludus:
                wm_to_ludus[wm_name] = wm_name
    
    print(f"Matched {len(wm_to_ludus)} cameras")
    
    total_mae = 0.0
    total_frames = 0
    all_results = []
    
    for wm_name, ludus_name in wm_to_ludus.items():
        wm_cam_dir = wm_dir / wm_name.replace(':', '_').replace('/', '_')
        ludus_cam_dir = ludus_dir / ludus_name.replace(':', '_').replace('/', '_')
        diff_cam_dir = diff_dir / wm_name.replace(':', '_').replace('/', '_')
        diff_cam_dir.mkdir(parents=True, exist_ok=True)
        
        wm_frames = sorted(wm_cam_dir.glob('frame_*.png'))
        ludus_frames = sorted(ludus_cam_dir.glob('frame_*.png'))
        
        n_frames = min(len(wm_frames), len(ludus_frames))
        cam_mae = 0.0
        
        for i in range(n_frames):
            wm_img = np.array(Image.open(wm_frames[i]))
            ludus_img = np.array(Image.open(ludus_frames[i]))
            
            diff = np.abs(wm_img.astype(np.float32) - ludus_img.astype(np.float32))
            mae = np.mean(diff)
            cam_mae += mae
            
            # Save diff (no amplification)
            diff_img = np.clip(diff, 0, 255).astype(np.uint8)
            Image.fromarray(diff_img).save(diff_cam_dir / f"diff_{i:04d}.png")
        
        avg_mae = cam_mae / n_frames if n_frames > 0 else 0
        total_mae += cam_mae
        total_frames += n_frames
        all_results.append((wm_name, avg_mae, n_frames))
        print(f"  {wm_name}: MAE={avg_mae:.2f} ({n_frames} frames)")
    
    overall_mae = total_mae / total_frames if total_frames > 0 else 0
    print(f"\nOverall MAE: {overall_mae:.2f}")
    print(f"Diff images saved to: {diff_dir}")
    
    # Save summary
    summary_path = diff_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Comparison Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Overall MAE: {overall_mae:.2f}\n")
        f.write(f"Total frames compared: {total_frames}\n\n")
        f.write(f"Per-camera results:\n")
        for name, mae, n in all_results:
            f.write(f"  {name}: MAE={mae:.2f} ({n} frames)\n")
    
    return overall_mae, list(wm_to_ludus.keys())


def create_comparison_videos(wm_dir: Path, ludus_dir: Path, diff_dir: Path,
                             video_dir: Path, camera_names: List[str], fps: int = 24):
    """Create comparison videos for each camera.
    
    Layout: 1920x1080
      Top row: LUDUS | WM_RENDER (each 960x540)
      Bottom row: DIFF centered (960x540 with black padding)
    """
    import subprocess
    from PIL import ImageDraw
    
    print(f"\n{'='*60}")
    print("CREATING COMPARISON VIDEOS")
    print(f"{'='*60}")
    
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Output dimensions
    out_w, out_h = 1920, 1080
    half_w, half_h = out_w // 2, out_h // 2  # 960x540
    
    for cam_name in camera_names:
        cam_dir_name = cam_name.replace(':', '_').replace('/', '_')
        output_video = video_dir / f'{cam_dir_name}_comparison.mp4'
        
        print(f"Creating video for {cam_name}...")
        
        wm_frames = sorted((wm_dir / cam_dir_name).glob('frame_*.png'))
        ludus_frames = sorted((ludus_dir / cam_dir_name).glob('frame_*.png'))
        diff_frames = sorted((diff_dir / cam_dir_name).glob('diff_*.png'))
        
        if not wm_frames or not ludus_frames or not diff_frames:
            print(f"  Missing frames, skipping")
            continue
        
        n_frames = min(len(wm_frames), len(ludus_frames), len(diff_frames))
        print(f"  {n_frames} frames")
        
        # Create temp directory for combined frames
        temp_dir = video_dir / f'temp_{cam_dir_name}'
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(n_frames):
            ludus_img = Image.open(ludus_frames[i]).resize((half_w, half_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
            wm_img = Image.open(wm_frames[i]).resize((half_w, half_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
            diff_img = Image.open(diff_frames[i]).resize((half_w, half_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
            
            def add_label(img, label):
                draw = ImageDraw.Draw(img)
                draw.rectangle([5, 5, 150, 30], fill=(0, 0, 0))
                draw.text((10, 8), label, fill=(255, 255, 255))
                return img
            
            ludus_labeled = add_label(ludus_img, 'LUDUS')
            wm_labeled = add_label(wm_img, 'WM_RENDER')
            diff_labeled = add_label(diff_img, 'DIFF')
            
            # Create 1920x1080 canvas
            combined = Image.new('RGB', (out_w, out_h), (0, 0, 0))
            
            # Top row: ludus left, wm right
            combined.paste(ludus_labeled, (0, 0))
            combined.paste(wm_labeled, (half_w, 0))
            
            # Bottom row: diff centered
            diff_x = (out_w - half_w) // 2  # Center horizontally
            combined.paste(diff_labeled, (diff_x, half_h))
            
            combined.save(temp_dir / f'frame_{i:04d}.png')
        
        # Create video with ffmpeg
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', str(temp_dir / 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            str(output_video)
        ]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f"  Saved: {output_video}")
        else:
            print(f"  Error creating video: {result.stderr.decode()[:200]}")
        
        # Cleanup temp
        for f in temp_dir.glob('*.png'):
            f.unlink()
        temp_dir.rmdir()
    
    print(f"\nVideos saved to: {video_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare wm_render and ludus_renderer')
    parser.add_argument('--scene', type=str, required=True,
                        help='Path to clipgt scene directory')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Target FPS for rendering (default: 10)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Output width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output height (default: 720)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: _images/comparison_<fps>hz)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to render')
    parser.add_argument('--native', action='store_true',
                        help='Use native frame rate (no subsampling)')
    parser.add_argument('--skip-wm', action='store_true',
                        help='Skip wm_render (use existing)')
    parser.add_argument('--skip-ludus', action='store_true',
                        help='Skip ludus_renderer (use existing)')
    parser.add_argument('--video', action='store_true',
                        help='Create comparison videos')
    parser.add_argument('--video-fps', type=int, default=24,
                        help='Video frame rate (default: 24)')
    parser.add_argument('--skip-diff', action='store_true',
                        help='Skip diff computation (use existing)')
    parser.add_argument('--msaa', type=int, default=0,
                        help='MSAA sample count for ludus (0=disabled, 2, 4, or 8)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_base = Path(args.output)
    else:
        if args.native:
            output_base = Path(__file__).parent.parent / "_images/comparison_native"
        else:
            output_base = Path(__file__).parent.parent / f"_images/comparison_{int(args.fps)}hz"
    
    wm_dir = output_base / "wm_render"
    ludus_dir = output_base / "ludus_render"
    diff_dir = output_base / "diffs"
    
    for d in [wm_dir, ludus_dir, diff_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Scene: {args.scene}")
    print(f"Output: {output_base}")
    if args.native:
        print(f"Mode: Native (all frames)")
    else:
        print(f"Target FPS: {args.fps}")
    print(f"Resolution: {args.width}x{args.height}")
    
    # Get timestamps and subsample
    timestamps = get_native_timestamps(args.scene)
    native_fps = 1_000_000 / (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0
    
    if args.native:
        frame_indices = list(range(len(timestamps)))
        target_fps = native_fps
    else:
        frame_indices = subsample_to_fps(timestamps, args.fps)
        target_fps = args.fps
    
    if args.max_frames:
        frame_indices = frame_indices[:args.max_frames]
    print(f"Native: {len(timestamps)} frames @ {native_fps:.1f} Hz")
    if args.native:
        print(f"Selected: {len(frame_indices)} frames (all)")
    else:
        print(f"Selected: {len(frame_indices)} frames @ ~{args.fps} Hz")
    
    # Save metadata
    meta_path = output_base / "metadata.txt"
    with open(meta_path, 'w') as f:
        f.write(f"scene: {args.scene}\n")
        f.write(f"native_frames: {len(timestamps)}\n")
        f.write(f"native_fps: {native_fps:.1f}\n")
        f.write(f"target_fps: {args.fps}\n")
        f.write(f"selected_frames: {len(frame_indices)}\n")
        f.write(f"frame_indices: {frame_indices}\n")
        f.write(f"width: {args.width}\n")
        f.write(f"height: {args.height}\n")
    
    # Render with wm_render
    if not args.skip_wm:
        camera_names_wm = render_wm(args.scene, wm_dir, frame_indices, args.width, args.height)
    else:
        # Get camera names from existing output
        camera_names_wm = [d.name for d in wm_dir.iterdir() if d.is_dir()]
        print(f"Skipping wm_render, using existing: {camera_names_wm}")
    
    # Render with ludus_renderer
    if not args.skip_ludus:
        camera_names_ludus = render_ludus(args.scene, ludus_dir, frame_indices, args.width, args.height,
                                          msaa_samples=args.msaa)
    else:
        camera_names_ludus = [d.name for d in ludus_dir.iterdir() if d.is_dir()]
        print(f"Skipping ludus_renderer, using existing: {camera_names_ludus}")
    
    # Compute diffs
    if not args.skip_diff:
        _, matched_cameras = compute_diffs(wm_dir, ludus_dir, diff_dir, camera_names_wm, camera_names_ludus)
    else:
        # Get camera names from existing diffs
        matched_cameras = [d.name for d in diff_dir.iterdir() if d.is_dir()]
        print(f"Skipping diff computation, using existing: {len(matched_cameras)} cameras")
    
    # Create comparison videos
    if args.video:
        video_dir = output_base / "videos"
        create_comparison_videos(wm_dir, ludus_dir, diff_dir, video_dir,
                                 matched_cameras, fps=args.video_fps)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {output_base}")
    if args.video:
        print(f"Videos: {output_base / 'videos'}")


if __name__ == "__main__":
    main()
