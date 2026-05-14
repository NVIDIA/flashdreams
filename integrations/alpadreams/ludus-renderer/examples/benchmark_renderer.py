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
Ludus Renderer Benchmarks

Performance benchmarking tools for the Ludus GPU renderer.

Includes:
- Single scene benchmarks (single/multi-camera)
- Multi-scene batch rendering benchmarks
- GPU memory profiling

Usage:
    # Single scene benchmark
    python benchmark_renderer.py --scene <path> --iters 10
    
    # Multi-camera benchmark (8 cameras per timestamp)
    python benchmark_renderer.py --scene <path> --multicam --iters 10
    
    # Multi-scene batch benchmark
    python benchmark_renderer.py --batch --scenes-dir <path> --iters 10
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from PIL import Image

# Default paths
DEFAULT_SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../example_data/test_hdmap'
)


from ludus_renderer.render_utils import (
    get_gpu_memory_stats, create_camera, get_available_cameras,
    load_scene_adapted as load_scene,
)


def print_stats(name: str, times_ms: List[float], n_items: int = 1):
    """Print timing statistics."""
    avg = np.mean(times_ms)
    std = np.std(times_ms)
    min_t = np.min(times_ms)
    max_t = np.max(times_ms)
    fps = n_items / (avg / 1000.0) if avg > 0 else 0
    print(f"  {name}:")
    print(f"    Avg: {avg:.2f}ms ({fps:.1f} FPS)")
    print(f"    Min: {min_t:.2f}ms, Max: {max_t:.2f}ms, Std: {std:.2f}ms")


def find_scene_dirs(scenes_dir: str) -> List[str]:
    """Find all scene directories in a folder."""
    scenes_dir = Path(scenes_dir)
    if not scenes_dir.exists():
        return []
    
    scene_paths = []
    for item in scenes_dir.iterdir():
        if item.is_dir():
            # Check if it has parquet files
            if list(item.glob("*.parquet")):
                scene_paths.append(str(item))
    
    return sorted(scene_paths)


def benchmark_single_scene(args):
    """Benchmark rendering a single scene."""
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.torch import LudusTimestampedContext
    from ludus_renderer.render_utils import render_all_frames
    
    device = torch.device('cuda')
    width, height = args.width, args.height
    n_iters = args.iters
    
    print("=" * 70)
    print("SINGLE SCENE BENCHMARK")
    print("=" * 70)
    
    # Load scene
    print(f"\nLoading scene: {args.scene}")
    t0 = time.time()
    scene = load_scene(args.scene, device)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")
    
    # Resample timestamps
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    n_frames = len(timestamps)
    print(f"  Frames: {n_frames}")
    
    # Create context and camera
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    
    bev_height = args.bev_height if args.bev else None
    camera = create_camera(width, height, device, bev=args.bev,
                           bev_height=args.bev_height, bev_fov=args.bev_fov)
    ctx.upload_cameras([camera])
    
    # Upload scene
    t0 = time.time()
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    upload_time = time.time() - t0
    print(f"  Upload time: {upload_time*1000:.1f}ms")
    
    mem = get_gpu_memory_stats()
    print(f"  GPU memory: {mem['allocated_mb']:.1f}MB allocated")
    
    # Warmup
    print(f"\nWarming up...")
    _, _ = render_all_frames(ctx, scene, scene_id, timestamps, width, height, 
                             device, bev_height=bev_height, verbose=False)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"\nBenchmarking ({n_iters} iterations, {n_frames} frames each)...")
    gpu_times = []
    total_times = []
    
    for i in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        
        _, timings = render_all_frames(ctx, scene, scene_id, timestamps, width, height,
                                       device, bev_height=bev_height, verbose=False)
        
        torch.cuda.synchronize()
        total_time = time.time() - t0
        
        gpu_times.append(timings['gpu_render'] * 1000)
        total_times.append(total_time * 1000)
        
        fps = n_frames / timings['gpu_render']
        print(f"  Iter {i+1}/{n_iters}: GPU={timings['gpu_render']*1000:.2f}ms ({fps:.0f} FPS)")
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Scene: {args.scene}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames per iteration: {n_frames}")
    print(f"Iterations: {n_iters}")
    print(f"Total frames rendered: {n_iters * n_frames}")
    print()
    print_stats("GPU Render", gpu_times, n_frames)
    print_stats("Total (GPU + CPU transfer)", total_times, n_frames)


def benchmark_multicam(args):
    """Benchmark multi-camera rendering (all cameras per timestamp)."""
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.torch import LudusTimestampedContext
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV
    from ludus_renderer.render_utils import (
        get_all_camera_poses, get_all_bev_camera_poses, compute_camera_poses,
    )
    
    device = torch.device('cuda')
    width, height = args.width, args.height
    n_iters = args.iters
    
    # Camera names to use
    camera_names = [
        'camera:front:wide:120fov',
        'camera:front:tele:30fov', 
        'camera:rear:fisheye:200fov',
        'camera:left:fisheye:200fov',
        'camera:right:fisheye:200fov',
        'camera:rear:left:70fov',
        'camera:rear:right:70fov',
        'BEV',
    ]
    n_cameras = len(camera_names)
    
    print("=" * 70)
    print("MULTI-CAMERA BENCHMARK")
    print("=" * 70)
    print(f"Cameras: {camera_names}")
    
    # Load scene
    print(f"\nLoading scene: {args.scene}")
    scene = load_scene(args.scene, device)
    
    # Resample timestamps  
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    n_frames = len(timestamps)
    print(f"  Frames: {n_frames}")
    print(f"  Total images per iteration: {n_frames * n_cameras}")
    
    # Create context
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    
    # Create cameras
    cameras = []
    for cam_name in camera_names:
        cameras.append(create_camera(width, height, device,
                                     bev=(cam_name == 'BEV'),
                                     scene=scene, camera_name=cam_name))
    
    ctx.upload_cameras(cameras)
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    
    # Pre-compute all camera poses for all timestamps
    print("\nPre-computing camera poses...")
    all_camera_poses = []
    for cam_name in camera_names:
        if cam_name == 'BEV':
            poses = get_all_bev_camera_poses(scene, timestamps, 80.0, device)
        else:
            if cam_name in scene.cameras:
                poses = get_all_camera_poses(scene, timestamps, cam_name, device)
            else:
                poses = torch.eye(4, device=device).unsqueeze(0).expand(n_frames, -1, -1)
        all_camera_poses.append(poses)
    
    all_camera_poses = torch.stack(all_camera_poses, dim=0)  # [n_cameras, n_frames, 4, 4]
    
    # Warmup
    print("Warming up...")
    for ts_idx in range(min(5, n_frames)):
        queries = [(scene_id, cam_idx, timestamps[ts_idx].item(), 
                   CAMERA_TYPE_BEV if camera_names[cam_idx] == 'BEV' else CAMERA_TYPE_REGULAR)
                  for cam_idx in range(n_cameras)]
        poses = all_camera_poses[:, ts_idx, :, :]
        _ = ctx.render_batch(queries, poses, resolution=(height, width))
    torch.cuda.synchronize()
    
    # Benchmark: per-timestamp rendering
    print(f"\nBenchmarking per-timestamp rendering ({n_iters} iterations)...")
    iter_times = []
    per_ts_times = []
    
    for iter_idx in range(n_iters):
        ts_times = []
        
        torch.cuda.synchronize()
        iter_start = time.time()
        
        for ts_idx in range(n_frames):
            torch.cuda.synchronize()
            t0 = time.time()
            
            queries = [(scene_id, cam_idx, timestamps[ts_idx].item(),
                       CAMERA_TYPE_BEV if camera_names[cam_idx] == 'BEV' else CAMERA_TYPE_REGULAR)
                      for cam_idx in range(n_cameras)]
            poses = all_camera_poses[:, ts_idx, :, :]
            
            images = ctx.render_batch(queries, poses, resolution=(height, width))
            
            torch.cuda.synchronize()
            ts_times.append((time.time() - t0) * 1000)
        
        iter_time = (time.time() - iter_start) * 1000
        iter_times.append(iter_time)
        per_ts_times.extend(ts_times)
        
        avg_ts = np.mean(ts_times)
        print(f"  Iter {iter_idx+1}/{n_iters}: {iter_time:.1f}ms total, {avg_ts:.2f}ms/timestamp")
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Scene: {args.scene}")
    print(f"Resolution: {width}x{height}")
    print(f"Cameras: {n_cameras}")
    print(f"Timestamps: {n_frames}")
    print(f"Images per iteration: {n_frames * n_cameras}")
    print(f"Iterations: {n_iters}")
    print()
    print_stats(f"Per-timestamp ({n_cameras} cameras)", per_ts_times, n_cameras)
    print_stats(f"Full iteration ({n_frames} timestamps)", iter_times, n_frames * n_cameras)


def benchmark_batch(args):
    """Benchmark multi-scene batch rendering."""
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.torch import LudusTimestampedContext
    
    device = torch.device('cuda')
    width, height = args.width, args.height
    n_iters = args.iters
    
    print("=" * 70)
    print("MULTI-SCENE BATCH BENCHMARK")
    print("=" * 70)
    
    # Find scenes
    scene_paths = find_scene_dirs(args.scenes_dir)
    if not scene_paths:
        print(f"No scenes found in {args.scenes_dir}")
        return
    
    print(f"Found {len(scene_paths)} scenes")
    
    # Load all scenes
    print("\nLoading scenes...")
    scenes = []
    all_timestamps = []
    
    for path in scene_paths:
        print(f"  Loading: {path}")
        scene = load_scene(path, device)
        scenes.append(scene)
        
        ts = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
        all_timestamps.append(ts)
    
    # Create unified context
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    
    camera = create_camera(width, height, device)
    ctx.upload_cameras([camera])
    
    # Upload all scenes
    print("\nUploading scenes to GPU...")
    scene_ids = []
    for i, scene in enumerate(scenes):
        sid = ctx.upload_scene(scene.timestamped_scene)
        scene_ids.append(sid)
    
    mem = get_gpu_memory_stats()
    print(f"  GPU memory: {mem['allocated_mb']:.1f}MB allocated")
    
    # Build combined query list
    total_frames = sum(len(ts) for ts in all_timestamps)
    print(f"\nTotal frames across all scenes: {total_frames}")
    
    # Benchmark
    print(f"\nBenchmarking ({n_iters} iterations)...")
    render_times = []
    
    for iter_idx in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Render each scene
        for scene_idx, (scene, timestamps) in enumerate(zip(scenes, all_timestamps)):
            n_frames = len(timestamps)
            scene_id = scene_ids[scene_idx]
            
            from ludus_renderer.render_utils import get_all_camera_poses
            if 'camera:front:wide:120fov' in scene.cameras:
                poses = get_all_camera_poses(scene, timestamps, 'camera:front:wide:120fov', device)
            else:
                poses = torch.eye(4, device=device).unsqueeze(0).expand(n_frames, -1, -1)
            
            queries = [(scene_id, 0, ts.item(), 0) for ts in timestamps]
            images = ctx.render_batch(queries, poses, resolution=(height, width))
        
        torch.cuda.synchronize()
        render_time = (time.time() - t0) * 1000
        render_times.append(render_time)
        
        fps = total_frames / (render_time / 1000)
        print(f"  Iter {iter_idx+1}/{n_iters}: {render_time:.1f}ms ({fps:.0f} FPS)")
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Scenes: {len(scenes)}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")
    print(f"Iterations: {n_iters}")
    print()
    print_stats("Full batch render", render_times, total_frames)


def main():
    parser = argparse.ArgumentParser(description='Ludus Renderer Benchmarks')
    parser.add_argument('--scene', type=str, default=DEFAULT_SCENE_PATH,
                        help='Path to clipgt scene directory')
    parser.add_argument('--scenes-dir', type=str, default=None,
                        help='Directory containing multiple scenes (for --batch)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Output image width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output image height (default: 720)')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--bev', action='store_true',
                        help='Use bird\'s eye view camera')
    parser.add_argument('--bev-height', type=float, default=80.0,
                        help='BEV camera height in meters (default: 80)')
    parser.add_argument('--bev-fov', type=float, default=60.0,
                        help='BEV camera FOV in degrees (default: 60)')
    parser.add_argument('--multicam', action='store_true',
                        help='Multi-camera benchmark (8 cameras per timestamp)')
    parser.add_argument('--batch', action='store_true',
                        help='Multi-scene batch benchmark')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.scenes_dir:
            print("Error: --scenes-dir required for --batch mode")
            return
        benchmark_batch(args)
    elif args.multicam:
        benchmark_multicam(args)
    else:
        benchmark_single_scene(args)


if __name__ == "__main__":
    main()
