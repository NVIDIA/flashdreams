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

"""Multi-scene batch rendering benchmark.

Loads multiple AV2 scenes, uploads them to a single GPU context, then
renders random frame windows across scenes in a single batched call.
Designed to validate and benchmark the mixed-scene rendering path needed
for online training.

Example — validate 4 scenes, single batch:

    python examples/render_multi_scene.py \\
        --scene-list example_data/scene_paths_10k.txt \\
        --num-scenes 4 --frames-per-scene 48

Benchmark 100 steps:

    python examples/render_multi_scene.py \\
        --scene-list example_data/scene_paths_10k.txt \\
        --num-scenes 4 --frames-per-scene 48 --benchmark 100
"""

import argparse
import os
import random
import time

import torch

from ludus_renderer.render_utils import (
    load_scene_adapted,
    create_bev_camera,
    get_all_bev_camera_poses,
)
from ludus_renderer.torch import LudusTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV
from ludus_renderer.util import resample_timestamps


def load_scenes(scene_paths, device, num_scenes):
    """Load num_scenes AV2 scenes, returning scene objects."""
    scenes = []
    for i, path in enumerate(scene_paths[:num_scenes]):
        t0 = time.time()
        scene = load_scene_adapted(path.strip(), device=device,
                                   include_ego_obstacle=True)
        dt = time.time() - t0
        n_ts = len(scene.ego_tracks.timestamps)
        print(f"  [{i+1}/{num_scenes}] {n_ts} ego poses, {dt:.2f}s")
        scenes.append(scene)
    return scenes


def sample_frame_window(scene, frames_per_scene, fps, device):
    """Pick a random contiguous window of timestamps from a scene.

    Resamples the scene's ego timestamps at the given fps, then picks a
    random window of frames_per_scene consecutive frames.

    Returns:
        timestamps: [frames_per_scene] int64 tensor on device
        poses: [frames_per_scene, 4, 4] float32 world-to-camera on device
    """
    ego_ts = scene.ego_tracks.timestamps
    timestep_us = 1_000_000 // fps
    duration_us = (ego_ts[-1] - ego_ts[0]).item()
    all_ts = resample_timestamps(ego_ts, timestep_us, duration_us)

    max_start = max(0, len(all_ts) - frames_per_scene)
    start = random.randint(0, max_start)
    ts_window = all_ts[start : start + frames_per_scene].to(device)
    return ts_window


def build_batch(scenes, scene_ids, frames_per_scene, fps,
                bev_height, width, height, device):
    """Build a mixed-scene batch of render queries.

    For each scene, samples a random frame window and computes BEV poses.

    Returns:
        scene_id_tensor:    [N] int32
        camera_id_tensor:   [N] int32 (all zeros — single camera)
        timestamps_tensor:  [N] int64
        camera_type_tensor: [N] int32
        poses_tensor:       [N, 4, 4] float32
        per_scene_ts:       list of per-scene timestamp tensors (for inspection)
    """
    all_scene_ids = []
    all_timestamps = []
    all_poses = []
    per_scene_ts = []

    for scene, sid in zip(scenes, scene_ids):
        ts_window = sample_frame_window(scene, frames_per_scene, fps, device)
        poses = get_all_bev_camera_poses(scene, ts_window, bev_height, device)
        poses = poses.squeeze(1)  # [N, 1, 4, 4] -> [N, 4, 4]

        n = len(ts_window)
        all_scene_ids.append(torch.full((n,), sid, dtype=torch.int32, device=device))
        all_timestamps.append(ts_window)
        all_poses.append(poses)
        per_scene_ts.append(ts_window)

    N = sum(len(t) for t in all_timestamps)
    return (
        torch.cat(all_scene_ids),
        torch.zeros(N, dtype=torch.int32, device=device),
        torch.cat(all_timestamps).to(torch.int64),
        torch.full((N,), CAMERA_TYPE_BEV, dtype=torch.int32, device=device),
        torch.cat(all_poses),
        per_scene_ts,
    )


def save_preview(images, scenes, scene_ids_list, per_scene_ts,
                 frames_per_scene, width, height, output_dir):
    """Save a grid preview image: one column per scene, 4 sample rows."""
    from PIL import Image
    import numpy as np

    n_scenes = len(scenes)
    sample_indices = [0, frames_per_scene // 3,
                      2 * frames_per_scene // 3, frames_per_scene - 1]
    n_rows = len(sample_indices)

    grid = Image.new('RGB', (width * n_scenes, height * n_rows))
    for col, scene_offset in enumerate(range(n_scenes)):
        base = scene_offset * frames_per_scene
        for row, fi in enumerate(sample_indices):
            img_tensor = images[base + fi, :, :, :3].flip(0)
            img = Image.fromarray(img_tensor.cpu().numpy())
            grid.paste(img, (col * width, row * height))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "multi_scene_preview.jpg")
    grid.save(path, quality=90)
    print(f"  Preview saved: {path}")


def run(args):
    device = torch.device('cuda')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Read scene paths
    with open(args.scene_list) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    if args.num_scenes > len(all_paths):
        raise ValueError(
            f"Requested {args.num_scenes} scenes but only "
            f"{len(all_paths)} in {args.scene_list}"
        )

    selected = random.sample(all_paths, args.num_scenes)

    # --- Load scenes ---
    print(f"\nLoading {args.num_scenes} scenes...")
    t_load_start = time.time()
    scenes = load_scenes(selected, device, args.num_scenes)
    t_load = time.time() - t_load_start
    print(f"  Total load time: {t_load:.2f}s "
          f"({t_load / args.num_scenes:.2f}s avg)")

    # --- Setup rendering context ---
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)

    bev_cam = create_bev_camera(args.width, args.height, device,
                                bev_height=args.bev_height,
                                fov_deg=args.bev_fov)
    ctx.upload_cameras([bev_cam])

    print(f"\nUploading {args.num_scenes} scenes to GPU...")
    t_upload_start = time.time()
    scene_ids = []
    for i, scene in enumerate(scenes):
        sid = ctx.upload_scene(scene.timestamped_scene)
        scene_ids.append(sid)
    t_upload = time.time() - t_upload_start
    print(f"  Upload time: {t_upload:.2f}s")

    total_frames = args.num_scenes * args.frames_per_scene
    print(f"\nRendering config:")
    print(f"  Scenes: {args.num_scenes}")
    print(f"  Frames/scene: {args.frames_per_scene}")
    print(f"  Total frames/batch: {total_frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  FPS (sample rate): {args.fps}")
    print(f"  BEV height: {args.bev_height}m, FOV: {args.bev_fov}°")

    # --- Single batch validation ---
    print(f"\n--- Single batch render ---")
    (scene_id_t, cam_id_t, ts_t, cam_type_t, poses_t,
     per_scene_ts) = build_batch(
        scenes, scene_ids, args.frames_per_scene, args.fps,
        args.bev_height, args.width, args.height, device,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    images = ctx.render(scene_id_t, cam_id_t, ts_t, cam_type_t, poses_t,
                        resolution=(args.height, args.width))
    torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"  Output shape: {list(images.shape)}")
    print(f"  Render time: {dt*1000:.1f}ms")
    print(f"  Throughput: {total_frames / dt:.0f} FPS")

    # Save preview
    output_dir = os.path.join(os.path.dirname(__file__), "../_images/multi_scene")
    save_preview(images, scenes, scene_ids, per_scene_ts,
                 args.frames_per_scene, args.width, args.height, output_dir)

    # --- Benchmark ---
    if args.benchmark and args.benchmark > 0:
        n_steps = args.benchmark
        print(f"\n--- Benchmark: {n_steps} steps ---")

        # Warmup
        for _ in range(3):
            s, c, t, ct, p, _ = build_batch(
                scenes, scene_ids, args.frames_per_scene, args.fps,
                args.bev_height, args.width, args.height, device,
            )
            ctx.render(s, c, t, ct, p, resolution=(args.height, args.width))
        torch.cuda.synchronize()

        timings = []
        for step in range(n_steps):
            s, c, t, ct, p, _ = build_batch(
                scenes, scene_ids, args.frames_per_scene, args.fps,
                args.bev_height, args.width, args.height, device,
            )
            torch.cuda.synchronize()
            t0 = time.time()
            ctx.render(s, c, t, ct, p, resolution=(args.height, args.width))
            torch.cuda.synchronize()
            dt = time.time() - t0
            timings.append(dt)

            if (step + 1) % max(1, n_steps // 10) == 0 or step == 0:
                fps = total_frames / dt
                print(f"  Step {step+1:4d}/{n_steps}: "
                      f"{dt*1000:.1f}ms ({fps:.0f} FPS)")

        timings_ms = [t * 1000 for t in timings]
        avg_ms = sum(timings_ms) / len(timings_ms)
        median_ms = sorted(timings_ms)[len(timings_ms) // 2]
        p95_ms = sorted(timings_ms)[int(len(timings_ms) * 0.95)]
        min_ms = min(timings_ms)
        max_ms = max(timings_ms)

        print(f"\n  Summary ({n_steps} steps, {total_frames} frames/step):")
        print(f"    Mean:   {avg_ms:.1f}ms ({total_frames / (avg_ms/1000):.0f} FPS)")
        print(f"    Median: {median_ms:.1f}ms ({total_frames / (median_ms/1000):.0f} FPS)")
        print(f"    P95:    {p95_ms:.1f}ms")
        print(f"    Min:    {min_ms:.1f}ms  Max: {max_ms:.1f}ms")
        print(f"    Budget: <167ms/step -> {'PASS' if p95_ms < 167 else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scene batch rendering benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene-list", required=True,
                        help="Text file with one tar path per line")
    parser.add_argument("--num-scenes", type=int, default=4,
                        help="Number of scenes to load (default: 4)")
    parser.add_argument("--frames-per-scene", type=int, default=48,
                        help="Frames to render per scene (default: 48)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Sampling rate in Hz for frame windows (default: 10)")
    parser.add_argument("--width", type=int, default=512,
                        help="Render width (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                        help="Render height (default: 512)")
    parser.add_argument("--bev-height", type=float, default=80.0,
                        help="BEV camera height in meters (default: 80)")
    parser.add_argument("--bev-fov", type=float, default=60.0,
                        help="BEV camera field of view in degrees (default: 60)")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Run N benchmark steps after validation (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save all rendered frames as individual images")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
