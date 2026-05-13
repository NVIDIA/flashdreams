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

"""Test and benchmark the 3-level scene cache (L1 CUDA / L2 CPU / L3 Disk).

Exercises the full L3 → L2 → L1 pipeline:
1. Cold load from tar (populating L3 disk cache on first run)
2. L3 disk cache hit
3. L2 CPU cache hit
4. Async prefetch
5. L1 CUDA tensor cache + direct GL upload
6. Async L1 prefetch
7. Render correctness: direct tar vs cached (pixel-identical check)

Example:

    python examples/test_scene_cache.py \\
        --scene-list example_data/scene_paths_10k.txt \\
        --cache-dir /tmp/ludus_cache \\
        --num-scenes 8 --prefetch-workers 4
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Test 3-level scene cache pipeline")
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU tests")
    args = parser.parse_args()

    from ludus_renderer.scene_cache import SceneDatabase, scene_key

    all_paths = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
    random.seed(args.seed)
    selected = random.sample(all_paths, min(args.num_scenes, len(all_paths)))

    db = SceneDatabase(
        cache_dir=args.cache_dir,
        max_cpu_bytes=256 * 1024**2,   # 256 MB L2 cache
        max_gpu_bytes=64 * 1024**2,    # 64 MB L1 cache (small for testing eviction)
        prefetch_workers=args.prefetch_workers,
    )

    keys = db.register_scenes(selected)
    print(f"Registered {len(keys)} scenes")

    # --- Test 1: Cold load (tar → L3 → L2) ---
    print(f"\n--- Test 1: Cold load (tar → L3 disk cache → L2 CPU cache) ---")
    t0 = time.perf_counter()
    scenes = db.ensure_cpu(keys)
    t_cold = time.perf_counter() - t0
    print(f"  Loaded {len(scenes)} scenes in {t_cold:.2f}s "
          f"({len(scenes) / t_cold:.1f} scenes/s)")
    print(f"  Stats: {db.stats}")

    # --- Test 2: L2 hit (should be instant) ---
    print(f"\n--- Test 2: L2 CPU cache hit ---")
    t0 = time.perf_counter()
    scenes2 = db.ensure_cpu(keys)
    t_l2 = time.perf_counter() - t0
    print(f"  Loaded {len(scenes2)} scenes in {t_l2 * 1000:.2f}ms")
    print(f"  Stats: {db.stats}")

    # --- Test 3: Clear L2, reload from L3 disk cache ---
    print(f"\n--- Test 3: L3 disk cache hit (after evicting L2) ---")
    db._cpu_cache._cache.clear()
    db._stats.l2_hits = 0
    db._stats.l2_misses = 0
    t0 = time.perf_counter()
    scenes3 = db.ensure_cpu(keys)
    t_l3 = time.perf_counter() - t0
    print(f"  Loaded {len(scenes3)} scenes in {t_l3 * 1000:.1f}ms "
          f"({t_cold / t_l3:.1f}x faster than cold load)")
    print(f"  Stats: {db.stats}")

    # --- Test 4: Prefetch ---
    print(f"\n--- Test 4: Async prefetch ---")
    db._cpu_cache._cache.clear()
    more_paths = random.sample(all_paths, min(args.num_scenes, len(all_paths)))
    more_keys = db.register_scenes(more_paths)

    t0 = time.perf_counter()
    db.prefetch(more_keys)
    print(f"  Prefetch submitted in {(time.perf_counter() - t0) * 1000:.1f}ms")

    # Wait for completion
    time.sleep(0.1)
    while db.stats.prefetch_completed < db.stats.prefetch_queued:
        time.sleep(0.1)
    t_prefetch = time.perf_counter() - t0
    print(f"  Prefetch completed in {t_prefetch:.2f}s")

    # Now ensure_cpu should be all L2 hits
    t0 = time.perf_counter()
    db.ensure_cpu(more_keys)
    t_after = time.perf_counter() - t0
    print(f"  Post-prefetch ensure_cpu: {t_after * 1000:.2f}ms")
    print(f"  Stats: {db.stats}")

    # --- Test 5: GPU upload (L2 → L1 → GL direct) ---
    if not args.no_gpu:
        print(f"\n--- Test 5: GPU upload (L2 → L1 CUDA → GL direct) ---")
        try:
            from ludus_renderer import LudusTimestampedContext, FThetaCamera
            from ludus_renderer.render_utils import create_bev_camera

            device = torch.device("cuda")
            ctx = LudusTimestampedContext(device=device)

            ctx.preallocate_buffers(max_scenes=len(keys), bytes_per_scene=2 * 1024 * 1024)
            print(f"  Pre-allocated GL buffers for {len(keys)} scenes @ 2MB")

            bev_cam = create_bev_camera(256, 256, device=device)
            ctx.upload_cameras([bev_cam])

            ctx.clear_scenes()
            t0 = time.perf_counter()
            scene_id_map = {}
            for k in keys:
                scene = db._ensure_l1(k)
                sid = ctx.upload_scene(scene.timestamped_scene)
                scene_id_map[k] = sid
            t_gpu = time.perf_counter() - t0
            print(f"  Uploaded {len(scene_id_map)} scenes in {t_gpu * 1000:.1f}ms")
            print(f"  Scene ID map: {scene_id_map}")
            print(f"  Stats: {db.stats}")

            # --- Test 5b: Async L1 prefetch ---
            print(f"\n--- Test 5b: Async L1 CUDA prefetch ---")
            extra_paths = random.sample(all_paths, min(4, len(all_paths)))
            extra_keys = db.register_scenes(extra_paths)
            for ek in extra_keys:
                db._cuda_cache.remove(ek)

            db.ensure_cpu(extra_keys)
            l1_before = db._cuda_cache.size
            t0 = time.perf_counter()
            db.prefetch_l1(extra_keys)
            t_submit = time.perf_counter() - t0
            torch.cuda.synchronize()
            t_done = time.perf_counter() - t0
            l1_after = db._cuda_cache.size
            print(f"  prefetch_l1 submit: {t_submit * 1000:.2f}ms, "
                  f"sync: {t_done * 1000:.2f}ms")
            print(f"  L1 size: {l1_before} → {l1_after} "
                  f"(+{l1_after - l1_before} promoted)")
            print(f"  Stats: {db.stats}")

        except Exception as e:
            import traceback
            print(f"  GPU test failed: {e}")
            traceback.print_exc()

    # --- Test 6: Render comparison (cached vs direct) ---
    if not args.no_gpu:
        print(f"\n--- Test 7: Render correctness (direct tar vs cached L1) ---")
        try:
            from ludus_renderer.render_utils import (
                load_scene_adapted, create_bev_camera,
                get_all_bev_camera_poses, SceneAdapter,
            )
            from ludus_renderer import (
                LudusTimestampedContext, CAMERA_TYPE_BEV,
                resample_timestamps,
            )
            from ludus_renderer.scene_cache import scene_to_device

            device = torch.device("cuda")
            W, H = 256, 256
            bev_height = 80.0
            fps = 10
            n_frames = 6
            test_paths = selected[:4]
            test_keys = [db.key_for_path(p) for p in test_paths]

            def render_scenes(scenes_list, label):
                ctx2 = LudusTimestampedContext(device=device)
                bev = create_bev_camera(W, H, device=device)
                ctx2.upload_cameras([bev])
                sids = []
                for sc in scenes_list:
                    sids.append(ctx2.upload_scene(sc.timestamped_scene))

                all_sid, all_ts, all_poses = [], [], []
                for sc, sid in zip(scenes_list, sids):
                    ego_ts = sc.ego_tracks.timestamps
                    timestep_us = 1_000_000 // fps
                    dur = (ego_ts[-1] - ego_ts[0]).item()
                    ts_all = resample_timestamps(ego_ts, timestep_us, dur)
                    ts_win = ts_all[:n_frames].to(device)
                    poses = get_all_bev_camera_poses(sc, ts_win, bev_height, device)
                    poses = poses.squeeze(1)
                    n = len(ts_win)
                    all_sid.append(torch.full((n,), sid, dtype=torch.int32, device=device))
                    all_ts.append(ts_win)
                    all_poses.append(poses)

                sid_t = torch.cat(all_sid)
                cam_t = torch.zeros_like(sid_t)
                ts_t = torch.cat(all_ts).to(torch.int64)
                type_t = torch.full_like(sid_t, CAMERA_TYPE_BEV)
                poses_t = torch.cat(all_poses)
                imgs = ctx2.render(sid_t, cam_t, ts_t, type_t, poses_t,
                                   resolution=(H, W))
                torch.cuda.synchronize()
                print(f"  {label}: rendered {imgs.shape[0]} frames")
                return imgs

            # A: Load directly from tar (ground truth)
            direct_scenes = []
            for p in test_paths:
                sc = load_scene_adapted(p, device=device)
                direct_scenes.append(sc)
            imgs_direct = render_scenes(direct_scenes, "Direct (tar)")

            # B: Load from cache via L1 CUDA path (scene_to_device)
            db._cpu_cache._cache.clear()
            db._cuda_cache._cache.clear()
            db._cuda_cache._total_bytes = 0
            cached_scenes = []
            for k in test_keys:
                cpu_sc = db._ensure_cpu_one(k)
                gpu_sc = scene_to_device(cpu_sc, device)
                cached_scenes.append(SceneAdapter(gpu_sc))
            imgs_cached = render_scenes(cached_scenes, "Cached (L1)")

            # Compare
            diff = (imgs_direct.float() - imgs_cached.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            nonzero = (diff.sum(dim=-1) > 0).sum().item()
            total_px = diff.shape[0] * diff.shape[1] * diff.shape[2]
            print(f"  Max pixel diff:  {max_diff:.1f}")
            print(f"  Mean pixel diff: {mean_diff:.4f}")
            print(f"  Nonzero pixels:  {nonzero} / {total_px}")
            if max_diff == 0:
                print(f"  PASS: Renders are pixel-identical")
            else:
                print(f"  WARNING: Renders differ (check serialization)")

            # Save comparison grid: columns = scenes, rows = [direct, cached, diff×10]
            from PIL import Image
            n_scenes_show = len(test_paths)
            sample_frame = n_frames // 2
            cols = []
            for si in range(n_scenes_show):
                idx = si * n_frames + sample_frame
                d = imgs_direct[idx, :, :, :3].flip(0).cpu().numpy()
                c = imgs_cached[idx, :, :, :3].flip(0).cpu().numpy()
                df = np.clip(diff[idx, :, :, :3].flip(0).cpu().numpy() * 10, 0, 255).astype(np.uint8)
                cols.append(np.concatenate([d, c, df], axis=0))
            grid = np.concatenate(cols, axis=1)
            out_dir = "_images/cache_test"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "compare.jpg")
            Image.fromarray(grid).save(out_path, quality=95)
            print(f"  Saved comparison: {out_path}")
            print(f"    Columns: {n_scenes_show} scenes, Rows: direct / cached / diff×10")

        except Exception as e:
            import traceback
            print(f"  Render comparison failed: {e}")
            traceback.print_exc()

    print(f"\n--- Summary ---")
    print(f"  Cold load (tar):     {t_cold:.2f}s")
    print(f"  L2 cache hit:        {t_l2 * 1000:.2f}ms")
    print(f"  L3 cache hit (disk): {t_l3 * 1000:.1f}ms")
    print(f"  Speedup L3 vs cold:  {t_cold / t_l3:.1f}x")
    print(f"  Final stats: {db.stats}")

    db.shutdown()


if __name__ == "__main__":
    main()
