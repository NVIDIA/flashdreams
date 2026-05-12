#!/usr/bin/env python3
"""Benchmark the 3-level scene cache under a simulated training loop.

Each iteration picks --num-scenes random scenes and a random 48-frame
window per scene, loads to L1 and uploads directly to GL, then renders
the batch. Scenes rotate between iterations to exercise cache eviction.

Example — 50 iterations, 4 scenes/iter, 48 frames/scene:

    python examples/benchmark_cache.py \
        --scene-list example_data/scene_paths_10k.txt \
        --cache-dir /tmp/ludus_cache \
        --num-scenes 4 --frames-per-scene 48 --iterations 50

Cold-load throughput test (16 workers):

    python examples/benchmark_cache.py \
        --scene-list example_data/scene_paths_10k.txt \
        --cache-dir /tmp/ludus_cache \
        --num-scenes 4 --iterations 20 --prefetch-workers 16
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch

from ludus_renderer import (
    LudusTimestampedContext,
    CAMERA_TYPE_BEV,
    resample_timestamps,
)
from ludus_renderer.render_utils import (
    create_bev_camera,
    get_all_bev_camera_poses,
    SceneAdapter,
)
from ludus_renderer.scene_cache import SceneDatabase, scene_bytes


def _fmt_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f}B"
    if b < 1024**2:
        return f"{b / 1024:.1f}KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f}MB"
    return f"{b / 1024**3:.2f}GB"


def build_render_batch(db, keys, scene_id_map, args, device):
    """Build a mixed-scene render batch with random frame windows."""
    all_sid, all_ts, all_poses = [], [], []

    for k in keys:
        sid = scene_id_map[k]
        scene = db._cuda_cache.get(k)
        if scene is None:
            scene = db._cpu_cache.get(k)
        if scene is None:
            raise RuntimeError(f"Scene {k} not in L1 or L2 cache")

        adapted = SceneAdapter(scene)
        ego_ts = adapted.ego_tracks.timestamps
        timestep_us = 1_000_000 // args.fps
        duration_us = (ego_ts[-1] - ego_ts[0]).item()
        all_resampled = resample_timestamps(ego_ts, timestep_us, duration_us)

        max_start = max(0, len(all_resampled) - args.frames_per_scene)
        start = random.randint(0, max_start)
        ts_win = all_resampled[start:start + args.frames_per_scene].to(device)

        poses = get_all_bev_camera_poses(adapted, ts_win, args.bev_height, device)
        poses = poses.squeeze(1)

        n = len(ts_win)
        all_sid.append(torch.full((n,), sid, dtype=torch.int32, device=device))
        all_ts.append(ts_win)
        all_poses.append(poses)

    return (
        torch.cat(all_sid),
        torch.zeros(sum(len(t) for t in all_ts), dtype=torch.int32, device=device),
        torch.cat(all_ts).to(torch.int64),
        torch.full((sum(len(t) for t in all_ts),), CAMERA_TYPE_BEV,
                    dtype=torch.int32, device=device),
        torch.cat(all_poses),
    )


def run(args):
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_paths = [l.strip() for l in Path(args.scene_list).read_text().splitlines()
                 if l.strip()]
    pool_size = min(args.scene_pool, len(all_paths))
    scene_pool = random.sample(all_paths, pool_size)

    print(f"Scene pool: {pool_size} scenes")
    print(f"Scenes/iter: {args.num_scenes}, Frames/scene: {args.frames_per_scene}")
    print(f"Resolution: {args.width}x{args.height}, FPS: {args.fps}")
    print(f"L1 (CUDA): {_fmt_bytes(args.max_gpu_bytes)}, "
          f"L2 (CPU): {_fmt_bytes(args.max_cpu_bytes)}")
    print(f"Prefetch workers: {args.prefetch_workers}, L1 prefetch: {args.prefetch_l1}")

    try:
        import lmdb
        print(f"LMDB: available")
    except ImportError:
        print(f"LMDB: not installed (using per-file L3 cache)")

    db = SceneDatabase(
        cache_dir=args.cache_dir,
        max_cpu_bytes=args.max_cpu_bytes,
        max_gpu_bytes=args.max_gpu_bytes,
        prefetch_workers=args.prefetch_workers,
    )
    all_keys = db.register_scenes(scene_pool)

    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    ctx.preallocate_buffers(max_scenes=args.num_scenes,
                            bytes_per_scene=2 * 1024 * 1024)
    print(f"Pre-allocated GL buffers: {args.num_scenes} scenes @ 2MB")
    bev = create_bev_camera(args.width, args.height, device,
                            bev_height=args.bev_height, fov_deg=args.bev_fov)
    ctx.upload_cameras([bev])

    total_frames = args.num_scenes * args.frames_per_scene
    n_iter = args.iterations

    t_ensure_list = []
    t_render_list = []
    t_total_list = []

    print(f"\n{'='*70}")
    print(f"{'Iter':>5}  {'upload':>12}  {'render':>10}  {'total':>10}  "
          f"{'FPS':>8}  {'L1':>6}")
    print(f"{'='*70}")

    for it in range(n_iter):
        iter_keys = random.sample(all_keys, args.num_scenes)

        if it + 1 < n_iter:
            next_keys = random.sample(all_keys, args.num_scenes)
            db.prefetch(next_keys)
            if args.prefetch_l1:
                db.prefetch_l1(next_keys)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx.clear_scenes()
        scene_id_map = {}
        for k in iter_keys:
            scene = db._ensure_l1(k)
            sid = ctx.upload_scene(scene.timestamped_scene)
            scene_id_map[k] = sid
        torch.cuda.synchronize()
        t_ensure = time.perf_counter() - t0

        # Build batch + render
        sid_t, cam_t, ts_t, type_t, poses_t = build_render_batch(
            db, iter_keys, scene_id_map, args, device)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        images = ctx.render(sid_t, cam_t, ts_t, type_t, poses_t,
                            resolution=(args.height, args.width))
        torch.cuda.synchronize()
        t_render = time.perf_counter() - t1

        t_total = t_ensure + t_render
        fps = total_frames / t_total if t_total > 0 else 0

        t_ensure_list.append(t_ensure)
        t_render_list.append(t_render)
        t_total_list.append(t_total)

        stats = db.stats
        if (it + 1) % max(1, n_iter // 20) == 0 or it == 0 or it == n_iter - 1:
            print(f"{it+1:5d}  {t_ensure*1000:10.1f}ms  {t_render*1000:8.1f}ms  "
                  f"{t_total*1000:8.1f}ms  {fps:7.0f}  "
                  f"{stats.l1_size:4d}")

    # --- Summary ---
    # Skip first iteration (cold start) for steady-state stats
    skip = min(3, n_iter - 1)
    warm_ensure = t_ensure_list[skip:]
    warm_render = t_render_list[skip:]
    warm_total = t_total_list[skip:]

    def _stats(vals):
        s = sorted(vals)
        n = len(s)
        return {
            "mean": sum(s) / n,
            "median": s[n // 2],
            "p95": s[int(n * 0.95)],
            "min": s[0],
            "max": s[-1],
        }

    stats_final = db.stats
    print(f"\n{'='*70}")
    print(f"Summary ({n_iter} iterations, skip first {skip} for warmup)")
    print(f"{'='*70}")

    if warm_total:
        es = _stats(warm_ensure)
        rs = _stats(warm_render)
        ts = _stats(warm_total)
        print(f"\n  upload (ms):      mean={es['mean']*1000:.1f}  "
              f"median={es['median']*1000:.1f}  p95={es['p95']*1000:.1f}  "
              f"min={es['min']*1000:.1f}  max={es['max']*1000:.1f}")
        print(f"  render (ms):      mean={rs['mean']*1000:.1f}  "
              f"median={rs['median']*1000:.1f}  p95={rs['p95']*1000:.1f}  "
              f"min={rs['min']*1000:.1f}  max={rs['max']*1000:.1f}")
        print(f"  total (ms):       mean={ts['mean']*1000:.1f}  "
              f"median={ts['median']*1000:.1f}  p95={ts['p95']*1000:.1f}  "
              f"min={ts['min']*1000:.1f}  max={ts['max']*1000:.1f}")
        print(f"  throughput (FPS): mean={total_frames / ts['mean']:.0f}  "
              f"median={total_frames / ts['median']:.0f}  "
              f"p95={total_frames / ts['p95']:.0f}")

    print(f"\n  Cache: {stats_final}")
    print(f"  L1 hit rate: {stats_final.l1_hit_rate:.1%}")
    print(f"  L2 hit rate: {stats_final.l2_hit_rate:.1%}")
    print(f"  L3 hit rate: {stats_final.l3_hit_rate:.1%}")
    print(f"  L1 evictions: {stats_final.l1_evictions}")

    db.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 3-level scene cache under simulated training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene-list", required=True,
                        help="Text file with scene tar paths")
    parser.add_argument("--cache-dir", required=True,
                        help="L3 disk cache directory")
    parser.add_argument("--num-scenes", type=int, default=4,
                        help="Scenes per iteration (default: 4)")
    parser.add_argument("--frames-per-scene", type=int, default=48,
                        help="Frame window size per scene (default: 48)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Sampling rate in Hz (default: 10)")
    parser.add_argument("--width", type=int, default=512,
                        help="Render width (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                        help="Render height (default: 512)")
    parser.add_argument("--bev-height", type=float, default=80.0,
                        help="BEV camera height in meters (default: 80)")
    parser.add_argument("--bev-fov", type=float, default=60.0,
                        help="BEV camera FOV in degrees (default: 60)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of training-loop iterations (default: 50)")
    parser.add_argument("--scene-pool", type=int, default=100,
                        help="Total scenes to sample from (default: 100)")
    parser.add_argument("--max-gpu-bytes", type=int, default=4 * 1024**3,
                        help="L1 CUDA cache capacity in bytes (default: 4GB)")
    parser.add_argument("--max-cpu-bytes", type=int, default=16 * 1024**3,
                        help="L2 CPU cache capacity in bytes (default: 16GB)")
    parser.add_argument("--prefetch-workers", type=int, default=8,
                        help="Background prefetch threads (default: 8)")
    parser.add_argument("--prefetch-l1", action="store_true",
                        help="Enable async L1 CUDA prefetch (L2→L1 on dedicated stream)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
