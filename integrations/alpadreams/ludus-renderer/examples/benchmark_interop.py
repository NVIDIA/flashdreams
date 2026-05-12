#!/usr/bin/env python3
"""Benchmark the rendering pipeline to identify where time is actually spent.

Measures each phase separately using CUDA events for GPU-accurate timing:
  1. Query packing (CPU → GPU)
  2. Scene upload (beginUpload / memcpy / endUpload)  
  3. Sync before draw (cudaGraphicsMap or semaphore signal/wait)
  4. GL draw calls
  5. Sync after draw (glFinish / semaphore signal)
  6. Color readback (map texture → cudaMemcpy3D → unmap)
  7. GPU → CPU transfer

Usage:
    uv run python examples/benchmark_interop.py --scene example_data/test_hdmap
    uv run python examples/benchmark_interop.py --scene /path/to/scene.tar --interop extmem
"""
import argparse
import time
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def benchmark(args):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    from ludus_renderer.torch import LudusTimestampedContext
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.render_utils import (
        load_scene_adapted as load_scene,
        create_camera, get_available_cameras,
        compute_camera_poses,
    )
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR

    print(f"=== Interop Pipeline Benchmark ===")
    print(f"Interop mode: {args.interop}")
    print(f"Resolution:   {args.width}x{args.height}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Warmup:       {args.warmup} iterations")
    print(f"Measure:      {args.iterations} iterations")
    print()

    ctx = LudusTimestampedContext(device=device, interop=args.interop)

    print(f"Loading scene: {args.scene}")
    t0 = time.time()
    scene = load_scene(args.scene, device)
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    load_time = time.time() - t0
    print(f"  Scene loaded in {load_time:.2f}s")

    cam_names = get_available_cameras(scene)
    if not cam_names:
        print("ERROR: No cameras found in scene")
        return

    cam_name = cam_names[0]
    camera = create_camera(args.width, args.height, device, scene=scene, camera_name=cam_name)
    ctx.upload_cameras([camera])

    # Get timestamps (same approach as render_hdmap_scene.py)
    ego_ts = scene.ego_tracks.timestamps
    timestep_us = 1000000 // 10  # 10 Hz
    duration_us = (ego_ts[-1] - ego_ts[0]).item()
    timestamps = resample_timestamps(ego_ts, timestep_us, duration_us)
    timestamps = timestamps.to(device)

    # Compute camera poses
    all_poses, camera_type_id = compute_camera_poses(
        scene, timestamps, device, camera_name=cam_name)

    n_ts = len(timestamps)
    batch = min(args.batch_size, n_ts)
    height, width = args.height, args.width

    scene_ids = torch.zeros(batch, dtype=torch.int32, device=device)
    camera_ids = torch.zeros(batch, dtype=torch.int32, device=device)
    camera_type_ids = torch.full((batch,), camera_type_id, dtype=torch.int32, device=device)
    ts_batch = timestamps[:batch]
    poses_batch = all_poses[:batch]

    queries_tensor = ctx.pack_queries_fast(scene_ids, camera_ids, ts_batch, camera_type_ids)
    resolution = (height, width)

    print(f"\nUsing camera: {cam_name}")
    print(f"Batch: {batch} queries at {width}x{height}")
    print()

    # Phase timing with CUDA events
    def make_events(n):
        return [torch.cuda.Event(enable_timing=True) for _ in range(n)]

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        images = ctx.render(
            scene_ids, camera_ids, ts_batch, camera_type_ids,
            poses_batch, resolution=resolution
        )
    torch.cuda.synchronize()
    print("  Done\n")

    # Measure full render call (the only thing we can time from Python level)
    print(f"Measuring full render call ({args.iterations} iterations)...")
    times_full = []
    for i in range(args.iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        images = ctx.render(
            scene_ids, camera_ids, ts_batch, camera_type_ids,
            poses_batch, resolution=resolution
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times_full.append(elapsed)
    
    times_full_ms = [t * 1000 for t in times_full]
    print(f"  Full render: {np.median(times_full_ms):.2f}ms median, "
          f"{np.mean(times_full_ms):.2f}ms mean, "
          f"{np.std(times_full_ms):.2f}ms std")
    print(f"  Per-query:   {np.median(times_full_ms)/batch:.3f}ms")
    print(f"  Throughput:  {batch/np.median(times_full)*1000:.0f} queries/s")
    print()

    # Measure GPU→CPU transfer separately
    print("Measuring GPU→CPU transfer...")
    times_transfer = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_images = images.cpu().numpy()
        elapsed = time.perf_counter() - t0
        times_transfer.append(elapsed * 1000)

    print(f"  GPU→CPU:     {np.median(times_transfer):.2f}ms median")
    print(f"  Data size:   {images.numel() * images.element_size() / 1e6:.1f} MB")
    print()

    # Measure render WITHOUT transfer
    render_only = np.median(times_full_ms)
    transfer_only = np.median(times_transfer)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Interop:       {args.interop}")
    print(f"  Batch:         {batch} queries @ {width}x{height}")
    print(f"  Full render:   {np.median(times_full_ms):.2f}ms")
    print(f"  GPU→CPU:       {transfer_only:.2f}ms")
    print(f"  Render only:   {render_only:.2f}ms (includes upload+sync+draw+readback)")
    print(f"  Throughput:    {batch/np.median(times_full)*1000:.0f} queries/s")
    fps = batch / np.median(times_full)
    print(f"  Effective FPS: {fps:.1f}")

    pixel_throughput = batch * width * height / np.median(times_full) / 1e6
    print(f"  Pixel rate:    {pixel_throughput:.0f} Mpix/s")
    print()

    # Run multiple batch sizes to see scaling
    if args.sweep:
        print("\n=== Batch Size Sweep ===")
        print(f"{'Batch':>6} {'Render(ms)':>11} {'Per-query(ms)':>14} {'Queries/s':>10}")
        for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if b > n_ts:
                break
            sids = torch.zeros(b, dtype=torch.int32, device=device)
            cids = torch.zeros(b, dtype=torch.int32, device=device)
            ctids = torch.zeros(b, dtype=torch.int32, device=device)
            ts_b = timestamps[:b]
            p_b = all_poses[:b]

            # warmup
            for _ in range(3):
                ctx.render(sids, cids, ts_b, ctids, p_b, resolution=resolution)
            torch.cuda.synchronize()

            sweep_times = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ctx.render(sids, cids, ts_b, ctids, p_b, resolution=resolution)
                torch.cuda.synchronize()
                sweep_times.append((time.perf_counter() - t0) * 1000)

            med = np.median(sweep_times)
            print(f"{b:>6} {med:>11.2f} {med/b:>14.3f} {b/med*1000:>10.0f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark interop pipeline phases')
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--interop', type=str, default='classic', choices=['classic', 'extmem'])
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--sweep', action='store_true',
                        help='Run batch size sweep (1, 2, 4, 8, ... 256)')
    args = parser.parse_args()
    benchmark(args)


if __name__ == '__main__':
    main()
