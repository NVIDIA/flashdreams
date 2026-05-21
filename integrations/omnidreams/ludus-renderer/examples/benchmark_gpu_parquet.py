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

"""Benchmark and validate GPU-native parquet decoder vs PyArrow path.

Usage (on GPU node):
    cd /home/jseo/nv/ludus-renderer && uv run python examples/benchmark_gpu_parquet.py

Runs:
  1. Single-scene correctness: compares FlatPolylineData from both paths
  2. Single-scene timing: measures per-step timings for both paths
  3. Multi-scene correctness: validates GPU vs PyArrow across many scenes
  4. Phase breakdown: per-phase timing of load_av2_scene
  5. Full scene timing: GPU-native vs PyArrow (20 runs, single scene)
  6. Batch decoder: GpuParquetDecoder with variable batch sizes
  7. Multi-scene e2e timing: GPU vs PyArrow across 20 different scenes
"""

import os
import random
import time
import sys

import numpy as np
import torch

SCENE_LIST = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "example_data", "scene_paths_all.txt"
)

TAR_PATH = (
    "/lustre/fsw/portfolios/av/projects/av_mlops_alpamayo/dataset/data_repo/"
    "production/0003/00030bff-8007-4806-a258-60fb659604cf/clipgt.2.2.2-8b99c2.tar"
)


def compare_flat(name: str, gpu_data, pyarrow_data) -> bool:
    """Compare FlatPolylineData from GPU-native vs PyArrow path."""
    if gpu_data is None and pyarrow_data is None:
        print(f"  {name}: both None -- OK")
        return True
    if gpu_data is None or pyarrow_data is None:
        print(f"  {name}: MISMATCH -- one is None (gpu={gpu_data is not None}, pyarrow={pyarrow_data is not None})")
        return False

    g_ts = gpu_data.timestamps_us.cpu()
    p_ts = pyarrow_data.timestamps_us.cpu()
    g_v = gpu_data.vertices.cpu()
    p_v = pyarrow_data.vertices.cpu()
    g_o = gpu_data.row_offsets.cpu()
    p_o = pyarrow_data.row_offsets.cpu()

    ok = True

    if g_ts.shape != p_ts.shape:
        print(f"  {name}: timestamps shape mismatch: {g_ts.shape} vs {p_ts.shape}")
        ok = False
    elif not torch.equal(g_ts, p_ts):
        diff_count = (g_ts != p_ts).sum().item()
        print(f"  {name}: timestamps differ in {diff_count}/{len(g_ts)} values")
        ok = False

    if g_v.shape != p_v.shape:
        print(f"  {name}: vertices shape mismatch: {g_v.shape} vs {p_v.shape}")
        ok = False
    else:
        max_diff = (g_v.float() - p_v.float()).abs().max().item()
        if max_diff > 1e-6:
            print(f"  {name}: vertices max diff = {max_diff}")
            ok = False

    if g_o.shape != p_o.shape:
        print(f"  {name}: row_offsets shape mismatch: {g_o.shape} vs {p_o.shape}")
        ok = False
    elif not torch.equal(g_o, p_o):
        print(f"  {name}: row_offsets differ")
        ok = False

    if ok:
        print(f"  {name}: MATCH ({g_v.shape[0]} verts, {len(g_ts)} rows)")
    return ok


def benchmark_single_scene():
    """Compare GPU-native vs PyArrow for a single scene."""
    from ludus_renderer.clipgt import load_av2_scene, _load_polylines_pyarrow, get_file_loader
    from ludus_renderer.gpu_parquet import is_gpu_parquet_available, load_polylines_gpu_native

    device = torch.device("cuda")

    if not is_gpu_parquet_available():
        print("ERROR: nvcomp not available, cannot benchmark GPU path")
        return False

    print("=" * 60)
    print("1. CORRECTNESS: GPU-native vs PyArrow (single scene)")
    print("=" * 60)

    # Load with PyArrow
    loader = get_file_loader(TAR_PATH)
    pyarrow_data = _load_polylines_pyarrow(loader)

    # Move to same device for comparison
    for k, v in pyarrow_data.items():
        if v is not None:
            pyarrow_data[k] = v.to(device)

    # Load with GPU-native
    gpu_data = load_polylines_gpu_native(TAR_PATH, device)

    all_ok = True
    for pq_name in ["cf_road_boundary.parquet", "dw_lane_line.parquet",
                     "cf_crosswalks.parquet", "cf_static_obstacle.parquet"]:
        ok = compare_flat(pq_name, gpu_data.get(pq_name), pyarrow_data.get(pq_name))
        all_ok = all_ok and ok

    print(f"\nOverall: {'ALL MATCH' if all_ok else 'MISMATCH DETECTED'}")

    print()
    print("=" * 60)
    print("2. TIMING: single scene load")
    print("=" * 60)

    n_warmup = 2
    n_runs = 10

    # Warmup
    for _ in range(n_warmup):
        load_polylines_gpu_native(TAR_PATH, device)
        torch.cuda.synchronize()

    # Time GPU-native
    gpu_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        load_polylines_gpu_native(TAR_PATH, device)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - t0)

    # Warmup PyArrow
    for _ in range(n_warmup):
        loader = get_file_loader(TAR_PATH)
        _load_polylines_pyarrow(loader)

    # Time PyArrow
    pa_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        loader = get_file_loader(TAR_PATH)
        _load_polylines_pyarrow(loader)
        pa_times.append(time.perf_counter() - t0)

    gpu_mean = np.mean(gpu_times) * 1000
    gpu_std = np.std(gpu_times) * 1000
    pa_mean = np.mean(pa_times) * 1000
    pa_std = np.std(pa_times) * 1000

    print(f"  GPU-native: {gpu_mean:.1f} +/- {gpu_std:.1f} ms")
    print(f"  PyArrow:    {pa_mean:.1f} +/- {pa_std:.1f} ms")
    print(f"  Speedup:    {pa_mean / gpu_mean:.2f}x")

    return all_ok


def compare_full_scene():
    """Compare full load_av2_scene output between GPU (unified tar read) and PyArrow."""
    from ludus_renderer.clipgt import load_av2_scene

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print("3. CORRECTNESS: full load_av2_scene (GPU unified vs PyArrow)")
    print("=" * 60)

    gpu_scene = load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=True)
    pa_scene = load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=False)
    torch.cuda.synchronize()

    all_ok = True

    # --- Ego track ---
    g_ego, p_ego = gpu_scene.ego_track, pa_scene.ego_track
    ts_ok = torch.equal(g_ego.timestamps.cpu(), p_ego.timestamps.cpu())
    pose_diff = (g_ego.poses_tquat.cpu().float() - p_ego.poses_tquat.cpu().float()).abs().max().item()
    ego_ok = ts_ok and pose_diff < 1e-6
    print(f"  ego_track timestamps: {'MATCH' if ts_ok else 'MISMATCH'} ({len(g_ego.timestamps)} frames)")
    print(f"  ego_track poses:      {'MATCH' if pose_diff < 1e-6 else 'MISMATCH'} (max diff {pose_diff:.2e})")
    all_ok = all_ok and ego_ok

    # --- Cameras (FThetaCamera: principal_point, image_size, fw_poly) ---
    # fw_poly tolerance is relaxed: GPU uses C++ normal-equations solver vs
    # PyArrow reference using numpy SVD-based polyfit.
    g_cams, p_cams = gpu_scene.cameras, pa_scene.cameras
    cam_ok = len(g_cams) == len(p_cams)
    max_fw_diff = 0.0
    if cam_ok:
        for gc, pc in zip(g_cams, p_cams):
            pp_diff = (gc.principal_point.cpu() - pc.principal_point.cpu()).abs().max().item()
            sz_diff = (gc.image_size.cpu() - pc.image_size.cpu()).abs().max().item()
            fw_diff = (gc.fw_poly.cpu() - pc.fw_poly.cpu()).abs().max().item()
            max_fw_diff = max(max_fw_diff, fw_diff)
            if pp_diff > 1e-6 or sz_diff > 1e-6 or fw_diff > 1e-3:
                cam_ok = False
                break
    fw_note = f", max fw_poly diff={max_fw_diff:.2e}" if max_fw_diff > 0 else ""
    print(f"  cameras:              {'MATCH' if cam_ok else 'MISMATCH'} ({len(g_cams)} cameras{fw_note})")
    all_ok = all_ok and cam_ok

    # --- Timestamped scene (polyline pools, polygon pools, cube pools) ---
    g_ts = gpu_scene.timestamped_scene
    p_ts = pa_scene.timestamped_scene

    n_gpoly = len(g_ts.polyline_pools) if g_ts.polyline_pools else 0
    n_ppoly = len(p_ts.polyline_pools) if p_ts.polyline_pools else 0
    poly_ok = n_gpoly == n_ppoly
    print(f"  polyline pools:       {'MATCH' if poly_ok else 'MISMATCH'} ({n_gpoly} vs {n_ppoly})")
    all_ok = all_ok and poly_ok

    n_gpgon = len(g_ts.polygon_pools) if g_ts.polygon_pools else 0
    n_ppgon = len(p_ts.polygon_pools) if p_ts.polygon_pools else 0
    pgon_ok = n_gpgon == n_ppgon
    print(f"  polygon pools:        {'MATCH' if pgon_ok else 'MISMATCH'} ({n_gpgon} vs {n_ppgon})")
    all_ok = all_ok and pgon_ok

    n_gcube = len(g_ts.cube_pools) if g_ts.cube_pools else 0
    n_pcube = len(p_ts.cube_pools) if p_ts.cube_pools else 0
    cube_ok = n_gcube == n_pcube
    print(f"  cube pools:           {'MATCH' if cube_ok else 'MISMATCH'} ({n_gcube} vs {n_pcube})")
    all_ok = all_ok and cube_ok

    print(f"\n  Full scene: {'ALL MATCH' if all_ok else 'MISMATCH DETECTED'}")
    return all_ok


def _sample_tar_paths(n: int, seed: int = 42) -> list:
    """Sample n tar paths from the pre-built scene list file.

    Uses reservoir sampling to avoid loading the entire 23M-line file.
    """
    rng = random.Random(seed)
    reservoir = []
    with open(SCENE_LIST) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    rng.shuffle(reservoir)
    return reservoir


def compare_full_scene_one(tar_path: str, device: torch.device, label: str = "") -> bool:
    """Compare GPU vs PyArrow load_av2_scene for a single tar. Returns True if match."""
    from ludus_renderer.clipgt import load_av2_scene

    gpu_scene = load_av2_scene(tar_path, device=device, verbose=False, use_gpu_decoder=True)
    torch.cuda.synchronize()
    pa_scene = load_av2_scene(tar_path, device=device, verbose=False, use_gpu_decoder=False)
    torch.cuda.synchronize()

    ok = True
    issues = []

    g_ego, p_ego = gpu_scene.ego_track, pa_scene.ego_track
    if not torch.equal(g_ego.timestamps.cpu(), p_ego.timestamps.cpu()):
        issues.append("ego_timestamps")
        ok = False
    pose_diff = (g_ego.poses_tquat.cpu().float() - p_ego.poses_tquat.cpu().float()).abs().max().item()
    if pose_diff > 1e-6:
        issues.append(f"ego_poses(diff={pose_diff:.2e})")
        ok = False

    g_cams, p_cams = gpu_scene.cameras, pa_scene.cameras
    if len(g_cams) != len(p_cams):
        issues.append(f"cam_count({len(g_cams)}vs{len(p_cams)})")
        ok = False
    else:
        for gc, pc in zip(g_cams, p_cams):
            pp_diff = (gc.principal_point.cpu() - pc.principal_point.cpu()).abs().max().item()
            sz_diff = (gc.image_size.cpu() - pc.image_size.cpu()).abs().max().item()
            fw_diff = (gc.fw_poly.cpu() - pc.fw_poly.cpu()).abs().max().item()
            if pp_diff > 1e-6 or sz_diff > 1e-6 or fw_diff > 1e-3:
                issues.append("cam_params")
                ok = False
                break

    g_ts = gpu_scene.timestamped_scene
    p_ts = pa_scene.timestamped_scene
    n_gp = len(g_ts.polyline_pools) if g_ts.polyline_pools else 0
    n_pp = len(p_ts.polyline_pools) if p_ts.polyline_pools else 0
    if n_gp != n_pp:
        issues.append(f"polyline_pools({n_gp}vs{n_pp})")
        ok = False
    n_gg = len(g_ts.polygon_pools) if g_ts.polygon_pools else 0
    n_pg = len(p_ts.polygon_pools) if p_ts.polygon_pools else 0
    if n_gg != n_pg:
        issues.append(f"polygon_pools({n_gg}vs{n_pg})")
        ok = False
    n_gc = len(g_ts.cube_pools) if g_ts.cube_pools else 0
    n_pc = len(p_ts.cube_pools) if p_ts.cube_pools else 0
    if n_gc != n_pc:
        issues.append(f"cube_pools({n_gc}vs{n_pc})")
        ok = False

    scene_id = os.path.basename(os.path.dirname(tar_path))
    status = "PASS" if ok else f"FAIL [{', '.join(issues)}]"
    print(f"  {label}{scene_id}: {status}")
    return ok


def validate_multi_scene(n_scenes: int = 50):
    """Validate GPU vs PyArrow across many scenes from different partitions."""
    if n_scenes <= 0:
        print("\n" + "=" * 60)
        print("3b. MULTI-SCENE CORRECTNESS (skipped, n_scenes=0)")
        print("=" * 60)
        return True

    from ludus_renderer.clipgt import load_av2_scene

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print(f"3b. MULTI-SCENE CORRECTNESS ({n_scenes} scenes)")
    print("=" * 60)

    # Warmup
    load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=True)
    load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=False)
    torch.cuda.synchronize()

    print(f"  Sampling {n_scenes} scenes from {SCENE_LIST}...")
    paths = _sample_tar_paths(n_scenes)
    print(f"  Sampled {len(paths)} scenes, testing each GPU vs PyArrow...")

    n_pass = 0
    n_fail = 0
    n_skip = 0
    t_start = time.perf_counter()
    for i, tar in enumerate(paths):
        if not os.path.isfile(tar):
            scene_id = os.path.basename(os.path.dirname(tar))
            print(f"  [{i+1:3d}/{len(paths)}] {scene_id}: SKIP (file not found)")
            n_skip += 1
            continue
        try:
            ok = compare_full_scene_one(tar, device, label=f"[{i+1:3d}/{len(paths)}] ")
        except Exception as e:
            import traceback
            scene_id = os.path.basename(os.path.dirname(tar))
            print(f"  [{i+1:3d}/{len(paths)}] {scene_id}: ERROR -- {type(e).__name__}: {e}")
            traceback.print_exc()
            ok = False
        elapsed = time.perf_counter() - t_start
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        if (i + 1) % 10 == 0:
            print(f"  --- {i+1}/{len(paths)} done, {elapsed:.1f}s elapsed, "
                  f"{n_pass} pass / {n_fail} fail / {n_skip} skip ---")

    print(f"\n  Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped out of {len(paths)}")
    return n_fail == 0


def benchmark_full_scene():
    """Benchmark full load_av2_scene (including egomotion, obstacles, pools)."""
    from ludus_renderer.clipgt import load_av2_scene

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print("4. PHASE BREAKDOWN: load_av2_scene GPU path (single run)")
    print("=" * 60)

    # Warmup
    load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=True)
    torch.cuda.synchronize()
    load_av2_scene(TAR_PATH, device=device, verbose=True, use_gpu_decoder=True)
    torch.cuda.synchronize()

    print()
    print("=" * 60)
    print("5. TIMING: full load_av2_scene (GPU-native vs PyArrow)")
    print("=" * 60)

    n_warmup = 3
    n_runs = 20

    for label, use_gpu in [("GPU-native", True), ("PyArrow", False)]:
        for _ in range(n_warmup):
            load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=use_gpu)
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            load_av2_scene(TAR_PATH, device=device, verbose=False, use_gpu_decoder=use_gpu)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean = np.mean(times) * 1000
        std = np.std(times) * 1000
        median = np.median(times) * 1000
        p5 = np.percentile(times, 5) * 1000
        p95 = np.percentile(times, 95) * 1000
        print(f"  {label:12s}: mean {mean:.1f} +/- {std:.1f} ms, "
              f"median {median:.1f}, p5 {p5:.1f}, p95 {p95:.1f}")


def benchmark_multi_scene_e2e(n_scenes: int = 20):
    """Benchmark load_av2_scene across many different scenes to average out noise."""
    from ludus_renderer.clipgt import load_av2_scene

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print(f"7. MULTI-SCENE E2E TIMING ({n_scenes} scenes, GPU vs PyArrow)")
    print("=" * 60)

    paths = _sample_tar_paths(n_scenes, seed=123)
    valid_paths = [p for p in paths if os.path.isfile(p)]
    if len(valid_paths) < n_scenes:
        print(f"  WARNING: only {len(valid_paths)}/{n_scenes} scene files found")
    if not valid_paths:
        print("  SKIPPED: no valid scene files")
        return

    # Warmup: run every scene once with both paths to prime FS cache + CUDA
    print(f"  Warmup: loading {len(valid_paths)} scenes (both paths) to prime caches...")
    for tar in valid_paths:
        try:
            load_av2_scene(tar, device=device, verbose=False, use_gpu_decoder=True)
            torch.cuda.synchronize()
            load_av2_scene(tar, device=device, verbose=False, use_gpu_decoder=False)
            torch.cuda.synchronize()
        except Exception:
            pass
    print("  Warmup done.\n")

    gpu_times = []
    pa_times = []

    for i, tar in enumerate(valid_paths):
        scene_id = os.path.basename(os.path.dirname(tar))
        try:
            # Alternate order: even scenes GPU-first, odd scenes PyArrow-first
            times_pair = {}
            order = [("GPU", True), ("PyArrow", False)]
            if i % 2 == 1:
                order = order[::-1]

            for label, use_gpu in order:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                load_av2_scene(tar, device=device, verbose=False, use_gpu_decoder=use_gpu)
                torch.cuda.synchronize()
                times_pair[label] = time.perf_counter() - t0

            gt = times_pair["GPU"]
            pt = times_pair["PyArrow"]
            gpu_times.append(gt)
            pa_times.append(pt)
            print(f"  [{i+1:3d}/{len(valid_paths)}] {scene_id}: "
                  f"GPU {gt*1000:6.1f} ms, PyArrow {pt*1000:6.1f} ms, "
                  f"speedup {pt/gt:.2f}x")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(valid_paths)}] {scene_id}: ERROR -- {e}")

    if gpu_times:
        gpu_arr = np.array(gpu_times) * 1000
        pa_arr = np.array(pa_times) * 1000
        speedups = np.array(pa_times) / np.array(gpu_times)
        print()
        print(f"  {'':12s}  {'mean':>7s}  {'std':>6s}  {'median':>7s}  {'p5':>6s}  {'p95':>6s}")
        print(f"  {'GPU-native':12s}: {gpu_arr.mean():7.1f}  {gpu_arr.std():6.1f}  "
              f"{np.median(gpu_arr):7.1f}  {np.percentile(gpu_arr,5):6.1f}  {np.percentile(gpu_arr,95):6.1f}")
        print(f"  {'PyArrow':12s}: {pa_arr.mean():7.1f}  {pa_arr.std():6.1f}  "
              f"{np.median(pa_arr):7.1f}  {np.percentile(pa_arr,5):6.1f}  {np.percentile(pa_arr,95):6.1f}")
        print(f"  Speedup:     mean {speedups.mean():.2f}x, median {np.median(speedups):.2f}x")


def benchmark_prefetch(n_scenes: int = 20):
    """Benchmark load_av2_scene with and without prefetch across different scenes.

    Measures both per-scene latency and total throughput. The prefetch benefit
    comes from overlapping I/O for scene N+1 with post-C++ processing of scene N,
    so we use `prefetch_next` (integrated into load_av2_scene) and also add a
    synthetic processing delay to simulate real workloads.
    """
    from ludus_renderer.clipgt import load_av2_scene, prefetch_scene

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print(f"8. PREFETCH BENCHMARK ({n_scenes} scenes, sequential)")
    print("=" * 60)

    paths = _sample_tar_paths(n_scenes, seed=456)
    valid_paths = [p for p in paths if os.path.isfile(p)]
    if len(valid_paths) < 2:
        print("  SKIPPED: need at least 2 valid scene files")
        return
    print(f"  Found {len(valid_paths)} scenes")

    for tar in valid_paths:
        try:
            load_av2_scene(tar, device=device, verbose=False, use_gpu_decoder=True)
            torch.cuda.synchronize()
        except Exception:
            pass

    n_runs = 10
    work_ms = 100

    # Collect per-scene load times across runs: scene_idx -> list of times
    per_scene_no_pf: dict = {i: [] for i in range(len(valid_paths))}
    per_scene_pf: dict = {i: [] for i in range(len(valid_paths))}

    for _ in range(n_runs):
        # Without prefetch
        torch.cuda.synchronize()
        for si, tar in enumerate(valid_paths):
            tl = time.perf_counter()
            load_av2_scene(tar, device=device, verbose=False, use_gpu_decoder=True)
            torch.cuda.synchronize()
            per_scene_no_pf[si].append(time.perf_counter() - tl)
            time.sleep(work_ms / 1000.0)

        # With prefetch_next
        torch.cuda.synchronize()
        prefetch_scene(valid_paths[0])
        for si, tar in enumerate(valid_paths):
            pf_next = valid_paths[si + 1] if si + 1 < len(valid_paths) else None
            tl = time.perf_counter()
            load_av2_scene(tar, device=device, verbose=False,
                           use_gpu_decoder=True, prefetch_next=pf_next)
            torch.cuda.synchronize()
            per_scene_pf[si].append(time.perf_counter() - tl)
            time.sleep(work_ms / 1000.0)

    # Per-scene results
    print(f"\n  {'scene':>4s}  {'no-pf median':>12s}  {'prefetch median':>15s}  {'saved':>7s}  id")
    all_no_pf = []
    all_pf = []
    for si in range(len(valid_paths)):
        no = np.median(per_scene_no_pf[si]) * 1000
        pf = np.median(per_scene_pf[si]) * 1000
        sv = no - pf
        scene_id = os.path.basename(os.path.dirname(valid_paths[si]))
        print(f"  {si+1:>4d}  {no:>10.2f} ms  {pf:>13.2f} ms  {sv:>+6.2f}  {scene_id}")
        all_no_pf.extend(per_scene_no_pf[si])
        all_pf.extend(per_scene_pf[si])

    no_arr = np.array(all_no_pf) * 1000
    pf_arr = np.array(all_pf) * 1000
    print(f"\n  {'':16s}  {'mean':>7s}  {'std':>6s}  {'median':>7s}  {'p5':>6s}  {'p95':>6s}")
    print(f"  {'No prefetch':16s}: {no_arr.mean():7.1f}  {no_arr.std():6.1f}  "
          f"{np.median(no_arr):7.1f}  {np.percentile(no_arr,5):6.1f}  {np.percentile(no_arr,95):6.1f}")
    print(f"  {'With prefetch':16s}: {pf_arr.mean():7.1f}  {pf_arr.std():6.1f}  "
          f"{np.median(pf_arr):7.1f}  {np.percentile(pf_arr,5):6.1f}  {np.percentile(pf_arr,95):6.1f}")
    print(f"  Prefetch saves: {np.median(no_arr) - np.median(pf_arr):.2f} ms (median)")


def benchmark_decoder_class():
    """Benchmark GpuParquetDecoder with variable batch sizes."""
    from ludus_renderer.gpu_parquet import GpuParquetDecoder

    device = torch.device("cuda")

    print()
    print("=" * 60)
    print("6. GpuParquetDecoder: pre-allocated, variable batch size")
    print("=" * 60)

    decoder = GpuParquetDecoder(device)

    # Warmup (includes CUDA JIT, buffer sizing)
    _ = decoder.load_scenes([TAR_PATH])
    _ = decoder.load_scenes([TAR_PATH])
    torch.cuda.synchronize()

    # Variable batch sizes
    for batch_size in [1, 2, 4, 8]:
        paths = [TAR_PATH] * batch_size

        n_runs = 5
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = decoder.load_scenes(paths)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean = np.mean(times) * 1000
        std = np.std(times) * 1000
        per_scene = mean / batch_size
        print(f"  batch={batch_size:2d}: {mean:7.1f} +/- {std:4.1f} ms  "
              f"({per_scene:.1f} ms/scene)")

    # Correctness: compare decoder output vs standalone
    from ludus_renderer.gpu_parquet import load_polylines_gpu_native
    dec_result = decoder.load_scene(TAR_PATH)
    ref_result = load_polylines_gpu_native(TAR_PATH, device)

    all_ok = True
    for pq_name in ["cf_road_boundary.parquet", "dw_lane_line.parquet",
                     "cf_crosswalks.parquet", "cf_static_obstacle.parquet"]:
        ok = compare_flat(f"[decoder] {pq_name}", dec_result.get(pq_name), ref_result.get(pq_name))
        all_ok = all_ok and ok
    print(f"\n  Decoder correctness: {'ALL MATCH' if all_ok else 'MISMATCH'}")


if __name__ == "__main__":
    n_multi = 50
    if len(sys.argv) > 1:
        n_multi = int(sys.argv[1])

    ok = benchmark_single_scene()
    ok2 = compare_full_scene()
    ok3 = validate_multi_scene(n_multi)
    benchmark_full_scene()
    benchmark_multi_scene_e2e(n_scenes=20)
    benchmark_prefetch(n_scenes=20)
    benchmark_decoder_class()
    all_ok = ok and ok2 and ok3
    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)
