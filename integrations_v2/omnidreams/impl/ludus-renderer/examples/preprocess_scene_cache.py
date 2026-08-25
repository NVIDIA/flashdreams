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

"""Offline preprocessing: convert AV2 tar scenes to L3 cache (LMDB or per-file).

Workers load and serialize scenes in parallel; the main thread writes to LMDB
(single-writer constraint). Falls back to per-file .pt cache if lmdb is not
installed.

Example — preprocess all 10k scenes with 32 workers:

    python examples/preprocess_scene_cache.py \
        --scene-list example_data/scene_paths_10k.txt \
        --cache-dir /fast/ludus_cache \
        --workers 32

Resume after interruption (skips already-cached scenes):

    python examples/preprocess_scene_cache.py \
        --scene-list example_data/scene_paths_10k.txt \
        --cache-dir /fast/ludus_cache \
        --workers 32 --skip-existing
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _load_and_serialize(args_tuple):
    """Worker: load scene from tar, return (key, serialized_bytes, elapsed)."""
    (tar_path,) = args_tuple

    from ludus_renderer import load_scene
    from ludus_renderer.scene_cache import _serialize_to_bytes, scene_key

    key = scene_key(tar_path)
    t0 = time.perf_counter()
    scene = load_scene(tar_path, device="cpu")
    data = _serialize_to_bytes(scene)
    elapsed = time.perf_counter() - t0
    return key, data, elapsed


def _load_and_save_file(args_tuple):
    """Worker: load scene and save as per-file .pt (fallback mode)."""
    tar_path, cache_dir, skip_existing = args_tuple

    from ludus_renderer import load_scene
    from ludus_renderer.scene_cache import (
        _cache_path,
        _resolve_cache_dir,
        load_scene_from_disk,
        save_scene_to_disk,
        scene_key,
    )

    key = scene_key(tar_path)
    versioned_dir = _resolve_cache_dir(cache_dir)
    out_path = _cache_path(versioned_dir, key)

    if skip_existing and out_path.exists():
        try:
            load_scene_from_disk(out_path)
            return key, 0.0, "skipped"
        except Exception:
            pass

    t0 = time.perf_counter()
    scene = load_scene(tar_path, device="cpu")
    save_scene_to_disk(scene, out_path)
    elapsed = time.perf_counter() - t0

    file_size = out_path.stat().st_size
    return key, elapsed, file_size


def main():
    parser = argparse.ArgumentParser(description="Preprocess AV2 scenes to disk cache")
    parser.add_argument(
        "--scene-list",
        type=str,
        required=True,
        help="Text file with scene tar paths, one per line",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Output directory for cached scenes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that are already cached",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N scenes"
    )
    parser.add_argument(
        "--no-lmdb",
        action="store_true",
        help="Force per-file .pt cache instead of LMDB",
    )
    args = parser.parse_args()

    all_paths = [
        l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()
    ]
    if args.limit:
        all_paths = all_paths[: args.limit]

    workers = args.workers or os.cpu_count()
    cache_dir = args.cache_dir

    try:
        import lmdb as _lmdb  # ty:ignore[unresolved-import]

        use_lmdb = not args.no_lmdb
    except ImportError:
        use_lmdb = False

    mode = "LMDB" if use_lmdb else "per-file .pt"
    print(f"Preprocessing {len(all_paths)} scenes → {cache_dir} ({mode})")
    print(f"Workers: {workers}, skip_existing: {args.skip_existing}")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if use_lmdb:
        _preprocess_lmdb(all_paths, cache_dir, workers, args.skip_existing)
    else:
        _preprocess_files(all_paths, cache_dir, workers, args.skip_existing)


def _preprocess_lmdb(all_paths, cache_dir, workers, skip_existing):
    from ludus_renderer.scene_cache import (
        LMDBSceneStore,
        _lmdb_path,
        _resolve_cache_dir,
        scene_key,
    )

    versioned_dir = _resolve_cache_dir(cache_dir)
    store = LMDBSceneStore(_lmdb_path(versioned_dir))

    if skip_existing:
        existing = set()
        for p in all_paths:
            k = scene_key(p)
            if store.contains(k):
                existing.add(k)
        to_process = [(p,) for p in all_paths if scene_key(p) not in existing]
        print(f"  Skipping {len(existing)} already-cached scenes")
    else:
        to_process = [(p,) for p in all_paths]

    done = 0
    failed = 0
    total_bytes = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load_and_serialize, t): t[0] for t in to_process}

        for fut in as_completed(futures):
            path = futures[fut]
            done += 1
            try:
                key, data, elapsed = fut.result()
                store.put_bytes(key, data)
                total_bytes += len(data)
                status = f"{len(data) / 1024:.0f}KB {elapsed:.1f}s"
            except Exception as e:
                failed += 1
                status = f"FAIL: {e}"

            if done % 100 == 0 or done == len(to_process):
                elapsed_total = time.perf_counter() - t0
                rate = (done - failed) / elapsed_total if elapsed_total > 0 else 0
                print(
                    f"  [{done}/{len(to_process)}] {status}  "
                    f"({rate:.1f} scenes/s, {failed} failed)"
                )

    store.close()

    elapsed_total = time.perf_counter() - t0
    processed = done - failed
    print(f"\nDone in {elapsed_total:.1f}s")
    print(f"  Processed: {processed}, Failed: {failed}")
    if processed > 0:
        print(
            f"  Total data: {total_bytes / 1024**2:.1f} MB ({total_bytes / processed / 1024:.1f} KB/scene avg)"
        )
        print(f"  Throughput: {processed / elapsed_total:.1f} scenes/s")


def _preprocess_files(all_paths, cache_dir, workers, skip_existing):
    tasks = [(p, cache_dir, skip_existing) for p in all_paths]

    done = 0
    skipped = 0
    failed = 0
    total_bytes = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load_and_save_file, t): t[0] for t in tasks}

        for fut in as_completed(futures):
            path = futures[fut]
            done += 1
            try:
                key, elapsed, result = fut.result()
                if result == "skipped":
                    skipped += 1
                    status = "skip"
                else:
                    total_bytes += result
                    status = f"{result / 1024:.0f}KB {elapsed:.1f}s"
            except Exception as e:
                failed += 1
                status = f"FAIL: {e}"

            if done % 100 == 0 or done == len(all_paths):
                elapsed_total = time.perf_counter() - t0
                rate = (
                    (done - skipped - failed) / elapsed_total
                    if elapsed_total > 0
                    else 0
                )
                print(
                    f"  [{done}/{len(all_paths)}] {status}  "
                    f"({rate:.1f} scenes/s, {skipped} skipped, {failed} failed)"
                )

    elapsed_total = time.perf_counter() - t0
    processed = done - skipped - failed
    print(f"\nDone in {elapsed_total:.1f}s")
    print(f"  Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    if processed > 0:
        print(
            f"  Cache size: {total_bytes / 1024**2:.1f} MB ({total_bytes / processed / 1024:.1f} KB/scene avg)"
        )
        print(f"  Throughput: {processed / elapsed_total:.1f} scenes/s")


if __name__ == "__main__":
    main()
