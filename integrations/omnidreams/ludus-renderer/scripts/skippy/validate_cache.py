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

"""Validate all cached .pt scene files by attempting to load each one.

Reports pass/fail counts and lists any corrupt files.

Usage:
    python scripts/skippy/validate_cache.py \
        --cache-dir /lustre/.../ludus_renderer_cache

    # With more workers:
    python scripts/skippy/validate_cache.py \
        --cache-dir /lustre/.../ludus_renderer_cache --workers 64
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _validate_one(pt_path_str: str) -> tuple[str, bool, str]:
    """Try loading a single .pt file. Returns (path, ok, error_msg)."""
    from ludus_renderer.scene_cache import load_scene_from_disk

    pt_path = Path(pt_path_str)
    try:
        load_scene_from_disk(pt_path)
        return pt_path_str, True, ""
    except Exception as e:
        return pt_path_str, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate cached .pt scene files")
    default_cache_dir = (
        f"/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos"
        f"/users/{os.environ['USER']}/ludus_renderer_cache"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=default_cache_dir,
        help="Cache root directory",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    args = parser.parse_args()

    from ludus_renderer.scene_cache import _resolve_cache_dir

    versioned_dir = _resolve_cache_dir(args.cache_dir)
    pt_files = sorted(str(p) for p in versioned_dir.rglob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {versioned_dir}")
        sys.exit(1)

    workers = args.workers or os.cpu_count()
    print(f"Validating {len(pt_files)} cached scenes from {versioned_dir}")
    print(f"Workers: {workers}")

    passed = 0
    failed = 0
    failures = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_validate_one, p): p for p in pt_files}
        for fut in as_completed(futures):
            path, ok, err = fut.result()
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((path, err))

            done = passed + failed
            if done % 500 == 0 or done == len(pt_files):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(pt_files)}] {passed} ok, {failed} bad "
                      f"({rate:.0f} files/s)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failures:
        print(f"\nCorrupt files ({len(failures)}):")
        for path, err in failures:
            print(f"  {path}: {err}")
        sys.exit(1)
    else:
        print("\nAll files valid.")


if __name__ == "__main__":
    main()
