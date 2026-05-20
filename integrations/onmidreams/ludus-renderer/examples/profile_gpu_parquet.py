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

"""Profile GPU-native parquet decoder -- two-pass architecture.

Measures:
  1. Total e2e load_scene time (steady-state, post-warmup)
  2. Per-step breakdown: I/O, upload, metadata, nvcomp, scan+unpack, gather
  3. Per-file decode times
  4. Batch scaling (the key metric: should be sub-linear with batched kernels)

Usage (on GPU node):
    cd /home/jseo/nv/ludus-renderer && uv run python examples/profile_gpu_parquet.py
"""

import time
import torch
import numpy as np

TAR_PATH = (
    "/lustre/fsw/portfolios/av/projects/av_mlops_alpamayo/dataset/data_repo/"
    "production/0003/00030bff-8007-4806-a258-60fb659604cf/clipgt.2.2.2-8b99c2.tar"
)


def profile_step(label, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {label:50s}: {dt:7.2f} ms")
    return result


def main():
    from ludus_renderer.gpu_parquet import (
        read_tar_to_pinned_buffer, scan_parquet_pages, batch_decompress_pages,
        decode_data_pages_gpu, rep_levels_to_offsets_gpu,
        POLYLINE_SPECS, PageInfo, GpuParquetDecoder,
    )

    device = torch.device("cuda")

    # ==== Warmup (JIT, CUDA context, caching allocator) ====
    print("Warming up (CUDA JIT + context)...")
    decoder = GpuParquetDecoder(device)
    for _ in range(3):
        decoder.load_scene(TAR_PATH)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # ==== 1. Total e2e time ====
    print("=" * 70)
    print("1. Total e2e time (steady-state)")
    print("=" * 70)

    n_runs = 20
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decoder.load_scene(TAR_PATH)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    print(f"  load_scene (decoder):  {np.mean(times):.2f} +/- {np.std(times):.2f} ms  "
          f"(min={min(times):.2f}, max={max(times):.2f})")

    # ==== 2. Per-step breakdown ====
    print(f"\n{'='*70}")
    print("2. Per-step breakdown (single file, post-warmup)")
    print("   NOTE: forced sync between steps inflates sum vs pipelined total")
    print(f"{'='*70}")

    pinned, entries = profile_step(
        "a. read_tar_to_pinned_buffer (CPU I/O)",
        lambda: read_tar_to_pinned_buffer(TAR_PATH),
    )
    print(f"     tar: {len(pinned)/1024/1024:.2f} MB, {len(entries)} entries")

    gpu_buffer = profile_step(
        "b. pinned -> GPU upload",
        lambda: pinned.to(device, non_blocking=False),
    )

    entry_map = {}
    for e in entries:
        base = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
        entry_map[base] = e

    pq_name = "cf_road_boundary.parquet"
    spec = POLYLINE_SPECS[pq_name]
    entry = entry_map[pq_name]
    pq_bytes = pinned[entry.offset:entry.offset + entry.size].numpy().tobytes()

    columns_needed = [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]
    page_index = profile_step(
        "c. scan_parquet_pages (CPU metadata)",
        lambda: scan_parquet_pages(pq_bytes, columns=columns_needed),
    )

    all_pages = []
    page_labels = []
    for col_path in columns_needed:
        col = page_index.columns[col_path]
        if col.dict_page is not None:
            all_pages.append(PageInfo(
                page_type=col.dict_page.page_type,
                data_offset=col.dict_page.data_offset + entry.offset,
                compressed_size=col.dict_page.compressed_size,
                uncompressed_size=col.dict_page.uncompressed_size,
            ))
            page_labels.append((col_path, "dict"))
        for i, dp in enumerate(col.data_pages):
            all_pages.append(PageInfo(
                page_type=dp.page_type,
                data_offset=dp.data_offset + entry.offset,
                compressed_size=dp.compressed_size,
                uncompressed_size=dp.uncompressed_size,
            ))
            page_labels.append((col_path, f"data_{i}"))

    decompressed = profile_step(
        "d. batch_decompress_pages (nvcomp GPU)",
        lambda: batch_decompress_pages(gpu_buffer, all_pages),
    )

    decomp_map = dict(zip(page_labels, decompressed))
    dp_tensors = [decomp_map[(cp, "data_0")] for cp in columns_needed]
    max_reps = [page_index.columns[cp].max_repetition_level for cp in columns_needed]
    max_defs = [page_index.columns[cp].max_definition_level for cp in columns_needed]
    num_vals = [page_index.columns[cp].num_values for cp in columns_needed]
    total_vals = sum(num_vals)

    out_rep, out_def, out_idx = profile_step(
        "e. decode_data_pages_gpu (CUDA SMEM kernel)",
        lambda: decode_data_pages_gpu(dp_tensors, max_reps, max_defs, num_vals, device),
    )
    print(f"     {len(dp_tensors)} data pages, {total_vals} values")

    dv = decomp_map[(spec.x_path, "dict")].view(torch.float32)
    xi = out_idx[:num_vals[0]].to(torch.int64)
    profile_step(
        "f. index_select (dict gather GPU)",
        lambda: torch.index_select(dv, 0, xi),
    )

    xr = out_rep[:num_vals[0]]
    profile_step(
        "g. rep_levels_to_offsets_gpu",
        lambda: rep_levels_to_offsets_gpu(xr),
    )

    # ==== 3. Batch scaling (the key test) ====
    print(f"\n{'='*70}")
    print("4. GpuParquetDecoder batch scaling (batched kernel launches)")
    print("   Expected: sub-linear scaling since GPU kernels are batched")
    print(f"{'='*70}")

    for bs in [1, 2, 4, 8, 16, 32]:
        paths = [TAR_PATH] * bs
        n_runs = 5
        batch_times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            decoder.load_scenes(paths)
            torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) * 1000)
        mean = np.mean(batch_times)
        per_scene = mean / bs
        speedup = times[0] / per_scene if per_scene > 0 else 0
        print(f"  batch={bs:2d}: {mean:8.1f} +/- {np.std(batch_times):5.1f} ms  "
              f"({per_scene:.1f} ms/scene, {speedup:.1f}x vs single)")


if __name__ == "__main__":
    main()
