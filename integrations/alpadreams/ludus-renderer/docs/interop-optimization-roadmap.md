# Rendering Pipeline Optimization Roadmap

## Benchmark Baseline (March 2025)

Measured on H100 80GB HBM3, 1280x720, classic `cudaGraphics*` interop.

| Batch | Render (ms) | Per-query (ms) | Queries/s |
|------:|------------:|---------------:|----------:|
| 1 | 6.10 | 6.10 | 164 |
| 2 | 7.07 | 3.53 | 283 |
| 4 | 8.92 | 2.23 | 449 |
| 8 | 12.52 | 1.56 | 639 |
| 16 | 19.93 | 1.25 | 803 |
| 32 | 34.35 | 1.07 | 932 |
| 64 | 67.22 | 1.05 | 952 |
| 128 | 135.30 | 1.06 | 946 |

**Key findings:**
- Per-query cost plateaus at **~1.05ms** for batch >= 32 → GPU-bound on draw calls
- Fixed overhead per batch call: **~5ms** (interop map/unmap + GL state setup)
- GPU→CPU transfer: **42ms for 32 frames (118 MB)** — larger than the render itself
- Throughput saturates at **~950 queries/s** regardless of batch size

### Where time is spent

```
Full pipeline for 32 queries at 1280x720 (34.35ms total):

  Fixed overhead (~5ms):
    ├── cudaGraphicsMapResources (SSBO buffers)     ~1-2ms
    ├── GL state setup (bind FBO, bind SSBOs)        ~1ms
    ├── cudaGraphicsMapResources (color texture)     ~1ms
    └── cudaGraphicsUnmapResources (cleanup)         ~1ms

  Per-query draw cost (~1.05ms × 32 = ~29ms):
    ├── Mesh shader dispatch (polylines)
    ├── Mesh shader dispatch (polygons)
    ├── Mesh shader dispatch (obstacles)
    └── MSAA resolve (if enabled)

  GPU→CPU transfer (42ms, measured separately):
    └── cudaMemcpy device→host (118 MB RGBA8)
```

The interop sync (map/unmap) is **<5% of total time**. Draw calls dominate.

---

## Phase 1: GL_EXT_memory_object + Vulkan Semaphores — DEPRIORITIZED

**Original plan:** Replace `cudaGraphics*` with CUDA VMM + `GL_EXT_memory_object`
for shared buffers, Vulkan timeline semaphores for sync.

**Status:** Implemented behind `InteropBackend` abstraction (`--interop extmem`).
Compilation and Vulkan device matching work. Blocked by:
- GL texture arrays require tiled memory layout; `GL_LINEAR_TILING_EXT` unsupported
  on H100/NVIDIA driver for `GL_TEXTURE_2D_ARRAY` → color texture must fall back
  to classic interop
- Vulkan ICD initialization (`libGLX_nvidia.so.0`) corrupts EGL display state →
  `eglGetCurrentDisplay()` returns `EGL_NO_DISPLAY` after `vkCreateInstance`

**Conclusion:** Benchmark shows interop sync is not the bottleneck (~5ms fixed
overhead vs 29ms draw + 42ms transfer). The engineering cost of working around
driver-level GL/Vulkan/EGL conflicts is not justified by the potential gain.

**Recommendation:** Keep the `InteropBackend` abstraction for future use, but
focus optimization effort on phases that target the actual bottlenecks.

---

## Phase 2: GPU→CPU Transfer Optimization — HIGH PRIORITY

The GPU→CPU transfer (42ms) exceeds the render time (34ms) for batch=32. This
is the single largest time sink.

### 2a: Pinned (page-locked) host memory

Allocate the output tensor in pinned memory so `cudaMemcpyAsync` can use DMA:

```python
output = torch.empty(..., pin_memory=True)
```

Expected improvement: 2-3× faster for large transfers.

### 2b: Async staging with double buffering

Overlap GPU→CPU transfer with the next batch's render:

```
Batch N:   Render(N) ──► Copy-to-staging(N) ──► DMA-to-host(N)
Batch N+1:               Render(N+1) ──────────► Copy-to-staging(N+1)
                              (overlapped)
```

The existing `ludusCopyBatchResultsToStaging` + `ludusGetStagingData` API already
supports this pattern but it's not used in the rendering scripts.

### 2c: Skip transfer entirely (GPU-resident pipeline)

If the consumer is a GPU model (training or inference), keep the rendered images
as GPU tensors. Eliminate the 42ms transfer completely.

---

## Phase 3: Draw Call Optimization — HIGH PRIORITY

The 1.05ms per-query floor at high batch sizes means the mesh shader dispatches
are the GPU bottleneck.

### 3a: Profile with nsys

```bash
nsys profile --trace=cuda,opengl \
  uv run python examples/benchmark_interop.py --scene ... --batch-size 64
```

Identify which shader pass (polyline, polygon, obstacle) dominates and whether
there are idle gaps between dispatches.

### 3b: Reduce overdraw / early-exit

The current dispatch model uses upper-bound task counts (e.g., `MAX_VARRAYS_PER_POOL
= 1000`). Many task shader invocations early-exit because there's no data.
Precomputing exact dispatch counts per scene could reduce wasted GPU work.

### 3c: Indirect draw with GPU-side count

Use `glDrawMeshTasksIndirectNV` with a GPU-generated count buffer. A small CUDA
kernel computes the exact dispatch count per primitive type per scene, avoiding
CPU round-trips and over-dispatch.

### 3d: Frustum culling in task shader

Add per-pool bounding box checks in the task shader to skip pools outside the
camera frustum. For cameras with limited FOV (non-BEV), this can eliminate
significant geometry.

---

## Phase 4: Async Frame Pipelining — MEDIUM PRIORITY

Use the existing double-buffered `bufferSets[2]` to overlap CUDA uploads and GL
rendering across consecutive frames. This eliminates the ~5ms fixed overhead
between batches.

```
Frame N:   Upload(setA) ──► Render(setA) ──► Readback(N)
Frame N+1:                  Upload(setB) ──► Render(setB) ──► Readback(N+1)
                   (overlapped)
```

With classic interop, this requires careful ordering of `cudaGraphicsMapResources`
calls. The `swapBufferSets` mechanism is already in place.

**Expected gain:** ~5ms per batch (the fixed overhead).

---

## Phase 5: Scene Upload Optimization — LOW PRIORITY

Scene upload is a one-time cost (~0.5s for small scenes, ~22s for large scenes).
Not a per-frame bottleneck.

Potential improvements:
- Parallel parquet decoding
- Streaming upload (start rendering before full scene is loaded)
- Scene caching / serialization

---

## Phase 6: Multi-Process / Multi-GPU — FUTURE

For training at scale, separate scene loading and rendering across processes or
GPUs. Relevant only after per-frame bottlenecks are addressed.

---

## Priority Summary

```
Impact vs Effort:

HIGH IMPACT, MODERATE EFFORT:
  Phase 2a: Pinned memory transfers        → saves ~20ms per batch
  Phase 2c: GPU-resident pipeline          → saves ~42ms per batch
  Phase 3a: nsys profiling                 → identifies draw bottleneck

HIGH IMPACT, HIGH EFFORT:
  Phase 3b-d: Draw call optimization       → reduces 1.05ms/query floor
  Phase 2b: Async staging                  → overlaps transfer with render

LOW IMPACT (DEPRIORITIZED):
  Phase 1: ExtMem interop                  → saves ~2-3ms (interop overhead)
  Phase 4: Frame pipelining                → saves ~5ms (fixed overhead)
  Phase 5: Scene upload                    → one-time cost, not per-frame
```

---

## Measuring Progress

| Metric | How to measure | Current baseline |
|--------|---------------|-----------------|
| Per-query render cost | `benchmark_interop.py --sweep` | 1.05ms @ batch=32 |
| Fixed overhead per batch | batch=1 minus (per-query × 1) | ~5ms |
| GPU→CPU transfer | `benchmark_interop.py` | 42ms for 32×1280×720 |
| GPU utilization | `nsys` trace — idle gaps between CUDA/GL | Not yet measured |
| End-to-end FPS | `render_hdmap_scene.py` timing output | ~950 queries/s |
