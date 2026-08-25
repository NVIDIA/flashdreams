# Rendering Pipeline Optimization Roadmap

## Benchmark Baseline (March 2025)

Measured on H100 80GB HBM3, 1280x720, CUDA software rasterizer (CudaRaster).

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
- Per-query cost plateaus at **~1.05ms** for batch >= 32 -- GPU-bound on rasterization
- Fixed overhead per batch call: **~5ms** (kernel launch + state setup)
- GPU->CPU transfer: **42ms for 32 frames (118 MB)** -- larger than the render itself
- Throughput saturates at **~950 queries/s** regardless of batch size

### Where time is spent

```
Full pipeline for 32 queries at 1280x720 (34.35ms total):

  Fixed overhead (~5ms):
    +-- CUDA kernel launch overhead              ~2ms
    +-- Buffer state setup                       ~2ms
    +-- Synchronization                          ~1ms

  Per-query rasterization cost (~1.05ms x 32 = ~29ms):
    +-- CudaRaster dispatch (polylines)
    +-- CudaRaster dispatch (polygons)
    +-- CudaRaster dispatch (obstacles)
    +-- MSAA resolve (if enabled)

  GPU->CPU transfer (42ms, measured separately):
    +-- cudaMemcpy device->host (118 MB RGBA8)
```

Rasterization dispatches dominate the per-query cost.

---

## Phase 2: GPU->CPU Transfer Optimization -- HIGH PRIORITY

The GPU->CPU transfer (42ms) exceeds the render time (34ms) for batch=32. This
is the single largest time sink.

### 2a: Pinned (page-locked) host memory

Allocate the output tensor in pinned memory so `cudaMemcpyAsync` can use DMA:

```python
output = torch.empty(..., pin_memory=True)
```

Expected improvement: 2-3x faster for large transfers.

### 2b: Async staging with double buffering

Overlap GPU->CPU transfer with the next batch's render:

```
Batch N:   Render(N) --> Copy-to-staging(N) --> DMA-to-host(N)
Batch N+1:               Render(N+1) ----------> Copy-to-staging(N+1)
                              (overlapped)
```

### 2c: Skip transfer entirely (GPU-resident pipeline)

If the consumer is a GPU model (training or inference), keep the rendered images
as GPU tensors. Eliminate the 42ms transfer completely.

---

## Phase 3: Rasterization Optimization -- HIGH PRIORITY

The 1.05ms per-query floor at high batch sizes means the CudaRaster kernel
dispatches are the GPU bottleneck.

### 3a: Profile with nsys

```bash
nsys profile --trace=cuda \
  uv run python examples/benchmark_renderer.py --scene ... --iters 10
```

Identify which rasterization pass (polyline, polygon, obstacle) dominates and
whether there are idle gaps between kernel launches.

### 3b: Reduce overdraw / early-exit

The current dispatch model uses upper-bound task counts (e.g., `MAX_VARRAYS_PER_POOL
= 1000`). Many invocations early-exit because there's no data.
Precomputing exact dispatch counts per scene could reduce wasted GPU work.

### 3c: Frustum culling

Add per-pool bounding box checks to skip pools outside the camera frustum. For
cameras with limited FOV (non-BEV), this can eliminate significant geometry.

---

## Phase 4: Scene Upload Optimization -- LOW PRIORITY

Scene upload is a one-time cost (~0.5s for small scenes, ~22s for large scenes).
Not a per-frame bottleneck.

Potential improvements:
- Parallel parquet decoding
- Streaming upload (start rendering before full scene is loaded)
- Scene caching / serialization

---

## Phase 5: Multi-Process / Multi-GPU -- FUTURE

For training at scale, separate scene loading and rendering across processes or
GPUs. Relevant only after per-frame bottlenecks are addressed.

---

## Priority Summary

```
Impact vs Effort:

HIGH IMPACT, MODERATE EFFORT:
  Phase 2a: Pinned memory transfers        -> saves ~20ms per batch
  Phase 2c: GPU-resident pipeline          -> saves ~42ms per batch
  Phase 3a: nsys profiling                 -> identifies rasterization bottleneck

HIGH IMPACT, HIGH EFFORT:
  Phase 3b-c: Rasterization optimization   -> reduces 1.05ms/query floor
  Phase 2b: Async staging                  -> overlaps transfer with render

LOW IMPACT:
  Phase 4: Scene upload                    -> one-time cost, not per-frame
```

---

## Measuring Progress

| Metric | How to measure | Current baseline |
|--------|---------------|-----------------|
| Per-query render cost | `benchmark_renderer.py --iters 10` | 1.05ms @ batch=32 |
| Fixed overhead per batch | batch=1 minus (per-query x 1) | ~5ms |
| GPU->CPU transfer | `benchmark_renderer.py` | 42ms for 32x1280x720 |
| GPU utilization | `nsys` trace -- idle gaps between CUDA kernels | Not yet measured |
| End-to-end FPS | `render_hdmap_scene.py` timing output | ~950 queries/s |
