# CUDA Rasterizer Port Notes

This directory is a port of the HPG-2011 NVIDIA CUDA rasterizer. It has been
adapted to act as the backend for the ludus renderer. Most of the
tile-level pipeline (triangle setup, bin raster, coarse raster, fine raster) is
upstream code; the host wrapper, the viewport/multi-image plumbing, and the
deterministic tiebreaker are local deviations.

These notes capture the non-obvious design decisions that aren't visible
from the diff alone — primarily for whoever next has to debug a corrupted
tile or extend the pixel pipe.

---

## Deterministic tiebreaker: per-pixel resident triangle index

The tiebreaker resolves which of N depth-equal fragments wins the pixel.
That decision needs the triangle index of the *current resident*
fragment so it can compare keys against an incoming candidate. There is no
free place in the existing tile state to store that — the upstream code only
keeps `tileColor` and `tileDepth` in shared memory.

### Where the resident triIdx lives

A framebuffer-shaped `int32` buffer (`m_triIdxBufferRaw` on the host, exposed
to the kernel as `c_crParams.triIdxBuffer` + `triIdxStride`). Each cell stores
the triangle index of the fragment that currently owns that pixel, or `-1`
when no fragment has written there since the last clear.

Lifecycle:

- Allocated in `CudaRaster::setBufferSize` at tile-aligned stride and
  `(m_numImages-1)*m_height + alignedH` rows. The over-allocation past the
  last image's logical bottom row is so that edge tiles (whose tile-aligned
  writes can extend past the viewport) never go out-of-bounds. The existing
  color/depth surfaces hide the same overshoot via `surf2Dwrite`'s silent OOB
  drop — a linear buffer would actually corrupt memory.
- Initialized to `-1` (`cudaMemset(..., 0xFF, ...)`) at allocation time and
  re-cleared in `drawTriangles` whenever `m_deferredClear` is set.
- Freed in `~CudaRaster` and on every `setBufferSize` call.

### Why global memory and not shared

The natural shape was `__shared__ S32 s_tileTriIdx[CR_FINE_MAX_WARPS][CR_TILE_SQR]`
in `fineRasterImpl_SingleSample` — same layout as `s_tileColor`/`s_tileDepth`,
zero L2 traffic, no allocation. That adds 5 KB to the kernel's static shared
footprint (`CR_FINE_MAX_WARPS=20`, `CR_TILE_SQR=64`, 4 bytes/cell), bringing
the total from ~47.25 KB to ~52.25 KB. Compiling that fails on Ampere
(compute_86) with:

```
ptxas error : Entry function 'crDefaultPipe_fineRaster' uses too much shared
data (0xd100 bytes, 0xc000 max)
```

— the 48 KB static shared-memory cap.

Two ways to make shared work were considered and rejected:

1. **Shrink another shared array.** The largest soft target is
   `s_temp[CR_FINE_MAX_WARPS][80]` at ~6.25 KB. `s_temp` is the per-warp
   scratch used as the prefix-sum staging buffer in `triangleSegmentScan`,
   the warp-wide write-counter in the fine raster's tile loop
   (`temp[16] = atomicAdd(...)`), and the buffer for `findFragment`. The 80
   slots aren't all live at once, but they're addressed at fixed offsets by
   several distinct stages and audited carefully across the bin/coarse/fine
   files. Shrinking it to make room for the tiebreaker buffer would mean a
   per-stage size audit and would silently corrupt tiles if any one site
   over-runs the new bound. Not worth it for a 5 KB gain.

2. **Narrower per-pixel type.** Storing `S16` would halve the array to
   2.5 KB and fit comfortably. Production draw calls already exceed
   65 535 triangles in a single dispatch (large meshes, full-frame
   tessellation), so a 16-bit triangle index would silently wrap and corrupt
   tiebreaker decisions for any draw past the 65 K boundary. Not safe.

Dynamic shared memory (`extern __shared__`) was also possible but adds host-
side `cuFuncSetAttribute(..., MaxDynamicSharedSizeBytes, ...)` plumbing and
opts the kernel into a different occupancy regime. Given that the global-
memory access pattern stays L1-resident (one tile at a time, one SM at a
time, contiguous within the warp), the win wasn't worth the complexity.

### Why no atomics on the global buffer

The fine raster already arbitrates per-pixel writes for color and depth:
after the warp-wide `atomicMin(pDepth, depth)` + `__syncwarp`, only the
elected lane (single lane per pixel via `__match_any_sync` ×
`__ballot_sync` × `tiebreakerLaneWins`) writes color and triangle index.
The tiebreaker triIdx write rides on the same arbitration —
the contended lanes have already been narrowed to one writer per pixel per
round, so a plain store is sufficient.

The pre-ROP zkill read (the equal-depth branch in `fineRasterImpl_SingleSample`)
happens before that arbitration but is read-only. It can race with a
concurrent writer in another iteration of the per-fragment loop. The next
iteration's `__ballot_sync` / `__syncwarp` re-establishes ordering, and the
`volatile` qualifier on the pointer prevents the compiler from caching the
load in a register across that barrier.

### What was removed when this landed

Before the global buffer, the resident triangle index was reverse-engineered
from the color buffer:

```
int oldTriIdx = (oldColor == 0) ? -1 : (int)oldColor - 1;
```

This was correct only when the active fragment shader was `TriIdShader` (which
writes `triIdx + 1` into the color buffer). It silently produced wrong
tiebreakers for any other shader, *and* required `tiebreakerColors` and
`TriIdShader` to be wired together at the call site. The global buffer
decouples tiebreaker correctness from the choice of fragment shader.

---

## Other deviations from upstream

### Viewport offsets

Upstream addresses the surface as `tileX << (CR_TILE_LOG2 + 2)`,
`tileY << CR_TILE_LOG2`, i.e. tile (0, 0) is always at the surface origin.
We support partial-viewport draws (offset within the framebuffer), so
`CRParams::surfaceOffsetX/Y` are added in at every tile-load and tile-write
in `fineRasterImpl_SingleSample` (and the dead MSAA path for consistency).

`crClearSurfaces` accepts `offsetX`/`offsetY`, but the active deferred-clear
path clears the whole backing surface (`m_width` by `m_height * m_numImages`)
at offset `(0, 0)`. Viewport offsets only bias rasterizer writes. That keeps
multi-tile draws from inheriting stale pixels outside the first tile.

### Multi-image / sub-range draws

`drawTriangles(ranges, ...)` lets the host dispatch each "image" with its
own `[firstTri, numTrisDraw)` slice of a shared vertex/index/triangle data
buffer, written into its own `surfaceOffsetY` band. The pipeline launches
once per image (`launchRange` lambda in `CudaRaster.cpp`), uploading new
params each time. The shared geometry buffers stay indexed by the *global*
triangle index, so `BinRaster.inl` and `TriangleSetup.inl` were updated to
compute `globalTri = c_crParams.firstTri + drawTriIdx` whenever they touch
`indexBuffer`, `triSubtris`, `triHeader`, or `triData`. Anywhere the kernel
loops over "all triangles in this draw" (e.g. the bin raster batch loop), the
bound is `numTrisDraw`, not `numTris`.

### Tile-segment cap

Upstream sized `m_maxTileSegs` as `(numTris-1)/CR_TILE_SEG_SIZE + 1 + slack`,
which is the *total* segment count across the whole frame and badly
under-counts when one triangle covers many tiles (each covered tile needs its
own segment entry). The cap is now:

```
triTileSegs = (numTris-1)/CR_TILE_SEG_SIZE + 1
m_maxTileSegs = max(numTiles,
                    min(numTiles * triTileSegs, 1 << 20)) + slack
```

The `1 << 20` ceiling is a memory budget — without it large `numTris` ×
`numTiles` would over-allocate.

### Equal-depth z-test

Upstream killed the incoming fragment when `depth >= oldDepth`, so equal-depth
fragments were dropped before the tiebreaker could run. The pre-ROP zkill in
`fineRasterImpl_SingleSample` now splits the comparison: `depth > oldDepth`
strictly kills, and `depth == oldDepth` defers to
`tiebreakerCandidateWins(triIdx, key, oldTriIdx)` (or kills outright when
the deterministic tiebreaker is disabled).

### Tiebreaker key and warp-lane election

`getTriangleTiebreakerKey(triIdx)` reads the optional per-vertex
`tiebreakerColors` buffer for the triangle's three vertices and returns
`max(c0, c1, c2)`. The intended use is "key = tile id" (so all triangles
sharing a tile compare equal on key and fall back to the secondary
`triIdx` order). When `tiebreakerColors` is null, the key collapses to
`triIdx` and the tiebreaker becomes "highest triIdx wins".

`tiebreakerLaneWins(candidateMask, triIdx, key)` in the executeROP path
elects exactly one writer when multiple lanes simultaneously won the depth
race for the same pixel. The fast path `__popc(candidateMask) == 1`
short-circuits the 32-lane `__shfl_sync` scan in the typical
one-fragment-per-pixel case, which is the dominant pattern under
production fragment distributions.

## `__activemask()` Audit (Reviewer #4)

The current port still uses `__activemask()` as the mask argument to several
Volta+ `_sync` intrinsics. `__activemask()` is a scheduler snapshot: it says
which lanes happen to be co-issued at that instruction, not which lanes the
algorithm intends to participate. It avoids referencing inactive lanes, but it
can make divergent-loop behavior depend on scheduler timing.

Future cleanup should use one of these patterns at every `_sync` intrinsic:

1. Full-warp convergence asserted by control flow: use `0xFFFFFFFFu`, with a
   comment naming the convergence point.
2. Predicate-derived participation: compute
   `unsigned mask = __ballot_sync(0xFFFFFFFFu, hasWork)` at a converged point,
   then use `mask` for later `_sync` calls inside the `hasWork` path.
3. Loop-arrival mask: derive the mask before entering a divergent loop, then
   either re-converge with `__syncwarp(mask)` inside the body or use atomics
   when subgroups may arrive separately.

Checklist for the audit:

- `BinRaster.inl`: per-warp prefix-sum scan sites around the 113/117 block are
  pattern 1 candidates.
- `BinRaster.inl`: sites around 221, 233, 241, 246, 258, 266, 321, and 346 are
  predicate-derived-work candidates and should be classified under pattern 2
  before changing masks.
- `CoarseRaster.inl`: sites around 308, 339, 400, and 496 are pattern 1
  candidates if the surrounding `__syncthreads()` still proves full-warp
  convergence.
- `CoarseRaster.inl`: the Case B inner-loop ballot around 411 is a pattern 3
  site. The current atomic write combines partial subgroups, but the mask
  should still come from algorithmic loop-arrival intent.
- `CoarseRaster.inl`: sites around 785 and 820 should be classified during the
  same audit; both currently inherit `__activemask()` behavior from upstream
  scan/write helpers.
- `FineRaster.inl`: sites around 178, 514, 866, and 869 are pattern 1
  candidates if their scopes remain converged.
- `FineRaster.inl`: ROP-path sites around 657, 688, 1006, and 1037 are
  predicate-derived-work candidates.
- `FineRaster.inl`: sites around 714, 745, 936, 939, 1076, and 1107 should be
  classified with the same ROP/scan mask audit.
- `FineRaster.inl`: `executeROP_SingleSample` now also uses
  `U32 activeMask = __activemask()` and derives `__match_any_sync` /
  `__ballot_sync` masks from it. Audit those together with the ROP sites.
- `Util.hpp`: delete `singleLane()` or replace it with a helper that accepts an
  explicit participation mask. "Single lane" is only meaningful relative to a
  known group of participating lanes.

## Depth Peeling Not Implemented

The public API still exposes depth-peeling controls, but the active draw
path does not implement depth peeling end-to-end:

- `setRenderModeFlags(RenderModeFlag_EnableDepthPeeling)` records the flag on
  the host, but `CudaRasterKernels.cu` instantiates only the default pipe with
  `FW::RenderModeFlag_EnableDepth`. The runtime flag bit never reaches a
  depth-peeling kernel.
- `drawTriangles(..., peel=true, ...)` stores `m_peelEnabled`; the active
  launch path does not use it to select a different pipe or surface.
- `swapDepthAndPeel()` swaps only the linear readback pointers. It does not
  swap `m_depthSurfaceObj`, `m_depthArray`, or `m_peelArray`.
- No active kernel writes to `m_peelArray`.

A complete future implementation needs at least a depth-peel pipe
instantiation, host-side selection of the peel pipe when `peel=true`,
surface-object/array ownership that makes `swapDepthAndPeel()` meaningful, and
ROP writes that populate the peel buffer. The positive-contract
`test_depth_peeling_*` tests stay `xfail(strict=True)`, and their paired
`*_currently_crashes_with_cuda700` markers stay plain tests until the failure
mode changes.

## Multi-Image Without Ranges

The active multi-image contract is range-driven. When `ranges != nullptr`,
`CudaRaster::drawTriangles` launches once per image and shifts each launch by
`imageIdx * m_height` in `surfaceOffsetY`.

When `ranges == nullptr`, the active path launches once and renders image 0
only. That matches the pinned
`test_multi_image_regression_second_image_remains_empty_for_single_draw`
behavior. A future all-images-without-ranges contract should choose one of
these semantics explicitly: replicate image 0 to every image, clear every image
but render only image 0, or reject `numImages > 1 && ranges == nullptr` with a
clear error.

### Dead paths still in the source

Two upstream code paths are unreachable in this build but kept verbatim:

- **MSAA fine raster** (`fineRasterImpl_MultiSample` in `FineRaster.inl`).
  The host always sets `samplesLog2 = 0` / `numSamples = 1` and only
  instantiates the single-sample template. The MSAA path received the
  `surfaceOffsetX/Y` rewrite for consistency, but it isn't exercised by any
  test or the production pipeline.
- **Kernel-side deferred clear** (`if (c_crParams.deferredClear)` branches
  in the fine raster tile-load). The host always sets
  `params.deferredClear = 0` and clears the surface up front via
  `crClearSurfaces` instead. The kernel branches stay in case we ever
  want a tile-local clear, but they're untested and would also need
  triIdx-buffer clearing wired in if revived.

## Active Surface

The active draw entry point is:

```cpp
bool CudaRaster::drawTriangles(const int32_t* ranges, bool peel, cudaStream_t stream)
```

It supports single-image and per-image range draws. It now ports upstream's
retry-and-grow behavior for internal rasterizer queues. If the subtriangle
count exceeds `CR_MAXSUBTRIS_SIZE`, the method returns `false` so callers can
handle the irrecoverable overflow without crashing the process.

## Incomplete Work

Each item below has an in-code `TODO(port)` cross-reference.

- `TODO(port): peel` - Depth peeling is not wired through the active draw path.
  See "Depth Peeling Not Implemented" above.
- `TODO(port): activemask` - `_sync` intrinsic masks still need a site-by-site
  audit. See "`__activemask()` Audit (Reviewer #4)" above.
- `TODO(port): multi-image-without-ranges` - multi-image draws are range-driven;
  without `ranges`, only image 0 is rendered. See "Multi-Image Without Ranges"
  above.
- `TODO(port): profiling` - Upstream `getStats()` and `getProfilingInfo()`
  still exist in the legacy host code, including CUevent-based per-stage timing
  and profile-counter readback, but they are not exposed through the active API.
- `TODO(port): emulation` - Upstream host-side stage emulation
  (`emulateTriangleSetup`, `emulateBinRaster`, `emulateCoarseRaster`,
  `emulateFineRaster`, plus helpers) is not invoked by the active draw path.
  Keep it available for future CPU-vs-GPU stage debugging until the Phase 2
  audit decides whether to remove or expose it.
- `TODO(port): debug-params` - Upstream `setDebugParams(...)` and
  `DebugParams` are not exposed through the active API.
- `TODO(port): gl-path` - The non-CUDA GL plugin path in
  `ludus_renderer/_ops/_plugin.py` references source files that are not present
  in this branch, including `glutil.cpp`, `ludus_gl.cpp`,
  `torch_bindings_gl.cpp`, and `torch_rasterize_gl.cpp`.

## Pending Phase 2 Audit

Do not delete inactive legacy host code until it has been audited. For each
item, decide whether the active code is a faithful 1:1 port of the same
semantics. Confirmed 1:1 ports can be deleted; unported capabilities should stay
with targeted TODOs.

Audit inventory:

- `void drawTriangles(void)`
- `launchStages(...)`
- `emulateTriangleSetup`, `emulateBinRaster`, `emulateCoarseRaster`,
  `emulateFineRaster`
- `setupTriangle` and `setupPleq`
- `getStats()` and `getProfilingInfo()`
- `setDebugParams(...)`
- Five `#if 0` FW-typed setters
- Dead or legacy state in `CudaRaster.hpp`, including FW surfaces, profiling
  members, `Stats`, and `DebugParams`

## Addressed Divergences

- The active draw path uses upstream slack values for queue growth:
  `4096 / 256 / 4096` for subtriangles, bin segments, and tile segments.
- The active draw path ports upstream retry-and-grow handling after the setup,
  bin, coarse, and fine stages.
- The active draw path no longer reports unconditional success when the
  internal queue hard cap is exceeded.

## New Integration Features

These are intentional integration features that upstream CudaRaster did not
have:

- `cudaStream_t` draw execution.
- Per-image `ranges` for multi-image rasterization.
- `cudaSurfaceObject_t` targets so CudaRaster can render into PyTorch-owned
  buffers.
- Deterministic tiebreaker controls used by the cleanroom contract tests.

## Test Pointers

The cleanroom public API contract lives in
`integrations/alpadreams/ludus-renderer/tests/test_cudaraster_api.py`. Its
module docstring describes the marker conventions and remaining validation
gaps.
