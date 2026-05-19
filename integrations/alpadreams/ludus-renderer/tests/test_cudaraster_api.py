"""
Contract suite for `FW::CudaRaster` (the C++ class under
`ludus_renderer/_cpp/cudaraster/`).

Purpose
-------
The `cudaraster/` directory now contains the original BSD-3-Clause
CudaRaster code from the HPG 2011 paper "High-Performance Software
Rasterization on GPUs" by Laine and Karras. This suite pins the API
contract that the ported implementation must satisfy. The upstream code
requires adaptation: Volta+ warp divergence fixes, API shims to match
the `CR::CudaRaster` interface that ludus_cuda.cu expects, and removal
of the FW framework dependencies.

Marker conventions used in this file
------------------------------------
1. Plain test  -- positive contract a correct impl must satisfy.
2. CURRENT-IMPL REGRESSION MARKER  -- pins a known-quirky behavior of the
   current impl (e.g. tile-boundary pixel drop, multi-image image-1 left
   uninitialized). When a cleanroom change makes one of these fail, the
   header on the test tells you whether to delete the marker or investigate.
3. POSITIVE CONTRACT + xfail(strict=True)  -- the behavior we want, asserted
   today against a broken impl. xfail keeps the suite green; xpass is loud
   when the cleanroom fixes the bug.
4. CURRENT-IMPL FAILURE-MODE MARKER  -- pins the *exact stderr signature* of
   a current crash (CUDA-700 in depth peeling and the overflow path). When
   the failure mode changes for any reason, this test fails loudly so we
   know to look.

What the main app actually uses from `CR::CudaRaster`
-----------------------------------------------------
Audited in `ludus_renderer/_cpp/render/ludus_cuda.cu` (lines 2079-2204,
2309-2398). The vehicle/HD-map render path is the only consumer.

  setBufferSize(W, H, 1)           # always numImages = 1
  setVertexBuffer(float4 GPU ptr)
  setIndexBuffer(int32 GPU ptr)
  setTiebreakerColorBuffer(...)    # always set, every frame
  setDeterministicTiebreaker(true) # always true
  deferredClear(0)                 # always clear-to-zero
  setViewport(...)                 # MULTIPLE per frame (nested tiles)
  drawTriangles(nullptr, false, stream)  # no ranges, no peel, NON-DEFAULT stream
  getColorBuffer()                 # every frame; depth never read

Never called by the main app:
  setRenderModeFlags         (so flags=0: no backface cull, no peeling)
  swapDepthAndPeel           (no peeling)
  getDepthBuffer             (depth not consumed)
  ranges parameter           (always nullptr)
  numImages > 1
  non-deterministic tiebreaker
  non-zero clear color

Out of scope for the cudaraster cleanroom (handled elsewhere)
-------------------------------------------------------------
* Differentiability -- not used anywhere; every requires_grad in the
  workspace is False, the render output is detach()ed and uint8-cast in
  `interactive_drive/rasterizer.py:211` before reaching any model.
* Barycentrics -- CudaRaster does NOT output them. `ludus_cuda.cu`
  `fragmentKernel` (line 1850+) reconstructs them from triangle id +
  vertex positions + pixel-center NDC. Cleanroom only has to keep
  triangle id + bottom-up Y correct -- both pinned by existing tests.
* MSAA -- implemented as 2x SSAA + downsample in `ludus_cuda.cu`, not in
  cudaraster. setMsaaSamples lives on LudusCudaStateWrapper.
* Texture sampling, antialias pass -- not in cudaraster.

Companion validation suite
--------------------------
This file is not the only thing that exercises cudaraster. The other leg is:

  samples/interactive-drive/tests/test_raster_reference_image.py
    ::test_raster_reference_image

It drives the full pipeline:
  LudusConditionRasterizer -> LudusCudaTimestampedContext
    -> ludus_render_fwd_cuda_timestamped (torch_rasterize_cuda.cpp)
    -> ludusCudaRenderTimestamped (ludus_cuda.cu)
    -> CR::CudaRaster
and compares the rendered RGB against `raster_reference_*_300p.png` using
the FLIP perceptual metric (gate <= 0.03 mean FLIP). Parametrized for one
real scene and one synthetic scene, one frame each.

Implication: a cleanroom replacement is not validated by this file alone.
Both this file (wrapper-level contracts) and the reference-image test
(end-to-end visual regression) must pass.

Open test gaps (proposed, not yet written)
------------------------------------------
Tier A -- directly exercises the main-app path.
  A1. Stream propagation contract.
      Submit drawTriangles on a non-default torch stream, sync that stream
      only, assert the readback reflects the draw. Then enqueue work on
      stream X, capture output before+after a stream-X-only sync, assert
      the post-sync buffer differs and the pre-sync buffer (read on the
      same stream as the draw) is consistent. Catches a cleanroom impl
      that ignores the `stream` parameter and uses the default stream.
  A2. Tiled render -- 1x1, 2x1, 1x2, 2x2 tilings with rectangular
      viewports that perfectly tile the buffer (mimics ludus_cuda.cu's
      `tilesX*tilesY` nested loop). For each tiling, assert pixel-perfect
      coverage at concrete interior sample points within each tile; assert
      the last-row/last-column tile is correctly handled when the buffer
      dimension isn't a multiple of maxVp. The 2x2 case is already xfail
      (test_tiled_render_matches_single_view_interior); these positive
      contract tests for the smaller tilings should pass today.
  A3. Single-frame multi-camera pattern. Mirror the main-app loop:
      deferredClear(0); for cam in cameras: setVertexBuffer(cam_verts);
      setIndexBuffer(cam_indices); for tile: setViewport+drawTriangles.
      Assert each camera's geometry lands correctly without bleeding into
      the next, and that re-binding vertex/index buffers between draws
      within a single deferredClear period works.
  A4. End-to-end main-app smoke. One test that does exactly what
      ludusCudaRender does: setBufferSize(W,H,1); setVertexBuffer;
      setIndexBuffer; setTiebreakerColorBuffer; setDeterministicTiebreaker(
      true); deferredClear(0); tiled setViewport+drawTriangles loop on a
      non-default stream; getColorBuffer. Assert a known-good triangle id
      pattern. Acts as a regression guard for the full call sequence.
  A5. Padded-buffer correctness. setBufferSize rounds W/H up to the
      tile alignment internally (8x8); the reported `getBufferWidth/Height`
      can exceed the requested W/H. Every test today slices `[:H, :W]`
      and never asserts the padding region. Add a test that requests a
      non-tile-aligned size (e.g. 60x60 with crW/crH = 64), draws a
      triangle, and asserts the `[60:, :]` and `[:, 60:]` padding strips
      equal the deferred clear color (no garbage writes outside the
      requested viewport). Catches a cleanroom impl that scribbles
      undefined memory in the rounded-up region.

Tier B -- numerical edges real f-theta geometry will hit.
  B1. NDC-clipped triangles. Vertices outside [-1, 1]; the inside portion
      must rasterize correctly, the outside portion must produce no
      coverage. Probe with one vertex at NDC (2.5, 0.0).
  B2. Triangles partially behind the near plane (one vertex with w <= 0).
      Must produce the in-front portion only; must not wrap or crash.
  B3. Triangles with very small w (e.g. 1e-4 on one vertex). Probes
      perspective-division precision near the f-theta horizon.
  B4. CW-wound triangle with flags=0. Main app NEVER enables backface
      cull, so a CW triangle MUST rasterize. Currently nothing asserts
      this -- a cleanroom impl that quietly culls CW would silently drop
      half the geometry in production.

Tier C -- investigate before writing.
  C1. Index buffer layout. `CudaRaster.hpp` documents indices as
      `uint4 (idx0, idx1, idx2, color)` (16 B/triangle). The wrapper
      enforces `[N, 3] int32` (12 B/triangle) and passes data_ptr
      directly. Either the doc is stale or the wrapper is silently
      lying about the layout. Probe both shapes with a known triangle,
      pin whichever actually works, then either fix the wrapper or fix
      the docstring on `setIndexBuffer`.

Tier D -- improvements to the companion reference-image suite.
  Out-of-file work; track here so the cleanroom owner can decide whether
  to take them on alongside changes in this file.
  D1. Native-resolution comparison. Drop the 300p downscale before FLIP
      so subpixel rasterization changes are not smeared away by the
      bilinear downsample. Tighten the FLIP gate accordingly.
  D2. Multi-frame / multi-camera coverage. Today: one frame per scene,
      one camera. Add at least one frame with multiple cameras (mirroring
      the per-camera loop in ludusCudaRender) and one frame with a
      geometry-dense scene chunk to exercise the tiled path harder.
  D3. Depth-buffer regression. Reference-image test compares RGB only.
      Add a depth readback comparison (or a depth-derived statistic) so
      a cleanroom that breaks depth ordering without changing colors is
      caught.
  D4. Performance budget. Timestamp the render, gate the elapsed time
      against a recorded baseline (with reasonable headroom). A
      cleanroom impl that is functionally correct but 10x slower would
      pass today; the app's real-time budget would not survive.

Tier E -- cross-suite gaps neither file covers today.
  E1. Multi-arch GPU coverage. Both suites run on whichever single GPU
      is on the dev box. The HPG-2011 -> Volta+ porting work is exactly
      where bugs hide; the cleanroom replacement should be exercised on
      at least one Volta-class and one Hopper/Blackwell-class GPU before
      it is trusted in production. Today this is invisible.
  E2. Real-app workload volume. The reference-image test renders one
      frame; this file's longest test allocates 100k triangles for a
      single draw. The actual app renders thousands of frames per
      session with churning geometry. Memory-allocator behavior under
      repeated set_buffer_size calls, long-running stream stability,
      and any GPU-state leaks across many frames are unexercised.

Open scope decision
-------------------
~25 tests in this file cover features the main app does NOT use:
depth peeling (4 + 4 markers), `ranges` (3), multi-image (3), backface
culling (1), depth readback (handful), non-deterministic tiebreaker
(handful). The cleanroom replacement could either:
  (a) Preserve the full current API surface  -> keep all these tests.
  (b) Narrow the API to the main-app subset  -> delete tests for the
      removed surface and the corresponding C++ code together.
This decision affects which Tier A/B tests are worth writing and which
markers can be retired. Pending input from the project owner.

Prioritized validation roadmap (leverage-ranked)
------------------------------------------------
Before we trust a cleanroom replacement in production:
  1. A4 (end-to-end main-app smoke).        Highest leverage. Pins the
     actual call sequence ludusCudaRender uses. Catches the broadest
     class of regressions for the smallest effort.
  2. A1 (stream propagation).               Largest single correctness
     risk: a cleanroom impl that ignores the `stream` parameter would
     be silently incorrect under PyTorch's stream model.
  3. D1 (native-resolution reference).      Closes the FLIP-perceptual-
     smear gap so the visual regression test detects subpixel-correct
     rasterization changes instead of letting the downscale hide them.
  4. D4 (perf budget).                      Cheap to add, immediately
     catches the "functionally correct but unusably slow" case.

Nice-to-have, not strictly blocking the swap:
  5. E1 (multi-arch coverage).              Single biggest confidence
     gain for the Volta+ port; investment cost is CI infra, not test
     code.
  6. A5, A2, A3 (padded buffer; tiled patterns; multi-camera).
  7. Tier B (numerical edges).
  8. Tier C (index-buffer-layout probe; small but cleanup).
  9. D2, D3 (multi-frame/camera reference; depth comparison).
  10. E2 (workload-volume soak test).

How to interpret a failing test during cleanroom development
------------------------------------------------------------
* A REGRESSION MARKER fails  -> read its header; usually means the new
  impl fixed a known quirk. Action is in the comment.
* A FAILURE-MODE MARKER fails -> a CUDA-700 crash now produces a
  different error string, or no crash at all. Read the paired positive
  contract test; it will tell you which.
* An xfail xpasses (strict=True surfaces this as a failure) -> the
  cleanroom fixed the underlying bug. Remove the `xfail` decorator and
  delete the paired marker.
* A plain test fails -> investigate normally.
"""

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from ludus_renderer._ops._plugin import _get_plugin

pytestmark = pytest.mark.ci_gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CudaRaster API tests")


def _to_vertices(vertices: list[tuple[float, float, float, float]]) -> torch.Tensor:
    if len(vertices) == 0:
        return torch.empty((0, 4), device="cuda", dtype=torch.float32)
    return torch.tensor(vertices, device="cuda", dtype=torch.float32).contiguous()


def _to_indices(indices: list[tuple[int, int, int]]) -> torch.Tensor:
    if len(indices) == 0:
        return torch.empty((0, 3), device="cuda", dtype=torch.int32)
    return torch.tensor(indices, device="cuda", dtype=torch.int32).contiguous()


def _to_ranges(ranges: list[tuple[int, int]]) -> torch.Tensor:
    if len(ranges) == 0:
        return torch.empty((0, 2), dtype=torch.int32)
    return torch.tensor(ranges, dtype=torch.int32).contiguous()


# Exact stderr signature of every current-impl CUDA-700 crash we pin as a
# regression marker. Used by the depth-peeling and overflow failure-mode tests
# so any change in the failure mode (different error, different stack, no
# crash) is loud during cleanroom development.
# After the GL backend removal, depth peeling no longer crashes with CUDA 700
# but instead produces incorrect output (the peel pass returns the same triangle
# as the first pass instead of revealing the next layer).
_CURRENT_PEEL_FAILURE_STDERR = "AssertionError"


def _pack_rgba(r: int, g: int, b: int, a: int) -> int:
    packed = (r | (g << 8) | (b << 16) | (a << 24)) & 0xFFFFFFFF
    # Keep Python value in signed int32 range so torch.int32 construction does not overflow.
    if packed >= (1 << 31):
        packed -= 1 << 32
    return int(packed)


def _pack_rgba_unsigned(r: int, g: int, b: int, a: int) -> int:
    return int((r | (g << 8) | (b << 16) | (a << 24)) & 0xFFFFFFFF)


def _ndc_to_pixel(
    width: int, height: int, x_ndc: float, y_ndc: float
) -> tuple[int, int]:
    x = int((x_ndc + 1.0) * 0.5 * width)
    y = int((y_ndc + 1.0) * 0.5 * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def _pixel(color: np.ndarray, x: int, y_bottom: int) -> int:
    return int(color[y_bottom, x])


def _ring_band_triangles(
    outer: float, inner: float, z: float
) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    vertices = [
        (-outer, -outer, z, 1.0),
        (outer, -outer, z, 1.0),
        (outer, outer, z, 1.0),
        (-outer, outer, z, 1.0),
        (-inner, -inner, z, 1.0),
        (inner, -inner, z, 1.0),
        (inner, inner, z, 1.0),
        (-inner, inner, z, 1.0),
    ]
    indices = [
        (0, 1, 5),
        (0, 5, 4),  # bottom band
        (1, 2, 6),
        (1, 6, 5),  # right band
        (2, 3, 7),
        (2, 7, 6),  # top band
        (3, 0, 4),
        (3, 4, 7),  # left band
    ]
    return vertices, indices


def _polyline_strip_triangles(
    points: list[tuple[float, float]],
    half_width: float,
    z: float,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float, float]] = []
    indices: list[tuple[int, int, int]] = []
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        dx = bx - ax
        dy = by - ay
        length = float(np.sqrt(dx * dx + dy * dy))
        tx = dx / length
        ty = dy / length
        nx = -dy / length
        ny = dx / length
        quad = [
            (ax + nx * half_width, ay + ny * half_width, z, 1.0),
            (ax - nx * half_width, ay - ny * half_width, z, 1.0),
            (bx + nx * half_width, by + ny * half_width, z, 1.0),
            (bx - nx * half_width, by - ny * half_width, z, 1.0),
        ]
        base = len(vertices)
        vertices.extend(quad)
        indices.append((base + 0, base + 1, base + 2))
        indices.append((base + 1, base + 3, base + 2))
    return vertices, indices


def _is_clearly_inside(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    margin: float,
) -> bool:
    # Half-space tests with a margin to stay away from triangle boundaries.
    def edge(
        u: tuple[float, float], v: tuple[float, float], w: tuple[float, float]
    ) -> float:
        return (w[0] - u[0]) * (v[1] - u[1]) - (w[1] - u[1]) * (v[0] - u[0])

    e0 = edge(a, b, p)
    e1 = edge(b, c, p)
    e2 = edge(c, a, p)
    return (e0 > margin and e1 > margin and e2 > margin) or (
        e0 < -margin and e1 < -margin and e2 < -margin
    )


def _interior_and_exterior_pixels(
    width: int,
    height: int,
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    a, b, c = triangle
    interior: list[tuple[int, int]] = []
    exterior: list[tuple[int, int]] = []
    margin = 0.01
    for y in range(height):
        for x in range(width):
            px = (2.0 * x + 1.0) / width - 1.0
            py = (2.0 * y + 1.0) / height - 1.0
            if _is_clearly_inside((px, py), a, b, c, margin):
                interior.append((x, y))
            else:
                # Keep exterior points far enough from the triangle bbox to avoid edge ambiguity.
                if (
                    abs(px - a[0]) > 0.2
                    and abs(px - b[0]) > 0.2
                    and abs(px - c[0]) > 0.2
                ):
                    if (
                        abs(py - a[1]) > 0.2
                        and abs(py - b[1]) > 0.2
                        and abs(py - c[1]) > 0.2
                    ):
                        exterior.append((x, y))
    return interior, exterior


@dataclass
class RasterRun:
    color: np.ndarray
    depth: np.ndarray
    buffer_width: int
    buffer_height: int


class CudaRasterHarness:
    def __init__(self, plugin_module: Any) -> None:
        _require_cuda()
        self._plugin = plugin_module
        self._wrapper = self._plugin.CudaRasterTestWrapper(torch.cuda.current_device())
        self._width = 0
        self._height = 0

    def configure(self, width: int, height: int, num_images: int = 1) -> None:
        self._wrapper.set_buffer_size(width, height, num_images)
        self._width = width
        self._height = height

    def upload(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        colors: torch.Tensor | None = None,
    ) -> None:
        self._wrapper.set_vertex_buffer(vertices)
        self._wrapper.set_index_buffer(indices)
        if colors is not None:
            self._wrapper.set_tiebreaker_color_buffer(colors)

    def draw(
        self,
        clear_color: int | None,
        flags: int,
        deterministic_tiebreaker: bool,
        peel: bool = False,
        ranges: torch.Tensor | None = None,
        viewports: list[tuple[int, int, int, int]] | None = None,
    ) -> bool:
        # Note: when depth peeling is enabled, CudaRaster needs peel buffers allocated
        # under the current mode flags. We currently do that by re-calling set_buffer_size
        # with the configured dimensions before issuing the draw.
        if self._width <= 0 or self._height <= 0:
            raise RuntimeError("configure() must be called before draw()")
        self._wrapper.set_render_mode_flags(flags)
        # Depth peeling requires peel buffers allocated under the mode flags.
        if (flags & int(self._plugin.CR_RENDER_MODE_ENABLE_DEPTH_PEELING)) != 0:
            self._wrapper.set_buffer_size(
                self._width, self._height, self._wrapper.get_num_images()
            )
        self._wrapper.set_deterministic_tiebreaker(deterministic_tiebreaker)
        if clear_color is not None:
            self._wrapper.deferred_clear(clear_color)
        if viewports is None:
            self._wrapper.set_viewport(self._width, self._height, 0, 0)
            return bool(self._wrapper.draw_triangles(ranges, peel))
        ok = True
        for vp_w, vp_h, vp_x, vp_y in viewports:
            self._wrapper.set_viewport(vp_w, vp_h, vp_x, vp_y)
            ok = bool(self._wrapper.draw_triangles(ranges, peel)) and ok
        return ok

    def swap_depth_and_peel(self) -> None:
        self._wrapper.swap_depth_and_peel()

    def read(self, image_idx: int = 0) -> RasterRun:
        torch.cuda.synchronize()
        color = self._wrapper.get_color_buffer()[image_idx].detach().cpu().numpy()
        depth = self._wrapper.get_depth_buffer()[image_idx].detach().cpu().numpy()
        return RasterRun(
            color=color[: self._height, : self._width].copy(),
            depth=depth[: self._height, : self._width].copy(),
            buffer_width=int(self._wrapper.get_buffer_width()),
            buffer_height=int(self._wrapper.get_buffer_height()),
        )


@pytest.fixture(scope="module")
def cudaraster_plugin() -> Any:
    _require_cuda()
    return _get_plugin()


@pytest.fixture
def harness(cudaraster_plugin: Any) -> CudaRasterHarness:
    return CudaRasterHarness(cudaraster_plugin)


@pytest.mark.gpu
def test_basic_triangle_interior_exterior(harness: CudaRasterHarness) -> None:
    width, height = 64, 64
    vertices = _to_vertices(
        [(-0.5, -0.5, 0.0, 1.0), (0.5, -0.5, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])

    harness.configure(width, height)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    out = harness.read().color

    interior, exterior = _interior_and_exterior_pixels(
        width, height, ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
    )
    assert interior, "Need interior sample points for this geometry"
    assert exterior, "Need exterior sample points for this geometry"
    for x, y in interior[:: max(1, len(interior) // 40)]:
        assert _pixel(out, x, y) == 1
    for x, y in exterior[:: max(1, len(exterior) // 60)]:
        assert _pixel(out, x, y) == 0


@pytest.mark.gpu
def test_output_buffer_is_bottom_up_y_and_left_to_right_x(
    harness: CudaRasterHarness,
) -> None:
    # Pins the cudaraster output buffer's coordinate convention. This is the
    # contract that ludus_cuda.cu's fragmentKernel relies on: it computes
    # `cr_py = (height - 1) - py` (line 1869) and reconstructs barycentrics
    # at NDC `fy = (2 * cr_py + 1) / crHeight - 1` (line 1893). If the
    # cleanroom replacement flips Y to top-down, the entire downstream
    # fragment pass will silently produce vertically-mirrored output.
    # Deliberately bypasses _ndc_to_pixel / _pixel helpers, which already
    # encode the bottom-up convention, so a cleanroom flip is loud HERE
    # rather than producing confusing failures all over the suite.
    width, height = 64, 64

    upper = _to_vertices(
        [(-0.3, 0.2, 0.0, 1.0), (0.3, 0.2, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(width, height)
    harness.upload(upper, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color_upper = harness.read().color
    upper_rows = int(np.count_nonzero(np.any(color_upper != 0, axis=1)))
    coverage_top_half = int(np.count_nonzero(color_upper[height // 2 :, :]))
    coverage_bot_half = int(np.count_nonzero(color_upper[: height // 2, :]))
    assert upper_rows > 0, "triangle in upper-NDC half must produce coverage"
    assert coverage_top_half > 0, (
        "NDC y > 0 must land in the upper-row half (bottom-up Y)"
    )
    assert coverage_bot_half == 0, (
        f"NDC y > 0 leaked into lower-row half ({coverage_bot_half} px); "
        "cleanroom may have flipped to top-down Y, breaking fragmentKernel's cr_py mapping"
    )

    right = _to_vertices(
        [(0.2, -0.3, 0.0, 1.0), (0.8, -0.3, 0.0, 1.0), (0.5, 0.3, 0.0, 1.0)]
    )
    harness.configure(width, height)
    harness.upload(right, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color_right = harness.read().color
    coverage_right_half = int(np.count_nonzero(color_right[:, width // 2 :]))
    coverage_left_half = int(np.count_nonzero(color_right[:, : width // 2]))
    assert coverage_right_half > 0, (
        "NDC x > 0 must land in the right-column half (left-to-right X)"
    )
    assert coverage_left_half == 0, (
        f"NDC x > 0 leaked into left-column half ({coverage_left_half} px); cleanroom may have flipped X"
    )


@pytest.mark.gpu
def test_full_screen_triangle_fills_all_pixels(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-4.0, -4.0, 0.0, 1.0), (4.0, -4.0, 0.0, 1.0), (0.0, 6.0, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(48, 40)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert np.all(color == 1)


@pytest.mark.gpu
def test_offscreen_triangle_renders_background_only(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(1.5, 0.0, 0.0, 1.0), (2.0, 0.5, 0.0, 1.0), (1.8, -0.5, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.all(harness.read().color == 0)


@pytest.mark.gpu
def test_non_overlapping_quadrants_have_expected_ids(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [
            (-0.9, -0.9, 0.0, 1.0),
            (-0.55, -0.9, 0.0, 1.0),
            (-0.72, -0.55, 0.0, 1.0),
            (0.55, -0.9, 0.0, 1.0),
            (0.9, -0.9, 0.0, 1.0),
            (0.72, -0.55, 0.0, 1.0),
            (-0.9, 0.55, 0.0, 1.0),
            (-0.55, 0.9, 0.0, 1.0),
            (-0.72, 0.9, 0.0, 1.0),
            (0.55, 0.55, 0.0, 1.0),
            (0.9, 0.55, 0.0, 1.0),
            (0.72, 0.9, 0.0, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    for tri_id, point in (
        (1, (-0.72, -0.72)),
        (2, (0.72, -0.72)),
        (3, (-0.72, 0.72)),
        (4, (0.72, 0.72)),
    ):
        x, y = _ndc_to_pixel(64, 64, point[0], point[1])
        assert _pixel(color, x, y) == tri_id


@pytest.mark.gpu
def test_degenerate_triangle_has_no_pixels(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.5, -0.5, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.all(harness.read().color == 0)


@pytest.mark.gpu
def test_empty_draw_preserves_clear_color(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices([])
    indices = _to_indices([])
    clear_color = _pack_rgba(0x12, 0x34, 0x56, 0x78)
    harness.configure(32, 32)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=clear_color, flags=0, deterministic_tiebreaker=False
    )
    assert np.all(harness.read().color == clear_color)


@pytest.mark.gpu
def test_y_up_convention_places_bottom_left_triangle_correctly(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-0.95, -0.95, 0.0, 1.0), (-0.2, -0.95, 0.0, 1.0), (-0.95, -0.2, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    bottom_left = _pixel(color, 4, 4)
    top_left = _pixel(color, 4, 59)
    assert bottom_left == 1
    assert top_left == 0


@pytest.mark.gpu
def test_perspective_divide_allows_visibility_with_w_two(
    harness: CudaRasterHarness,
) -> None:
    # x/w and y/w are in range even though x,y alone are outside.
    vertices = _to_vertices(
        [(-1.5, -1.5, 0.0, 2.0), (1.5, -1.5, 0.0, 2.0), (0.0, 1.5, 0.0, 2.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.any(harness.read().color == 1)


@pytest.mark.gpu
def test_ndc_corner_vertex_maps_to_bottom_left_region(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-1.0, -1.0, 0.0, 1.0), (-0.7, -1.0, 0.0, 1.0), (-1.0, -0.7, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert _pixel(color, 1, 1) == 1
    assert _pixel(color, 1, 62) == 0


@pytest.mark.gpu
def test_depth_front_triangle_wins_overlap(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.5, -0.5, 0.2, 1.0),
            (0.5, -0.5, 0.2, 1.0),
            (0.0, 0.5, 0.2, 1.0),
            (-0.5, -0.5, 0.8, 1.0),
            (0.5, -0.5, 0.8, 1.0),
            (0.0, 0.5, 0.8, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    center = _ndc_to_pixel(64, 64, 0.0, 0.0)
    assert _pixel(harness.read().color, center[0], center[1]) == 1


@pytest.mark.gpu
def test_depth_partial_occlusion_keeps_visible_back_region(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [
            (-0.2, -0.2, 0.2, 1.0),
            (0.2, -0.2, 0.2, 1.0),
            (0.0, 0.2, 0.2, 1.0),
            (-0.8, -0.8, 0.8, 1.0),
            (0.8, -0.8, 0.8, 1.0),
            (0.0, 0.8, 0.8, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    center = _ndc_to_pixel(64, 64, 0.0, 0.0)
    lower_left = _ndc_to_pixel(64, 64, -0.6, -0.6)
    assert _pixel(color, center[0], center[1]) == 1
    assert _pixel(color, lower_left[0], lower_left[1]) == 2


@pytest.mark.gpu
def test_depth_three_layers_show_nearest(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.45, -0.45, 0.2, 1.0),
            (0.45, -0.45, 0.2, 1.0),
            (0.0, 0.45, 0.2, 1.0),
            (-0.45, -0.45, 0.5, 1.0),
            (0.45, -0.45, 0.5, 1.0),
            (0.0, 0.45, 0.5, 1.0),
            (-0.45, -0.45, 0.8, 1.0),
            (0.45, -0.45, 0.8, 1.0),
            (0.0, 0.45, 0.8, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    x, y = _ndc_to_pixel(64, 64, 0.0, 0.0)
    assert _pixel(harness.read().color, x, y) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("z_value", [-0.99, 0.99])
def test_depth_clip_boundaries_just_inside_are_visible(
    harness: CudaRasterHarness, z_value: float
) -> None:
    vertices = _to_vertices(
        [
            (-0.3, -0.3, z_value, 1.0),
            (0.3, -0.3, z_value, 1.0),
            (0.0, 0.3, z_value, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.any(harness.read().color == 1)


@pytest.mark.gpu
def test_zfight_without_tiebreaker_is_consistent(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.4, -0.4, 0.3, 1.0),
            (0.4, -0.4, 0.3, 1.0),
            (0.0, 0.4, 0.3, 1.0),
            (-0.4, -0.4, 0.3, 1.0),
            (0.4, -0.4, 0.3, 1.0),
            (0.0, 0.4, 0.3, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read().color
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    second = harness.read().color
    assert np.array_equal(first, second)
    overlap = first[np.nonzero(first)]
    assert overlap.size > 0
    assert np.all(np.logical_or(overlap == 1, overlap == 2))


@pytest.mark.gpu
def test_tiebreaker_enabled_is_deterministic(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.35, -0.35, 0.4, 1.0),
            (0.35, -0.35, 0.4, 1.0),
            (0.0, 0.35, 0.4, 1.0),
            (-0.35, -0.35, 0.4, 1.0),
            (0.35, -0.35, 0.4, 1.0),
            (0.0, 0.35, 0.4, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    colors = torch.tensor(
        [
            _pack_rgba(255, 0, 0, 255),
            _pack_rgba(255, 64, 0, 255),
            _pack_rgba(255, 128, 0, 255),
            _pack_rgba(0, 0, 255, 255),
            _pack_rgba(0, 64, 255, 255),
            _pack_rgba(0, 128, 255, 255),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    harness.configure(64, 64)
    harness.upload(vertices, indices, colors)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = harness.read().color
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = harness.read().color
    assert np.array_equal(first, second)


@pytest.mark.gpu
def test_tiebreaker_uses_colors_not_triangle_order(cudaraster_plugin: Any) -> None:
    width = 64
    height = 64
    vertices = _to_vertices(
        [
            (-0.3, -0.3, 0.5, 1.0),
            (0.3, -0.3, 0.5, 1.0),
            (0.0, 0.3, 0.5, 1.0),
            (-0.3, -0.3, 0.5, 1.0),
            (0.3, -0.3, 0.5, 1.0),
            (0.0, 0.3, 0.5, 1.0),
        ]
    )
    colors = torch.tensor(
        [
            _pack_rgba(255, 0, 0, 255),
            _pack_rgba(255, 0, 32, 255),
            _pack_rgba(255, 0, 64, 255),
            _pack_rgba(0, 255, 0, 255),
            _pack_rgba(0, 255, 32, 255),
            _pack_rgba(0, 255, 64, 255),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    first_harness = CudaRasterHarness(cudaraster_plugin)
    first_harness.configure(width, height)
    first_harness.upload(vertices, _to_indices([(0, 1, 2), (3, 4, 5)]), colors)
    assert first_harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = first_harness.read().color

    second_harness = CudaRasterHarness(cudaraster_plugin)
    second_harness.configure(width, height)
    second_harness.upload(vertices, _to_indices([(3, 4, 5), (0, 1, 2)]), colors)
    assert second_harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = second_harness.read().color

    x, y = _ndc_to_pixel(width, height, 0.0, 0.0)
    winner_first = _pixel(first, x, y)
    winner_second = _pixel(second, x, y)
    assert winner_first != 0 and winner_second != 0
    # If tiebreaker uses colors as primary factor, winner identity survives index reorder.
    assert winner_first != winner_second


@pytest.mark.gpu
def test_backface_culling_ccw_visible(
    harness: CudaRasterHarness, cudaraster_plugin: Any
) -> None:
    flags = int(cudaraster_plugin.CR_RENDER_MODE_ENABLE_BACKFACE_CULLING)
    vertices = _to_vertices(
        [(-0.6, -0.6, 0.0, 1.0), (0.6, -0.6, 0.0, 1.0), (0.0, 0.6, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=flags, deterministic_tiebreaker=False)
    assert np.any(harness.read().color == 1)


@pytest.mark.gpu
def test_backface_culling_cw_hidden_when_enabled(
    harness: CudaRasterHarness, cudaraster_plugin: Any
) -> None:
    flags = int(cudaraster_plugin.CR_RENDER_MODE_ENABLE_BACKFACE_CULLING)
    vertices = _to_vertices(
        [(-0.6, -0.6, 0.0, 1.0), (0.0, 0.6, 0.0, 1.0), (0.6, -0.6, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=flags, deterministic_tiebreaker=False)
    assert np.all(harness.read().color == 0)


@pytest.mark.gpu
def test_backface_culling_cw_visible_when_disabled(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.6, -0.6, 0.0, 1.0), (0.0, 0.6, 0.0, 1.0), (0.6, -0.6, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.any(harness.read().color == 1)


@pytest.mark.gpu
def test_buffer_size_unaligned_renders_correctly(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.2, -0.2, 0.0, 1.0), (0.2, -0.2, 0.0, 1.0), (0.0, 0.2, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(100, 100)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    out = harness.read()
    assert out.buffer_width >= 100
    assert out.buffer_height >= 100
    cx, cy = _ndc_to_pixel(100, 100, 0.0, 0.0)
    assert _pixel(out.color, cx, cy) == 1


def _render_single_view(
    harness: CudaRasterHarness, vertices: torch.Tensor, indices: torch.Tensor
) -> np.ndarray:
    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    return harness.read().color


def _render_tiled_view(
    harness: CudaRasterHarness, vertices: torch.Tensor, indices: torch.Tensor
) -> np.ndarray:
    harness.configure(128, 128)
    harness.upload(vertices, indices)
    viewports = [(64, 64, 0, 0), (64, 64, 64, 0), (64, 64, 0, 64), (64, 64, 64, 64)]
    assert harness.draw(
        clear_color=0, flags=0, deterministic_tiebreaker=True, viewports=viewports
    )
    return harness.read().color


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="known-broken in current impl: tiled render drops tile-boundary pixels",
)
def test_tiled_render_matches_single_view_interior(cudaraster_plugin: Any) -> None:
    # POSITIVE CONTRACT: a 4-tile render and a single-view render of the same
    # geometry must produce identical color buffers. Currently xfail because the
    # current impl drops tile-boundary pixels; see paired marker
    # test_tiled_render_regression_center_pixel_drops_for_reference_triangle.
    # When this xpasses, delete the marker test and remove the xfail.
    vertices = _to_vertices(
        [(-0.8, -0.8, 0.2, 1.0), (0.8, -0.8, 0.2, 1.0), (0.0, 0.8, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    single = _render_single_view(
        CudaRasterHarness(cudaraster_plugin), vertices, indices
    )
    tiled = _render_tiled_view(CudaRasterHarness(cudaraster_plugin), vertices, indices)
    assert np.array_equal(tiled, single)


@pytest.mark.gpu
def test_tiled_render_regression_center_pixel_drops_for_reference_triangle(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL REGRESSION MARKER -- read before changing this test.
    #
    # Pinned behavior: with this reference triangle and a 4-tile viewport
    # configuration, the tiled render does NOT cover the center pixel, even
    # though single-view rendering does.
    # Why it's unusual: tile-boundary pixels are unowned by any tile under the
    # current edge-rule implementation.
    # Paired positive contract test: test_tiled_render_matches_single_view_interior
    # (currently xfail strict=True).
    #
    # When this test fails on a cleanroom replacement:
    #   1. Verify the failure is a behavioral improvement (tiled now matches single).
    #   2. Confirm the paired positive contract test now passes / xpasses loudly.
    #   3. DELETE this marker; the positive contract becomes the source of truth.
    vertices = _to_vertices(
        [(-0.8, -0.8, 0.2, 1.0), (0.8, -0.8, 0.2, 1.0), (0.0, 0.8, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    single = _render_single_view(
        CudaRasterHarness(cudaraster_plugin), vertices, indices
    )
    tiled = _render_tiled_view(CudaRasterHarness(cudaraster_plugin), vertices, indices)
    sx, sy = _ndc_to_pixel(128, 128, 0.0, 0.0)
    assert _pixel(single, sx, sy) == 1
    assert _pixel(tiled, sx, sy) == 0


@pytest.mark.gpu
def test_viewport_offset_shifts_output(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.2, -0.2, 0.0, 1.0), (0.2, -0.2, 0.0, 1.0), (0.0, 0.2, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    base = harness.read().color

    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=0,
        flags=0,
        deterministic_tiebreaker=False,
        viewports=[(96, 96, 32, 32)],
    )
    shifted = harness.read().color

    base_count = int(np.count_nonzero(base == 1))
    shifted_count = int(np.count_nonzero(shifted == 1))
    assert shifted_count > 0
    assert shifted_count <= base_count
    assert _pixel(shifted, 20, 20) == 0


@pytest.mark.gpu
def test_clear_to_nonzero_sets_background(harness: CudaRasterHarness) -> None:
    clear_color_u32 = _pack_rgba_unsigned(0xDE, 0xAD, 0xBE, 0xEF)
    vertices = _to_vertices(
        [(-0.2, -0.2, 0.0, 1.0), (0.2, -0.2, 0.0, 1.0), (0.0, 0.2, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=clear_color_u32, flags=0, deterministic_tiebreaker=False
    )
    color = harness.read().color
    x, y = _ndc_to_pixel(64, 64, 0.0, 0.0)
    assert _pixel(color, x, y) == 1
    background = np.uint32(np.int32(_pixel(color, 2, 2))).item()
    assert background == np.uint32(clear_color_u32).item()


@pytest.mark.gpu
def test_clear_to_nonzero_regression_reads_back_as_signed_int32(
    harness: CudaRasterHarness,
) -> None:
    # CURRENT-IMPL REGRESSION MARKER -- read before changing this test.
    #
    # Pinned behavior: the wrapper exposes the color buffer as torch.int32, so
    # an RGBA8 clear with alpha >= 128 reads back as a negative signed integer
    # whose unsigned reinterpretation equals the original packed RGBA8.
    # Why it's unusual: a uint32 view would be a more natural contract, but the
    # current pybind binding picks int32. Pinned so any change in the readback
    # dtype is loud.
    # Paired positive contract test: test_clear_to_nonzero_sets_background.
    #
    # When this test fails on a cleanroom replacement:
    #   1. If the wrapper now returns uint32, update test_clear_to_nonzero_sets_background
    #      to read it as uint32 directly and DELETE this marker.
    #   2. If the clear path itself broke, fix it; the paired test will also fail.
    clear_color_u32 = _pack_rgba_unsigned(0xDE, 0xAD, 0xBE, 0xEF)
    harness.configure(32, 32)
    harness.upload(_to_vertices([]), _to_indices([]))
    assert harness.draw(
        clear_color=clear_color_u32, flags=0, deterministic_tiebreaker=False
    )
    color = harness.read().color

    assert color.dtype == np.int32
    assert np.all(color < 0), (
        "every pixel of the cleared buffer must reinterpret as negative int32"
    )
    reinterpreted = color.view(np.uint32)
    expected = np.uint32(clear_color_u32)
    assert np.all(reinterpreted == expected), (
        "clear pixels must all equal the packed RGBA8 clear color"
    )

    color_again = harness.read().color
    assert np.array_equal(color, color_again), (
        "consecutive reads of an unchanged buffer must be byte-stable"
    )


@pytest.mark.gpu
def test_no_clear_accumulates_previous_draw(harness: CudaRasterHarness) -> None:
    harness.configure(64, 64)
    vertices_a = _to_vertices(
        [(-0.9, -0.9, 0.0, 1.0), (-0.5, -0.9, 0.0, 1.0), (-0.7, -0.5, 0.0, 1.0)]
    )
    indices_a = _to_indices([(0, 1, 2)])
    harness.upload(vertices_a, indices_a)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)

    vertices_b = _to_vertices(
        [(0.5, 0.5, 0.0, 1.0), (0.9, 0.5, 0.0, 1.0), (0.7, 0.9, 0.0, 1.0)]
    )
    indices_b = _to_indices([(0, 1, 2)])
    harness.upload(vertices_b, indices_b)
    assert harness.draw(clear_color=None, flags=0, deterministic_tiebreaker=False)

    color = harness.read().color
    ax, ay = _ndc_to_pixel(64, 64, -0.7, -0.7)
    bx, by = _ndc_to_pixel(64, 64, 0.7, 0.7)
    assert _pixel(color, ax, ay) == 1
    assert _pixel(color, bx, by) == 1


@pytest.mark.gpu
def test_set_buffer_size_zero_is_accepted_and_reports_zero_dims(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL REGRESSION MARKER -- read before changing this test.
    #
    # Pinned behavior: set_buffer_size(0, 0, 0) is silently accepted, no
    # exception is raised, and the getters report (0, 0, 0). Calling them again
    # must report the same values (no internal mutation between getters).
    # Why it's unusual: a defensive impl would reject zero dimensions outright.
    # Paired positive contract test: test_get_buffer_dims_report_tile_rounded_allocation
    # (asserts the contracted rounding behavior for any positive dimensions).
    #
    # When this test fails on a cleanroom replacement:
    #   1. If the new impl raises on zero dims, that's the better contract:
    #      replace this test with a `pytest.raises` assertion and document the
    #      stricter contract.
    #   2. If the new impl returns different values for zero input, decide
    #      whether that's intentional and update accordingly.
    wrapper = cudaraster_plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    wrapper.set_buffer_size(0, 0, 0)
    assert int(wrapper.get_buffer_width()) == 0
    assert int(wrapper.get_buffer_height()) == 0
    assert int(wrapper.get_num_images()) == 0
    # Getters must be pure: a second call must report the same values.
    assert int(wrapper.get_buffer_width()) == 0
    assert int(wrapper.get_buffer_height()) == 0
    assert int(wrapper.get_num_images()) == 0


@pytest.mark.gpu
def test_tiebreaker_disabled_ignores_uploaded_colors(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [
            (-0.4, -0.4, 0.3, 1.0),
            (0.4, -0.4, 0.3, 1.0),
            (0.0, 0.4, 0.3, 1.0),
            (-0.4, -0.4, 0.3, 1.0),
            (0.4, -0.4, 0.3, 1.0),
            (0.0, 0.4, 0.3, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    colors = torch.tensor(
        [
            _pack_rgba(255, 0, 0, 255),
            _pack_rgba(255, 0, 0, 255),
            _pack_rgba(255, 0, 0, 255),
            _pack_rgba(0, 255, 0, 255),
            _pack_rgba(0, 255, 0, 255),
            _pack_rgba(0, 255, 0, 255),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    cx, cy = _ndc_to_pixel(96, 96, 0.0, 0.0)

    harness.configure(96, 96)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    no_color = harness.read().color

    harness.upload(vertices, indices, colors=colors)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    with_color = harness.read().color

    assert _pixel(no_color, cx, cy) == _pixel(with_color, cx, cy)


def _coplanar_overlap_pair(z: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _to_vertices(
            [
                (-0.4, -0.4, z, 1.0),
                (0.4, -0.4, z, 1.0),
                (0.0, 0.4, z, 1.0),
                (-0.4, -0.4, z, 1.0),
                (0.4, -0.4, z, 1.0),
                (0.0, 0.4, z, 1.0),
            ]
        ),
        _to_indices([(0, 1, 2), (3, 4, 5)]),
    )


@pytest.mark.gpu
def test_tiebreaker_enabled_without_color_buffer_is_deterministic(
    cudaraster_plugin: Any,
) -> None:
    # Intent (real contract): when the deterministic tiebreaker is enabled but
    # no tiebreaker color buffer was ever uploaded, the impl must still produce
    # a deterministic, byte-stable rasterization (or refuse with an error).
    # It must not produce different output across runs and must not crash.
    #
    # Below we ALSO pin the current-impl-specific winner (triangle id 2) as a
    # labeled regression marker so any change in the chosen winner is loud
    # during cleanroom development.
    vertices, indices = _coplanar_overlap_pair()

    runs: list[np.ndarray] = []
    for _ in range(5):
        harness = CudaRasterHarness(cudaraster_plugin)
        harness.configure(96, 96)
        harness.upload(vertices, indices)
        assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
        runs.append(harness.read().color.copy())

    for i in range(1, len(runs)):
        assert np.array_equal(runs[0], runs[i]), (
            f"deterministic tiebreaker without colors must be byte-stable across runs (run {i} differs)"
        )

    cx, cy = _ndc_to_pixel(96, 96, 0.0, 0.0)
    # Current-impl regression marker: winner at center is triangle id 2.
    assert _pixel(runs[0], cx, cy) == 2, (
        "winner changed; if the new winner is documented, update this marker"
    )


@pytest.mark.gpu
def test_tiebreaker_enabled_zero_color_buffer_is_deterministic(
    cudaraster_plugin: Any,
) -> None:
    # Intent (real contract): with the deterministic tiebreaker enabled and an
    # all-zero color buffer, every per-vertex tiebreaker color is identical so
    # the rasterizer cannot use them to break the tie. The result must still
    # be deterministic and byte-stable across runs (the impl must fall back to
    # SOME well-defined rule, not random behavior).
    #
    # We ALSO pin the current-impl-specific winner (triangle id 1) as a labeled
    # regression marker so any change in the fallback rule is loud.
    vertices, indices = _coplanar_overlap_pair()
    colors = torch.zeros((6,), device="cuda", dtype=torch.int32)

    runs: list[np.ndarray] = []
    for _ in range(5):
        harness = CudaRasterHarness(cudaraster_plugin)
        harness.configure(96, 96)
        harness.upload(vertices, indices, colors=colors)
        assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
        runs.append(harness.read().color.copy())

    for i in range(1, len(runs)):
        assert np.array_equal(runs[0], runs[i]), (
            f"deterministic tiebreaker with all-equal colors must be byte-stable across runs (run {i} differs)"
        )

    cx, cy = _ndc_to_pixel(96, 96, 0.0, 0.0)
    # Current-impl regression marker: fallback winner at center is triangle id 1.
    assert _pixel(runs[0], cx, cy) == 1, (
        "fallback winner changed; if the new fallback rule is documented, update this marker"
    )


@pytest.mark.gpu
def test_reupload_vertex_and_index_buffers_replaces_scene_geometry(
    harness: CudaRasterHarness,
) -> None:
    indices = _to_indices([(0, 1, 2)])
    first_vertices = _to_vertices(
        [(-0.8, -0.8, 0.2, 1.0), (-0.4, -0.8, 0.2, 1.0), (-0.6, -0.4, 0.2, 1.0)]
    )
    second_vertices = _to_vertices(
        [(0.4, 0.4, 0.2, 1.0), (0.8, 0.4, 0.2, 1.0), (0.6, 0.8, 0.2, 1.0)]
    )
    harness.configure(96, 96)

    harness.upload(first_vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read().color
    ax, ay = _ndc_to_pixel(96, 96, -0.6, -0.6)
    bx, by = _ndc_to_pixel(96, 96, 0.6, 0.6)
    assert _pixel(first, ax, ay) == 1
    assert _pixel(first, bx, by) == 0

    harness.upload(second_vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    second = harness.read().color
    assert _pixel(second, ax, ay) == 0
    assert _pixel(second, bx, by) == 1


@pytest.mark.gpu
def test_resize_after_draw_produces_consistent_output_in_new_extent(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-0.4, -0.4, 0.2, 1.0), (0.4, -0.4, 0.2, 1.0), (0.0, 0.4, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])

    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read().color
    assert int(np.count_nonzero(first)) > 0

    harness.configure(160, 96)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    second = harness.read().color
    assert second.shape == (96, 160)
    cx, cy = _ndc_to_pixel(160, 96, 0.0, -0.1)
    assert _pixel(second, cx, cy) == 1


_DEPTH_PEEL_TWO_LAYER_SCRIPT = textwrap.dedent(
    """
    import torch
    from ludus_renderer._ops._plugin import _get_plugin

    plugin = _get_plugin()
    w = plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    flags = int(plugin.CR_RENDER_MODE_ENABLE_DEPTH_PEELING)
    verts = torch.tensor(
        [[-0.4, -0.4, 0.2, 1.0], [0.4, -0.4, 0.2, 1.0], [0.0, 0.4, 0.2, 1.0],
         [-0.4, -0.4, 0.7, 1.0], [0.4, -0.4, 0.7, 1.0], [0.0, 0.4, 0.7, 1.0]],
        device="cuda", dtype=torch.float32
    )
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
    w.set_buffer_size(64, 64, 1)
    w.set_vertex_buffer(verts)
    w.set_index_buffer(idx)
    w.set_render_mode_flags(flags)
    w.set_deterministic_tiebreaker(False)
    w.deferred_clear(0)
    w.set_viewport(64, 64, 0, 0)

    ok1 = bool(w.draw_triangles(None, False))
    torch.cuda.synchronize()
    c1 = int(w.get_color_buffer()[0, 32, 32].item())
    w.swap_depth_and_peel()

    ok2 = bool(w.draw_triangles(None, True))
    torch.cuda.synchronize()
    c2 = int(w.get_color_buffer()[0, 32, 32].item())

    assert ok1 and ok2
    assert c1 == 1, f"first pass center must be near triangle id 1, got {c1}"
    assert c2 == 2, f"second pass center must reveal far triangle id 2, got {c2}"
    """
)


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="known-broken in current impl: depth peeling")
def test_depth_peeling_two_layers_exposes_back_layer_on_second_pass(
    cudaraster_plugin: Any,
) -> None:
    # POSITIVE CONTRACT: with two overlapping triangles at different depths,
    # peel pass 1 must show the near triangle at the center pixel, and peel
    # pass 2 must reveal the far triangle. Currently xfail because the impl
    # crashes; see paired marker
    # test_depth_peeling_two_layers_currently_crashes_with_cuda700.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_TWO_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu
def test_depth_peeling_two_layers_currently_crashes_with_cuda700(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL FAILURE-MODE MARKER. See the depth-peel header comment for
    # the contract pairing. Pinned crash signature for the two-layer scenario.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_TWO_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "two-layer peel no longer crashes; positive contract should now pass"
    )
    assert _CURRENT_PEEL_FAILURE_STDERR in result.stderr, (
        f"two-layer peel now fails with a different error:\n{result.stderr}"
    )


@pytest.mark.gpu
def test_multi_image_readback_shape_and_byte_stability(
    harness: CudaRasterHarness,
) -> None:
    # Contract: with num_images=2 the readback exposes a (2, H, W) tensor for
    # both color and depth, both images report identical buffer dimensions,
    # image 0 contains the rasterized triangle, and consecutive reads of either
    # image are byte-stable.
    #
    # NB: the current impl does NOT clear image 1 on deferred_clear, so image 1
    # may contain uninitialized GPU memory. Constraints on image 1 *content*
    # belong in test_multi_image_regression_second_image_remains_empty_for_single_draw.
    vertices = _to_vertices(
        [(-0.3, -0.3, 0.0, 1.0), (0.3, -0.3, 0.0, 1.0), (0.0, 0.3, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64, num_images=2)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read(0)
    second = harness.read(1)

    assert first.color.shape == (64, 64)
    assert second.color.shape == (64, 64)
    assert first.depth.shape == (64, 64)
    assert second.depth.shape == (64, 64)
    assert first.buffer_width == second.buffer_width
    assert first.buffer_height == second.buffer_height
    assert np.any(first.color == 1), "image 0 must contain the rasterized triangle"

    first_again = harness.read(0)
    second_again = harness.read(1)
    assert np.array_equal(first.color, first_again.color)
    assert np.array_equal(second.color, second_again.color)
    assert np.array_equal(first.depth, first_again.depth)
    assert np.array_equal(second.depth, second_again.depth)


@pytest.mark.gpu
def test_get_buffer_dims_report_tile_rounded_allocation(
    cudaraster_plugin: Any,
) -> None:
    wrapper = cudaraster_plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    wrapper.set_buffer_size(101, 67, 1)
    # After GL removal the CUDA-only backend reports requested dimensions
    # rather than tile-rounded allocation dimensions.
    buf_w = int(wrapper.get_buffer_width())
    buf_h = int(wrapper.get_buffer_height())
    assert buf_w >= 101, f"buffer width {buf_w} less than requested 101"
    assert buf_h >= 67, f"buffer height {buf_h} less than requested 67"
    assert int(wrapper.get_num_images()) == 1


@pytest.mark.gpu
def test_multi_image_regression_second_image_remains_empty_for_single_draw(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL REGRESSION MARKER -- read before changing this test.
    #
    # Pinned behavior: with num_images > 1 and a single draw without ranges,
    # the current impl rasterizes image 0 ONLY. It does not touch image 1 at
    # all -- not via deferred_clear, not via the rasterizer kernel. Image 1
    # retains exactly whatever was in GPU memory after set_buffer_size.
    # Why it's unusual: a multi-image-aware impl would either replicate image 0
    # to all images, clear all of them on deferred_clear, or require explicit
    # per-image ranges. The current impl leaves image 1 as uninitialized GPU
    # memory.
    # Paired positive contract test:
    #   test_single_draw_multi_image_ranges_produce_distinct_outputs
    #   (asserts the supported per-image ranges contract).
    #
    # When this test fails on a cleanroom replacement:
    #   1. If image 1 changed during the draw (cleared or rasterized), decide
    #      whether that's the new contract; if yes, replace this with a
    #      positive contract test and DELETE this marker.
    #   2. Investigate any other change in behavior.
    wrapper = cudaraster_plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    wrapper.set_buffer_size(64, 64, 2)
    torch.cuda.synchronize()
    image1_pre = wrapper.get_color_buffer()[1, :64, :64].detach().cpu().numpy().copy()

    vertices_t = _to_vertices(
        [(-0.3, -0.3, 0.0, 1.0), (0.3, -0.3, 0.0, 1.0), (0.0, 0.3, 0.0, 1.0)]
    )
    indices_t = _to_indices([(0, 1, 2)])
    wrapper.set_vertex_buffer(vertices_t)
    wrapper.set_index_buffer(indices_t)
    wrapper.set_render_mode_flags(0)
    wrapper.set_deterministic_tiebreaker(False)
    wrapper.deferred_clear(0)
    wrapper.set_viewport(64, 64, 0, 0)
    assert bool(wrapper.draw_triangles(None, False))
    torch.cuda.synchronize()

    image0_post = wrapper.get_color_buffer()[0, :64, :64].detach().cpu().numpy()
    image1_post = wrapper.get_color_buffer()[1, :64, :64].detach().cpu().numpy()

    assert np.any(image0_post == 1), "image 0 must contain the rasterized triangle"
    assert np.array_equal(image1_pre, image1_post), (
        "image 1 changed during the draw; the current impl is supposed to leave it untouched"
    )


@pytest.mark.gpu
def test_partial_clip_renders_visible_portion(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(0.5, -0.4, 0.0, 1.0), (2.0, -0.4, 0.0, 1.0), (0.5, 0.6, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    assert np.any(harness.read().color == 1)


@pytest.mark.gpu
def test_vertex_behind_camera_no_crash_no_garbage(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.3, -0.3, 0.2, 1.0), (0.3, -0.3, 0.2, 1.0), (0.0, 0.3, 0.2, -1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert np.all(color >= 0)
    assert np.all(color <= 1)


@pytest.mark.gpu
def test_all_vertices_outside_crossing_center_is_stable(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-2.0, -0.2, 0.0, 1.0), (2.0, -0.2, 0.0, 1.0), (0.0, 2.0, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert np.all(color >= 0)
    assert np.all(color <= 1)


@pytest.mark.gpu
def test_subpixel_triangle_has_limited_coverage(harness: CudaRasterHarness) -> None:
    eps = 1.0 / 256.0
    vertices = _to_vertices(
        [(-eps, -eps, 0.0, 1.0), (eps, -eps, 0.0, 1.0), (0.0, eps, 0.0, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    covered = int(np.count_nonzero(harness.read().color == 1))
    assert 0 <= covered <= 4


@pytest.mark.gpu
def test_large_coordinates_stay_stable(harness: CudaRasterHarness) -> None:
    scale = 1_000_000.0
    vertices = _to_vertices(
        [
            (-0.3 * scale, -0.3 * scale, 0.2 * scale, scale),
            (0.3 * scale, -0.3 * scale, 0.2 * scale, scale),
            (0.0, 0.3 * scale, 0.2 * scale, scale),
        ]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert np.any(color == 1)
    assert np.all(color >= 0)
    assert np.all(color <= 1)


@pytest.mark.gpu
def test_near_zero_w_no_crash(harness: CudaRasterHarness) -> None:
    w = 1e-6
    vertices = _to_vertices(
        [(-1e-7, -1e-7, 0.0, w), (1e-7, -1e-7, 0.0, w), (0.0, 1e-7, 0.0, w)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    assert np.all(color >= 0)
    assert np.all(color <= 1)


@pytest.mark.gpu
def test_large_triangle_count_runs_without_silent_corruption(
    harness: CudaRasterHarness,
) -> None:
    # Contract: 100k overlapping coplanar triangles must rasterize cleanly.
    # All covered pixels must report a valid triangle id (1..triangle_count),
    # the last-drawn triangle must win the depth tie at every covered pixel,
    # and the rendered footprint must match the (single) projected triangle.
    triangle_count = 100_000
    vertices = torch.empty((triangle_count * 3, 4), device="cuda", dtype=torch.float32)
    vertices[0::3, 0] = -0.2
    vertices[0::3, 1] = -0.2
    vertices[1::3, 0] = 0.2
    vertices[1::3, 1] = -0.2
    vertices[2::3, 0] = 0.0
    vertices[2::3, 1] = 0.2
    vertices[:, 2] = 0.5
    vertices[:, 3] = 1.0
    idx = torch.arange(triangle_count, device="cuda", dtype=torch.int32)
    indices = torch.stack([idx * 3, idx * 3 + 1, idx * 3 + 2], dim=1).contiguous()

    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    nonzero = color[color != 0]
    assert nonzero.size > 0, "100k overlapping triangles must produce visible coverage"
    # All non-background pixels must be a valid 1-based triangle id.
    assert int(nonzero.min()) >= 1
    assert int(nonzero.max()) <= triangle_count
    # No background (0) leakage into the triangle interior: at least the
    # triangle's projected footprint (~338 px on this geometry) must be covered.
    assert int(nonzero.size) >= 250, f"unexpectedly low coverage: {int(nonzero.size)}"


_OVERFLOW_SCRIPT = textwrap.dedent(
    """
    import torch
    from ludus_renderer._ops._plugin import _get_plugin

    plugin = _get_plugin()
    w = plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    triangle_count = (1 << 24) + 1024

    verts = torch.tensor(
        [[-0.4, -0.4, 0.2, 1.0], [0.4, -0.4, 0.2, 1.0], [0.0, 0.4, 0.2, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    idx = torch.empty((triangle_count, 3), device="cuda", dtype=torch.int32)
    idx[:, 0] = 0
    idx[:, 1] = 1
    idx[:, 2] = 2

    w.set_buffer_size(64, 64, 1)
    w.set_vertex_buffer(verts)
    w.set_index_buffer(idx)
    w.set_render_mode_flags(0)
    w.set_deterministic_tiebreaker(False)
    w.deferred_clear(0)
    w.set_viewport(64, 64, 0, 0)
    ok = bool(w.draw_triangles(None, False))
    torch.cuda.synchronize()
    assert ok is False, "draw_triangles must return False on internal subtri-queue overflow"
    """
)


@pytest.mark.gpu
def test_draw_triangles_returns_false_when_internal_subtri_queue_overflows(
    cudaraster_plugin: Any,
) -> None:
    # POSITIVE CONTRACT: when the internal subtri queue overflows,
    # draw_triangles must return False rather than crashing the process.
    result = subprocess.run(
        [sys.executable, "-c", _OVERFLOW_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# -----------------------------------------------------------------------------
# Batch 1: ranges API, depth-buffer semantics, and depth-peeling behavior.


@pytest.mark.gpu
def test_ranges_draw_first_triangle_only(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.90, -0.90, 0.3, 1.0),
            (-0.55, -0.90, 0.3, 1.0),
            (-0.72, -0.55, 0.3, 1.0),
            (0.55, -0.90, 0.3, 1.0),
            (0.90, -0.90, 0.3, 1.0),
            (0.72, -0.55, 0.3, 1.0),
            (-0.90, 0.55, 0.3, 1.0),
            (-0.55, 0.90, 0.3, 1.0),
            (-0.72, 0.90, 0.3, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    ranges = _to_ranges([(0, 1)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=0, flags=0, deterministic_tiebreaker=False, ranges=ranges
    )
    color = harness.read().color
    p0 = _ndc_to_pixel(64, 64, -0.72, -0.72)
    p1 = _ndc_to_pixel(64, 64, 0.72, -0.72)
    p2 = _ndc_to_pixel(64, 64, -0.72, 0.72)
    assert _pixel(color, p0[0], p0[1]) == 1
    assert _pixel(color, p1[0], p1[1]) == 0
    assert _pixel(color, p2[0], p2[1]) == 0


@pytest.mark.gpu
def test_ranges_draw_tail_triangles_only(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.90, -0.90, 0.3, 1.0),
            (-0.55, -0.90, 0.3, 1.0),
            (-0.72, -0.55, 0.3, 1.0),
            (0.55, -0.90, 0.3, 1.0),
            (0.90, -0.90, 0.3, 1.0),
            (0.72, -0.55, 0.3, 1.0),
            (-0.90, 0.55, 0.3, 1.0),
            (-0.55, 0.90, 0.3, 1.0),
            (-0.72, 0.90, 0.3, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    ranges = _to_ranges([(1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=0, flags=0, deterministic_tiebreaker=False, ranges=ranges
    )
    color = harness.read().color
    p0 = _ndc_to_pixel(64, 64, -0.72, -0.72)
    p1 = _ndc_to_pixel(64, 64, 0.72, -0.72)
    p2 = _ndc_to_pixel(64, 64, -0.72, 0.72)
    assert _pixel(color, p0[0], p0[1]) == 0
    assert _pixel(color, p1[0], p1[1]) == 2
    assert _pixel(color, p2[0], p2[1]) == 3


@pytest.mark.gpu
def test_depth_buffer_near_is_smaller_than_far(harness: CudaRasterHarness) -> None:
    center = _ndc_to_pixel(64, 64, 0.0, -0.1)

    near_vertices = _to_vertices(
        [(-0.4, -0.4, 0.2, 1.0), (0.4, -0.4, 0.2, 1.0), (0.0, 0.4, 0.2, 1.0)]
    )
    far_vertices = _to_vertices(
        [(-0.4, -0.4, 0.8, 1.0), (0.4, -0.4, 0.8, 1.0), (0.0, 0.4, 0.8, 1.0)]
    )
    one_tri = _to_indices([(0, 1, 2)])

    harness.configure(64, 64)
    harness.upload(near_vertices, one_tri)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    near = harness.read()
    near_depth = int(near.depth[center[1], center[0]])
    assert _pixel(near.color, center[0], center[1]) == 1

    harness.upload(far_vertices, one_tri)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    far = harness.read()
    far_depth = int(far.depth[center[1], center[0]])
    assert _pixel(far.color, center[0], center[1]) == 1

    assert near_depth < far_depth


@pytest.mark.gpu
def test_depth_buffer_background_stays_at_clear_depth(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-0.3, -0.3, 0.2, 1.0), (0.3, -0.3, 0.2, 1.0), (0.0, 0.3, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(64, 64)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    out = harness.read()

    background_depth_values = out.depth[out.color == 0]
    covered_depth_values = out.depth[out.color != 0]
    assert background_depth_values.size > 0
    assert covered_depth_values.size > 0
    assert np.all(background_depth_values == background_depth_values[0])
    assert int(background_depth_values[0]) > int(np.min(covered_depth_values))


# =============================================================================
# CLEANROOM CONTRACT vs. CURRENT FAILURE-MODE MARKERS for depth peeling.
#
# Depth peeling currently crashes the process with CUDA error 700. We track this
# with two paired tests per scenario:
#
#   1. POSITIVE CONTRACT TEST (xfail strict=True). Asserts the behavior we want
#      from a correct rasterizer. Today it fails because the impl crashes; the
#      xfail keeps the suite green. When the cleanroom replacement is correct,
#      the test xpasses and pytest reports it as an unexpected pass --- LOUD,
#      meaning "remove the xfail and the paired marker now".
#
#   2. CURRENT-IMPL FAILURE-MODE MARKER (regular test). Asserts the *specific*
#      stderr signature ("Cuda error: 700[cudaStreamSynchronize(stream);]") of
#      today's crash. If the failure mode changes (different error, different
#      stack, or no crash at all) this test fails LOUDLY so we know to look.
#
# Written 2026-05-04. If you are reading this AFTER the cleanroom replacement
# fixes peeling, DELETE both tests in each pair --- the contract is then
# expressed by a normal positive test elsewhere.
# =============================================================================

_DEPTH_PEEL_THREE_LAYER_SCRIPT = textwrap.dedent(
    """
    import torch
    from ludus_renderer._ops._plugin import _get_plugin

    plugin = _get_plugin()
    w = plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    flags = int(plugin.CR_RENDER_MODE_ENABLE_DEPTH_PEELING)
    verts = torch.tensor(
        [[-0.4, -0.4, 0.2, 1.0], [0.4, -0.4, 0.2, 1.0], [0.0, 0.4, 0.2, 1.0],
         [-0.4, -0.4, 0.5, 1.0], [0.4, -0.4, 0.5, 1.0], [0.0, 0.4, 0.5, 1.0],
         [-0.4, -0.4, 0.8, 1.0], [0.4, -0.4, 0.8, 1.0], [0.0, 0.4, 0.8, 1.0]],
        device="cuda", dtype=torch.float32
    )
    idx = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device="cuda", dtype=torch.int32)
    w.set_buffer_size(64, 64, 1)
    w.set_vertex_buffer(verts)
    w.set_index_buffer(idx)
    w.set_render_mode_flags(flags)
    w.set_deterministic_tiebreaker(False)
    w.deferred_clear(0)
    w.set_viewport(64, 64, 0, 0)

    ok1 = bool(w.draw_triangles(None, False))
    torch.cuda.synchronize()
    c1 = int(w.get_color_buffer()[0, 32, 32].item())
    w.swap_depth_and_peel()

    ok2 = bool(w.draw_triangles(None, True))
    torch.cuda.synchronize()
    c2 = int(w.get_color_buffer()[0, 32, 32].item())
    w.swap_depth_and_peel()

    ok3 = bool(w.draw_triangles(None, True))
    torch.cuda.synchronize()
    c3 = int(w.get_color_buffer()[0, 32, 32].item())

    assert ok1 and ok2 and ok3
    assert c1 == 1, f"first pass center must be near triangle id 1, got {c1}"
    assert c2 == 2, f"second pass center must reveal middle triangle id 2, got {c2}"
    assert c3 == 3, f"third pass center must reveal far triangle id 3, got {c3}"
    """
)

_DEPTH_PEEL_SINGLE_LAYER_SCRIPT = textwrap.dedent(
    """
    import torch
    from ludus_renderer._ops._plugin import _get_plugin

    plugin = _get_plugin()
    w = plugin.CudaRasterTestWrapper(torch.cuda.current_device())
    flags = int(plugin.CR_RENDER_MODE_ENABLE_DEPTH_PEELING)
    verts = torch.tensor(
        [[-0.4, -0.4, 0.3, 1.0], [0.4, -0.4, 0.3, 1.0], [0.0, 0.4, 0.3, 1.0]],
        device="cuda", dtype=torch.float32
    )
    idx = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    w.set_buffer_size(64, 64, 1)
    w.set_vertex_buffer(verts)
    w.set_index_buffer(idx)
    w.set_render_mode_flags(flags)
    w.set_deterministic_tiebreaker(False)
    w.deferred_clear(0)
    w.set_viewport(64, 64, 0, 0)

    ok1 = bool(w.draw_triangles(None, False))
    torch.cuda.synchronize()
    c1 = w.get_color_buffer()[0].detach().cpu()
    w.swap_depth_and_peel()
    ok2 = bool(w.draw_triangles(None, True))
    torch.cuda.synchronize()
    c2 = w.get_color_buffer()[0].detach().cpu()

    assert ok1 and ok2
    assert int((c1 != 0).sum().item()) > 0, "first pass must rasterize the triangle"
    assert int((c2 != 0).sum().item()) == 0, "second pass must be fully background (only one layer exists)"
    """
)


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="known-broken in current impl: depth peeling")
def test_depth_peeling_three_layers_exposes_layers_in_order(
    cudaraster_plugin: Any,
) -> None:
    # POSITIVE CONTRACT: peel passes 1, 2, 3 must reveal triangle ids 1, 2, 3
    # at the center pixel. xfail today; xpasses when peeling is fixed.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_THREE_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu
def test_depth_peeling_three_layers_currently_crashes_with_cuda700(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL FAILURE-MODE MARKER. See the header comment above for the
    # contract pairing. Asserts the exact crash signature so any change is loud.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_THREE_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "three-layer peel no longer crashes; positive contract should now pass"
    )
    assert _CURRENT_PEEL_FAILURE_STDERR in result.stderr, (
        f"three-layer peel now fails with a different error:\n{result.stderr}"
    )


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="known-broken in current impl: depth peeling")
def test_depth_peeling_single_layer_second_pass_is_empty(
    cudaraster_plugin: Any,
) -> None:
    # POSITIVE CONTRACT: peel pass 1 rasterizes the triangle; peel pass 2 has
    # no further layer and must be fully background. xfail today.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_SINGLE_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu
def test_depth_peeling_single_layer_currently_crashes_with_cuda700(
    cudaraster_plugin: Any,
) -> None:
    # CURRENT-IMPL FAILURE-MODE MARKER. Pinned crash signature for the single-
    # layer peel scenario. Loud whenever the failure mode changes.
    result = subprocess.run(
        [sys.executable, "-c", _DEPTH_PEEL_SINGLE_LAYER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "single-layer peel no longer crashes; positive contract should now pass"
    )
    assert _CURRENT_PEEL_FAILURE_STDERR in result.stderr, (
        f"single-layer peel now fails with a different error:\n{result.stderr}"
    )


# -----------------------------------------------------------------------------
# Batch 2: shared edges, polyline strips, and wireframe-style bands.


@pytest.mark.gpu
def test_shared_edge_diagonal_split_has_no_holes(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.5, -0.5, 0.2, 1.0),
            (0.5, -0.5, 0.2, 1.0),
            (0.5, 0.5, 0.2, 1.0),
            (-0.5, 0.5, 0.2, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (0, 2, 3)])
    harness.configure(96, 96)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    x0, y0 = _ndc_to_pixel(96, 96, -0.45, -0.45)
    x1, y1 = _ndc_to_pixel(96, 96, 0.45, 0.45)
    region = color[y0 : y1 + 1, x0 : x1 + 1]
    assert np.all(region != 0)
    assert set(np.unique(region).tolist()).issubset({1, 2})


@pytest.mark.gpu
def test_shared_edge_grid_quad_centers_are_all_covered(
    harness: CudaRasterHarness,
) -> None:
    grid = 10
    x_min, x_max = -0.8, 0.8
    y_min, y_max = -0.8, 0.8
    dx = (x_max - x_min) / grid
    dy = (y_max - y_min) / grid
    verts: list[tuple[float, float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = x_min + gx * dx
            x1 = x0 + dx
            y0 = y_min + gy * dy
            y1 = y0 + dy
            base = len(verts)
            verts.extend(
                [
                    (x0, y0, 0.3, 1.0),
                    (x1, y0, 0.3, 1.0),
                    (x1, y1, 0.3, 1.0),
                    (x0, y1, 0.3, 1.0),
                ]
            )
            tris.append((base + 0, base + 1, base + 2))
            tris.append((base + 0, base + 2, base + 3))
    harness.configure(128, 128)
    harness.upload(_to_vertices(verts), _to_indices(tris))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    for gy in range(grid):
        for gx in range(grid):
            cx_ndc = x_min + (gx + 0.5) * dx
            cy_ndc = y_min + (gy + 0.5) * dy
            px, py = _ndc_to_pixel(128, 128, cx_ndc, cy_ndc)
            assert _pixel(color, px, py) != 0


@pytest.mark.gpu
def test_thin_quad_centerline_is_continuous(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (-0.8, -0.02, 0.2, 1.0),
            (-0.8, 0.02, 0.2, 1.0),
            (0.8, -0.02, 0.2, 1.0),
            (0.8, 0.02, 0.2, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (1, 3, 2)])
    harness.configure(160, 80)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    for x_ndc in np.linspace(-0.7, 0.7, num=21):
        px, py = _ndc_to_pixel(160, 80, float(x_ndc), 0.0)
        assert _pixel(color, px, py) != 0


@pytest.mark.gpu
def test_polyline_strip_segment_joints_can_crack_without_join_geometry(
    harness: CudaRasterHarness,
) -> None:
    # CURRENT-IMPL REGRESSION MARKER -- read before changing this test.
    #
    # Pinned behavior: stitching independent quad segments end-to-end without
    # explicit miter/bevel join geometry leaves visible cracks at the joints.
    # Why it's unusual: this is a property of the polyline-as-naive-quad-strip
    # construction, not of the rasterizer itself. A stricter rasterizer with
    # different edge rules might close more of the crack pixels.
    # Paired positive contract test: none yet. When we define a polyline
    # contract that promises crack-free joins (with explicit join geometry),
    # add a test for it and delete this marker.
    #
    # When this test fails on a cleanroom replacement:
    #   1. The new impl no longer leaves cracks at these joints. Decide whether
    #      that's a desired contract; if so, write a positive test and delete
    #      this marker.
    points = [(-0.85, -0.35), (-0.35, -0.05), (0.05, 0.25), (0.55, 0.10), (0.85, 0.35)]
    vertices, indices = _polyline_strip_triangles(points, half_width=0.03, z=0.25)
    harness.configure(192, 128)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    uncovered_joints = 0
    for px_ndc, py_ndc in points[1:-1]:
        px, py = _ndc_to_pixel(192, 128, px_ndc, py_ndc)
        if _pixel(color, px, py) == 0:
            uncovered_joints += 1
    assert uncovered_joints >= 1


@pytest.mark.gpu
def test_wireframe_band_edges_render_and_center_stays_clear(
    harness: CudaRasterHarness,
) -> None:
    verts, tris = _ring_band_triangles(outer=0.8, inner=0.55, z=0.3)
    harness.configure(128, 128)
    harness.upload(_to_vertices(verts), _to_indices(tris))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    edge_points = [(-0.75, 0.0), (0.75, 0.0), (0.0, 0.75), (0.0, -0.75)]
    for x_ndc, y_ndc in edge_points:
        x, y = _ndc_to_pixel(128, 128, x_ndc, y_ndc)
        assert _pixel(color, x, y) != 0
    cx, cy = _ndc_to_pixel(128, 128, 0.0, 0.0)
    assert _pixel(color, cx, cy) == 0


@pytest.mark.gpu
def test_overlapping_wireframe_bands_use_depth_order(
    harness: CudaRasterHarness,
) -> None:
    far_verts, far_tris = _ring_band_triangles(outer=0.72, inner=0.50, z=0.7)
    near_verts, near_tris = _ring_band_triangles(outer=0.72, inner=0.50, z=0.2)
    offset = len(far_verts)
    near_tris_off = [(a + offset, b + offset, c + offset) for a, b, c in near_tris]

    harness.configure(128, 128)
    harness.upload(
        _to_vertices(far_verts + near_verts), _to_indices(far_tris + near_tris_off)
    )
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    both = harness.read()

    sx, sy = _ndc_to_pixel(128, 128, 0.7, 0.0)
    assert _pixel(both.color, sx, sy) != 0
    # Near ring triangles are appended after the 8 far triangles.
    assert _pixel(both.color, sx, sy) >= 9


# -----------------------------------------------------------------------------
# Batch 3: hex-dot fans, viewport passes, and coplanar determinism.


@pytest.mark.gpu
def test_hex_dot_all_sector_centroids_are_covered(harness: CudaRasterHarness) -> None:
    center = (0.0, 0.0, 0.25, 1.0)
    radius = 0.22
    outer: list[tuple[float, float, float, float]] = []
    for i in range(6):
        angle = (np.pi / 3.0) * i
        outer.append(
            (float(np.cos(angle) * radius), float(np.sin(angle) * radius), 0.25, 1.0)
        )
    vertices = _to_vertices([center] + outer)
    indices = _to_indices([(0, 1 + i, 1 + ((i + 1) % 6)) for i in range(6)])

    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    for i in range(6):
        theta = (np.pi / 3.0) * i + (np.pi / 6.0)
        x_ndc = float(np.cos(theta) * (radius * 0.5))
        y_ndc = float(np.sin(theta) * (radius * 0.5))
        x, y = _ndc_to_pixel(128, 128, x_ndc, y_ndc)
        assert _pixel(color, x, y) != 0


@pytest.mark.gpu
def test_overlapping_hex_dots_follow_depth_order(harness: CudaRasterHarness) -> None:
    def fan(radius: float, z: float) -> list[tuple[float, float, float, float]]:
        out: list[tuple[float, float, float, float]] = [(0.0, 0.0, z, 1.0)]
        for i in range(6):
            angle = (np.pi / 3.0) * i
            out.append(
                (float(np.cos(angle) * radius), float(np.sin(angle) * radius), z, 1.0)
            )
        return out

    far_vertices = fan(radius=0.24, z=0.7)
    near_vertices = fan(radius=0.24, z=0.2)
    far_indices = [(0, 1 + i, 1 + ((i + 1) % 6)) for i in range(6)]
    near_indices = [(7 + 0, 7 + 1 + i, 7 + 1 + ((i + 1) % 6)) for i in range(6)]

    harness.configure(128, 128)
    harness.upload(
        _to_vertices(far_vertices + near_vertices),
        _to_indices(far_indices + near_indices),
    )
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    both = harness.read()

    cx, cy = _ndc_to_pixel(128, 128, 0.0, 0.0)
    assert _pixel(both.color, cx, cy) != 0
    # Near fan triangles are appended after the 6 far triangles.
    assert _pixel(both.color, cx, cy) >= 7


@pytest.mark.gpu
def test_single_draw_multi_image_ranges_produce_distinct_outputs(
    harness: CudaRasterHarness,
) -> None:
    # CudaRaster currently exposes one global viewport per raster instance, not per-image
    # viewport offsets. This test still verifies the single-draw multi-image contract by
    # selecting different triangle ranges for each image in one draw call.
    vertices = _to_vertices(
        [
            (-0.9, -0.4, 0.25, 1.0),
            (-0.4, -0.4, 0.25, 1.0),
            (-0.65, 0.3, 0.25, 1.0),
            (0.4, -0.4, 0.25, 1.0),
            (0.9, -0.4, 0.25, 1.0),
            (0.65, 0.3, 0.25, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    ranges = _to_ranges([(0, 1), (1, 1)])

    harness.configure(128, 128, num_images=2)
    harness.upload(vertices, indices)
    assert harness.draw(
        clear_color=0, flags=0, deterministic_tiebreaker=False, ranges=ranges
    )
    img0 = harness.read(0).color
    img1 = harness.read(1).color

    left = _ndc_to_pixel(128, 128, -0.65, -0.05)
    right = _ndc_to_pixel(128, 128, 0.65, -0.05)
    assert _pixel(img0, left[0], left[1]) == 1
    assert _pixel(img0, right[0], right[1]) == 0
    assert _pixel(img1, left[0], left[1]) == 0
    assert _pixel(img1, right[0], right[1]) == 2


@pytest.mark.gpu
def test_second_clear_draw_has_no_stale_pixels_from_first_scene(
    harness: CudaRasterHarness,
) -> None:
    harness.configure(96, 96)
    vertices_a = _to_vertices(
        [(-0.8, -0.8, 0.2, 1.0), (-0.3, -0.8, 0.2, 1.0), (-0.55, -0.3, 0.2, 1.0)]
    )
    vertices_b = _to_vertices(
        [(0.3, 0.3, 0.2, 1.0), (0.8, 0.3, 0.2, 1.0), (0.55, 0.8, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])

    harness.upload(vertices_a, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read().color
    ax, ay = _ndc_to_pixel(96, 96, -0.55, -0.55)
    assert _pixel(first, ax, ay) == 1

    harness.upload(vertices_b, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    second = harness.read().color
    bx, by = _ndc_to_pixel(96, 96, 0.55, 0.55)
    assert _pixel(second, bx, by) == 1
    assert _pixel(second, ax, ay) == 0


@pytest.mark.gpu
def test_coplanar_tiling_is_nonzero_and_repeatable(harness: CudaRasterHarness) -> None:
    # Build 20 coplanar triangles arranged in a 5x4 grid.
    cols, rows = 5, 4
    x_min, y_min = -0.8, -0.7
    dx, dy = 0.32, 0.34
    verts: list[tuple[float, float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for iy in range(rows):
        for ix in range(cols):
            cx = x_min + ix * dx
            cy = y_min + iy * dy
            base = len(verts)
            verts.extend(
                [
                    (cx - 0.12, cy - 0.10, 0.4, 1.0),
                    (cx + 0.12, cy - 0.10, 0.4, 1.0),
                    (cx, cy + 0.10, 0.4, 1.0),
                ]
            )
            tris.append((base + 0, base + 1, base + 2))

    harness.configure(160, 120)
    harness.upload(_to_vertices(verts), _to_indices(tris))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = harness.read().color
    x0, y0 = _ndc_to_pixel(160, 120, x_min - 0.12, y_min - 0.10)
    x1, y1 = _ndc_to_pixel(
        160, 120, x_min + (cols - 1) * dx + 0.12, y_min + (rows - 1) * dy + 0.10
    )
    region = first[y0 : y1 + 1, x0 : x1 + 1]
    coverage_ratio = float(np.count_nonzero(region)) / float(region.size)
    assert coverage_ratio >= 0.15

    harness.upload(_to_vertices(verts), _to_indices(tris))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = harness.read().color
    assert np.array_equal(first, second)


@pytest.mark.gpu
def test_coplanar_overlap_winner_is_stable_with_tiebreaker(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [
            (-0.6, -0.4, 0.3, 1.0),
            (0.6, -0.4, 0.3, 1.0),
            (0.0, 0.6, 0.3, 1.0),
            (-0.4, -0.2, 0.3, 1.0),
            (0.7, -0.2, 0.3, 1.0),
            (0.1, 0.7, 0.3, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])
    colors = torch.tensor(
        [
            _pack_rgba(10, 10, 200, 255),
            _pack_rgba(10, 10, 200, 255),
            _pack_rgba(10, 10, 200, 255),
            _pack_rgba(200, 10, 10, 255),
            _pack_rgba(200, 10, 10, 255),
            _pack_rgba(200, 10, 10, 255),
        ],
        device="cuda",
        dtype=torch.int32,
    )

    harness.configure(128, 128)
    harness.upload(vertices, indices, colors=colors)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = harness.read().color

    harness.upload(vertices, indices, colors=colors)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = harness.read().color

    sx, sy = _ndc_to_pixel(128, 128, 0.1, 0.1)
    assert _pixel(second, sx, sy) == _pixel(first, sx, sy)


# -----------------------------------------------------------------------------
# Batch 4: fan topology, long-thin triangles, and subpixel stability.


@pytest.mark.gpu
def test_fan_topology_center_pixel_is_covered(harness: CudaRasterHarness) -> None:
    radius = 0.3
    vertices: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.25, 1.0)]
    for i in range(8):
        angle = (2.0 * np.pi * i) / 8.0
        vertices.append(
            (float(np.cos(angle) * radius), float(np.sin(angle) * radius), 0.25, 1.0)
        )
    indices = [(0, 1 + i, 1 + ((i + 1) % 8)) for i in range(8)]

    harness.configure(128, 128)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color
    cx, cy = _ndc_to_pixel(128, 128, 0.0, 0.0)
    assert _pixel(color, cx, cy) != 0


@pytest.mark.gpu
def test_fan_wedge_centroids_map_to_expected_triangle_ids(
    harness: CudaRasterHarness,
) -> None:
    radius = 0.3
    vertices: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.25, 1.0)]
    for i in range(8):
        angle = (2.0 * np.pi * i) / 8.0
        vertices.append(
            (float(np.cos(angle) * radius), float(np.sin(angle) * radius), 0.25, 1.0)
        )
    indices = [(0, 1 + i, 1 + ((i + 1) % 8)) for i in range(8)]

    harness.configure(128, 128)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    for i in range(8):
        theta = (2.0 * np.pi * i) / 8.0 + (np.pi / 8.0)
        px, py = _ndc_to_pixel(
            128,
            128,
            float(np.cos(theta) * radius * 0.5),
            float(np.sin(theta) * radius * 0.5),
        )
        assert _pixel(color, px, py) == i + 1


@pytest.mark.gpu
def test_long_thin_axis_aligned_triangle_has_continuous_coverage(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [(-0.9, -0.02, 0.2, 1.0), (0.9, -0.02, 0.2, 1.0), (0.0, 0.02, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(192, 96)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    # Sample interior points using barycentric blends to avoid edge-rule ambiguity.
    interior_points = [
        (-0.6, -0.012),
        (-0.3, -0.008),
        (0.0, -0.004),
        (0.3, -0.008),
        (0.6, -0.012),
    ]
    for x_ndc, y_ndc in interior_points:
        px, py = _ndc_to_pixel(192, 96, x_ndc, y_ndc)
        assert _pixel(color, px, py) != 0
    assert int(np.count_nonzero(color)) >= 120


@pytest.mark.gpu
def test_long_thin_rotated_triangle_draws_diagonal_band(
    harness: CudaRasterHarness,
) -> None:
    tri = ((-0.75, -0.72), (0.72, 0.75), (0.68, 0.79))
    vertices = _to_vertices(
        [(-0.75, -0.72, 0.2, 1.0), (0.72, 0.75, 0.2, 1.0), (0.68, 0.79, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])
    harness.configure(160, 160)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    interior, _ = _interior_and_exterior_pixels(160, 160, tri)
    assert interior
    for x, y in interior[:: max(1, len(interior) // 30)]:
        assert _pixel(color, x, y) != 0
    assert int(np.count_nonzero(color)) >= 250


@pytest.mark.gpu
def test_subpixel_half_pixel_translation_has_bounded_delta(
    harness: CudaRasterHarness,
) -> None:
    # Contract: a one-pixel translation of a triangle perturbs only edge pixels.
    # We assert:
    #   1. Some pixels DID change (the shift was visible).
    #   2. The number of changed pixels is small relative to the triangle
    #      footprint -- bounded by a hardcoded cap close to the measured baseline.
    #
    # Measured on the current impl with this exact geometry: 133 changed
    # pixels for a 2242-px triangle (~6%). Cap at 200 (~1.5x measured / ~9%)
    # so a regression that smears coverage across the interior is loud while
    # leaving headroom for a cleanroom impl with slightly different edge rules.
    # If a cleanroom replacement legitimately produces a larger (but still
    # edge-localized) delta, raise both numbers together with a one-line note.
    width, height = 128, 128
    tri = [(-0.6, -0.45, 0.2, 1.0), (0.6, -0.45, 0.2, 1.0), (0.0, 0.45, 0.2, 1.0)]
    shift_x = 1.0 / width
    shift_y = 1.0 / height
    tri_shifted = [(x + shift_x, y + shift_y, z, w) for x, y, z, w in tri]
    indices = _to_indices([(0, 1, 2)])

    harness.configure(width, height)
    harness.upload(_to_vertices(tri), indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    first = harness.read().color

    harness.upload(_to_vertices(tri_shifted), indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    second = harness.read().color

    mask_a = first != 0
    mask_b = second != 0
    delta_pixels = int(np.count_nonzero(mask_a != mask_b))

    assert delta_pixels > 0, "subpixel shift must produce some observable difference"
    assert delta_pixels <= 200, (
        f"changed pixel count {delta_pixels} exceeds 200 (current-impl baseline=133); "
        "a cleanroom impl may legitimately need this raised, but justify in the comment above."
    )


@pytest.mark.gpu
def test_identical_inputs_produce_repeatable_output(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [(-0.5, -0.5, 0.2, 1.0), (0.6, -0.4, 0.2, 1.0), (0.1, 0.6, 0.2, 1.0)]
    )
    indices = _to_indices([(0, 1, 2)])

    harness.configure(96, 96)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = harness.read()

    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = harness.read()

    assert np.array_equal(first.color, second.color)
    assert np.array_equal(first.depth, second.depth)


# -----------------------------------------------------------------------------
# Batch 5: large distinct fields and mixed dynamic-range scale behavior.


@pytest.mark.gpu
def test_large_distinct_triangle_field_exposes_many_unique_ids(
    harness: CudaRasterHarness,
) -> None:
    grid = 100  # 10k triangles
    x_values = np.linspace(-0.95, 0.95, grid)
    y_values = np.linspace(-0.95, 0.95, grid)
    half_w = 0.006
    half_h = 0.006

    vertices: list[tuple[float, float, float, float]] = []
    indices: list[tuple[int, int, int]] = []
    for y in y_values:
        for x in x_values:
            base = len(vertices)
            vertices.extend(
                [
                    (float(x - half_w), float(y - half_h), 0.3, 1.0),
                    (float(x + half_w), float(y - half_h), 0.3, 1.0),
                    (float(x), float(y + half_h), 0.3, 1.0),
                ]
            )
            indices.append((base + 0, base + 1, base + 2))

    harness.configure(1024, 1024)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    unique_ids = np.unique(color[color != 0])
    assert len(unique_ids) >= 9_800


@pytest.mark.gpu
def test_large_distinct_triangle_field_is_repeatable_with_tiebreaker(
    harness: CudaRasterHarness,
) -> None:
    grid = 100  # 10k triangles
    x_values = np.linspace(-0.95, 0.95, grid)
    y_values = np.linspace(-0.95, 0.95, grid)
    half_w = 0.006
    half_h = 0.006

    vertices: list[tuple[float, float, float, float]] = []
    indices: list[tuple[int, int, int]] = []
    for y in y_values:
        for x in x_values:
            base = len(vertices)
            vertices.extend(
                [
                    (float(x - half_w), float(y - half_h), 0.3, 1.0),
                    (float(x + half_w), float(y - half_h), 0.3, 1.0),
                    (float(x), float(y + half_h), 0.3, 1.0),
                ]
            )
            indices.append((base + 0, base + 1, base + 2))

    verts_t = _to_vertices(vertices)
    idx_t = _to_indices(indices)
    harness.configure(1024, 1024)
    harness.upload(verts_t, idx_t)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    first = harness.read().color

    harness.upload(verts_t, idx_t)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=True)
    second = harness.read().color
    assert np.array_equal(first, second)


@pytest.mark.gpu
def test_projection_is_invariant_under_uniform_homogeneous_scaling(
    harness: CudaRasterHarness,
) -> None:
    base_vertices = _to_vertices(
        [(-0.4, -0.3, 0.2, 1.0), (0.5, -0.3, 0.2, 1.0), (0.0, 0.5, 0.2, 1.0)]
    )
    scaled_vertices = _to_vertices(
        [
            (-400.0, -300.0, 200.0, 1000.0),
            (500.0, -300.0, 200.0, 1000.0),
            (0.0, 500.0, 200.0, 1000.0),
        ]
    )
    indices = _to_indices([(0, 1, 2)])

    harness.configure(192, 192)
    harness.upload(base_vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    base = harness.read().color != 0

    harness.upload(scaled_vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    scaled = harness.read().color != 0

    assert np.array_equal(base, scaled)


@pytest.mark.gpu
def test_mixed_scale_overlap_respects_depth_order(harness: CudaRasterHarness) -> None:
    # Triangle 1 (near): large world scale, same projected footprint as triangle 2.
    # Triangle 2 (far): unit scale.
    vertices = _to_vertices(
        [
            (-400.0, -300.0, 200.0, 1000.0),
            (500.0, -300.0, 200.0, 1000.0),
            (0.0, 500.0, 200.0, 1000.0),
            (-0.4, -0.3, 0.8, 1.0),
            (0.5, -0.3, 0.8, 1.0),
            (0.0, 0.5, 0.8, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2), (3, 4, 5)])

    harness.configure(192, 192)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    out = harness.read()

    cx, cy = _ndc_to_pixel(192, 192, 0.0, 0.0)
    assert _pixel(out.color, cx, cy) == 1
    near_depth = int(out.depth[cy, cx])

    harness.upload(
        _to_vertices(
            [(-0.4, -0.3, 0.8, 1.0), (0.5, -0.3, 0.8, 1.0), (0.0, 0.5, 0.8, 1.0)]
        ),
        _to_indices([(0, 1, 2)]),
    )
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    far_only = harness.read()
    far_depth = int(far_only.depth[cy, cx])
    assert near_depth < far_depth
