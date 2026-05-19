"""Focused invariants for the CudaRaster cleanroom port.

These tests pin small, risky porting assumptions that are easier to verify
directly than through the broad API contract suite.
"""

import sys
from pathlib import Path
from typing import Any

# Allow bare import of the sibling test module under --import-mode=importlib
# (pytest's importlib mode does not add the test directory to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import torch
import torch.utils.cpp_extension
from ludus_renderer._ops._plugin import _get_plugin
from test_cudaraster_api import (
    CudaRasterHarness,
    _ndc_to_pixel,
    _pixel,
    _require_cuda,
    _to_indices,
    _to_vertices,
)

pytestmark = pytest.mark.ci_gpu

ROOT = Path(__file__).resolve().parents[1] / "ludus_renderer" / "_cpp" / "cudaraster"


@pytest.fixture(scope="module")
def cudaraster_plugin() -> Any:
    _require_cuda()
    return _get_plugin()


@pytest.fixture
def harness(cudaraster_plugin: Any) -> CudaRasterHarness:
    return CudaRasterHarness(cudaraster_plugin)


@pytest.fixture(scope="module")
def rop_lane_mask_helper() -> Any:
    _require_cuda()
    helper_src = Path(__file__).with_name("cuda") / "rop_lane_mask_invariant.cu"
    return torch.utils.cpp_extension.load(
        name="cudaraster_rop_lane_mask_invariant",
        sources=[str(helper_src)],
        extra_include_paths=[str(ROOT), str(ROOT / "framework")],
        extra_cflags=["-DFW_DO_NOT_OVERRIDE_NEW_DELETE"],
        extra_cuda_cflags=["-DFW_DO_NOT_OVERRIDE_NEW_DELETE", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


@pytest.fixture(scope="module")
def bin_raster_arbitration_helper() -> Any:
    _require_cuda()
    helper_src = (
        Path(__file__).with_name("cuda") / "bin_raster_arbitration_invariants.cu"
    )
    return torch.utils.cpp_extension.load(
        name="cudaraster_bin_raster_arbitration_invariants",
        sources=[str(helper_src)],
        extra_include_paths=[str(ROOT), str(ROOT / "framework")],
        extra_cflags=["-DFW_DO_NOT_OVERRIDE_NEW_DELETE"],
        extra_cuda_cflags=["-DFW_DO_NOT_OVERRIDE_NEW_DELETE", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


@pytest.mark.gpu
def test_rop_lane_mask_replacement_matches_upstream_arbitration_order(
    rop_lane_mask_helper: Any,
) -> None:
    values = list(rop_lane_mask_helper.run_rop_lane_mask_invariant())
    assert len(values) == 64

    cases = [
        ("reverse", 0, lambda lane: (1 << lane) - 1),
        (
            "forward",
            32,
            lambda lane: (0xFFFFFFFF ^ ((1 << (lane + 1)) - 1)) & 0xFFFFFFFF,
        ),
    ]
    for label, offset, expected_for_lane in cases:
        replacement = [int(v) for v in values[offset : offset + 32]]
        expected = [expected_for_lane(lane) for lane in range(32)]

        assert replacement == expected, label
        # The fine raster only requires that __popc(mask) is a permutation of
        # [0, 31] across the warp -- it is used as a unique per-lane index into
        # a 32-slot scratch buffer.
        assert sorted(mask.bit_count() for mask in replacement) == list(range(32)), (
            label
        )


@pytest.mark.gpu
def test_clipped_cw_triangle_renders_with_backface_culling_disabled(
    harness: CudaRasterHarness,
) -> None:
    vertices = _to_vertices(
        [
            (
                1.6,
                -0.3,
                0.2,
                1.0,
            ),  # v0 is outside +X, forcing the clipped-subtriangle path.
            (-0.2, -0.5, 0.2, 1.0),
            (-0.2, 0.5, 0.2, 1.0),
        ]
    )
    indices = _to_indices([(0, 1, 2)])  # CW in screen space before clipping.

    harness.configure(128, 128)
    harness.upload(vertices, indices)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    probe_points = [(0.2, -0.05), (0.4, -0.1), (0.7, -0.2)]
    for x_ndc, y_ndc in probe_points:
        x, y = _ndc_to_pixel(128, 128, x_ndc, y_ndc)
        assert _pixel(color, x, y) != 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "label, nums",
    [
        ("ramp_lane_idx_mod_8", [(lane % 8) for lane in range(32)]),
        ("dense_max_three_bits", [7] * 32),
        ("sparse_one_lane_only", [0] * 31 + [5]),
        ("alternating_zero_one", [(lane & 1) for lane in range(32)]),
        ("zero_warp", [0] * 32),
    ],
)
def test_bin_raster_warp_total_broadcast_lands_warp_total(
    bin_raster_arbitration_helper: Any, label: str, nums: list[int]
) -> None:
    # Pins BinRaster.inl Fix A: only lane 31 may write `s_broadcast[warpId+16]`
    # with `myIdx + num`. With this gate, the broadcast slot must equal the
    # warp total regardless of lane store ordering. If the gate moves to a
    # different lane (or is removed), this slot would receive that lane's
    # partial prefix or an undefined value under ITS.
    actual = int(bin_raster_arbitration_helper.run_warp_total_broadcast(nums))
    expected = sum(nums)
    assert actual == expected, label


@pytest.mark.gpu
@pytest.mark.parametrize(
    "label, totals",
    [
        ("ramp_one_through_sixteen", list(range(1, 17))),
        ("uniform_five_each", [5] * 16),
        ("sparse_first_warp_only", [11] + [0] * 15),
        ("sparse_last_warp_only", [0] * 15 + [11]),
        ("zero_block", [0] * 16),
    ],
)
def test_bin_raster_block_total_lands_inclusive_scan_total(
    bin_raster_arbitration_helper: Any, label: str, totals: list[int]
) -> None:
    # Pins BinRaster.inl Fix B: only lane (CR_BIN_WARPS - 1) may write
    # `s_bufCount = bufCount + val`. With this gate, the broadcast slot must
    # equal the inclusive scan's last value regardless of lane store
    # ordering. The inclusive-scan output array is also returned so we can
    # assert that the upstream Hillis-Steele step pattern produces the
    # canonical inclusive prefix sum (i.e. lane k = sum(totals[0..k])).
    out = bin_raster_arbitration_helper.run_block_total_inclusive_scan(totals)
    prefix = [int(v) for v in out["prefix"]]
    actual_buf = int(out["buf_count"])

    expected_prefix = []
    running = 0
    for value in totals:
        running += value
        expected_prefix.append(running)

    assert prefix == expected_prefix, (
        f"{label}: prefix scan diverged from inclusive sum"
    )
    assert actual_buf == expected_prefix[-1], (
        f"{label}: s_bufCount must equal block total"
    )


# -----------------------------------------------------------------------------
# Volta+ ITS warp-sync stress tests.
#
# After the CoarseRaster Case B atomicOr fix (CoarseRaster.inl, the ballot
# write inside the per-cell loop), the cudaraster kernels still contain at
# least seven Hillis-Steele scan patterns of the form
#
#     volatile U32* p = &shared[lane + 16];
#     p[0] = v;  v = combine(v, p[-1]);
#     p[0] = v;  v = combine(v, p[-2]);
#     ...
#
# These scans assume pre-Volta lockstep SIMT: each step's `p[0] = v` is
# visible to neighbours' next-step reads without explicit synchronisation.
# `volatile` orders accesses within one thread but does nothing across lanes.
# Under Volta+ Independent Thread Scheduling (ITS) lanes can advance
# independently and read stale neighbour values.
#
# The most divergent scan -- CoarseRaster's tile-emit prefix sum at
# CoarseRaster.inl L509-L515 -- has explicit per-lane divergence
# (`if (threadIdx.x >= K)`) baked into each step. The other scans are less
# divergent in the body but still rely on warp-synchronous execution.
#
# The three tests below stress the production pipeline with geometry that
# pushes these scans into their hot regimes:
#   * test_front_occluder_no_back_id_leaks_in_full_footprint:
#       Generic depth-overlap leakage probe; catches drops at any stage that
#       cause the front triangle to be missed (back leaks through).
#   * test_dense_overlap_in_single_tile_frontmost_triangle_wins:
#       Pushes a full warp's worth of triangles into one fine-raster tile,
#       stressing FineRaster's scan32_value fragment-count scan.
#   * test_per_tile_emit_imbalance_does_not_misroute_triangles:
#       Builds a per-warp emit imbalance (one heavy tile, many light tiles in
#       the same coarse-raster scan window), stressing CoarseRaster's
#       divergent tile-emit prefix sum.
#
# These tests pin the *visible behaviour* of the production pipeline. They
# don't isolate which scan is broken -- they fail any time any scan in the
# pipeline drops a triangle from a tile. Once they fail, narrowing down to
# the specific scan requires kernel-level instrumentation (the same
# DELETEME-printf workflow we used for the CoarseRaster Case B bug).
# -----------------------------------------------------------------------------


@pytest.mark.gpu
def test_front_occluder_no_back_id_leaks_in_full_footprint(
    harness: CudaRasterHarness,
) -> None:
    # A small front triangle entirely inside a larger back triangle, with the
    # front in front (smaller z). Across the full projected footprint of the
    # front, every covered pixel must report the front's id and the front's
    # depth -- the back must not leak through anywhere.
    #
    # Existing depth-overlap tests in test_cudaraster_api.py only check 1-2
    # individual pixels. This test asserts the contract over the entire front
    # footprint, so a tile-localised drop (front missing at some tiles, back
    # showing through) is caught.
    width = 128
    height = 128
    vertices = _to_vertices(
        [
            # Back triangle (large), z = 0.7. Triangle id 1 in indices_both.
            (-0.8, -0.8, 0.7, 1.0),
            (0.8, -0.8, 0.7, 1.0),
            (0.0, 0.8, 0.7, 1.0),
            # Front triangle (smaller, fully inside back's footprint), z = 0.2.
            # Triangle id 2 in indices_both.
            (-0.4, -0.4, 0.2, 1.0),
            (0.4, -0.4, 0.2, 1.0),
            (0.0, 0.4, 0.2, 1.0),
        ]
    )
    indices_front_only = _to_indices([(3, 4, 5)])
    indices_both = _to_indices([(0, 1, 2), (3, 4, 5)])

    harness.configure(width, height)

    harness.upload(vertices, indices_front_only)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    front_only_color = harness.read().color
    front_mask = front_only_color != 0
    assert front_mask.any(), "front triangle alone must produce coverage"

    harness.upload(vertices, indices_both)
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    combined = harness.read()

    front_id = 2
    leakage_mask = front_mask & (combined.color != front_id)
    leak_count = int(leakage_mask.sum())
    assert leak_count == 0, (
        f"{leak_count} pixels inside the front triangle's footprint show a different id "
        f"than {front_id} (the front triangle); the back is leaking through"
    )

    # The back's id must still appear somewhere outside the front (otherwise
    # the test isn't actually exercising the overlap path).
    back_visible = (~front_mask) & (combined.color == 1)
    assert back_visible.any(), (
        "back triangle is supposed to be visible outside the front's footprint"
    )


@pytest.mark.gpu
def test_dense_overlap_in_single_tile_frontmost_triangle_wins(
    harness: CudaRasterHarness,
) -> None:
    # Pack 32 small overlapping triangles into a single 8x8 fine-raster tile,
    # each successively closer (smaller z) than the previous. The frontmost
    # (last-index) triangle wins by depth at every covered pixel inside the
    # cluster.
    #
    # Stress target: FineRaster's per-tile fragment processing, including
    # scan32_value (the per-warp Hillis-Steele scan over fragment counts at
    # FineRaster.inl L308-L317). If that scan miscomputes any lane's prefix,
    # a fragment ends up in the wrong queue slot and the front-most triangle
    # can be dropped at the per-pixel ROP. Symptom: an earlier triangle's id
    # surfaces at the tile's centre.
    width = 128
    height = 128
    triangle_count = 32  # exactly one warp's worth of FineRaster lanes
    target_tx, target_ty = 8, 8

    cx_ndc = (2.0 * (target_tx * 8 + 4.5)) / width - 1.0
    cy_ndc = (2.0 * (target_ty * 8 + 4.5)) / height - 1.0
    h = 0.025  # ~3-pixel half-width, well inside an 8x8 tile

    vertices: list[tuple[float, float, float, float]] = []
    indices: list[tuple[int, int, int]] = []
    z_far = 0.7
    z_near = 0.2
    for i in range(triangle_count):
        z = z_far - (z_far - z_near) * (i / max(1, triangle_count - 1))
        # Tiny per-triangle x offset so they aren't bit-identical (avoids any
        # potential dedup path) while still all hitting the same tile centre.
        offset = (i / triangle_count) * 0.0008
        base = len(vertices)
        vertices.extend(
            [
                (cx_ndc - h + offset, cy_ndc - h, z, 1.0),
                (cx_ndc + h + offset, cy_ndc - h, z, 1.0),
                (cx_ndc + offset, cy_ndc + h, z, 1.0),
            ]
        )
        indices.append((base, base + 1, base + 2))

    harness.configure(width, height)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    expected_id = triangle_count  # 1-indexed; last triangle is the frontmost
    center_x = target_tx * 8 + 4
    center_y = target_ty * 8 + 4
    actual = _pixel(color, center_x, center_y)
    assert actual == expected_id, (
        f"tile ({target_tx},{target_ty}) centre pixel ({center_x},{center_y}) "
        f"expected id {expected_id} (frontmost of {triangle_count} overlapping triangles), got {actual}"
    )


@pytest.mark.gpu
def test_per_tile_emit_imbalance_does_not_misroute_triangles(
    harness: CudaRasterHarness,
) -> None:
    # CoarseRaster's tile-emit prefix sum (CoarseRaster.inl L509-L515) is
    # warp-wide and contains explicit per-lane divergent control flow
    # (`if (threadIdx.x >= K)`) between scan steps. This is the most
    # divergence-prone of the seven Hillis-Steele scans the port inherited
    # from upstream, and the most likely to misbehave under Volta+ ITS
    # without explicit __syncwarp(). A bug in the scan corrupts the
    # emit-to-tile mapping, sending triangles into the wrong tile's segment.
    #
    # Build geometry that drives the divergent path: one heavy tile (~30
    # overlapping triangles) and many light tiles (1 distinct triangle each)
    # spanning the full first scan window of one coarse-raster warp.
    # If the prefix sum misroutes any triangle, a light tile's centre pixel
    # will show the wrong id (or no id).
    width = 128
    height = 128

    def _tile_center_ndc(tx: int, ty: int) -> tuple[float, float]:
        cx = tx * 8 + 4
        cy = ty * 8 + 4
        x_ndc = (2.0 * (cx + 0.5)) / width - 1.0
        y_ndc = (2.0 * (cy + 0.5)) / height - 1.0
        return x_ndc, y_ndc

    def _small_triangle_in_tile(
        tx: int, ty: int, z: float, jitter: float = 0.0
    ) -> list[tuple[float, float, float, float]]:
        cx_ndc, cy_ndc = _tile_center_ndc(tx, ty)
        h = 0.018  # ~2.3 pixel half-width; comfortably inside an 8x8 tile
        return [
            (cx_ndc - h + jitter, cy_ndc - h, z, 1.0),
            (cx_ndc + h + jitter, cy_ndc - h, z, 1.0),
            (cx_ndc + jitter, cy_ndc + h, z, 1.0),
        ]

    heavy_tx, heavy_ty = 4, 0
    heavy_count = 30

    # Pick 31 distinct light tiles inside the first scan window of one
    # coarse-raster warp (rows 0-1 of the bin, all 16 columns) excluding the
    # heavy tile. With CR_BIN_SIZE=16 and CR_COARSE_WARPS=4, warp 0's first
    # iteration of the outer loop covers exactly tileInBin in [0..32), which
    # is rows 0 and 1 of the bin.
    light_tiles: list[tuple[int, int]] = []
    for ty in range(2):
        for tx in range(16):
            if (tx, ty) == (heavy_tx, heavy_ty):
                continue
            light_tiles.append((tx, ty))
    light_tiles = light_tiles[:31]
    assert len(light_tiles) == 31

    vertices: list[tuple[float, float, float, float]] = []
    indices: list[tuple[int, int, int]] = []

    for i in range(heavy_count):
        z = 0.7 - 0.5 * (i / max(1, heavy_count - 1))
        jitter = (i / heavy_count) * 0.0006
        base = len(vertices)
        vertices.extend(_small_triangle_in_tile(heavy_tx, heavy_ty, z, jitter))
        indices.append((base, base + 1, base + 2))
    heavy_id_last = len(indices)  # 1-indexed; frontmost heavy triangle

    light_id_first = len(indices) + 1  # 1-indexed
    for tx, ty in light_tiles:
        base = len(vertices)
        vertices.extend(_small_triangle_in_tile(tx, ty, 0.5))
        indices.append((base, base + 1, base + 2))

    harness.configure(width, height)
    harness.upload(_to_vertices(vertices), _to_indices(indices))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    color = harness.read().color

    failures: list[str] = []
    for i, (tx, ty) in enumerate(light_tiles):
        expected_id = light_id_first + i
        center_x = tx * 8 + 4
        center_y = ty * 8 + 4
        actual = _pixel(color, center_x, center_y)
        if actual != expected_id:
            failures.append(
                f"light tile ({tx},{ty}) expected {expected_id} got {actual}"
            )

    heavy_center_x = heavy_tx * 8 + 4
    heavy_center_y = heavy_ty * 8 + 4
    actual_heavy = _pixel(color, heavy_center_x, heavy_center_y)
    if actual_heavy != heavy_id_last:
        failures.append(
            f"heavy tile ({heavy_tx},{heavy_ty}) expected {heavy_id_last} (frontmost), got {actual_heavy}"
        )

    assert not failures, (
        f"{len(failures)} tiles got the wrong triangle id; coarse-raster prefix sum misrouted "
        f"triangles. First few: " + "; ".join(failures[:5])
    )


@pytest.mark.gpu
def test_many_overlapping_triangles_do_not_overflow_tile_segments(
    harness: CudaRasterHarness,
) -> None:
    triangle_count = 4096
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
    assert np.count_nonzero(color) > 0, (
        "overlapping triangles produced no visible coverage"
    )


@pytest.mark.gpu
def test_clipped_backface_swap_preserves_depth_plane(
    harness: CudaRasterHarness,
) -> None:
    # Pins the barycentric tuple swap in TriangleSetup.inl's clipped path.
    # When backface culling is disabled and a clipped triangle comes out
    # backfacing in screen space, p1<->p2, v1<->v2, vidx.y<->vidx.z, rcpW.y<->z,
    # and the polygon-space (s,t) tuple bb1<->bb2 are all swapped together
    # before setupTriangle. The plane equations setupTriangle writes (including
    # the depth plane) must be identical to those produced by an already-CCW
    # input that goes through the same clipped path.
    #
    # Geometry: a triangle whose v0 lies outside the +X frustum, so the clipper
    # always splits it into multiple subtriangles. The CW vs CCW versions
    # differ only in vertex order; the visible pixels (and their depth values)
    # must match exactly.
    vertices = _to_vertices(
        [
            (1.6, -0.3, 0.4, 1.0),
            (-0.2, -0.5, 0.2, 1.0),
            (-0.2, 0.5, 0.6, 1.0),
        ]
    )

    harness.configure(128, 128)

    harness.upload(vertices, _to_indices([(0, 1, 2)]))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    cw = harness.read()

    harness.upload(vertices, _to_indices([(0, 2, 1)]))
    assert harness.draw(clear_color=0, flags=0, deterministic_tiebreaker=False)
    ccw = harness.read()

    coverage_cw = cw.color != 0
    coverage_ccw = ccw.color != 0
    assert coverage_cw.any(), "clipped CW triangle produced no coverage"
    assert np.array_equal(coverage_cw, coverage_ccw), (
        "CW and CCW clipped triangles disagree on coverage; the position swap is broken"
    )

    covered = coverage_cw
    assert np.array_equal(cw.depth[covered], ccw.depth[covered]), (
        "CW and CCW clipped triangles disagree on depth at covered pixels; "
        "the barycentric tuple swap in the clipped path is wrong"
    )
