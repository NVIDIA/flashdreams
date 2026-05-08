"""Focused invariants for the CudaRaster cleanroom port.

These tests pin small, risky porting assumptions that are easier to verify
directly than through the broad API contract suite.

Known limitation: most of the warp-arbitration tests in this file currently
exercise hand-written copies of the production patterns rather than the
production code itself. They are specification tests -- they pin what the
pattern should compute and pin the lane-gate choice -- but they do NOT
defend against regressions in the production source. See the TODO blocks at
the top of `tests/cuda/rop_lane_mask_invariant.cu` and
`tests/cuda/bin_raster_arbitration_invariants.cu` for the planned fix
(extracting the patterns into shared __device__ inline helpers that both
the production code and the tests include). End-to-end regression coverage
for the production paths currently falls on the API-level tests in
`test_cudaraster_api.py`.

The `test_clipped_*` tests below DO exercise production code -- they go
through the real rasterizer pipeline -- and are not affected by this
limitation.
"""

from pathlib import Path

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


@pytest.fixture(scope="module")
def cudaraster_plugin() -> object:
    _require_cuda()
    return _get_plugin(gl=False)


@pytest.fixture
def harness(cudaraster_plugin: object) -> CudaRasterHarness:
    return CudaRasterHarness(cudaraster_plugin)


@pytest.fixture(scope="module")
def rop_lane_mask_helper() -> object:
    _require_cuda()
    helper_src = Path(__file__).with_name("cuda") / "rop_lane_mask_invariant.cu"
    return torch.utils.cpp_extension.load(
        name="cudaraster_rop_lane_mask_invariant",
        sources=[str(helper_src)],
        extra_cuda_cflags=["-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


@pytest.fixture(scope="module")
def bin_raster_arbitration_helper() -> object:
    _require_cuda()
    helper_src = Path(__file__).with_name("cuda") / "bin_raster_arbitration_invariants.cu"
    return torch.utils.cpp_extension.load(
        name="cudaraster_bin_raster_arbitration_invariants",
        sources=[str(helper_src)],
        extra_cuda_cflags=["-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


@pytest.mark.gpu
def test_rop_lane_mask_replacement_matches_upstream_arbitration_order(rop_lane_mask_helper: object) -> None:
    values = list(rop_lane_mask_helper.run_rop_lane_mask_invariant())
    assert len(values) == 128

    cases = [
        ("reverse", 0, lambda lane: (1 << lane) - 1),
        ("forward", 64, lambda lane: (0xFFFFFFFF ^ ((1 << (lane + 1)) - 1)) & 0xFFFFFFFF),
    ]
    for label, offset, expected_for_lane in cases:
        ordered = [int(v) for v in values[offset : offset + 32]]
        replacement = [int(v) for v in values[offset + 32 : offset + 64]]
        expected = [expected_for_lane(lane) for lane in range(32)]

        assert ordered == expected, label
        assert replacement == expected, label
        # The fine raster only requires that __popc(mask) is a permutation of
        # [0, 31] across the warp -- it is used as a unique per-lane index into
        # a 32-slot scratch buffer.
        assert sorted(mask.bit_count() for mask in replacement) == list(range(32)), label


@pytest.mark.gpu
def test_clipped_cw_triangle_renders_with_backface_culling_disabled(harness: CudaRasterHarness) -> None:
    vertices = _to_vertices(
        [
            (1.6, -0.3, 0.2, 1.0),  # v0 is outside +X, forcing the clipped-subtriangle path.
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
    bin_raster_arbitration_helper: object, label: str, nums: list[int]
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
    bin_raster_arbitration_helper: object, label: str, totals: list[int]
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

    assert prefix == expected_prefix, f"{label}: prefix scan diverged from inclusive sum"
    assert actual_buf == expected_prefix[-1], f"{label}: s_bufCount must equal block total"


@pytest.mark.gpu
def test_clipped_backface_swap_preserves_depth_plane(harness: CudaRasterHarness) -> None:
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
