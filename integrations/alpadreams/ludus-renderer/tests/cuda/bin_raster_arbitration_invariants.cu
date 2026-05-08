// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Two arbitration patterns from BinRaster.inl, isolated for unit testing.
//
// Both patterns existed verbatim in the upstream HPG-2011 CudaRaster and
// relied on warp-synchronous "highest active lane stores last" semantics
// when several lanes write the same shared memory word. On Volta+ ITS the
// store ordering is undefined, so the upstream loops needed a single-lane
// gate to be deterministic.
//
//   Fix A -- s_broadcast warp-total race (BinRaster.inl ~line 119):
//     32 converged lanes do an exclusive prefix sum of `num` across the
//     warp. Lane k's `myIdx + num` equals the sum of nums[0..k]. The warp
//     total is therefore lane 31's `myIdx + num`, and is broadcast to a
//     shared slot for the next stage. Upstream had every lane write the
//     slot with its own value; the fix gates the write to lane 31.
//
//   Fix B -- s_bufCount block-total race (BinRaster.inl ~line 148):
//     The first CR_BIN_WARPS lanes of one warp inclusive-scan the per-warp
//     totals via repeated reads of pre-zeroed shared-memory pad slots.
//     After the scan, lane (CR_BIN_WARPS - 1) holds the block total.
//     Upstream had every participating lane write `s_bufCount`; the fix
//     gates the write to that final lane.
//
// These helpers replay each pattern in isolation with caller-supplied per-
// lane inputs and report the resulting broadcast / total values back to the
// host. The Python suite uses them to pin the lane-gate placement.
//
// KNOWN LIMITATION: SPECIFICATION TEST, NOT REGRESSION TEST.
// The kernels below contain hand-written copies of the patterns from
// `cuda/BinRaster.inl`. Because the test code is a separate copy, a
// regression in the production code (e.g. someone changes the gate from
// `threadIdx.x == 31` to `threadIdx.x == 0`, or removes a `__ballot_sync`
// step) does NOT cause this test to fail. The test pins the *spec* of what
// each pattern should compute, but the actual production functions are
// unverified by this file. End-to-end coverage for BinRaster currently
// comes from `test_large_distinct_triangle_field_exposes_many_unique_ids`
// in `test_cudaraster_api.py`.
//
// TODO: Replace the copies with calls into shared `__device__ __inline__`
// helpers. Concretely: extract the prefix-sum-and-broadcast pattern (Fix A)
// and the inclusive-scan-and-block-total pattern (Fix B) from
// `BinRaster.inl` into a new header (e.g. `cuda/BinRasterScans.inl`) that
// both `BinRaster.inl` and this test source `#include`. Replace the inline
// snippets in `BinRaster.inl` with calls to those helpers. After that
// change, breaking the production function breaks this test, and the spec
// and regression coverage merge.

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

constexpr int kFixBLanes = 16; // mirrors CR_BIN_WARPS in BinRaster

__device__ __forceinline__ unsigned laneMaskLt()
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

// Replays the upstream Fix A pattern. One warp; per-lane `num` input from
// `numIn`; returns lane 31's `myIdx + num` via `broadcastOut[0]`, which is
// the value the gated write deposits in BinRaster's `s_broadcast` slot.
__global__ void fixABroadcastKernel(const unsigned* numIn, unsigned* broadcastOut)
{
    __shared__ unsigned sBroadcast;
    if (threadIdx.x == 0)
        sBroadcast = 0xDEADBEEFu;
    __syncwarp();

    unsigned num = numIn[threadIdx.x];

    // Mirrors BinRaster.inl exactly: ballot-bit scan for bits 0..2 of `num`.
    unsigned myIdx = __popc(__ballot_sync(0xFFFFFFFFu, num & 1u) & laneMaskLt());
    if (__any_sync(0xFFFFFFFFu, num > 1))
    {
        myIdx += __popc(__ballot_sync(0xFFFFFFFFu, num & 2u) & laneMaskLt()) * 2u;
        myIdx += __popc(__ballot_sync(0xFFFFFFFFu, num & 4u) & laneMaskLt()) * 4u;
    }

    // Single-lane gate (the fix). If this gate moved or were removed, the
    // broadcast slot would receive an undefined lane's prefix value under ITS.
    if (threadIdx.x == 31)
        sBroadcast = myIdx + num;
    __syncwarp();

    if (threadIdx.x == 0)
        broadcastOut[0] = sBroadcast;
}

// Replays the upstream Fix B pattern. The first CR_BIN_WARPS lanes inclusive-
// scan a vector of per-warp totals via the upstream shared-memory pad layout.
// Reports the inclusive-scan vector via `prefixOut[0..CR_BIN_WARPS)` and the
// final block total (i.e. the value `s_bufCount` should hold after the gated
// write) via `bufCountOut[0]`.
__global__ void fixBBlockTotalKernel(const unsigned* totalsIn,
                                     unsigned* prefixOut,
                                     unsigned* bufCountOut)
{
    // Layout matches BinRaster's s_broadcast: the first CR_BIN_WARPS slots
    // are zero pads so `ptr[-k]` reads return 0 for the first iterations.
    __shared__ unsigned sBroadcast[2 * kFixBLanes];
    __shared__ unsigned sBufCount;

    if (threadIdx.x < 2 * kFixBLanes)
        sBroadcast[threadIdx.x] = 0u;
    if (threadIdx.x == 0)
        sBufCount = 0xDEADBEEFu;
    __syncwarp();

    if (threadIdx.x < kFixBLanes)
        sBroadcast[threadIdx.x + kFixBLanes] = totalsIn[threadIdx.x];
    __syncwarp();

    if (threadIdx.x < kFixBLanes)
    {
        // Inclusive scan via the same series of shared-memory step adds the
        // upstream loop performs. The bufCount value here represents the
        // pre-existing accumulator; using zero keeps the test self-contained.
        unsigned bufCount = 0u;
        volatile unsigned* ptr = &sBroadcast[threadIdx.x + kFixBLanes];
        unsigned val = *ptr;
        val += ptr[-1]; *ptr = val;
        val += ptr[-2]; *ptr = val;
        val += ptr[-4]; *ptr = val;
        val += ptr[-8]; *ptr = val;

        prefixOut[threadIdx.x] = val;

        // Single-lane gate (the fix). The gated write deposits the block
        // total into `s_bufCount`; if it moved to a different lane, the
        // resulting value would be that lane's partial prefix instead.
        if (threadIdx.x == kFixBLanes - 1)
            sBufCount = bufCount + val;
    }
    __syncwarp();

    if (threadIdx.x == 0)
        bufCountOut[0] = sBufCount;
}

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        char msg[512];
        std::snprintf(msg, sizeof(msg), "%s: %s", op, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

unsigned run_warp_total_broadcast(const std::vector<unsigned>& nums)
{
    if (nums.size() != 32)
        throw std::runtime_error("nums must contain exactly 32 entries (one per warp lane)");

    unsigned* deviceIn = nullptr;
    unsigned* deviceOut = nullptr;
    unsigned hostOut = 0u;

    checkCuda(cudaMalloc(&deviceIn, sizeof(unsigned) * 32), "cudaMalloc(in)");
    checkCuda(cudaMalloc(&deviceOut, sizeof(unsigned)), "cudaMalloc(out)");
    checkCuda(cudaMemcpy(deviceIn, nums.data(), sizeof(unsigned) * 32, cudaMemcpyHostToDevice),
              "cudaMemcpy(in)");

    fixABroadcastKernel<<<1, 32>>>(deviceIn, deviceOut);
    checkCuda(cudaGetLastError(), "fixABroadcastKernel launch");
    checkCuda(cudaDeviceSynchronize(), "fixABroadcastKernel sync");
    checkCuda(cudaMemcpy(&hostOut, deviceOut, sizeof(unsigned), cudaMemcpyDeviceToHost),
              "cudaMemcpy(out)");

    cudaFree(deviceIn);
    cudaFree(deviceOut);
    return hostOut;
}

py::dict run_block_total_inclusive_scan(const std::vector<unsigned>& totals)
{
    if (totals.size() != static_cast<size_t>(kFixBLanes))
        throw std::runtime_error("totals must contain exactly CR_BIN_WARPS (=16) entries");

    unsigned* deviceIn = nullptr;
    unsigned* devicePrefix = nullptr;
    unsigned* deviceBuf = nullptr;
    std::vector<unsigned> prefix(kFixBLanes, 0u);
    unsigned hostBuf = 0u;

    checkCuda(cudaMalloc(&deviceIn, sizeof(unsigned) * kFixBLanes), "cudaMalloc(in)");
    checkCuda(cudaMalloc(&devicePrefix, sizeof(unsigned) * kFixBLanes), "cudaMalloc(prefix)");
    checkCuda(cudaMalloc(&deviceBuf, sizeof(unsigned)), "cudaMalloc(buf)");
    checkCuda(cudaMemcpy(deviceIn, totals.data(), sizeof(unsigned) * kFixBLanes,
                         cudaMemcpyHostToDevice), "cudaMemcpy(in)");

    fixBBlockTotalKernel<<<1, 32>>>(deviceIn, devicePrefix, deviceBuf);
    checkCuda(cudaGetLastError(), "fixBBlockTotalKernel launch");
    checkCuda(cudaDeviceSynchronize(), "fixBBlockTotalKernel sync");
    checkCuda(cudaMemcpy(prefix.data(), devicePrefix, sizeof(unsigned) * kFixBLanes,
                         cudaMemcpyDeviceToHost), "cudaMemcpy(prefix)");
    checkCuda(cudaMemcpy(&hostBuf, deviceBuf, sizeof(unsigned), cudaMemcpyDeviceToHost),
              "cudaMemcpy(buf)");

    cudaFree(deviceIn);
    cudaFree(devicePrefix);
    cudaFree(deviceBuf);

    py::dict out;
    out["prefix"] = prefix;
    out["buf_count"] = hostBuf;
    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run_warp_total_broadcast", &run_warp_total_broadcast,
          "Replay BinRaster's per-warp prefix-sum-and-broadcast pattern (Fix A).");
    m.def("run_block_total_inclusive_scan", &run_block_total_inclusive_scan,
          "Replay BinRaster's per-block CR_BIN_WARPS-lane inclusive scan (Fix B).");
}
