// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Exercises BinRaster arbitration helpers in isolation with caller-provided
// lane inputs. The helpers are shared with production code via
// cuda/BinRasterScans.cuh, so these checks are direct regression tests.

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda/BinRasterScans.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

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

    FW::binRasterPerLanePrefix3Bit(num, &sBroadcast);
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
    __shared__ unsigned sBroadcast[2 * CR_BIN_WARPS];
    __shared__ unsigned sBufCount;

    if (threadIdx.x < 2 * CR_BIN_WARPS)
        sBroadcast[threadIdx.x] = 0u;
    if (threadIdx.x == 0)
        sBufCount = 0xDEADBEEFu;
    __syncwarp();

    if (threadIdx.x < CR_BIN_WARPS)
        sBroadcast[threadIdx.x + CR_BIN_WARPS] = totalsIn[threadIdx.x];
    __syncwarp();

    if (threadIdx.x < CR_BIN_WARPS)
    {
        volatile unsigned* ptr = &sBroadcast[threadIdx.x + CR_BIN_WARPS];
        unsigned val = FW::binRasterPerWarpInclusiveScan(ptr, threadIdx.x, 0u, &sBufCount);

        prefixOut[threadIdx.x] = val;
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
    if (totals.size() != static_cast<size_t>(CR_BIN_WARPS))
        throw std::runtime_error("totals must contain exactly CR_BIN_WARPS (=16) entries");

    unsigned* deviceIn = nullptr;
    unsigned* devicePrefix = nullptr;
    unsigned* deviceBuf = nullptr;
    std::vector<unsigned> prefix(CR_BIN_WARPS, 0u);
    unsigned hostBuf = 0u;

    checkCuda(cudaMalloc(&deviceIn, sizeof(unsigned) * CR_BIN_WARPS), "cudaMalloc(in)");
    checkCuda(cudaMalloc(&devicePrefix, sizeof(unsigned) * CR_BIN_WARPS), "cudaMalloc(prefix)");
    checkCuda(cudaMalloc(&deviceBuf, sizeof(unsigned)), "cudaMalloc(buf)");
    checkCuda(cudaMemcpy(deviceIn, totals.data(), sizeof(unsigned) * CR_BIN_WARPS,
                         cudaMemcpyHostToDevice), "cudaMemcpy(in)");

    fixBBlockTotalKernel<<<1, 32>>>(deviceIn, devicePrefix, deviceBuf);
    checkCuda(cudaGetLastError(), "fixBBlockTotalKernel launch");
    checkCuda(cudaDeviceSynchronize(), "fixBBlockTotalKernel sync");
    checkCuda(cudaMemcpy(prefix.data(), devicePrefix, sizeof(unsigned) * CR_BIN_WARPS,
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
