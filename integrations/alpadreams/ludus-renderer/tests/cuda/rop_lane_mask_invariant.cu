// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// KNOWN LIMITATION: SPECIFICATION TEST, NOT REGRESSION TEST.
// The kernels below contain hand-written copies of the lanemask_lt/gt usage
// and an emulation of the original upstream busy-wait that this port
// replaced. The production replacement lives in
// `ludus_renderer/_cpp/cudaraster/cuda/FineRaster.inl::determineROPLaneMask`.
// Because the test code is a separate copy, a regression in the production
// function (e.g. the lt/gt branches getting swapped) does NOT cause this
// test to fail. The test pins the *spec* (what the function should compute)
// and the algebraic invariant that __popc(mask) is a permutation of [0,31].
//
// TODO: Replace the copy with a shared `__device__ __inline__` helper.
// Concretely: extract `determineROPLaneMask` from FineRaster.inl into a new
// header (e.g. `cuda/RopLaneMask.inl`) that both FineRaster.inl and this
// test source `#include`. The kernel here should then call the production
// function directly. After that change, breaking the production function
// breaks this test, and the spec/regression coverage merge.

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

constexpr int kNumValues = 2 * 64;

__device__ __forceinline__ unsigned laneMaskLt()
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned laneMaskGt()
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__global__ void ropLaneMaskInvariantKernel(unsigned* out)
{
    __shared__ unsigned warpTemp;

    const unsigned lane = threadIdx.x;
    const bool reverseLanes = (blockIdx.x == 0);

    if (lane >= 32)
        return;

    unsigned orderedMask = reverseLanes ? (1u << lane) : ~0u;

    // Ordered emulation of the upstream single-shared-word arbitration loop.
    // The original loop depended on warp-synchronous write ordering. This
    // sequence models the intended ascending-lane arbitration explicitly.
    #pragma unroll
    for (unsigned winner = 0; winner < 32; ++winner)
    {
        if (lane == winner)
            warpTemp = lane;
        __syncwarp();

        const unsigned observed = warpTemp;
        orderedMask ^= 1u << observed;
        __syncwarp();

        if (observed == lane)
            break;
    }

    const unsigned replacementMask = reverseLanes ? laneMaskLt() : laneMaskGt();
    const unsigned caseOffset = blockIdx.x * 64;
    out[caseOffset + lane] = orderedMask;
    out[caseOffset + 32 + lane] = replacementMask;
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

std::vector<unsigned> run_rop_lane_mask_invariant()
{
    unsigned* deviceOut = nullptr;
    std::vector<unsigned> hostOut(kNumValues, 0u);

    checkCuda(cudaMalloc(&deviceOut, sizeof(unsigned) * kNumValues), "cudaMalloc");
    ropLaneMaskInvariantKernel<<<2, 32>>>(deviceOut);
    checkCuda(cudaGetLastError(), "ropLaneMaskInvariantKernel launch");
    checkCuda(cudaDeviceSynchronize(), "ropLaneMaskInvariantKernel synchronize");
    checkCuda(cudaMemcpy(hostOut.data(), deviceOut, sizeof(unsigned) * kNumValues, cudaMemcpyDeviceToHost), "cudaMemcpy");
    checkCuda(cudaFree(deviceOut), "cudaFree");

    return hostOut;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run_rop_lane_mask_invariant", &run_rop_lane_mask_invariant);
}
