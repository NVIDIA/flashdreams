// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda/RopLaneMask.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

constexpr int kNumValues = 2 * 32;

__global__ void ropLaneMaskInvariantKernel(unsigned* out)
{
    const unsigned lane = threadIdx.x;
    const bool reverseLanes = (blockIdx.x == 0);

    if (lane >= 32)
        return;

    const unsigned replacementMask = reverseLanes
        ? FW::determineROPLaneMask<FW::BlendReplace, FW::RenderModeFlag_EnableDepth>()
        : FW::determineROPLaneMask<FW::BlendReplace, 0>();
    const unsigned caseOffset = blockIdx.x * 32;
    out[caseOffset + lane] = replacementMask;
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
