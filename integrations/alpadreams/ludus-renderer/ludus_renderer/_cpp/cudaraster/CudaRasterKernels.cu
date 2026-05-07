// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cuda/PixelPipe.hpp"
#include "cuda/PrivateDefs.hpp"
#include <stdint.h>

// Definitions of the pipeline state variables declared (extern) in
// cuda/PixelPipe.inl. Kept at global scope so nvcc's static host-side
// stub registers them via __cudaRegisterVar with external linkage.

__constant__ FW::CRParams    c_crParams;
__device__   FW::CRAtomics   g_crAtomics;
__constant__ FW::S32         c_profLaunchIdx;
__constant__ CUdeviceptr     c_profData;

#include "cuda/PixelPipe.inl"

using FW::PixelPipeSpec;

CR_DEFINE_PIXEL_PIPE(
    crDefaultPipe,
    FW::GouraudVertex,
    FW::GouraudShader,
    FW::BlendReplace,
    0,
    FW::RenderModeFlag_EnableDepth | FW::RenderModeFlag_EnableLerp)

namespace
{

__global__ void crClearBuffersKernel(uint32_t* color, uint32_t* depth, size_t count, uint32_t clearColor, uint32_t clearDepth)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    color[idx] = clearColor;
    depth[idx] = clearDepth;
}

}

void crClearBuffers(uint32_t* color, uint32_t* depth, size_t count, uint32_t clearColor, uint32_t clearDepth, cudaStream_t stream)
{
    if (!count)
        return;

    const int blockSize = 256;
    int gridSize = (int)((count + blockSize - 1) / blockSize);
    crClearBuffersKernel<<<gridSize, blockSize, 0, stream>>>(color, depth, count, clearColor, clearDepth);
}
