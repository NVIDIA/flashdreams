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
#include <cstdio>
#include <cstdlib>

// Definitions of the pipeline state variables declared (extern) in
// cuda/PixelPipe.inl. Kept at global scope so nvcc's static host-side
// stub registers them via __cudaRegisterVar with external linkage.

__constant__ FW::CRParams    c_crParams;
__device__   FW::CRAtomics   g_crAtomics;
__constant__ FW::S32         c_profLaunchIdx;
__constant__ CUdeviceptr     c_profData;

#include "cuda/PixelPipe.inl"

using FW::PixelPipeSpec;

namespace FW
{

// Simple shader that outputs triangle index + 1 (0 = no triangle, 1+ = triIdx)
class TriIdShader : public FragmentShaderBase
{
public:
    __device__ __inline__ void run(void) { m_color = (U32)(m_triIdx + 1); }
};

}

// Pixel pipe using only ShadedVertexBase (4 floats) and outputting triangle IDs
CR_DEFINE_PIXEL_PIPE(
    crDefaultPipe,
    FW::ShadedVertexBase,
    FW::TriIdShader,
    FW::BlendReplace,
    0,
    FW::RenderModeFlag_EnableDepth)

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

__global__ void crClearSurfacesKernel(cudaSurfaceObject_t colorSurf, cudaSurfaceObject_t depthSurf,
                                       int width, int height, uint32_t clearColor, uint32_t clearDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    surf2Dwrite<uint32_t>(clearColor, colorSurf, x * sizeof(uint32_t), y);
    surf2Dwrite<uint32_t>(clearDepth, depthSurf, x * sizeof(uint32_t), y);
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

void crClearSurfaces(cudaSurfaceObject_t colorSurf, cudaSurfaceObject_t depthSurf,
                     int width, int height, uint32_t clearColor, uint32_t clearDepth, cudaStream_t stream)
{
    if (width <= 0 || height <= 0)
        return;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    crClearSurfacesKernel<<<grid, block, 0, stream>>>(colorSurf, depthSurf, width, height, clearColor, clearDepth);
}

void crCopyFromArray(uint32_t* dst, cudaArray_t src, int width, int height, cudaStream_t stream)
{
    if (width <= 0 || height <= 0 || !dst || !src)
        return;

    cudaMemcpy2DFromArrayAsync(dst, width * sizeof(uint32_t), src,
                                0, 0, width * sizeof(uint32_t), height,
                                cudaMemcpyDeviceToDevice, stream);
}

//------------------------------------------------------------------------
// Host-callable wrappers for pipeline stages.
//------------------------------------------------------------------------

// Debug helper - set CR_DEBUG_SYNC=1 env var to enable sync after each stage
static bool crDebugSync()
{
    static int val = -1;
    if (val < 0) {
        const char* env = getenv("CR_DEBUG_SYNC");
        val = (env && env[0] == '1') ? 1 : 0;
    }
    return val != 0;
}

static void crCheckError(const char* stage)
{
    if (!crDebugSync()) return;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", stage, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "Stage %s completed OK\n", stage);
    }
}

void crUploadParams(const FW::CRParams& params, cudaStream_t stream)
{
    if (crDebugSync()) {
        fprintf(stderr, "CRParams: numTris=%d, viewport=%dx%d, bins=%dx%d (%d), tiles=%dx%d (%d)\n",
                params.numTris, params.viewportWidth, params.viewportHeight,
                params.widthBins, params.heightBins, params.numBins,
                params.widthTiles, params.heightTiles, params.numTiles);
        fprintf(stderr, "CRParams: binBatchSize=%d, maxSubtris=%d, maxBinSegs=%d, maxTileSegs=%d\n",
                params.binBatchSize, params.maxSubtris, params.maxBinSegs, params.maxTileSegs);
        fprintf(stderr, "CRParams: triSubtris=%p, triHeader=%p, triData=%p\n",
                (void*)params.triSubtris, (void*)params.triHeader, (void*)params.triData);
        fprintf(stderr, "CRParams: t_vertexBuffer=%llu, t_triHeader=%llu, t_triData=%llu\n",
                (unsigned long long)params.t_vertexBuffer, (unsigned long long)params.t_triHeader, (unsigned long long)params.t_triData);
    }
    cudaMemcpyToSymbolAsync(c_crParams, &params, sizeof(FW::CRParams), 0, cudaMemcpyHostToDevice, stream);
}

void crInitAtomics(int numTris, cudaStream_t stream)
{
    FW::CRAtomics atomics;
    atomics.numSubtris = numTris;
    atomics.binCounter = 0;
    atomics.numBinSegs = 0;
    atomics.coarseCounter = 0;
    atomics.numTileSegs = 0;
    atomics.numActiveTiles = 0;
    atomics.fineCounter = 0;
    cudaMemcpyToSymbolAsync(g_crAtomics, &atomics, sizeof(FW::CRAtomics), 0, cudaMemcpyHostToDevice, stream);
}

void crReadAtomics(FW::CRAtomics* atomics, cudaStream_t stream)
{
    cudaMemcpyFromSymbolAsync(atomics, g_crAtomics, sizeof(FW::CRAtomics), 0, cudaMemcpyDeviceToHost, stream);
}

void crDebugReadTriSubtris(uint8_t* dst, CUdeviceptr src, int numTris, cudaStream_t stream)
{
    cudaMemcpyAsync(dst, (void*)src, numTris, cudaMemcpyDeviceToHost, stream);
}

void crLaunchSetup(int numTris, cudaStream_t stream)
{
    if (numTris <= 0)
        return;

    dim3 block(32, CR_SETUP_WARPS);
    int numBlocks = (numTris + block.x * block.y - 1) / (block.x * block.y);
    dim3 grid(numBlocks, 1);
    if (crDebugSync()) fprintf(stderr, "Launching setup: %d tris, grid=%d, block=(%d,%d)\n", numTris, numBlocks, block.x, block.y);
    crDefaultPipe_triangleSetup<<<grid, block, 0, stream>>>();
    crCheckError("setup");
}

void crLaunchBin(cudaStream_t stream)
{
    dim3 block(32, CR_BIN_WARPS);
    dim3 grid(CR_BIN_STREAMS_SIZE, 1);
    if (crDebugSync()) fprintf(stderr, "Launching bin: grid=%d, block=(%d,%d)\n", CR_BIN_STREAMS_SIZE, block.x, block.y);
    crDefaultPipe_binRaster<<<grid, block, 0, stream>>>();
    crCheckError("bin");
}

void crLaunchCoarse(int numSMs, cudaStream_t stream)
{
    dim3 block(32, CR_COARSE_WARPS);
    dim3 grid(numSMs, 1);
    if (crDebugSync()) fprintf(stderr, "Launching coarse: grid=%d, block=(%d,%d)\n", numSMs, block.x, block.y);
    crDefaultPipe_coarseRaster<<<grid, block, 0, stream>>>();
    crCheckError("coarse");
}

void crLaunchFine(int numSMs, int numFineWarps, cudaStream_t stream)
{
    dim3 block(32, numFineWarps);
    dim3 grid(numSMs, 1);
    if (crDebugSync()) fprintf(stderr, "Launching fine: grid=%d, block=(%d,%d)\n", numSMs, block.x, block.y);
    crDefaultPipe_fineRaster<<<grid, block, 0, stream>>>();
    crCheckError("fine");
}
