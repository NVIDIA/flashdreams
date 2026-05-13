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
#include <cassert>
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

// PORT_NOTES.md: depth peeling needs a separate pipe instantiation before this flag can work.
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
                                       int width, int height, int offsetX, int offsetY,
                                       uint32_t clearColor, uint32_t clearDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    surf2Dwrite<uint32_t>(clearColor, colorSurf, (x + offsetX) * sizeof(uint32_t), y + offsetY);
    surf2Dwrite<uint32_t>(clearDepth, depthSurf, (x + offsetX) * sizeof(uint32_t), y + offsetY);
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
                     int width, int height, int offsetX, int offsetY,
                     uint32_t clearColor, uint32_t clearDepth, cudaStream_t stream)
{
    if (width <= 0 || height <= 0)
        return;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    crClearSurfacesKernel<<<grid, block, 0, stream>>>(colorSurf, depthSurf, width, height, offsetX, offsetY, clearColor, clearDepth);
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

// Pipeline diagnostic gate.
//
// Set CR_DEBUG_SYNC=1 in the environment to:
//   * Insert a cudaDeviceSynchronize() after every stage launch (setup, bin,
//     coarse, fine), so kernel errors are reported against the correct stage
//     instead of surfacing later as opaque API failures.
//   * Read back g_crAtomics after each stage and print the per-stage counter
//     deltas (numSubtris / binCounter / numBinSegs / coarseCounter /
//     numTileSegs / numActiveTiles / fineCounter) to stderr. Useful for
//     localizing which stage is dropping work when a draw produces
//     unexpected output.
//   * Print param uploads and atomics resets so the host-side pipeline
//     wiring is visible.
//
// Off by default. The barrier and readback are not free; do not enable in
// production or perf measurements. The check is a single env-var read cached
// across the process lifetime, so the disabled path is effectively zero cost.
static bool crDebugSync()
{
    static int val = -1;
    if (val < 0) {
        const char* env = getenv("CR_DEBUG_SYNC");
        val = (env && env[0] == '1') ? 1 : 0;
    }
    return val != 0;
}

// Diagnostic stage barrier. No-op unless CR_DEBUG_SYNC=1 (see crDebugSync).
static void crCheckError(const char* stage)
{
    if (!crDebugSync()) return;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", stage, cudaGetErrorString(err));
        return;
    }
    void* devPtr = nullptr;
    FW::CRAtomics readback;
    memset(&readback, 0, sizeof(readback));
    cudaError_t symErr = cudaGetSymbolAddress(&devPtr, g_crAtomics);
    if (symErr == cudaSuccess && devPtr)
        cudaMemcpy(&readback, devPtr, sizeof(FW::CRAtomics), cudaMemcpyDeviceToHost);
    fprintf(stderr,
            "stage=%-7s numSubtris=%d binCounter=%d numBinSegs=%d coarseCounter=%d numTileSegs=%d numActiveTiles=%d fineCounter=%d\n",
            stage, readback.numSubtris, readback.binCounter, readback.numBinSegs,
            readback.coarseCounter, readback.numTileSegs, readback.numActiveTiles,
            readback.fineCounter);
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

    // Use cudaGetSymbolAddress for reliable symbol access in dynamically loaded .so
    void* devPtr = nullptr;
    cudaError_t err = cudaGetSymbolAddress(&devPtr, c_crParams);
    assert(err == cudaSuccess && devPtr && "cudaGetSymbolAddress(c_crParams) failed");
    cudaMemcpyAsync(devPtr, &params, sizeof(FW::CRParams), cudaMemcpyHostToDevice, stream);
}

__global__ void crResetAtomicsKernel(int numSubtris)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_crAtomics.numSubtris = numSubtris;
        g_crAtomics.binCounter = 0;
        g_crAtomics.numBinSegs = 0;
        g_crAtomics.coarseCounter = 0;
        g_crAtomics.numTileSegs = 0;
        g_crAtomics.numActiveTiles = 0;
        g_crAtomics.fineCounter = 0;
        __threadfence();
    }
}

void crInitAtomics(int numTris, cudaStream_t stream)
{
    if (crDebugSync())
        fprintf(stderr, "crInitAtomics: launching kernel to reset atomics, numSubtris=%d\n", numTris);

    // Use a kernel to reset atomics - this ensures we're writing to the same
    // g_crAtomics symbol that the other kernels use
    crResetAtomicsKernel<<<1, 1, 0, stream>>>(numTris);

    if (crDebugSync()) {
        cudaStreamSynchronize(stream);
        void* devPtr = nullptr;
        cudaError_t err = cudaGetSymbolAddress(&devPtr, g_crAtomics);
        assert(err == cudaSuccess && devPtr);
        FW::CRAtomics readback;
        cudaMemcpy(&readback, devPtr, sizeof(FW::CRAtomics), cudaMemcpyDeviceToHost);
        fprintf(stderr, "crInitAtomics verify (host): binCounter=%d\n", readback.binCounter);
    }
}

void crReadAtomics(FW::CRAtomics* atomics, cudaStream_t stream)
{
    void* devPtr = nullptr;
    cudaError_t err = cudaGetSymbolAddress(&devPtr, g_crAtomics);
    assert(err == cudaSuccess && devPtr && "cudaGetSymbolAddress(g_crAtomics) failed");
    cudaMemcpyAsync(atomics, devPtr, sizeof(FW::CRAtomics), cudaMemcpyDeviceToHost, stream);
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
    if (crDebugSync()) {
        cudaStreamSynchronize(stream);
        void* devPtr = nullptr;
        cudaGetSymbolAddress(&devPtr, g_crAtomics);
        FW::CRAtomics readback;
        if (devPtr)
            cudaMemcpy(&readback, devPtr, sizeof(FW::CRAtomics), cudaMemcpyDeviceToHost);
        fprintf(stderr, "BEFORE bin launch: binCounter=%d (devPtr=%p)\n", readback.binCounter, devPtr);
        fprintf(stderr, "Launching bin: grid=%d, block=(%d,%d)\n", CR_BIN_STREAMS_SIZE, block.x, block.y);
    }
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
