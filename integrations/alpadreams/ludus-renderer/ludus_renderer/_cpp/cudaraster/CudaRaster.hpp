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

#pragma once

//------------------------------------------------------------------------
// This is a slimmed-down and modernized version of the original
// CudaRaster codebase that accompanied the HPG 2011 paper
// "High-Performance Software Rasterization on GPUs" by Laine and Karras.
// Modifications have been made to accommodate post-Volta execution model
// with warp divergence. Support for shading, blending, quad rendering,
// and supersampling have been removed as unnecessary for nvdiffrast.
//------------------------------------------------------------------------

namespace CR
{

class RasterImpl;

//------------------------------------------------------------------------
// Interface class to isolate user from implementation details.
//------------------------------------------------------------------------

class CudaRaster
{
public:
    enum
    {
        RenderModeFlag_EnableBackfaceCulling = 1 << 0,   // Enable backface culling.
        RenderModeFlag_EnableDepthPeeling    = 1 << 1,   // Enable depth peeling. Must have a peel buffer set.
    };

public:
					        CudaRaster				(void);
					        ~CudaRaster				(void);

    void                    setBufferSize           (int width, int height, int numImages);              // Width and height are internally rounded up to multiples of tile size (8x8) for buffer sizes.
    void                    setViewport             (int width, int height, int offsetX, int offsetY);   // Tiled rendering viewport setup.
    void                    setRenderModeFlags      (unsigned int renderModeFlags);                      // Affects all subsequent calls to drawTriangles(). Defaults to zero.
    void                    deferredClear           (unsigned int clearColor);                           // Clears color and depth buffers during next call to drawTriangles().
    void                    setVertexBuffer         (void* vertices, int numVertices);                   // GPU pointer managed by caller. Vertex positions in clip space as float4 (x, y, z, w).
    void                    setIndexBuffer          (void* indices, int numTriangles);                   // GPU pointer managed by caller. Triangle index+color quadruplets as uint4 (idx0, idx1, idx2, color).
    void                    setTiebreakerColorBuffer(void* colors);                                      // GPU pointer managed by caller. Per-vertex packed RGBA8 colors for deterministic z-fight tiebreaker.
    void                    setDeterministicTiebreaker(bool enable);                                     // When true, uses vertex colors as deterministic z-fight tiebreaker instead of triIdx.
    bool                    drawTriangles           (const int* ranges, bool peel, cudaStream_t stream); // Ranges (offsets and counts) as #triangles entries, not as bytes. If NULL, draw all triangles. Returns false in case of internal overflow.
    void*                   getColorBuffer          (void);                                              // GPU pointer managed by CudaRaster.
    void*                   getDepthBuffer          (void);                                              // GPU pointer managed by CudaRaster.
    void                    swapDepthAndPeel        (void);                                              // Swap depth and peeling buffers.

private:
					        CudaRaster           	(const CudaRaster&); // forbidden
	CudaRaster&             operator=           	(const CudaRaster&); // forbidden

private:
    RasterImpl*             m_impl;                 // Opaque pointer to implementation.
};

//------------------------------------------------------------------------
} // namespace CR
