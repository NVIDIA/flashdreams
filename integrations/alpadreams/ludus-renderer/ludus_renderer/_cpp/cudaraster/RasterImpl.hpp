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
#include "PrivateDefs.hpp"
#include "Buffer.hpp"
#include "CudaRaster.hpp"

namespace CR
{
//------------------------------------------------------------------------

class RasterImpl
{
public:
					        RasterImpl				(void);
					        ~RasterImpl				(void);

    void                    setBufferSize           (Vec3i size);
    void                    setViewport             (Vec2i size, Vec2i offset);
    void                    setRenderModeFlags      (U32 flags) { m_renderModeFlags = flags; }
    void                    deferredClear           (U32 color) { m_deferredClear = true; m_clearColor = color; }
    void                    setVertexBuffer         (void* ptr, int numVertices) { m_vertexPtr = ptr; m_numVertices = numVertices; } // GPU pointer.
    void                    setIndexBuffer          (void* ptr, int numTriangles) { m_indexPtr = ptr; m_numTriangles = numTriangles; } // GPU pointer.
    void                    setTiebreakerColorBuffer(void* ptr) { m_tiebreakerColorPtr = ptr; }
    void                    setDeterministicTiebreaker(bool enable) { m_deterministicTiebreaker = enable; }
    bool                    drawTriangles           (const Vec2i* ranges, bool peel, cudaStream_t stream);
    void*                   getColorBuffer          (void) { return m_colorBuffer.getPtr(); } // GPU pointer.
    void*                   getDepthBuffer          (void) { return m_depthBuffer.getPtr(); } // GPU pointer.
    void                    swapDepthAndPeel        (void);
    int                     getBufferWidth          (void) const { return m_bufferSizePixels.x; }
    int                     getBufferHeight         (void) const { return m_bufferSizePixels.y; }
    int                     getNumImages            (void) const { return m_numImages; }
    size_t                  getTotalBufferSizes     (void) const;

private:
    void                    launchStages            (bool instanceMode, bool peel, cudaStream_t stream);

    // State.

    unsigned int            m_renderModeFlags;
    bool                    m_deferredClear;
    unsigned int            m_clearColor;
    void*                   m_vertexPtr;
    void*                   m_indexPtr;
    void*                   m_tiebreakerColorPtr;
    bool                    m_deterministicTiebreaker;
    int                     m_numVertices;          // Input buffer size.
    int                     m_numTriangles;         // Input buffer size.
    size_t                  m_bufferSizesReported;  // Previously reported buffer sizes.

    // Surfaces.

    Buffer                  m_colorBuffer;
    Buffer                  m_depthBuffer;
    Buffer                  m_peelBuffer;
    int                     m_numImages;
    Vec2i                   m_bufferSizePixels;     // Internal buffer size.
    Vec2i                   m_bufferSizeVp;         // Total viewport size.
    Vec2i                   m_sizePixels;           // Internal size at which all computation is done, buffers reserved, etc.
    Vec2i                   m_sizeVp;               // Size to which output will be cropped outside, determines viewport size.
    Vec2i                   m_offsetPixels;         // Viewport offset for tiled rendering.
    Vec2i                   m_sizeBins;
    S32                     m_numBins;
    Vec2i                   m_sizeTiles;
    S32                     m_numTiles;

    // Launch sizes etc.

    S32                     m_numSMs;
    S32                     m_numCoarseBlocksPerSM;
    S32                     m_numFineBlocksPerSM;
    S32                     m_numFineWarpsPerBlock;

    // Global intermediate buffers. Individual images have offsets to these.

    Buffer                  m_crAtomics;
    HostBuffer              m_crAtomicsHost;
    HostBuffer              m_crImageParamsHost;
    Buffer                  m_crImageParamsExtra;
    Buffer                  m_triSubtris;
    Buffer                  m_triHeader;
    Buffer                  m_triData;
    Buffer                  m_triTiebreaker;
    Buffer                  m_binFirstSeg;
    Buffer                  m_binTotal;
    Buffer                  m_binSegData;
    Buffer                  m_binSegNext;
	Buffer                  m_binSegCount;
    Buffer                  m_activeTiles;
    Buffer                  m_tileFirstSeg;
    Buffer                  m_tileSegData;
    Buffer                  m_tileSegNext;
    Buffer                  m_tileSegCount;

    // Actual buffer sizes.

    S32                     m_maxSubtris;
    S32                     m_maxBinSegs;
    S32                     m_maxTileSegs;
};

//------------------------------------------------------------------------
} // namespace CR
