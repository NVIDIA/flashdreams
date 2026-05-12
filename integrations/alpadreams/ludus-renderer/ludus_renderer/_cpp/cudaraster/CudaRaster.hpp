/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/PrivateDefs.hpp"
#include <cstdint>
#include <memory>

namespace FW
{
class CudaSurface;
class CudaModule;
class Buffer;
struct PixelPipeSpec;
class String;
}

namespace CR
{
//------------------------------------------------------------------------
// CudaRaster host-side public interface.
//------------------------------------------------------------------------

class CudaRaster
{
public:
    enum RenderModeFlag
    {
        RenderModeFlag_EnableBackfaceCulling = 1u << 0,
        RenderModeFlag_EnableDepthPeeling = 1u << 1,
    };

public:
                            CudaRaster               (void);
                            ~CudaRaster              (void);

    void                    setBufferSize            (int width, int height, int numImages);
    void                    setViewport              (int width, int height, int offsetX, int offsetY);
    void                    setRenderModeFlags       (unsigned int flags);
    void                    deferredClear            (unsigned int clearColor);
    void                    setVertexBuffer          (const float* vertices, int numVertices);
    void                    setIndexBuffer           (const int32_t* indices, int numTriangles);
    void                    setTiebreakerColorBuffer (const uint32_t* colors);
    void                    setTiebreakerColorBuffer (const int32_t* colors)
    {
        setTiebreakerColorBuffer(reinterpret_cast<const uint32_t*>(colors));
    }
    void                    setDeterministicTiebreaker(bool enable);
    bool                    drawTriangles            (const int32_t* ranges, bool peel, cudaStream_t stream);
    void                    swapDepthAndPeel         (void);

    const uint32_t*         getColorBuffer           (void) const;
    const uint32_t*         getDepthBuffer           (void) const;
    int                     getBufferWidth           (void) const;
    int                     getBufferHeight          (void) const;
    int                     getNumImages             (void) const;

private:
    // TODO(port): profiling - expose upstream statistics/timing through the
    // new public API or remove after the Phase 2 audit. See PORT_NOTES.md.
    struct Stats // Statistics for the previous call to drawTriangles().
    {
        FW::F32             setupTime;  // Seconds spent in TriangleSetup.
        FW::F32             binTime;    // Seconds spent in BinRaster.
        FW::F32             coarseTime; // Seconds spent in CoarseRaster.
        FW::F32             fineTime;   // Seconds spent in FineRaster.
    };

    // TODO(port): debug-params - decide how upstream debug controls map to
    // the raw-pointer/cudaStream_t API. See PORT_NOTES.md.
    struct DebugParams // Host-side emulation of individual stages, for debugging purposes.
    {
        bool                emulateTriangleSetup;
        bool                emulateBinRaster;
        bool                emulateCoarseRaster;
        bool                emulateFineRaster;      // Only supports GouraudShader, BlendReplace, and BlendSrcOver.

        DebugParams(void)
        {
            emulateTriangleSetup    = false;
            emulateBinRaster        = false;
            emulateCoarseRaster     = false;
            emulateFineRaster       = false;
        }
    };

private:
    // Legacy upstream entry points kept private while public API is migrated.
    // TODO(port): audit these in Phase 2; 1:1 ports can be deleted, while
    // unported capabilities should keep targeted TODOs. See PORT_NOTES.md.
    void                    setSurfaces             (FW::CudaSurface* color, FW::CudaSurface* depth);
    void                    deferredClear           (const FW::Vec4f& color, FW::F32 depth);
    void                    setPixelPipe            (FW::CudaModule* module, const FW::String& name);
    void                    setVertexBuffer         (FW::Buffer* buf, FW::S64 ofs);
    void                    setIndexBuffer          (FW::Buffer* buf, FW::S64 ofs, int numTris);
    void                    drawTriangles           (void);
    Stats                   getStats                (void);
    FW::String              getProfilingInfo        (void);
    void                    setDebugParams          (const DebugParams& p);

private:
    void                    launchStages            (void);

    FW::Vec3i               setupPleq               (const FW::Vec3f& values, const FW::Vec2i& v0, const FW::Vec2i& d1, const FW::Vec2i& d2, FW::S32 area, int samplesLog2);

    bool                    setupTriangle           (int triIdx,
                                                     const FW::Vec4f& v0, const FW::Vec4f& v1, const FW::Vec4f& v2,
                                                     const FW::Vec2f& b0, const FW::Vec2f& b1, const FW::Vec2f& b2,
                                                     const FW::Vec3i& vidx);

    // TODO(port): emulation - host-side reference stage emulation is not
    // currently invoked by the active draw path. See PORT_NOTES.md.
    void                    emulateTriangleSetup    (void);
    void                    emulateBinRaster        (void);
    void                    emulateCoarseRaster     (void);
    void                    emulateFineRaster       (void);

private:
                            CudaRaster              (const CudaRaster&); // forbidden
    CudaRaster&             operator=               (const CudaRaster&); // forbidden

private:
    // State.

    FW::CudaSurface*        m_colorBuffer;
    FW::CudaSurface*        m_depthBuffer;

    uint32_t*               m_colorBufferRaw;
    uint32_t*               m_depthBufferRaw;
    uint32_t*               m_peelBufferRaw;
    int32_t*                m_triIdxBufferRaw;      // Per-pixel resident-fragment triangle index for deterministic tiebreaker.
    int                     m_triIdxStride;         // Row stride of m_triIdxBufferRaw in S32 elements (tile-aligned).
    cudaArray_t             m_colorArray;
    cudaArray_t             m_depthArray;
    cudaArray_t             m_peelArray;
    int                     m_width;
    int                     m_height;
    int                     m_numImages;
    int                     m_viewportWidth;
    int                     m_viewportHeight;
    int                     m_viewportOffsetX;
    int                     m_viewportOffsetY;
    unsigned int            m_renderModeFlags;
    bool                    m_deterministicTiebreaker;
    const uint32_t*         m_tiebreakerColors;
    const int32_t*          m_ranges;
    bool                    m_peelEnabled;
    cudaTextureObject_t     m_vertexTexObj;
    cudaTextureObject_t     m_triHeaderTexObj;
    cudaTextureObject_t     m_triDataTexObj;
    cudaSurfaceObject_t     m_colorSurfaceObj;
    cudaSurfaceObject_t     m_depthSurfaceObj;

    bool                    m_deferredClear;
    FW::U32                 m_clearColor;
    FW::U32                 m_clearDepth;

    FW::Buffer*             m_vertexBuffer;
    FW::S64                 m_vertexOfs;
    FW::Buffer*             m_indexBuffer;
    FW::S64                 m_indexOfs;
    FW::S32                 m_numTris;
    const float*            m_vertexBufferRaw;
    const int32_t*          m_indexBufferRaw;
    int                     m_numVertices;

    // Surfaces.

    FW::Vec2i               m_viewportSize;
    FW::Vec2i               m_sizePixels;
    FW::Vec2i               m_sizeBins;
    FW::S32                 m_numBins;
    FW::Vec2i               m_sizeTiles;
    FW::S32                 m_numTiles;
    FW::S32                 m_numSamples;
    FW::S32                 m_samplesLog2;

    // Pixel pipe.

    FW::CudaModule*         m_module;
    std::unique_ptr<FW::PixelPipeSpec> m_pipeSpec;
    FW::S32                 m_numSMs;
    FW::S32                 m_numFineWarps;

    // Buffers.

    FW::S32                 m_binBatchSize;

    FW::S32                 m_maxSubtris;
    std::unique_ptr<FW::Buffer> m_triSubtris;
    std::unique_ptr<FW::Buffer> m_triHeader;
    std::unique_ptr<FW::Buffer> m_triData;

    FW::S32                 m_maxBinSegs;
    std::unique_ptr<FW::Buffer> m_binFirstSeg;
    std::unique_ptr<FW::Buffer> m_binTotal;
    std::unique_ptr<FW::Buffer> m_binSegData;
    std::unique_ptr<FW::Buffer> m_binSegNext;
    std::unique_ptr<FW::Buffer> m_binSegCount;

    FW::S32                 m_maxTileSegs;
    std::unique_ptr<FW::Buffer> m_activeTiles;
    std::unique_ptr<FW::Buffer> m_tileFirstSeg;
    std::unique_ptr<FW::Buffer> m_tileSegData;
    std::unique_ptr<FW::Buffer> m_tileSegNext;
    std::unique_ptr<FW::Buffer> m_tileSegCount;

    // Stats, profiling, debug.

    CUevent                 m_evSetupBegin;
    CUevent                 m_evBinBegin;
    CUevent                 m_evCoarseBegin;
    CUevent                 m_evFineBegin;
    CUevent                 m_evFineEnd;
    std::unique_ptr<FW::Buffer> m_profData;
    DebugParams             m_debug;
};

//------------------------------------------------------------------------
}
