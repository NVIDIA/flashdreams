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
#include "CudaSurface.hpp"
#include "gui/Image.hpp"
#include "gpu/CudaModule.hpp"
#include "cuda/PixelPipe.hpp"
#include "cuda/PrivateDefs.hpp"

namespace FW
{
//------------------------------------------------------------------------
// CudaRaster host-side public interface.
//------------------------------------------------------------------------

class CudaRaster
{
public:
    struct Stats // Statistics for the previous call to drawTriangles().
    {
        F32                 setupTime;  // Seconds spent in TriangleSetup.
        F32                 binTime;    // Seconds spent in BinRaster.
        F32                 coarseTime; // Seconds spent in CoarseRaster.
        F32                 fineTime;   // Seconds spent in FineRaster.
    };

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

public:
					        CudaRaster				(void);
					        ~CudaRaster				(void);

    void                    setSurfaces             (CudaSurface* color, CudaSurface* depth);       // Set before calling other methods.
    void                    deferredClear           (const Vec4f& color = 0.0f, F32 depth = 1.0f);  // Clear surfaces on the next call to drawTriangles().

    void                    setPixelPipe            (CudaModule* module, const String& name);       // See CR_DEFINE_PIXEL_PIPE() in PixelPipe.hpp.
    void                    setVertexBuffer         (Buffer* buf, S64 ofs);
    void                    setIndexBuffer          (Buffer* buf, S64 ofs, int numTris);
	void			        drawTriangles			(void);                                         // Draw all triangles specified by the current index buffer.

    Stats                   getStats                (void);
    String                  getProfilingInfo        (void);                                         // See CR_PROFILING_MODE in PixelPipe.hpp.
    void                    setDebugParams          (const DebugParams& p);

private:
    void                    launchStages            (void);

    Vec3i                   setupPleq               (const Vec3f& values, const Vec2i& v0, const Vec2i& d1, const Vec2i& d2, S32 area, int samplesLog2);

    bool                    setupTriangle           (int triIdx,
                                                     const Vec4f& v0, const Vec4f& v1, const Vec4f& v2,
                                                     const Vec2f& b0, const Vec2f& b1, const Vec2f& b2,
                                                     const Vec3i& vidx);

    void                    emulateTriangleSetup    (void);
    void                    emulateBinRaster        (void);
    void                    emulateCoarseRaster     (void);
    void                    emulateFineRaster       (void);

private:
					        CudaRaster           	(const CudaRaster&); // forbidden
	CudaRaster&             operator=           	(const CudaRaster&); // forbidden

private:
    // State.

    CudaSurface*            m_colorBuffer;
    CudaSurface*            m_depthBuffer;

    bool                    m_deferredClear;
    U32                     m_clearColor;
    U32                     m_clearDepth;

    Buffer*                 m_vertexBuffer;
    S64                     m_vertexOfs;
    Buffer*                 m_indexBuffer;
    S64                     m_indexOfs;
    S32                     m_numTris;

    // Surfaces.

    Vec2i                   m_viewportSize;
    Vec2i                   m_sizePixels;
    Vec2i                   m_sizeBins;
    S32                     m_numBins;
    Vec2i                   m_sizeTiles;
    S32                     m_numTiles;
    S32                     m_numSamples;
    S32                     m_samplesLog2;

    // Pixel pipe.

    CudaModule*             m_module;
    CudaKernel              m_setupKernel;
    CudaKernel              m_binKernel;
    CudaKernel              m_coarseKernel;
    CudaKernel              m_fineKernel;
    PixelPipeSpec           m_pipeSpec;
    S32                     m_numSMs;
    S32                     m_numFineWarps;

    // Buffers.

    S32                     m_binBatchSize;

    S32                     m_maxSubtris;
    Buffer                  m_triSubtris;
    Buffer                  m_triHeader;
    Buffer                  m_triData;

    S32                     m_maxBinSegs;
    Buffer                  m_binFirstSeg;
    Buffer                  m_binTotal;
    Buffer                  m_binSegData;
    Buffer                  m_binSegNext;
	Buffer				    m_binSegCount;

    S32                     m_maxTileSegs;
    Buffer                  m_activeTiles;
    Buffer                  m_tileFirstSeg;
    Buffer                  m_tileSegData;
    Buffer                  m_tileSegNext;
    Buffer                  m_tileSegCount;

    // Stats, profiling, debug.

    CUevent                 m_evSetupBegin;
    CUevent                 m_evBinBegin;
    CUevent                 m_evCoarseBegin;
    CUevent                 m_evFineBegin;
    CUevent                 m_evFineEnd;
    Buffer                  m_profData;
    DebugParams             m_debug;
};

//------------------------------------------------------------------------
}
