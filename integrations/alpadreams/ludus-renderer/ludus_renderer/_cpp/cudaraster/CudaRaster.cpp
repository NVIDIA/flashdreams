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

#include "CudaRaster.hpp"

using namespace FW;

//------------------------------------------------------------------------

static const struct
{
    const char* format;
} g_profCounters[] =
{
#define LAMBDA(ID, FORMAT) { FORMAT },
CR_PROFILING_COUNTERS(LAMBDA)
#undef LAMBDA
};

//------------------------------------------------------------------------

static const struct
{
    S32         parent;
    const char* format;
} g_profTimers[] =
{
#define LAMBDA(ID, PARENT, FORMAT) { (int)&((CRProfTimerOrder*)NULL)->PARENT, FORMAT },
CR_PROFILING_TIMERS(LAMBDA)
#undef LAMBDA
};

//------------------------------------------------------------------------

CudaRaster::CudaRaster(void)
:   m_colorBuffer   (NULL),
    m_depthBuffer   (NULL),

    m_deferredClear (false),
    m_clearColor    (0),
    m_clearDepth    (0),

    m_vertexBuffer  (NULL),
    m_vertexOfs     (0),
    m_indexBuffer   (NULL),
    m_indexOfs      (0),
    m_numTris       (0),

    m_module        (NULL),
    m_numSMs        (1),
    m_numFineWarps  (1),

    m_maxSubtris    (1),
    m_maxBinSegs    (1),
    m_maxTileSegs   (1)
{
    // Check CUDA version, compute capability, and NVCC availability.

    CudaModule::staticInit();
    if (!CudaModule::isAvailable())
        fail("CudaRaster: No CUDA-capable devices found!");
    if (CudaModule::getDriverVersion() < 40)
        fail("CudaRaster: CUDA 4.0 or later is required!");
    if (CudaModule::getComputeCapability() < 20)
        fail("CudaRaster: Compute capability 2.0 or better is required!");

    // Create CUDA events.

    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_evSetupBegin, 0));
    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_evBinBegin, 0));
    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_evCoarseBegin, 0));
    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_evFineBegin, 0));
    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_evFineEnd, 0));

    // Allocate fixed-size buffers.

    m_binFirstSeg.resizeDiscard(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
    m_binTotal.resizeDiscard(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
    m_activeTiles.resizeDiscard(CR_MAXTILES_SQR * sizeof(S32));
    m_tileFirstSeg.resizeDiscard(CR_MAXTILES_SQR * sizeof(S32));
}

//------------------------------------------------------------------------

CudaRaster::~CudaRaster(void)
{
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_evSetupBegin));
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_evBinBegin));
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_evCoarseBegin));
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_evFineBegin));
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_evFineEnd));
}

//------------------------------------------------------------------------

void CudaRaster::setSurfaces(CudaSurface* color, CudaSurface* depth)
{
    m_colorBuffer = color;
    m_depthBuffer = depth;
    if (!m_colorBuffer && !m_depthBuffer)
        return;

    // Check for errors.

    if (!m_colorBuffer)
        fail("CudaRaster: No color buffer specified!");

    if (!m_depthBuffer)
        fail("CudaRaster: No depth buffer specified!");

    if (m_colorBuffer->getFormat() != CudaSurface::Format_R8_G8_B8_A8)
        fail("CudaRaster: Unsupported color buffer format!");

    if (m_depthBuffer->getFormat() != CudaSurface::Format_Depth32)
        fail("CudaRaster: Unsupported depth buffer format!");

    if (m_colorBuffer->getSize() != m_depthBuffer->getSize())
        fail("CudaRaster: Mismatch in size between surfaces!");

    if (m_colorBuffer->getNumSamples() != m_depthBuffer->getNumSamples())
        fail("CudaRaster: Mismatch in multisampling between surfaces!");

    // Initialize parameters.

    m_viewportSize  = m_colorBuffer->getSize();
    m_sizePixels    = m_colorBuffer->getRoundedSize();
    m_sizeTiles     = m_sizePixels >> CR_TILE_LOG2;
    m_numTiles      = m_sizeTiles.x * m_sizeTiles.y;
    m_sizeBins      = (m_sizeTiles + CR_BIN_SIZE - 1) >> CR_BIN_LOG2;
    m_numBins       = m_sizeBins.x * m_sizeBins.y;
    m_numSamples    = m_colorBuffer->getNumSamples();
    m_samplesLog2   = m_colorBuffer->getSamplesLog2();
}

//------------------------------------------------------------------------

void CudaRaster::deferredClear(const Vec4f& color, F32 depth)
{
    m_deferredClear = true;
    m_clearColor = color.toABGR();
    m_clearDepth = encodeDepth((U32)min((U64)(depth * exp2(32)), (U64)FW_U32_MAX));
}

//------------------------------------------------------------------------

void CudaRaster::setPixelPipe(CudaModule* module, const String& name)
{
    m_module = module;
    if (!module)
        return;

    // Query kernels.

    if (!m_module->hasKernel(name + "_triangleSetup") ||
        !m_module->hasKernel(name + "_binRaster") ||
        !m_module->hasKernel(name + "_coarseRaster") ||
        !m_module->hasKernel(name + "_fineRaster"))
    {
        fail("CudaRaster: Invalid pixel pipe!");
    }

    m_setupKernel   = m_module->getKernel(name + "_triangleSetup");
    m_binKernel     = m_module->getKernel(name + "_binRaster");
    m_coarseKernel  = m_module->getKernel(name + "_coarseRaster");
    m_fineKernel    = m_module->getKernel(name + "_fineRaster");

    // Query spec.

    m_pipeSpec = *(const PixelPipeSpec*)m_module->getGlobal(name + "_spec").getPtr();

    // Query launch bounds.

    CudaModule::checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&m_numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CudaModule::getDeviceHandle()));
    CudaModule::checkError("cuFuncGetAttribute", cuFuncGetAttribute(&m_numFineWarps, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_fineKernel.getHandle()));
    m_numFineWarps = min(m_numFineWarps / 32, CR_FINE_MAX_WARPS);
}

//------------------------------------------------------------------------

void CudaRaster::setVertexBuffer(Buffer* buf, S64 ofs)
{
    m_vertexBuffer = buf;
    m_vertexOfs = ofs;
}

//------------------------------------------------------------------------

void CudaRaster::setIndexBuffer(Buffer* buf, S64 ofs, int numTris)
{
    m_indexBuffer = buf;
    m_indexOfs = ofs;
    m_numTris = numTris;
}

//------------------------------------------------------------------------

void CudaRaster::drawTriangles(void)
{
    int maxSubtrisSlack     = 4096;     // x 81B    = 324KB
    int maxBinSegsSlack     = 256;      // x 2137B  = 534KB
    int maxTileSegsSlack    = 4096;     // x 136B   = 544KB

    // Check for errors.

    if (!m_colorBuffer)
        fail("CudaRaster: Surfaces not set!");

    if (!m_module)
        fail("CudaRaster: Pixel pipe not set!");

    if (!m_vertexBuffer)
        fail("CudaRaster: Vertex buffer not set!");

    if (!m_indexBuffer)
        fail("CudaRaster: Index buffer not set!");

    if (m_pipeSpec.samplesLog2 != m_colorBuffer->getSamplesLog2())
        fail("CudaRaster: Mismatch in multisampling between pixel pipe and surface!");

    // Select batch size for BinRaster and estimate buffer sizes.
    {
        int roundSize  = CR_BIN_WARPS * 32;
        int minBatches = CR_BIN_STREAMS_SIZE * 2;
        int maxRounds  = 32;

        m_binBatchSize = clamp(m_numTris / (roundSize * minBatches), 1, maxRounds) * roundSize;
        m_maxSubtris = max(m_maxSubtris, m_numTris + maxSubtrisSlack);
        m_maxBinSegs = max(m_maxBinSegs, max(m_numBins * CR_BIN_STREAMS_SIZE, (m_numTris - 1) / CR_BIN_SEG_SIZE + 1) + maxBinSegsSlack);
        m_maxTileSegs = max(m_maxTileSegs, max(m_numTiles, (m_numTris - 1) / CR_TILE_SEG_SIZE + 1) + maxTileSegsSlack);
    }

    // Retry until successful.

    for (;;)
    {
        // Allocate buffers.

        if (m_maxSubtris > CR_MAXSUBTRIS_SIZE)
            fail("CudaRaster: CR_MAXSUBTRIS_SIZE exceeded!");

        m_triSubtris.resizeDiscard(m_maxSubtris * sizeof(U8));
        m_triHeader.resizeDiscard(m_maxSubtris * sizeof(CRTriangleHeader));
        m_triData.resizeDiscard(m_maxSubtris * sizeof(CRTriangleData));

        m_binSegData.resizeDiscard(m_maxBinSegs * CR_BIN_SEG_SIZE * sizeof(S32));
        m_binSegNext.resizeDiscard(m_maxBinSegs * sizeof(S32));
        m_binSegCount.resizeDiscard(m_maxBinSegs * sizeof(S32));

        m_tileSegData.resizeDiscard(m_maxTileSegs * CR_TILE_SEG_SIZE * sizeof(S32));
        m_tileSegNext.resizeDiscard(m_maxTileSegs * sizeof(S32));
        m_tileSegCount.resizeDiscard(m_maxTileSegs * sizeof(S32));

        // No profiling => launch stages.

        if (m_pipeSpec.profilingMode == ProfilingMode_Default)
            launchStages();

        // Otherwise => setup data buffer, and launch multiple times.

        else
        {
            int numCounters     = FW_ARRAY_SIZE(g_profCounters);
            int numTimers       = FW_ARRAY_SIZE(g_profTimers);
            int totalWarps      = m_numSMs * max(CR_BIN_WARPS, CR_COARSE_WARPS, m_numFineWarps);
            int bytesPerWarp    = max(numCounters * 64 * (int)sizeof(S64), numTimers * 32 * (int)sizeof(U32));

            m_profData.resizeDiscard(totalWarps * bytesPerWarp);
            m_profData.clear(0);
            *(CUdeviceptr*)m_module->getGlobal("c_profData").getMutablePtrDiscard() = m_profData.getMutableCudaPtr();

            int numLaunches = (m_pipeSpec.profilingMode == ProfilingMode_Timers) ? numTimers : 1;
            for (int i = 0; i < numLaunches; i++)
            {
                *(S32*)m_module->getGlobal("c_profLaunchIdx").getMutablePtrDiscard() = i;
                launchStages();
            }
        }

        // No overflows => done.

        const CRAtomics& atomics = *(const CRAtomics*)m_module->getGlobal("g_crAtomics").getPtr();
        if (atomics.numSubtris <= m_maxSubtris && atomics.numBinSegs <= m_maxBinSegs && atomics.numTileSegs <= m_maxTileSegs)
            break;

        // Grow buffers and retry.

        m_maxSubtris = max(m_maxSubtris, atomics.numSubtris + maxSubtrisSlack);
        m_maxBinSegs = max(m_maxBinSegs, atomics.numBinSegs + maxBinSegsSlack);
        m_maxTileSegs = max(m_maxTileSegs, atomics.numTileSegs + maxTileSegsSlack);
    }

    m_deferredClear = false;
}

//------------------------------------------------------------------------

CudaRaster::Stats CudaRaster::getStats(void)
{
    Stats stats;
    memset(&stats, 0, sizeof(Stats));
    CudaModule::sync(false);

    cuEventElapsedTime(&stats.setupTime,    m_evSetupBegin,     m_evBinBegin);
    cuEventElapsedTime(&stats.binTime,      m_evBinBegin,       m_evCoarseBegin);
    cuEventElapsedTime(&stats.coarseTime,   m_evCoarseBegin,    m_evFineBegin);
    cuEventElapsedTime(&stats.fineTime,     m_evFineBegin,      m_evFineEnd);

    stats.setupTime     *= 1.0e-3f;
    stats.binTime       *= 1.0e-3f;
    stats.coarseTime    *= 1.0e-3f;
    stats.fineTime      *= 1.0e-3f;
    return stats;
}

//------------------------------------------------------------------------

String CudaRaster::getProfilingInfo(void)
{
    String s;
    s += "\n";

    if (!m_module)
    {
        s += "Pixel pipe not set!\n";
    }

    // ProfilingMode_Default.

    if (m_pipeSpec.profilingMode == ProfilingMode_Default)
    {
        Stats               stats           = getStats();
        const CRAtomics&    atomics         = *(const CRAtomics*)m_module->getGlobal("g_crAtomics").getPtr();
        F32                 pctCoef         = 100.0f / (stats.setupTime + stats.binTime + stats.coarseTime + stats.fineTime);
        int                 bytesPerSubtri  = (int)(sizeof(U8) + sizeof(CRTriangleHeader) + sizeof(CRTriangleData));
        int                 bytesPerBinSeg  = (CR_BIN_SEG_SIZE + 2) * (int)sizeof(S32);
        int                 bytesPerTileSeg = (CR_TILE_SEG_SIZE + 2) * (int)sizeof(S32);

        s += "ProfilingMode_Default\n";
        s += "---------------------\n";
        s += "\n";
        s += sprintf("%-16s%.3f ms (%.0f%%)\n", "triangleSetup",    stats.setupTime * 1.0e3f,   stats.setupTime * pctCoef);
        s += sprintf("%-16s%.3f ms (%.0f%%)\n", "binRaster",        stats.binTime * 1.0e3f,     stats.binTime * pctCoef);
        s += sprintf("%-16s%.3f ms (%.0f%%)\n", "coarseRaster",     stats.coarseTime * 1.0e3f,  stats.coarseTime * pctCoef);
        s += sprintf("%-16s%.3f ms (%.0f%%)\n", "fineRaster",       stats.fineTime * 1.0e3f,    stats.fineTime * pctCoef);
        s += "\n";
        s += sprintf("%-16s%-10d(%.1f MB)\n", "numSubtris",   atomics.numSubtris,  (F32)(atomics.numSubtris * bytesPerSubtri) * exp2(-20));
        s += sprintf("%-16s%-10d(%.1f MB)\n", "numBinSegs",   atomics.numBinSegs,  (F32)(atomics.numBinSegs * bytesPerBinSeg) * exp2(-20));
        s += sprintf("%-16s%-10d(%.1f MB)\n", "numTileSegs",  atomics.numTileSegs, (F32)(atomics.numTileSegs * bytesPerTileSeg) * exp2(-20));
    }

    // ProfilingMode_Counters.

    else if (m_pipeSpec.profilingMode == ProfilingMode_Counters)
    {
        const S64*  counterPtr  = (const S64*)m_profData.getPtr();
        int         numCounters = FW_ARRAY_SIZE(g_profCounters);
        int         numWarps    = (int)m_profData.getSize() / (numCounters * 64 * (int)sizeof(S64));

        s += "ProfilingMode_Counters\n";
        s += "----------------------\n";
        s += "\n";

        for (int i = 0; i < numCounters; i++)
        {
            S64 num = 0;
            S64 denom = 0;
            for (int j = 0; j < numWarps; j++)
            for (int k = 0; k < 32; k++)
            {
                int idx = (j * numCounters + i) * 64 + k;
                num += counterPtr[idx + 0];
                denom += counterPtr[idx + 32];
            }
            s += sprintf(g_profCounters[i].format, (F64)num / max((F64)denom, 1.0));
        }
    }

    // ProfilingMode_Timers.

    else if (m_pipeSpec.profilingMode == ProfilingMode_Timers)
    {
        const U32*  timerPtr    = (const U32*)m_profData.getPtr();
        int         numTimers   = FW_ARRAY_SIZE(g_profTimers);
        int         numWarps    = (int)m_profData.getSize() / (numTimers * 32 * (int)sizeof(U32));

        Array<F64> timers;
        for (int i = 0; i < numTimers; i++)
        {
            U64 launchTotal = 0;
            for (int j = 0; j < numWarps; j++)
            {
                U32 warpTotal = 0;
                for (int k = 0; k < 32; k++)
                    warpTotal += timerPtr[(j * numTimers + i) * 32 + k];
                launchTotal += warpTotal;
            }
            timers.add((F64)launchTotal);
        }
        timers.add(0.0);

        s += "ProfilingMode_Timers\n";
        s += "--------------------\n";
        s += "\n";
        for (int i = 0; i < numTimers; i++)
            s += sprintf(g_profTimers[i].format, timers[i] / max(timers[g_profTimers[i].parent], 1.0) * 100.0);
    }

    else
    {
        s += "Invalid profiling mode!\n";
    }

    s += "\n";
    return s;
}

//------------------------------------------------------------------------

void CudaRaster::setDebugParams(const DebugParams& p)
{
    m_debug = p;
}

//------------------------------------------------------------------------

void CudaRaster::launchStages(void)
{
    // Set parameters.
    {
        CRParams& p = *(CRParams*)m_module->getGlobal("c_crParams").getMutablePtrDiscard();

        p.numTris           = m_numTris;
        p.vertexBuffer      = m_vertexBuffer->getCudaPtr(m_vertexOfs);
        p.indexBuffer       = m_indexBuffer->getCudaPtr(m_indexOfs);

        p.viewportWidth     = m_viewportSize.x;
        p.viewportHeight    = m_viewportSize.y;
        p.widthPixels       = m_sizePixels.x;
        p.heightPixels      = m_sizePixels.y;

        p.widthBins         = m_sizeBins.x;
        p.heightBins        = m_sizeBins.y;
        p.numBins           = m_numBins;

        p.widthTiles        = m_sizeTiles.x;
        p.heightTiles       = m_sizeTiles.y;
        p.numTiles          = m_numTiles;

        p.binBatchSize      = m_binBatchSize;

        p.deferredClear     = (m_deferredClear) ? 1 : 0;
        p.clearColor        = m_clearColor;
        p.clearDepth        = m_clearDepth;

        p.maxSubtris        = m_maxSubtris;
        p.triSubtris        = m_triSubtris.getMutableCudaPtrDiscard();
        p.triHeader         = m_triHeader.getMutableCudaPtrDiscard();
        p.triData           = m_triData.getMutableCudaPtrDiscard();

        p.maxBinSegs        = m_maxBinSegs;
        p.binFirstSeg       = m_binFirstSeg.getMutableCudaPtrDiscard();
        p.binTotal          = m_binTotal.getMutableCudaPtrDiscard();
        p.binSegData        = m_binSegData.getMutableCudaPtrDiscard();
        p.binSegNext        = m_binSegNext.getMutableCudaPtrDiscard();
        p.binSegCount		= m_binSegCount.getMutableCudaPtrDiscard();

        p.maxTileSegs       = m_maxTileSegs;
        p.activeTiles       = m_activeTiles.getMutableCudaPtrDiscard();
        p.tileFirstSeg      = m_tileFirstSeg.getMutableCudaPtrDiscard();
        p.tileSegData       = m_tileSegData.getMutableCudaPtrDiscard();
        p.tileSegNext       = m_tileSegNext.getMutableCudaPtrDiscard();
        p.tileSegCount      = m_tileSegCount.getMutableCudaPtrDiscard();
    }

    // Initialize atomics.
    {
        CRAtomics& a        = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtrDiscard();
        a.numSubtris        = m_numTris;
        a.binCounter        = 0;
        a.numBinSegs        = 0;
        a.coarseCounter     = 0;
        a.numTileSegs       = 0;
        a.numActiveTiles    = 0;
        a.fineCounter       = 0;
    }

    // Bind textures and surfaces.

    CUdeviceptr vertexPtr = m_vertexBuffer->getCudaPtr(m_vertexOfs);
    S64 vertexSize = m_vertexBuffer->getSize() - m_vertexOfs;

    m_module->setTexRef("t_vertexBuffer",   vertexPtr, vertexSize, CU_AD_FORMAT_FLOAT, 4);
    m_module->setTexRef("t_triHeader",      m_triHeader, CU_AD_FORMAT_UNSIGNED_INT32, 4);
    m_module->setTexRef("t_triData",        m_triData, CU_AD_FORMAT_UNSIGNED_INT32, 4);

    m_module->setSurfRef("s_colorBuffer",   m_colorBuffer->getCudaArray());
    m_module->setSurfRef("s_depthBuffer",   m_depthBuffer->getCudaArray());

    // Launch triangleSetup().

    CudaModule::checkError("cuEventRecord", cuEventRecord(m_evSetupBegin, NULL));

    if (!m_debug.emulateTriangleSetup)
    {
        m_setupKernel.preferShared().launch(m_numTris, Vec2i(32, CR_SETUP_WARPS));
    }
    else
    {
        emulateTriangleSetup();
        m_triSubtris.getCudaPtr();
        m_triHeader.getCudaPtr();
        m_triData.getCudaPtr();
    }

    // Launch binRaster().

    CudaModule::checkError("cuEventRecord", cuEventRecord(m_evBinBegin, NULL));

    if (!m_debug.emulateBinRaster)
    {
        Vec2i block(32, CR_BIN_WARPS);
        m_binKernel.preferShared().launch(Vec2i(CR_BIN_STREAMS_SIZE, 1) * block, block);
    }
    else
    {
        emulateBinRaster();
        m_binFirstSeg.getCudaPtr();
        m_binTotal.getCudaPtr();
        m_binSegData.getCudaPtr();
        m_binSegNext.getCudaPtr();
	    m_binSegCount.getCudaPtr();
    }

    // Launch coarseRaster().

    CudaModule::checkError("cuEventRecord", cuEventRecord(m_evCoarseBegin, NULL));

    if (!m_debug.emulateCoarseRaster)
    {
        Vec2i block(32, CR_COARSE_WARPS);
        m_coarseKernel.preferShared().launch(Vec2i(m_numSMs, 1) * block, block);
    }
    else
    {
        emulateCoarseRaster();
        m_activeTiles.getCudaPtr();
        m_tileFirstSeg.getCudaPtr();
        m_tileSegData.getCudaPtr();
        m_tileSegNext.getCudaPtr();
        m_tileSegCount.getCudaPtr();
    }

    // Launch fineRaster().

    CudaModule::checkError("cuEventRecord", cuEventRecord(m_evFineBegin, NULL));

    if (!m_debug.emulateFineRaster)
    {
        Vec2i block(32, m_numFineWarps);
        m_fineKernel.preferShared().launch(Vec2i(m_numSMs, 1) * block, block);
    }
    else
    {
        emulateFineRaster();
    }

    CudaModule::checkError("cuEventRecord", cuEventRecord(m_evFineEnd, NULL));
}

//------------------------------------------------------------------------

Vec3i CudaRaster::setupPleq(const Vec3f& values, const Vec2i& v0, const Vec2i& d1, const Vec2i& d2, S32 area, int samplesLog2)
{
    F64 t0 = (F64)values.x;
    F64 t1 = (F64)values.y - t0;
    F64 t2 = (F64)values.z - t0;
    F64 xc = (t1 * (F64)d2.y - t2 * (F64)d1.y) / (F64)area;
    F64 yc = (t2 * (F64)d1.x - t1 * (F64)d2.x) / (F64)area;

    Vec2i center = (v0 * 2 + min(d1.x, d2.x, 0) + max(d1.x, d2.x, 0)) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    Vec2i vc = v0 - (center << (CR_SUBPIXEL_LOG2 - samplesLog2));

    Vec3i pleq;
    pleq.x = (U32)(S64)floor(xc * exp2(CR_SUBPIXEL_LOG2 - samplesLog2) + 0.5);
    pleq.y = (U32)(S64)floor(yc * exp2(CR_SUBPIXEL_LOG2 - samplesLog2) + 0.5);
    pleq.z = (U32)(S64)floor(t0 - xc * (F64)vc.x - yc * (F64)vc.y + 0.5);
    pleq.z -= pleq.x * center.x + pleq.y * center.y;
    return pleq;
}

//------------------------------------------------------------------------

bool CudaRaster::setupTriangle(
    int triIdx,
    const Vec4f& v0, const Vec4f& v1, const Vec4f& v2,
    const Vec2f& b0, const Vec2f& b1, const Vec2f& b2,
    const Vec3i& vidx)
{
    // Snap vertices.

    Vec2f viewScale = Vec2f(m_viewportSize << (CR_SUBPIXEL_LOG2 - 1));
    Vec3f rcpW = 1.0f / Vec3f(v0.w, v1.w, v2.w);
    Vec2i p0 = Vec2i((S32)floor(v0.x * rcpW.x * viewScale.x + 0.5f), (S32)floor(v0.y * rcpW.x * viewScale.y + 0.5f));
    Vec2i p1 = Vec2i((S32)floor(v1.x * rcpW.y * viewScale.x + 0.5f), (S32)floor(v1.y * rcpW.y * viewScale.y + 0.5f));
    Vec2i p2 = Vec2i((S32)floor(v2.x * rcpW.z * viewScale.x + 0.5f), (S32)floor(v2.y * rcpW.z * viewScale.y + 0.5f));
    Vec2i d1 = p1 - p0;
    Vec2i d2 = p2 - p0;

    // Backfacing or degenerate => cull.

    S32 area = d1.x * d2.y - d1.y * d2.x;
    if (area <= 0)
        return false;

    // AABB falls between samples => cull.

    Vec2i lo = min(p0, p1, p2);
    Vec2i hi = max(p0, p1, p2);

    int sampleSize = 1 << (CR_SUBPIXEL_LOG2 - m_samplesLog2);
    Vec2i bias = (m_viewportSize << (CR_SUBPIXEL_LOG2 - 1)) - sampleSize / 2;
    Vec2i loc = (lo + bias + sampleSize - 1) & -sampleSize;
    Vec2i hic = (hi + bias) & -sampleSize;

    if (loc.x > hic.x || loc.y > hic.y)
        return false;

    // AABB covers 1 or 2 samples => cull if they are not covered.

    int diff = hic.x + hic.y - loc.x - loc.y;
    if (diff <= sampleSize)
    {
        loc -= bias;
        Vec2i t0 = p0 - loc;
        Vec2i t1 = p1 - loc;
        Vec2i t2 = p2 - loc;
        S64 e0 = (S64)t0.x * t1.y - (S64)t0.y * t1.x;
        S64 e1 = (S64)t1.x * t2.y - (S64)t1.y * t2.x;
        S64 e2 = (S64)t2.x * t0.y - (S64)t2.y * t0.x;

        if (e0 < 0 || e1 < 0 || e2 < 0)
        {
            if (diff == 0)
                return false;

            hic -= bias;
            t0 = p0 - hic;
            t1 = p1 - hic;
            t2 = p2 - hic;
            e0 = (S64)t0.x * t1.y - (S64)t0.y * t1.x;
            e1 = (S64)t1.x * t2.y - (S64)t1.y * t2.x;
            e2 = (S64)t2.x * t0.y - (S64)t2.y * t0.x;

            if (e0 < 0 || e1 < 0 || e2 < 0)
                return false;
        }
    }

    // Setup plane equations.

    Vec3f zvert = lerp(Vec3f(CR_DEPTH_MIN), Vec3f(CR_DEPTH_MAX), Vec3f(v0.z, v1.z, v2.z) * rcpW * 0.5f + 0.5f);
    Vec3f wvert = rcpW * (min(v0.w, v1.w, v2.w) * (F32)CR_BARY_MAX);
    Vec3f uvert = Vec3f(b0.x, b1.x, b2.x) * wvert;
    Vec3f vvert = Vec3f(b0.y, b1.y, b2.y) * wvert;

    Vec2i wv0 = p0 + (m_viewportSize << (CR_SUBPIXEL_LOG2 - 1));
    Vec2i zv0 = wv0 - (1 << (CR_SUBPIXEL_LOG2 - m_samplesLog2 - 1));
    Vec3i zpleq = setupPleq(zvert, zv0, d1, d2, area, m_samplesLog2);
    Vec3i wpleq = setupPleq(wvert, wv0, d1, d2, area, m_samplesLog2 + 1);
    Vec3i upleq = setupPleq(uvert, wv0, d1, d2, area, m_samplesLog2 + 1);
    Vec3i vpleq = setupPleq(vvert, wv0, d1, d2, area, m_samplesLog2 + 1);
    U32 zmin = (U32)max(floor(min(zvert) + 0.5f) - CR_LERP_ERROR(m_samplesLog2), 0.0f);
    U32 zslope = (U32)min(((U64)abs(zpleq.x) + abs(zpleq.y)) * (m_numSamples / 2), (U64)FW_U32_MAX);

    // Write CRTriangleData.

    CRTriangleData& td = ((CRTriangleData*)m_triData.getMutablePtr())[triIdx];
    td.zx = zpleq.x, td.zy = zpleq.y, td.zb = zpleq.z; td.zslope = zslope;
    td.wx = wpleq.x, td.wy = wpleq.y, td.wb = wpleq.z;
    td.ux = upleq.x, td.uy = upleq.y, td.ub = upleq.z;
    td.vx = vpleq.x, td.vy = vpleq.y, td.vb = vpleq.z;
    td.vi0 = vidx.x, td.vi1 = vidx.y, td.vi2 = vidx.z;

    // Write CRTriangleHeader.

    CRTriangleHeader& th = ((CRTriangleHeader*)m_triHeader.getMutablePtr())[triIdx];
    th.v0x = (S16)p0.x, th.v0y = (S16)p0.y;
    th.v1x = (S16)p1.x, th.v1y = (S16)p1.y;
    th.v2x = (S16)p2.x, th.v2y = (S16)p2.y;
    U32 f01 = (U8)cover8x8_selectFlips(d1.x, d1.y);
    U32 f12 = (U8)cover8x8_selectFlips(d2.x - d1.x, d2.y - d1.y);
    U32 f20 = (U8)cover8x8_selectFlips(-d2.x, -d2.y);
	th.misc = (zmin & 0xfffff000u) | (f01 << 6) | (f12 << 2) | (f20 >> 2);
    return true;
}

//------------------------------------------------------------------------

void CudaRaster::emulateTriangleSetup(void)
{
    const U8*               vertexBuffer    = (const U8*)m_vertexBuffer->getPtr(m_vertexOfs);
    const Vec3i*            indexBuffer     = (const Vec3i*)m_indexBuffer->getPtr(m_indexOfs);

    CRAtomics&              atomics         = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtr();
    U8*                     triSubtris      = (U8*)m_triSubtris.getMutablePtr();
    CRTriangleHeader*       triHeader       = (CRTriangleHeader*)m_triHeader.getMutablePtr();
    CRTriangleData*         triData         = (CRTriangleData*)m_triData.getMutablePtr();

    for (int triIdx = 0; triIdx < m_numTris; triIdx++)
    {
        const Vec3i& vidx = indexBuffer[triIdx];
        int numVerts = 3;
        bool needToClip = true;

        // Read vertices.

        Vec4f v[9];
        for (int i = 0; i < 3; i++)
            v[i] = *(const Vec4f*)(vertexBuffer + vidx[i] * m_pipeSpec.vertexStructSize);

        // Outside view frustum => cull.

        if ((v[0].x < -v[0].w && v[1].x < -v[1].w && v[2].x < -v[2].w) ||
            (v[0].x > +v[0].w && v[1].x > +v[1].w && v[2].x > +v[2].w) ||
            (v[0].y < -v[0].w && v[1].y < -v[1].w && v[2].y < -v[2].w) ||
            (v[0].y > +v[0].w && v[1].y > +v[1].w && v[2].y > +v[2].w) ||
            (v[0].z < -v[0].w && v[1].z < -v[1].w && v[2].z < -v[2].w) ||
            (v[0].z > +v[0].w && v[1].z > +v[1].w && v[2].z > +v[2].w))
        {
            numVerts = 0;
            needToClip = false;
        }

        // Within depth range => try to project.

        if (v[0].z >= -v[0].w && v[1].z >= -v[1].w && v[2].z >= -v[2].w &&
            v[0].z <= +v[0].w && v[1].z <= +v[1].w && v[2].z <= +v[2].w)
        {
            Vec2f viewScale = Vec2f(m_viewportSize << (CR_SUBPIXEL_LOG2 - 1));
            Vec2f p0 = v[0].getXY() / v[0].w * viewScale;
            Vec2f p1 = v[1].getXY() / v[1].w * viewScale;
            Vec2f p2 = v[2].getXY() / v[2].w * viewScale;
            Vec2f lo = min(p0, p1, p2);
            Vec2f hi = max(p0, p1, p2);

            // Within S16 range and small enough => no need to clip.
            // Note: aabbLimit comes from the fact that cover8x8
            // does not support guardband with maximal viewport.

            F32 aabbLimit = (F32)((1 << (CR_MAXVIEWPORT_LOG2 + CR_SUBPIXEL_LOG2)) - 1);
            if (min(lo) >= -32768.5f && max(hi) < 32767.5f && max(hi - lo) <= aabbLimit)
                needToClip = false;
        }

        // Clip if needed.

        Vec2f b[9];
        b[0] = Vec2f(0.0f, 0.0f);
        b[1] = Vec2f(1.0f, 0.0f);
        b[2] = Vec2f(0.0f, 1.0f);

        if (needToClip)
        {
            Vec4f v0 = v[0];
            Vec4f d1 = v[1] - v[0];
            Vec4f d2 = v[2] - v[0];

            F32 bary[18];
            numVerts = clipTriangleWithFrustum(bary, &v0.x, &v[1].x, &v[2].x, &d1.x, &d2.x);

            for (int i = 0; i < numVerts; i++)
            {
                b[i] = Vec2f(bary[i * 2 + 0], bary[i * 2 + 1]);
                v[i] = v0 + d1 * b[i].x + d2 * b[i].y;
            }
        }

        // Setup subtriangles.

        int numSubtris = 0;
        for (int i = 0; i < numVerts - 2; i++)
        {
            int subtriIdx = (numSubtris == 0) ? triIdx : min(atomics.numSubtris + numSubtris, m_maxSubtris - 1);
            if (setupTriangle(subtriIdx, v[0], v[i + 1], v[i + 2], b[0], b[i + 1], b[i + 2], vidx))
                numSubtris++;
        }
        triSubtris[triIdx] = (U8)numSubtris;

        // More than one subtriangle => create indirect reference.

        if (numSubtris > 1)
        {
            if (atomics.numSubtris < m_maxSubtris)
            {
                triHeader[atomics.numSubtris] = triHeader[triIdx];
                triData[atomics.numSubtris] = triData[triIdx];
            }
            triHeader[triIdx].misc = atomics.numSubtris;
            atomics.numSubtris += numSubtris;
        }
    }
}

//------------------------------------------------------------------------

void CudaRaster::emulateBinRaster(void)
{
    // Initialize.

    const U8*               triSubtris      = (const U8*)m_triSubtris.getPtr();
    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)m_triHeader.getPtr();

    CRAtomics&              atomics         = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtr();
    S32*                    binFirstSeg     = (S32*)m_binFirstSeg.getMutablePtr();
    S32*                    binTotal        = (S32*)m_binTotal.getMutablePtr();
    S32*                    binSegData      = (S32*)m_binSegData.getMutablePtr();
    S32*                    binSegNext      = (S32*)m_binSegNext.getMutablePtr();
    S32*                    binSegCount		= (S32*)m_binSegCount.getMutablePtr();

    if (atomics.numSubtris > m_maxSubtris)
        return;

    Array<S32> batchTris;
    Array<S32> currSeg(NULL, m_numBins * CR_BIN_STREAMS_SIZE);
    Array<S32> idxInSeg(NULL, m_numBins * CR_BIN_STREAMS_SIZE);

    for (int i = 0; i < m_numBins * CR_BIN_STREAMS_SIZE; i++)
    {
        binFirstSeg[i] = -1;
        binTotal[i] = 0;
        currSeg[i] = -1;
        idxInSeg[i] = CR_BIN_SEG_SIZE;
    }

    // Loop over batches.

    for (int batchIdx = 0; batchIdx * m_binBatchSize < m_numTris; batchIdx++)
    {
        // Collect triangles.

        batchTris.clear();
        int batchStart = batchIdx * m_binBatchSize;
        int batchEnd = min(batchStart + m_binBatchSize, m_numTris);
        for (int triIdx = batchStart; triIdx < batchEnd; triIdx++)
        {
			int numSubtris = triSubtris[triIdx];
            for (int subtriIdx = 0; subtriIdx < numSubtris; subtriIdx++)
                batchTris.add((triIdx << 3) | ((numSubtris == 1) ? 7 : subtriIdx));
        }

        // Rasterize each triangle to bins.

        for (int idxInBatch = 0; idxInBatch < batchTris.getSize(); idxInBatch++)
        {
            int triIdx = batchTris[idxInBatch];
            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = triHeader[dataIdx].misc + subtriIdx;

            // Read vertices and compute AABB.

            const CRTriangleHeader& tri = triHeader[dataIdx];
            Vec2i v0 = Vec2i(tri.v0x, tri.v0y);
            Vec2i d01 = Vec2i(tri.v1x, tri.v1y) - v0;
            Vec2i d02 = Vec2i(tri.v2x, tri.v2y) - v0;
            v0 += m_viewportSize * CR_SUBPIXEL_SIZE / 2;
            Vec2i lo = v0 + min(0, d01, d02);
            Vec2i hi = v0 + max(0, d01, d02);

            // Check against each bin.

            for (int binIdx = 0; binIdx < m_numBins; binIdx++)
            {
                int binX = binIdx % m_sizeBins.x;
                int binY = binIdx / m_sizeBins.x;
                int half = CR_BIN_SIZE * CR_TILE_SIZE * CR_SUBPIXEL_SIZE / 2;
                Vec2i center = (Vec2i(binX, binY) * 2 + 1) * half;

                // Outside AABB => skip.

                if (lo.x >= center.x + half || lo.y >= center.y + half || hi.x <= center.x - half || hi.y <= center.y - half)
                    continue;

                // No intersection => skip.

                Vec2i p0 = center - v0;
                Vec2i p1 = p0 - d01;
                Vec2i d12 = d02 - d01;
                if ((S64)p0.x * d01.y - (S64)p0.y * d01.x >= (abs(d01.x) + abs(d01.y)) * half) continue;
                if ((S64)p0.y * d02.x - (S64)p0.x * d02.y >= (abs(d02.x) + abs(d02.y)) * half) continue;
                if ((S64)p1.x * d12.y - (S64)p1.y * d12.x >= (abs(d12.x) + abs(d12.y)) * half) continue;

                // Segment full => allocate a new one.

                int si = binIdx * CR_BIN_STREAMS_SIZE + batchIdx % CR_BIN_STREAMS_SIZE;
                if (idxInSeg[si] == CR_BIN_SEG_SIZE)
                {
                    int segIdx = min(atomics.numBinSegs++, m_maxBinSegs - 1);
                    if (currSeg[si] == -1)
                        binFirstSeg[si] = segIdx;
                    else
                        binSegNext[currSeg[si]] = segIdx;

                    binSegNext[segIdx] = -1;
					binSegCount[segIdx] = CR_BIN_SEG_SIZE;
                    currSeg[si] = segIdx;
                    idxInSeg[si] = 0;
                }

                // Append to the current segment.

                binSegData[currSeg[si] * CR_BIN_SEG_SIZE + idxInSeg[si]] = triIdx;
                idxInSeg[si]++;
                binTotal[si]++;
            }
        }

        // Flush between batches.

        for (int i = 0; i < m_numBins * CR_BIN_STREAMS_SIZE; i++)
		{
			if (idxInSeg[i] != CR_BIN_SEG_SIZE)
				binSegCount[currSeg[i]] = idxInSeg[i];
            idxInSeg[i] = CR_BIN_SEG_SIZE;
		}
    }
}

//------------------------------------------------------------------------

void CudaRaster::emulateCoarseRaster(void)
{
    // Initialize.

    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)m_triHeader.getPtr();

    const S32*              binFirstSeg     = (const S32*)m_binFirstSeg.getPtr();
    const S32*              binSegData      = (const S32*)m_binSegData.getPtr();
    const S32*              binSegNext      = (const S32*)m_binSegNext.getPtr();
    const S32*              binSegCount     = (const S32*)m_binSegCount.getPtr();

    CRAtomics&              atomics         = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtr();
    S32*                    activeTiles     = (S32*)m_activeTiles.getMutablePtr();
    S32*                    tileFirstSeg    = (S32*)m_tileFirstSeg.getMutablePtr();
    S32*                    tileSegData     = (S32*)m_tileSegData.getMutablePtr();
    S32*                    tileSegNext     = (S32*)m_tileSegNext.getMutablePtr();
    S32*                    tileSegCount    = (S32*)m_tileSegCount.getMutablePtr();

    Array<S32> mergedTris;
    Array<S32> currSeg(NULL, m_numTiles);
    Array<S32> idxInSeg(NULL, m_numTiles);

    if (atomics.numSubtris > m_maxSubtris || atomics.numBinSegs > m_maxBinSegs)
        return;

    for (int i = 0; i < m_numTiles; i++)
    {
        tileFirstSeg[i] = -1;
        currSeg[i] = -1;
        idxInSeg[i] = CR_TILE_SEG_SIZE;
    }

    // Process each bin.

    for (int binIdx = 0; binIdx < m_numBins; binIdx++)
    {
        int binTileX = (binIdx % m_sizeBins.x) * CR_BIN_SIZE;
        int binTileY = (binIdx / m_sizeBins.x) * CR_BIN_SIZE;

        // Merge streams.

        mergedTris.clear();
        S32 streamSeg[CR_BIN_STREAMS_SIZE];
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
            streamSeg[i] = binFirstSeg[binIdx * CR_BIN_STREAMS_SIZE + i];

        for (;;)
        {
            // Pick the stream with the lowest triangle index.

            S64 smin = FW_S64_MAX;
            for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
                if (streamSeg[i] != -1)
                    smin = min(smin, ((S64)binSegData[streamSeg[i] * CR_BIN_SEG_SIZE] << 32) | i);
            if (smin == FW_S64_MAX)
                break;

            // Consume one segment from the stream.

            int segIdx = streamSeg[(S32)smin];
            streamSeg[(S32)smin] = binSegNext[segIdx];
            for (int i = 0; i < binSegCount[segIdx]; i++)
				mergedTris.add(binSegData[segIdx * CR_BIN_SEG_SIZE + i]);
        }

        // Rasterize each triangle into tiles.

        for (int mergedIdx = 0; mergedIdx < mergedTris.getSize(); mergedIdx++)
        {
            int triIdx = mergedTris[mergedIdx];
            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = triHeader[dataIdx].misc + subtriIdx;

            // Read vertices and compute AABB.

            const CRTriangleHeader& tri = triHeader[dataIdx];
            Vec2i v0 = Vec2i(tri.v0x, tri.v0y);
            Vec2i d01 = Vec2i(tri.v1x, tri.v1y) - v0;
            Vec2i d02 = Vec2i(tri.v2x, tri.v2y) - v0;
            v0 += m_viewportSize * CR_SUBPIXEL_SIZE / 2;
            Vec2i lo = v0 + min(0, d01, d02);
            Vec2i hi = v0 + max(0, d01, d02);

            // Check against each tile.

            for (int tileInBin = 0; tileInBin < CR_BIN_SQR; tileInBin++)
            {
                int tileX = tileInBin % CR_BIN_SIZE + binTileX;
                int tileY = tileInBin / CR_BIN_SIZE + binTileY;
                int half = CR_TILE_SIZE * CR_SUBPIXEL_SIZE / 2;
                Vec2i center = (Vec2i(tileX, tileY) * 2 + 1) * half;

                // Outside viewport => skip.

                if (tileX >= m_sizeTiles.x || tileY >= m_sizeTiles.y)
                    continue;

                // No intersection => skip.

                if (lo.x >= center.x + half || lo.y >= center.y + half || hi.x <= center.x - half || hi.y <= center.y - half)
                    continue;

                Vec2i p0 = center - v0;
                Vec2i p1 = p0 - d01;
                Vec2i d12 = d02 - d01;
                if ((S64)p0.x * d01.y - (S64)p0.y * d01.x >= (abs(d01.x) + abs(d01.y)) * half) continue;
                if ((S64)p0.y * d02.x - (S64)p0.x * d02.y >= (abs(d02.x) + abs(d02.y)) * half) continue;
                if ((S64)p1.x * d12.y - (S64)p1.y * d12.x >= (abs(d12.x) + abs(d12.y)) * half) continue;

                // Segment full => allocate a new one.

                int si = tileX + tileY * m_sizeTiles.x;
                if (idxInSeg[si] == CR_TILE_SEG_SIZE)
                {
                    int segIdx = min(atomics.numTileSegs++, m_maxTileSegs - 1);
                    if (currSeg[si] == -1)
                        tileFirstSeg[si] = segIdx;
                    else
                        tileSegNext[currSeg[si]] = segIdx;

                    tileSegNext[segIdx] = -1;
                    tileSegCount[segIdx] = CR_TILE_SEG_SIZE;
                    currSeg[si] = segIdx;
                    idxInSeg[si] = 0;
                }

                // Append to the current segment.

                tileSegData[currSeg[si] * CR_TILE_SEG_SIZE + idxInSeg[si]] = triIdx;
                idxInSeg[si]++;
            }
        }
    }

    // Flush.

    for (int i = 0; i < m_numTiles; i++)
    {
        if (currSeg[i] == -1 && m_deferredClear)
            tileFirstSeg[i] = -1;
        if (currSeg[i] != -1 || m_deferredClear)
            activeTiles[atomics.numActiveTiles++] = i;
		if (idxInSeg[i] != CR_TILE_SEG_SIZE)
			tileSegCount[currSeg[i]] = idxInSeg[i];
    }
}

//------------------------------------------------------------------------

void CudaRaster::emulateFineRaster(void)
{
    // Initialize.

    const U8*               vertexBuffer    = (const U8*)m_vertexBuffer->getPtr(m_vertexOfs);
    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)m_triHeader.getPtr();
    const CRTriangleData*   triData         = (const CRTriangleData*)m_triData.getPtr();

    CRAtomics&              atomics         = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtr();
    const S32*              activeTiles     = (S32*)m_activeTiles.getPtr();
    const S32*              tileFirstSeg    = (S32*)m_tileFirstSeg.getPtr();
    const S32*              tileSegData     = (S32*)m_tileSegData.getPtr();
    const S32*              tileSegNext     = (S32*)m_tileSegNext.getPtr();
    const S32*              tileSegCount    = (S32*)m_tileSegCount.getPtr();

    bool                    enableBlend     = (String(m_pipeSpec.blendShaderName) == "BlendSrcOver");

    Array<S32> mergedTris;
    Array<U32> colorBuffer(NULL, m_sizePixels.y * m_sizePixels.x * m_numSamples);
    Array<U32> depthBuffer(NULL, m_sizePixels.y * m_sizePixels.x * m_numSamples);

    if (atomics.numSubtris > m_maxSubtris || atomics.numBinSegs > m_maxBinSegs || atomics.numTileSegs > m_maxTileSegs)
        return;

    // Deferred clear => clear framebuffer.

    if (m_deferredClear)
    {
        for (int i = 0; i < colorBuffer.getSize(); i++)
        {
            colorBuffer[i] = m_clearColor;
            depthBuffer[i] = m_clearDepth;
        }
    }

    // Otherwise => download framebuffer.

    else
    {
        CUDA_MEMCPY2D copy;
        copy.srcXInBytes    = 0;
        copy.srcY           = 0;
        copy.srcMemoryType  = CU_MEMORYTYPE_ARRAY;
        copy.srcArray       = m_colorBuffer->getCudaArray();
        copy.dstXInBytes    = 0;
        copy.dstY           = 0;
        copy.dstMemoryType  = CU_MEMORYTYPE_HOST;
        copy.dstHost        = colorBuffer.getPtr();
        copy.dstPitch       = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.WidthInBytes   = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.Height         = m_sizePixels.y;

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));

        copy.srcArray       = m_depthBuffer->getCudaArray();
        copy.dstHost        = depthBuffer.getPtr();

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));
    }

    // Process each tile-triangle intersection.

    for (int activeIdx = 0; activeIdx < atomics.numActiveTiles; activeIdx++)
    {
        int tileIdx = activeTiles[activeIdx];
        Vec2i tilePixelPos = Vec2i(tileIdx % m_sizeTiles.x, tileIdx / m_sizeTiles.x) * CR_TILE_SIZE;

        // Collect triangles.

        mergedTris.clear();
        for (int segIdx = tileFirstSeg[tileIdx]; segIdx != -1; segIdx = tileSegNext[segIdx])
            for (int i = 0; i < tileSegCount[segIdx]; i++)
                mergedTris.add(tileSegData[segIdx * CR_TILE_SEG_SIZE + i]);

        // Rasterize each triangle into framebuffer.

        for (int mergedIdx = 0; mergedIdx < mergedTris.getSize(); mergedIdx++)
        {
            int triIdx = mergedTris[mergedIdx];
            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = triHeader[dataIdx].misc + subtriIdx;

            // Read vertices.

		    const CRTriangleHeader& th  = triHeader[dataIdx];
		    const CRTriangleData&   td  = triData[dataIdx];
            const GouraudVertex&    vd0 = *(const GouraudVertex*)(vertexBuffer + td.vi0 * m_pipeSpec.vertexStructSize);
            const GouraudVertex&    vd1 = *(const GouraudVertex*)(vertexBuffer + td.vi1 * m_pipeSpec.vertexStructSize);
            const GouraudVertex&    vd2 = *(const GouraudVertex*)(vertexBuffer + td.vi2 * m_pipeSpec.vertexStructSize);

            Vec2i v0 = Vec2i(th.v0x, th.v0y) + m_viewportSize * (CR_SUBPIXEL_SIZE / 2);
            Vec2i v1 = Vec2i(th.v1x, th.v1y) + m_viewportSize * (CR_SUBPIXEL_SIZE / 2);
            Vec2i v2 = Vec2i(th.v2x, th.v2y) + m_viewportSize * (CR_SUBPIXEL_SIZE / 2);

            // Setup edge functions.

            Vec2i d0 = v1 - v0;
            Vec2i d1 = v2 - v1;
            Vec2i d2 = v0 - v2;
            S64 b0 = (S64)v0.x * d0.y - (S64)v0.y * d0.x;
            S64 b1 = (S64)v1.x * d1.y - (S64)v1.y * d1.x;
            S64 b2 = (S64)v2.x * d2.y - (S64)v2.y * d2.x;
            S64 c0 = b0 + (abs(d0.x) + abs(d0.y)) * (CR_SUBPIXEL_SIZE / 2);
            S64 c1 = b1 + (abs(d1.x) + abs(d1.y)) * (CR_SUBPIXEL_SIZE / 2);
            S64 c2 = b2 + (abs(d2.x) + abs(d2.y)) * (CR_SUBPIXEL_SIZE / 2);
            if (d0.y > 0 || (d0.y == 0 && d0.x <= 0)) b0--;
            if (d1.y > 0 || (d1.y == 0 && d1.x <= 0)) b1--;
            if (d2.y > 0 || (d2.y == 0 && d2.x <= 0)) b2--;

            // Check against each pixel.

            for (int pixelIdx = 0; pixelIdx < CR_TILE_SQR; pixelIdx++)
            {
                Vec2i pixelPos = tilePixelPos + Vec2i(pixelIdx % CR_TILE_SIZE, pixelIdx / CR_TILE_SIZE);
                int pixelOfs = (tilePixelPos.x + pixelPos.y * m_sizePixels.x) * m_numSamples + (pixelIdx % CR_TILE_SIZE);

                // Test pixel coverage (conservative).

                S64 xx = (S64)(pixelPos.x * CR_SUBPIXEL_SIZE + CR_SUBPIXEL_SIZE / 2);
                S64 yy = (S64)(pixelPos.y * CR_SUBPIXEL_SIZE + CR_SUBPIXEL_SIZE / 2);
                if (xx * d0.y - yy * d0.x > c0) continue;
                if (xx * d1.y - yy * d1.x > c1) continue;
                if (xx * d2.y - yy * d2.x > c2) continue;

                // Test sample coverage (exact).
                // Test and update depth.

                U32 coverMask = 0;
                U32 writeMask = 0;
                for (int i = 0; i < m_numSamples; i++)
                {
                    U32 sampleX = pixelPos.x * m_numSamples + c_msaaPatterns[m_samplesLog2][i];
                    U32 sampleY = pixelPos.y * m_numSamples + i;

                    S64 xx = (S64)((sampleX * 2 + 1) << (CR_SUBPIXEL_LOG2 - m_samplesLog2 - 1));
                    S64 yy = (S64)((sampleY * 2 + 1) << (CR_SUBPIXEL_LOG2 - m_samplesLog2 - 1));
                    if (xx * d0.y - yy * d0.x > b0) continue;
                    if (xx * d1.y - yy * d1.x > b1) continue;
                    if (xx * d2.y - yy * d2.x > b2) continue;

                    coverMask |= 1 << i;

                    if ((m_pipeSpec.renderModeFlags & RenderModeFlag_EnableDepth) != 0)
                    {
                        U32 depth = td.zx * sampleX + td.zy * sampleY + td.zb;
                        if (depth >= depthBuffer[pixelOfs + i * CR_TILE_SIZE])
                            continue;
                        depthBuffer[pixelOfs + i * CR_TILE_SIZE] = depth;
                    }
                    writeMask |= 1 << i;
                }

                // No samples to write => skip shader & ROP.

                if (writeMask == 0)
                    continue;

                // Interpolate color.

                Vec4f color;
                if ((m_pipeSpec.renderModeFlags & RenderModeFlag_EnableLerp) == 0)
                    color = vd2.color;
                else
                {
                    int ctr = selectMSAACentroid(m_samplesLog2, coverMask);
                    int sampleX = pixelPos.x * m_numSamples * 2 + ((ctr == -1) ? m_numSamples : c_msaaPatterns[m_samplesLog2][ctr] * 2 + 1);
                    int sampleY = pixelPos.y * m_numSamples * 2 + ((ctr == -1) ? m_numSamples : ctr * 2 + 1);
                    F32 w = 1.0f / (F32)(td.wx * sampleX + td.wy * sampleY + td.wb);
                    F32 u = w * (F32)(td.ux * sampleX + td.uy * sampleY + td.ub);
                    F32 v = w * (F32)(td.vx * sampleX + td.vy * sampleY + td.vb);
                    color = vd0.color + (vd1.color - vd0.color) * u + (vd2.color - vd0.color) * v;
                }

                // Blend.

                U32 src = color.toABGR();
                U32 srcFactor = src >> 24;
                U32 dstFactor = 255 - srcFactor;

                for (int i = 0; i < m_numSamples; i++)
                {
                    if ((writeMask & (1 << i)) != 0)
                    {
                        U32& dst = colorBuffer[pixelOfs + i * CR_TILE_SIZE];
                        if (!enableBlend)
                            dst = src;
                        else
                            dst =
                                ((((((src >> 0)  & 0xFF) * srcFactor + ((dst >> 0)  & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 0)  |
                                ((((((src >> 8)  & 0xFF) * srcFactor + ((dst >> 8)  & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 8)  |
                                ((((((src >> 16) & 0xFF) * srcFactor + ((dst >> 16) & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 16) |
                                ((((((src >> 24) & 0xFF) * srcFactor + ((dst >> 24) & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 24);
                    }
                }
            }
        }
    }

    // Upload framebuffer.
    {
        CUDA_MEMCPY2D copy;
        copy.srcXInBytes    = 0;
        copy.srcY           = 0;
        copy.srcMemoryType  = CU_MEMORYTYPE_HOST;
        copy.srcHost        = colorBuffer.getPtr();
        copy.srcPitch       = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.dstXInBytes    = 0;
        copy.dstY           = 0;
        copy.dstMemoryType  = CU_MEMORYTYPE_ARRAY;
        copy.dstArray       = m_colorBuffer->getCudaArray();
        copy.WidthInBytes   = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.Height         = m_sizePixels.y;

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));

        copy.srcHost        = depthBuffer.getPtr();
        copy.dstArray       = m_depthBuffer->getCudaArray();

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));
    }
}

//------------------------------------------------------------------------
