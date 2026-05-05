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
#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Projected triangle.
//------------------------------------------------------------------------

struct CRTriangleHeader
{
    S16 v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    S16 v0y;
    S16 v1x;
    S16 v1y;
    S16 v2x;
    S16 v2y;

    U32 misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
};

//------------------------------------------------------------------------

struct CRTriangleData
{
    U32 zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    U32 zy;
    U32 zb;
    U32 zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

    S32 wx;     // wx * (sampleX * 2 + 1) + wy * (sampleY * 2 + 1) + wb = minClipW / clipW * CR_BARY_MAX
    S32 wy;
    S32 wb;

    S32 ux;     // ux * (sampleX * 2 + 1) + uy * (sampleY * 2 + 1) + ub = baryU * minClipW / clipW * CR_BARY_MAX
    S32 uy;
    S32 ub;

    S32 vx;     // vx * (sampleX * 2 + 1) + vy * (sampleY * 2 + 1) + vb = baryV * minClipW / clipW * CR_BARY_MAX
    S32 vy;
    S32 vb;

    U32 vi0;    // Vertex indices.
    U32 vi1;
    U32 vi2;
};

//------------------------------------------------------------------------
// Device-side globals.
//------------------------------------------------------------------------

struct CRParams
{
    // Common.

    S32         numTris;
    CUdeviceptr vertexBuffer;       // numVerts * ShadedVertexSubclass
    CUdeviceptr indexBuffer;        // numTris * int3

    S32         viewportWidth;      // Viewport size. May be smaller than framebuffer.
    S32         viewportHeight;
    S32         widthPixels;        // widthTiles * CR_TILE_SIZE
    S32         heightPixels;       // heightTiles * CR_TILE_SIZE

    S32         widthBins;          // ceil(viewportWidth / CR_BIN_SIZE)
    S32         heightBins;         // ceil(viewportHeight / CR_BIN_SIZE)
    S32         numBins;            // widthBins * heightBins

    S32         widthTiles;         // ceil(viewportWidth / CR_TILE_SIZE)
    S32         heightTiles;        // ceil(viewportHeight / CR_TILE_SIZE)
    S32         numTiles;           // widthTiles * heightTiles

    S32         binBatchSize;       // Number of triangles per batch.

    S32         deferredClear;      // 1 = Clear framebuffer before rendering triangles.
    U32         clearColor;
    U32         clearDepth;

    // Setup output / bin input.

    S32         maxSubtris;
    CUdeviceptr triSubtris;         // maxSubtris * U8
    CUdeviceptr triHeader;          // maxSubtris * CRTriangleHeader
    CUdeviceptr triData;            // maxSubtris * CRTriangleData

    // Bin output / coarse input.

    S32         maxBinSegs;
    CUdeviceptr binFirstSeg;        // CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * (S32 segIdx), -1 = none
    CUdeviceptr binTotal;           // CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * (S32 numTris)
    CUdeviceptr binSegData;         // maxBinSegs * CR_BIN_SEG_SIZE * (S32 triIdx)
    CUdeviceptr binSegNext;         // maxBinSegs * (S32 segIdx), -1 = none
    CUdeviceptr binSegCount;        // maxBinSegs * (S32 numEntries)

    // Coarse output / fine input.

    S32         maxTileSegs;
    CUdeviceptr activeTiles;        // CR_MAXTILES_SQR * (S32 tileIdx)
    CUdeviceptr tileFirstSeg;       // CR_MAXTILES_SQR * (S32 segIdx), -1 = none
    CUdeviceptr tileSegData;        // maxTileSegs * CR_TILE_SEG_SIZE * (S32 triIdx)
    CUdeviceptr tileSegNext;        // maxTileSegs * (S32 segIdx), -1 = none
    CUdeviceptr tileSegCount;       // maxTileSegs * (S32 numEntries)
};

//------------------------------------------------------------------------

struct CRAtomics
{
    // Setup.

    S32         numSubtris;         // = numTris

    // Bin.

    S32         binCounter;         // = 0
    S32         numBinSegs;         // = 0

    // Coarse.

    S32         coarseCounter;      // = 0
    S32         numTileSegs;        // = 0
    S32         numActiveTiles;     // = 0

    // Fine.

    S32         fineCounter;        // = 0
};

//------------------------------------------------------------------------

struct PixelPipeSpec
{
    S32         samplesLog2;
    S32         vertexStructSize;
    U32         renderModeFlags;
    S32         profilingMode;
    char        blendShaderName[128];
};

//------------------------------------------------------------------------
// Profiling.
//------------------------------------------------------------------------

// Each counter stores separate numerator and denominator.
#define CR_PROFILING_COUNTERS(X) \
    X(SetupHeader,          "TriangleSetup:\n") \
    X(SetupViewportCull,    "- Viewport cull        %.1f%%\n") \
    X(SetupBackfaceCull,    "- Backface cull        %.1f%%\n") \
    X(SetupBetweenPixelsCull,"- Between pixels cull  %.1f%%\n") \
    X(SetupClipped,         "- Clipped              %.1f%%\n") \
    X(SetupSamplesPerTri,   "- Avg. samples / tri   %.2f\n\n") \
    X(BinHeader,            "BinRaster:\n") \
    X(BinInputOverflow,     "- Input overflows      %.0f\n") \
    X(BinTrisPerRound,      "- Avg. triangles/round %.1f\n") \
    X(BinTriBBArea,         "- Avg. tri bb size     %.1f\n") \
    X(BinTriSinglePath,     "- Coverage single path %.1f%%\n") \
    X(BinTriFastPath,       "- Coverage fast path   %.1f%%\n") \
    X(BinTriSlowPath,       "- Coverage slow path   %.1f%%\n") \
    X(BinTriSegAlloc,       "- Segment allocs/round %.1f\n\n") \
    X(CoarseHeader,         "CoarseRaster:\n") \
    X(CoarseBins,           "- Bins                 %.0f\n") \
    X(CoarseRoundsPerBin,   "- Rounds / Bin         %.1f\n") \
    X(CoarseMergePerRound,  "- Merge / Round        %.1f\n") \
    X(CoarseTrisPerRound,   "- Triangles / Round    %.1f\n") \
    X(CoarseTilesPerRound,  "- Tiles / Round        %.1f\n") \
    X(CoarseEmitsPerRound,  "- Emits / Round        %.1f\n") \
    X(CoarseAllocsPerRound, "- Allocs / Round       %.1f\n") \
    X(CoarseEmitsPerTri,    "- Emits / Triangle     %.2f\n") \
    X(CoarseCaseA,          "- Case A               %.0f%%\n") \
    X(CoarseCaseB,          "- Case B               %.0f%%\n") \
    X(CoarseCaseC,          "- Case C               %.0f%%\n\n") \
    X(FineHeader,           "FineRaster:\n") \
    X(FineTriangleCull,     "- Triangles culled\n") \
    X(FineStreamEndCull,    "  - End of stream      %.1f%%\n") \
    X(FineEarlyZCull,       "  - Early Z kill       %.1f%%\n") \
    X(FineEmptyCull,        "  - Empty coverage     %.1f%%\n") \
    X(FineZKill,            "- Z kills              %.1f%%\n") \
    X(FineMSAAKill,         "- MSAA kills           %.1f%%\n") \
    X(FineWarpUtil,         "- Post-kill warp util. %.1f%%\n") \
    X(FineBlendRounds,      "- Blend attempt rounds %.2f\n") \
    X(FineTriPerTile,       "- Avg. tri/tile        %.0f\n") \
    X(FineFragPerTri,       "- Avg. frag/tri        %.1f\n") \
    X(FineFragPerTile,      "- Avg. frag/tile       %.0f\n")

// Each timer is displayed as percentage relative to a parent timer.
#define CR_PROFILING_TIMERS(X) \
    X(SetupTotal,           None,               "TriangleSetup:\n") \
    X(SetupCompute,         None,               "- Compute\n") \
    X(SetupCullSnap,        SetupTotal,         "  - Cull & snap      %4.1f%%\n") \
    X(SetupPleq,            SetupTotal,         "  - Pleq setup       %4.1f%%\n") \
    X(SetupClip,            SetupTotal,         "  - Clip             %4.1f%%\n") \
    X(SetupMemory,          None,               "- Memory\n") \
    X(SetupVertexRead,      SetupTotal,         "  - Vertex read      %4.1f%%\n") \
    X(SetupNumSubWrite,     SetupTotal,         "  - NumSubtris write %4.1f%%\n") \
    X(SetupTriHeaderWrite,  SetupTotal,         "  - TriHeader write  %4.1f%%\n") \
    X(SetupTriDataWrite,    SetupTotal,         "  - TriData write    %4.1f%%\n") \
    X(SetupMarshal,         None,               "- Marshal\n") \
    X(SetupAllocSub,        SetupTotal,         "  - Allocate subtris %4.1f%%\n\n") \
    X(BinTotal,             None,               "BinRaster:\n") \
    X(BinCompute,           BinTotal,           "- Compute\n") \
    X(BinRasterize,         BinTotal,           "  - Rasterize        %4.1f%%\n") \
    X(BinRasterAtomic,      BinTotal,           "  - Raster atomics   %4.1f%%\n") \
    X(BinMemory,            BinTotal,           "- Memory\n") \
    X(BinReadTriHeader,     BinTotal,           "  - Read tri header  %4.1f%%\n") \
    X(BinReadTriangle,      BinTotal,           "  - Read triangle    %4.1f%%\n") \
    X(BinWrite,             BinTotal,           "  - Enqueue write    %4.1f%%\n") \
    X(BinMarshal,           BinTotal,           "- Marshal\n") \
    X(BinPickBin,           BinTotal,           "  - Pick bin         %4.1f%%\n") \
    X(BinCompactSubtri,     BinTotal,           "  - Compact subtri   %4.1f%%\n") \
    X(BinCount,             BinTotal,           "  - Count emit       %4.1f%%\n") \
    X(BinAlloc,             BinTotal,           "  - Allocate segs    %4.1f%%\n") \
    X(BinEnqueue,           BinTotal,           "  - Enqueue logic    %4.1f%%\n\n") \
    X(CoarseTotal,          None,               "CoarseRaster:\n") \
    X(CoarseCompute,        None,               "- Compute\n") \
    X(CoarseRasterize,      CoarseTotal,        "  - Rasterize        %4.1f%%\n") \
    X(CoarseRasterAtomic,   CoarseTotal,        "  - Raster atomics   %4.1f%%\n") \
    X(CoarseMemory,         None,               "- Memory\n") \
    X(CoarseStreamRead,     CoarseTotal,        "  - Stream read      %4.1f%%\n") \
    X(CoarseStreamWrite,    CoarseTotal,        "  - Stream write     %4.1f%%\n") \
    X(CoarseTriRead,        CoarseTotal,        "  - Triangle read    %4.1f%%\n") \
    X(CoarseMarshal,        None,               "- Marshal\n") \
    X(CoarseSort,           CoarseTotal,        "  - Sort             %4.1f%%\n") \
    X(CoarseBinInit,        CoarseTotal,        "  - Bin init         %4.1f%%\n") \
    X(CoarseMerge,          CoarseTotal,        "  - Merge            %4.1f%%\n") \
    X(CoarseMergeSum,       CoarseTotal,        "  - Merge prefsum    %4.1f%%\n") \
    X(CoarseCount,          CoarseTotal,        "  - Count            %4.1f%%\n") \
    X(CoarseCountSum,       CoarseTotal,        "  - Count prefsum    %4.1f%%\n") \
    X(CoarseEmit,           CoarseTotal,        "  - Emit             %4.1f%%\n") \
    X(CoarseEmitBitFind,    CoarseTotal,        "  - Emit bitfind     %4.1f%%\n") \
    X(CoarsePatch,          CoarseTotal,        "  - Patch ptrs       %4.1f%%\n") \
    X(CoarseBinDeinit,      CoarseTotal,        "  - Bin deinit       %4.1f%%\n\n") \
    X(FineTotal,            None,               "FineRaster:\n") \
    X(FineShade,            FineTotal,          "- Shader             %4.1f%%\n") \
    X(FineCompute,          FineTotal,          "- Compute\n") \
    X(FineUpdateTileZ,      FineTotal,          "  - Tile Z update    %4.1f%%\n") \
    X(FineEarlyZCull,       FineTotal,          "  - LRZ cull         %4.1f%%\n") \
    X(FinePixelCoverage,    FineTotal,          "  - Pixel coverage   %4.1f%%\n") \
    X(FineFragmentScan,     FineTotal,          "  - Fragment scan    %4.1f%%\n") \
    X(FineFindBit,          FineTotal,          "  - Bit finder       %4.1f%%\n") \
    X(FineZKill,            FineTotal,          "  - Z kill           %4.1f%%\n") \
    X(FineSampleCoverage,   FineTotal,          "  - Sample coverage  %4.1f%%\n") \
    X(FineROPConfResolve,   FineTotal,          "  - ROP conf resolve %4.1f%%\n") \
    X(FineROPBlend,         FineTotal,          "  - ROP blend        %4.1f%%\n") \
    X(FineMemory,           FineTotal,          "- Memory\n") \
    X(FineReadTile,         FineTotal,          "  - Read tile        %4.1f%%\n") \
    X(FineReadTriangle,     FineTotal,          "  - Read triangle    %4.1f%%\n") \
    X(FineReadZData,        FineTotal,          "  - Read tri Z data  %4.1f%%\n") \
    X(FineROPRead,          FineTotal,          "  - ROP frag read    %4.1f%%\n") \
    X(FineROPWrite,         FineTotal,          "  - ROP frag write   %4.1f%%\n") \
    X(FineWriteTile,        FineTotal,          "  - Write tile       %4.1f%%\n") \
    X(FineMarshal,          FineTotal,          "- Marshal\n") \
    X(FinePickTile,         FineTotal,          "  - Pick tile        %4.1f%%\n") \
    X(FineFragmentEnqueue,  FineTotal,          "  - Fragment enq     %4.1f%%\n") \
    X(FineFragmentDistr,    FineTotal,          "  - Fragment distr   %4.1f%%\n") \

//------------------------------------------------------------------------

struct CRProfCounterOrder
{
#define LAMBDA(ID, FORMAT) U8 ID;
    CR_PROFILING_COUNTERS(LAMBDA)
#undef LAMBDA
    U8 None;
};

struct CRProfTimerOrder
{
#define LAMBDA(ID, PARENT, FORMAT) U8 ID;
    CR_PROFILING_TIMERS(LAMBDA)
#undef LAMBDA
    U8 None;
};

//------------------------------------------------------------------------
}
