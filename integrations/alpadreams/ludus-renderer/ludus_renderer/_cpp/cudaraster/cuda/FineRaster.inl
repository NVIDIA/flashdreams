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

//------------------------------------------------------------------------
// Shader wrappers.
//------------------------------------------------------------------------

template <int SamplesLog2>
__device__ __inline__ void computeBarys(
    Vec3f& bary, Vec3f& baryDX, Vec3f& baryDY,
    const Vec3i& wpleq, const Vec3i& upleq, const Vec3i& vpleq,
    int sampleX, int sampleY)
{
    F32 w = 1.0f / (F32)(wpleq.x * sampleX + wpleq.y * sampleY + wpleq.z);
    F32 u = w * (F32)(upleq.x * sampleX + upleq.y * sampleY + upleq.z);
    F32 v = w * (F32)(vpleq.x * sampleX + vpleq.y * sampleY + vpleq.z);
    bary = Vec3f(1.0f - u - v, u, v);

    F32 wd = w * (F32)(1 << (SamplesLog2 + 1));
    F32 udx = wd * ((F32)upleq.x - u * (F32)wpleq.x);
    F32 udy = wd * ((F32)upleq.y - u * (F32)wpleq.y);
    F32 vdx = wd * ((F32)vpleq.x - v * (F32)wpleq.x);
    F32 vdy = wd * ((F32)vpleq.y - v * (F32)wpleq.y);
    baryDX = Vec3f(-udx - vdx, udx, vdx);
    baryDY = Vec3f(-udy - vdy, udy, vdy);
}

//------------------------------------------------------------------------

template <class VertexClass, class FragmentShaderClass, int SamplesLog2, U32 RenderModeFlags>
__device__ __inline__ void runFragmentShader(
    FragmentShaderClass& fs,
    int triIdx, int dataIdx, int pixelX, int pixelY, U32 centroid, volatile U32* shared)
{
    // Initialize.

    uint4 t3 = tex1Dfetch(t_triData, dataIdx * 4 + 3); // vb, vi0, vi1, vi2

    fs.m_triIdx         = triIdx;
    fs.m_vertIdx        = Vec3i(t3.y, t3.z, t3.w);
    fs.m_pixelPos       = Vec2i(pixelX, pixelY);
    fs.m_vertexBytes    = sizeof(VertexClass);
    fs.m_shared         = (volatile F32*)shared;

    fs.m_color          = 0xFF0000FF;
    fs.m_discard        = false;

    // Interpolation disabled => sample varyings at the last vertex.

    if ((RenderModeFlags & RenderModeFlag_EnableLerp) == 0)
    {
        fs.m_center     = Vec3f(0.0f, 0.0f, 1.0f);
        fs.m_centerDX   = 0.0f;
        fs.m_centerDY   = 0.0f;

        fs.m_centroid   = Vec3f(0.0f, 0.0f, 1.0f);
        fs.m_centroidDX = 0.0f;
        fs.m_centroidDY = 0.0f;
    }

    // Interpolation enabled => compute barys.

    else
    {
        // Fetch pleqs.

        uint4 t1 = tex1Dfetch(t_triData, dataIdx * 4 + 1); // wx, wy, wb, ux
        uint4 t2 = tex1Dfetch(t_triData, dataIdx * 4 + 2); // uy, ub, vx, vy
        Vec3i wpleq(t1.x, t1.y, t1.z);
        Vec3i upleq(t1.w, t2.x, t2.y);
        Vec3i vpleq(t2.z, t2.w, t3.x);

        // Compute barys for pixel center.

        computeBarys<SamplesLog2>(
            fs.m_center, fs.m_centerDX,
            fs.m_centerDY, wpleq, upleq, vpleq,
            (pixelX * 2 + 1) << SamplesLog2,
            (pixelY * 2 + 1) << SamplesLog2);

        // Compute barys for triangle centroid.

        if (SamplesLog2 == 0)
        {
            fs.m_centroid = fs.m_center;
            fs.m_centroidDX = fs.m_centerDX;
            fs.m_centroidDY = fs.m_centerDY;
        }
        else
        {
            computeBarys<SamplesLog2>(
                fs.m_centroid, fs.m_centroidDX, fs.m_centroidDY,
                wpleq, upleq, vpleq,
                (pixelX << (SamplesLog2 + 1)) + (centroid & 0xF),
                (pixelY << (SamplesLog2 + 1)) + (centroid >> 4));
        }
    }

    // Run shader.

    fs.run();
}

//------------------------------------------------------------------------

template <class BlendShaderClass>
__device__ __inline__ void runBlendShader(
    BlendShaderClass& bs,
    int triIdx, int pixelX, int pixelY, int sampleIdx, U32 src, U32 dst)
{
    bs.m_triIdx     = triIdx;
    bs.m_pixelPos   = Vec2i(pixelX, pixelY);
    bs.m_sampleIdx  = sampleIdx;
    bs.m_src        = src;
    bs.m_dst        = dst;

    bs.m_color      = 0xFF0000FF;
    bs.m_writeColor = true;

    bs.run();
}

//------------------------------------------------------------------------
// Utility funcs.
//------------------------------------------------------------------------

template <int SamplesLog2>
__device__ __inline__ void setupCentroidLUT(volatile U8* lut)
{
    for (int mask = threadIdx.x + threadIdx.y * 32; mask < (1 << (1 << SamplesLog2)); mask += blockDim.y * 32)
    {
        U32 value = 0x11 << SamplesLog2;
        int y = selectMSAACentroid(SamplesLog2, mask);
        if (y != -1)
        {
            int x = c_msaaPatterns[SamplesLog2][y];
            value = x * 0x02 + y * 0x20 + 0x11;
        }
        lut[mask] = (U8)value;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void initTileZMax(U32& tileZMax, bool& tileZUpd, volatile U32* tileDepth)
{
    tileZMax = CR_DEPTH_MAX;
    tileZUpd = (::min(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]) < tileZMax);
}

template <U32 RenderModeFlags>
__device__ __inline__ void updateTileZMax(U32& tileZMax, bool& tileZUpd, volatile U32* tileDepth, volatile U32* temp)
{
    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0 && __any(tileZUpd))
    {
        U32 z = ::max(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]);
        temp[threadIdx.x + 16] = z;
        z = ::max(z, temp[threadIdx.x + 16 -  1]); temp[threadIdx.x + 16] = z;
        z = ::max(z, temp[threadIdx.x + 16 -  2]); temp[threadIdx.x + 16] = z;
        z = ::max(z, temp[threadIdx.x + 16 -  4]); temp[threadIdx.x + 16] = z;
        z = ::max(z, temp[threadIdx.x + 16 -  8]); temp[threadIdx.x + 16] = z;
        z = ::max(z, temp[threadIdx.x + 16 - 16]); temp[threadIdx.x + 16] = z;
        tileZMax = temp[47];
        tileZUpd = false;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void getTriangle(S32& triIdx, S32& dataIdx, uint4& triHeader, S32& segment)
{
    const CRTriangleHeader* triHeaderPtr    = (const CRTriangleHeader*)c_crParams.triHeader;
    const S32*              tileSegData     = (const S32*)c_crParams.tileSegData;
    const S32*              tileSegNext     = (const S32*)c_crParams.tileSegNext;
    const S32*              tileSegCount    = (const S32*)c_crParams.tileSegCount;

    if (threadIdx.x >= tileSegCount[segment])
    {
        CR_COUNT(FineStreamEndCull, 100, 1);
        triIdx = -1;
        dataIdx = -1;
    }
    else
    {
        CR_COUNT(FineStreamEndCull, 0, 1);
        int subtriIdx = tileSegData[segment * CR_TILE_SEG_SIZE + threadIdx.x];
        triIdx = subtriIdx >> 3;
        dataIdx = triIdx;
        subtriIdx &= 7;
        if (subtriIdx != 7)
            dataIdx = triHeaderPtr[triIdx].misc + subtriIdx;
        triHeader = tex1Dfetch(t_triHeader, dataIdx);
    }

    // count triangles per tile using thread 0
    CR_COUNT(FineTriPerTile, tileSegCount[segment], 0);

    // advance to next segment
    segment = tileSegNext[segment];
}

//------------------------------------------------------------------------

template <U32 RenderModeFlags>
__device__ __inline__ bool earlyZCull(uint4 triHeader, U32 tileZMax)
{
    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
    {
        U32 zmin = triHeader.w & 0xFFFFF000;
        CR_COUNT(FineEarlyZCull, (zmin >= tileZMax) ? 100 : 0, 1);
        if (zmin >= tileZMax)
            return true;
    }
    return false;
}

//------------------------------------------------------------------------

template <int SamplesLog2>
__device__ __inline__ U64 trianglePixelCoverage(const uint4& triHeader, int tileX, int tileY, volatile U64* s_cover8x8_lut)
{
    int baseX = (tileX << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((c_crParams.viewportWidth  - 1) << (CR_SUBPIXEL_LOG2 - 1));
    int baseY = (tileY << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((c_crParams.viewportHeight - 1) << (CR_SUBPIXEL_LOG2 - 1));

    // extract S16 vertex positions while subtracting tile coordinates
    S32 v0x  = sub_s16lo_s16lo(triHeader.x, baseX);
    S32 v0y  = sub_s16hi_s16lo(triHeader.x, baseY);
    S32 v01x = sub_s16lo_s16lo(triHeader.y, triHeader.x);
    S32 v01y = sub_s16hi_s16hi(triHeader.y, triHeader.x);
    S32 v20x = sub_s16lo_s16lo(triHeader.x, triHeader.z);
    S32 v20y = sub_s16hi_s16hi(triHeader.x, triHeader.z);

    // extract flipbits
    U32 f01 = (triHeader.w >> 6) & 0x3C;
    U32 f12 = (triHeader.w >> 2) & 0x3C;
    U32 f20 = (triHeader.w << 2) & 0x3C;

    // compute per-edge coverage masks
    U64 c01, c12, c20;
    if (SamplesLog2 == 0)
    {
        c01 = cover8x8_exact_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
        c12 = cover8x8_exact_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
        c20 = cover8x8_exact_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);
    }
    else
    {
        c01 = cover8x8_conservative_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
        c12 = cover8x8_conservative_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
        c20 = cover8x8_conservative_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);
    }

    // combine masks
    return c01 & c12 & c20;
}

//------------------------------------------------------------------------

template <int SamplesLog2>
__device__ __inline__ U32 triangleSampleCoverage(const uint4& triHeader, int pixelX, int pixelY)
{
    int baseX = (pixelX << CR_SUBPIXEL_LOG2) - ((c_crParams.viewportWidth  - 1) << (CR_SUBPIXEL_LOG2 - 1));
    int baseY = (pixelY << CR_SUBPIXEL_LOG2) - ((c_crParams.viewportHeight - 1) << (CR_SUBPIXEL_LOG2 - 1));

    // extract S16 vertex positions while subtracting pixel coordinates
    S32 v0x  = sub_s16lo_s16lo(triHeader.x, baseX);
    S32 v0y  = sub_s16hi_s16lo(triHeader.x, baseY);
    S32 v01x = sub_s16lo_s16lo(triHeader.y, triHeader.x);
    S32 v01y = sub_s16hi_s16hi(triHeader.y, triHeader.x);
    S32 v20x = sub_s16lo_s16lo(triHeader.x, triHeader.z);
    S32 v20y = sub_s16hi_s16hi(triHeader.x, triHeader.z);

    // compute per-edge coverage masks
    U32 c01 = coverMSAA_fast(SamplesLog2, v0x, v0y, v01x, v01y);
    U32 c12 = coverMSAA_fast(SamplesLog2, v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y);
    U32 c20 = coverMSAA_fast(SamplesLog2, v0x, v0y, v20x, v20y);

    // combine masks
    return c01 & c12 & c20;
}

//------------------------------------------------------------------------

__device__ __inline__ U32 scan32_value(U32 value, volatile U32* temp)
{
    temp[threadIdx.x + 16] = value;
    value += temp[threadIdx.x + 16 -  1], temp[threadIdx.x + 16] = value;
    value += temp[threadIdx.x + 16 -  2], temp[threadIdx.x + 16] = value;
    value += temp[threadIdx.x + 16 -  4], temp[threadIdx.x + 16] = value;
    value += temp[threadIdx.x + 16 -  8], temp[threadIdx.x + 16] = value;
    value += temp[threadIdx.x + 16 - 16], temp[threadIdx.x + 16] = value;
    return value;
}

__device__ __inline__ volatile const U32& scan32_total(volatile U32* temp)
{
    return temp[47];
}

//------------------------------------------------------------------------

template <class BlendShaderClass, U32 RenderModeFlags>
__device__ __inline__ U32 determineROPLaneMask(volatile U32& warpTemp) // mask of lanes that should process an earlier fragment than this lane
{
    bool reverseLanes = true;
    if ((RenderModeFlags & RenderModeFlag_EnableDepth) == 0)
    {
        BlendShaderClass bs;
        if (!bs.needsDst())
            reverseLanes = false;
    }

    U32 mask = (reverseLanes) ? (1u << threadIdx.x) : ~0u;
    do
    {
        warpTemp = threadIdx.x;
        mask ^= 1u << warpTemp;
    }
    while (warpTemp != threadIdx.x);
    return mask;
}

template <U32 RenderModeFlags>
__device__ __inline__ S32 findBit(U64 mask, int idx)
{
    U32 x = getLo(mask);
    int  pop = __popc(x);
    bool p   = (pop <= idx);
    if (p) x = getHi(mask);
    if (p) idx -= pop;
    int bit = p ? 32 : 0;

    pop = __popc(x & 0x0000ffffu);
    p   = (pop <= idx);
    if (p) x >>= 16;
    if (p) bit += 16;
    if (p) idx -= pop;

    if ((RenderModeFlags & RenderModeFlag_EnableQuads) == 0)
    {
        // Optimized variant.
        // Assumes that scanlines do not contain holes, and doesn't thus support quad rendering.
        // Counts scanlines LSB->MSB, but bits within them MSB->LSB.
        // 21 instructions.

        U32 tmp = x & 0x000000ffu;
        pop = __popc(tmp);
        p   = (pop <= idx);
        if (p) tmp = x & 0x0000ff00u;
        if (p) idx -= pop;

        return findLeadingOne(tmp) + bit - idx;
    }
    else
    {
        // Generic variant. Counts bits LSB->MSB.
        // 33 instructions.

        pop = __popc(x & 0x000000ffu);
        p   = (pop <= idx);
        if (p) x >>= 8;
        if (p) bit += 8;
        if (p) idx -= pop;

        pop = __popc(x & 0x0000000fu);
        p   = (pop <= idx);
        if (p) x >>= 4;
        if (p) bit += 4;
        if (p) idx -= pop;

        pop = __popc(x & 0x00000003u);
        p   = (pop <= idx);
        if (p) x >>= 2;
        if (p) bit += 2;
        if (p) idx -= pop;

        if (idx >= (x & 1))
            bit++;
        return bit;
    }
}

__device__ __inline__ U64 quadCoverage(U64 mask)
{
    mask |= mask >> 1;
    mask |= mask >> 8;
    return mask & 0x0055005500550055;
}

template <U32 RenderModeFlags>
__device__ __inline__ int numFragments(U64 coverage)
{
    if ((RenderModeFlags & RenderModeFlag_EnableQuads) == 0)
        return __popcll(coverage);
    else
        return __popcll(quadCoverage(coverage)) << 2;
}

template <U32 RenderModeFlags>
__device__ __inline__ int findFragment(U64 coverage, int fragIdx)
{
    if ((RenderModeFlags & RenderModeFlag_EnableQuads) == 0)
        return findBit<RenderModeFlags>(coverage, fragIdx);
    else
    {
        int t = findBit<RenderModeFlags>(quadCoverage(coverage), fragIdx >> 2);
        return t + (threadIdx.x & 1) + ((threadIdx.x & 2) << 2);
    }
}

//------------------------------------------------------------------------
// Single-sample implementation.
//------------------------------------------------------------------------

template <class BlendShaderClass, U32 RenderModeFlags>
__device__ __inline__ void executeROP_SingleSample(
    int triIdx, int pixelX, int pixelY,
    U32 color, U32 depth, volatile U32* pColor, volatile U32* pDepth,
	U32& timerTotal)
{
    BlendShaderClass bs;
    int rounds = 0;

    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
    {
		CR_TIMER_IN(FineROPConfResolve);
        do
        {
            rounds++;
			CR_TIMER_OUT_DEP(FineROPConfResolve, rounds);
			CR_TIMER_IN(FineROPBlend);
            *pDepth = depth;
			U32 sColor = *pColor;
			runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, pixelY, 0, color, sColor);
            if (bs.m_writeColor)
                *pColor = bs.m_color;
			CR_TIMER_OUT(FineROPBlend);
			CR_TIMER_IN(FineROPConfResolve);
        }
        while (depth < *pDepth);
		CR_TIMER_OUT(FineROPConfResolve);
    }
    else if (!bs.needsDst())
    {
        rounds++;
		CR_TIMER_IN(FineROPBlend);
        runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, pixelY, 0, color, 0);
        if (bs.m_writeColor)
            *pColor = bs.m_color;
		CR_TIMER_OUT(FineROPBlend);
    }
    else
    {
		CR_TIMER_IN(FineROPConfResolve);
        do
        {
            rounds++;
			CR_TIMER_OUT_DEP(FineROPConfResolve, rounds);
			CR_TIMER_IN(FineROPBlend);
            *pDepth = threadIdx.x;
			U32 sColor = *pColor;
			runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, pixelY, 0, color, sColor);
            if (bs.m_writeColor)
                *pColor = bs.m_color;
			CR_TIMER_OUT(FineROPBlend);
			CR_TIMER_IN(FineROPConfResolve);
        }
        while (*pDepth != threadIdx.x);
		CR_TIMER_OUT(FineROPConfResolve);
    }

    #if (CR_PROFILING_MODE == ProfilingMode_Counters)
        CR_COUNT(FineBlendRounds, 0, 1);
        for (int i = 0; __any(i < rounds); i++)
            CR_COUNT(FineBlendRounds, 1, 0);
    #endif
}

//------------------------------------------------------------------------

template <class VertexClass, class FragmentShaderClass, class BlendShaderClass, U32 RenderModeFlags>
__device__ __inline__ void fineRasterImpl_SingleSample(void)
{
                                                                            // for 20 warps:
    __shared__ volatile U64 s_cover8x8_lut[CR_COVER8X8_LUT_SIZE];           // 6KB
    __shared__ volatile U32 s_tileColor   [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_tileDepth   [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_triangleIdx [CR_FINE_MAX_WARPS][64];          // 5KB  original triangle index
    __shared__ volatile U32 s_triDataIdx  [CR_FINE_MAX_WARPS][64];          // 5KB  CRTriangleData index
    __shared__ volatile U64 s_triangleCov [CR_FINE_MAX_WARPS][64];          // 10KB coverage mask
    __shared__ volatile U32 s_triangleFrag[CR_FINE_MAX_WARPS][64];          // 5KB  fragment index
    __shared__ volatile U32 s_temp        [CR_FINE_MAX_WARPS][80];          // 6.25KB
                                                                            // = 47.25KB total

    const S32*      activeTiles     = (const S32*)c_crParams.activeTiles;
    const S32*      tileFirstSeg    = (const S32*)c_crParams.tileFirstSeg;

    volatile U32*   tileColor       = s_tileColor[threadIdx.y];
    volatile U32*   tileDepth       = s_tileDepth[threadIdx.y];
    volatile U32*   triangleIdx     = s_triangleIdx[threadIdx.y];
    volatile U32*   triDataIdx      = s_triDataIdx[threadIdx.y];
    volatile U64*   triangleCov     = s_triangleCov[threadIdx.y];
    volatile U32*   triangleFrag    = s_triangleFrag[threadIdx.y];
    volatile U32*   temp            = s_temp[threadIdx.y];

    if (g_crAtomics.numSubtris > c_crParams.maxSubtris || g_crAtomics.numBinSegs > c_crParams.maxBinSegs || g_crAtomics.numTileSegs > c_crParams.maxTileSegs)
        return;

    CR_TIMER_INIT();
	CR_TIMER_IN(FineTotal);

    U32 ropLaneMask = determineROPLaneMask<BlendShaderClass, RenderModeFlags>(temp[0]);
    temp[threadIdx.x] = 0; // first 16 elements of temp are always zero
    cover8x8_setupLUT(s_cover8x8_lut);
    __syncthreads();

    // loop over tiles
    for (;;)
    {
        CR_TIMER_IN(FinePickTile);

        // pick a tile
        if (threadIdx.x == 0)
            temp[16] = atomicAdd(&g_crAtomics.fineCounter, 1);
        int activeIdx = temp[16];
        if (activeIdx >= g_crAtomics.numActiveTiles)
        {
            CR_TIMER_OUT(FinePickTile);
            break;
        }

        int tileIdx = activeTiles[activeIdx];
        S32 segment = tileFirstSeg[tileIdx];
        int tileY = idiv_fast(tileIdx, c_crParams.widthTiles);
        int tileX = tileIdx - tileY * c_crParams.widthTiles;

        // initialize per-tile state
        int triRead = 0, triWrite = 0;
        int fragRead = 0, fragWrite = 0;
        triangleFrag[63] = 0; // "previous triangle"

        CR_TIMER_OUT_DEP(FinePickTile, triangleFrag[63]);
		CR_TIMER_IN(FineReadTile);

        // deferred clear => clear tile
        if (c_crParams.deferredClear)
        {
			tileColor[threadIdx.x] = c_crParams.clearColor;
            tileDepth[threadIdx.x] = c_crParams.clearDepth;
            tileColor[threadIdx.x + 32] = c_crParams.clearColor;
            tileDepth[threadIdx.x + 32] = c_crParams.clearDepth;
        }

        // otherwise => read tile from framebuffer
        else
        {
            int surfX = (tileX << (CR_TILE_LOG2 + 2)) + ((threadIdx.x & (CR_TILE_SIZE - 1)) << 2);
            int surfY = (tileY << CR_TILE_LOG2) + (threadIdx.x >> CR_TILE_LOG2);
			tileColor[threadIdx.x] = surf2Dread<U32>(s_colorBuffer, surfX, surfY);
            tileDepth[threadIdx.x] = surf2Dread<U32>(s_depthBuffer, surfX, surfY);
            tileColor[threadIdx.x + 32] = surf2Dread<U32>(s_colorBuffer, surfX, surfY + 4);
            tileDepth[threadIdx.x + 32] = surf2Dread<U32>(s_depthBuffer, surfX, surfY + 4);
        }

		CR_TIMER_OUT_DEP(FineReadTile, tileDepth[threadIdx.x + 32]);

        CR_TIMER_IN(FinePickTile);
        U32 tileZMax;
        bool tileZUpd;
        initTileZMax(tileZMax, tileZUpd, tileDepth);
        CR_TIMER_OUT_DEP(FinePickTile, tileZMax);

        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            if (fragWrite - fragRead < 32 && segment >= 0)
            {
                // update tile z
                CR_TIMER_IN(FineUpdateTileZ);
                updateTileZMax<RenderModeFlags>(tileZMax, tileZUpd, tileDepth, temp);
                CR_TIMER_OUT_DEP(FineUpdateTileZ, tileZMax);

                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    CR_TIMER_IN(FineReadTriangle);
                    S32 triIdx, dataIdx;
                    uint4 triHeader;
                    getTriangle(triIdx, dataIdx, triHeader, segment);
					CR_TIMER_OUT_DEP(FineReadTriangle, triHeader);

                    // early z cull
                    CR_TIMER_IN(FineEarlyZCull);
                    if (triIdx >= 0 && earlyZCull<RenderModeFlags>(triHeader, tileZMax))
                        triIdx = -1;
                    CR_TIMER_OUT_DEP(FineEarlyZCull, triIdx);

                    // determine coverage
                    CR_TIMER_IN(FinePixelCoverage);
                    U64 coverage = trianglePixelCoverage<0>(triHeader, tileX, tileY, s_cover8x8_lut);
                    S32 pop = (triIdx == -1) ? 0 : numFragments<RenderModeFlags>(coverage);
                    CR_COUNT(FineEmptyCull, (pop == 0 && triIdx != -1) ? 100 : 0, (triIdx == -1) ? 0 : 1);
                    CR_COUNT(FineFragPerTri, pop, (triIdx == -1) ? 0 : 1);
                    CR_TIMER_OUT_DEP(FinePixelCoverage, pop);

                    // fragment count scan
                    CR_TIMER_IN(FineFragmentScan);
                    U32 frag = scan32_value(pop, temp);
                    CR_TIMER_OUT_DEP(FineFragmentScan, frag);
                    CR_TIMER_IN(FineFragmentEnqueue);
                    frag += fragWrite; // frag now holds cumulative fragment count
                    fragWrite += scan32_total(temp);

                    // queue non-empty triangles
                    U32 goodMask = __ballot(pop != 0);
                    if (pop != 0)
                    {
                        int idx = (triWrite + __popc(goodMask & getLaneMaskLt())) & 63;
                        triangleIdx [idx] = triIdx;
                        triDataIdx  [idx] = dataIdx;
                        triangleFrag[idx] = frag;
                        triangleCov [idx] = coverage;
                    }
                    triWrite += __popc(goodMask);
					CR_TIMER_OUT_DEP(FineFragmentEnqueue, triWrite);
                }
                while (fragWrite - fragRead < 32 && segment >= 0);
            }

            // end of segment?
            if (fragRead == fragWrite)
                break;

            CR_TIMER_IN(FineFragmentDistr);

            // tag triangle boundaries
            temp[threadIdx.x + 16] = 0;
            if (triRead + threadIdx.x < triWrite)
            {
                int idx = triangleFrag[(triRead + threadIdx.x) & 63] - fragRead;
                if (idx <= 32)
                    temp[idx + 16 - 1] = 1;
            }

            int ropLaneIdx = __popc(ropLaneMask);
            U32 boundaryMask = __ballot(temp[ropLaneIdx + 16]);

            // distribute fragments
            CR_TIMER_OUT_DEP(FineFragmentDistr, boundaryMask);
            if (ropLaneIdx < fragWrite - fragRead)
            {
                int triBufIdx = (triRead + __popc(boundaryMask & ropLaneMask)) & 63;
                int fragIdx = add_sub(fragRead, ropLaneIdx, triangleFrag[(triBufIdx - 1) & 63]);
                CR_TIMER_IN(FineFindBit);
                U64 coverage = triangleCov[triBufIdx];
                int pixelInTile = findFragment<RenderModeFlags>(coverage, fragIdx);
				CR_TIMER_OUT_DEP(FineFindBit, pixelInTile);
                int triIdx = triangleIdx[triBufIdx];
                int dataIdx = triDataIdx[triBufIdx];

                // determine pixel position
                U32 pixelX = (tileX << CR_TILE_LOG2) + (pixelInTile & 7);
                U32 pixelY = (tileY << CR_TILE_LOG2) + (pixelInTile >> 3);

                CR_COUNT(SetupSamplesPerTri, (((U32)(coverage >> pixelInTile) & 1) != 0) ? 1 : 0, 0);

                // depth test
                U32 depth = 0;
                bool zkill = false;
                if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
                {
                    CR_TIMER_IN(FineReadZData);
                    uint4 zdata = tex1Dfetch(t_triData, dataIdx * 4);
                    CR_TIMER_OUT_DEP(FineReadZData, zdata);
					CR_TIMER_IN(FineZKill);
                    depth = zdata.x * pixelX + zdata.y * pixelY + zdata.z;
                    U32 oldDepth = tileDepth[pixelInTile];
                    if (depth >= oldDepth)
                        zkill = true;
                    else if (oldDepth == tileZMax)
                        tileZUpd = true; // we are replacing previous zmax => need to update
					CR_TIMER_OUT_DEP(FineZKill, tileZUpd);
                    CR_COUNT(FineZKill, zkill ? 100 : 0, 1);
                }

                if ((RenderModeFlags & RenderModeFlag_EnableQuads) != 0 || !zkill)
                {
                    // run fragment shader
                    CR_COUNT(FineWarpUtil, 100, singleLane() ? 32 : 0);
                    CR_TIMER_IN(FineShade);
                    FragmentShaderClass fragShader;
                    runFragmentShader<VertexClass, FragmentShaderClass, 0, RenderModeFlags>(
                        fragShader,
                        triIdx, dataIdx, pixelX, pixelY, 0x11, &temp[16]);
                    CR_TIMER_OUT(FineShade);

                    // run ROP
                    bool covered = (((U32)(coverage >> pixelInTile) & 1) != 0);
                    if (((RenderModeFlags & RenderModeFlag_EnableQuads) == 0 || (covered && !zkill)) && !fragShader.m_discard)
                    {
					    executeROP_SingleSample<BlendShaderClass, RenderModeFlags>(
                            triIdx, pixelX, pixelY, fragShader.m_color, depth,
                            &tileColor[pixelInTile], &tileDepth[pixelInTile],
							timerTotal);
                    }
                }
            }

            // update counters
            fragRead = ::min(fragRead + 32, fragWrite);
            triRead += __popc(boundaryMask);
        }

        CR_COUNT(FineFragPerTile, fragRead, 1);
        CR_COUNT(FineTriPerTile, 0, 1);

        // Write tile back to the framebuffer.

        CR_TIMER_IN(FineWriteTile);
        {
            int surfX = (tileX << (CR_TILE_LOG2 + 2)) + ((threadIdx.x & (CR_TILE_SIZE - 1)) << 2);
            int surfY = (tileY << CR_TILE_LOG2) + (threadIdx.x >> CR_TILE_LOG2);
            surf2Dwrite<U32>(tileColor[threadIdx.x], s_colorBuffer, surfX, surfY);
            surf2Dwrite<U32>(tileDepth[threadIdx.x], s_depthBuffer, surfX, surfY);
            surf2Dwrite<U32>(tileColor[threadIdx.x + 32], s_colorBuffer, surfX, surfY + 4);
            surf2Dwrite<U32>(tileDepth[threadIdx.x + 32], s_depthBuffer, surfX, surfY + 4);
        }
        CR_TIMER_OUT(FineWriteTile);
    }

    CR_TIMER_OUT(FineTotal);
    CR_TIMER_DEINIT();
}

//------------------------------------------------------------------------
// Multisample implementation.
//------------------------------------------------------------------------

template <class BlendShaderClass, U32 RenderModeFlags>
__device__ __inline__ U32 executeROP_MultiSample(
    int triIdx, int pixelX, int pixelY, int sampleIdx, int surfX,
    bool covered, U32 color, U32 depth, volatile U32* temp,
	U32& timerTotal)
{
    BlendShaderClass bs;
    int rounds = 0;
    U32 newDepth = 0;

    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
    {
		CR_TIMER_IN(FineROPRead);
        U32 oldDepth = surf2Dread<U32>(s_depthBuffer, surfX, pixelY);
        *temp = oldDepth;
		CR_TIMER_OUT_DEP(FineROPRead, oldDepth);
	    CR_TIMER_IN(FineROPConfResolve);
        if (covered && depth < oldDepth)
        {
            do
            {
                rounds++;
                *temp = depth;
			    CR_TIMER_OUT_DEP(FineROPConfResolve, rounds);
				CR_TIMER_IN(FineROPRead);
                U32 dst = surf2Dread<U32>(s_colorBuffer, surfX, pixelY);
				CR_TIMER_OUT_DEP(FineROPRead, dst);
				CR_TIMER_IN(FineROPBlend);
                runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, pixelY, sampleIdx, color, dst);
				CR_TIMER_OUT(FineROPBlend);
				CR_TIMER_IN(FineROPWrite);
                if (bs.m_writeColor)
                    surf2Dwrite<U32>(bs.m_color, s_colorBuffer, surfX, pixelY);
				CR_TIMER_OUT(FineROPWrite);
			    CR_TIMER_IN(FineROPConfResolve);
            }
            while (depth < *temp);
        }

	    CR_TIMER_OUT(FineROPConfResolve);
		CR_TIMER_IN(FineROPWrite);
        newDepth = *temp;
        if (newDepth != oldDepth)
            surf2Dwrite<U32>(newDepth, s_depthBuffer, surfX, pixelY);
		CR_TIMER_OUT(FineROPWrite);
    }
    else if (covered)
    {
        if (!bs.needsDst())
        {
            rounds++;
			CR_TIMER_IN(FineROPBlend);
            runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, sampleIdx, pixelY, color, 0);
			CR_TIMER_OUT(FineROPBlend);
			CR_TIMER_IN(FineROPWrite);
            if (bs.m_writeColor)
                surf2Dwrite<U32>(bs.m_color, s_colorBuffer, surfX, pixelY);
			CR_TIMER_OUT(FineROPWrite);
        }
        else
        {
			CR_TIMER_IN(FineROPConfResolve);
            do
            {
                rounds++;
                *temp = threadIdx.x;
				CR_TIMER_OUT_DEP(FineROPConfResolve, rounds);
				CR_TIMER_IN(FineROPRead);
                U32 dst = surf2Dread<U32>(s_colorBuffer, surfX, pixelY);
				CR_TIMER_OUT_DEP(FineROPRead, dst);
				CR_TIMER_IN(FineROPBlend);
                runBlendShader<BlendShaderClass>(bs, triIdx, pixelX, pixelY, sampleIdx, color, dst);
				CR_TIMER_OUT(FineROPBlend);
			    CR_TIMER_IN(FineROPWrite);
                if (bs.m_writeColor)
                    surf2Dwrite<U32>(bs.m_color, s_colorBuffer, surfX, pixelY);
			    CR_TIMER_OUT(FineROPWrite);
				CR_TIMER_IN(FineROPConfResolve);
            }
            while (*temp != threadIdx.x);
			CR_TIMER_OUT(FineROPConfResolve);
        }
    }

    #if (CR_PROFILING_MODE == ProfilingMode_Counters)
        if (__any(rounds != 0))
        {
            CR_COUNT(FineBlendRounds, 1, 1);
            for (int i = 1; __any(i < rounds); i++)
                CR_COUNT(FineBlendRounds, 1, 0);
        }
    #endif
    return newDepth;
}

//------------------------------------------------------------------------

template <class VertexClass, class FragmentShaderClass, class BlendShaderClass, int SamplesLog2, U32 RenderModeFlags>
__device__ __inline__ void fineRasterImpl_MultiSample(void)
{
                                                                            // for 20 warps:
    __shared__ volatile U64 s_cover8x8_lut[CR_COVER8X8_LUT_SIZE];           // 6KB
    __shared__ volatile U8  s_centroid_lut[1 << (1 << SamplesLog2)];        // 0.25KB
    __shared__ volatile U32 s_tileDepth   [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_triangleIdx [CR_FINE_MAX_WARPS][64];          // 5KB  original triangle index
    __shared__ volatile U32 s_triDataIdx  [CR_FINE_MAX_WARPS][64];          // 5KB  CRTriangleData index
    __shared__ volatile U64 s_triangleCov [CR_FINE_MAX_WARPS][64];          // 10KB coverage mask
    __shared__ volatile U32 s_triangleFrag[CR_FINE_MAX_WARPS][64];          // 5KB  fragment index
    __shared__ volatile U32 s_temp        [CR_FINE_MAX_WARPS][80];          // 6.25KB
                                                                            // = 42.5KB total

    const S32*      activeTiles     = (const S32*)c_crParams.activeTiles;
    const S32*      tileFirstSeg    = (const S32*)c_crParams.tileFirstSeg;

    volatile U32*   tileDepth       = s_tileDepth[threadIdx.y];
    volatile U32*   triangleIdx     = s_triangleIdx[threadIdx.y];
    volatile U32*   triDataIdx      = s_triDataIdx[threadIdx.y];
    volatile U64*   triangleCov     = s_triangleCov[threadIdx.y];
    volatile U32*   triangleFrag    = s_triangleFrag[threadIdx.y];
    volatile U32*   temp            = s_temp[threadIdx.y];

    if (g_crAtomics.numSubtris > c_crParams.maxSubtris || g_crAtomics.numBinSegs > c_crParams.maxBinSegs || g_crAtomics.numTileSegs > c_crParams.maxTileSegs)
        return;

    CR_TIMER_INIT();
    CR_TIMER_IN(FineTotal);

    U32 ropLaneMask = determineROPLaneMask<BlendShaderClass, RenderModeFlags>(temp[0]);
    temp[threadIdx.x] = 0; // first 16 elements of temp are always zero
    cover8x8_setupLUT(s_cover8x8_lut);
    setupCentroidLUT<SamplesLog2>(s_centroid_lut);
    __syncthreads();

    // loop over tiles
    for (;;)
    {
        CR_TIMER_IN(FinePickTile);

        // pick a tile
        if (threadIdx.x == 0)
            temp[16] = atomicAdd(&g_crAtomics.fineCounter, 1);
        int activeIdx = temp[16];
        if (activeIdx >= g_crAtomics.numActiveTiles)
        {
            CR_TIMER_OUT(FinePickTile);
            break;
        }

        int tileIdx = activeTiles[activeIdx];
        S32 segment = tileFirstSeg[tileIdx];
        int tileY = idiv_fast(tileIdx, c_crParams.widthTiles);
        int tileX = tileIdx - tileY * c_crParams.widthTiles;
        int tileSurfX = tileX << (CR_TILE_LOG2 + SamplesLog2 + 2);

        // initialize per-tile state
        int triRead = 0, triWrite = 0;
        int fragRead = 0, fragWrite = 0;
        triangleFrag[63] = 0; // "previous triangle"

        // deferred clear => clear tile
        if (c_crParams.deferredClear)
        {
            int surfX = (tileX << (CR_TILE_LOG2 + (SamplesLog2 + 2))) + ((threadIdx.x & (CR_TILE_SIZE - 1)) << 2);
            int surfY = (tileY << CR_TILE_LOG2) + (threadIdx.x >> CR_TILE_LOG2);
            for (int i = 0; i < (1 << SamplesLog2); i++)
            {
                surf2Dwrite<U32>(c_crParams.clearColor, s_colorBuffer, surfX, surfY);
                surf2Dwrite<U32>(c_crParams.clearDepth, s_depthBuffer, surfX, surfY);
                surf2Dwrite<U32>(c_crParams.clearColor, s_colorBuffer, surfX, surfY + 4);
                surf2Dwrite<U32>(c_crParams.clearDepth, s_depthBuffer, surfX, surfY + 4);
                surfX += CR_TILE_SIZE << 2;
            }
        }

        U32 tileZMax = CR_DEPTH_MAX;
        bool tileZUpd = false;
        tileDepth[threadIdx.x] = CR_DEPTH_MAX;
        tileDepth[threadIdx.x + 32] = CR_DEPTH_MAX;

        CR_TIMER_OUT_DEP(FinePickTile, tileDepth[threadIdx.x + 32]);

        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            if (fragWrite - fragRead < 32 && segment >= 0)
            {
                // update tile z
                CR_TIMER_IN(FineUpdateTileZ);
                updateTileZMax<RenderModeFlags>(tileZMax, tileZUpd, tileDepth, temp);
                CR_TIMER_OUT_DEP(FineUpdateTileZ, tileZMax);

                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    CR_TIMER_IN(FineReadTriangle);
                    S32 triIdx, dataIdx;
                    uint4 triHeader;
                    getTriangle(triIdx, dataIdx, triHeader, segment);
					CR_TIMER_OUT_DEP(FineReadTriangle, triHeader);

                    // early z cull
                    CR_TIMER_IN(FineEarlyZCull);
                    if (triIdx >= 0 && earlyZCull<RenderModeFlags>(triHeader, tileZMax))
                        triIdx = -1;
                    CR_TIMER_OUT_DEP(FineEarlyZCull, triIdx);

                    // determine pixel coverage
                    CR_TIMER_IN(FinePixelCoverage);
                    U64 coverage = trianglePixelCoverage<SamplesLog2>(triHeader, tileX, tileY, s_cover8x8_lut);
                    S32 pop = (triIdx == -1) ? 0 : numFragments<RenderModeFlags>(coverage);
                    CR_COUNT(FineEmptyCull, (pop == 0 && triIdx != -1) ? 100 : 0, (triIdx == -1) ? 0 : 1);
                    CR_COUNT(FineFragPerTri, pop, (triIdx == -1) ? 0 : 1);
                    CR_TIMER_OUT_DEP(FinePixelCoverage, pop);

                    // fragment count scan
                    CR_TIMER_IN(FineFragmentScan);
                    U32 frag = scan32_value(pop, temp);
                    CR_TIMER_OUT_DEP(FineFragmentScan, frag);
                    CR_TIMER_IN(FineFragmentEnqueue);
                    frag += fragWrite; // frag now holds cumulative fragment count
                    fragWrite += scan32_total(temp);

                    // queue non-empty triangles
                    U32 goodMask = __ballot(pop != 0);
                    if (pop != 0)
                    {
                        int idx = (triWrite + __popc(goodMask & getLaneMaskLt())) & 63;
                        triangleIdx [idx] = triIdx;
                        triDataIdx  [idx] = dataIdx;
                        triangleFrag[idx] = frag;
                        triangleCov [idx] = coverage;
                    }
                    triWrite += __popc(goodMask);
                    CR_TIMER_OUT_DEP(FineFragmentEnqueue, triWrite);
                }
                while (fragWrite - fragRead < 32 && segment >= 0);
            }

            // end of segment?
            if (fragRead == fragWrite)
                break;

            CR_TIMER_IN(FineFragmentDistr);

			// tag triangle boundaries
            temp[threadIdx.x + 16] = 0;
            if (triRead + threadIdx.x < triWrite)
            {
                int idx = triangleFrag[(triRead + threadIdx.x) & 63] - fragRead;
                if (idx <= 32)
                    temp[idx + 16 - 1] = 1;
            }

            int ropLaneIdx = __popc(ropLaneMask);
            U32 boundaryMask = __ballot(temp[ropLaneIdx + 16]);

            // distribute fragments
            CR_TIMER_OUT_DEP(FineFragmentDistr, boundaryMask);
            if (ropLaneIdx < fragWrite - fragRead)
            {
                int triBufIdx = (triRead + __popc(boundaryMask & ropLaneMask)) & 63;
                int fragIdx = add_sub(fragRead, ropLaneIdx, triangleFrag[(triBufIdx - 1) & 63]);
                CR_TIMER_IN(FineFindBit);
                U64 coverage = triangleCov[triBufIdx];
                int pixelInTile = findFragment<RenderModeFlags>(coverage, fragIdx);
                CR_TIMER_OUT_DEP(FineFindBit, pixelInTile);
                int triIdx = triangleIdx[triBufIdx];
                int dataIdx = triDataIdx[triBufIdx];

                // determine pixel position
                U32 pixelX = (tileX << CR_TILE_LOG2) + (pixelInTile & 7);
                U32 pixelY = (tileY << CR_TILE_LOG2) + (pixelInTile >> 3);

                CR_COUNT(SetupSamplesPerTri, __popc(triangleSampleCoverage<SamplesLog2>(tex1Dfetch(t_triHeader, dataIdx), pixelX, pixelY)), 0);

                // conservative depth test
                uint4   zdata   = make_uint4(0, 0, 0, 0);
                U32     oldZMax = 0;
                U32     zbase   = 0;
                bool    zkill   = false;

                if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
                {
                    CR_TIMER_IN(FineReadZData);
                    zdata = tex1Dfetch(t_triData, dataIdx * 4);
                    CR_TIMER_OUT_DEP(FineReadZData, zdata);
                    CR_TIMER_IN(FineZKill);
                    oldZMax = tileDepth[pixelInTile];
                    zbase = ((zdata.x * pixelX + zdata.y * pixelY) << SamplesLog2) + zdata.z;
                    U32 zmin = ((zdata.x + zdata.y) << max(SamplesLog2 - 1, 0)) + zbase - zdata.w;
                    zkill = (zmin >= oldZMax && zmin < zmin + zdata.w * 2);
                    CR_TIMER_OUT_DEP(FineZKill, zkill);
                }

                // determine sample coverage
                U32 sampleMask = 0;
                if ((RenderModeFlags & RenderModeFlag_EnableQuads) != 0 || !zkill)
                {
                    CR_TIMER_IN(FineSampleCoverage);
                    uint4 triHeader = tex1Dfetch(t_triHeader, dataIdx);
                    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
                    {
                        U32 zmin = triHeader.w & 0xFFFFF000;
                        zkill = (zmin >= oldZMax); // can happen if zslope is very high
                    }
                    if (!zkill)
                        sampleMask = triangleSampleCoverage<SamplesLog2>(triHeader, pixelX, pixelY);
                    CR_TIMER_OUT_DEP(FineSampleCoverage, sampleMask);
                }

                CR_COUNT(FineZKill, (zkill) ? 100 : 0, 1);
                CR_COUNT(FineMSAAKill, (sampleMask == 0 && !zkill) ? 100 : 0, 1);

                // execute shader and rop
                if ((RenderModeFlags & RenderModeFlag_EnableQuads) != 0 || sampleMask != 0)
                {
                    // run fragment shader
                    CR_COUNT(FineWarpUtil, 100, singleLane() ? 32 : 0);
                    CR_TIMER_IN(FineShade);
                    FragmentShaderClass fragShader;
                    runFragmentShader<VertexClass, FragmentShaderClass, SamplesLog2, RenderModeFlags>(
                        fragShader,
                        triIdx, dataIdx, pixelX, pixelY, s_centroid_lut[sampleMask], &temp[16]);
                    CR_TIMER_OUT(FineShade);

					// evaluate depth and execute ROP
                    if (((RenderModeFlags & RenderModeFlag_EnableQuads) == 0 || (!zkill && sampleMask != 0)) && !fragShader.m_discard)
                    {
					    U32 newZMax = 0;
                        int surfX = tileSurfX + ((pixelInTile & 7) << 2);

					    for (int i = 0; i < (1 << SamplesLog2); i++)
					    {
						    newZMax = ::max(newZMax, executeROP_MultiSample<BlendShaderClass, RenderModeFlags>(
                                triIdx, pixelX, pixelY, i, surfX,
							    ((sampleMask & (1 << i)) != 0),
							    fragShader.m_color,
							    zdata.x * c_msaaPatterns[SamplesLog2][i] + zdata.y * i + zbase,
							    &temp[pixelInTile + 16],
								timerTotal));

                            surfX += 1 << (CR_TILE_LOG2 + 2);
					    }

					    if (newZMax < oldZMax)
					    {
							CR_TIMER_IN(FineROPWrite);
						    tileDepth[pixelInTile] = newZMax;
							CR_TIMER_OUT(FineROPWrite);
						    if (oldZMax == tileZMax)
							    tileZUpd = true;
					    }
                    }
                }
            }

            // update counters
            fragRead = ::min(fragRead + 32, fragWrite);
            triRead += __popc(boundaryMask);
        }

        CR_COUNT(FineFragPerTile, fragRead, 1);
        CR_COUNT(FineTriPerTile, 0, 1);
    }

    CR_TIMER_OUT(FineTotal);
    CR_TIMER_DEINIT();
}

//------------------------------------------------------------------------
