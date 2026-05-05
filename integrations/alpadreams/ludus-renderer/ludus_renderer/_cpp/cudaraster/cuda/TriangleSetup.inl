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

__device__ __inline__ void snapTriangle(
    float4 v0, float4 v1, float4 v2,
    int2& p0, int2& p1, int2& p2, float3& rcpW, int2& lo, int2& hi)
{
    F32 viewScaleX = (F32)(c_crParams.viewportWidth  << (CR_SUBPIXEL_LOG2 - 1));
    F32 viewScaleY = (F32)(c_crParams.viewportHeight << (CR_SUBPIXEL_LOG2 - 1));
    rcpW = make_float3(1.0f / v0.w, 1.0f / v1.w, 1.0f / v2.w);
    p0 = make_int2(f32_to_s32_sat(v0.x * rcpW.x * viewScaleX), f32_to_s32_sat(v0.y * rcpW.x * viewScaleY));
    p1 = make_int2(f32_to_s32_sat(v1.x * rcpW.y * viewScaleX), f32_to_s32_sat(v1.y * rcpW.y * viewScaleY));
    p2 = make_int2(f32_to_s32_sat(v2.x * rcpW.z * viewScaleX), f32_to_s32_sat(v2.y * rcpW.z * viewScaleY));
    lo = make_int2(min_min(p0.x, p1.x, p2.x), min_min(p0.y, p1.y, p2.y));
    hi = make_int2(max_max(p0.x, p1.x, p2.x), max_max(p0.y, p1.y, p2.y));
}

//------------------------------------------------------------------------
// 0 = Visible.
// 1 = Backfacing.
// 2 = Between pixels.

template <int SamplesLog2>
__device__ __inline__ int prepareTriangle(
    int2 p0, int2 p1, int2 p2, int2 lo, int2 hi,
    int2& d1, int2& d2, S32& area)
{
    // Backfacing or degenerate => cull.

    d1 = make_int2(p1.x - p0.x, p1.y - p0.y);
    d2 = make_int2(p2.x - p0.x, p2.y - p0.y);
    area = d1.x * d2.y - d1.y * d2.x;

    if (area <= 0)
        return 1; // Backfacing.

    // AABB falls between samples => cull.

    int sampleSize = 1 << (CR_SUBPIXEL_LOG2 - SamplesLog2);
    int biasX = (c_crParams.viewportWidth  << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int biasY = (c_crParams.viewportHeight << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int lox = (int)add_add(lo.x, sampleSize - 1, biasX) & -sampleSize;
    int loy = (int)add_add(lo.y, sampleSize - 1, biasY) & -sampleSize;
    int hix = (hi.x + biasX) & -sampleSize;
    int hiy = (hi.y + biasY) & -sampleSize;

    if (lox > hix || loy > hiy)
        return 2; // Between pixels.

    // AABB covers 1 or 2 samples => cull if they are not covered.

    int diff = add_sub(hix, hiy, lox) - loy;
    if (diff <= sampleSize)
    {
        int2 t0 = make_int2(add_sub(p0.x, biasX, lox), add_sub(p0.y, biasY, loy));
        int2 t1 = make_int2(add_sub(p1.x, biasX, lox), add_sub(p1.y, biasY, loy));
        int2 t2 = make_int2(add_sub(p2.x, biasX, lox), add_sub(p2.y, biasY, loy));
        S32 e0 = t0.x * t1.y - t0.y * t1.x;
        S32 e1 = t1.x * t2.y - t1.y * t2.x;
        S32 e2 = t2.x * t0.y - t2.y * t0.x;

        if (e0 < 0 || e1 < 0 || e2 < 0)
        {
            if (diff == 0)
                return 2; // Between pixels.

            t0 = make_int2(add_sub(p0.x, biasX, hix), add_sub(p0.y, biasY, hiy));
            t1 = make_int2(add_sub(p1.x, biasX, hix), add_sub(p1.y, biasY, hiy));
            t2 = make_int2(add_sub(p2.x, biasX, hix), add_sub(p2.y, biasY, hiy));
            e0 = t0.x * t1.y - t0.y * t1.x;
            e1 = t1.x * t2.y - t1.y * t2.x;
            e2 = t2.x * t0.y - t2.y * t0.x;

            if (e0 < 0 || e1 < 0 || e2 < 0)
                return 2; // Between pixels.
        }
    }

    // Otherwise => proceed to output the triangle.

    return 0; // Visible.
}

//------------------------------------------------------------------------

template <int SamplesLog2, U32 RenderModeFlags>
__device__ __inline__ void setupTriangle(
    CRTriangleHeader* th, CRTriangleData* td, int3 vidx,
    float4 v0, float4 v1, float4 v2,
    float2 b0, float2 b1, float2 b2,
    int2 p0, int2 p1, int2 p2, float3 rcpW,
    int2 d1, int2 d2, S32 area,
    U32& timerTotal)
{
    CR_TIMER_IN(SetupPleq);
    U32 dep = 0;

    F32 areaRcp;
    int2 wv0;

    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0 ||
        (RenderModeFlags & RenderModeFlag_EnableLerp) != 0)
    {
        areaRcp = 1.0f / (F32)area;
        wv0.x = p0.x + (c_crParams.viewportWidth  << (CR_SUBPIXEL_LOG2 - 1));
        wv0.y = p0.y + (c_crParams.viewportHeight << (CR_SUBPIXEL_LOG2 - 1));
    }

    // Setup depth plane equation.

    uint3 zpleq;
    U32 zmin = 0, zslope = 0;
    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
    {
        F32 zcoef = (F32)(CR_DEPTH_MAX - CR_DEPTH_MIN) * 0.5f;
        F32 zbias = (F32)(CR_DEPTH_MAX + CR_DEPTH_MIN) * 0.5f;
        float3 zvert;
        zvert.x = (v0.z * zcoef) * rcpW.x + zbias;
        zvert.y = (v1.z * zcoef) * rcpW.y + zbias;
        zvert.z = (v2.z * zcoef) * rcpW.z + zbias;

        int2 zv0;
        zv0.x = wv0.x - (1 << (CR_SUBPIXEL_LOG2 - SamplesLog2 - 1));
        zv0.y = wv0.y - (1 << (CR_SUBPIXEL_LOG2 - SamplesLog2 - 1));
        zpleq = setupPleq(zvert, zv0, d1, d2, areaRcp, SamplesLog2);

        zmin = f32_to_u32_sat(fminf(fminf(zvert.x, zvert.y), zvert.z) - (F32)CR_LERP_ERROR(SamplesLog2));
        if (SamplesLog2 != 0)
        {
            U32 tmp = ::abs((S32)zpleq.x) + ::abs(::max((S32)zpleq.y, -FW_S32_MAX));
            zslope = tmp << max(SamplesLog2 - 1, 0);
            if ((zslope >> max(SamplesLog2 - 1, 0)) != tmp)
                zslope = FW_U32_MAX;
        }

        dep += zpleq.x + zpleq.y + zpleq.z + zmin + zslope;
    }

    // Setup lerp plane equations.

    uint3 wpleq, upleq, vpleq;
    if ((RenderModeFlags & RenderModeFlag_EnableLerp) != 0)
    {
        F32 wcoef = fminf(fminf(v0.w, v1.w), v2.w) * (F32)CR_BARY_MAX;
        float3 wvert = make_float3(wcoef * rcpW.x, wcoef * rcpW.y, wcoef * rcpW.z);
        float3 uvert = make_float3(b0.x * wvert.x, b1.x * wvert.y, b2.x * wvert.z);
        float3 vvert = make_float3(b0.y * wvert.x, b1.y * wvert.y, b2.y * wvert.z);

        wpleq = setupPleq(wvert, wv0, d1, d2, areaRcp, SamplesLog2 + 1);
        upleq = setupPleq(uvert, wv0, d1, d2, areaRcp, SamplesLog2 + 1);
        vpleq = setupPleq(vvert, wv0, d1, d2, areaRcp, SamplesLog2 + 1);
        dep += wpleq.x + wpleq.y + wpleq.z + upleq.x + upleq.y + upleq.z;
    }

    CR_TIMER_OUT_DEP(SetupPleq, dep);

    // Write CRTriangleData.

    CR_TIMER_IN(SetupTriDataWrite);

    if ((RenderModeFlags & RenderModeFlag_EnableDepth) != 0)
        *(uint4*)&td->zx = make_uint4(zpleq.x, zpleq.y, zpleq.z, zslope);

    if ((RenderModeFlags & RenderModeFlag_EnableLerp) == 0)
        *(uint4*)&td->vb = make_uint4(0, vidx.x, vidx.y, vidx.z);
    else
    {
        *(uint4*)&td->wx = make_uint4(wpleq.x, wpleq.y, wpleq.z, upleq.x);
        *(uint4*)&td->uy = make_uint4(upleq.y, upleq.z, vpleq.x, vpleq.y);
        *(uint4*)&td->vb = make_uint4(vpleq.z, vidx.x, vidx.y, vidx.z);
    }

    CR_TIMER_OUT(SetupTriDataWrite);

    // Determine flipbits.

    CR_TIMER_IN(SetupTriHeaderWrite);

    U32 f01 = cover8x8_selectFlips(d1.x, d1.y);
    U32 f12 = cover8x8_selectFlips(d2.x - d1.x, d2.y - d1.y);
    U32 f20 = cover8x8_selectFlips(-d2.x, -d2.y);

    // Write CRTriangleHeader.

    *(uint4*)th = make_uint4(
        prmt(p0.x, p0.y, 0x5410),
        prmt(p1.x, p1.y, 0x5410),
        prmt(p2.x, p2.y, 0x5410),
        (zmin & 0xfffff000u) | (f01 << 6) | (f12 << 2) | (f20 >> 2));

    CR_TIMER_OUT(SetupTriHeaderWrite);
}

//------------------------------------------------------------------------

template <class VertexClass, int SamplesLog2, U32 RenderModeFlags>
__device__ __inline__ void triangleSetupImpl(void)
{
    __shared__ F32 s_bary[CR_SETUP_WARPS * 32][18];
    F32* bary = s_bary[threadIdx.x + threadIdx.y * 32];

    const int3*         indexBuffer = (const int3*)c_crParams.indexBuffer;
    U8*                 triSubtris  = (U8*)c_crParams.triSubtris;
    CRTriangleHeader*   triHeader   = (CRTriangleHeader*)c_crParams.triHeader;
    CRTriangleData*     triData     = (CRTriangleData*)c_crParams.triData;

    int2 p0, p1, p2, lo, hi, d1, d2;
    float3 rcpW;
    S32 area;

    // Pick a task.

    int taskIdx = threadIdx.x + 32 * (threadIdx.y + CR_SETUP_WARPS * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= c_crParams.numTris)
        return;

    // Read vertices.

    CR_TIMER_INIT();
    CR_TIMER_IN(SetupTotal);
    CR_TIMER_IN(SetupVertexRead);

    int3 vidx = indexBuffer[taskIdx];
    int stride = sizeof(VertexClass) / sizeof(Vec4f);
    float4 v0 = tex1Dfetch(t_vertexBuffer, vidx.x * stride);
    float4 v1 = tex1Dfetch(t_vertexBuffer, vidx.y * stride);
    float4 v2 = tex1Dfetch(t_vertexBuffer, vidx.z * stride);

    CR_TIMER_OUT_DEP(SetupVertexRead, v0.x + v1.x + v2.x);
    CR_TIMER_IN(SetupCullSnap);

    CR_COUNT_LARGE_GRID(SetupViewportCull, 0, 1);
    CR_COUNT_LARGE_GRID(SetupBackfaceCull, 0, 1);
    CR_COUNT_LARGE_GRID(SetupBetweenPixelsCull, 0, 1);
    CR_COUNT_LARGE_GRID(SetupClipped, 0, 1);

    // Outside view frustum => cull.

    if (v0.w < fabsf(v0.x) | v0.w < fabsf(v0.y) | v0.w < fabsf(v0.z))
    {
        if ((v0.w < +v0.x & v1.w < +v1.x & v2.w < +v2.x) |
            (v0.w < -v0.x & v1.w < -v1.x & v2.w < -v2.x) |
            (v0.w < +v0.y & v1.w < +v1.y & v2.w < +v2.y) |
            (v0.w < -v0.y & v1.w < -v1.y & v2.w < -v2.y) |
            (v0.w < +v0.z & v1.w < +v1.z & v2.w < +v2.z) |
            (v0.w < -v0.z & v1.w < -v1.z & v2.w < -v2.z))
        {
            CR_COUNT_LARGE_GRID(SetupViewportCull, 100, 0);

            CR_TIMER_OUT(SetupCullSnap);
            CR_TIMER_IN(SetupNumSubWrite);
            triSubtris[taskIdx] = 0;
            CR_TIMER_OUT(SetupNumSubWrite);
            CR_TIMER_OUT(SetupTotal);
            CR_TIMER_DEINIT_LARGE_GRID();
            return;
        }
    }

    // Inside depth range => try to snap vertices.

    if (v0.w >= fabsf(v0.z) & v1.w >= fabsf(v1.z) & v2.w >= fabsf(v2.z))
    {
        // Inside S16 range and small enough => fast path.
        // Note: aabbLimit comes from the fact that cover8x8
        // does not support guardband with maximal viewport.

        snapTriangle(v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        S32 loxy = ::min(lo.x, lo.y);
        S32 hixy = ::max(hi.x, hi.y);
        S32 aabbLimit = (1 << (CR_MAXVIEWPORT_LOG2 + CR_SUBPIXEL_LOG2)) - 1;

        if (loxy >= -32768 && hixy <= 32767 && hixy - loxy <= aabbLimit)
        {
            int res = prepareTriangle<SamplesLog2>(p0, p1, p2, lo, hi, d1, d2, area);
            CR_TIMER_OUT_DEP(SetupCullSnap, res);
            CR_TIMER_IN(SetupNumSubWrite);
            triSubtris[taskIdx] = (res == 0) ? 1 : 0;
            CR_TIMER_OUT(SetupNumSubWrite);

            CR_COUNT_LARGE_GRID(SetupBackfaceCull, (res == 1) ? 100 : 0, 0);
            CR_COUNT_LARGE_GRID(SetupBetweenPixelsCull, (res == 2) ? 100 : 0, 0);

            if (res == 0)
                setupTriangle<SamplesLog2, RenderModeFlags>(
                    &triHeader[taskIdx], &triData[taskIdx], vidx,
                    v0, v1, v2,
                    make_float2(0.0f, 0.0f),
                    make_float2(1.0f, 0.0f),
                    make_float2(0.0f, 1.0f),
                    p0, p1, p2, rcpW,
                    d1, d2, area,
                    timerTotal);

            CR_TIMER_OUT(SetupTotal);
            CR_TIMER_DEINIT_LARGE_GRID();
            return;
        }
    }

    CR_TIMER_OUT(SetupCullSnap);

    // Clip to view frustum.

    CR_TIMER_IN(SetupClip);
    CR_COUNT_LARGE_GRID(SetupClipped, 100, 0);

    float4 ov0 = v0;
    float4 od1 = make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, v1.w - v0.w);
    float4 od2 = make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, v2.w - v0.w);
    int numVerts = clipTriangleWithFrustum(bary, &ov0.x, &v1.x, &v2.x, &od1.x, &od2.x);

    // Count non-culled subtriangles.

    v0.x = ov0.x + od1.x * bary[0] + od2.x * bary[1];
    v0.y = ov0.y + od1.y * bary[0] + od2.y * bary[1];
    v0.z = ov0.z + od1.z * bary[0] + od2.z * bary[1];
    v0.w = ov0.w + od1.w * bary[0] + od2.w * bary[1];
    v1.x = ov0.x + od1.x * bary[2] + od2.x * bary[3];
    v1.y = ov0.y + od1.y * bary[2] + od2.y * bary[3];
    v1.z = ov0.z + od1.z * bary[2] + od2.z * bary[3];
    v1.w = ov0.w + od1.w * bary[2] + od2.w * bary[3];
    float4 tv1 = v1;

    int numSubtris = 0;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        snapTriangle(v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle<SamplesLog2>(p0, p1, p2, lo, hi, d1, d2, area) == 0)
            numSubtris++;

        v1 = v2;
    }

    CR_TIMER_OUT(SetupClip);
    CR_TIMER_IN(SetupNumSubWrite);
    triSubtris[taskIdx] = numSubtris;
    CR_TIMER_OUT(SetupNumSubWrite);

    // Multiple subtriangles => allocate.

    CR_TIMER_IN(SetupAllocSub);
    int subtriBase = taskIdx;
    if (numSubtris > 1)
    {
        subtriBase = atomicAdd(&g_crAtomics.numSubtris, numSubtris);
        triHeader[taskIdx].misc = subtriBase;
        if (subtriBase + numSubtris > c_crParams.maxSubtris)
            numVerts = 0;
    }
    CR_TIMER_OUT_DEP(SetupAllocSub, subtriBase);

    // Setup subtriangles.

    CR_TIMER_IN(SetupCullSnap);
    v1 = tv1;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        snapTriangle(v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle<SamplesLog2>(p0, p1, p2, lo, hi, d1, d2, area) == 0)
        {
            CR_TIMER_OUT(SetupCullSnap);

            setupTriangle<SamplesLog2, RenderModeFlags>(
                &triHeader[subtriBase], &triData[subtriBase], vidx,
                v0, v1, v2,
                make_float2(bary[0], bary[1]),
                make_float2(bary[i * 2 - 2], bary[i * 2 - 1]),
                make_float2(bary[i * 2 + 0], bary[i * 2 + 1]),
                p0, p1, p2, rcpW,
                d1, d2, area,
                timerTotal);

            subtriBase++;
            CR_TIMER_IN(SetupCullSnap);
        }

        v1 = v2;
    }

    CR_TIMER_OUT(SetupCullSnap);
    CR_TIMER_OUT(SetupTotal);
    CR_TIMER_DEINIT_LARGE_GRID();
}

//------------------------------------------------------------------------
