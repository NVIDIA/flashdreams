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

__device__ __inline__ int globalTileIdx(int tileInBin)
{
    int tileX = tileInBin & (CR_BIN_SIZE - 1);
    int tileY = tileInBin >> CR_BIN_LOG2;
    return tileX + tileY * c_crParams.widthTiles;
}

//------------------------------------------------------------------------

__device__ __inline__ void coarseRasterImpl(void)
{
    // Common.

    __shared__ volatile U32 s_workCounter;
    __shared__ volatile U32 s_scanTemp          [CR_COARSE_WARPS][48];              // 3KB

    // Input.

    __shared__ volatile U32 s_binOrder          [CR_MAXBINS_SQR];                   // 1KB
    __shared__ volatile S32 s_binStreamCurrSeg  [CR_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile S32 s_binStreamFirstTri [CR_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile S32 s_triQueue          [CR_COARSE_QUEUE_SIZE];             // 4KB
    __shared__ volatile S32 s_triQueueWritePos;
    __shared__ volatile U32 s_binStreamSelectedOfs;
    __shared__ volatile U32 s_binStreamSelectedSize;

    // Output.

    __shared__ volatile U32 s_warpEmitMask      [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    __shared__ volatile U32 s_warpEmitPrefixSum [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    __shared__ volatile U32 s_tileEmitPrefixSum [CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    __shared__ volatile U32 s_tileAllocPrefixSum[CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    __shared__ volatile S32 s_tileStreamCurrOfs [CR_BIN_SQR];                       // 1KB
    __shared__ volatile U32 s_firstAllocSeg;
    __shared__ volatile U32 s_firstActiveIdx;

    // Pointers and constants.

    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)c_crParams.triHeader;
    const S32*              binFirstSeg     = (const S32*)c_crParams.binFirstSeg;
    const S32*              binTotal        = (const S32*)c_crParams.binTotal;
    const S32*              binSegData      = (const S32*)c_crParams.binSegData;
    const S32*              binSegNext      = (const S32*)c_crParams.binSegNext;
    const S32*              binSegCount     = (const S32*)c_crParams.binSegCount;
    S32*                    activeTiles     = (S32*)c_crParams.activeTiles;
    S32*                    tileFirstSeg    = (S32*)c_crParams.tileFirstSeg;
    S32*                    tileSegData     = (S32*)c_crParams.tileSegData;
    S32*                    tileSegNext     = (S32*)c_crParams.tileSegNext;
    S32*                    tileSegCount    = (S32*)c_crParams.tileSegCount;

    int tileLog     = CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
    int thrInBlock  = threadIdx.x + threadIdx.y * 32;
    int emitShift   = CR_BIN_LOG2 * 2 + 5; // We scan ((numEmits << emitShift) | numAllocs) over tiles.

    if (g_crAtomics.numSubtris > c_crParams.maxSubtris || g_crAtomics.numBinSegs > c_crParams.maxBinSegs)
        return;

    CR_TIMER_INIT();
    CR_TIMER_IN(CoarseTotal);

    // Initialize sharedmem arrays.

    CR_TIMER_IN(CoarseSort);
    s_tileEmitPrefixSum[0] = 0;
    s_tileAllocPrefixSum[0] = 0;
    s_scanTemp[threadIdx.y][threadIdx.x] = 0;

    // Sort bins in descending order of triangle count.

    for (int binIdx = thrInBlock; binIdx < c_crParams.numBins; binIdx += CR_COARSE_WARPS * 32)
    {
        int count = 0;
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
            count += binTotal[(binIdx << CR_BIN_STREAMS_LOG2) + i];
        s_binOrder[binIdx] = (~count << (CR_MAXBINS_LOG2 * 2)) | binIdx;
    }

    __syncthreads();
    sortShared(s_binOrder, c_crParams.numBins);

    CR_TIMER_SYNC();
    CR_TIMER_OUT(CoarseSort);

    // Process each bin by one block.

    for (;;)
    {
        // Pick a bin for the block.

        CR_TIMER_IN(CoarseBinInit);

        if (thrInBlock == 0)
            s_workCounter = atomicAdd(&g_crAtomics.coarseCounter, 1);
        __syncthreads();

        int workCounter = s_workCounter;
        if (workCounter >= c_crParams.numBins)
        {
            CR_TIMER_OUT(CoarseBinInit);
            break;
        }

        U32 binOrder = s_binOrder[workCounter];
        bool binEmpty = ((~binOrder >> (CR_MAXBINS_LOG2 * 2)) == 0);
        if (binEmpty && !c_crParams.deferredClear)
        {
            CR_TIMER_OUT(CoarseBinInit);
            break;
        }

        int binIdx = binOrder & (CR_MAXBINS_SQR - 1);

        // Initialize input/output streams.

        int triQueueWritePos = 0;
        int triQueueReadPos = 0;

        if (thrInBlock < CR_BIN_STREAMS_SIZE)
        {
            int segIdx = binFirstSeg[(binIdx << CR_BIN_STREAMS_LOG2) + thrInBlock];
            s_binStreamCurrSeg[thrInBlock] = segIdx;
            s_binStreamFirstTri[thrInBlock] = (segIdx == -1) ? ~0u : binSegData[segIdx << CR_BIN_SEG_LOG2];
        }

        for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            s_tileStreamCurrOfs[tileInBin] = -CR_TILE_SEG_SIZE;

        // Initialize per-bin state.

        int binY = idiv_fast(binIdx, c_crParams.widthBins);
        int binX = binIdx - binY * c_crParams.widthBins;
        int originX = (binX << (CR_BIN_LOG2 + tileLog)) - (c_crParams.viewportWidth << (CR_SUBPIXEL_LOG2 - 1));
        int originY = (binY << (CR_BIN_LOG2 + tileLog)) - (c_crParams.viewportHeight << (CR_SUBPIXEL_LOG2 - 1));
        int maxTileXInBin = ::min(c_crParams.widthTiles - (binX << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int maxTileYInBin = ::min(c_crParams.heightTiles - (binY << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int binTileIdx = (binX + binY * c_crParams.widthTiles) << CR_BIN_LOG2;

        CR_TIMER_SYNC();
        CR_TIMER_OUT(CoarseBinInit);
        CR_COUNT(CoarseBins, (thrInBlock == 0) ? 1 : 0, 0);
        CR_COUNT(CoarseRoundsPerBin, 0, 1);

        // Entire block: Merge input streams and process triangles.

        if (!binEmpty)
        do
        {
            //------------------------------------------------------------------------
            // Merge.
            //------------------------------------------------------------------------

            CR_TIMER_IN(CoarseMerge);
            CR_COUNT(CoarseRoundsPerBin, 1, 0);
            CR_COUNT(CoarseMergePerRound, 0, 1);

            // Entire block: Not enough triangles => merge and queue segments.
            // NOTE: The bin exit criterion assumes that we queue more triangles than we actually need.

            while (triQueueWritePos - triQueueReadPos <= CR_COARSE_WARPS * 32)
            {
                CR_COUNT(CoarseMergePerRound, 1, 0);

                // First warp: Choose the segment with the lowest initial triangle index.

                if (thrInBlock < CR_BIN_STREAMS_SIZE)
                {
                    // Find the stream with the lowest triangle index.

                    U32 firstTri = s_binStreamFirstTri[thrInBlock];
                    U32 t = firstTri;
                    volatile U32* p = &s_scanTemp[0][thrInBlock + 16];

                    CR_TIMER_OUT(CoarseMerge);
                    CR_TIMER_IN(CoarseMergeSum);

                    #if (CR_BIN_STREAMS_SIZE > 1)
                        p[0] = t, t = ::min(t, p[-1]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 2)
                        p[0] = t, t = ::min(t, p[-2]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 4)
                        p[0] = t, t = ::min(t, p[-4]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 8)
                        p[0] = t, t = ::min(t, p[-8]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 16)
                        p[0] = t, t = ::min(t, p[-16]);
                    #endif
                    p[0] = t;

                    CR_TIMER_OUT_DEP(CoarseMergeSum, t);
                    CR_TIMER_IN(CoarseMerge);

                    // Consume and broadcast.

                    if (s_scanTemp[0][CR_BIN_STREAMS_SIZE - 1 + 16] == firstTri)
                    {
                        int segIdx = s_binStreamCurrSeg[thrInBlock];
                        s_binStreamSelectedOfs = segIdx << CR_BIN_SEG_LOG2;
                        if (segIdx != -1)
                        {
                            int segSize = binSegCount[segIdx];
                            int segNext = binSegNext[segIdx];
                            s_binStreamSelectedSize = segSize;
                            s_triQueueWritePos = triQueueWritePos + segSize;
                            s_binStreamCurrSeg[thrInBlock] = segNext;
                            s_binStreamFirstTri[thrInBlock] = (segNext == -1) ? ~0u : binSegData[segNext << CR_BIN_SEG_LOG2];
                        }
                    }
                }

                // No more segments => break.

                __syncthreads();
                triQueueWritePos = s_triQueueWritePos;
                int segOfs = s_binStreamSelectedOfs;
                if (segOfs < 0)
                    break;

                int segSize = s_binStreamSelectedSize;
                __syncthreads();

                // Fetch triangles into the queue.

                CR_TIMER_OUT(CoarseMerge);
                CR_TIMER_IN(CoarseStreamRead);

                for (int idxInSeg = CR_COARSE_WARPS * 32 - 1 - thrInBlock; idxInSeg < segSize; idxInSeg += CR_COARSE_WARPS * 32)
                {
                    S32 triIdx = binSegData[segOfs + idxInSeg];
                    s_triQueue[(triQueueWritePos - segSize + idxInSeg) & (CR_COARSE_QUEUE_SIZE - 1)] = triIdx;
                }

                CR_TIMER_SYNC();
                CR_TIMER_OUT(CoarseStreamRead);
                CR_TIMER_IN(CoarseMerge);
            }

            // All threads: Clear emit masks.

            for (int maskIdx = thrInBlock; maskIdx < CR_COARSE_WARPS * CR_BIN_SQR; maskIdx += CR_COARSE_WARPS * 32)
                s_warpEmitMask[maskIdx >> (CR_BIN_LOG2 * 2)][maskIdx & (CR_BIN_SQR - 1)] = 0;

            __syncthreads();
            CR_TIMER_OUT(CoarseMerge);

            //------------------------------------------------------------------------
            // Raster.
            //------------------------------------------------------------------------

            // Triangle per thread: Read from the queue.

            CR_TIMER_IN(CoarseTriRead);

            int triIdx = -1;
            if (triQueueReadPos + thrInBlock < triQueueWritePos)
                triIdx = s_triQueue[(triQueueReadPos + thrInBlock) & (CR_COARSE_QUEUE_SIZE - 1)];

            uint4 triData = make_uint4(0, 0, 0, 0);
            if (triIdx != -1)
            {
                int dataIdx = triIdx >> 3;
                int subtriIdx = triIdx & 7;
                if (subtriIdx != 7)
                    dataIdx = triHeader[dataIdx].misc + subtriIdx;
                triData = tex1Dfetch(t_triHeader, dataIdx);
            }

            CR_TIMER_SYNC();
            CR_TIMER_OUT_DEP(CoarseTriRead, triData);
            CR_COUNT(CoarseTrisPerRound, (triIdx == -1) ? 0 : 1, (thrInBlock == 0) ? 1 : 0);

            // 32 triangles per warp: Record emits (= tile intersections).

            CR_TIMER_IN(CoarseRasterize);

            if (__any(triIdx != -1))
            {
                S32 v0x = sub_s16lo_s16lo(triData.x, originX);
                S32 v0y = sub_s16hi_s16lo(triData.x, originY);
                S32 d01x = sub_s16lo_s16lo(triData.y, triData.x);
                S32 d01y = sub_s16hi_s16hi(triData.y, triData.x);
                S32 d02x = sub_s16lo_s16lo(triData.z, triData.x);
                S32 d02y = sub_s16hi_s16hi(triData.z, triData.x);

                // Compute tile-based AABB.

                int lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                int loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);
                int hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                int hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);
                int sizex = add_sub(hix, 1, lox);
                int sizey = add_sub(hiy, 1, loy);
                int area = sizex * sizey;

                // Miscellaneous init.

                U8* currPtr = (U8*)&s_warpEmitMask[threadIdx.y][lox + (loy << CR_BIN_LOG2)];
                int ptrYInc = CR_BIN_SIZE * 4 - (sizex << 2);
                U32 maskBit = 1 << threadIdx.x;

                CR_COUNT(CoarseCaseA, 0, 1);
                CR_COUNT(CoarseCaseB, 0, 1);
                CR_COUNT(CoarseCaseC, 0, 1);

                // Case A: All AABBs are small => record the full AABB using atomics.

                if (__all(sizex <= 2 && sizey <= 2))
                {
                    CR_COUNT(CoarseCaseA, 100, 0);
                    CR_TIMER_OUT(CoarseRasterize);
                    CR_TIMER_IN(CoarseRasterAtomic);

                    if (triIdx != -1)
                    {
                        atomicOr((U32*)currPtr, maskBit);
                        if (sizex == 2) atomicOr((U32*)(currPtr + 4), maskBit);
                        if (sizey == 2) atomicOr((U32*)(currPtr + CR_BIN_SIZE * 4), maskBit);
                        if (sizex == 2 && sizey == 2) atomicOr((U32*)(currPtr + 4 + CR_BIN_SIZE * 4), maskBit);
                    }

                    CR_TIMER_OUT(CoarseRasterAtomic);
                    CR_TIMER_IN(CoarseRasterize);
                }
                else
                {
                    // Compute warp-AABB (scan-32).

                    U32 aabbMask = add_sub(2 << hix, 0x20000 << hiy, 1 << lox) - (0x10000 << loy);
                    if (triIdx == -1)
                        aabbMask = 0;

                    volatile U32* p = &s_scanTemp[threadIdx.y][threadIdx.x + 16];
                    p[0] = aabbMask, aabbMask |= p[-1];
                    p[0] = aabbMask, aabbMask |= p[-2];
                    p[0] = aabbMask, aabbMask |= p[-4];
                    p[0] = aabbMask, aabbMask |= p[-8];
                    p[0] = aabbMask, aabbMask |= p[-16];
                    p[0] = aabbMask, aabbMask = s_scanTemp[threadIdx.y][47];

                    U32 maskX = aabbMask & 0xFFFF;
                    U32 maskY = aabbMask >> 16;
                    int wlox = findLeadingOne(maskX ^ (maskX - 1));
                    int wloy = findLeadingOne(maskY ^ (maskY - 1));
                    int whix = findLeadingOne(maskX);
                    int whiy = findLeadingOne(maskY);
                    int warea = (add_sub(whix, 1, wlox)) * (add_sub(whiy, 1, wloy));

                    // Initialize edge functions.

                    S32 d12x = d02x - d01x;
                    S32 d12y = d02y - d01y;
                    v0x -= lox << tileLog;
                    v0y -= loy << tileLog;

                    S32 t01 = v0x * d01y - v0y * d01x;
                    S32 t02 = v0y * d02x - v0x * d02y;
                    S32 t12 = d01x * d12y - d01y * d12x - t01 - t02;
                    S32 b01 = add_sub(t01 >> tileLog, ::max(d01x, 0), ::min(d01y, 0));
                    S32 b02 = add_sub(t02 >> tileLog, ::max(d02y, 0), ::min(d02x, 0));
                    S32 b12 = add_sub(t12 >> tileLog, ::max(d12x, 0), ::min(d12y, 0));

                    d01x += sizex * d01y;
                    d02x += sizex * d02y;
                    d12x += sizex * d12y;

                    // Case B: Warp-AABB is not much larger than largest AABB => Check tiles in warp-AABB, record using ballots.

                    if (__any(warea * 4 <= area * 8))
                    {
                        CR_COUNT(CoarseCaseB, 100, 0);
                        if (triIdx != -1)
                        {
                            for (int y = wloy; y <= hiy; y++)
                            {
                                if (y < loy) continue;
                                for (int x = wlox; x <= hix; x++)
                                {
                                    if (x < lox) continue;
                                    *(U32*)currPtr = __ballot(b01 >= 0 && b02 >= 0 && b12 >= 0);
                                    currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                }
                                currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x;
                            }
                        }
                    }

                    // Case C: General case => Check tiles in AABB, record using atomics.

                    else
                    {
                        CR_COUNT(CoarseCaseC, 100, 0);
                        CR_TIMER_OUT(CoarseRasterize);
                        CR_TIMER_IN(CoarseRasterAtomic);

                        if (triIdx != -1)
                        {
                            U8* skipPtr = currPtr + (sizex << 2);
                            U8* endPtr  = currPtr + (sizey << (CR_BIN_LOG2 + 2));
                            do
                            {
                                if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                    atomicOr((U32*)currPtr, maskBit);
                                currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                if (currPtr == skipPtr)
                                    currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x, skipPtr += CR_BIN_SIZE * 4;
                            }
                            while (currPtr != endPtr);
                        }

                        CR_TIMER_OUT(CoarseRasterAtomic);
                        CR_TIMER_IN(CoarseRasterize);
                    }
                }
            }

            __syncthreads();
            CR_TIMER_OUT(CoarseRasterize);

            //------------------------------------------------------------------------
            // Count.
            //------------------------------------------------------------------------

            CR_TIMER_IN(CoarseCount);

            // Tile per thread: Initialize prefix sums.

            for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            {
                // Compute prefix sum of emits over warps.

                U8* srcPtr = (U8*)&s_warpEmitMask[0][tileInBin];
                U8* dstPtr = (U8*)&s_warpEmitPrefixSum[0][tileInBin];
                int tileEmits = 0;
                for (int i = 0; i < CR_COARSE_WARPS; i++)
                {
                    tileEmits += __popc(*(U32*)srcPtr);
                    *(U32*)dstPtr = tileEmits;
                    srcPtr += (CR_BIN_SQR + 1) * 4;
                    dstPtr += (CR_BIN_SQR + 1) * 4;
                }

                CR_COUNT(CoarseTilesPerRound, (tileEmits == 0) ? 0 : 1, (tileInBin == 0) ? 1 : 0);

                // Determine the number of segments to allocate.

                int spaceLeft = -s_tileStreamCurrOfs[tileInBin] & (CR_TILE_SEG_SIZE - 1);
                int tileAllocs = (tileEmits - spaceLeft + CR_TILE_SEG_SIZE - 1) >> CR_TILE_SEG_LOG2;
                volatile U32* p = &s_tileEmitPrefixSum[tileInBin + 1];

                // All counters within the warp are small => compute prefix sum using ballot.

                if (!__any(tileEmits >= 2))
                {
                    U32 m = getLaneMaskLe();
                    *p = (__popc(__ballot(tileEmits & 1) & m) << emitShift) | __popc(__ballot(tileAllocs & 1) & m);
                }

                // Otherwise => scan-32 within the warp.

                else
                {
                    CR_TIMER_OUT(CoarseCount);
                    CR_TIMER_IN(CoarseCountSum);

                    U32 sum = (tileEmits << emitShift) | tileAllocs;
                    *p = sum; if (threadIdx.x >= 1)  sum += p[-1];
                    *p = sum; if (threadIdx.x >= 2)  sum += p[-2];
                    *p = sum; if (threadIdx.x >= 4)  sum += p[-4];
                    *p = sum; if (threadIdx.x >= 8)  sum += p[-8];
                    *p = sum; if (threadIdx.x >= 16) sum += p[-16];
                    *p = sum;

                    CR_TIMER_OUT_DEP(CoarseCountSum, sum);
                    CR_TIMER_IN(CoarseCount);
                }
            }

            // First warp: Scan-8.

            __syncthreads();
            CR_TIMER_OUT(CoarseCount);
            CR_TIMER_IN(CoarseCountSum);

            if (thrInBlock < CR_BIN_SQR / 32)
            {
                int sum = s_tileEmitPrefixSum[(thrInBlock << 5) + 32];
                volatile U32* p = &s_scanTemp[0][thrInBlock + 16];
                p[0] = sum;
                #if (CR_BIN_SQR > 1 * 32)
                    sum += p[-1], p[0] = sum;
                #endif
                #if (CR_BIN_SQR > 2 * 32)
                    sum += p[-2], p[0] = sum;
                #endif
                #if (CR_BIN_SQR > 4 * 32)
                    sum += p[-4], p[0] = sum;
                #endif
            }

            __syncthreads();
            CR_TIMER_OUT(CoarseCountSum);
            CR_TIMER_IN(CoarseCount);

            // Tile per thread: Finalize prefix sums.
            // Single thread: Allocate segments.

            for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            {
                int sum = s_tileEmitPrefixSum[tileInBin + 1] + s_scanTemp[0][(tileInBin >> 5) + 15];
                int numEmits = sum >> emitShift;
                int numAllocs = sum & ((1 << emitShift) - 1);
                s_tileEmitPrefixSum[tileInBin + 1] = numEmits;
                s_tileAllocPrefixSum[tileInBin + 1] = numAllocs;

                if (tileInBin == CR_BIN_SQR - 1 && numAllocs != 0)
                {
                    int t = atomicAdd(&g_crAtomics.numTileSegs, numAllocs);
                    s_firstAllocSeg = (t + numAllocs <= c_crParams.maxTileSegs) ? t : 0;
                }
            }

            __syncthreads();
            int firstAllocSeg   = s_firstAllocSeg;
            int totalEmits      = s_tileEmitPrefixSum[CR_BIN_SQR];
            int totalAllocs     = s_tileAllocPrefixSum[CR_BIN_SQR];

            CR_COUNT(CoarseEmitsPerRound, totalEmits, 1);
            CR_COUNT(CoarseAllocsPerRound, totalAllocs, 1);
            CR_COUNT(CoarseEmitsPerTri, (thrInBlock == 0) ? totalEmits : 0, (triIdx == -1) ? 0 : 1);
            CR_TIMER_OUT(CoarseCount);

            //------------------------------------------------------------------------
            // Emit.
            //------------------------------------------------------------------------

            CR_TIMER_IN(CoarseEmit);

            // Emit per thread: Write triangle index to globalmem.

            for (int emitInBin = thrInBlock; emitInBin < totalEmits; emitInBin += CR_COARSE_WARPS * 32)
            {
                // Find tile in bin.

                U8* tileBase = (U8*)&s_tileEmitPrefixSum[0];
                U8* tilePtr = tileBase;
                U8* ptr;

                #if (CR_BIN_SQR > 128)
                    ptr = tilePtr + 0x80 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 64)
                    ptr = tilePtr + 0x40 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 32)
                    ptr = tilePtr + 0x20 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 16)
                    ptr = tilePtr + 0x10 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 8)
                    ptr = tilePtr + 0x08 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 4)
                    ptr = tilePtr + 0x04 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 2)
                    ptr = tilePtr + 0x02 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 1)
                    ptr = tilePtr + 0x01 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif

                int tileInBin = (tilePtr - tileBase) >> 2;
                int emitInTile = emitInBin - *(U32*)tilePtr;

                // Find warp in tile.

                int warpStep = (CR_BIN_SQR + 1) * 4;
                U8* warpBase = (U8*)&s_warpEmitPrefixSum[0][tileInBin] - warpStep;
                U8* warpPtr = warpBase;

                #if (CR_COARSE_WARPS > 8)
                    ptr = warpPtr + 0x08 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 4)
                    ptr = warpPtr + 0x04 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 2)
                    ptr = warpPtr + 0x02 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 1)
                    ptr = warpPtr + 0x01 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif

                int warpInTile = (warpPtr - warpBase) >> (CR_BIN_LOG2 * 2 + 2);
                U32 emitMask = *(U32*)(warpPtr + warpStep + ((U8*)s_warpEmitMask - (U8*)s_warpEmitPrefixSum));
                int emitInWarp = emitInTile - *(U32*)(warpPtr + warpStep) + __popc(emitMask);

                // Find thread in warp.

                CR_TIMER_OUT_DEP(CoarseEmit, emitInWarp);
                CR_TIMER_IN(CoarseEmitBitFind);

                int threadInWarp = 0;
                int pop = __popc(emitMask & 0xFFFF);
                bool pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x10;
                if (pred) threadInWarp += 0x10;

                pop = __popc(emitMask & 0xFF);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x08;
                if (pred) threadInWarp += 0x08;

                pop = __popc(emitMask & 0xF);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x04;
                if (pred) threadInWarp += 0x04;

                pop = __popc(emitMask & 0x3);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x02;
                if (pred) threadInWarp += 0x02;

                if (emitInWarp >= (emitMask & 1))
                    threadInWarp++;

                CR_TIMER_OUT_DEP(CoarseEmitBitFind, threadInWarp);
                CR_TIMER_IN(CoarseEmit);

                // Figure out where to write.

                int currOfs = s_tileStreamCurrOfs[tileInBin];
                int spaceLeft = -currOfs & (CR_TILE_SEG_SIZE - 1);
                int outOfs = emitInTile;

                if (outOfs < spaceLeft)
                    outOfs += currOfs;
                else
                {
                    int allocLo = firstAllocSeg + s_tileAllocPrefixSum[tileInBin];
                    outOfs += (allocLo << CR_TILE_SEG_LOG2) - spaceLeft;
                }

                // Write.

                int queueIdx = warpInTile * 32 + threadInWarp;
                int triIdx = s_triQueue[(triQueueReadPos + queueIdx) & (CR_COARSE_QUEUE_SIZE - 1)];

                CR_TIMER_OUT_DEP(CoarseEmit, triIdx);
                CR_TIMER_IN(CoarseStreamWrite);
                tileSegData[outOfs] = triIdx;
                CR_TIMER_OUT(CoarseStreamWrite);
                CR_TIMER_IN(CoarseEmit);
            }

            CR_TIMER_SYNC();
            CR_TIMER_OUT(CoarseEmit);

            //------------------------------------------------------------------------
            // Patch.
            //------------------------------------------------------------------------

            CR_TIMER_IN(CoarsePatch);

            // Allocated segment per thread: Initialize next-pointer and count.

            for (int i = CR_COARSE_WARPS * 32 - 1 - thrInBlock; i < totalAllocs; i += CR_COARSE_WARPS * 32)
            {
                int segIdx = firstAllocSeg + i;
                tileSegNext[segIdx] = segIdx + 1;
                tileSegCount[segIdx] = CR_TILE_SEG_SIZE;
            }

            // Tile per thread: Fix previous segment's next-pointer and update s_tileStreamCurrOfs.

            __syncthreads();
            for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            {
                int oldOfs = s_tileStreamCurrOfs[tileInBin];
                int newOfs = oldOfs + s_warpEmitPrefixSum[CR_COARSE_WARPS - 1][tileInBin];
                int allocLo = s_tileAllocPrefixSum[tileInBin];
                int allocHi = s_tileAllocPrefixSum[tileInBin + 1];

                if (allocLo != allocHi)
                {
                    S32* nextPtr = &tileSegNext[(oldOfs - 1) >> CR_TILE_SEG_LOG2];
                    if (oldOfs < 0)
                        nextPtr = &tileFirstSeg[binTileIdx + globalTileIdx(tileInBin)];
                    *nextPtr = firstAllocSeg + allocLo;

                    newOfs--;
                    newOfs &= CR_TILE_SEG_SIZE - 1;
                    newOfs |= (firstAllocSeg + allocHi - 1) << CR_TILE_SEG_LOG2;
                    newOfs++;
                }
                s_tileStreamCurrOfs[tileInBin] = newOfs;
            }

            CR_TIMER_SYNC();
            CR_TIMER_OUT(CoarsePatch);

            // Advance queue read pointer.
            // Queue became empty => bin done.

            triQueueReadPos += CR_COARSE_WARPS * 32;
        }
        while (triQueueReadPos < triQueueWritePos);

        // Tile per thread: Fix next-pointer and count of the last segment.
        // 32 tiles per warp: Count active tiles.

        CR_TIMER_IN(CoarseBinDeinit);
        __syncthreads();

        for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
        {
            int tileX = tileInBin & (CR_BIN_SIZE - 1);
            int tileY = tileInBin >> CR_BIN_LOG2;
            bool force = (c_crParams.deferredClear & tileX <= maxTileXInBin & tileY <= maxTileYInBin);

            int ofs = s_tileStreamCurrOfs[tileInBin];
            int segIdx = (ofs - 1) >> CR_TILE_SEG_LOG2;
            int segCount = ofs & (CR_TILE_SEG_SIZE - 1);

            if (ofs >= 0)
                tileSegNext[segIdx] = -1;
            else if (force)
            {
                s_tileStreamCurrOfs[tileInBin] = 0;
                tileFirstSeg[binTileIdx + tileX + tileY * c_crParams.widthTiles] = -1;
            }

            if (segCount != 0)
                tileSegCount[segIdx] = segCount;

            s_scanTemp[0][(tileInBin >> 5) + 16] = __popc(__ballot(ofs >= 0 | force));
        }

        // First warp: Scan-8.
        // One thread: Allocate space for active tiles.

        __syncthreads();
        if (thrInBlock < CR_BIN_SQR / 32)
        {
            volatile U32* p = &s_scanTemp[0][thrInBlock + 16];
            U32 sum = p[0];
            #if (CR_BIN_SQR > 1 * 32)
                sum += p[-1], p[0] = sum;
            #endif
            #if (CR_BIN_SQR > 2 * 32)
                sum += p[-2], p[0] = sum;
            #endif
            #if (CR_BIN_SQR > 4 * 32)
                sum += p[-4], p[0] = sum;
            #endif

            if (thrInBlock == CR_BIN_SQR / 32 - 1)
                s_firstActiveIdx = atomicAdd(&g_crAtomics.numActiveTiles, sum);
        }

        // Tile per thread: Output active tiles.

        __syncthreads();
        for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
        {
            if (s_tileStreamCurrOfs[tileInBin] < 0)
                continue;

            int activeIdx = s_firstActiveIdx;
            activeIdx += s_scanTemp[0][(tileInBin >> 5) + 15];
            activeIdx += __popc(__ballot(true) & getLaneMaskLt());
            activeTiles[activeIdx] = binTileIdx + globalTileIdx(tileInBin);
        }

        CR_TIMER_SYNC();
        CR_TIMER_OUT(CoarseBinDeinit);
    }

    CR_TIMER_OUT(CoarseTotal);
    CR_TIMER_DEINIT();
}

//------------------------------------------------------------------------
