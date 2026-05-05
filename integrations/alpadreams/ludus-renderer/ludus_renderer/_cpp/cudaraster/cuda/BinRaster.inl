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

__device__ __inline__ void binRasterImpl(void)
{
    __shared__ volatile U32 s_broadcast [CR_BIN_WARPS + 16];
    __shared__ volatile S32 s_outOfs    [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_outTotal  [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_overIndex [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_outMask   [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    __shared__ volatile S32 s_outCount  [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    __shared__ volatile S32 s_triBuf    [CR_BIN_WARPS*32*4];                // triangle ring buffer
    __shared__ volatile U32 s_batchPos;
    __shared__ volatile U32 s_bufCount;
    __shared__ volatile U32 s_overTotal;
    __shared__ volatile U32 s_allocBase;

    const U8*               triSubtris      = (const U8*)c_crParams.triSubtris;
    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)c_crParams.triHeader;
    S32*                    binFirstSeg     = (S32*)c_crParams.binFirstSeg;
    S32*                    binTotal        = (S32*)c_crParams.binTotal;
    S32*                    binSegData      = (S32*)c_crParams.binSegData;
    S32*                    binSegNext      = (S32*)c_crParams.binSegNext;
    S32*                    binSegCount     = (S32*)c_crParams.binSegCount;

    if (g_crAtomics.numSubtris > c_crParams.maxSubtris)
        return;

    CR_TIMER_INIT();
    CR_TIMER_IN(BinTotal);

    // per-thread state
    int thrInBlock = threadIdx.x + threadIdx.y * 32;
    int batchPos = 0;

    // first 16 elements of s_broadcast are always zero
    if (thrInBlock < 16)
        s_broadcast[thrInBlock] = 0;

    // initialize output linked lists and offsets
    if (thrInBlock < c_crParams.numBins)
    {
        binFirstSeg[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = -1;
        s_outOfs[thrInBlock] = -CR_BIN_SEG_SIZE;
        s_outTotal[thrInBlock] = 0;
    }

    // repeat until done
    for(;;)
    {
        CR_TIMER_IN(BinPickBin);
        // get batch
        if (thrInBlock == 0)
            s_batchPos = atomicAdd(&g_crAtomics.binCounter, c_crParams.binBatchSize);
        __syncthreads();
        batchPos = s_batchPos;
        CR_TIMER_OUT_DEP(BinPickBin, batchPos);

        // all batches done?
        if (batchPos >= c_crParams.numTris)
            break;

        // per-thread state
        int bufIndex = 0;
        int bufCount = 0;
        int batchEnd = ::min(batchPos + c_crParams.binBatchSize, c_crParams.numTris);

        // loop over batch as long as we have triangles in it
        do
        {
            // read more triangles
            while (bufCount < CR_BIN_WARPS*32 && batchPos < batchEnd)
            {
                // get subtriangle count

                CR_TIMER_IN(BinReadTriHeader);
                int triIdx = batchPos + thrInBlock;
                int num = 0;
                if (triIdx < batchEnd)
                    num = triSubtris[triIdx];
                CR_COUNT(SetupSamplesPerTri, 0, (num != 0) ? 1 : 0);
                CR_TIMER_OUT_DEP(BinReadTriHeader, num);

                CR_TIMER_IN(BinCompactSubtri);
                // cumulative sum of subtriangles within each warp
                U32 myIdx = __popc(__ballot(num & 1) & getLaneMaskLt());
                if (__any(num > 1))
                {
                    myIdx += __popc(__ballot(num & 2) & getLaneMaskLt()) * 2;
                    myIdx += __popc(__ballot(num & 4) & getLaneMaskLt()) * 4;
                }
                s_broadcast[threadIdx.y + 16] = myIdx + num;
                __syncthreads();

                // cumulative sum of per-warp subtriangle counts
                if (thrInBlock < CR_BIN_WARPS)
                {
                    volatile U32* ptr = &s_broadcast[thrInBlock + 16];
                    U32 val = *ptr;
                    #if (CR_BIN_WARPS > 1)
                        val += ptr[-1]; *ptr = val;
                    #endif
                    #if (CR_BIN_WARPS > 2)
                        val += ptr[-2]; *ptr = val;
                    #endif
                    #if (CR_BIN_WARPS > 4)
                        val += ptr[-4]; *ptr = val;
                    #endif
                    #if (CR_BIN_WARPS > 8)
                        val += ptr[-8]; *ptr = val;
                    #endif
                    #if (CR_BIN_WARPS > 16)
                        val += ptr[-16]; *ptr = val;
                    #endif

                    // initially assume that we consume everything
                    s_batchPos = batchPos + CR_BIN_WARPS * 32;
                    s_bufCount = bufCount + val;
                }
                __syncthreads();

                // skip if no subtriangles
                if (num)
                {
                    // calculate write position for first subtriangle
                    U32 pos = bufCount + myIdx + s_broadcast[threadIdx.y + 16 - 1];

                    // only write if entire triangle fits
                    if (pos + num <= FW_ARRAY_SIZE(s_triBuf))
                    {
                        pos += bufIndex; // adjust for current start position
                        pos &= FW_ARRAY_SIZE(s_triBuf)-1;
                        if (num == 1)
                            s_triBuf[pos] = triIdx * 8 + 7; // single triangle
                        else
                        {
                            for (int i=0; i < num; i++)
                            {
                                s_triBuf[pos] = triIdx * 8 + i;
                                pos++;
                                pos &= FW_ARRAY_SIZE(s_triBuf)-1;
                            }
                        }
                    } else if (pos <= FW_ARRAY_SIZE(s_triBuf))
                    {
                        // this triangle is the first that failed, overwrite total count and triangle count
                        s_batchPos = batchPos + thrInBlock;
                        s_bufCount = pos;
                        CR_COUNT(BinInputOverflow, 1, 0);
                    }
                }

                // update triangle counts
                __syncthreads();
                batchPos = s_batchPos;
                bufCount = s_bufCount;

                CR_TIMER_OUT_DEP(BinCompactSubtri, bufCount);
            }

            CR_COUNT(BinTrisPerRound, ::min(CR_BIN_WARPS*32, bufCount), 1);
//            CR_TIMER_OUT(BinCompact);
//            CR_TIMER_IN(BinRaster);

            // make every warp clear its output buffers
            for (int i=threadIdx.x; i < c_crParams.numBins; i += 32)
                s_outMask[threadIdx.y][i] = 0;

            // choose our triangle
            CR_TIMER_IN(BinReadTriangle);
            uint4 triData = make_uint4(0, 0, 0, 0);
            if (thrInBlock < bufCount)
            {
                U32 triPos = bufIndex + thrInBlock;
                triPos &= FW_ARRAY_SIZE(s_triBuf)-1;

                // find triangle
                int triIdx = s_triBuf[triPos];
                int dataIdx = triIdx >> 3;
                int subtriIdx = triIdx & 7;
                if (subtriIdx != 7)
                    dataIdx = triHeader[dataIdx].misc + subtriIdx;

                // read triangle

                triData = tex1Dfetch(t_triHeader, dataIdx);
            }
            CR_TIMER_SYNC();
            CR_TIMER_OUT_DEP(BinReadTriangle, triData);

            // setup bounding box and edge functions, and rasterize
            CR_TIMER_IN(BinRasterize);
            S32 lox, loy, hix, hiy;
            if (thrInBlock < bufCount)
            {
                S32 v0x = add_s16lo_s16lo(triData.x, c_crParams.viewportWidth  * (CR_SUBPIXEL_SIZE >> 1));
                S32 v0y = add_s16hi_s16lo(triData.x, c_crParams.viewportHeight * (CR_SUBPIXEL_SIZE >> 1));
                S32 d01x = sub_s16lo_s16lo(triData.y, triData.x);
                S32 d01y = sub_s16hi_s16hi(triData.y, triData.x);
                S32 d02x = sub_s16lo_s16lo(triData.z, triData.x);
                S32 d02y = sub_s16hi_s16hi(triData.z, triData.x);
                int binLog = CR_BIN_LOG2 + CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
                lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> binLog, 0, c_crParams.widthBins  - 1);
                loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> binLog, 0, c_crParams.heightBins - 1);
                hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> binLog, 0, c_crParams.widthBins  - 1);
                hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> binLog, 0, c_crParams.heightBins - 1);
                CR_COUNT(BinTriBBArea, (hix-lox+1)*(hiy-loy+1), 1);

                CR_COUNT(BinTriSinglePath, 0, 1);
                CR_COUNT(BinTriFastPath,   0, 1);
                CR_COUNT(BinTriSlowPath,   0, 1);

                U32 bit = 1 << threadIdx.x;
                bool multi = (hix != lox || hiy != loy);
                if (!__any(multi))
                {
                    CR_COUNT(BinTriSinglePath, 100, 0);
                    int binIdx = lox + c_crParams.widthBins * loy;
                    bool won;
                    CR_TIMER_OUT_DEP(BinRasterize, binIdx);
                    CR_TIMER_IN(BinRasterAtomic);
                    do
                    {
                        s_broadcast[threadIdx.y + 16] = binIdx;
                        int winner = s_broadcast[threadIdx.y + 16];
                        won = (binIdx == winner);
                        U32 mask = __ballot(won);
                        s_outMask[threadIdx.y][winner] = mask;
                    } while (!won);
                    CR_TIMER_OUT_DEP(BinRasterAtomic, won);
                    CR_TIMER_IN(BinRasterize);
                } else
                {
                    bool complex = (hix > lox+1 || hiy > loy+1);
                    if (!__any(complex))
                    {
                        CR_COUNT(BinTriFastPath, 100, 0);
                        int binIdx = lox + c_crParams.widthBins * loy;
                        CR_TIMER_OUT_DEP(BinRasterize, binIdx);
                        CR_TIMER_IN(BinRasterAtomic);
                        atomicOr((U32*)&s_outMask[threadIdx.y][binIdx], bit);
                        if (hix > lox) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + 1], bit);
                        if (hiy > loy) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + c_crParams.widthBins], bit);
                        if (hix > lox && hiy > loy) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + c_crParams.widthBins + 1], bit);
                        CR_TIMER_OUT(BinRasterAtomic);
                        CR_TIMER_IN(BinRasterize);
                    } else
                    {
                        CR_COUNT(BinTriSlowPath, 100, 0);
                        S32 d12x = d02x - d01x, d12y = d02y - d01y;
                        v0x -= lox << binLog, v0y -= loy << binLog;

                        S32 t01 = v0x * d01y - v0y * d01x;
                        S32 t02 = v0y * d02x - v0x * d02y;
                        S32 t12 = d01x * d12y - d01y * d12x - t01 - t02;
                        S32 b01 = add_sub(t01 >> binLog, ::max(d01x, 0), ::min(d01y, 0));
                        S32 b02 = add_sub(t02 >> binLog, ::max(d02y, 0), ::min(d02x, 0));
                        S32 b12 = add_sub(t12 >> binLog, ::max(d12x, 0), ::min(d12y, 0));

                        int width = hix - lox + 1;
                        d01x += width * d01y;
                        d02x += width * d02y;
                        d12x += width * d12y;

                        U8* currPtr = (U8*)&s_outMask[threadIdx.y][lox + loy * c_crParams.widthBins];
                        U8* skipPtr = (U8*)&s_outMask[threadIdx.y][(hix + 1) + loy * c_crParams.widthBins];
                        U8* endPtr  = (U8*)&s_outMask[threadIdx.y][lox + (hiy + 1) * c_crParams.widthBins];
                        int stride  = c_crParams.widthBins * 4;
                        int ptrYInc = stride - width * 4;

                        CR_TIMER_OUT_DEP(BinRasterize, b01|b02|b12|d01x|d02x|d12x);
                        CR_TIMER_IN(BinRasterAtomic);
                        do
                        {
                            if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                atomicOr((U32*)currPtr, bit);
                            currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                            if (currPtr == skipPtr)
                                currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x, skipPtr += stride;
                        }
                        while (currPtr != endPtr);
                        CR_TIMER_OUT_DEP(BinRasterAtomic, currPtr);
                        CR_TIMER_IN(BinRasterize);
                    }
                }
            }

            // count per-bin contributions
            s_overTotal = 0; // overflow counter

            // ensure that out masks are done
            __syncthreads();
            CR_TIMER_OUT_DEP(BinRasterize, s_overTotal);

            CR_TIMER_IN(BinCount);

            int overIndex = -1;
            if (thrInBlock < c_crParams.numBins)
            {
                U8* srcPtr = (U8*)&s_outMask[0][thrInBlock];
                U8* dstPtr = (U8*)&s_outCount[0][thrInBlock];
                int total = 0;
                for (int i = 0; i < CR_BIN_WARPS; i++)
                {
                    total += __popc(*(U32*)srcPtr);
                    *(U32*)dstPtr = total;
                    srcPtr += (CR_MAXBINS_SQR + 1) * 4;
                    dstPtr += (CR_MAXBINS_SQR + 1) * 4;
                }

                // overflow => request a new segment
                int ofs = s_outOfs[thrInBlock];
                if (((ofs - 1) >> CR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> CR_BIN_SEG_LOG2))
                {
                    U32 mask = __ballot(true);
                    overIndex = __popc(mask & getLaneMaskLt());
                    if (overIndex == 0)
                        s_broadcast[threadIdx.y + 16] = atomicAdd((U32*)&s_overTotal, __popc(mask));
                    overIndex += s_broadcast[threadIdx.y + 16];
                    s_overIndex[thrInBlock] = overIndex;
                }
            }

            // sync after overTotal is ready
            __syncthreads();
            CR_TIMER_OUT(BinCount);

            // at least one segment overflowed => allocate segments
            U32 overTotal = s_overTotal;
            U32 allocBase = 0;
            CR_COUNT(BinTriSegAlloc, overTotal, 1);
            if (overTotal > 0)
            {
                CR_TIMER_IN(BinAlloc);
                // allocate memory
                if (thrInBlock == 0)
                {
                    U32 allocBase = atomicAdd(&g_crAtomics.numBinSegs, overTotal);
                    s_allocBase = (allocBase + overTotal <= c_crParams.maxBinSegs) ? allocBase : 0;
                }
                __syncthreads();
                allocBase = s_allocBase;

                // did my bin overflow?
                if (overIndex != -1)
                {
                    // calculate new segment index
                    int segIdx = allocBase + overIndex;

                    // add to linked list
                    if (s_outOfs[thrInBlock] < 0)
                        binFirstSeg[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = segIdx;
                    else
                        binSegNext[(s_outOfs[thrInBlock] - 1) >> CR_BIN_SEG_LOG2] = segIdx;

                    // defaults
                    binSegNext [segIdx] = -1;
                    binSegCount[segIdx] = CR_BIN_SEG_SIZE;
                }
               CR_TIMER_OUT(BinAlloc);
            }

            CR_TIMER_IN(BinEnqueue);

            // concurrent emission -- each warp handles its own triangle
            if (thrInBlock < bufCount)
            {
                int triPos  = (bufIndex + thrInBlock) & (FW_ARRAY_SIZE(s_triBuf) - 1);
                int currBin = lox + loy * c_crParams.widthBins;
                int skipBin = (hix + 1) + loy * c_crParams.widthBins;
                int endBin  = lox + (hiy + 1) * c_crParams.widthBins;
                int binYInc = c_crParams.widthBins - (hix - lox + 1);

                // loop over triangle's bins
                do
                {
                    U32 outMask = s_outMask[threadIdx.y][currBin];
                    if (outMask & (1<<threadIdx.x))
                    {
                        int idx = __popc(outMask & getLaneMaskLt());
                        if (threadIdx.y > 0)
                            idx += s_outCount[threadIdx.y-1][currBin];

                        int base = s_outOfs[currBin];
                        int free = (-base) & (CR_BIN_SEG_SIZE - 1);
                        if (idx >= free)
                            idx += ((allocBase + s_overIndex[currBin]) << CR_BIN_SEG_LOG2) - free;
                        else
                            idx += base;

                        CR_TIMER_OUT_DEP(BinEnqueue, idx);
                        CR_TIMER_IN(BinWrite);
                        binSegData[idx] = s_triBuf[triPos];
                        CR_TIMER_OUT(BinWrite);
                        CR_TIMER_IN(BinEnqueue);
                    }

                    currBin++;
                    if (currBin == skipBin)
                        currBin += binYInc, skipBin += c_crParams.widthBins;
                }
                while (currBin != endBin);
            }

            // wait all triangles to finish, then replace overflown segment offsets
            __syncthreads();
            if (thrInBlock < c_crParams.numBins)
            {
                U32 total  = s_outCount[CR_BIN_WARPS - 1][thrInBlock];
                U32 oldOfs = s_outOfs[thrInBlock];
                if (overIndex == -1)
                    s_outOfs[thrInBlock] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    addr = ((addr - 1) & (CR_BIN_SEG_SIZE - 1)) + 1;
                    addr += (allocBase + overIndex) << CR_BIN_SEG_LOG2;
                    s_outOfs[thrInBlock] = addr;
                }
                s_outTotal[thrInBlock] += total;
            }

            // these triangles are now done
            int count = ::min(bufCount, CR_BIN_WARPS * 32);
            bufCount -= count;
            bufIndex += count;
            bufIndex &= FW_ARRAY_SIZE(s_triBuf)-1;

            CR_TIMER_OUT_DEP(BinEnqueue, bufIndex);
        }
        while (bufCount > 0 || batchPos < batchEnd);

        // flush all bins
        CR_TIMER_IN(BinEnqueue);
        if (thrInBlock < c_crParams.numBins)
        {
            int ofs = s_outOfs[thrInBlock];
            if (ofs & (CR_BIN_SEG_SIZE-1))
            {
                int seg = ofs >> CR_BIN_SEG_LOG2;
                binSegCount[seg] = ofs & (CR_BIN_SEG_SIZE-1);
                s_outOfs[thrInBlock] = (ofs + CR_BIN_SEG_SIZE - 1) & -CR_BIN_SEG_SIZE;
            }
        }
        CR_TIMER_OUT(BinEnqueue);
    }

    // output totals
    if (thrInBlock < c_crParams.numBins)
        binTotal[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = s_outTotal[thrInBlock];

    CR_TIMER_OUT(BinTotal);
    CR_TIMER_DEINIT();
}

//------------------------------------------------------------------------
