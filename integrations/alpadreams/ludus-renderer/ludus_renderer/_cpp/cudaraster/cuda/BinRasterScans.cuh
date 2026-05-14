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

#include "Constants.hpp"
#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

__device__ __inline__ U32 binRasterPerLanePrefix3Bit(U32 num, volatile U32* warpTotalSlot)
{
    U32 myIdx = __popc(__ballot_sync(0xFFFFFFFFu, num & 1) & getLaneMaskLt());
    if (__any_sync(0xFFFFFFFFu, num > 1))
    {
        myIdx += __popc(__ballot_sync(0xFFFFFFFFu, num & 2) & getLaneMaskLt()) * 2;
        myIdx += __popc(__ballot_sync(0xFFFFFFFFu, num & 4) & getLaneMaskLt()) * 4;
    }
    if (threadIdx.x == 31)
        *warpTotalSlot = myIdx + num;
    return myIdx;
}

__device__ __inline__ U32 binRasterPerWarpInclusiveScan(
    volatile U32* slot, int laneIdx, U32 priorBufCount, volatile U32* bufCountOut)
{
    U32 val = *slot;
    #if (CR_BIN_WARPS > 1)
        val += slot[-1]; *slot = val;
    #endif
    #if (CR_BIN_WARPS > 2)
        val += slot[-2]; *slot = val;
    #endif
    #if (CR_BIN_WARPS > 4)
        val += slot[-4]; *slot = val;
    #endif
    #if (CR_BIN_WARPS > 8)
        val += slot[-8]; *slot = val;
    #endif
    #if (CR_BIN_WARPS > 16)
        val += slot[-16]; *slot = val;
    #endif
    if (laneIdx == CR_BIN_WARPS - 1)
        *bufCountOut = priorBufCount + val;
    return val;
}

//------------------------------------------------------------------------
}
