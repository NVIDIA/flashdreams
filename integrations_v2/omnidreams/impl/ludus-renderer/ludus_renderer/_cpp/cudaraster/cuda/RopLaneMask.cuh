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

#include "PixelPipe.hpp"
#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

template <class BlendShaderClass, U32 RenderModeFlags>
__device__ __inline__ U32 determineROPLaneMask() // mask of lanes that should process an earlier fragment than this lane
{
    bool reverseLanes = true;
    if ((RenderModeFlags & RenderModeFlag_EnableDepth) == 0)
    {
        BlendShaderClass bs;
        if (!bs.needsDst())
            reverseLanes = false;
    }

    // Volta+ replacement for upstream busy-wait on shared memory write arbitration.
    // Empirical Volta+ trace of the original loop with reverseLanes=true returned
    // bits 0..threadIdx-1 set, i.e. %lanemask_lt. By the same XOR-sequence algebra
    // (initial mask ~0u, toggled by bits 0..threadIdx), reverseLanes=false produces
    // bits threadIdx+1..31 set, i.e. %lanemask_gt. Both are valid permutations:
    // __popc gives a unique rank in [0,31] across the warp, as required.
    U32 mask;
    if (reverseLanes)
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    else
        asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

//------------------------------------------------------------------------
}
