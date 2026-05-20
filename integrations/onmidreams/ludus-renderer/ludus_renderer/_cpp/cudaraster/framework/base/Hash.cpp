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

#include "base/Hash.hpp"

using namespace FW;

//------------------------------------------------------------------------

U32 FW::hashBuffer(const void* ptr, int size)
{
    FW_ASSERT(size >= 0);
    FW_ASSERT(ptr || !size);

    if ((((S32)(UPTR)ptr | size) & 3) == 0)
        return hashBufferAlign(ptr, size);

    const U8*   src     = (const U8*)ptr;
    U32         a       = FW_HASH_MAGIC;
    U32         b       = FW_HASH_MAGIC;
    U32         c       = FW_HASH_MAGIC;

    while (size >= 12)
    {
        a += src[0] + (src[1] << 8) + (src[2] << 16) + (src[3] << 24);
        b += src[4] + (src[5] << 8) + (src[6] << 16) + (src[7] << 24);
        c += src[8] + (src[9] << 8) + (src[10] << 16) + (src[11] << 24);
        FW_JENKINS_MIX(a, b, c);
        src += 12;
        size -= 12;
    }

    switch (size)
    {
    case 11: c += src[10] << 16;
    case 10: c += src[9] << 8;
    case 9:  c += src[8];
    case 8:  b += src[7] << 24;
    case 7:  b += src[6] << 16;
    case 6:  b += src[5] << 8;
    case 5:  b += src[4];
    case 4:  a += src[3] << 24;
    case 3:  a += src[2] << 16;
    case 2:  a += src[1] << 8;
    case 1:  a += src[0];
    case 0:  break;
    }

    c += size;
    FW_JENKINS_MIX(a, b, c);
    return c;
}

//------------------------------------------------------------------------

U32 FW::hashBufferAlign(const void* ptr, int size)
{
    FW_ASSERT(size >= 0);
    FW_ASSERT(ptr || !size);
    FW_ASSERT(((UPTR)ptr & 3) == 0);
    FW_ASSERT((size & 3) == 0);

    const U32*  src     = (const U32*)ptr;
    U32         a       = FW_HASH_MAGIC;
    U32         b       = FW_HASH_MAGIC;
    U32         c       = FW_HASH_MAGIC;

    while (size >= 12)
    {
        a += src[0];
        b += src[1];
        c += src[2];
        FW_JENKINS_MIX(a, b, c);
        src += 3;
        size -= 12;
    }

    switch (size)
    {
    case 8: b += src[1];
    case 4: a += src[0];
    case 0: break;
    }

    c += size;
    FW_JENKINS_MIX(a, b, c);
    return c;
}

//------------------------------------------------------------------------
