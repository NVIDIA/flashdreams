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

#include "base/Random.hpp"
#include "base/DLLImports.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define FW_USE_MERSENNE_TWISTER 0

//------------------------------------------------------------------------
// Mersenne Twister
//------------------------------------------------------------------------

#if FW_USE_MERSENNE_TWISTER
extern "C"
{
#include "3rdparty/mt19937ar/mt19937ar_ctx.h"
}
#endif

//------------------------------------------------------------------------
// RANROT-A
//------------------------------------------------------------------------

class RanrotA
{
private:
    S32     p1;
    S32     p2;
    U32     buffer[11];

public:
    void reset(U32 seed)
    {
        if (seed == 0)
            seed--;

        for (int i = 0; i < (int)FW_ARRAY_SIZE(buffer); i++)
        {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            buffer[i] = seed;
        }

        p1 = 0;
        p2 = 7;

        for (int i = 0; i < (int)FW_ARRAY_SIZE(buffer); i++)
            get();
    }

    U32 get(void)
    {
        U32 x = buffer[p1] + buffer[p2];
        x = (x << 13) | (x >> 19);
        buffer[p1] = x;

        p1--;
        p1 += (p1 >> 31) & FW_ARRAY_SIZE(buffer);
        p2--;
        p2 += (p2 >> 31) & FW_ARRAY_SIZE(buffer);
        return x;
    }
};

//------------------------------------------------------------------------
// Common functionality.
//------------------------------------------------------------------------

void Random::reset(void)
{
    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks))
        failWin32Error("QueryPerformanceCounter");
    reset(ticks.LowPart);
}

//------------------------------------------------------------------------

void Random::reset(U32 seed)
{
    resetImpl(seed);

    m_normalF32Valid    = false;
    m_normalF32         = 0.0f;
    m_normalF64Valid    = false;
    m_normalF64         = 0.0;
}

//------------------------------------------------------------------------

void Random::reset(const Random& other)
{
    assignImpl(other);

    m_normalF32Valid    = other.m_normalF32Valid;
    m_normalF32         = other.m_normalF32;
    m_normalF64Valid    = other.m_normalF64Valid;
    m_normalF64         = other.m_normalF64;
}

//------------------------------------------------------------------------

int Random::read(void* ptr, int size)
{
    for (int i = 0; i < size; i++)
        ((U8*)ptr)[i] = (U8)getU32();
    return max(size, 0);
}

//------------------------------------------------------------------------

F32 Random::getF32Normal(void)
{
    m_normalF32Valid = (!m_normalF32Valid);
    if (!m_normalF32Valid)
        return m_normalF32;

    F32 a, b, c;
    do
    {
        a = (F32)getU32() * (2.0f / 4294967296.0f) + (1.0f / 4294967296.0f - 1.0f);
        b = (F32)getU32() * (2.0f / 4294967296.0f) + (1.0f / 4294967296.0f - 1.0f);
        c = a * a + b * b;
    }
    while (c >= 1.0f);

    c = sqrt(-2.0f * log(c) / c);
    m_normalF32 = b * c;
    return a * c;
}

//------------------------------------------------------------------------

F64 Random::getF64Normal(void)
{
    m_normalF64Valid = (!m_normalF64Valid);
    if (!m_normalF64Valid)
        return m_normalF64;

    F64 a, b, c;
    do
    {
        a = (F64)getU64() * (2.0 / 18446744073709551616.0) + (1.0 / 18446744073709551616.0 - 1.0);
        b = (F64)getU64() * (2.0 / 18446744073709551616.0) + (1.0 / 18446744073709551616.0 - 1.0);
        c = a * a + b * b;
    }
    while (c >= 1.0);

    c = sqrt(-2.0 * log(c) / c);
    m_normalF64 = b * c;
    return a * c;
}

//------------------------------------------------------------------------
// Implementation wrappers.
//------------------------------------------------------------------------

void Random::initImpl(void)
{
#if FW_USE_MERSENNE_TWISTER
    m_impl = (void*)new genrand;
#else
    m_impl = (void*)new RanrotA;
#endif
}

//------------------------------------------------------------------------

void Random::deinitImpl(void)
{
#if FW_USE_MERSENNE_TWISTER
    delete (genrand*)m_impl;
#else
    delete (RanrotA*)m_impl;
#endif
}

//------------------------------------------------------------------------

void Random::resetImpl(U32 seed)
{
#if FW_USE_MERSENNE_TWISTER
    init_genrand((genrand*)m_impl, seed);
#else
    ((RanrotA*)m_impl)->reset(seed);
#endif
}

//------------------------------------------------------------------------

void Random::assignImpl(const Random& other)
{
#if FW_USE_MERSENNE_TWISTER
    *(genrand*)m_impl = *(const genrand*)other.m_impl;
#else
    *(RanrotA*)m_impl = *(const RanrotA*)other.m_impl;
#endif
}

//------------------------------------------------------------------------

U32 Random::getImpl(void)
{
#if FW_USE_MERSENNE_TWISTER
    return genrand_int32((genrand*)m_impl);
#else
    return ((RanrotA*)m_impl)->get();
#endif
}

//------------------------------------------------------------------------
