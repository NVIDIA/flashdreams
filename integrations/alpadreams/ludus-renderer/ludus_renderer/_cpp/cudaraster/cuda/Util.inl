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

#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

__device__ __inline__ U32 idiv_fast(U32 a, U32 b)
{
    return f32_to_u32_sat_rmi(((F32)a + 0.5f) / (F32)b);
}

//------------------------------------------------------------------------

__device__ __inline__ U32 toABGR(float4 color)
{
	// 11 instructions: 4*FFMA, 4*F2I, 3*PRMT
	U32 x = f32_to_u32_sat_rmi(fma_rm(color.x, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 y = f32_to_u32_sat_rmi(fma_rm(color.y, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 z = f32_to_u32_sat_rmi(fma_rm(color.z, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 w = f32_to_u32_sat_rmi(fma_rm(color.w, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	return prmt(prmt(x, y, 0x0073), prmt(z, w, 0x0073), 0x5410);
}

//------------------------------------------------------------------------

__device__ __inline__ U32 blendABGR(U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor)
{
	U32 a = vmad_b0_b3(src, srcColorFactor, vmad_b0_b3(dst, dstColorFactor, 0)) * 0x010101 + 0x800000;
	U32 b = vmad_b1_b3(src, srcColorFactor, vmad_b1_b3(dst, dstColorFactor, 0)) * 0x010101 + 0x800000;
	U32 c = vmad_b2_b3(src, srcColorFactor, vmad_b2_b3(dst, dstColorFactor, 0)) * 0x010101 + 0x800000;
	U32 d = vmad_b3_b3(src, srcAlphaFactor, vmad_b3_b3(dst, dstAlphaFactor, 0)) * 0x010101 + 0x800000;
	return prmt(prmt(a, b, 0x0073), prmt(c, d, 0x0073), 0x5410);
}

//------------------------------------------------------------------------

__device__ __inline__ U32 blendABGRClamp(U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor)
{
    U32 a = ::min(vmad_b0_b3(src, srcColorFactor, vmad_b0_b3(dst, dstColorFactor, 0)), 255 * 255) * 0x010101 + 0x800000;
    U32 b = ::min(vmad_b1_b3(src, srcColorFactor, vmad_b1_b3(dst, dstColorFactor, 0)), 255 * 255) * 0x010101 + 0x800000;
    U32 c = ::min(vmad_b2_b3(src, srcColorFactor, vmad_b2_b3(dst, dstColorFactor, 0)), 255 * 255) * 0x010101 + 0x800000;
    U32 d = ::min(vmad_b3_b3(src, srcAlphaFactor, vmad_b3_b3(dst, dstAlphaFactor, 0)), 255 * 255) * 0x010101 + 0x800000;
	return prmt(prmt(a, b, 0x0073), prmt(c, d, 0x0073), 0x5410);
}

//------------------------------------------------------------------------
// v0 = subpixels relative to the bottom-left sampling point

__device__ __inline__ uint3 setupPleq(float3 values, int2 v0, int2 d1, int2 d2, F32 areaRcp, int samplesLog2)
{
    F32 mx = fmaxf(fmaxf(values.x, values.y), values.z);
    int sh = ::min(::max((__float_as_int(mx) >> 23) - (127 + 22), 0), 8);
    S32 t0 = (U32)values.x >> sh;
    S32 t1 = ((U32)values.y >> sh) - t0;
    S32 t2 = ((U32)values.z >> sh) - t0;

    U32 rcpMant = (__float_as_int(areaRcp) & 0x007FFFFF) | 0x00800000;
    int rcpShift = (23 + 127) - (__float_as_int(areaRcp) >> 23);

    uint3 pleq;
    S64 xc = ((S64)t1 * d2.y - (S64)t2 * d1.y) * rcpMant;
    S64 yc = ((S64)t2 * d1.x - (S64)t1 * d2.x) * rcpMant;
    pleq.x = (U32)(xc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2 - samplesLog2)));
    pleq.y = (U32)(yc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2 - samplesLog2)));

    S32 centerX = (v0.x * 2 + min_min(d1.x, d2.x, 0) + max_max(d1.x, d2.x, 0)) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    S32 centerY = (v0.y * 2 + min_min(d1.y, d2.y, 0) + max_max(d1.y, d2.y, 0)) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    S32 vcx = v0.x - (centerX << (CR_SUBPIXEL_LOG2 - samplesLog2));
    S32 vcy = v0.y - (centerY << (CR_SUBPIXEL_LOG2 - samplesLog2));

    pleq.z = t0 << sh;
    pleq.z -= (U32)(((xc >> 13) * vcx + (yc >> 13) * vcy) >> (rcpShift - (sh + 13)));
    pleq.z -= pleq.x * centerX + pleq.y * centerY;
    return pleq;
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_exact_ref(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S64 curr = (S64)ox * dy - (S64)oy * dx;
    S64 stepX = (S64)-dy << CR_SUBPIXEL_LOG2;
    S64 stepY = (S64)+dx << CR_SUBPIXEL_LOG2;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    return cover8x8_generateMask_ref(curr, stepX, stepY);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_conservative_ref(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S64 curr = (S64)ox * dy - (S64)oy * dx;
    S64 stepX = (S64)-dy << CR_SUBPIXEL_LOG2;
    S64 stepY = (S64)+dx << CR_SUBPIXEL_LOG2;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    curr += (::abs(stepX) + ::abs(stepY)) >> 1;
    return cover8x8_generateMask_ref(curr, stepX, stepY);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_generateMask_ref(S64 curr, S64 stepX, S64 stepY)
{
    stepY -= stepX * 7;
    U32 lo = 0;
    U32 hi = 0;
    for (int i = 0; i < 32; i++)
    {
        lo = slct(lo | (1 << i), lo, getHi(curr));
        curr += ((i & 7) == 7) ? stepY : stepX;
    }
    for (int i = 0; i < 32; i++)
    {
        hi = slct(hi | (1 << i), hi, getHi(curr));
        curr += ((i & 7) == 7) ? stepY : stepX;
    }
    return combineLoHi(lo, hi);
}

//------------------------------------------------------------------------

__device__ __inline__ bool cover8x8_missesTile(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S32 bias = 7 << (CR_SUBPIXEL_LOG2 - 1);
    S64 center = (S64)(bias - ox) * dy - (S64)(bias - oy) * dx;
    S32 extent = (::abs(dx) + ::abs(dy)) << (CR_SUBPIXEL_LOG2 + 2);
    return (::abs(center) >= extent);
}

//------------------------------------------------------------------------

__device__ __inline__ void cover8x8_setupLUT(volatile U64* lut)
{
    for (S32 lutIdx = threadIdx.x + blockDim.x * threadIdx.y; lutIdx < CR_COVER8X8_LUT_SIZE; lutIdx += blockDim.x * blockDim.y)
    {
        int half       = (lutIdx < (12 << 5)) ? 0 : 1;
        int yint       = (lutIdx >> 5) - half * 12 - 3;
        U32 shape      = ((lutIdx >> 2) & 7) << (31 - 2);
        S32 slctSwapXY = lutIdx << (31 - 1);
        S32 slctNegX   = lutIdx << (31 - 0);
        S32 slctCompl  = slctSwapXY ^ slctNegX;

        U64 mask = 0;
        int xlo = half * 4;
        int xhi = xlo + 4;
        for (int x = xlo; x < xhi; x++)
        {
            int ylo = slct(0, ::max(yint, 0), slctCompl);
            int yhi = slct(::min(yint, 8), 8, slctCompl);
            for (int y = ylo; y < yhi; y++)
            {
                int xx = slct(x, y, slctSwapXY);
                int yy = slct(y, x, slctSwapXY);
                xx = slct(xx, 7 - xx, slctNegX);
                mask |= (U64)1 << (xx + yy * 8);
            }
            yint += shape >> 31;
            shape <<= 1;
        }
        lut[lutIdx] = mask;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_exact_fast(S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut) // 52 instr
{
    F32  yinitBias  = (F32)(1 << (31 - CR_MAXVIEWPORT_LOG2 - CR_SUBPIXEL_LOG2 * 2));
    F32  yinitScale = (F32)(1 << (32 - CR_SUBPIXEL_LOG2));
    F32  yincScale  = 65536.0f * 65536.0f;

    S32  slctFlipY  = flips << (31 - CR_FLIPBIT_FLIP_Y);
    S32  slctFlipX  = flips << (31 - CR_FLIPBIT_FLIP_X);
    S32  slctSwapXY = flips << (31 - CR_FLIPBIT_SWAP_XY);

    // Evaluate cross product.

    S32 t = ox * dy - oy * dx;
    F32 det = (F32)slct(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);
    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    F32 xrcp  = 1.0f / (F32)::abs(slct(dx, dy, slctSwapXY));
    F32 yzero = det * yinitScale * xrcp + yinitBias;
    S64 yinit = f32_to_s64(slct(yzero, -yzero, slctFlipY));
    U32 yinc  = f32_to_u32_sat((F32)::abs(slct(dy, dx, slctSwapXY)) * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookupMask(yinit, yinc, flips, lut);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_conservative_fast(S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut) // 54 instr
{
    F32  halfPixel  = (F32)(1 << (CR_SUBPIXEL_LOG2 - 1));
    F32  yinitBias  = (F32)(1 << (31 - CR_MAXVIEWPORT_LOG2 - CR_SUBPIXEL_LOG2 * 2));
    F32  yinitScale = (F32)(1 << (32 - CR_SUBPIXEL_LOG2));
    F32  yincScale  = 65536.0f * 65536.0f;

    S32  slctFlipY  = flips << (31 - CR_FLIPBIT_FLIP_Y);
    S32  slctFlipX  = flips << (31 - CR_FLIPBIT_FLIP_X);
    S32  slctSwapXY = flips << (31 - CR_FLIPBIT_SWAP_XY);

    // Evaluate cross product.

    S32 t = ox * dy - oy * dx;
    F32 det = (F32)slct(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);

    F32 xabs = (F32)::abs(slct(dx, dy, slctSwapXY));
    F32 yabs = (F32)::abs(slct(dy, dx, slctSwapXY));
    det = det + xabs * halfPixel + yabs * halfPixel;

    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    F32 xrcp  = 1.0f / xabs;
    F32 yzero = det * yinitScale * xrcp + yinitBias;
    S64 yinit = f32_to_s64(slct(yzero, -yzero, slctFlipY));
    U32 yinc  = f32_to_u32_sat(yabs * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookupMask(yinit, yinc, flips, lut);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_lookupMask(S64 yinit, U32 yinc, U32 flips, volatile const U64* lut)
{
    // First half.

    U32 yfrac = getLo(yinit);
    U32 shape = add_clamp_0_x(getHi(yinit) + 4, 0, 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    int oct = flips & ((1 << CR_FLIPBIT_FLIP_X) | (1 << CR_FLIPBIT_SWAP_XY));
    U64 mask = *(U64*)((U8*)lut + oct + (shape << 5));

    // Second half.

    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    shape = add_clamp_0_x(getHi(yinit) + 4, __popc(shape & 15), 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    mask |= *(U64*)((U8*)lut + oct + (shape << 5) + (12 << 8));
    return (flips >= (1 << CR_FLIPBIT_COMPL)) ? ~mask : mask;
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_exact_noLUT(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S32 curr = ox * dy - oy * dx;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    return cover8x8_generateMask_noLUT(curr, dx, dy);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_conservative_noLUT(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S32 curr = ox * dy - oy * dx;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    curr += (::abs(dx) + ::abs(dy)) << (CR_SUBPIXEL_LOG2 - 1);
    return cover8x8_generateMask_noLUT(curr, dx, dy);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_generateMask_noLUT(S32 curr, S32 dx, S32 dy)
{
    curr += (dx - dy) * (7 << CR_SUBPIXEL_LOG2);
    S32 stepX = dy << (CR_SUBPIXEL_LOG2 + 1);
    S32 stepYorig = -dx - dy * 7;
    S32 stepY = stepYorig << (CR_SUBPIXEL_LOG2 + 1);

    U32 hi = isetge(curr, 0);
    U32 frac = curr + curr;
    for (int i = 62; i >= 32; i--)
        add_add_carry(frac, frac, ((i & 7) == 7) ? stepY : stepX, hi, hi, hi);

	U32 lo = 0;
    for (int i = 31; i >= 0; i--)
        add_add_carry(frac, frac, ((i & 7) == 7) ? stepY : stepX, lo, lo, lo);

	lo ^= lo >> 1,  hi ^= hi >> 1;
	lo ^= lo >> 2,  hi ^= hi >> 2;
	lo ^= lo >> 4,  hi ^= hi >> 4;
	lo ^= lo >> 8,  hi ^= hi >> 8;
	lo ^= lo >> 16, hi ^= hi >> 16;

	if (dy < 0)
    {
        lo ^= 0x55AA55AA;
        hi ^= 0x55AA55AA;
    }
	if (stepYorig < 0)
    {
        lo ^= 0xFF00FF00;
        hi ^= 0x00FF00FF;
    }
	if ((hi & 1) != 0)
		lo = ~lo;

    return combineLoHi(lo, hi);
}

//------------------------------------------------------------------------

__device__ __inline__ U32 coverMSAA_ref(int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy)
{
    S64 base = (S64)ox * dy - (S64)oy * dx;
    S64 stepX = (S64)-dy << (CR_SUBPIXEL_LOG2 - samplesLog2);
    S64 stepY = (S64)+dx << (CR_SUBPIXEL_LOG2 - samplesLog2);
    base -= ((stepX + stepY) * ((1 << samplesLog2) - 1)) >> 1;
    if (dy > 0 || (dy == 0 && dx <= 0)) base--; // exclusive

    U32 mask = 0;
    for (int i = 0; i < (1 << samplesLog2); i++)
    {
        S64 value = base;
        value += c_msaaPatterns[samplesLog2][i] * stepX;
        value += i * stepY;
        mask = slct(mask | (1 << i), mask, getHi(value));
    }
    return mask;
}

//------------------------------------------------------------------------

#define S(SAMPLES_LOG2, X, Y) \
    { \
        (X * 2 + 1 - (1 << SAMPLES_LOG2)) << (CR_SUBPIXEL_LOG2 - SAMPLES_LOG2 - 1), \
        (Y * 2 + 1 - (1 << SAMPLES_LOG2)) << (CR_SUBPIXEL_LOG2 - SAMPLES_LOG2 - 1) \
    }
FW_CUDA_CONST int c_msaaPatternsFast[4][16][2] = { CR_MSAA_PATTERNS(S) };
#undef S

__device__ __inline__ U32 coverMSAA_fast(int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy) // 37 instr for 8 samples
{
    S32 base = ox * dy - oy * dx;
    if (dy > 0 || (dy == 0 && dx <= 0)) base--; // exclusive

    U32 mask = 0;
    for (int i = 0; i < (1 << samplesLog2); i++)
    {
        S32 value = base;
        value -= c_msaaPatternsFast[samplesLog2][i][0] * dy;
        value += c_msaaPatternsFast[samplesLog2][i][1] * dx;
        mask = slct(mask | (1 << i), mask, value);
    }
    return mask;
}

//------------------------------------------------------------------------

template <class T> __device__ __inline__ void sortShared(T* ptr, int numItems)
{
    int thrInBlock = threadIdx.x + threadIdx.y * blockDim.x;
    int range = 16;

    // Use transposition sort within each 16-wide subrange.

    int base = thrInBlock * 2;
    if (base < numItems - 1)
    {
        bool tryOdd = (base < numItems - 2 && (~base & (range - 2)) != 0);
        T mid = ptr[base + 1];

        for (int iter = 0; iter < range; iter += 2)
        {
            // Evens.

            T tmp = ptr[base + 0];
            if (tmp > mid)
            {
                ptr[base + 0] = mid;
                mid = tmp;
            }

            // Odds.

            if (tryOdd)
            {
                tmp = ptr[base + 2];
                if (mid > tmp)
                {
                    ptr[base + 2] = mid;
                    mid = tmp;
                }
            }
        }
        ptr[base + 1] = mid;
    }

    // Multiple subranges => Merge hierarchically.

    for (; range < numItems; range <<= 1)
    {
        // Assuming that we would insert the current item into the other
        // subrange, use binary search to find the appropriate slot.

        __syncthreads();

        T item;
        int slot;
        if (thrInBlock < numItems)
        {
            item = ptr[thrInBlock];
            slot = (thrInBlock & -range) ^ range;
            if (slot < numItems)
            {
                T tmp = ptr[slot];
                bool inclusive = ((thrInBlock & range) != 0);
                if (tmp < item || (inclusive && tmp == item))
                {
                    for (int step = (range >> 1); step != 0; step >>= 1)
                    {
                        int probe = slot + step;
                        if (probe < numItems)
                        {
                            tmp = ptr[probe];
                            if (tmp < item || (inclusive && tmp == item))
                                slot = probe;
                        }
                    }
                    slot++;
                }
            }
        }

        // Store the item at an appropriate place.

        __syncthreads();

        if (thrInBlock < numItems)
            ptr[slot + (thrInBlock & (range * 2 - 1)) - range] = item;
    }
}

//------------------------------------------------------------------------
}
