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
#include "base/DLLImports.hpp"
#include "base/Math.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Multisample patterns used by NVIDIA GF100.
// c_msaaPatterns[log2(samplesPerPixel)][sampleY] = sampleX

#define CR_MSAA_PATTERNS(S) \
    /* MODE_1X1 (CENTER_1) */                       { S(0,0,0) }, \
    /* MODE_2X1 (DIAGONAL_CENTERED_2) */            { S(1,0,0), S(1,1,1) }, \
    /* MODE_2X2 (SQUARE_ROTATED_4) */               { S(2,1,0), S(2,3,1), S(2,0,2), S(2,2,3) }, \
    /* MODE_4X2_D3D (DX10.1 8-sample pattern) */    { S(3,7,0), S(3,2,1), S(3,4,2), S(3,0,3), S(3,6,4), S(3,3,5), S(3,1,6), S(3,5,7) }, \
    /* MODE_4X4 (NROOK_16) */                     /*{ S(4,1,0), S(4,8,1), S(4,4,2), S(4,11,3), S(4,15,4), S(4,7,5), S(4,3,6), S(4,12,7), S(4,0,8), S(4,9,9), S(4,5,10), S(4,13,11), S(4,2,12), S(4,10,13), S(4,6,14), S(4,14,15) },*/ \

#define S(SAMPLES_LOG2, X, Y) X
FW_CUDA_CONST int c_msaaPatterns[4][16] = { CR_MSAA_PATTERNS(S) };
#undef S

//------------------------------------------------------------------------

#define CR_HASH_MAGIC (0x9e3779b9u)

#define CR_JENKINS_MIX(a, b, c)   \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15);

//------------------------------------------------------------------------

FW_CUDA_FUNC int    selectMSAACentroid      (int samplesLog2, U32 sampleMask);
FW_CUDA_FUNC U32    cover8x8_selectFlips    (S32 dx, S32 dy);
FW_CUDA_FUNC int    clipPolygonWithPlane    (F32* baryOut, const F32* baryIn, int numIn, F32 v0, F32 v1, F32 v2);
FW_CUDA_FUNC int    clipTriangleWithFrustum (F32* bary, const F32* v0, const F32* v1, const F32* v2, const F32* d1, const F32* d2);
FW_CUDA_FUNC U32    encodeDepth             (U32 depth);
FW_CUDA_FUNC U32    decodeDepth             (U32 depth);

//------------------------------------------------------------------------

#if FW_CUDA

#if FW_64
#   define PTX_PTR(P) "l"(P)
#else
#   define PTX_PTR(P) "r"(P)
#endif

__device__ __inline__ U32   getLo                   (U64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ S32   getLo                   (S64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ U32   getHi                   (U64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ S32   getHi                   (S64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ U64   combineLoHi             (U32 lo, U32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ S64   combineLoHi             (S32 lo, S32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskLe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }
__device__ __inline__ int   findLeadingOne          (U32 v)                 { U32 r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
__device__ __inline__ bool  singleLane              (void)                  { return ((__ballot(true) & getLaneMaskLt()) == 0); }

__device__ __inline__ void  add_add_carry           (U32& rlo, U32 alo, U32 blo, U32& rhi, U32 ahi, U32 bhi) { U64 r = combineLoHi(alo, ahi) + combineLoHi(blo, bhi); rlo = getLo(r); rhi = getHi(r); }
__device__ __inline__ S32   f32_to_s32_sat          (F32 a)                 { S32 v; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat          (F32 a)                 { U32 v; asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat_rmi      (F32 a)                 { U32 v; asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u8_sat           (F32 a)                 { U32 v; asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ S64   f32_to_s64              (F32 a)                 { S64 v; asm("cvt.rni.s64.f32 %0, %1;" : "=l"(v) : "f"(a)); return v; }
__device__ __inline__ S32   add_s16lo_s16lo			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16lo			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16lo_s16hi			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16hi			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16lo			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16lo			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16hi			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16hi			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16lo			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16lo			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16hi			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16hi			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b0					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b0, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b1					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b2					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b2, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b3					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b3, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   vmad_b0					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b0_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_mask8				(U32 a, U32 b)			{ U32 v; U32 z=0; asm("vadd.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ U32   sub_mask8				(U32 a, U32 b)			{ U32 v; U32 z=0; asm("vsub.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ S32   max_max					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_min					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   max_add					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_add					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_add					(U32 a, U32 b, U32 c)	{ U32 v; asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   sub_add					(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_sub					(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
__device__ __inline__ S32   add_clamp_0_x			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b0			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b2			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b2, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   prmt					(U32 a, U32 b, U32 c)   { U32 v; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   u32lo_sext              (U32 a)                 { U32 v; asm("cvt.s16.u32 %0, %1;" : "=r"(v) : "r"(a)); return v; }
__device__ __inline__ U32   slct                    (U32 a, U32 b, S32 c)   { U32 v; asm("slct.u32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   slct                    (S32 a, S32 b, S32 c)   { S32 v; asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ F32   slct                    (F32 a, F32 b, S32 c)   { F32 v; asm("slct.f32.s32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "r"(c)); return v; }
__device__ __inline__ U32   isetge                  (S32 a, S32 b)          { U32 v; asm("set.ge.u32.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ F64   rcp_approx              (F64 a)                 { F64 v; asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(v) : "d"(a)); return v; }
__device__ __inline__ F32   fma_rm                  (F32 a, F32 b, F32 c)   { F32 v; asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "f"(c)); return v; }
__device__ __inline__ U32   idiv_fast               (U32 a, U32 b);

__device__ __inline__ U32   cachedLoad              (const U32* p)          { U32 v; asm("ld.global.ca.u32 %0, [%1];" : "=r"(v) : PTX_PTR(p)); return v; }
__device__ __inline__ uint2 cachedLoad              (const uint2* p)        { uint2 v; asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : PTX_PTR(p)); return v; }
__device__ __inline__ uint4 cachedLoad              (const uint4* p)        { uint4 v; asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : PTX_PTR(p)); return v; }
__device__ __inline__ void  cachedStore             (U32* p, U32 v)         { asm("st.global.wb.u32 [%0], %1;" :: PTX_PTR(p), "r"(v)); }
__device__ __inline__ void  cachedStore             (uint2* p, uint2 v)     { asm("st.global.wb.v2.u32 [%0], {%1, %2};" :: PTX_PTR(p), "r"(v.x), "r"(v.y)); }
__device__ __inline__ void  cachedStore             (uint4* p, uint4 v)     { asm("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: PTX_PTR(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }

__device__ __inline__ U32   uncachedLoad            (const U32* p)          { U32 v; asm("ld.global.cg.u32 %0, [%1];" : "=r"(v) : PTX_PTR(p)); return v; }
__device__ __inline__ uint2 uncachedLoad            (const uint2* p)        { uint2 v; asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : PTX_PTR(p)); return v; }
__device__ __inline__ uint4 uncachedLoad            (const uint4* p)        { uint4 v; asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : PTX_PTR(p)); return v; }
__device__ __inline__ void  uncachedStore           (U32* p, U32 v)         { asm("st.global.cg.u32 [%0], %1;" :: PTX_PTR(p), "r"(v)); }
__device__ __inline__ void  uncachedStore           (uint2* p, uint2 v)     { asm("st.global.cg.v2.u32 [%0], {%1, %2};" :: PTX_PTR(p), "r"(v.x), "r"(v.y)); }
__device__ __inline__ void  uncachedStore           (uint4* p, uint4 v)     { asm("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" :: PTX_PTR(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }

__device__ __inline__ U32   toABGR					(float4 color);
__device__ __inline__ U32   blendABGR               (U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor); // Uses 8 highest bits of xxxFactor.
__device__ __inline__ U32   blendABGRClamp          (U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor); // Clamps the result to 255.

__device__ __inline__ uint3 setupPleq               (float3 values, int2 v0, int2 d1, int2 d2, F32 areaRcp, int samplesLog2);

__device__ __inline__ U64   cover8x8_exact_ref          (S32 ox, S32 oy, S32 dx, S32 dy); // reference implementation
__device__ __inline__ U64   cover8x8_conservative_ref   (S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U64   cover8x8_generateMask_ref   (S64 curr, S64 stepX, S64 stepY);
__device__ __inline__ bool  cover8x8_missesTile         (S32 ox, S32 oy, S32 dx, S32 dy);

__device__ __inline__ void  cover8x8_setupLUT           (volatile U64* lut);
__device__ __inline__ U64   cover8x8_exact_fast         (S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut); // Assumes viewport <= 2^11, subpixels <= 2^4, no guardband.
__device__ __inline__ U64   cover8x8_conservative_fast  (S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut);
__device__ __inline__ U64   cover8x8_lookupMask         (S64 yinit, U32 yinc, U32 flips, volatile const U64* lut);

__device__ __inline__ U64   cover8x8_exact_noLUT        (S32 ox, S32 oy, S32 dx, S32 dy); // optimized reference implementation, does not require look-up table
__device__ __inline__ U64   cover8x8_conservative_noLUT (S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U64   cover8x8_generateMask_noLUT (S32 curr, S32 dx, S32 dy);

__device__ __inline__ U32   coverMSAA_ref               (int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U32   coverMSAA_fast              (int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy);

template <class T> __device__ __inline__ void sortShared(T* ptr, int numItems); // Assumes that numItems <= threadsInBlock. Must sync before & after the call.

#endif

//------------------------------------------------------------------------

FW_CUDA_FUNC int selectMSAACentroid(int samplesLog2, U32 sampleMask)
{
    int numSamples = 1 << samplesLog2;
    if (sampleMask == 0 || sampleMask == (1u << numSamples) - 1)
        return -1;

    int bestSample = -1;
    int bestDist = FW_S32_MAX;
    for (int i = 0; i < numSamples; i++)
    {
        if ((sampleMask & (1 << i)) != 0)
        {
            int dist = sqr(c_msaaPatterns[samplesLog2][i] * 2 + 1 - numSamples) + sqr(i * 2 + 1 - numSamples);
            if (dist < bestDist)
            {
                bestSample = i;
                bestDist = dist;
            }
        }
    }
    return bestSample;
}

//------------------------------------------------------------------------

FW_CUDA_FUNC U32 cover8x8_selectFlips(S32 dx, S32 dy) // 10 instr
{
    U32 flips = 0;
    if (dy > 0 || (dy == 0 && dx <= 0))
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y) ^ (1 << CR_FLIPBIT_COMPL);
    if (dx > 0)
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y);
    if (::abs(dx) < ::abs(dy))
        flips ^= (1 << CR_FLIPBIT_SWAP_XY) ^ (1 << CR_FLIPBIT_FLIP_Y);
    return flips;
}

//------------------------------------------------------------------------

FW_CUDA_FUNC int clipPolygonWithPlane(F32* baryOut, const F32* baryIn, int numIn, F32 v0, F32 v1, F32 v2)
{
    int numOut = 0;
    if (numIn >= 3)
    {
        int ai = (numIn - 1) * 2;
        F32 av = v0 + v1 * baryIn[ai + 0] + v2 * baryIn[ai + 1];
        for (int bi = 0; bi < numIn * 2; bi += 2)
        {
            F32 bv = v0 + v1 * baryIn[bi + 0] + v2 * baryIn[bi + 1];
            if (av * bv < 0.0f)
            {
                F32 bc = av / (av - bv);
                F32 ac = 1.0f - bc;
                baryOut[numOut + 0] = baryIn[ai + 0] * ac + baryIn[bi + 0] * bc;
                baryOut[numOut + 1] = baryIn[ai + 1] * ac + baryIn[bi + 1] * bc;
                numOut += 2;
            }
            if (bv >= 0.0f)
            {
                baryOut[numOut + 0] = baryIn[bi + 0];
                baryOut[numOut + 1] = baryIn[bi + 1];
                numOut += 2;
            }
            ai = bi;
            av = bv;
        }
    }
    return (numOut >> 1);
}

//------------------------------------------------------------------------
// bary = &Vec2f[9] (output)
// v0 = &Vec4f(clipPos0)
// v1 = &Vec4f(clipPos1)
// v2 = &Vec4f(clipPos2)
// d1 = &Vec4f(clipPos1 - clipPos0)
// d2 = &Vec4f(clipPos2 - clipPos0)

FW_CUDA_FUNC int clipTriangleWithFrustum(F32* bary, const F32* v0, const F32* v1, const F32* v2, const F32* d1, const F32* d2)
{
    int num = 3;
    bary[0] = 0.0f, bary[1] = 0.0f;
    bary[2] = 1.0f, bary[3] = 0.0f;
    bary[4] = 0.0f, bary[5] = 1.0f;

    if ((v0[3] < fabsf(v0[0])) | (v1[3] < fabsf(v1[0])) | (v2[3] < fabsf(v2[0])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[0], d1[3] + d1[0], d2[3] + d2[0]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[0], d1[3] - d1[0], d2[3] - d2[0]);
    }
    if ((v0[3] < fabsf(v0[1])) | (v1[3] < fabsf(v1[1])) | (v2[3] < fabsf(v2[1])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[1], d1[3] + d1[1], d2[3] + d2[1]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[1], d1[3] - d1[1], d2[3] - d2[1]);
    }
    if ((v0[3] < fabsf(v0[2])) | (v1[3] < fabsf(v1[2])) | (v2[3] < fabsf(v2[2])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[2], d1[3] + d1[2], d2[3] + d2[2]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[2], d1[3] - d1[2], d2[3] - d2[2]);
    }
    return num;
}

//------------------------------------------------------------------------

FW_CUDA_FUNC U32 encodeDepth(U32 depth)
{
    F64 v = (F64)depth;
    v /= 65536.0 * 65536.0 - 1.0;;
    v = clamp(v, 0.0, 1.0);
    v *= (F64)(CR_DEPTH_MAX - CR_DEPTH_MIN);
    v += (F64)CR_DEPTH_MIN;
    return (U32)v;
}

//------------------------------------------------------------------------

FW_CUDA_FUNC U32 decodeDepth(U32 depth)
{
    F64 v = (F64)depth;
    v -= (F64)CR_DEPTH_MIN;
    v *= 1.0 / (F64)(CR_DEPTH_MAX - CR_DEPTH_MIN);
    v = clamp(v, 0.0, 1.0);
    v *= 65536.0 * 65536.0 - 1.0;
    return (U32)v;
}

//------------------------------------------------------------------------
}
