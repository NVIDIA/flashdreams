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

#include "PixelPipe.hpp"
#include "PrivateDefs.hpp"
#include "Util.inl"

namespace FW
{
//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

extern "C" __constant__ CRParams    c_crParams;
extern "C" __device__   CRAtomics   g_crAtomics;
extern "C" __constant__ S32         c_profLaunchIdx;
extern "C" __constant__ CUdeviceptr c_profData;

extern "C" texture<float4, 1>   t_vertexBuffer;
extern "C" texture<uint4, 1>    t_triHeader;
extern "C" texture<uint4, 1>    t_triData;

extern "C" surface<void, 2>     s_colorBuffer;
extern "C" surface<void, 2>     s_depthBuffer;

//------------------------------------------------------------------------
// FragmentShaderBase.
//------------------------------------------------------------------------

__device__ __inline__ Vec4f FragmentShaderBase::getVaryingAtVertex(int varyingIdx, int vertIdx) const
{
    float4 t = tex1Dfetch(t_vertexBuffer, vertIdx * (m_vertexBytes / sizeof(Vec4f)) + varyingIdx + 1);
    return Vec4f(t.x, t.y, t.z, t.w);
}

//------------------------------------------------------------------------

__device__ __inline__ Vec4f FragmentShaderBase::interpolateVarying(int varyingIdx, const Vec3f& bary) const
{
    Vec4f v0 = getVaryingAtVertex(varyingIdx, m_vertIdx.x);
    Vec4f v1 = getVaryingAtVertex(varyingIdx, m_vertIdx.y);
    Vec4f v2 = getVaryingAtVertex(varyingIdx, m_vertIdx.z);
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

//------------------------------------------------------------------------
// Common shaders.
//------------------------------------------------------------------------

__device__ __inline__ void GouraudShader::run(void)
{
    m_color = toABGR(interpolateVarying(0, m_centroid));
}

//------------------------------------------------------------------------

__device__ __inline__ void BlendSrcOver::run(void)
{
    m_color = blendABGR(m_src, m_dst, m_src, ~m_src, m_src, ~m_src);
}

//------------------------------------------------------------------------

__device__ __inline__ void BlendAdditive::run(void)
{
    m_color = blendABGRClamp(m_src, m_dst, ~0, ~0, ~0, ~0);
}

//------------------------------------------------------------------------
// Profiling.
//------------------------------------------------------------------------

#ifndef CR_PROFILING_MODE
#   define CR_PROFILING_MODE ProfilingMode_Default
#endif

#if (CR_PROFILING_MODE == ProfilingMode_Counters)
#   define CR_COUNT(ID, NUM, DENOM)             incProfilingCounter((int)&((CRProfCounterOrder*)NULL)->ID, NUM, DENOM)
#   define CR_COUNT_LARGE_GRID(ID, NUM, DENOM)  incProfilingCounterLargeGrid((int)&((CRProfCounterOrder*)NULL)->ID, NUM, DENOM)
#else
#   define CR_COUNT(ID, NUM, DENOM)
#   define CR_COUNT_LARGE_GRID(ID, NUM, DENOM)
#endif

#if (CR_PROFILING_MODE == ProfilingMode_Timers)
#   define CR_TIMER_INIT()                      U32 timerTotal = 0; timerTotal &= 0
#   define CR_TIMER_IN(ID)                      timerTotal -= queryProfilingTimer((int)(S64)&((CRProfTimerOrder*)NULL)->ID, 0)
#   define CR_TIMER_OUT(ID)                     timerTotal += queryProfilingTimer((int)(S64)&((CRProfTimerOrder*)NULL)->ID, 0)
#   define CR_TIMER_OUT_DEP(ID, DEP)            timerTotal += queryProfilingTimer((int)(S64)&((CRProfTimerOrder*)NULL)->ID, DEP)
#   define CR_TIMER_SYNC()                      __syncthreads()
#   define CR_TIMER_DEINIT()                    writeProfilingTimer(timerTotal)
#   define CR_TIMER_DEINIT_LARGE_GRID()         writeProfilingTimerLargeGrid(timerTotal)
#else
#   define CR_TIMER_INIT()                      U32 timerTotal = 0; timerTotal &= 0
#   define CR_TIMER_IN(ID)
#   define CR_TIMER_OUT(ID)
#   define CR_TIMER_OUT_DEP(ID, DEP)
#   define CR_TIMER_SYNC()
#   define CR_TIMER_DEINIT()
#   define CR_TIMER_DEINIT_LARGE_GRID()
#endif

//------------------------------------------------------------------------

__device__ __inline__ void incProfilingCounter(int counterIdx, S64 num, S64 denom)
{
    int numCounters = sizeof(CRProfCounterOrder) - 1;
    int warpIdx = threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y);
    S64* ptr = &((S64*)c_profData)[(warpIdx * numCounters + counterIdx) * 64 + threadIdx.x];

    if (num != 0)
        ptr[0] += num;
    if (denom != 0)
        ptr[32] += denom;
}

//------------------------------------------------------------------------

__device__ __inline__ void incProfilingCounterLargeGrid(int counterIdx, S64 num, S64 denom)
{
    __shared__ volatile U64 s_warpTotal[48];
    bool isLeader = singleLane();
    int numCounters = sizeof(CRProfCounterOrder) - 1;
    U64* ptr = &((U64*)c_profData)[(threadIdx.y * numCounters + counterIdx) * 64];

    s_warpTotal[threadIdx.y] = 0;
    atomicAdd((U64*)&s_warpTotal[threadIdx.y], num);
    if (isLeader && s_warpTotal[threadIdx.y] != 0)
        atomicAdd(&ptr[0], s_warpTotal[threadIdx.y]);

    s_warpTotal[threadIdx.y] = 0;
    atomicAdd((U64*)&s_warpTotal[threadIdx.y], denom);
    if (isLeader && s_warpTotal[threadIdx.y] != 0)
        atomicAdd(&ptr[32], s_warpTotal[threadIdx.y]);
}

//------------------------------------------------------------------------

__constant__ U32 c_zero = 0;

__device__ __inline__ U32 queryProfilingTimer(int timerIdx, U32 dep)
{
    return ((dep & c_zero) == 0 && c_profLaunchIdx == timerIdx && singleLane()) ? clock() : 0;
}

__device__ __inline__ U32 queryProfilingTimer(int timerIdx, S32 dep) { return queryProfilingTimer(timerIdx, (U32)dep); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, F32 dep) { return queryProfilingTimer(timerIdx, (U32)__float_as_int(dep)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, bool dep) { return queryProfilingTimer(timerIdx, (U32)(dep ? 2 : 1)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const int2& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const int3& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y | dep.z)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const int4& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y | dep.z | dep.w)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const uint2& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const uint3& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y | dep.z)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const uint4& dep) { return queryProfilingTimer(timerIdx, (U32)(dep.x | dep.y | dep.z | dep.w)); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const float2& dep) { return queryProfilingTimer(timerIdx, (U32)(__float_as_int(dep.x) | __float_as_int(dep.y))); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const float3& dep) { return queryProfilingTimer(timerIdx, (U32)(__float_as_int(dep.x) | __float_as_int(dep.y) | __float_as_int(dep.z))); }
__device__ __inline__ U32 queryProfilingTimer(int timerIdx, const float4& dep) { return queryProfilingTimer(timerIdx, (U32)(__float_as_int(dep.x) | __float_as_int(dep.y) | __float_as_int(dep.z) | __float_as_int(dep.w))); }

//------------------------------------------------------------------------

__device__ __inline__ void writeProfilingTimer(U32 timerTotal)
{
    int numTimers = sizeof(CRProfTimerOrder) - 1;
    int warpIdx = threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y);
    ((U32*)c_profData)[(warpIdx * numTimers + c_profLaunchIdx) * 32 + threadIdx.x] += timerTotal;
}

//------------------------------------------------------------------------

__device__ __inline__ void writeProfilingTimerLargeGrid(U32 timerTotal)
{
    __shared__ volatile U32 s_warpTotal[48];
    s_warpTotal[threadIdx.y] = 0;
    atomicAdd((U32*)&s_warpTotal[threadIdx.y], timerTotal);

    if (threadIdx.x == 0)
    {
        int numTimers = sizeof(CRProfTimerOrder) - 1;
        atomicAdd(&((U32*)c_profData)[(threadIdx.y * numTimers + c_profLaunchIdx) * 32], s_warpTotal[threadIdx.y]);
    }
}

//------------------------------------------------------------------------
// Stage implementations.
//------------------------------------------------------------------------

#include "TriangleSetup.inl"
#include "BinRaster.inl"
#include "CoarseRaster.inl"
#include "FineRaster.inl"

//------------------------------------------------------------------------
// Pixel pipe definition.
//------------------------------------------------------------------------

#define CR_DEFINE_PIXEL_PIPE(PIPE_NAME, VERTEX_STRUCT, FRAGMENT_SHADER, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS) \
    \
    extern "C" __global__ void __launch_bounds__(CR_SETUP_WARPS * 32, CR_SETUP_OPT_BLOCKS) PIPE_NAME ## _triangleSetup(void) \
    { \
        FW::triangleSetupImpl<VERTEX_STRUCT, SAMPLES_LOG2, RENDER_MODE_FLAGS>(); \
    } \
    \
    extern "C" __global__ void __launch_bounds__(CR_BIN_WARPS * 32, 1) PIPE_NAME ## _binRaster(void) \
    { \
        FW::binRasterImpl(); \
    } \
    \
    extern "C" __global__ void __launch_bounds__(CR_COARSE_WARPS * 32, 1) PIPE_NAME ## _coarseRaster(void) \
    { \
        FW::coarseRasterImpl(); \
    } \
    \
    extern "C" __global__ void __launch_bounds__(CR_FINE_OPT_WARPS * 32, 1) PIPE_NAME ## _fineRaster(void) \
    { \
        if (SAMPLES_LOG2 == 0) \
            FW::fineRasterImpl_SingleSample<VERTEX_STRUCT, FRAGMENT_SHADER, BLEND_SHADER, RENDER_MODE_FLAGS>(); \
        else \
            FW::fineRasterImpl_MultiSample<VERTEX_STRUCT, FRAGMENT_SHADER, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS>(); \
    } \
    extern "C" __constant__ PixelPipeSpec PIPE_NAME ## _spec = \
    { \
        /* samplesLog2 */       SAMPLES_LOG2, \
        /* vertexStructSize */  (int)sizeof(VERTEX_STRUCT), \
        /* renderModeFlags */   RENDER_MODE_FLAGS, \
        /* profilingMode */     CR_PROFILING_MODE, \
        /* blendShaderName */   #BLEND_SHADER, \
    };

//------------------------------------------------------------------------
}
