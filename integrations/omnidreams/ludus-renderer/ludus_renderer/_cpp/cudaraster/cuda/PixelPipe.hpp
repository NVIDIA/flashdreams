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
#include "Util.hpp"

//------------------------------------------------------------------------
// CudaRaster device-side public interface.
//------------------------------------------------------------------------

namespace FW
{
//------------------------------------------------------------------------
// Flags to specify the rendering mode.
//------------------------------------------------------------------------

enum
{
    RenderModeFlag_EnableDepth  = 1 << 0,   // Enable depth test and depth write.
    RenderModeFlag_EnableLerp   = 1 << 1,   // Enable varying interpolation.
    RenderModeFlag_EnableQuads  = 1 << 2,   // Enable numerical derivatives in fragment shader. Degrades performance.
};

//------------------------------------------------------------------------
// Shaded vertex base class.
//------------------------------------------------------------------------

struct ShadedVertexBase
{
    Vec4f   clipPos;

    // Subclass can add a Vec4f for each varying.
    // Other data types are forbidden.
};

//------------------------------------------------------------------------
// Fragment shader base class.
//------------------------------------------------------------------------

class FragmentShaderBase
{
public:
    __device__ __inline__ Vec4f getVaryingAtVertex  (int varyingIdx, int vertIdx) const;
    __device__ __inline__ Vec4f interpolateVarying  (int varyingIdx, const Vec3f& bary) const;

    // Numerical derivatives (only valid when RenderModeFlag_EnableQuads is set).

#if FW_CUDA
    __device__ __inline__ F32   dFdx                (F32 v) const { m_shared[threadIdx.x] = v; return m_shared[threadIdx.x | 1] - m_shared[threadIdx.x & ~1]; }
    __device__ __inline__ F32   dFdy                (F32 v) const { m_shared[threadIdx.x] = v; return m_shared[threadIdx.x | 2] - m_shared[threadIdx.x & ~2]; }
    __device__ __inline__ Vec2f dFdx                (const Vec2f& v) const { return Vec2f(dFdx(v.x), dFdx(v.y)); }
    __device__ __inline__ Vec2f dFdy                (const Vec2f& v) const { return Vec2f(dFdy(v.x), dFdy(v.y)); }
    __device__ __inline__ Vec3f dFdx                (const Vec3f& v) const { return Vec3f(dFdx(v.x), dFdx(v.y), dFdx(v.z)); }
    __device__ __inline__ Vec3f dFdy                (const Vec3f& v) const { return Vec3f(dFdy(v.x), dFdy(v.y), dFdy(v.z)); }
    __device__ __inline__ Vec4f dFdx                (const Vec4f& v) const { return Vec4f(dFdx(v.x), dFdx(v.y), dFdx(v.z), dFdx(v.w)); }
    __device__ __inline__ Vec4f dFdy                (const Vec4f& v) const { return Vec4f(dFdy(v.x), dFdy(v.y), dFdy(v.z), dFdy(v.w)); }
#endif

    // Override by the subclass:

    __device__ __inline__ void  run                 (void) {}

public:
    // Inputs.

    S32     m_triIdx;       // Triangle index.
    Vec3i   m_vertIdx;      // Vertex indices.
    Vec2i   m_pixelPos;     // Integer pixel position.
    S32     m_vertexBytes;  // sizeof(ShadedVertexClass)
    volatile F32* m_shared; // 32 entries for the warp.

    Vec3f   m_center;       // Barycentrics at pixel center.
    Vec3f   m_centerDX;     // dFdx(m_center)
    Vec3f   m_centerDY;     // dFdy(m_center)

    Vec3f   m_centroid;     // Barycentrics at triangle centroid.
    Vec3f   m_centroidDX;   // dFdx(m_center)
    Vec3f   m_centroidDY;   // dFdy(m_center)

    // Outputs.

    U32     m_color;        // ABGR_8888.
    bool    m_discard;      // True to cull the fragment.
};

//------------------------------------------------------------------------
// Blend shader base class.
//------------------------------------------------------------------------

class BlendShaderBase
{
public:
    // Override by the subclass:

    __device__ __inline__ bool  needsDst    (void)  { return true; } // Must be a constant.
    __device__ __inline__ void  run         (void)  {}

public:
    // Inputs.

    S32     m_triIdx;       // Triangle index.
    Vec2i   m_pixelPos;     // Integer pixel position.
    S32     m_sampleIdx;    // MSAA sample index within the pixel.
    U32     m_src;          // Color from fragment shader.
    U32     m_dst;          // Color from framebuffer.

    // Outputs.

    U32     m_color;        // Blended color.
    bool    m_writeColor;   // False to disable color write.
};

//------------------------------------------------------------------------
// Common shaders.
//------------------------------------------------------------------------

struct GouraudVertex : ShadedVertexBase
{
    Vec4f   color;          // Varying 0.
};

//------------------------------------------------------------------------

class GouraudShader : public FragmentShaderBase
{
public:
    __device__ __inline__ void  run         (void);
};

//------------------------------------------------------------------------

class BlendReplace : public BlendShaderBase // dst = src
{
public:
    __device__ __inline__ bool  needsDst    (void)  { return false; }
    __device__ __inline__ void  run         (void)  { m_color = m_src; }
};

//------------------------------------------------------------------------

class BlendSrcOver : public BlendShaderBase // dst = lerp(dst, src, src.a)
{
public:
    __device__ __inline__ void  run         (void);
};

//------------------------------------------------------------------------

class BlendAdditive : public BlendShaderBase // dst += src
{
public:
    __device__ __inline__ void  run         (void);
};

//------------------------------------------------------------------------

class BlendDepthOnly : public BlendShaderBase // dst = dst
{
public:
    __device__ __inline__ bool  needsDst    (void)  { return false; }
    __device__ __inline__ void  run         (void)  { m_writeColor = false; }
};

//------------------------------------------------------------------------
// Pixel pipe definition.
//------------------------------------------------------------------------
/*
// Compiling device-side code is up to the user. Shaders and rendering
// mode are selected by defining one or more pixel pipes. Once compiled,
// the pipes may be used on the host side through CudaRaster::setPixelPipe().

#include "PixelPipe.inl"

CR_DEFINE_PIXEL_PIPE(PipeName, ShadedVertexClass, FragmentShaderClass, BlendShaderClass, SamplesLog2, RenderModeFlags)
CR_DEFINE_PIXEL_PIPE(AnotherPipeName, ...)

// PipeName             = Identifier string for setPixelPipe().
// ShadedVertexClass    = Name of the vertex struct, e.g. GouraudVertex.
// FragmentShaderClass  = Name of the fragment shader class, e.g. GouraudShader.
// BlendShaderClass     = Name of the blend shader class, e.g. BlendReplace.
// SamplesLog2          = Base-2 logarithm of samples per pixel.
// RenderModeFlags      = Logical OR of RenderModeFlag_XXX.
*/
//------------------------------------------------------------------------
// Profiling.
//------------------------------------------------------------------------
/*
// To select the type of information returned by CudaRaster::getProfilingInfo(),
// define CR_PROFILING_MODE before including PixelPipe.inl. Example:

#define CR_PROFILING_MODE ProfilingMode_Counters
#include "PixelPipe.inl"
*/

#define ProfilingMode_Default   0   // Performance, memory footprint.
#define ProfilingMode_Counters  1   // Internal counters. Degrades performance significantly.
#define ProfilingMode_Timers    2   // Internal timing breakdown. Degrades performance significantly.

#define ProfilingMode_First     ProfilingMode_Default
#define ProfilingMode_Last      ProfilingMode_Timers

//------------------------------------------------------------------------
}
