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
#include "gpu/GLContext.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Render target for CudaRaster, visible in OpenGL as a 2D texture.
//------------------------------------------------------------------------

class CudaSurface
{
public:
    enum Format
    {
        Format_R8_G8_B8_A8  = 0,    // U8 red, U8 green, U8 blue, U8 alpha
        Format_Depth32,             // U32 depth

        Format_Max
    };

public:
                        CudaSurface     (const Vec2i& size, Format format, int numSamples = 1);
                        ~CudaSurface    (void);

    const Vec2i&        getSize         (void) const    { return m_size; }          // Original size specified in the constructor.
    const Vec2i&        getRoundedSize  (void) const    { return m_roundedSize; }   // Rounded to full 8x8 tiles.
    const Vec2i&        getTextureSize  (void) const    { return m_textureSize; }   // 8x8 tiles are replicated horizontally for MSAA.
    Format              getFormat       (void) const    { return m_format; }
    int                 getNumSamples   (void) const    { return m_numSamples; }
    int                 getSamplesLog2  (void) const    { return popc8(m_numSamples - 1); } // log2(numSamples)

    GLuint              getGLTexture    (void);             // Invalidates the CUDA array.
    CUarray             getCudaArray    (void);             // Invalidates the GL texture.

    void                resolveToScreen (GLContext* gl);    // Resolves MSAA and writes pixels into the current GL render target.

private:
                        CudaSurface     (const CudaSurface&); // forbidden
    CudaSurface&        operator=       (const CudaSurface&); // forbidden

private:
    Vec2i               m_size;
    Vec2i               m_roundedSize;
    Vec2i               m_textureSize;
    Format              m_format;
    S32                 m_numSamples;

    GLuint              m_glTexture;
    CUgraphicsResource  m_cudaResource;
    bool                m_isMapped;
    CUarray             m_cudaArray;
};

//------------------------------------------------------------------------
}
