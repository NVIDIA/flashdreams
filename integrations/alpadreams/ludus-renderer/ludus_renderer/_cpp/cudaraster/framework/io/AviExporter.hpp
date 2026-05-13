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
#include "io/File.hpp"
#include "gui/Image.hpp"

namespace FW
{

//------------------------------------------------------------------------

class AviExporter
{
public:
                    AviExporter         (const String& fileName, Vec2i size, int fps);
                    ~AviExporter        (void);

    const Image&    getFrame            (void) const    { return m_frame; }
    Image&          getFrame            (void)          { return m_frame; }
    int             getFPS              (void) const    { return m_fps; }

    void            exportFrame         (void);
    void            flush               (void);

private:
    void            setTag              (int ofs, const char* tag);
    void            setS32              (int ofs, S32 value);
    void            writeHeader         (void);

private:
                    AviExporter         (const AviExporter&); // forbidden
    AviExporter&    operator=           (const AviExporter&); // forbidden

private:
    File            m_file;
    Vec2i           m_size;
    Image           m_frame;
    S32             m_fps;

    S32             m_lineBytes;
    S32             m_frameBytes;
    S32             m_numFrames;
    Array<U8>       m_buffer;
};

//------------------------------------------------------------------------
}
