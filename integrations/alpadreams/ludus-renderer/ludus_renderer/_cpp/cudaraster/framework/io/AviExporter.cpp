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

#include "io/AviExporter.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define AVI_HEADER_SIZE 0xe0

//------------------------------------------------------------------------

AviExporter::AviExporter(const String& fileName, Vec2i size, int fps)
:   m_file  (fileName, File::Create),
    m_size  (size),
    m_frame (size, ImageFormat::R8_G8_B8),
    m_fps   (fps)
{
    FW_ASSERT(size.min() > 0 && fps > 0);
    m_lineBytes  = (m_size.x * 3 + 3) & -4;
    m_frameBytes = m_lineBytes * m_size.y;
    m_numFrames  = 0;
    writeHeader();
}

//------------------------------------------------------------------------

AviExporter::~AviExporter(void)
{
}

//------------------------------------------------------------------------

void AviExporter::exportFrame(void)
{
    // Increment frame counter and allocate buffer.

    m_numFrames++;
    m_buffer.resize(8 + m_frameBytes + 8 + m_numFrames * 16);

    // Output frame header.

    int ofs = 0;
    setTag(ofs + 0x00, "00db");
    setS32(ofs + 0x04, m_frameBytes);
    ofs += 8;

    // Output each scanline.

    for (int y = 0; y < m_size.y; y++)
    {
        const U8* src = m_frame.getPtr(Vec2i(0, y));
        U8* dst = m_buffer.getPtr(ofs);
        for (int x = m_size.x; x > 0; x--)
        {
            *dst++ = src[2]; // B
            *dst++ = src[1]; // G
            *dst++ = src[0]; // R
            src += 3;
        }
        ofs += m_lineBytes;
    }

    // Output index header.

    setTag(ofs + 0x00, "idx1");             // AVIOLDINDEX.fcc
    setS32(ofs + 0x04, m_numFrames * 16);   // cb
    ofs += 8;

    // Output the offset of each frame.

    for (int i = 0; i < m_numFrames; i++)
    {
        S32 dwOffset = 4 + i * (m_frameBytes + 8);
        setTag(ofs + 0x00, "00db");         // dwChunkId
        setS32(ofs + 0x04, 0x10);           // dwFlags
        setS32(ofs + 0x08, dwOffset);       // dwOffset
        setS32(ofs + 0x0c, m_frameBytes);   // dwSize
        ofs += 16;
    }

    // Write the buffer.

    FW_ASSERT(ofs == m_buffer.getSize());
    m_file.write(m_buffer.getPtr(), m_buffer.getSize());

    // Update the header, since some fields depend on numFrames.

    m_file.seek(0);
    writeHeader();

    // Seek to the beginning of the next frame.

    m_file.seek(AVI_HEADER_SIZE + (S64)m_numFrames * (m_frameBytes + 8));
}

//------------------------------------------------------------------------

void AviExporter::flush(void)
{
    m_file.flush();
}

//------------------------------------------------------------------------

void AviExporter::setTag(int ofs, const char* tag)
{
    FW_ASSERT(tag);
    FW_ASSERT(ofs >= 0 && ofs + 4 <= m_buffer.getSize());
    U8* ptr = m_buffer.getPtr(ofs);

    ptr[0] = tag[0];
    ptr[1] = tag[1];
    ptr[2] = tag[2];
    ptr[3] = tag[3];
}

//------------------------------------------------------------------------

void AviExporter::setS32(int ofs, S32 value)
{
    FW_ASSERT(ofs >= 0 && ofs + 4 <= m_buffer.getSize());
    U8* ptr = m_buffer.getPtr(ofs);

    ptr[0] = (U8)value;
    ptr[1] = (U8)(value >> 8);
    ptr[2] = (U8)(value >> 16);
    ptr[3] = (U8)(value >> 24);
}

//------------------------------------------------------------------------

void AviExporter::writeHeader(void)
{
    S32 dwMicroSecPerFrame    = 1000000 / m_fps;
    S32 dwMaxBytesPerSec      = m_frameBytes * m_fps;
    S32 dwSuggestedBufferSize = m_frameBytes + 8;
    S32 rcFrame               = m_size.x | (m_size.y << 16);
    S32 riffSize              = AVI_HEADER_SIZE + m_numFrames * (m_frameBytes + 24);
    S32 moviSize              = 4 + m_numFrames * (m_frameBytes + 8);

    // Initialize to zero.

    m_buffer.resize(AVI_HEADER_SIZE);
    memset(m_buffer.getPtr(), 0, m_buffer.getSize());

    // Write non-zero fields.

    setTag(0x00, "RIFF");
    setS32(0x04, riffSize);
    setTag(0x08, "AVI ");
    setTag(0x0c, "LIST");
    setS32(0x10, 0xd4 - 0x14);
    setTag(0x14, "hdrl");
    setTag(0x18, "avih");                   // AVIMAINHEADER
    setS32(0x1c, 0x58 - 0x20);
    setS32(0x20, dwMicroSecPerFrame);       // dwMicroSecPerFrame
    setS32(0x24, dwMaxBytesPerSec);         // dwMaxBytesPerSec
    setS32(0x2c, 0x810);                    // dwFlags
    setS32(0x30, m_numFrames);              // AVIMAINHEADER.dwTotalFrames
    setS32(0x38, 1);                        // dwStreams
    setS32(0x3c, dwSuggestedBufferSize);    // dwSuggestedBufferSize
    setS32(0x40, m_size.x);                 // dwWidth
    setS32(0x44, m_size.y);                 // dwHeight

    setTag(0x58, "LIST");
    setS32(0x5c, 0xd4 - 0x60);
    setTag(0x60, "strl");
    setTag(0x64, "strh");                   // AVISTREAMINFO
    setS32(0x68, 0xa4 - 0x6c);
    setTag(0x6c, "vids");                   // fccType
    setTag(0x70, "DIB ");                   // fccHandler
    setS32(0x80, 1);                        // dwScale
    setS32(0x84, m_fps);                    // dwRate
    setS32(0x8c, m_numFrames);              // AVISTREAMINFO.dwLength
    setS32(0x90, dwSuggestedBufferSize);    // dwSuggestedBufferSize
    setS32(0x94, -1);                       // dwQuality
    setS32(0x98, m_frameBytes);             // dwSampleSize
    setS32(0xa0, rcFrame);                  // rcFrame.right/bottom

    setTag(0xa4, "strf");                   // BITMAPINFOHEADER
    setS32(0xa8, 0xd4 - 0xac);
    setS32(0xac, 0xd4 - 0xac);              // biSize
    setS32(0xb0, m_size.x);                 // biWidth
    setS32(0xb4, -m_size.y);                // biHeight
    setS32(0xb8, 0x00180001);               // biPlanes, biBitCount
    setS32(0xc0, m_frameBytes);             // biSizeImage

    setTag(0xd4, "LIST");
    setS32(0xd8, moviSize);
    setTag(0xdc, "movi");

    // Write.

    m_file.write(m_buffer.getPtr(), m_buffer.getSize());
}

//------------------------------------------------------------------------
