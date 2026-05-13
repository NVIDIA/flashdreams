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

#include "io/ImageRawPngIO.hpp"
#include "gui/Image.hpp"
#include "io/Stream.hpp"

using namespace FW;

//------------------------------------------------------------------------

namespace FW
{

class Output
{
public:
                    Output      (OutputStream& s);
                    ~Output     (void)          {}

    Output&         operator<<  (int byte)      { m_stream << (U8)byte; m_crc = (m_crc >> 8) ^ m_crcTable[(m_crc ^ byte) & 0xFF]; return *this; }
    void            writeDWord  (U32 value)     { *this << (value >> 24) << (value >> 16) << (value >> 8) << value; }

    void            resetCRC    (U32 value)     { m_crc = value; }
    U32             getCRC      (void) const    { return m_crc; }

private:
                    Output      (const Output&); // forbidden
    Output&         operator=   (const Output&); // forbidden

private:
    OutputStream&   m_stream;
    U32             m_crcTable[256];
    U32             m_crc;
};

}

//------------------------------------------------------------------------

Output::Output(OutputStream& s)
:   m_stream    (s),
    m_crc       (0)
{
    for (int i = 0; i < 256; i++)
    {
        U32 v = i;
        for (int j = 0; j < 8; j++)
            v = (v >> 1) ^ (0xedb88320u & (~(v & 1) + 1));
        m_crcTable[i] = v;
    }
}

//------------------------------------------------------------------------

void FW::exportRawPngImage(OutputStream& stream, const Image* image)
{
    FW_ASSERT(image);

    // Convert image.

    Vec2i           size        = image->getSize();
    bool            empty       = (size.min() <= 0);
    bool            hasAlpha    = image->getFormat().hasChannel(ImageFormat::ChannelType_A);
    ImageFormat::ID format      = (hasAlpha) ? ImageFormat::R8_G8_B8_A8 : ImageFormat::R8_G8_B8;
    const Image*    source      = image;
    Image*          converted   = NULL;

    if (empty || image->getFormat().getID() != format)
    {
        size = Vec2i(max(size.x, 1), max(size.y, 1));
        converted = new Image(size, format);
        if (empty)
            converted->clear();
        else
            *converted = *image;
        source = converted;
    }

    // Write header.

    Output out(stream);
    int bpp = (hasAlpha) ? 4 : 3;
    int blockLen = size.x * bpp + 1;

    out << 0x89 << 'P' << 'N' << 'G' << 0x0D << 0x0A << 0x1A << 0x0A;
    out << 0x00 << 0x00 << 0x00 << 0x0D << 'I' << 'H' << 'D' << 'R';
    out.resetCRC(0x575e51f5);
    out.writeDWord(size.x);
    out.writeDWord(size.y);
    out << 8 << ((hasAlpha) ? 6 : 2) << 0 << 0 << 0;
    out.writeDWord(~out.getCRC());
    out.writeDWord(size.y * (blockLen + 5) + 6);
    out << 'I' << 'D' << 'A' << 'T' << 0x78 << 0x01;
    out.resetCRC(0x13e5812d);

    // Write image data.

    int adlerA = 1;
    int adlerB = 0;

    for (int y = 0; y < size.y; y++)
    {
        out << ((y == size.y - 1) ? 1 : 0);
        out << blockLen << (blockLen >> 8) << ~blockLen << (~blockLen >> 8);
        out << 0;
        adlerB = (adlerA + adlerB) % 65521;

        const U8* scanline = (const U8*)source->getPtr(Vec2i(0, y));
        for (int x = 0; x < blockLen - 1; x++)
        {
            out << scanline[x];
            adlerA = (adlerA + scanline[x]) % 65521;
            adlerB = (adlerA + adlerB) % 65521;
        }
    }

    out << (adlerB >> 8) << adlerB << (adlerA >> 8) << adlerA;

    // Write footer.

    out.writeDWord(~out.getCRC());
    out << 0x00 << 0x00 << 0x00 << 0x00;
    out << 'I' << 'E' << 'N' << 'D' << 0xAE << 0x42 << 0x60 << 0x82;
    delete converted;
}

//------------------------------------------------------------------------
