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

#include "io/ImageTiffIO.hpp"
#include "gui/Image.hpp"
#include "base/Hash.hpp"
#include "io/Stream.hpp"

using namespace FW;

//------------------------------------------------------------------------

namespace FW
{

class Input
{
public:
                    Input       (InputStream& s, bool le, int ofs)  : m_stream(s), m_littleEndian(le), m_data(NULL, ofs), m_ofs(ofs) {}
                    ~Input      (void)                              {}

    U32             tell        (void) const                        { return m_ofs; }
    void            seek        (U32 ofs)                           { fill(ofs); m_ofs = ofs; }

    const U8*       read        (int num)                           { m_ofs += num; fill(m_ofs); return m_data.getPtr(m_ofs - num); }
    U8              readU8      (void)                              { return *read(1); }
    U16             readU16     (void)                              { const U8* p = read(2); return (U16)((m_littleEndian) ? (p[0] | (p[1] << 8)) : ((p[0] << 8) | p[1])); }
    U32             readU32     (void)                              { const U8* p = read(4); return (m_littleEndian) ? (p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24)) : ((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]); }

private:
    void            fill        (U32 ofs)                           { U32 old = m_data.getSize(); if (ofs > old) { m_data.resize(ofs); m_stream.readFully(m_data.getPtr(old), ofs - old); } }

private:
                    Input       (const Input&); // forbidden
    Input&          operator=   (const Input&); // forbidden

private:
    InputStream&    m_stream;
    bool            m_littleEndian;
    Array<U8>       m_data;
    U32             m_ofs;
};

enum TiffType
{
    Tiff_Byte       = 1,
    Tiff_Ascii      = 2,
    Tiff_Short      = 3,
    Tiff_Long       = 4,
    Tiff_Rational   = 5
};

}

//------------------------------------------------------------------------

Image* FW::importTiffImage(InputStream& stream)
{
    // Detect endianess and check format.

    U8 endianTag = stream.readU8();
    Input in(stream, (endianTag == 'I'), 1);
    if ((endianTag != 'I' && endianTag != 'M') || in.readU8() != endianTag || in.readU16() != 42)
        setError("Not a TIFF file!");

    // Read directory header.

    U32 dirOfs = in.readU32();
    in.seek(dirOfs);
    int numEntries = in.readU16();
    if (!dirOfs || !numEntries)
        setError("Corrupt TIFF directory!");

    // Read directory entries.

    Hash<U32, Array<U32> > entries;
    for (int i = 0; i < numEntries && !hasError(); i++)
    {
        U16 tag   = in.readU16();
        U16 type  = in.readU16();
        U32 count = in.readU32();
        U32 ofs   = in.readU32();

        // Check type and count.

        int typeSize;
        switch (type)
        {
        case 1:     typeSize = 1; break;    // BYTE
        case 3:     typeSize = 2; break;    // SHORT
        case 4:     typeSize = 4; break;    // LONG
        default:    typeSize = 0; break;
        }

        if (!typeSize)
            continue;

        // Seek to the value.

        U32 oldOfs = in.tell();
        if (typeSize * count <= 4)
            in.seek(oldOfs - 4);
        else
            in.seek(ofs);

        // Read value.

        Array<U32>& data = entries.add(tag);
        data.resize(count);
        for (int j = 0; j < (int)count; j++)
        {
            switch (typeSize)
            {
            case 1:     data[j] = in.readU8(); break;
            case 2:     data[j] = in.readU16(); break;
            case 4:     data[j] = in.readU32(); break;
            default:    FW_ASSERT(false); break;
            }
        }
        in.seek(oldOfs);
    }

    // Find relevant entries and check their sizes.

    const Array<U32>* width         = entries.search(256);  // ImageWidth
    const Array<U32>* height        = entries.search(257);  // ImageLength
    const Array<U32>* numBits       = entries.search(258);  // BitsPerSample
    const Array<U32>* compression   = entries.search(259);  // Compression
    const Array<U32>* photometric   = entries.search(262);  // PhotometricInterpretation
    const Array<U32>* stripOfs      = entries.search(273);  // StripOffsets
    const Array<U32>* numChannels   = entries.search(277);  // SamplesPerPixel
    const Array<U32>* stripBytes    = entries.search(279);  // StripByteCounts
    const Array<U32>* predictor     = entries.search(317);  // Predictor
    const Array<U32>* extraSamples  = entries.search(338);  // ExtraSamples
    const Array<U32>* sampleFormat  = entries.search(339);  // SampleFormat

    if (!width          || width->getSize()         != 1 ||
        !height         || height->getSize()        != 1 ||
        !numBits        || numBits->getSize()       == 0 ||
        !compression    || compression->getSize()   != 1 ||
        !photometric    || photometric->getSize()   != 1 ||
        !stripOfs       || stripOfs->getSize()      == 0 ||
        !numChannels    || numChannels->getSize()   != 1 ||
        !stripBytes     || stripBytes->getSize()    != stripOfs->getSize() ||
        (predictor      && predictor->getSize()     != 1) ||
        (extraSamples   && extraSamples->getSize()  != 1) ||
        (sampleFormat   && sampleFormat->getSize()  != 1))
    {
        setError("Corrupt TIFF directory!");
    }

    if (hasError())
        return NULL;

    // Interpret size.

    Vec2i size(width->get(0), height->get(0));
    if (size.min() <= 0)
        setError("Invalid TIFF size!");

    // Interpret compression.

    bool packBits = false;
    switch (compression->get(0))
    {
    case 1:     packBits = false; break;
    case 32773: packBits = true; break;
    default:    setError("Unsupported TIFF compression %d!", compression->get(0)); break;
    }

    if (predictor && predictor->get(0) != 1)
        setError("Unsupported TIFF predictor %d!", predictor->get(0));

    // Read color format.

    bool floats = false;
    if (sampleFormat)
    {
        switch (sampleFormat->get(0))
        {
        case 1:     floats = false; break;
        case 3:     floats = true; break;
        default:    setError("Unsupported TIFF sample format %d!", sampleFormat->get(0)); break;
        }
    }

    U32 photo = photometric->get(0);
    int numColor = numChannels->get(0);
    int numAlpha = (extraSamples) ? extraSamples->get(0) : 0;
    if (numBits->getSize() != numColor + numAlpha)
        setError("Invalid TIFF color format!");
    for (int i = 0; i < numBits->getSize(); i++)
        if (numBits->get(i) != ((floats) ? 32u : 8u))
            setError("Invalid TIFF color format!");

    // Interpret color format.

    ImageFormat format;
    switch (photo)
    {
    case 1: // MinIsBlack
        if ((numColor == 0 && numAlpha == 1) || (numColor == 1 && numAlpha == 0))
            format = (floats) ? ImageFormat::A_F32 : ImageFormat::A8;
        else
            setError("Unsupported TIFF monochrome color format!");
        break;

    case 2: // RGB
        if (numColor == 3 && numAlpha == 0)
            format = (floats) ? ImageFormat::RGB_Vec3f : ImageFormat::R8_G8_B8;
        else if ((numColor == 3 && numAlpha == 1) || (numColor == 4 && numAlpha == 0))
            format = (floats) ? ImageFormat::RGBA_Vec4f : ImageFormat::R8_G8_B8_A8;
        else
            setError("Unsupported TIFF RGB color format!");
        break;

    default:
        setError("Unsupported TIFF photometric interpretation %d!", photo);
    }

    // Error => fail.

    if (hasError())
        return NULL;

    // Create image.

    Image* image = new Image(size, format);
    U8* data = image->getMutablePtr();
    const U8* dataEnd = data + image->getStride() * size.y;

    // Read each strip of image data.

    U8* dst = data;
    for (int i = 0; i < stripOfs->getSize() && !hasError(); i++)
    {
        int size = stripBytes->get(i);
        in.seek(stripOfs->get(i));
        const U8* src = in.read(size);
        const U8* srcEnd = src + size;

        // Uncompressed => just read the bytes.

        if (!packBits)
        {
            dst += size;
            if (dst > dataEnd)
                break;
            memcpy(dst - size, src, size);
            continue;
        }

        // PackBits => read each RLE packet.

        while (src < srcEnd)
        {
            int n = *src++;
            if (n < 128)
            {
                n++;
                dst += n;
                src += n;
                if (dst > dataEnd || src > srcEnd)
                    break;
                memcpy(dst - n, src - n, n);
            }
            else if (n > 128)
            {
                n = 257 - n;
                dst += n;
                src++;
                if (dst > dataEnd || src > srcEnd)
                    break;
                memset(dst - n, src[-1], n);
            }
        }

        if (dst > dataEnd || src != srcEnd)
            setError("Corrupt TIFF image data!");
    }

    if (dst != dataEnd)
        setError("Corrupt TIFF image data!");

    // Float-based format => fix endianess.

    if (floats)
        for (U8* ptr = data; ptr < dataEnd; ptr += 4)
            if (endianTag == 'I')
                *(U32*)ptr = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
            else
                *(U32*)ptr = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];

    // Handle errors.

    if (hasError())
    {
        delete image;
        return NULL;
    }
    return image;
}

//------------------------------------------------------------------------

void FW::exportTiffImage(OutputStream& stream, const Image* image)
{
    // Select format and convert.
    // - RGBA and floats are not well-supported.
    // - PackBits compression does not make sense for RGB data.

    FW_ASSERT(image);
    Vec2i size = image->getSize();
    bool empty = (size.min() <= 0);
    const Image* source = image;
    Image* converted = NULL;

    if (empty ||
        image->getFormat().getID() != ImageFormat::R8_G8_B8 ||
        (int)image->getStride() != size.x * 4)
    {
        size = Vec2i(max(size.x, 1), max(size.y, 1));
        converted = new Image(size, ImageFormat::R8_G8_B8);
        if (empty)
            converted->clear();
        else
            *converted = *image;
        source = converted;
    }

    int numBytes = (int)source->getStride() * size.y;

    // Write.

    stream.writeU8('I');                    // 0x00: endianess tag
    stream.writeU8('I');                    // 0x01: (little-endian)
    stream.writeU16LE(42);                  // 0x02: format identifier
    stream.writeU32LE(0x08);                // 0x04: IFD offset
    stream.writeU16LE(12);                  // 0x08: number of IFD entries

    stream.writeU16LE(256);                 // 0x0A: Tag = ImageWidth
    stream.writeU16LE(4);                   // 0x0C: Type = LONG
    stream.writeU32LE(1);                   // 0x0E: Count
    stream.writeU32LE(size.x);              // 0x12: Value

    stream.writeU16LE(257);                 // 0x16: Tag = ImageLength
    stream.writeU16LE(4);                   // 0x18: Type = LONG
    stream.writeU32LE(1);                   // 0x1A: Count
    stream.writeU32LE(size.y);              // 0x1E: Value

    stream.writeU16LE(258);                 // 0x22: Tag = BitsPerSample
    stream.writeU16LE(3);                   // 0x24: Type = SHORT
    stream.writeU32LE(3);                   // 0x26: Count
    stream.writeU32LE(0x9E);                // 0x2A: Value offset

    stream.writeU16LE(259);                 // 0x2E: Tag = Compression
    stream.writeU16LE(3);                   // 0x30: Type = SHORT
    stream.writeU32LE(1);                   // 0x32: Count
    stream.writeU32LE(1);                   // 0x36: Value = No compression

    stream.writeU16LE(262);                 // 0x3A: Tag = PhotometricInterpretation
    stream.writeU16LE(3);                   // 0x3C: Type = SHORT
    stream.writeU32LE(1);                   // 0x3E: Count
    stream.writeU32LE(2);                   // 0x42: Value = RGB

    stream.writeU16LE(273);                 // 0x46: Tag = StripOffsets
    stream.writeU16LE(4);                   // 0x48: Type = LONG
    stream.writeU32LE(1);                   // 0x4A: Count
    stream.writeU32LE(0xAC);                // 0x4E: Value

    stream.writeU16LE(277);                 // 0x52: Tag = SamplesPerPixel
    stream.writeU16LE(3);                   // 0x54: Type = SHORT
    stream.writeU32LE(1);                   // 0x56: Count
    stream.writeU32LE(3);                   // 0x5A: Value

    stream.writeU16LE(278);                 // 0x5E: Tag = RowsPerStrip
    stream.writeU16LE(4);                   // 0x60: Type = LONG
    stream.writeU32LE(1);                   // 0x62: Count
    stream.writeU32LE(size.y);              // 0x66: Value

    stream.writeU16LE(279);                 // 0x6A: Tag = StripByteCounts
    stream.writeU16LE(4);                   // 0x6C: Type = LONG
    stream.writeU32LE(1);                   // 0x6E: Count
    stream.writeU32LE(numBytes);            // 0x72: Value

    stream.writeU16LE(282);                 // 0x76: Tag = XResolution
    stream.writeU16LE(5);                   // 0x78: Type = RATIONAL
    stream.writeU32LE(1);                   // 0x7A: Count
    stream.writeU32LE(0xA4);                // 0x7E: Value offset

    stream.writeU16LE(283);                 // 0x82: Tag = YResolution
    stream.writeU16LE(5);                   // 0x84: Type = RATIONAL
    stream.writeU32LE(1);                   // 0x86: Count
    stream.writeU32LE(0xA4);                // 0x8A: Value offset

    stream.writeU16LE(296);                 // 0x8E: Tag = ResolutionUnit
    stream.writeU16LE(3);                   // 0x90: Type = SHORT
    stream.writeU32LE(1);                   // 0x92: Count
    stream.writeU32LE(2);                   // 0x96: Value = Inch

    stream.writeU32LE(0);                   // 0x9A: next IFD offset
    stream.writeU16LE(8);                   // 0x9E: BitsPerSample[0]
    stream.writeU16LE(8);                   // 0xA0: BitsPerSample[1]
    stream.writeU16LE(8);                   // 0xA2: BitsPerSample[2]
    stream.writeU32LE(72);                  // 0xA4: Resolution numerator
    stream.writeU32LE(1);                   // 0xA8: Resolution denominator

    stream.write(source->getPtr(), numBytes); // 0xAC: image data

    // Clean up.

    delete converted;
}

//------------------------------------------------------------------------
