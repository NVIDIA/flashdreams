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

#include "io/ImageBmpIO.hpp"
#include "gui/Image.hpp"
#include "io/Stream.hpp"

using namespace FW;

//------------------------------------------------------------------------

Image* FW::importBmpImage(InputStream& stream)
{
    // Read header.

    int fileType = stream.readU16LE();      // bfType
    stream.readU32LE();                     // bfSize
    stream.readU16LE();                     // bfReserved1
    stream.readU16LE();                     // bfReserved2
    int dataStart = stream.readU32LE();     // bfOffBits
    int infoSize = stream.readU32LE();      // biSize
    int width = stream.readU32LE();         // biWidth
    int height = stream.readU32LE();        // biHeight
    stream.readU16LE();                     // biPlanes
    int bpp = stream.readU16LE();           // biBitCount
    int compression = stream.readU32LE();   // biCompression
    stream.readU32LE();                     // biSizeImage
    stream.readU32LE();                     // biXPelsPerMeter
    stream.readU32LE();                     // biYPelsPerMeter
    int paletteSize = stream.readU32LE();   // biClrUsed
    stream.readU32LE();                     // biClrImportant

    // Interpret header.

    if (fileType != 19778)
        setError("Not a BMP file!");
    if (infoSize < 40)
        setError("Invalid BMP header size!");
    if (width < 1 || height < 1)
        setError("Invalid BMP dimensions!");

    bool formatOk;
    switch (bpp)
    {
    case 1:     formatOk = (compression == 0); break;
    case 4:     formatOk = (compression == 0 || compression == 2); break;
    case 8:     formatOk = (compression == 0 || compression == 1); break;
    case 24:    formatOk = (compression == 0); break;
    default:    formatOk = false; break;
    }
    if (!formatOk)
        setError("Unsupported BMP format %d/%d!", bpp, compression);

    if (bpp == 24)
        paletteSize = 0;
    else if (paletteSize == 0)
        paletteSize = 1 << bpp;
    else if (paletteSize < 1 || paletteSize > (1 << bpp))
        setError("Invalid BMP palette size!");

    int headerEnd = 54;
    int paletteStart = infoSize + 14;
    int paletteEnd = paletteStart + paletteSize * 4;
    if (dataStart < paletteEnd)
        setError("Invalid BMP data offset!");

    if (hasError())
        return NULL;

    // Read palette.

    Array<U8> palette(NULL, paletteSize * 4);
    for (int i = headerEnd; i < paletteStart; i++)
        stream.readU8();
    stream.read(palette.getPtr(), palette.getNumBytes());

    // Decode each scanline.

    Image* image = new Image(Vec2i(width, height), ImageFormat::R8_G8_B8);
    Array<S32> indices(NULL, width);
    for (int i = paletteEnd; i < dataStart; i++)
        stream.readU8();

    for (int y = height - 1; y >= 0; y--)
    {
        U8* ptr = (U8*)image->getMutablePtr(Vec2i(0, y));
        int align = 0;

        // Raw color data.

        if (bpp == 24)
        {
            for (int x = 0; x < width; x++)
            {
                ptr[2] = stream.readU8();
                ptr[1] = stream.readU8();
                ptr[0] = stream.readU8();
                ptr += 3;
            }
            align = width * 3;
        }

        // Palette indices.

        else
        {
            // 2-bit palette.

            if (bpp == 1)
            {
                for (int x = 0; x < width; x += 8)
                {
                    U8 v = stream.readU8();
                    for (int i = 0; i < 8 && x + i < width; i++)
                    {
                        indices[x + i] = v >> 7;
                        v <<= 1;
                    }
                }
                align = (width + 7) >> 3;
            }

            // 4-bit palette, uncompressed.

            else if (bpp == 4 && !compression)
            {
                for (int x = 0; x < width; x += 2)
                {
                    U8 v = stream.readU8();
                    indices[x] = v >> 4;
                    if (x + 1 < width)
                        indices[x + 1] = v & 0xF;
                }
                align = (width + 1) >> 1;
            }

            // 8-bit palette, uncompressed.

            else if (bpp == 8 && !compression)
            {
                for (int x = 0; x < width; x++)
                    indices[x] = stream.readU8();
                align = width;
            }

            // 4-bit palette, RLE compression.

            else if (bpp == 4 && compression)
            {
                int x = 0;
                bool ok = false;
                for (;;)
                {
                    U8 first = stream.readU8();
                    U8 second = stream.readU8();

                    if (first != 0)
                    {
                        if (x + first > width)
                            break;
                        for (int i = 0; i < first; i++)
                            indices[x++] = ((i & 1) == 0) ? (second >> 4) : (second & 0xF);
                    }
                    else if (second < 3)
                    {
                        ok = (second == 0);
                        break;
                    }
                    else
                    {
                        if (x + second > width)
                            break;
                        for (int i = 0; i < second; i += 2)
                        {
                            U8 v = stream.readU8();
                            indices[x++] = v >> 4;
                            if (i + 1 < second)
                                indices[x++] = v & 0xF;
                        }
                        if (((second - 1) & 2) == 0)
                            stream.readU8();
                    }
                }

                if (!ok || x != width)
                    setError("Corrupt BMP 4-bit RLE data!");
            }

            // 8-bit palette, RLE compression.

            else if (bpp == 8 && compression)
            {
                int x = 0;
                bool ok = false;
                for (;;)
                {
                    U8 first = stream.readU8();
                    U8 second = stream.readU8();

                    if (first != 0)
                    {
                        if (x + first > width)
                            break;
                        for (int i = 0; i < first; i++)
                            indices[x++] = second;
                    }
                    else if (second < 3)
                    {
                        ok = (second == 0);
                        break;
                    }
                    else
                    {
                        if (x + second > width)
                            break;
                        for (int i = 0; i < second; i++)
                            indices[x++] = stream.readU8();
                        if ((second & 1) != 0)
                            stream.readU8();
                    }
                }

                if (!ok || x != width)
                    setError("Corrupt BMP 8-bit RLE data!");
            }

            for (int x = 0; x < width; x++)
            {
                if (indices[x] < 0 || indices[x] >= paletteSize)
                {
                    setError("Invalid BMP color index!");
                    break;
                }
                const U8* p = &palette[indices[x] * 4];
                *ptr++ = p[2];
                *ptr++ = p[1];
                *ptr++ = p[0];
            }
        }

        // Padding.

        while ((align & 3) != 0)
        {
            stream.readU8();
            align++;
        }
    }

    // Footer.

    if ((bpp == 4 || bpp == 8) && compression)
        if (stream.readU8() != 0 || stream.readU8() != 1)
            setError("Corrupt BMP 8-bit RLE data!");

    // Handle errors.

    if (hasError())
    {
        delete image;
        return NULL;
    }
    return image;
}

//------------------------------------------------------------------------

void FW::exportBmpImage(OutputStream& stream, const Image* image)
{
    // Select format and convert.

    FW_ASSERT(image);
    Vec2i size = image->getSize();
    bool empty = (size.min() <= 0);
    const Image* source = image;
    Image* converted = NULL;

    if (empty || image->getFormat().getID() != ImageFormat::R8_G8_B8)
    {
        size = Vec2i(max(size.x, 1), max(size.y, 1));
        converted = new Image(size, ImageFormat::R8_G8_B8);
        if (empty)
            converted->clear();
        else
            *converted = *image;
        source = converted;
    }

    // Write header.

    int lineBytes = (size.x * 3 + 3) & -4;
    int fileBytes = lineBytes * size.y + 54;

    stream.writeU16LE(19778);       // bfType
    stream.writeU32LE(fileBytes);   // bfSize
    stream.writeU16LE(0);           // bfReserved1
    stream.writeU16LE(0);           // bfReserved2
    stream.writeU32LE(54);          // bfOffBits
    stream.writeU32LE(40);          // biSize
    stream.writeU32LE(size.x);      // biWidth
    stream.writeU32LE(size.y);      // biHeight
    stream.writeU16LE(1);           // biPlanes
    stream.writeU16LE(24);          // biBitCount
    stream.writeU32LE(0);           // biCompression
    stream.writeU32LE(0);           // biSizeImage
    stream.writeU32LE(0);           // biXPelsPerMeter
    stream.writeU32LE(0);           // biYPelsPerMeter
    stream.writeU32LE(0);           // biClrUsed
    stream.writeU32LE(0);           // biClrImportant

    // Write image data.

    for (int y = size.y - 1; y >= 0; y--)
    {
        const U8* ptr = (const U8*)image->getPtr(Vec2i(0, y));
        for (int x = 0; x < size.x; x++)
        {
            stream.writeU8(ptr[2]);
            stream.writeU8(ptr[1]);
            stream.writeU8(ptr[0]);
            ptr += 3;
        }
        for (int x = size.x * 3; x < lineBytes; x++)
            stream.writeU8(0);
    }

    // Clean up.

    delete converted;
}

//------------------------------------------------------------------------
