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

#include "io/ImageTargaIO.hpp"
#include "gui/Image.hpp"
#include "io/Stream.hpp"

using namespace FW;

//------------------------------------------------------------------------

Image* FW::importTargaImage(InputStream& stream)
{
    // Read header.

    int idLength = stream.readU8();     // idlength
    int cmaptype = stream.readU8();     // colourmaptype
    int datatype = stream.readU8();     // datatypecode
    stream.readU16LE();                 // colourmaporigin
    stream.readU16LE();                 // colourmaplength
    stream.readU8();                    // colourmapdepth
    stream.readU16LE();                 // x_origin
    stream.readU16LE();                 // y_origin
    int width = stream.readU16LE();     // width
    int height = stream.readU16LE();    // height
    int pixelBits = stream.readU8();    // bitsperpixel
    U32 flags = stream.readU8();        // imagedescriptor

    // Interpret header.

    int attribBits = (flags & 0x0F);
    bool rightToLeft = ((flags & 0x10) != 0);
    bool bottomToTop = ((flags & 0x20) == 0);
    int interleaving = ((flags >> 6) & 0x3);

    bool rle = false;
    switch (datatype)
    {
    case 2:     rle = false; break;
    case 10:    rle = true; break;
    default:    setError("Unsupported Targa datatype %d!", datatype); break;
    }

    ImageFormat format;
    if (pixelBits == 16 && attribBits <= 1)
        format = ImageFormat::RGBA_5551;
    else if (pixelBits == 24 && attribBits == 0)
        format = ImageFormat::R8_G8_B8;
    else if (pixelBits == 32 && (attribBits == 8 || attribBits == 0))
        format = ImageFormat::R8_G8_B8_A8;
    else
        setError("Unsupported Targa color format %d/%d!", pixelBits, attribBits);

    if (cmaptype != 0)
        setError("Targa colormaps not supported!");
    if (interleaving != 0)
        setError("Targa interleaving not supported!");

    if (hasError())
        return NULL;

    // Skip identification field.

    for (int i = 0; i < idLength; i++)
        stream.readU8();

    // Decode image data.

    Image* image = new Image(Vec2i(width, height), format);
    int bpp = image->getBPP();
    int stride = (int)image->getStride();
    U8* data = image->getMutablePtr();

    if (!rle)
        stream.readFully(data, stride * height);
    else
    {
        U8* ptr = data;
        U8* imageEnd = ptr + stride * height;
        while (ptr < imageEnd)
        {
            int header = stream.readU8();
            U8* end = ptr + ((header & 0x7F) + 1) * bpp;
            if (end > imageEnd)
            {
                setError("Corrupt Targa image data!");
                break;
            }

            if ((header & 0x80) == 0)
            {
                stream.readFully(ptr, (int)(end - ptr));
                ptr = end;
            }
            else
            {
                stream.readFully(ptr, bpp);
                ptr += bpp;
                while (ptr < end)
                    *ptr++ = ptr[-bpp];
            }
        }
    }

    // Handle errors.

    if (hasError())
    {
        delete image;
        return NULL;
    }

    // Convert color format.

    U8* ptr = data;
    switch (bpp)
    {
    case 2:
        for (int i = width * height; i > 0; i--)
        {
            U32 v = ptr[0] | (ptr[1] << 8);
            *(U16*)ptr = (U16)((v << 1) | ((attribBits == 0) ? 1 : (v >> 15)));
            ptr += bpp;
        }
        break;

    case 3:
    case 4:
        for (int i = width * height; i > 0; i--)
        {
            swap(ptr[0], ptr[2]);
            ptr += bpp;
        }
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    // Flip if needed.

    if (rightToLeft)
        image->flipX();
    if (bottomToTop)
        image->flipY();
    return image;
}

//------------------------------------------------------------------------

void FW::exportTargaImage(OutputStream& stream, const Image* image)
{
    // Select format and convert.

    FW_ASSERT(image);
    Vec2i size = image->getSize();
    bool empty = (size.min() <= 0);
    bool hasAlpha = image->getFormat().hasChannel(ImageFormat::ChannelType_A);
    const Image* source = image;
    Image* converted = NULL;

    if (empty ||
        image->getFormat().getID() != ImageFormat::ABGR_8888 ||
        image->getStride() != size.x * 4)
    {
        size = Vec2i(max(size.x, 1), max(size.y, 1));
        converted = new Image(size, ImageFormat::ABGR_8888);
        if (empty)
            converted->clear();
        else
            *converted = *image;
        source = converted;
    }

    // Write header.

    stream.writeU8(0);                          // idlength
    stream.writeU8(0);                          // colourmaptype
    stream.writeU8(10);                         // datatypecode
    stream.writeU16LE(0);                       // colourmaporigin
    stream.writeU16LE(0);                       // colourmaplength
    stream.writeU8(0);                          // colourmapdepth
    stream.writeU16LE(0);                       // x_origin
    stream.writeU16LE(0);                       // y_origin
    stream.writeU16LE(size.x);                  // width
    stream.writeU16LE(size.y);                  // height
    stream.writeU8((hasAlpha) ? 32 : 24);       // bitsperpixel
    stream.writeU8((hasAlpha) ? 0x28 : 0x20);   // imagedescriptor

    // Compress image data.

    const U32* ptr = (const U32*)source->getPtr();
    const U32* end = ptr + size.x * size.y;

    while (ptr < end)
    {
        const U32* start = ptr;
        while (ptr < end && ptr - start < 128 && ptr[0] == start[0])
            ptr++;

        if (ptr - start >= 2)
        {
            stream.writeU8((U32)(ptr - start) + 127);
            start = ptr - 1;
        }
        else
        {
            while (ptr < end && ptr - start < 128 && (ptr + 1 == end || ptr[1] != ptr[0]))
                ptr++;
            stream.writeU8((U32)(ptr - start) - 1);
        }

        while (start < ptr)
        {
            U32 v = *start++;
            stream.writeU8(v >> 16);
            stream.writeU8(v >> 8);
            stream.writeU8(v);
            if (hasAlpha)
                stream.writeU8(v >> 24);
        }
    }

    // Clean up.

    delete converted;
}

//------------------------------------------------------------------------
