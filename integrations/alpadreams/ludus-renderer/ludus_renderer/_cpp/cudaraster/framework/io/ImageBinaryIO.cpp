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

#include "io/ImageBinaryIO.hpp"
#include "gui/Image.hpp"
#include "io/Stream.hpp"

using namespace FW;

//------------------------------------------------------------------------

Image* FW::importBinaryImage(InputStream& stream)
{
    // ImageHeader.

    char formatID[9];
    stream.readFully(formatID, 8);
    formatID[8] = '\0';
    if (String(formatID) != "BinImage")
    {
        setError("Not a binary image file!");
        return NULL;
    }

    S32 version;
    stream >> version;
    if (version != 1)
    {
        setError("Unsupported binary image version!");
        return NULL;
    }

    S32 width, height, bpp, numChannels;
    stream >> width >> height >> bpp >> numChannels;
    if (width < 0 || height < 0 || bpp < 0 || numChannels < 0)
    {
        setError("Corrupt binary image data!");
        return NULL;
    }

    // Array of ImageChannel.

    ImageFormat format;
    for (int i = 0; i < numChannels; i++)
    {
        S32 ctype, cformat;
        ImageFormat::Channel c;
        stream >> ctype >> cformat >> c.wordOfs >> c.wordSize >> c.fieldOfs >> c.fieldSize;
        if (ctype < 0 || cformat < 0 || cformat >= ImageFormat::ChannelFormat_Max ||
            c.wordOfs < 0 || (c.wordSize != 1 && c.wordSize != 2 && c.wordSize != 4) ||
            c.fieldOfs < 0 || c.fieldSize <= 0 || c.fieldOfs + c.fieldSize > c.wordSize * 8 ||
            (cformat == ImageFormat::ChannelFormat_Float && c.fieldSize != 32))
        {
            setError("Corrupt binary image data!");
            return NULL;
        }

        c.type = (ImageFormat::ChannelType)ctype;
        c.format = (ImageFormat::ChannelFormat)cformat;
        format.addChannel(c);
    }

    if (bpp != format.getBPP())
    {
        setError("Corrupt binary image data!");
        return NULL;
    }

    // Image data.

    Image* image = new Image(Vec2i(width, height), format);
    stream.readFully(image->getMutablePtr(), width * height * bpp);

    // Handle errors.

    if (hasError())
    {
        delete image;
        return NULL;
    }
    return image;
}

//------------------------------------------------------------------------

void FW::exportBinaryImage(OutputStream& stream, const Image* image)
{
    FW_ASSERT(image);
    const Vec2i& size = image->getSize();
    const ImageFormat& format = image->getFormat();
    int bpp = format.getBPP();

    // ImageHeader.

    stream.write("BinImage", 8);
    stream << (S32)1 << (S32)size.x << (S32)size.y << (S32)bpp << (S32)format.getNumChannels();

    // Array of ImageChannel.

    for (int i = 0; i < format.getNumChannels(); i++)
    {
        const ImageFormat::Channel& c = format.getChannel(i);
        stream << (S32)c.type << (S32)c.format << (S32)c.wordOfs << (S32)c.wordSize << (S32)c.fieldOfs << (S32)c.fieldSize;
    }

    // Image data.

    if (image->getStride() == size.x * bpp)
        stream.write(image->getPtr(), size.x * size.y * bpp);
    else
    for (int y = 0; y < size.y; y++)
        stream.write(image->getPtr(Vec2i(0, y)), size.x * bpp);
}

//------------------------------------------------------------------------
