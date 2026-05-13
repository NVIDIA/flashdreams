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

#include "io/ImageLodePngIO.hpp"
#include "io/Stream.hpp"
#include "gui/Image.hpp"

#include "3rdparty/lodepng/lodepng.h"

using namespace FW;

//------------------------------------------------------------------------

Image* FW::importLodePngImage(InputStream& stream)
{
    // Read the entire input stream.

    Array<U8> dataBuffer;
    int blockSize = 4096;
    for (;;)
    {
        int pos = dataBuffer.getSize();
        dataBuffer.resize(pos + blockSize);
        int num = stream.read(dataBuffer.getPtr(pos), blockSize);
        if (num < blockSize)
        {
            dataBuffer.resize(pos + num);
            break;
        }
    }

    if (hasError())
        return NULL;

    // Decode image info.

    LodePNG::Decoder decoder;
    decoder.inspect(dataBuffer.getPtr(), dataBuffer.getSize());
    Vec2i size(decoder.getWidth(), decoder.getHeight());
    bool hasAlpha = (LodePNG_InfoColor_canHaveAlpha(&decoder.getInfoPng().color) != 0);

    if (decoder.hasError())
        setError("importLodePngImage(): LodePNG error %d!", decoder.getError());
    if (min(size) <= 0)
        setError("importLodePngImage(): Invalid image size!");
    if (hasError())
        return NULL;

    // Decode image data.

    int numBytes = size.x * size.y * ((hasAlpha) ? 4 : 3);
    std::vector<U8> pixelBuffer;
    pixelBuffer.reserve(numBytes);
    decoder.getInfoRaw().color.colorType = (hasAlpha) ? 6 : 2;
    decoder.decode(pixelBuffer, dataBuffer.getPtr(), dataBuffer.getSize());

    if (decoder.hasError())
        setError("importLodePngImage(): LodePNG error %d!", decoder.getError());
    if ((int)pixelBuffer.size() != numBytes)
        setError("importLodePngImage(): Incorrect amount of pixel data!");
    if (hasError())
        return NULL;

    // Create image.

    Image* image = new Image(size, (hasAlpha) ? ImageFormat::R8_G8_B8_A8 : ImageFormat::R8_G8_B8);
    image->getBuffer().set(&pixelBuffer[0], numBytes);
    return image;
}

//------------------------------------------------------------------------

void FW::exportLodePngImage(OutputStream& stream, const Image* image)
{
    // Select format and convert.

    FW_ASSERT(image);
    const Vec2i& size = image->getSize();
    bool hasAlpha = image->getFormat().hasChannel(ImageFormat::ChannelType_A);
    ImageFormat::ID format = (hasAlpha) ? ImageFormat::R8_G8_B8_A8 : ImageFormat::R8_G8_B8;

    Image* converted = NULL;
    if (image->getFormat().getID() != format)
    {
        converted = new Image(size, format);
        *converted = *image;
    }

    // Encode image.

    LodePNG::Encoder encoder;
    int colorType = (hasAlpha) ? 6 : 2;
    encoder.getSettings().autoLeaveOutAlphaChannel = false;
    encoder.getInfoRaw().color.colorType = colorType;
    encoder.getInfoPng().color.colorType = colorType;

    std::vector<U8> dataBuffer;
    encoder.encode(dataBuffer, (U8*)((converted) ? converted : image)->getPtr(), size.x, size.y);

    if (encoder.hasError())
        setError("exportLodePngImage(): LodePNG error %d!", encoder.getError());

    // Write to the output stream.

    if (!hasError() && !dataBuffer.empty())
        stream.write(&dataBuffer[0], (int)dataBuffer.size());

    // Clean up.

    delete converted;
}

//------------------------------------------------------------------------
