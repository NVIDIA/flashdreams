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
#include "base/Defs.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Image;
class InputStream;
class OutputStream;

//------------------------------------------------------------------------

Image*  importBinaryImage   (InputStream& stream);
void    exportBinaryImage   (OutputStream& stream, const Image* image);

//------------------------------------------------------------------------
/*

Binary image file format v1
---------------------------

- the basic units of data are 32-bit little-endian ints and floats

BinaryImage
    0       7       struct  ImageHeader
    3       n*6     struct  array of ImageChannel (ImageHeader.numChannels)
    ?       ?       struct  image data (ImageHeader.width * ImageHeader.height * ImageHeader.bpp)
    ?

ImageHeader
    0       2       bytes   formatID (must be "BinImage")
    2       1       int     formatVersion (must be 1)
    3       1       int     width
    4       1       int     height
    5       1       int     bpp
    6       1       int     numChannels
    7

ImageChannel
    0       1       int     type (see ImageFormat::ChannelType)
    1       1       int     format (see ImageFormat::ChannelFormat)
    2       1       int     wordOfs
    3       1       int     wordSize
    4       1       int     fieldOfs
    5       1       int     fieldSize
    6

*/
//------------------------------------------------------------------------
}
