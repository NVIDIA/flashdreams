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
#include "gui/Image.hpp"
#include "base/Hash.hpp"
#include "base/Thread.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Texture
{
private:
    struct Data
    {
        S32         refCount;
        Spinlock    refCountLock;

        String      id;
        bool        isInHash;
        Image*      image;
        GLuint      glTexture;
        CUarray     cudaArray;
        Texture*    nextMip;
    };

public:
                    Texture         (void)                                  : m_data(NULL) {}
                    Texture         (const Texture& other)                  : m_data(NULL) { set(other); }
                    Texture         (Image* image, const String& id = ""); // takes ownership of the image
                    ~Texture        (void)                                  { if (m_data) unreferData(m_data); }

    static Texture  find            (const String& id);
    static Texture  import          (const String& fileName);

    bool            exists          (void) const                            { return (m_data && m_data->image && m_data->image->getSize().min() > 0); }
    String          getID           (void) const                            { return (m_data) ? m_data->id : ""; }
    const Image*    getImage        (void) const                            { return (m_data) ? m_data->image : NULL; }
    Vec2i           getSize         (void) const                            { return (exists()) ? getImage()->getSize() : 0; }

    void            clear           (void)                                  { if (m_data) unreferData(m_data); m_data = NULL; }
    void            set             (const Texture& other);

    GLuint          getGLTexture    (const ImageFormat::ID desiredFormat = ImageFormat::ID_Max, bool generateMipmaps = true) const;
    CUarray         getCudaArray    (const ImageFormat::ID desiredFormat = ImageFormat::ID_Max) const;
    Texture         getMipLevel     (int level) const;

    Texture&        operator=       (const Texture& other)                  { set(other); return *this; }
    bool            operator==      (const Texture& other) const            { return (m_data == other.m_data); }
    bool            operator!=      (const Texture& other) const            { return (m_data != other.m_data); }

private:
    static Data*    findData        (const String& id);
    static Data*    createData      (const String& id);
    static void     referData       (Data* data);
    static void     unreferData     (Data* data);

private:
    static Hash<String, Data*>* s_hash;

    Data*           m_data;
};

//------------------------------------------------------------------------
}
