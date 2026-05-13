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

#include "3d/Texture.hpp"
#include "gpu/CudaModule.hpp"

using namespace FW;

//------------------------------------------------------------------------

Hash<String, Texture::Data*>* Texture::s_hash = NULL;

//------------------------------------------------------------------------

Texture::Texture(Image* image, const String& id)
{
    m_data = createData(id);
    m_data->image = image;
}

//------------------------------------------------------------------------

Texture Texture::find(const String& id)
{
    Texture tex;
    tex.m_data = findData(id);
    return tex;
}

//------------------------------------------------------------------------

Texture Texture::import(const String& fileName)
{
    Texture tex;
    tex.m_data = findData(fileName);
    if (!tex.m_data)
    {
        tex.m_data = createData(fileName);
        tex.m_data->image = importImage(fileName);
    }
    return tex;
}

//------------------------------------------------------------------------

void Texture::set(const Texture& other)
{
    Data* old = m_data;
    m_data = other.m_data;
    if (m_data)
        referData(m_data);
    if (old)
        unreferData(old);
}

//------------------------------------------------------------------------

GLuint Texture::getGLTexture(ImageFormat::ID desiredFormat, bool generateMipmaps) const
{
    if (!m_data)
        return 0;
    if (m_data->glTexture == 0 && m_data->image)
        m_data->glTexture = m_data->image->createGLTexture(desiredFormat, generateMipmaps);
    return m_data->glTexture;
}

//------------------------------------------------------------------------

CUarray Texture::getCudaArray(const ImageFormat::ID desiredFormat) const
{
    if (!m_data)
        return NULL;
    if (!m_data->cudaArray && m_data->image)
        m_data->cudaArray = m_data->image->createCudaArray(desiredFormat);
    return m_data->cudaArray;
}

//------------------------------------------------------------------------

Texture Texture::getMipLevel(int level) const
{
    const Texture* curr = this;
    for (int i = 0; i < level && curr->exists(); i++)
    {
        if (!curr->m_data->nextMip)
        {
            Image* scaled = curr->m_data->image->downscale2x();
            if (!scaled)
                break;
            curr->m_data->nextMip = new Texture(scaled);
        }
        curr = curr->m_data->nextMip;
    }
    return *curr;
}

//------------------------------------------------------------------------

Texture::Data* Texture::findData(const String& id)
{
    Data** found = (s_hash) ? s_hash->search(id) : NULL;
    if (!found)
        return NULL;

    referData(*found);
    return *found;
}

//------------------------------------------------------------------------

Texture::Data* Texture::createData(const String& id)
{
    Data* data      = new Data;
    data->id        = id;
    data->refCount  = 1;
    data->isInHash  = false;
    data->image     = NULL;
    data->glTexture = 0;
    data->cudaArray = NULL;
    data->nextMip   = NULL;

    // Update hash.

    if (id.getLength())
    {
        if (!s_hash)
            s_hash = new Hash<String, Data*>;

        Data** old = s_hash->search(id);
        if (old)
        {
            s_hash->remove(id);
            (*old)->isInHash = false;
        }

        s_hash->add(id, data);
        data->isInHash = true;
    }
    return data;
}

//------------------------------------------------------------------------

void Texture::referData(Data* data)
{
    FW_ASSERT(data);
    data->refCountLock.enter();
    data->refCount++;
    data->refCountLock.leave();
}

//------------------------------------------------------------------------

void Texture::unreferData(Data* data)
{
    FW_ASSERT(data);

    // Decrease refcount.

    data->refCountLock.enter();
    int refCount = --data->refCount;
    data->refCountLock.leave();

    if (refCount != 0)
        return;

    // Remove from hash.

    if (data->isInHash)
    {
        FW_ASSERT(s_hash);
        s_hash->remove(data->id);
        if (!s_hash->getSize())
        {
            delete s_hash;
            s_hash = NULL;
        }
    }

    // Delete data.

    if (data->glTexture != 0)
        glDeleteTextures(1, &data->glTexture);

    if (data->cudaArray)
        CudaModule::checkError("cuArrayDestroy", cuArrayDestroy(data->cudaArray));

    delete data->image;
    delete data->nextMip;
    delete data;
}

//------------------------------------------------------------------------
