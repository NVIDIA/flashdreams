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

#include "io/StateDump.hpp"

using namespace FW;

//------------------------------------------------------------------------

void StateDump::readFromStream(InputStream& s)
{
    clear();
    S32 num;
    s >> num;
    for (int i = 0; i < num; i++)
    {
        String id;
        Array<U8>* data = new Array<U8>;
        s >> id >> *data;
        setInternal(data, id);
    }
}

//------------------------------------------------------------------------

void StateDump::writeToStream(OutputStream& s) const
{
    s << m_values.getSize();
    for (int i = m_values.firstSlot(); i != -1; i = m_values.nextSlot(i))
        s << m_values.getSlot(i).key << *m_values.getSlot(i).value;
}

//------------------------------------------------------------------------

void StateDump::clear(void)
{
    for (int i = m_values.firstSlot(); i != -1; i = m_values.nextSlot(i))
        delete m_values.getSlot(i).value;
    m_values.clear();
}

//------------------------------------------------------------------------

void StateDump::add(const StateDump& other)
{
    if (&other != this)
        for (int i = other.m_values.firstSlot(); i != -1; i = other.m_values.nextSlot(i))
            setInternal(new Array<U8>(*other.m_values.getSlot(i).value), other.m_values.getSlot(i).key);
}

//------------------------------------------------------------------------

const Array<U8>* StateDump::get(const String& id) const
{
    Array<U8>* const* data = m_values.search(xlateId(id));
    return (data) ? *data : NULL;
}

//------------------------------------------------------------------------

bool StateDump::get(void* ptr, int size, const String& id)
{
    FW_ASSERT(size >= 0);
    FW_ASSERT(ptr || !size);

    Array<U8>* const* data = m_values.search(xlateId(id));
    if (!data)
        return false;

    FW_ASSERT((*data)->getNumBytes() == size);
    memcpy(ptr, (*data)->getPtr(), size);
    return true;
}

//------------------------------------------------------------------------

void StateDump::set(const void* ptr, int size, const String& id)
{
    setInternal(new Array<U8>((const U8*)ptr, size), xlateId(id));
}

//------------------------------------------------------------------------

void StateDump::setInternal(Array<U8>* data, const String& id)
{
    if (m_values.contains(id))
        delete m_values.remove(id);
    if (data)
        m_values.add(id, data);
}

//------------------------------------------------------------------------
