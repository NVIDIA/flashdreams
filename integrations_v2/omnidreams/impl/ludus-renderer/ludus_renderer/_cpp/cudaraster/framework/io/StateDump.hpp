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
#include "base/Hash.hpp"
#include "io/Stream.hpp"

namespace FW
{

//------------------------------------------------------------------------

class StateDump : public Serializable
{
public:
                                StateDump       (void)                          {}
                                StateDump       (const StateDump& other)        { add(other); }
    virtual                     ~StateDump      (void)                          { clear(); }

    virtual void                readFromStream  (InputStream& s);
    virtual void                writeToStream   (OutputStream& s) const;

    void                        clear           (void);
    void                        add             (const StateDump& other);
    void                        set             (const StateDump& other)        { if (&other != this) { clear(); add(other); } }

    void                        pushOwner       (const String& id)              { m_owners.add(xlateId(id + "::")); }
    void                        popOwner        (void)                          { m_owners.removeLast(); }

    bool                        has             (const String& id) const        { return m_values.contains(xlateId(id)); }
    const Array<U8>*            get             (const String& id) const;
    bool                        get             (void* ptr, int size, const String& id);
    template <class T> bool     get             (T& value, const String& id) const;
    template <class T> bool     get             (T& value, const String& id, const T& defValue) const;
    template <class T> T        get             (const String& id, const T& defValue) const;

    void                        set             (const void* ptr, int size, const String& id);
    template <class T> void     set             (const T& value, const String& id);
    void                        unset           (const String& id)              { setInternal(NULL, xlateId(id)); }

    StateDump&                  operator=       (const StateDump& other)        { set(other); return *this; }

private:
    String                      xlateId         (const String& id) const        { return (m_owners.getSize()) ? m_owners.getLast() + id : id; }
    void                        setInternal     (Array<U8>* data, const String& id); // takes ownership of data

private:
    Hash<String, Array<U8>* >   m_values;
    Array<String>               m_owners;

    mutable MemoryInputStream   m_memIn;
    MemoryOutputStream          m_memOut;
};

//------------------------------------------------------------------------

template <class T> bool StateDump::get(T& value, const String& id) const
{
    const Array<U8>* data = get(id);
    if (!data)
        return false;

    m_memIn.reset(*data);
    m_memIn >> value;
    FW_ASSERT(m_memIn.getOffset() == data->getSize());
    return true;
}

//------------------------------------------------------------------------

template <class T> bool StateDump::get(T& value, const String& id, const T& defValue) const
{
    if (get(value, id))
        return true;
    value = defValue;
    return false;
}

//------------------------------------------------------------------------

template <class T> T StateDump::get(const String& id, const T& defValue) const
{
    T value;
    get(value, id, defValue);
    return value;
}

//------------------------------------------------------------------------

template <class T> void StateDump::set(const T& value, const String& id)
{
    m_memOut.clear();
    m_memOut << value;
    Array<U8>& data = m_memOut.getData();
    set(data.getPtr(), data.getNumBytes(), id);
}

//------------------------------------------------------------------------
}
