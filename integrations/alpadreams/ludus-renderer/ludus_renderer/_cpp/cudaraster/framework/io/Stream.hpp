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
#include "base/Array.hpp"
#include "base/String.hpp"
#include "base/Math.hpp"

#include <stdarg.h>

namespace FW
{

//------------------------------------------------------------------------

class InputStream
{
public:
                            InputStream             (void)          {}
    virtual                 ~InputStream            (void)          {}

    virtual int             read                    (void* ptr, int size) = 0; // out of data => partial result
    void                    readFully               (void* ptr, int size); // out of data => failure

    U8                      readU8                  (void)          { U8 b; readFully(&b, sizeof(b)); return b; }
    U16                     readU16BE               (void)          { U8 b[2]; readFully(b, sizeof(b)); return (U16)((b[0] << 8) | b[1]); }
    U16                     readU16LE               (void)          { U8 b[2]; readFully(b, sizeof(b)); return (U16)((b[1] << 8) | b[0]); }
    U32                     readU32BE               (void)          { U8 b[4]; readFully(b, sizeof(b)); return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]; }
    U32                     readU32LE               (void)          { U8 b[4]; readFully(b, sizeof(b)); return (b[3] << 24) | (b[2] << 16) | (b[1] << 8) | b[0]; }
    U64                     readU64BE               (void)          { U8 b[8]; readFully(b, sizeof(b)); return ((U64)b[0] << 56) | ((U64)b[1] << 48) | ((U64)b[2] << 40) | ((U64)b[3] << 32) | ((U64)b[4] << 24) | ((U64)b[5] << 16) | ((U64)b[6] << 8) | (U64)b[7]; }
    U64                     readU64LE               (void)          { U8 b[8]; readFully(b, sizeof(b)); return ((U64)b[7] << 56) | ((U64)b[6] << 48) | ((U64)b[5] << 40) | ((U64)b[4] << 32) | ((U64)b[3] << 24) | ((U64)b[2] << 16) | ((U64)b[1] << 8) | (U64)b[0]; }
};

//------------------------------------------------------------------------

class OutputStream
{
public:
                            OutputStream            (void)          {}
    virtual                 ~OutputStream           (void)          {}

    virtual void            write                   (const void* ptr, int size) = 0;
    virtual void            flush                   (void) = 0;

    void                    writeU8                 (U32 v)         { U8 b[1]; b[0] = (U8)v; write(b, sizeof(b)); }
    void                    writeU16BE              (U32 v)         { U8 b[2]; b[0] = (U8)(v >> 8); b[1] = (U8)v; write(b, sizeof(b)); }
    void                    writeU16LE              (U32 v)         { U8 b[2]; b[1] = (U8)(v >> 8); b[0] = (U8)v; write(b, sizeof(b)); }
    void                    writeU32BE              (U32 v)         { U8 b[4]; b[0] = (U8)(v >> 24); b[1] = (U8)(v >> 16); b[2] = (U8)(v >> 8); b[3] = (U8)v; write(b, sizeof(b)); }
    void                    writeU32LE              (U32 v)         { U8 b[4]; b[3] = (U8)(v >> 24); b[2] = (U8)(v >> 16); b[1] = (U8)(v >> 8); b[0] = (U8)v; write(b, sizeof(b)); }
    void                    writeU64BE              (U64 v)         { U8 b[8]; b[0] = (U8)(v >> 56); b[1] = (U8)(v >> 48); b[2] = (U8)(v >> 40); b[3] = (U8)(v >> 32); b[4] = (U8)(v >> 24); b[5] = (U8)(v >> 16); b[6] = (U8)(v >> 8); b[7] = (U8)v; write(b, sizeof(b)); }
    void                    writeU64LE              (U64 v)         { U8 b[8]; b[7] = (U8)(v >> 56); b[6] = (U8)(v >> 48); b[5] = (U8)(v >> 40); b[4] = (U8)(v >> 32); b[3] = (U8)(v >> 24); b[2] = (U8)(v >> 16); b[1] = (U8)(v >> 8); b[0] = (U8)v; write(b, sizeof(b)); }
};

//------------------------------------------------------------------------

class BufferedInputStream : public InputStream
{
public:
                            BufferedInputStream     (InputStream& stream, int bufferSize = 64 << 10);
    virtual                 ~BufferedInputStream    (void);

    virtual int             read                    (void* ptr, int size);
    char*                   readLine                (bool combineWithBackslash = false, bool normalizeWhitespace = false);

    bool                    fillBuffer              (int size);
    int                     getBufferSize           (void)          { return m_numRead - m_numConsumed; }
    U8*                     getBufferPtr            (void)          { return m_buffer.getPtr(m_numConsumed); }
    void                    consumeBuffer           (int num);

private:
                            BufferedInputStream     (const BufferedInputStream&); // forbidden
    BufferedInputStream&    operator=               (const BufferedInputStream&); // forbidden

private:
    InputStream&            m_stream;
    Array<U8>               m_buffer;
    S32                     m_numRead;
    S32                     m_numConsumed;
};

//------------------------------------------------------------------------

class BufferedOutputStream : public OutputStream
{
public:
                            BufferedOutputStream    (OutputStream& stream, int bufferSize = 64 << 10, bool writeOnLF = false, bool emulateCR = false);
    virtual                 ~BufferedOutputStream   (void);

    virtual void            write                   (const void* ptr, int size);
    void                    writef                  (const char* fmt, ...);
    void                    writefv                 (const char* fmt, va_list args);
    virtual void            flush                   (void);

    S32                     getNumBytesWritten      (void) const    { return m_numFlushed + m_numValid; }

private:
    void                    addValid                (int size);
    void                    flushInternal           (void);

private:
                            BufferedOutputStream    (const BufferedOutputStream&); // forbidden
    BufferedOutputStream&   operator=               (const BufferedOutputStream&); // forbidden

private:
    OutputStream&           m_stream;
    bool                    m_writeOnLF;
    bool                    m_emulateCR;

    Array<U8>               m_buffer;
    S32                     m_numValid;
    S32                     m_lineStart;
    S32                     m_currOfs;
    S32                     m_numFlushed;
};

//------------------------------------------------------------------------

class MemoryInputStream : public InputStream
{
public:
                            MemoryInputStream       (void)                      { reset(); }
                            MemoryInputStream       (const void* ptr, int size) { reset(ptr, size); }
    template <class T> explicit MemoryInputStream   (const Array<T>& data)      { reset(data); }
    virtual                 ~MemoryInputStream      (void);

    virtual int             read                    (void* ptr, int size);

    int                     getOffset               (void) const                { return m_ofs; }
    void                    seek                    (int ofs)                   { FW_ASSERT(ofs >= 0 && ofs <= m_size); m_ofs = ofs; }

    void                    reset                   (void)                      { m_ptr = NULL; m_size = 0; m_ofs = 0; }
    void                    reset                   (const void* ptr, int size) { FW_ASSERT(size >= 0); FW_ASSERT(ptr || !size); m_ptr = (const U8*)ptr; m_size = size; m_ofs = 0; }
    template <class T> void reset                   (const Array<T>& data)      { reset(data.getPtr(), data.getNumBytes()); }

private:
                            MemoryInputStream       (const MemoryInputStream&); // forbidden
    MemoryInputStream&      operator=               (const MemoryInputStream&); // forbidden

private:
    const U8*               m_ptr;
    S32                     m_size;
    S32                     m_ofs;
};

//------------------------------------------------------------------------

class MemoryOutputStream : public OutputStream
{
public:
                            MemoryOutputStream      (int capacity = 0) { m_data.setCapacity(capacity); }
    virtual                 ~MemoryOutputStream     (void);

    virtual void            write                   (const void* ptr, int size);
    virtual void            flush                   (void);

    void                    clear                   (void)          { m_data.clear(); }
    Array<U8>&              getData                 (void)          { return m_data; }
    const Array<U8>&        getData                 (void) const    { return m_data; }

private:
                            MemoryOutputStream      (const MemoryOutputStream&); // forbidden
    MemoryOutputStream&     operator=               (const MemoryOutputStream&); // forbidden

private:
    Array<U8>               m_data;
};

//------------------------------------------------------------------------

class Serializable
{
public:
                            Serializable            (void)          {}
    virtual                 ~Serializable           (void)          {}

    virtual void            readFromStream          (InputStream& s) = 0;
    virtual void            writeToStream           (OutputStream& s) const = 0;
};

//------------------------------------------------------------------------
// Primitive types.
//------------------------------------------------------------------------

inline InputStream&     operator>>  (InputStream& s, U8& v)         { v = s.readU8(); return s; }
inline InputStream&     operator>>  (InputStream& s, U16& v)        { v = s.readU16LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, U32& v)        { v = s.readU32LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, U64& v)        { v = s.readU64LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, S8& v)         { v = s.readU8(); return s; }
inline InputStream&     operator>>  (InputStream& s, S16& v)        { v = s.readU16LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, S32& v)        { v = s.readU32LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, S64& v)        { v = s.readU64LE(); return s; }
inline InputStream&     operator>>  (InputStream& s, F32& v)        { v = bitsToFloat(s.readU32LE()); return s; }
inline InputStream&     operator>>  (InputStream& s, F64& v)        { v = bitsToDouble(s.readU64LE()); return s; }
inline InputStream&     operator>>  (InputStream& s, char& v)       { v = s.readU8(); return s; }
inline InputStream&     operator>>  (InputStream& s, bool& v)       { v = (s.readU8() != 0); return s; }

inline OutputStream&    operator<<  (OutputStream& s, U8 v)         { s.writeU8(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, U16 v)        { s.writeU16LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, U32 v)        { s.writeU32LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, U64 v)        { s.writeU64LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, S8 v)         { s.writeU8(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, S16 v)        { s.writeU16LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, S32 v)        { s.writeU32LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, S64 v)        { s.writeU64LE(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, F32 v)        { s.writeU32LE(floatToBits(v)); return s; }
inline OutputStream&    operator<<  (OutputStream& s, F64 v)        { s.writeU64LE(doubleToBits(v)); return s; }
inline OutputStream&    operator<<  (OutputStream& s, char v)       { s.writeU8(v); return s; }
inline OutputStream&    operator<<  (OutputStream& s, bool v)       { s.writeU8((v) ? 1 : 0); return s; }

//------------------------------------------------------------------------
// Types defined or included by this header.
//------------------------------------------------------------------------

inline InputStream& operator>>(InputStream& s, Serializable& v)
{
    v.readFromStream(s);
    return s;
}

//------------------------------------------------------------------------

inline OutputStream& operator<<(OutputStream& s, const Serializable& v)
{
    v.writeToStream(s);
    return s;
}

//------------------------------------------------------------------------

template <class T> InputStream& operator>>(InputStream& s, Array<T>& v)
{
    S32 len;
    s >> len;
    v.reset(len);
    for (int i = 0; i < len; i++)
        s >> v[i];
    return s;
}

//------------------------------------------------------------------------

template <class T> OutputStream& operator<<(OutputStream& s, const Array<T>& v)
{
    s << v.getSize();
    for (int i = 0; i < v.getSize(); i++)
        s << v[i];
    return s;
}

//------------------------------------------------------------------------

inline InputStream& operator>>(InputStream& s, String& v)
{
    S32 len;
    s >> len;
    Array<char> t(NULL, len + 1);
    for (int i = 0; i < len; i++)
        s >> t[i];
    t[len] = '\0';
    v.set(t.getPtr());
    return s;
}

//------------------------------------------------------------------------

inline OutputStream& operator<<(OutputStream& s, const String& v)
{
    s << v.getLength();
    for (int i = 0; i < v.getLength(); i++)
        s << v[i];
    return s;
}

//------------------------------------------------------------------------

template <class T, int L, class S> InputStream& operator>>(InputStream& s, VectorBase<T, L, S>& v)
{
    for (int i = 0; i < L; i++)
        s >> v[i];
    return s;
}

//------------------------------------------------------------------------

template <class T, int L, class S> OutputStream& operator<<(OutputStream& s, const VectorBase<T, L, S>& v)
{
    for (int i = 0; i < L; i++)
        s << v[i];
    return s;
}

//------------------------------------------------------------------------

template <class T, int L, class S> InputStream& operator>>(InputStream& s, MatrixBase<T, L, S>& v)
{
    for (int i = 0; i < L * L; i++)
        s >> v.getPtr()[i];
    return s;
}

//------------------------------------------------------------------------

template <class T, int L, class S> OutputStream& operator<<(OutputStream& s, const MatrixBase<T, L, S>& v)
{
    for (int i = 0; i < L * L; i++)
        s << v.getPtr()[i];
    return s;
}

//------------------------------------------------------------------------
}
