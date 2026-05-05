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
#include "io/Stream.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Buffer : public Serializable
{
public:
    enum Module
    {
        CPU             = 1 << 0,
        GL              = 1 << 1,
        Cuda            = 1 << 2,

        Module_None     = 0,
        Module_All      = (1 << 3) - 1
    };

    enum Hint
    {
        Hint_PageLock   = 1 << 0,
        Hint_CudaGL     = 1 << 1,

        Hint_None       = 0,
        Hint_All        = (1 << 2) - 1
    };

public:
    explicit        Buffer              (U32 hints = Hint_None)                 { init(0, hints, 1); }
    explicit        Buffer              (const void* ptr, S64 size, U32 hints = Hint_None, int align = 1) { init(size, hints, align); if (ptr) set(ptr); }
    template <class T> explicit Buffer  (const Array<T>& data, U32 hints = Hint_None, int align = 1) { init(data.getNumBytes(), hints, align); set(data.getPtr()); }
    template <class T> explicit Buffer  (const Array<T>& data, int start, int end, U32 hints = Hint_None, int align = 1) { init((end - start) * sizeof(T), hints, align); set(data.getPtr(start)); }
                    Buffer              (Buffer& other)                         { init(other.getSize(), other.getHints(), other.getAlign()); setRange(0, other, 0, other.getSize()); }
    virtual         ~Buffer             (void)                                  { deinit(); }

    void            wrapCPU             (void* cpuPtr, S64 size);
    void            wrapGL              (GLuint glBuffer);
    void            wrapCuda            (CUdeviceptr cudaPtr, S64 size);

    S64             getSize             (void) const                            { return m_size; }
    U32             getHints            (void) const                            { return m_hints; }
    int             getAlign            (void) const                            { return m_align; }
    void            setHintsAndAlign    (U32 hints, int align)                  { realloc(m_size, validateHints(hints, align, m_original), align); }
    void            setHints            (U32 hints)                             { setHintsAndAlign(hints, m_align); }
    void            setAlign            (int align)                             { setHintsAndAlign(m_hints, align); }

    void            reset               (U32 hints, int align)                  { deinit(); init(0, hints, align); }
    void            reset               (U32 hints)                             { reset(hints, m_align); }
    void            reset               (void)                                  { reset(m_hints, m_align); }
    void            reset               (const void* ptr, S64 size, U32 hints, int align) { deinit(); init(size, hints, align); if (ptr) setRange(0, ptr, size); }
    void            reset               (const void* ptr, S64 size, U32 hints)  { reset(ptr, size, hints, m_align); }
    void            reset               (const void* ptr, S64 size)             { reset(ptr, size, m_hints, m_align); }
    void            resize              (S64 size)                              { realloc(size, m_hints, m_align); }
    void            resizeDiscard       (S64 size)                              { if (m_size != size) reset(NULL, size, m_hints, m_align); }
    void            free                (Module module);

    void            getRange            (void* dst, S64 srcOfs, S64 size, bool async = false, CUstream cudaStream = NULL) const;
    void            get                 (void* ptr)                             { getRange(ptr, 0, getSize()); }
    template <class T> void get         (Array<T>& data)                        { FW_ASSERT(data.getNumBytes() == getSize()); get(data.getPtr()); }

    void            setRange            (S64 dstOfs, const void* src, S64 size, bool async = false, CUstream cudaStream = NULL);
    void            setRange            (S64 dstOfs, Buffer& src, S64 srcOfs, S64 size, bool async = false, CUstream cudaStream = NULL);
    void            set                 (const void* ptr)                       { setRange(0, ptr, getSize()); }
    void            set                 (const void* ptr, S64 size)             { resizeDiscard(size); setRange(0, ptr, size); }
    void            set                 (Buffer& other)                         { if (&other != this) { resizeDiscard(other.getSize()); setRange(0, other, 0, other.getSize()); } }
    template <class T> void set         (const Array<T>& data)                  { set(data.getPtr(), data.getNumBytes()); }
    template <class T> void set         (const Array<T>& data, int start, int end) { set(data.getPtr(start), (end - start) * sizeof(T)); }

    void            clearRange          (S64 dstOfs, int value, S64 size, bool async = false, CUstream cudaStream = NULL);
    void            clear               (int value = 0)                         { clearRange(0, value, m_size); }

    void            setOwner            (Module module, bool modify, bool async = false, CUstream cudaStream = NULL, S64 validSize = -1);
    Module          getOwner            (void) const                            { return m_owner; }
    void            discard             (void)                                  { m_dirty = 0; }

    const U8*       getPtr              (S64 ofs = 0)                           { FW_ASSERT(ofs >= 0 && ofs <= m_size); setOwner(CPU, false); return m_cpuPtr + ofs; }
    GLuint          getGLBuffer         (void)                                  { setOwner(GL, false); return m_glBuffer; }
    CUdeviceptr     getCudaPtr          (S64 ofs = 0)                           { FW_ASSERT(ofs >= 0 && ofs <= m_size); setOwner(Cuda, false); return m_cudaPtr + (U32)ofs; }

    U8*             getMutablePtr       (S64 ofs = 0)                           { FW_ASSERT(ofs >= 0 && ofs <= m_size); setOwner(CPU, true); return m_cpuPtr + ofs; }
    GLuint          getMutableGLBuffer  (void)                                  { setOwner(GL, true); return m_glBuffer; }
    CUdeviceptr     getMutableCudaPtr   (S64 ofs = 0)                           { FW_ASSERT(ofs >= 0 && ofs <= m_size); setOwner(Cuda, true); return m_cudaPtr + (U32)ofs; }

    U8*             getMutablePtrDiscard(S64 ofs = 0)                           { discard(); return getMutablePtr(ofs); }
    GLuint          getMutableGLBufferDiscard(void)                             { discard(); return getMutableGLBuffer(); }
    CUdeviceptr     getMutableCudaPtrDiscard(S64 ofs = 0)                       { discard(); return getMutableCudaPtr(ofs); }

    Buffer&         operator=           (Buffer& other)                         { set(other); return *this; }
    U8              operator[]          (S64 idx)                               { FW_ASSERT(idx < m_size); return *getPtr(idx); }

    virtual void    readFromStream      (InputStream& s);
    virtual void    writeToStream       (OutputStream& s) const;

    static void     memcpyHtoD          (CUdeviceptr dst, const void* src, S64 size, bool async = false, CUstream cudaStream = NULL) { memcpyXtoX(NULL, dst, src, NULL, size, async, cudaStream); }
    static void     memcpyDtoH          (void* dst, CUdeviceptr src, S64 size, bool async = false, CUstream cudaStream = NULL) { memcpyXtoX(dst, NULL, NULL, src, size, async, cudaStream); }
    static void     memcpyDtoD          (CUdeviceptr dst, CUdeviceptr src, S64 size, bool async = false, CUstream cudaStream = NULL) { memcpyXtoX(NULL, dst, NULL, src, size, async, cudaStream); }

private:
    static U32      validateHints       (U32 hints, int align, Module original);

    void            init                (S64 size, U32 hints, int align);
    void            deinit              (void);
    void            wrap                (Module module, S64 size);
    void            realloc             (S64 size, U32 hints, int align);
    void            validateCPU         (bool async, CUstream cudaStream, S64 validSize);

    static void     cpuAlloc            (U8*& cpuPtr, U8*& cpuBase, S64 size, U32 hints, int align);
    static void     cpuFree             (U8*& cpuPtr, U8*& cpuBase, U32 hints);
    static void     glAlloc             (GLuint& glBuffer, S64 size, const void* data);
    static void     glFree              (GLuint& glBuffer, bool& cudaGLReg);
    static void     cudaAlloc           (CUdeviceptr& cudaPtr, CUdeviceptr& cudaBase, bool& cudaGLReg, S64 size, GLuint glBuffer, U32 hints, int align);
    static void     cudaFree            (CUdeviceptr& cudaPtr, CUdeviceptr& cudaBase, GLuint glBuffer, U32 hints);

    static void     checkSize           (S64 size, int bits, const String& funcName);

    static void     memcpyXtoX          (void* dstHost, CUdeviceptr dstDevice, const void* srcHost, CUdeviceptr srcDevice, S64 size, bool async, CUstream cudaStream);

private:
    U32             m_hints;
    S32             m_align;
    S64             m_size;
    Module          m_original;
    Module          m_owner;
    U32             m_exists;
    U32             m_dirty;

    U8*             m_cpuPtr;
    U8*             m_cpuBase;
    GLuint          m_glBuffer;
    CUdeviceptr     m_cudaPtr;
    CUdeviceptr     m_cudaBase;
    bool            m_cudaGLReg;
};

//------------------------------------------------------------------------
}
