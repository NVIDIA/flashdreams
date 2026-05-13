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

namespace FW
{
//------------------------------------------------------------------------

class CudaModule;

//------------------------------------------------------------------------
// Automatic translation of kernel parameters:
//
// MyType*                          => mutable CUdeviceptr, valid for ONE element
// CudaKernel::Param(MyType*, N)    => mutable CUdeviceptr, valid for N elements
// Array<MyType>                    => mutable CUdeviceptr, valid for all elements
// Buffer                           => mutable CUdeviceptr, valid for the whole buffer
// Image                            => mutable CUdeviceptr, valid for all pixels
// MyType                           => MyType, passed by value (int, float, Vec4f, etc.)
//------------------------------------------------------------------------

class CudaKernel
{
public:
    struct Param // Wrapper for converting kernel parameters to CUDA-compatible types.
    {
        S32             size;
        S32             align;
        const void*     value;
        CUdeviceptr     cudaPtr;
        Buffer          buffer;

        template <class T>  Param           (const T& v)                { size = sizeof(T); align = __alignof(T); value = &v; }
        template <class T>  Param           (const T* ptr, int num = 1) { buffer.wrapCPU(ptr, num * sizeof(T)); setCudaPtr(buffer.getCudaPtr()); }
        template <class T>  Param           (T* ptr, int num = 1)       { buffer.wrapCPU(ptr, num * sizeof(T)); setCudaPtr(buffer.getMutableCudaPtr()); }
        template <class T>  Param           (const Array<T>& v)         { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setCudaPtr(buffer.getCudaPtr()); }
        template <class T>  Param           (Array<T>& v)               { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setCudaPtr(buffer.getMutableCudaPtr()); }
        template <class T>  Param           (const Array64<T>& v)       { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setCudaPtr(buffer.getCudaPtr()); }
        template <class T>  Param           (Array64<T>& v)             { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setCudaPtr(buffer.getMutableCudaPtr()); }
                            Param           (Buffer& v)                 { setCudaPtr(v.getMutableCudaPtr()); }
                            Param           (const Image& v)            { setCudaPtr(v.getBuffer().getCudaPtr()); }
                            Param           (Image& v)                  { setCudaPtr(v.getBuffer().getMutableCudaPtr()); }
        void                setCudaPtr      (CUdeviceptr ptr)           { size = sizeof(CUdeviceptr); align = __alignof(CUdeviceptr); value = &cudaPtr; cudaPtr = ptr; }
    };

    typedef const Param& P; // To reduce the amount of code in setParams() overloads.

public:
                        CudaKernel          (CudaModule* module = NULL, CUfunction function = NULL);
                        CudaKernel          (const CudaKernel& other)   { operator=(other); }
                        ~CudaKernel         (void);

    CudaModule*         getModule           (void) const                { return m_module; }
    CUfunction          getHandle           (void) const                { return m_function; }
    int                 getAttribute        (CUfunction_attribute attrib) const;

    CudaKernel&         setParams           (const void* ptr, int size) { m_params.set((const U8*)ptr, size); }
    CudaKernel&         setParams           (const Param* const* params, int numParams);

    CudaKernel&         setParams           (void)                                                                                                  { return setParams((const Param* const*)NULL, 0); }
    CudaKernel&         setParams           (P p0)                                                                                                  { const Param* p[] = { &p0 }; return setParams(p, 1); }
    CudaKernel&         setParams           (P p0, P p1)                                                                                            { const Param* p[] = { &p0, &p1 }; return setParams(p, 2); }
    CudaKernel&         setParams           (P p0, P p1, P p2)                                                                                      { const Param* p[] = { &p0, &p1, &p2 }; return setParams(p, 3); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3)                                                                                { const Param* p[] = { &p0, &p1, &p2, &p3 }; return setParams(p, 4); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4)                                                                          { const Param* p[] = { &p0, &p1, &p2, &p3, &p4 }; return setParams(p, 5); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5)                                                                    { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5 }; return setParams(p, 6); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6)                                                              { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6 }; return setParams(p, 7); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7)                                                        { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7 }; return setParams(p, 8); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8)                                                  { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8 }; return setParams(p, 9); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9)                                            { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9 }; return setParams(p, 10); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10)                                     { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10 }; return setParams(p, 11); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11)                              { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11 }; return setParams(p, 12); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12)                       { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12 }; return setParams(p, 13); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13)                { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13 }; return setParams(p, 14); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14)         { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14 }; return setParams(p, 15); }
    CudaKernel&         setParams           (P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15)  { const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15 }; return setParams(p, 16); }

    CudaKernel&         preferL1            (void)                      { m_preferL1 = true; return *this; }
    CudaKernel&         preferShared        (void)                      { m_preferL1 = false; return *this; }
    CudaKernel&         setSharedBankSize   (int bytes)                 { FW_ASSERT(bytes == 4 || bytes == 8); m_sharedBankSize = bytes; return *this; }

    CudaKernel&         setAsync            (CUstream stream = NULL)    { m_async = true; m_stream = stream; return *this; }
    CudaKernel&         cancelAsync         (void)                      { m_async = false; return *this; }

    Vec2i               getDefaultBlockSize (void) const;                                           // Smallest block that reaches maximal occupancy.
    CudaKernel&         setGrid             (int numThreads, const Vec2i& blockSize = 0);           // Generates at least numThreads.
    CudaKernel&         setGrid             (const Vec2i& sizeThreads, const Vec2i& blockSize = 0); // Generates at least sizeThreads in both X and Y.

    CudaKernel&         launch              (void);
    CudaKernel&         launch              (int numThreads, const Vec2i& blockSize = 0)                                { setGrid(numThreads, blockSize); return launch(); }
    CudaKernel&         launch              (const Vec2i& sizeThreads, const Vec2i& blockSize = 0)                      { setGrid(sizeThreads, blockSize); return launch(); }
    F32                 launchTimed         (bool yield = true); // Returns GPU time in seconds.
    F32                 launchTimed         (int numThreads, const Vec2i& blockSize = 0, bool yield = true)             { setGrid(numThreads, blockSize); return launchTimed(yield); }
    F32                 launchTimed         (const Vec2i& sizeThreads, const Vec2i& blockSize = 0, bool yield = true)   { setGrid(sizeThreads, blockSize); return launchTimed(yield); }

    CudaKernel&         sync                (bool yield = true); // False = low latency but keeps the CPU busy. True = long latency but relieves the CPU.

    CudaKernel&         operator=           (const CudaKernel& other);

private:
    bool                prepareLaunch       (void);
    void                performLaunch       (void);

private:
    CudaModule*         m_module;
    CUfunction          m_function;

    Array<U8>           m_params;
    bool                m_preferL1;
    S32                 m_sharedBankSize;
    bool                m_async;
    CUstream            m_stream;
    Vec2i               m_gridSize;
    Vec2i               m_blockSize;
};

//------------------------------------------------------------------------
}
