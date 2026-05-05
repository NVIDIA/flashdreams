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

#pragma warning(disable:4530) // C++ exception handler used, but unwind semantics are not enabled.
#include <new>
#include <string.h>

namespace FW
{
//------------------------------------------------------------------------

#ifndef NULL
#   define NULL 0
#endif

#ifdef _DEBUG
#   define FW_DEBUG 1
#else
#   define FW_DEBUG 0
#endif

#ifdef _M_X64
#   define FW_64    1
#else
#   define FW_64    0
#endif

#ifdef __CUDACC__
#   define FW_CUDA 1
#else
#   define FW_CUDA 0
#endif

#if FW_CUDA && !defined(__CUDA_ARCH__)
#   define __CUDA_ARCH__ 100 // e.g. 120 = compute capability 1.2
#endif

#if (FW_DEBUG || defined(FW_ENABLE_ASSERT)) && !FW_CUDA
#   define FW_ASSERT(X) ((X) ? ((void)0) : FW::fail("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X))
#else
#   define FW_ASSERT(X) ((void)0)
#endif

#if FW_CUDA
#   define FW_CUDA_FUNC     __device__ __inline__
#   define FW_CUDA_CONST    __constant__
#else
#   define FW_CUDA_FUNC     inline
#   define FW_CUDA_CONST    static const
#endif

#define FW_UNREF(X)         ((void)(X))
#define FW_ARRAY_SIZE(X)    ((int)(sizeof(X) / sizeof((X)[0])))

//------------------------------------------------------------------------

typedef unsigned char       U8;
typedef unsigned short      U16;
typedef unsigned int        U32;
typedef signed char         S8;
typedef signed short        S16;
typedef signed int          S32;
typedef float               F32;
typedef double              F64;
typedef void                (*FuncPtr)(void);

#if FW_CUDA
typedef unsigned long long  U64;
typedef signed long long    S64;
#else
typedef unsigned __int64    U64;
typedef signed __int64      S64;
#endif

#if FW_64
typedef S64                 SPTR;
typedef U64                 UPTR;
#else
typedef __w64 S32           SPTR;
typedef __w64 U32           UPTR;
#endif

//------------------------------------------------------------------------

#define FW_U32_MAX          (0xFFFFFFFFu)
#define FW_S32_MIN          (~0x7FFFFFFF)
#define FW_S32_MAX          (0x7FFFFFFF)
#define FW_U64_MAX          ((U64)(S64)-1)
#define FW_S64_MIN          ((S64)-1 << 63)
#define FW_S64_MAX          (~((S64)-1 << 63))
#define FW_F32_MIN          (1.175494351e-38f)
#define FW_F32_MAX          (3.402823466e+38f)
#define FW_F64_MIN          (2.2250738585072014e-308)
#define FW_F64_MAX          (1.7976931348623158e+308)
#define FW_PI               (3.14159265358979323846f)

//------------------------------------------------------------------------

#if !FW_CUDA

class String;

// Common functionality.

void*           malloc          (size_t size);
void            free            (void* ptr);
void*           realloc         (void* ptr, size_t size);

void            printf          (const char* fmt, ...);
String          sprintf         (const char* fmt, ...);

// Error handling.

void            setError        (const char* fmt, ...);
String          clearError      (void);
bool            restoreError    (const String& old);
bool            hasError        (void);
const String&   getError        (void);

void            fail            (const char* fmt, ...);
void            failWin32Error  (const char* funcName);
void            failIfError     (void);

// Re-entrancy. Called internally by Main and Window.

int             incNestingLevel (int delta);
bool            setDiscardEvents(bool discard);
bool            getDiscardEvents(void);

// Logging.

void            pushLogFile     (const String& name, bool append = true);
void            popLogFile      (void);
bool            hasLogFile      (void);

// Memory profiling.

size_t          getMemoryUsed   (void);
void            pushMemOwner    (const char* id);
void            popMemOwner     (void);
void            printMemStats   (void);

// Performance profiling.

void            profileStart    (void);
void            profilePush     (const char* id);
void            profilePop      (void);
void            profileEnd      (bool printResults = true);

#endif

//------------------------------------------------------------------------
// min(), max(), clamp().

template <class T> FW_CUDA_FUNC void swap(T& a, T& b) { T t = a; a = b; b = t; }

#define FW_SPECIALIZE_MINMAX(TEMPLATE, T, MIN, MAX) \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b) { return MIN; } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b) { return MAX; } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c) { return min(min(a, b), c); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c) { return max(max(a, b), c); } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c, T d) { return min(min(min(a, b), c), d); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c, T d) { return max(max(max(a, b), c), d); } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c, T d, T e) { return min(min(min(min(a, b), c), d), e); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c, T d, T e) { return max(max(max(max(a, b), c), d), e); } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c, T d, T e, T f) { return min(min(min(min(min(a, b), c), d), e), f); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c, T d, T e, T f) { return max(max(max(max(max(a, b), c), d), e), f); } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c, T d, T e, T f, T g) { return min(min(min(min(min(min(a, b), c), d), e), f), g); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c, T d, T e, T f, T g) { return max(max(max(max(max(max(a, b), c), d), e), f), g); } \
    TEMPLATE FW_CUDA_FUNC T min(T a, T b, T c, T d, T e, T f, T g, T h) { return min(min(min(min(min(min(min(a, b), c), d), e), f), g), h); } \
    TEMPLATE FW_CUDA_FUNC T max(T a, T b, T c, T d, T e, T f, T g, T h) { return max(max(max(max(max(max(max(a, b), c), d), e), f), g), h); } \
    TEMPLATE FW_CUDA_FUNC T clamp(T v, T lo, T hi) { return min(max(v, lo), hi); }

FW_SPECIALIZE_MINMAX(template <class T>, T&, (a < b) ? a : b, (a > b) ? a : b)
FW_SPECIALIZE_MINMAX(template <class T>, const T&, (a < b) ? a : b, (a > b) ? a : b)

#if FW_CUDA
FW_SPECIALIZE_MINMAX(, U32, ::min(a, b), ::max(a, b))
FW_SPECIALIZE_MINMAX(, S32, ::min(a, b), ::max(a, b))
FW_SPECIALIZE_MINMAX(, U64, ::min(a, b), ::max(a, b))
FW_SPECIALIZE_MINMAX(, S64, ::min(a, b), ::max(a, b))
FW_SPECIALIZE_MINMAX(, F32, ::fminf(a, b), ::fmaxf(a, b))
FW_SPECIALIZE_MINMAX(, F64, ::fmin(a, b), ::fmax(a, b))
#endif

//------------------------------------------------------------------------
// CUDA utilities.

#if FW_CUDA
#   define globalThreadIdx (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))
#endif

//------------------------------------------------------------------------
// new, delete.
}

#ifndef FW_DO_NOT_OVERRIDE_NEW_DELETE
#if !FW_CUDA

inline void*    operator new        (size_t size)       { return FW::malloc(size); }
inline void*    operator new[]      (size_t size)       { return FW::malloc(size); }
inline void     operator delete     (void* ptr)         { return FW::free(ptr); }
inline void     operator delete[]   (void* ptr)         { return FW::free(ptr); }

#endif
#endif // FW_DO_NOT_OVERRIDE_NEW_DELETE

//------------------------------------------------------------------------
