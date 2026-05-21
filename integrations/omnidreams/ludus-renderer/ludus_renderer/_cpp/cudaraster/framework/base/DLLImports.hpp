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

//------------------------------------------------------------------------

#define FW_USE_CUDA 1

//------------------------------------------------------------------------

#include <math.h>

#if (FW_USE_CUDA)
#   include <cuda.h>
#   ifdef _MSC_VER
#   pragma warning(push,3)
#   endif
#       include <vector_functions.h> // float4, etc.
#   ifdef _MSC_VER
#   pragma warning(pop)
#   endif
#endif

#if (!FW_CUDA) && defined(_WIN32)
#   define _WIN32_WINNT 0x0600
#   define WIN32_LEAN_AND_MEAN
#   define _KERNEL32_
#   define _WINMM_
#   include <windows.h>
#   undef min
#   undef max

#   pragma warning(push,3)
#   include <mmsystem.h>
#   pragma warning(pop)

#   define _SHLWAPI_
#   include <shlwapi.h>
#endif

//------------------------------------------------------------------------

namespace FW
{
#if (!FW_CUDA)
void    setCudaDLLName      (const String& name);
void    initDLLImports      (void);
void    deinitDLLImports    (void);
#endif
}

//------------------------------------------------------------------------
// CUDA definitions.
//------------------------------------------------------------------------

#if (!FW_USE_CUDA)
#   define CUDA_VERSION 2010
#   define CUDAAPI __stdcall

typedef enum { CUDA_SUCCESS = 0}        CUresult;
typedef struct { FW::S32 x, y; }        int2;
typedef struct { FW::S32 x, y, z; }     int3;
typedef struct { FW::S32 x, y, z, w; }  int4;
typedef struct { FW::F32 x, y; }        float2;
typedef struct { FW::F32 x, y, z; }     float3;
typedef struct { FW::F32 x, y, z, w; }  float4;
typedef struct { FW::F64 x, y; }        double2;
typedef struct { FW::F64 x, y, z; }     double3;
typedef struct { FW::F64 x, y, z, w; }  double4;

typedef void*   CUfunction;
typedef void*   CUmodule;
typedef int     CUdevice;
typedef size_t  CUdeviceptr;
typedef void*   CUcontext;
typedef void*   CUdevprop;
typedef int     CUdevice_attribute;
typedef int     CUjit_option;
typedef void*   CUtexref;
typedef void*   CUarray;
typedef int     CUarray_format;
typedef int     CUaddress_mode;
typedef int     CUfilter_mode;
typedef void*   CUstream;
typedef void*   CUevent;
typedef void*   CUDA_MEMCPY2D;
typedef void*   CUDA_MEMCPY3D;
typedef void*   CUDA_ARRAY_DESCRIPTOR;
typedef void*   CUDA_ARRAY3D_DESCRIPTOR;
typedef int     CUfunction_attribute;

#endif

#if (CUDA_VERSION < 3010)
typedef void* CUsurfref;
#endif

#if (CUDA_VERSION < 3020)
typedef unsigned int    CUsize_t;
#else
typedef size_t          CUsize_t;
#endif

//------------------------------------------------------------------------

#if (!FW_CUDA)
#   define FW_DLL_IMPORT_RETV(RET, CALL, NAME, PARAMS, PASS)        bool isAvailable_ ## NAME(void);
#   define FW_DLL_IMPORT_VOID(RET, CALL, NAME, PARAMS, PASS)        bool isAvailable_ ## NAME(void);
#   define FW_DLL_DECLARE_RETV(RET, CALL, NAME, PARAMS, PASS)       bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   define FW_DLL_DECLARE_VOID(RET, CALL, NAME, PARAMS, PASS)       bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   if (FW_USE_CUDA)
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#   else
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   endif
#   include "base/DLLImports.inl"
#   undef FW_DLL_IMPORT_RETV
#   undef FW_DLL_IMPORT_VOID
#   undef FW_DLL_DECLARE_RETV
#   undef FW_DLL_DECLARE_VOID
#   undef FW_DLL_IMPORT_CUDA
#   undef FW_DLL_IMPORT_CUV2
#endif

//------------------------------------------------------------------------
