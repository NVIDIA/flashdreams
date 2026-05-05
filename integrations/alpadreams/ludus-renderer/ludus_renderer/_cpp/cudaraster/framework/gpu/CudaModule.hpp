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
#include "gpu/GLContext.hpp"
#include "gpu/CudaKernel.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CudaModule
{
public:
                        CudaModule          (const void* cubin);
                        CudaModule          (const String& cubinFile);
                        ~CudaModule         (void);

    CUmodule            getHandle           (void) { return m_module; }

    bool                hasKernel           (const String& name);
    CudaKernel          getKernel           (const String& name);

    Buffer&             getGlobal           (const String& name);
    void                updateGlobals       (bool async = false, CUstream stream = NULL); // copy to the device if modified
    
    CUtexref            getTexRef           (const String& name);
    void                setTexRefMode       (CUtexref texRef, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                setTexRef           (const String& name, Buffer& buf, CUarray_format format, int numComponents);
    void                setTexRef           (const String& name, CUdeviceptr ptr, S64 size, CUarray_format format, int numComponents);
    void                setTexRef           (const String& name, CUarray cudaArray, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                setTexRef           (const String& name, const Image& image, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                unsetTexRef         (const String& name);
    void                updateTexRefs       (CUfunction kernel);

    CUsurfref           getSurfRef          (const String& name);
    void                setSurfRef          (const String& name, CUarray cudaArray);

    static void         staticInit          (void);
    static void         staticDeinit        (void);
    static bool         isAvailable         (void)      { staticInit(); return s_available; }
    static S64          getMemoryUsed       (void);
    static void         sync                (bool yield = true); // False = low latency but keeps the CPU busy. True = long latency but relieves the CPU.
    static void         checkError          (const char* funcName, CUresult res);
    static const char*  decodeError         (CUresult res);

    static CUdevice     getDeviceHandle     (void)      { staticInit(); return s_device; }
    static int          getDriverVersion    (void); // e.g. 23 = 2.3
    static int          getComputeCapability(void); // e.g. 13 = 1.3
    static int          getDeviceAttribute  (CUdevice_attribute attrib);
    static CUevent      getStartEvent       (void)      { staticInit(); return s_startEvent; }
    static CUevent      getEndEvent         (void)      { staticInit(); return s_endEvent; }

private:
    static CUdevice     selectDevice        (void);
    static void         printDeviceInfo     (CUdevice device);
    static Vec2i        selectGridSize      (int numBlocks); // TODO: remove
    CUfunction          findKernel          (const String& name);

private:
                        CudaModule          (const CudaModule&); // forbidden
    CudaModule&         operator=           (const CudaModule&); // forbidden

private:
    static bool         s_inited;
    static bool         s_available;
    static CUdevice     s_device;
    static CUcontext    s_context;
    static CUevent      s_startEvent;
    static CUevent      s_endEvent;

    CUmodule            m_module;
    Hash<String, CUfunction> m_kernels;
    Array<Buffer*>      m_globals;
    Hash<String, S32>   m_globalHash;
    Array<CUtexref>     m_texRefs;
    Hash<String, S32>   m_texRefHash;
};

//------------------------------------------------------------------------
}
