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

#include "gpu/CudaKernel.hpp"
#include "gpu/CudaModule.hpp"
#include "base/Timer.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaKernel::CudaKernel(CudaModule* module, CUfunction function)
:   m_module            (module),
    m_function          (function),
    m_preferL1          (true),
    m_sharedBankSize    (4),
    m_async             (false),
    m_stream            (NULL),
    m_gridSize          (1, 1),
    m_blockSize         (1, 1)
{
}

//------------------------------------------------------------------------

CudaKernel::~CudaKernel(void)
{
}

//------------------------------------------------------------------------

int CudaKernel::getAttribute(CUfunction_attribute attrib) const
{
    int value = 0;
#if (!FW_USE_CUDA)
    FW_UNREF(attrib);
#elif (CUDA_VERSION >= 2020)
    if (m_function && CudaModule::getDriverVersion() >= 22)
        CudaModule::checkError("cuFuncGetAttribute", cuFuncGetAttribute(&value, attrib, m_function));
#endif
    return value;
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::setParams(const Param* const* params, int numParams)
{
    FW_ASSERT(numParams == 0 || params);
    FW_ASSERT(numParams >= 0);

    int size = 0;
    for (int i = 0; i < numParams; i++)
    {
        size = (size + params[i]->align - 1) & -params[i]->align;
        size += params[i]->size;
    }
    m_params.reset(size);

    int ofs = 0;
    for (int i = 0; i < numParams; i++)
    {
        ofs = (ofs + params[i]->align - 1) & -params[i]->align;
        memcpy(m_params.getPtr(ofs), params[i]->value, params[i]->size);
        ofs += params[i]->size;
    }
    return *this;
}

//------------------------------------------------------------------------

Vec2i CudaKernel::getDefaultBlockSize(void) const
{
    int arch = CudaModule::getComputeCapability();
#if (CUDA_VERSION >= 2020)
    int driver = CudaModule::getDriverVersion();
#endif

    // Details available => choose smallest block that reaches maximal occupancy.

#if (CUDA_VERSION >= 2020)
    if (m_function && driver >= 22)
    {
        int warpSize            = max(CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE), 1);
        int warpRounding        = 2;
        int maxBlocksPerSM      = 8;
        int maxSharedPerSM      = CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK);
        int maxThreadsPerBlock  = getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        int maxThreadsPerSM     = (arch >= 20 && arch < 30) ? 1536 : maxThreadsPerBlock;
        int sharedPerBlock      = getAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);

#if (CUDA_VERSION >= 4000)
        if (driver >= 40)
            maxThreadsPerSM = CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
#endif

        if (arch >= 20 && m_preferL1 && sharedPerBlock <= maxSharedPerSM / 3)
            maxSharedPerSM /= 3;

        int numBlocks = maxBlocksPerSM;
        if (sharedPerBlock > 0)
            numBlocks = min(numBlocks, maxSharedPerSM / sharedPerBlock);
        numBlocks = min(numBlocks, maxThreadsPerSM / warpSize / warpRounding);
        numBlocks = max(numBlocks, 1);

        int numWarps = maxThreadsPerBlock / warpSize;
        numWarps = min(numWarps, maxThreadsPerSM / numBlocks / warpSize);
        numWarps -= numWarps % warpRounding;
        numWarps = max(numWarps, 1);
        return Vec2i(warpSize, numWarps);
    }
#endif

    // Otherwise => guess based on GPU architecture.

    return Vec2i(32, (arch < 20) ? 2 : 4);
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::setGrid(int numThreads, const Vec2i& blockSize)
{
    FW_ASSERT(numThreads >= 0);
    m_blockSize = (min(blockSize) > 0) ? blockSize : getDefaultBlockSize();

    int maxGridWidth = 65536;
#if FW_USE_CUDA
    int tmp = CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    if (tmp != 0)
        maxGridWidth = tmp;
#endif

    int threadsPerBlock = m_blockSize.x * m_blockSize.y;
    m_gridSize = Vec2i((numThreads + threadsPerBlock - 1) / threadsPerBlock, 1);
    while (m_gridSize.x > maxGridWidth)
    {
        m_gridSize.x = (m_gridSize.x + 1) >> 1;
        m_gridSize.y <<= 1;
    }

    return *this;
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::setGrid(const Vec2i& sizeThreads, const Vec2i& blockSize)
{
    FW_ASSERT(min(sizeThreads) >= 0);
    m_blockSize = (min(blockSize) > 0) ? blockSize : getDefaultBlockSize();
    m_gridSize = (sizeThreads + m_blockSize - 1) / m_blockSize;
    return *this;
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::launch(void)
{
    if (prepareLaunch())
        performLaunch();
    return *this;
}

//------------------------------------------------------------------------

F32 CudaKernel::launchTimed(bool yield)
{
    // Prepare and sync before timing.

    if (!prepareLaunch())
        return 0.0f;
    sync(false); // wait is short => spin

    // Events not supported => use CPU-based timer.

    CUevent startEvent = CudaModule::getStartEvent();
    CUevent endEvent = CudaModule::getEndEvent();

    if (!startEvent)
    {
        Timer timer(true);
        performLaunch();
        sync(false); // need accurate timing => spin
        return timer.getElapsed();
    }

    // Launch and record events.

    CudaModule::checkError("cuEventRecord", cuEventRecord(startEvent, NULL));
    performLaunch();
    CudaModule::checkError("cuEventRecord", cuEventRecord(endEvent, NULL));
    sync(yield);

    // Query GPU time between the events.

    F32 time = 0.0f;
    CudaModule::checkError("cuEventElapsedTime", cuEventElapsedTime(&time, startEvent, endEvent));
    return time * 1.0e-3f;
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::sync(bool yield)
{
    CudaModule::sync(yield);
    return *this;
}

//------------------------------------------------------------------------

bool CudaKernel::prepareLaunch(void)
{
    // Nothing to do => skip.

    if (!m_module || !m_function || min(m_gridSize) == 0)
        return false;

    // Set parameters.

    CudaModule::checkError("cuParamSetSize", cuParamSetSize(m_function, m_params.getSize()));
    if (m_params.getSize())
        CudaModule::checkError("cuParamSetv", cuParamSetv(m_function, 0, m_params.getPtr(), m_params.getSize()));

    // Set L1 and shared memory configuration.

#if (CUDA_VERSION >= 3000)
    if (isAvailable_cuFuncSetCacheConfig())
        CudaModule::checkError("cuFuncSetCacheConfig", cuFuncSetCacheConfig(m_function,
            (m_preferL1) ? CU_FUNC_CACHE_PREFER_L1 : CU_FUNC_CACHE_PREFER_SHARED));
#endif

#if (CUDA_VERSION >= 4020)
    if (isAvailable_cuFuncSetSharedMemConfig())
        CudaModule::checkError("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(m_function,
            (m_sharedBankSize == 4) ? CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE : CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));
#endif

    // Set block size.

    CudaModule::checkError("cuFuncSetBlockShape", cuFuncSetBlockShape(m_function, m_blockSize.x, m_blockSize.y, 1));

    // Update globals.

    m_module->updateGlobals();
    m_module->updateTexRefs(m_function);
    return true;
}

//------------------------------------------------------------------------

void CudaKernel::performLaunch(void)
{
    if (m_async && isAvailable_cuLaunchGridAsync())
        CudaModule::checkError("cuLaunchGridAsync", cuLaunchGridAsync(m_function, m_gridSize.x, m_gridSize.y, m_stream));
    else
        CudaModule::checkError("cuLaunchGrid", cuLaunchGrid(m_function, m_gridSize.x, m_gridSize.y));
}

//------------------------------------------------------------------------

CudaKernel& CudaKernel::operator=(const CudaKernel& other)
{
    m_module    = other.m_module;
    m_function  = other.m_function;
    m_params    = other.m_params;
    m_preferL1  = other.m_preferL1;
    m_async     = other.m_async;
    m_stream    = other.m_stream;
    m_gridSize  = other.m_gridSize;
    m_blockSize = other.m_blockSize;
    return *this;
}

//------------------------------------------------------------------------
