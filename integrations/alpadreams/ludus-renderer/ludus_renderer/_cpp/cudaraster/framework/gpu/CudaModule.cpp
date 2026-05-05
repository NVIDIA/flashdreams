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

#include "gpu/CudaModule.hpp"
#include "base/Thread.hpp"
#include "base/Timer.hpp"
#include "gpu/CudaCompiler.hpp"
#include "gui/Image.hpp"

using namespace FW;

//------------------------------------------------------------------------

bool        CudaModule::s_inited        = false;
bool        CudaModule::s_available     = false;
CUdevice    CudaModule::s_device        = NULL;
CUcontext   CudaModule::s_context       = NULL;
CUevent     CudaModule::s_startEvent    = NULL;
CUevent     CudaModule::s_endEvent      = NULL;

//------------------------------------------------------------------------

CudaModule::CudaModule(const void* cubin)
{
    staticInit();
    checkError("cuModuleLoadData", cuModuleLoadData(&m_module, cubin));
}

//------------------------------------------------------------------------

CudaModule::CudaModule(const String& cubinFile)
{
    staticInit();
    checkError("cuModuleLoad", cuModuleLoad(&m_module, cubinFile.getPtr()));
}

//------------------------------------------------------------------------

CudaModule::~CudaModule(void)
{
    for (int i = 0; i < m_globals.getSize(); i++)
        delete m_globals[i];

    checkError("cuModuleUnload", cuModuleUnload(m_module));
}

//------------------------------------------------------------------------

bool CudaModule::hasKernel(const String& name)
{
    return (findKernel(name) != NULL);
}

//------------------------------------------------------------------------

CudaKernel CudaModule::getKernel(const String& name)
{
    CUfunction kernel = findKernel(name);
    if (!kernel)
        fail("CudaModule: Kernel not found '%s'!", name.getPtr());
    return CudaKernel(this, kernel);
}

//------------------------------------------------------------------------

Buffer& CudaModule::getGlobal(const String& name)
{
    S32* found = m_globalHash.search(name);
    if (found)
        return *m_globals[*found];

    CUdeviceptr ptr;
    CUsize_t size;
    checkError("cuModuleGetGlobal", cuModuleGetGlobal(&ptr, &size, m_module, name.getPtr()));

    Buffer* buffer = new Buffer;
    buffer->wrapCuda(ptr, size);

    m_globalHash.add(name, m_globals.getSize());
    m_globals.add(buffer);
    return *buffer;
}

//------------------------------------------------------------------------

void CudaModule::updateGlobals(bool async, CUstream stream)
{
    for (int i = 0; i < m_globals.getSize(); i++)
        m_globals[i]->setOwner(Buffer::Cuda, true, async, stream);
}

//------------------------------------------------------------------------

CUtexref CudaModule::getTexRef(const String& name)
{
    S32* found = m_texRefHash.search(name);
    if (found)
        return m_texRefs[*found];

    CUtexref texRef;
    checkError("cuModuleGetTexRef", cuModuleGetTexRef(&texRef, m_module, name.getPtr()));

    m_texRefHash.add(name, m_texRefs.getSize());
    m_texRefs.add(texRef);
    return texRef;
}

//------------------------------------------------------------------------

void CudaModule::setTexRefMode(CUtexref texRef, bool wrap, bool bilinear, bool normalizedCoords, bool readAsInt)
{
#if (!FW_USE_CUDA)

    FW_UNREF(texRef);
    FW_UNREF(wrap);
    FW_UNREF(bilinear);
    FW_UNREF(normalizedCoords);
    FW_UNREF(readAsInt);
    fail("CudaModule::setTexRefMode(): Built without FW_USE_CUDA!");

#else

    CUaddress_mode addressMode = (wrap) ? CU_TR_ADDRESS_MODE_WRAP : CU_TR_ADDRESS_MODE_CLAMP;
    CUfilter_mode filterMode = (bilinear) ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;

    U32 flags = 0;
    if (normalizedCoords)
        flags |= CU_TRSF_NORMALIZED_COORDINATES;
    if (readAsInt)
        flags |= CU_TRSF_READ_AS_INTEGER;

    for (int dim = 0; dim < 3; dim++)
        checkError("cuTexRefSetAddressMode", cuTexRefSetAddressMode(texRef, dim, addressMode));
    checkError("cuTexRefSetFilterMode", cuTexRefSetFilterMode(texRef, filterMode));
    checkError("cuTexRefSetFlags", cuTexRefSetFlags(texRef, flags));

#endif
}

//------------------------------------------------------------------------

void CudaModule::setTexRef(const String& name, Buffer& buf, CUarray_format format, int numComponents)
{
    setTexRef(name, buf.getCudaPtr(), buf.getSize(), format, numComponents);
}

//------------------------------------------------------------------------

void CudaModule::setTexRef(const String& name, CUdeviceptr ptr, S64 size, CUarray_format format, int numComponents)
{
    CUtexref texRef = getTexRef(name);
    checkError("cuTexRefSetFormat", cuTexRefSetFormat(texRef, format, numComponents));
    checkError("cuTexRefSetAddress", cuTexRefSetAddress(NULL, texRef, ptr, (U32)size));
}

//------------------------------------------------------------------------

void CudaModule::setTexRef(const String& name, CUarray cudaArray, bool wrap, bool bilinear, bool normalizedCoords, bool readAsInt)
{
    CUtexref texRef = getTexRef(name);
    setTexRefMode(texRef, wrap, bilinear, normalizedCoords, readAsInt);

#if FW_USE_CUDA
    checkError("cuTexRefSetArray", cuTexRefSetArray(texRef, cudaArray, CU_TRSA_OVERRIDE_FORMAT));
#else
    FW_UNREF(cudaArray);
#endif
}

//------------------------------------------------------------------------

void CudaModule::setTexRef(const String& name, const Image& image, bool wrap, bool bilinear, bool normalizedCoords, bool readAsInt)
{
    FW_UNREF(name);
    FW_UNREF(image);
    FW_UNREF(wrap);
    FW_UNREF(bilinear);
    FW_UNREF(normalizedCoords);
    FW_UNREF(readAsInt);

#if (!FW_USE_CUDA)

    fail("CudaModule::setTexRef(Image): Built without FW_USE_CUDA!");

#elif (CUDA_VERSION < 2020)

    fail("CudaModule: setTexRef(Image) requires CUDA 2.2 or later!");

#else

    CUDA_ARRAY_DESCRIPTOR desc;
    ImageFormat format = image.chooseCudaFormat(&desc);
    if (format != image.getFormat())
        fail("CudaModule: Unsupported image format in setTexRef(Image)!");

    CUtexref texRef = getTexRef(name);
    setTexRefMode(texRef, wrap, bilinear, normalizedCoords, readAsInt);
    checkError("cuTexRefSetAddress2D", cuTexRefSetAddress2D(texRef, &desc, image.getBuffer().getCudaPtr(), (size_t)image.getStride()));

#endif
}

//------------------------------------------------------------------------

void CudaModule::unsetTexRef(const String& name)
{
    CUtexref texRef = getTexRef(name);
    checkError("cuTexRefSetAddress", cuTexRefSetAddress(NULL, texRef, NULL, 0));
}

//------------------------------------------------------------------------

void CudaModule::updateTexRefs(CUfunction kernel)
{
#if (!FW_USE_CUDA)

    FW_UNREF(kernel);
    fail("CudaModule::updateTexRefs(): Built without FW_USE_CUDA!");

#else

    if (getDriverVersion() >= 32)
        return;

    for (int i = 0; i < m_texRefs.getSize(); i++)
        checkError("cuParamSetTexRef", cuParamSetTexRef(kernel, CU_PARAM_TR_DEFAULT, m_texRefs[i]));
#endif
}

//------------------------------------------------------------------------

CUsurfref CudaModule::getSurfRef(const String& name)
{
#if (CUDA_VERSION >= 3010)
    CUsurfref surfRef;
    checkError("cuModuleGetSurfRef", cuModuleGetSurfRef(&surfRef, m_module, name.getPtr()));
    return surfRef;
#else
    FW_UNREF(name);
    fail("CudaModule: getSurfRef() requires CUDA 3.1 or later!");
    return NULL;
#endif
}

//------------------------------------------------------------------------

void CudaModule::setSurfRef(const String& name, CUarray cudaArray)
{
#if (CUDA_VERSION >= 3010)
    checkError("cuSurfRefSetArray", cuSurfRefSetArray(getSurfRef(name), cudaArray, 0));
#else
    FW_UNREF(name);
    FW_UNREF(cudaArray);
    fail("CudaModule: setSurfRef() requires CUDA 3.1 or later!");
#endif
}

//------------------------------------------------------------------------

void CudaModule::staticInit(void)
{
    if (s_inited)
        return;
    s_inited = true;
    s_available = false;

    if (!isAvailable_cuInit())
        return;

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS)
    {
#if FW_USE_CUDA
        if (res != CUDA_ERROR_NO_DEVICE)
            checkError("cuInit", res);
#endif
        return;
    }

    s_available = true;
    s_device = selectDevice();
    printDeviceInfo(s_device);

    U32 flags = 0;
#if FW_USE_CUDA
    flags |= CU_CTX_SCHED_SPIN; // use sync() if you want to yield
#endif
#if (CUDA_VERSION >= 2030)
    if (getDriverVersion() >= 23)
        flags |= CU_CTX_LMEM_RESIZE_TO_MAX; // reduce launch overhead with large localmem
#endif

    if (!isAvailable_cuGLCtxCreate())
        checkError("cuCtxCreate", cuCtxCreate(&s_context, flags, s_device));
    else
    {
        GLContext::staticInit();
        checkError("cuGLCtxCreate", cuGLCtxCreate(&s_context, flags, s_device));
    }

    if (isAvailable_cuEventCreate())
    {
        checkError("cuEventCreate", cuEventCreate(&s_startEvent, 0));
        checkError("cuEventCreate", cuEventCreate(&s_endEvent, 0));
    }
}

//------------------------------------------------------------------------

void CudaModule::staticDeinit(void)
{
    if (!s_inited)
        return;
    s_inited = false;

    if (s_startEvent)
        checkError("cuEventDestroy", cuEventDestroy(s_startEvent));
    s_startEvent = NULL;

    if (s_endEvent)
        checkError("cuEventDestroy", cuEventDestroy(s_endEvent));
    s_endEvent = NULL;

    if (s_context)
        checkError("cuCtxDestroy", cuCtxDestroy(s_context));
    s_context = NULL;

    s_device = NULL;
}

//------------------------------------------------------------------------

S64 CudaModule::getMemoryUsed(void)
{
    staticInit();
    if (!s_available)
        return 0;

    CUsize_t free = 0;
    CUsize_t total = 0;
    cuMemGetInfo(&free, &total);
    return total - free;
}

//------------------------------------------------------------------------

void CudaModule::sync(bool yield)
{
    if (!s_inited)
        return;

    if (!yield || !s_endEvent)
    {
            checkError("cuCtxSynchronize", cuCtxSynchronize());
        return;
    }

#if FW_USE_CUDA
    checkError("cuEventRecord", cuEventRecord(s_endEvent, NULL));
    for (;;)
    {
        CUresult res = cuEventQuery(s_endEvent);
        if (res != CUDA_ERROR_NOT_READY)
        {
            checkError("cuEventQuery", res);
            break;
        }
        Thread::yield();
    }
#endif
}

//------------------------------------------------------------------------

const char* CudaModule::decodeError(CUresult res)
{
    const char* error;
    switch (res)
    {
    default:                                        error = "Unknown CUresult"; break;
    case CUDA_SUCCESS:                              error = "No error"; break;

#if FW_USE_CUDA
    case CUDA_ERROR_INVALID_VALUE:                  error = "Invalid value"; break;
    case CUDA_ERROR_OUT_OF_MEMORY:                  error = "Out of memory"; break;
    case CUDA_ERROR_NOT_INITIALIZED:                error = "Not initialized"; break;
    case CUDA_ERROR_DEINITIALIZED:                  error = "Deinitialized"; break;
    case CUDA_ERROR_NO_DEVICE:                      error = "No device"; break;
    case CUDA_ERROR_INVALID_DEVICE:                 error = "Invalid device"; break;
    case CUDA_ERROR_INVALID_IMAGE:                  error = "Invalid image"; break;
    case CUDA_ERROR_INVALID_CONTEXT:                error = "Invalid context"; break;
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        error = "Context already current"; break;
    case CUDA_ERROR_MAP_FAILED:                     error = "Map failed"; break;
    case CUDA_ERROR_UNMAP_FAILED:                   error = "Unmap failed"; break;
    case CUDA_ERROR_ARRAY_IS_MAPPED:                error = "Array is mapped"; break;
    case CUDA_ERROR_ALREADY_MAPPED:                 error = "Already mapped"; break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:              error = "No binary for GPU"; break;
    case CUDA_ERROR_ALREADY_ACQUIRED:               error = "Already acquired"; break;
    case CUDA_ERROR_NOT_MAPPED:                     error = "Not mapped"; break;
    case CUDA_ERROR_INVALID_SOURCE:                 error = "Invalid source"; break;
    case CUDA_ERROR_FILE_NOT_FOUND:                 error = "File not found"; break;
    case CUDA_ERROR_INVALID_HANDLE:                 error = "Invalid handle"; break;
    case CUDA_ERROR_NOT_FOUND:                      error = "Not found"; break;
    case CUDA_ERROR_NOT_READY:                      error = "Not ready"; break;
    case CUDA_ERROR_LAUNCH_FAILED:                  error = "Launch failed"; break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        error = "Launch out of resources"; break;
    case CUDA_ERROR_LAUNCH_TIMEOUT:                 error = "Launch timeout"; break;
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  error = "Launch incompatible texturing"; break;
    case CUDA_ERROR_UNKNOWN:                        error = "Unknown error"; break;
#endif

#if (CUDA_VERSION >= 4000) // TODO: Some of these may exist in earlier versions, too.
    case CUDA_ERROR_PROFILER_DISABLED:              error = "Profiler disabled"; break;
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       error = "Profiler not initialized"; break;
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:       error = "Profiler already started"; break;
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       error = "Profiler already stopped"; break;
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            error = "Not mapped as array"; break;
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          error = "Not mapped as pointer"; break;
    case CUDA_ERROR_ECC_UNCORRECTABLE:              error = "ECC uncorrectable"; break;
    case CUDA_ERROR_UNSUPPORTED_LIMIT:              error = "Unsupported limit"; break;
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         error = "Context already in use"; break;
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: error = "Shared object symbol not found"; break;
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      error = "Shared object init failed"; break;
    case CUDA_ERROR_OPERATING_SYSTEM:               error = "Operating system error"; break;
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    error = "Peer access already enabled"; break;
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        error = "Peer access not enabled"; break;
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         error = "Primary context active"; break;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:           error = "Context is destroyed"; break;
#endif
    }
    return error;
}

//------------------------------------------------------------------------

void CudaModule::checkError(const char* funcName, CUresult res)
{
    if (res != CUDA_SUCCESS)
        fail("%s() failed: %s!", funcName, decodeError(res));
}

//------------------------------------------------------------------------

int CudaModule::getDriverVersion(void)
{
    int version = 2010;
#if (CUDA_VERSION >= 2020)
    if (isAvailable_cuDriverGetVersion())
        cuDriverGetVersion(&version);
#endif
    version /= 10;
    return version / 10 + version % 10;
}

//------------------------------------------------------------------------

int CudaModule::getComputeCapability(void)
{
    staticInit();
    if (!s_available)
        return 10;

    int major;
    int minor;
    checkError("cuDeviceComputeCapability", cuDeviceComputeCapability(&major, &minor, s_device));
    return major * 10 + minor;
}

//------------------------------------------------------------------------

int CudaModule::getDeviceAttribute(CUdevice_attribute attrib)
{
    staticInit();
    if (!s_available)
        return 0;

    int value;
    checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&value, attrib, s_device));
    return value;
}

//------------------------------------------------------------------------

CUdevice CudaModule::selectDevice(void)
{
#if (!FW_USE_CUDA)

    fail("Image::chooseCudaFormat(): Built without FW_USE_CUDA!");
    return 0;

#else

    int numDevices;
    CUdevice device = NULL;
    S32 bestScore = FW_S32_MIN;
    checkError("cuDeviceGetCount", cuDeviceGetCount(&numDevices));

    for (int i = 0; i < numDevices; i++)
    {
        CUdevice dev;
        checkError("cuDeviceGet", cuDeviceGet(&dev, i)); // TODO: Use cuGLGetDevices() on CUDA 4.1+.

        int clockRate;
        int numProcessors;
        checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
        checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&numProcessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

        S32 score = clockRate * numProcessors;
        if (score > bestScore)
        {
            device = dev;
            bestScore = score;
        }
    }

    if (bestScore == FW_S32_MIN)
        fail("CudaModule: No appropriate CUDA device found!");
    return device;

#endif
}

//------------------------------------------------------------------------

void CudaModule::printDeviceInfo(CUdevice device)
{
#if (!FW_USE_CUDA)

    FW_UNREF(device);
    fail("CudaModule::printDeviceInfo(): Built without FW_USE_CUDA!");

#else

    static const struct
    {
        CUdevice_attribute  attrib;
        const char*         name;
    } attribs[] =
    {
#define A21(ENUM, NAME) { CU_DEVICE_ATTRIBUTE_ ## ENUM, NAME },
#if (CUDA_VERSION >= 4000)
#   define A40(ENUM, NAME) A21(ENUM, NAME)
#else
#   define A40(ENUM, NAME) // TODO: Some of these may exist in earlier versions, too.
#endif

        A21(CLOCK_RATE,                         "Clock rate")
        A40(MEMORY_CLOCK_RATE,                  "Memory clock rate")
        A21(MULTIPROCESSOR_COUNT,               "Number of SMs")
//      A40(GLOBAL_MEMORY_BUS_WIDTH,            "DRAM bus width")
//      A40(L2_CACHE_SIZE,                      "L2 cache size")

        A21(MAX_THREADS_PER_BLOCK,              "Max threads per block")
        A40(MAX_THREADS_PER_MULTIPROCESSOR,     "Max threads per SM")
        A21(REGISTERS_PER_BLOCK,                "Max registers per block")
        A21(SHARED_MEMORY_PER_BLOCK,            "Max shared mem per block")
        A21(TOTAL_CONSTANT_MEMORY,              "Constant memory")
//      A21(WARP_SIZE,                          "Warp size")

        A21(MAX_BLOCK_DIM_X,                    "Max blockDim.x")
//      A21(MAX_BLOCK_DIM_Y,                    "Max blockDim.y")
//      A21(MAX_BLOCK_DIM_Z,                    "Max blockDim.z")
        A21(MAX_GRID_DIM_X,                     "Max gridDim.x")
//      A21(MAX_GRID_DIM_Y,                     "Max gridDim.y")
//      A21(MAX_GRID_DIM_Z,                     "Max gridDim.z")
//      A40(MAXIMUM_TEXTURE1D_WIDTH,            "Max tex1D.x")
//      A40(MAXIMUM_TEXTURE2D_WIDTH,            "Max tex2D.x")
//      A40(MAXIMUM_TEXTURE2D_HEIGHT,           "Max tex2D.y")
//      A40(MAXIMUM_TEXTURE3D_WIDTH,            "Max tex3D.x")
//      A40(MAXIMUM_TEXTURE3D_HEIGHT,           "Max tex3D.y")
//      A40(MAXIMUM_TEXTURE3D_DEPTH,            "Max tex3D.z")
//      A40(MAXIMUM_TEXTURE1D_LAYERED_WIDTH,    "Max layerTex1D.x")
//      A40(MAXIMUM_TEXTURE1D_LAYERED_LAYERS,   "Max layerTex1D.y")
//      A40(MAXIMUM_TEXTURE2D_LAYERED_WIDTH,    "Max layerTex2D.x")
//      A40(MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,   "Max layerTex2D.y")
//      A40(MAXIMUM_TEXTURE2D_LAYERED_LAYERS,   "Max layerTex2D.z")
//      A40(MAXIMUM_TEXTURE2D_ARRAY_WIDTH,      "Max array.x")
//      A40(MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,     "Max array.y")
//      A40(MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,  "Max array.z")

//      A21(MAX_PITCH,                          "Max memcopy pitch")
//      A21(TEXTURE_ALIGNMENT,                  "Texture alignment")
//      A40(SURFACE_ALIGNMENT,                  "Surface alignment")

        A40(CONCURRENT_KERNELS,                 "Concurrent launches supported")
        A21(GPU_OVERLAP,                        "Concurrent memcopy supported")
        A40(ASYNC_ENGINE_COUNT,                 "Max concurrent memcopies")
//      A40(KERNEL_EXEC_TIMEOUT,                "Kernel launch time limited")
//      A40(INTEGRATED,                         "Integrated with host memory")
        A40(UNIFIED_ADDRESSING,                 "Unified addressing supported")
        A40(CAN_MAP_HOST_MEMORY,                "Can map host memory")
        A40(ECC_ENABLED,                        "ECC enabled")

//      A40(TCC_DRIVER,                         "Driver is TCC")
//      A40(COMPUTE_MODE,                       "Compute exclusivity mode")

//      A40(PCI_BUS_ID,                         "PCI bus ID")
//      A40(PCI_DEVICE_ID,                      "PCI device ID")
//      A40(PCI_DOMAIN_ID,                      "PCI domain ID")

#undef A21
#undef A40
    };

    char name[256];
    int major;
    int minor;
    CUsize_t memory;

    checkError("cuDeviceGetName", cuDeviceGetName(name, FW_ARRAY_SIZE(name) - 1, device));
    checkError("cuDeviceComputeCapability", cuDeviceComputeCapability(&major, &minor, device));
    checkError("cuDeviceTotalMem", cuDeviceTotalMem(&memory, device));
    name[FW_ARRAY_SIZE(name) - 1] = '\0';

    printf("\n");
    printf("%-32s%s\n", sprintf("CUDA device %d", (int)device).getPtr(), name);
    printf("%-32s%s\n", "---", "---");
    printf("%-32s%d.%d\n", "Compute capability", major, minor);
    printf("%-32s%.0f megs\n", "Total memory", (F32)memory * exp2(-20));

    for (int i = 0; i < (int)FW_ARRAY_SIZE(attribs); i++)
    {
        int value;
        if (cuDeviceGetAttribute(&value, attribs[i].attrib, device) == CUDA_SUCCESS)
            printf("%-32s%d\n", attribs[i].name, value);
    }
    printf("\n");

#endif
}

//------------------------------------------------------------------------

Vec2i CudaModule::selectGridSize(int numBlocks)
{
    int maxWidth = 65536;
#if FW_USE_CUDA
    checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&maxWidth, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, s_device));
#endif

    Vec2i size(numBlocks, 1);
    while (size.x > maxWidth)
    {
        size.x = (size.x + 1) >> 1;
        size.y <<= 1;
    }
    return size;
}

//------------------------------------------------------------------------

CUfunction CudaModule::findKernel(const String& name)
{
    // Search from hash.

    CUfunction* found = m_kernels.search(name);
    if (found)
        return *found;

    // Search from module.

    CUfunction kernel = NULL;
    cuModuleGetFunction(&kernel, m_module, name.getPtr());
    if (!kernel)
        cuModuleGetFunction(&kernel, m_module, (String("__globfunc_") + name).getPtr());
    if (!kernel)
        return NULL;

    // Add to hash.

    m_kernels.add(name, kernel);
    return kernel;
}

//------------------------------------------------------------------------
