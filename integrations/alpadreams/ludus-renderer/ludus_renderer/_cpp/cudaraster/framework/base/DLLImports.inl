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

//------------------------------------------------------------------------

#ifndef FW_DLL_IMPORT_RETV
#   define FW_DLL_IMPORT_RETV(RET, CALL, NAME, PARAMS, PASS)
#   define FW_DLL_IMPORT_VOID(RET, CALL, NAME, PARAMS, PASS)
#   define FW_DLL_DECLARE_RETV(RET, CALL, NAME, PARAMS, PASS)
#   define FW_DLL_DECLARE_VOID(RET, CALL, NAME, PARAMS, PASS)
#   define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)
#   define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)
#endif

//------------------------------------------------------------------------
// CUDA Driver API
//------------------------------------------------------------------------

// CUDA 2.1

FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuInit,                                 (unsigned int Flags), (Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGet,                            (CUdevice* device, int ordinal), (device, ordinal))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetCount,                       (int* count), (count))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetName,                        (char* name, int len, CUdevice dev), (name, len, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceComputeCapability,              (int* major, int* minor, CUdevice dev), (major, minor, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetProperties,                  (CUdevprop* prop, CUdevice dev), (prop, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetAttribute,                   (int* pi, CUdevice_attribute attrib, CUdevice dev), (pi, attrib, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxDestroy,                           (CUcontext ctx), (ctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxAttach,                            (CUcontext* pctx, unsigned int flags), (pctx, flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxDetach,                            (CUcontext ctx), (ctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxPushCurrent,                       (CUcontext ctx), (ctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxPopCurrent,                        (CUcontext* pctx), (pctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetDevice,                         (CUdevice* device), (device))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxSynchronize,                       (void), ())
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleLoad,                           (CUmodule* module, const char* fname), (module, fname))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleLoadData,                       (CUmodule* module, const void* image), (module, image))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleLoadDataEx,                     (CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues), (module, image, numOptions, options, optionValues))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleLoadFatBinary,                  (CUmodule* module, const void* fatCubin), (module, fatCubin))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleUnload,                         (CUmodule hmod), (hmod))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleGetFunction,                    (CUfunction* hfunc, CUmodule hmod, const char* name), (hfunc, hmod, name))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleGetTexRef,                      (CUtexref* pTexRef, CUmodule hmod, const char* name), (pTexRef, hmod, name))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemFreeHost,                          (void* p), (p))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuFuncSetBlockShape,                    (CUfunction hfunc, int x, int y, int z), (hfunc, x, y, z))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuFuncSetSharedSize,                    (CUfunction hfunc, unsigned int bytes), (hfunc, bytes))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuArrayDestroy,                         (CUarray hArray), (hArray))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefCreate,                         (CUtexref* pTexRef), (pTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefDestroy,                        (CUtexref hTexRef), (hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetArray,                       (CUtexref hTexRef, CUarray hArray, unsigned int Flags), (hTexRef, hArray, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetFormat,                      (CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents), (hTexRef, fmt, NumPackedComponents))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetAddressMode,                 (CUtexref hTexRef, int dim, CUaddress_mode am), (hTexRef, dim, am))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetFilterMode,                  (CUtexref hTexRef, CUfilter_mode fm), (hTexRef, fm))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetFlags,                       (CUtexref hTexRef, unsigned int Flags), (hTexRef, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetArray,                       (CUarray* phArray, CUtexref hTexRef), (phArray, hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetAddressMode,                 (CUaddress_mode* pam, CUtexref hTexRef, int dim), (pam, hTexRef, dim))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetFilterMode,                  (CUfilter_mode* pfm, CUtexref hTexRef), (pfm, hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetFormat,                      (CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef), (pFormat, pNumChannels, hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetFlags,                       (unsigned int* pFlags, CUtexref hTexRef), (pFlags, hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuParamSetSize,                         (CUfunction hfunc, unsigned int numbytes), (hfunc, numbytes))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuParamSeti,                            (CUfunction hfunc, int offset, unsigned int value), (hfunc, offset, value))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuParamSetf,                            (CUfunction hfunc, int offset, float value), (hfunc, offset, value))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuParamSetv,                            (CUfunction hfunc, int offset, void* ptr, unsigned int numbytes), (hfunc, offset, ptr, numbytes))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuParamSetTexRef,                       (CUfunction hfunc, int texunit, CUtexref hTexRef), (hfunc, texunit, hTexRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuLaunch,                               (CUfunction f), (f))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuLaunchGrid,                           (CUfunction f, int grid_width, int grid_height), (f, grid_width, grid_height))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuLaunchGridAsync,                      (CUfunction f, int grid_width, int grid_height, CUstream hStream), (f, grid_width, grid_height, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventCreate,                          (CUevent* phEvent, unsigned int Flags), (phEvent, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventRecord,                          (CUevent hEvent, CUstream hStream), (hEvent, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventQuery,                           (CUevent hEvent), (hEvent))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventSynchronize,                     (CUevent hEvent), (hEvent))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventDestroy,                         (CUevent hEvent), (hEvent))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuEventElapsedTime,                     (float* pMilliseconds, CUevent hStart, CUevent hEnd), (pMilliseconds, hStart, hEnd))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuStreamCreate,                         (CUstream* phStream, unsigned int Flags), (phStream, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuStreamQuery,                          (CUstream hStream), (hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuStreamSynchronize,                    (CUstream hStream), (hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuStreamDestroy,                        (CUstream hStream), (hStream))

#if (CUDA_VERSION < 3020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceTotalMem,                       (unsigned int* bytes, CUdevice dev), (bytes, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxCreate,                            (CUcontext* pctx, unsigned int flags, CUdevice dev), (pctx, flags, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleGetGlobal,                      (CUdeviceptr* dptr, unsigned int* bytes, CUmodule hmod, const char* name), (dptr, bytes, hmod, name))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemGetInfo,                           (unsigned int* free, unsigned int* total), (free, total))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemAlloc,                             (CUdeviceptr* dptr, unsigned int bytesize), (dptr, bytesize))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemAllocPitch,                        (CUdeviceptr* dptr, unsigned int* pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes), (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemFree,                              (CUdeviceptr dptr), (dptr))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemGetAddressRange,                   (CUdeviceptr* pbase, unsigned int* psize, CUdeviceptr dptr), (pbase, psize, dptr))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemAllocHost,                         (void** pp, unsigned int bytesize), (pp, bytesize))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyHtoD,                           (CUdeviceptr dstDevice, const void* srcHost, unsigned int ByteCount), (dstDevice, srcHost, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyDtoH,                           (void* dstHost, CUdeviceptr srcDevice, unsigned int ByteCount), (dstHost, srcDevice, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyDtoD,                           (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount), (dstDevice, srcDevice, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyDtoA,                           (CUarray dstArray, unsigned int dstIndex, CUdeviceptr srcDevice, unsigned int ByteCount), (dstArray, dstIndex, srcDevice, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyAtoD,                           (CUdeviceptr dstDevice, CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount), (dstDevice, hSrc, SrcIndex, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyHtoA,                           (CUarray dstArray, unsigned int dstIndex, const void* pSrc, unsigned int ByteCount), (dstArray, dstIndex, pSrc, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyAtoH,                           (void* dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount), (dstHost, srcArray, srcIndex, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyAtoA,                           (CUarray dstArray, unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount), (dstArray, dstIndex, srcArray, srcIndex, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyHtoAAsync,                      (CUarray dstArray, unsigned int dstIndex, const void* pSrc, unsigned int ByteCount, CUstream hStream), (dstArray, dstIndex, pSrc, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyAtoHAsync,                      (void* dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount, CUstream hStream), (dstHost, srcArray, srcIndex, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy2D,                             (const CUDA_MEMCPY2D* pCopy), (pCopy))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy2DUnaligned,                    (const CUDA_MEMCPY2D* pCopy), (pCopy))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy3D,                             (const CUDA_MEMCPY3D* pCopy), (pCopy))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyHtoDAsync,                      (CUdeviceptr dstDevice, const void* srcHost, unsigned int ByteCount, CUstream hStream), (dstDevice, srcHost, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyDtoHAsync,                      (void* dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream), (dstHost, srcDevice, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy2DAsync,                        (const CUDA_MEMCPY2D* pCopy, CUstream hStream), (pCopy, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy3DAsync,                        (const CUDA_MEMCPY3D* pCopy, CUstream hStream), (pCopy, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD8,                             (CUdeviceptr dstDevice, unsigned char uc, unsigned int N), (dstDevice, uc, N))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD16,                            (CUdeviceptr dstDevice, unsigned short us, unsigned int N), (dstDevice, us, N))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD32,                            (CUdeviceptr dstDevice, unsigned int ui, unsigned int N), (dstDevice, ui, N))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D8,                           (CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height), (dstDevice, dstPitch, uc, Width, Height))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D16,                          (CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height), (dstDevice, dstPitch, us, Width, Height))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D32,                          (CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height), (dstDevice, dstPitch, ui, Width, Height))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuArrayCreate,                          (CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray), (pHandle, pAllocateArray))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuArrayGetDescriptor,                   (CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray), (pArrayDescriptor, hArray))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuArray3DCreate,                        (CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray), (pHandle, pAllocateArray))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuArray3DGetDescriptor,                 (CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray), (pArrayDescriptor, hArray))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetAddress,                     (unsigned int* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes), (ByteOffset, hTexRef, dptr, bytes))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefGetAddress,                     (CUdeviceptr* pdptr, CUtexref hTexRef), (pdptr, hTexRef))
#endif

// CUDA 2.2

#if (CUDA_VERSION >= 2020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDriverGetVersion,                     (int* driverVersion), (driverVersion))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemHostAlloc,                         (void** pp, size_t bytesize, unsigned int Flags), (pp, bytesize, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuFuncGetAttribute,                     (int* pi, CUfunction_attribute attrib, CUfunction hfunc), (pi, attrib, hfunc))
#endif

#if (CUDA_VERSION >= 2020 && CUDA_VERSION < 3020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemHostGetDevicePointer,              (CUdeviceptr* pdptr, void* p, unsigned int Flags), (pdptr, p, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuTexRefSetAddress2D,                   (CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, unsigned int Pitch), (hTexRef, desc, dptr, Pitch))
#endif

// CUDA 2.3

#if (CUDA_VERSION >= 2030)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemHostGetFlags,                      (unsigned int* pFlags, void* p), (pFlags, p))
#endif

// CUDA 3.0

#if (CUDA_VERSION >= 3000)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuFuncSetCacheConfig,                   (CUfunction hfunc, CUfunc_cache config), (hfunc, config))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsUnregisterResource,           (CUgraphicsResource resource), (resource))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsSubResourceGetMappedArray,    (CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel), (pArray, resource, arrayIndex, mipLevel))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsResourceSetMapFlags,          (CUgraphicsResource resource, unsigned int flags), (resource, flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsMapResources,                 (unsigned int count, CUgraphicsResource* resources, CUstream hStream), (count, resources, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsUnmapResources,               (unsigned int count, CUgraphicsResource* resources, CUstream hStream), (count, resources, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGetExportTable,                       (const void** ppExportTable, const CUuuid* pExportTableId), (ppExportTable, pExportTableId))
#endif

#if (CUDA_VERSION >= 3000 && CUDA_VERSION < 3020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyDtoDAsync,                      (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream), (dstDevice, srcDevice, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuGraphicsResourceGetMappedPointer,     (CUdeviceptr* pDevPtr, unsigned int* pSize, CUgraphicsResource resource), (pDevPtr, pSize, resource))
#endif

// CUDA 3.1

#if (CUDA_VERSION >= 3010)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuModuleGetSurfRef,                     (CUsurfref* pSurfRef, CUmodule hmod, const char* name), (pSurfRef, hmod, name))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuSurfRefSetArray,                      (CUsurfref hSurfRef, CUarray hArray, unsigned int Flags), (hSurfRef, hArray, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuSurfRefGetArray,                      (CUarray* phArray, CUsurfref hSurfRef), (phArray, hSurfRef))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxSetLimit,                          (CUlimit limit, size_t value), (limit, value))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetLimit,                          (size_t* pvalue, CUlimit limit), (pvalue, limit))
#endif

// CUDA 3.2

#if (CUDA_VERSION >= 3020)
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuDeviceTotalMem,                       (size_t* bytes, CUdevice dev), (bytes, dev))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuCtxCreate,                            (CUcontext* pctx, unsigned int flags, CUdevice dev), (pctx, flags, dev))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuModuleGetGlobal,                      (CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name), (dptr, bytes, hmod, name))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemGetInfo,                           (size_t* free, size_t* total), (free, total))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemAlloc,                             (CUdeviceptr* dptr, size_t bytesize), (dptr, bytesize))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemAllocPitch,                        (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes), (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemFree,                              (CUdeviceptr dptr), (dptr))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemGetAddressRange,                   (CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr), (pbase, psize, dptr))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemAllocHost,                         (void** pp, size_t bytesize), (pp, bytesize))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyHtoD,                           (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount), (dstDevice, srcHost, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyDtoH,                           (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount), (dstHost, srcDevice, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyDtoD,                           (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount), (dstDevice, srcDevice, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyDtoA,                           (CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount), (dstArray, dstOffset, srcDevice, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyAtoD,                           (CUdeviceptr dstDevice, CUarray hSrc, size_t srcOffset, size_t ByteCount), (dstDevice, hSrc, srcOffset, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyHtoA,                           (CUarray dstArray, size_t dstOffset, const void* pSrc, size_t ByteCount), (dstArray, dstOffset, pSrc, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyAtoH,                           (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount), (dstHost, srcArray, srcOffset, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyAtoA,                           (CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount), (dstArray, dstOffset, srcArray, srcOffset, ByteCount))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyHtoAAsync,                      (CUarray dstArray, size_t dstOffset, const void* pSrc, size_t ByteCount, CUstream hStream), (dstArray, dstOffset, pSrc, ByteCount, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyAtoHAsync,                      (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream), (dstHost, srcArray, srcOffset, ByteCount, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpy2D,                             (const CUDA_MEMCPY2D* pCopy), (pCopy))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpy2DUnaligned,                    (const CUDA_MEMCPY2D* pCopy), (pCopy))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpy3D,                             (const CUDA_MEMCPY3D* pCopy), (pCopy))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyHtoDAsync,                      (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream), (dstDevice, srcHost, ByteCount, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyDtoHAsync,                      (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream), (dstHost, srcDevice, ByteCount, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpy2DAsync,                        (const CUDA_MEMCPY2D* pCopy, CUstream hStream), (pCopy, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpy3DAsync,                        (const CUDA_MEMCPY3D* pCopy, CUstream hStream), (pCopy, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD8,                             (CUdeviceptr dstDevice, unsigned char uc, size_t N), (dstDevice, uc, N))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD16,                            (CUdeviceptr dstDevice, unsigned short us, size_t N), (dstDevice, us, N))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD32,                            (CUdeviceptr dstDevice, unsigned int ui, size_t N), (dstDevice, ui, N))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD2D8,                           (CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height), (dstDevice, dstPitch, uc, Width, Height))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD2D16,                          (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height), (dstDevice, dstPitch, us, Width, Height))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemsetD2D32,                          (CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height), (dstDevice, dstPitch, ui, Width, Height))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuArrayCreate,                          (CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray), (pHandle, pAllocateArray))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuArrayGetDescriptor,                   (CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray), (pArrayDescriptor, hArray))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuArray3DCreate,                        (CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray), (pHandle, pAllocateArray))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuArray3DGetDescriptor,                 (CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray), (pArrayDescriptor, hArray))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuTexRefSetAddress,                     (size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes), (ByteOffset, hTexRef, dptr, bytes))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuTexRefGetAddress,                     (CUdeviceptr* pdptr, CUtexref hTexRef), (pdptr, hTexRef))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemHostGetDevicePointer,              (CUdeviceptr* pdptr, void* p, unsigned int Flags), (pdptr, p, Flags))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuTexRefSetAddress2D,                   (CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch), (hTexRef, desc, dptr, Pitch))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuMemcpyDtoDAsync,                      (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream), (dstDevice, srcDevice, ByteCount, hStream))
FW_DLL_IMPORT_CUV2( CUresult,   CUDAAPI,    cuGraphicsResourceGetMappedPointer,     (CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource), (pDevPtr, pSize, resource))
#endif

#if (CUDA_VERSION >= 3020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetCacheConfig,                    (CUfunc_cache* pconfig), (pconfig))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxSetCacheConfig,                    (CUfunc_cache config), (config))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetApiVersion,                     (CUcontext ctx, unsigned int* version), (ctx, version))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD8Async,                        (CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream), (dstDevice, uc, N, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD16Async,                       (CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream), (dstDevice, us, N, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD32Async,                       (CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream), (dstDevice, ui, N, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D8Async,                      (CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream), (dstDevice, dstPitch, uc, Width, Height, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D16Async,                     (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream), (dstDevice, dstPitch, us, Width, Height, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemsetD2D32Async,                     (CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream), (dstDevice, dstPitch, ui, Width, Height, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuStreamWaitEvent,                      (CUstream hStream, CUevent hEvent, unsigned int Flags), (hStream, hEvent, Flags))
#endif

// CUDA 4.0

#if (CUDA_VERSION >= 4000)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxSetCurrent,                        (CUcontext ctx), (ctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetCurrent,                        (CUcontext* pctx), (pctx))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemHostRegister,                      (void* p, size_t bytesize, unsigned int Flags), (p, bytesize, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemHostUnregister,                    (void* p), (p))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy,                               (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount), (dst, src, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyPeer,                           (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount), (dstDevice, dstContext, srcDevice, srcContext, ByteCount))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy3DPeer,                         (const CUDA_MEMCPY3D_PEER* pCopy), (pCopy))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyAsync,                          (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream), (dst, src, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpyPeerAsync,                      (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream), (dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuMemcpy3DPeerAsync,                    (const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream), (pCopy, hStream))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuPointerGetAttribute,                  (void* data, CUpointer_attribute attribute, CUdeviceptr ptr), (data, attribute, ptr))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuLaunchKernel,                         (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra), (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceCanAccessPeer,                  (int* canAccessPeer, CUdevice dev, CUdevice peerDev), (canAccessPeer, dev, peerDev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxEnablePeerAccess,                  (CUcontext peerContext, unsigned int Flags), (peerContext, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxDisablePeerAccess,                 (CUcontext peerContext), (peerContext))
#endif

// CUDA 4.1

#if (CUDA_VERSION >= 4010)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetByPCIBusId,                  (CUdevice* dev, char* pciBusId), (dev, pciBusId))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuDeviceGetPCIBusId,                    (char* pciBusId, int len, CUdevice dev), (pciBusId, len, dev))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuIpcGetEventHandle,                    (CUipcEventHandle* pHandle, CUevent event), (pHandle, event))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuIpcOpenEventHandle,                   (CUevent* phEvent, CUipcEventHandle handle), (phEvent, handle))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuIpcGetMemHandle,                      (CUipcMemHandle* pHandle, CUdeviceptr dptr), (pHandle, dptr))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuIpcOpenMemHandle,                     (CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags), (pdptr, handle, Flags))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuIpcCloseMemHandle,                    (CUdeviceptr dptr), (dptr))
#endif

// CUDA 4.2

#if (CUDA_VERSION >= 4020)
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxGetSharedMemConfig,                (CUsharedconfig *pConfig), (pConfig))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuCtxSetSharedMemConfig,                (CUsharedconfig config), (config))
FW_DLL_IMPORT_CUDA( CUresult,   CUDAAPI,    cuFuncSetSharedMemConfig,               (CUfunction hfunc, CUsharedconfig config), (hfunc, config))
#endif

#if defined(_WIN32)
//------------------------------------------------------------------------
// WinBase
//------------------------------------------------------------------------

FW_DLL_IMPORT_VOID(void,        WINAPI,     InitializeConditionVariable,            (PCONDITION_VARIABLE ConditionVariable), (ConditionVariable))
FW_DLL_IMPORT_RETV(BOOL,        WINAPI,     SleepConditionVariableCS,               (PCONDITION_VARIABLE ConditionVariable, PCRITICAL_SECTION CriticalSection, DWORD dwMilliseconds), (ConditionVariable, CriticalSection, dwMilliseconds))
FW_DLL_IMPORT_VOID(void,        WINAPI,     WakeAllConditionVariable,               (PCONDITION_VARIABLE ConditionVariable), (ConditionVariable))
FW_DLL_IMPORT_VOID(void,        WINAPI,     WakeConditionVariable,                  (PCONDITION_VARIABLE ConditionVariable), (ConditionVariable))

//------------------------------------------------------------------------
// WinMM
//------------------------------------------------------------------------

FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutOpen,                            (OUT LPHWAVEOUT phwo, IN UINT uDeviceID, IN LPCWAVEFORMATEX pwfx, IN DWORD_PTR dwCallback, IN DWORD_PTR dwInstance, IN DWORD fdwOpen), (phwo, uDeviceID, pwfx, dwCallback, dwInstance, fdwOpen))
FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutClose,                           (IN OUT HWAVEOUT hwo), (hwo))
FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutPrepareHeader,                   (IN HWAVEOUT hwo, IN OUT LPWAVEHDR pwh, IN UINT cbwh), (hwo, pwh, cbwh))
FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutUnprepareHeader,                 (IN HWAVEOUT hwo, IN OUT LPWAVEHDR pwh, IN UINT cbwh), (hwo, pwh, cbwh))
FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutWrite,                           (IN HWAVEOUT hwo, IN OUT LPWAVEHDR pwh, IN UINT cbwh), (hwo, pwh, cbwh))
FW_DLL_IMPORT_RETV( MMRESULT,   WINAPI,     waveOutReset,                           (IN HWAVEOUT hwo), (hwo))

//------------------------------------------------------------------------
// ShLwAPI
//------------------------------------------------------------------------

FW_DLL_IMPORT_RETV( BOOL,   STDAPICALLTYPE, PathRelativePathToA,                    (LPSTR pszPath, LPCSTR pszFrom, DWORD dwAttrFrom, LPCSTR pszTo, DWORD dwAttrTo), (pszPath, pszFrom, dwAttrFrom, pszTo, dwAttrTo))

#endif // defined(_WIN32)

//------------------------------------------------------------------------
