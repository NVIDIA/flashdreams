// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "framework.h"
#include "vkutil.h"
#include <cstring>
#include <algorithm>
#if !defined(_WIN32)
#include <unistd.h>
#endif

// VK_CHECK / VK_DBG / ludus_vk_debug() live in vkutil.h (shared with the
// renderer translation unit).

#if defined(_WIN32)
static constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
static constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
static constexpr const char* kExternalMemoryExtension =
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
static constexpr const char* kExternalSemaphoreExtension =
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
static constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
static constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
static constexpr const char* kExternalMemoryExtension =
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
static constexpr const char* kExternalSemaphoreExtension =
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif

// ---------------------------------------------------------------------------
// Debug messenger (optional, only attached when validation layer is loaded).
// ---------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
        fflush(stderr);
    }
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// UUID pairing with a CUDA device. Zero-copy interop must never silently bind
// Vulkan and CUDA to different physical GPUs.
// ---------------------------------------------------------------------------

static bool matchCudaDevice(VkInstance /*instance*/, VkPhysicalDevice physDev, int cudaDeviceIdx)
{
    cudaDeviceProp cudaProps;
    if (cudaGetDeviceProperties(&cudaProps, cudaDeviceIdx) != cudaSuccess)
        return false;

    VkPhysicalDeviceProperties2 props2 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceIDProperties id = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    props2.pNext = &id;
    vkGetPhysicalDeviceProperties2(physDev, &props2);

    return memcmp(id.deviceUUID, cudaProps.uuid.bytes, VK_UUID_SIZE) == 0;
}

// ---------------------------------------------------------------------------
// Context creation
// ---------------------------------------------------------------------------

VkContext createVkContext(int cudaDeviceIdx)
{
    VkContext ctx = {};
    ctx.cudaDeviceIdx = cudaDeviceIdx;

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "ludus-renderer";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
    };
    std::vector<const char*> layers;

#ifndef NDEBUG
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availLayers.data());
    for (auto& l : availLayers) {
        if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            break;
        }
    }
#endif

    VkInstanceCreateInfo instCI = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instCI.pApplicationInfo = &appInfo;
    instCI.enabledExtensionCount = (uint32_t)instExts.size();
    instCI.ppEnabledExtensionNames = instExts.data();
    instCI.enabledLayerCount = (uint32_t)layers.size();
    instCI.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&instCI, nullptr, &ctx.instance));

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    TORCH_CHECK(devCount > 0, "No Vulkan physical devices found");
    std::vector<VkPhysicalDevice> physDevs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, physDevs.data());

    ctx.physicalDevice = VK_NULL_HANDLE;
    if (cudaDeviceIdx >= 0) {
        for (auto& pd : physDevs) {
            if (matchCudaDevice(ctx.instance, pd, cudaDeviceIdx)) {
                ctx.physicalDevice = pd;
                break;
            }
        }
    }
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        TORCH_CHECK(cudaDeviceIdx < 0,
            "No Vulkan physical device matches CUDA device ", cudaDeviceIdx,
            "; zero-copy CUDA/Vulkan external memory requires one physical GPU");
        ctx.physicalDevice = physDevs[0];
    }

    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &devProps);
    VK_DBG("[Vulkan] Device: %s (apiVersion %u.%u.%u)\n",
        devProps.deviceName,
        VK_API_VERSION_MAJOR(devProps.apiVersion),
        VK_API_VERSION_MINOR(devProps.apiVersion),
        VK_API_VERSION_PATCH(devProps.apiVersion));

    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &ctx.memProperties);

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(ctx.physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateDeviceExtensionProperties(ctx.physicalDevice, nullptr, &extCount, availExts.data());

    auto hasExt = [&](const char* name) {
        for (auto& e : availExts)
            if (strcmp(e.extensionName, name) == 0) return true;
        return false;
    };

    ctx.hasMeshShader = hasExt(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    ctx.hasFragmentShaderBarycentric = hasExt(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    ctx.hasExternalMemory = hasExt(kExternalMemoryExtension);
    ctx.hasExternalSemaphore = hasExt(kExternalSemaphoreExtension);

    TORCH_CHECK(ctx.hasMeshShader,
        "VK_EXT_mesh_shader is required but not supported by Vulkan device '",
        devProps.deviceName, "'");
    TORCH_CHECK(ctx.hasFragmentShaderBarycentric,
        "VK_KHR_fragment_shader_barycentric is required but not supported by "
        "Vulkan device '", devProps.deviceName, "'");
    TORCH_CHECK(ctx.hasExternalMemory,
        kExternalMemoryExtension, " is required for CUDA interop");
    TORCH_CHECK(ctx.hasExternalSemaphore,
        kExternalSemaphoreExtension,
        " is required for asynchronous CUDA interop");

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qfCount, qfProps.data());

    ctx.graphicsQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            ctx.graphicsQueueFamily = i;
            break;
        }
    }
    TORCH_CHECK(ctx.graphicsQueueFamily != UINT32_MAX, "No graphics queue family found");

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCI.queueFamilyIndex = ctx.graphicsQueueFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &queuePriority;

    std::vector<const char*> devExts = {
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        kExternalMemoryExtension,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        kExternalSemaphoreExtension,
    };
    if (ctx.hasFragmentShaderBarycentric)
        devExts.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);

    // Feature chain: VK_EXT_mesh_shader requires its own features struct,
    // plus maintenance4 to allow the task/mesh shader stages to express
    // workgroup sizes via local_size_x_id constants.
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    meshFeatures.taskShader = VK_TRUE;
    meshFeatures.meshShader = VK_TRUE;

    VkPhysicalDeviceMaintenance4Features maint4 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES};
    maint4.maintenance4 = VK_TRUE;
    meshFeatures.pNext = &maint4;

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timelineFeatures.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    barycentricFeatures.fragmentShaderBarycentric = VK_TRUE;
    maint4.pNext = &timelineFeatures;
    timelineFeatures.pNext = ctx.hasFragmentShaderBarycentric ? (void*)&barycentricFeatures : nullptr;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceFeatures supportedFeatures = {};
    vkGetPhysicalDeviceFeatures(ctx.physicalDevice, &supportedFeatures);
    TORCH_CHECK(supportedFeatures.geometryShader,
        "geometryShader is required for layered mesh-shader rasterization");
    TORCH_CHECK(supportedFeatures.shaderInt64,
        "shaderInt64 is required by timestamped Ludus shaders");
    TORCH_CHECK(supportedFeatures.fillModeNonSolid,
        "fillModeNonSolid is required by Ludus pipeline configuration");
    features2.features.multiDrawIndirect = VK_TRUE;
    features2.features.fillModeNonSolid = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
    // gl_Layer emitted by the mesh stage is represented by SPIR-V's Geometry
    // capability even though no classic geometry shader stage is present.
    features2.features.geometryShader = VK_TRUE;
    features2.pNext = &meshFeatures;

    VkDeviceCreateInfo devCI = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &queueCI;
    devCI.enabledExtensionCount = (uint32_t)devExts.size();
    devCI.ppEnabledExtensionNames = devExts.data();
    devCI.pNext = &features2;

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &devCI, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.graphicsQueueFamily, 0, &ctx.graphicsQueue);

    ctx.pfnCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)
        vkGetDeviceProcAddr(ctx.device, "vkCmdDrawMeshTasksEXT");
    TORCH_CHECK(ctx.pfnCmdDrawMeshTasksEXT != nullptr,
        "vkCmdDrawMeshTasksEXT not available even though VK_EXT_mesh_shader is reported");

    VkCommandPoolCreateInfo poolCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCI.queueFamilyIndex = ctx.graphicsQueueFamily;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolCI, nullptr, &ctx.commandPool));

    VkCommandBufferAllocateInfo allocCI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCI.commandPool = ctx.commandPool;
    allocCI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandBufferCount = VkContext::kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &allocCI, ctx.commandBuffers));

    VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < VkContext::kFramesInFlight; ++i)
        VK_CHECK(vkCreateFence(ctx.device, &fenceCI, nullptr, &ctx.fences[i]));

    VkExportSemaphoreCreateInfo exportSemaphore = {
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportSemaphore.handleTypes = kExternalSemaphoreHandleType;
    VkSemaphoreTypeCreateInfo timelineCI = {
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCI.initialValue = 0;
    timelineCI.pNext = &exportSemaphore;
    VkSemaphoreCreateInfo semaphoreCI = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreCI.pNext = &timelineCI;
    VK_CHECK(vkCreateSemaphore(ctx.device, &semaphoreCI, nullptr, &ctx.interopTimeline));

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphoreDesc = {};
#if defined(_WIN32)
    auto vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)
        vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreWin32HandleKHR");
    TORCH_CHECK(vkGetSemaphoreWin32HandleKHR,
        "vkGetSemaphoreWin32HandleKHR not available");
    VkSemaphoreGetWin32HandleInfoKHR semaphoreHandleInfo = {
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    semaphoreHandleInfo.semaphore = ctx.interopTimeline;
    semaphoreHandleInfo.handleType = kExternalSemaphoreHandleType;
    HANDLE semaphoreHandle = nullptr;
    VK_CHECK(vkGetSemaphoreWin32HandleKHR(
        ctx.device, &semaphoreHandleInfo, &semaphoreHandle));
    semaphoreDesc.type =
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32;
    semaphoreDesc.handle.win32.handle = semaphoreHandle;
    CUresult cr = cuImportExternalSemaphore(
        &ctx.cuInteropTimeline, &semaphoreDesc);
    // CUDA retains the imported object but does not take ownership of an NT
    // handle. Close our transport handle on both success and failure.
    CloseHandle(semaphoreHandle);
#else
    auto vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)
        vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR");
    TORCH_CHECK(vkGetSemaphoreFdKHR, "vkGetSemaphoreFdKHR not available");
    VkSemaphoreGetFdInfoKHR semaphoreHandleInfo = {
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    semaphoreHandleInfo.semaphore = ctx.interopTimeline;
    semaphoreHandleInfo.handleType = kExternalSemaphoreHandleType;
    int semaphoreFd = -1;
    VK_CHECK(vkGetSemaphoreFdKHR(
        ctx.device, &semaphoreHandleInfo, &semaphoreFd));
    semaphoreDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    semaphoreDesc.handle.fd = semaphoreFd;
    CUresult cr = cuImportExternalSemaphore(
        &ctx.cuInteropTimeline, &semaphoreDesc);
    // A successful CUDA import consumes an opaque fd. Retain no descriptor;
    // close it ourselves only when ownership was not transferred.
    if (cr != CUDA_SUCCESS) close(semaphoreFd);
#endif
    TORCH_CHECK(cr == CUDA_SUCCESS,
        "cuImportExternalSemaphore (timeline) failed: ", (int)cr);
    ctx.frameCursor = 0;
    ctx.nextTimelineValue = 0;

    VK_DBG("[Vulkan] Context ready (mesh_shader=EXT, barycentric=%s)\n",
        ctx.hasFragmentShaderBarycentric ? "KHR" : "off");

    return ctx;
}

void destroyVkContext(VkContext& ctx)
{
    if (ctx.device) {
        vkDeviceWaitIdle(ctx.device);
        if (ctx.cuInteropTimeline) {
            cuDestroyExternalSemaphore(ctx.cuInteropTimeline);
            ctx.cuInteropTimeline = nullptr;
        }
        if (ctx.interopTimeline)
            vkDestroySemaphore(ctx.device, ctx.interopTimeline, nullptr);
        for (uint32_t i = 0; i < VkContext::kFramesInFlight; ++i)
            if (ctx.fences[i]) vkDestroyFence(ctx.device, ctx.fences[i], nullptr);
        if (ctx.commandPool) vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        vkDestroyDevice(ctx.device, nullptr);
    }
    if (ctx.instance) vkDestroyInstance(ctx.instance, nullptr);
    memset(&ctx, 0, sizeof(VkContext));
}

uint64_t signalInteropTimelineFromCuda(VkContext& ctx, cudaStream_t stream)
{
    const uint64_t value = ++ctx.nextTimelineValue;
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params = {};
    params.params.fence.value = value;
    CUexternalSemaphore semaphore = ctx.cuInteropTimeline;
    CUresult cr = cuSignalExternalSemaphoresAsync(
        &semaphore, &params, 1, reinterpret_cast<CUstream>(stream));
    TORCH_CHECK(cr == CUDA_SUCCESS,
        "cuSignalExternalSemaphoresAsync failed: ", (int)cr);
    return value;
}

void waitInteropTimelineOnCuda(
    VkContext& ctx, cudaStream_t stream, uint64_t value)
{
    if (value == 0) return;
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params = {};
    params.params.fence.value = value;
    CUexternalSemaphore semaphore = ctx.cuInteropTimeline;
    CUresult cr = cuWaitExternalSemaphoresAsync(
        &semaphore, &params, 1, reinterpret_cast<CUstream>(stream));
    TORCH_CHECK(cr == CUDA_SUCCESS,
        "cuWaitExternalSemaphoresAsync failed: ", (int)cr);
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

uint32_t findMemoryType(
    const VkPhysicalDeviceMemoryProperties& memProps,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    TORCH_CHECK(false, "Failed to find suitable Vulkan memory type");
    return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// External buffer (SSBO with CUDA import)
// ---------------------------------------------------------------------------

VkExternalBuffer createExternalBuffer(
    VkContext& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    bool cudaImportable)
{
    VkExternalBuffer buf = {};
    buf.size = size;

    if (size == 0) return buf;

    VkExternalMemoryBufferCreateInfo extBufCI = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBufCI.handleTypes = kExternalMemoryHandleType;

    bool dedicatedOnly = false;
    if (cudaImportable) {
        VkPhysicalDeviceExternalBufferInfo externalInfo = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        externalInfo.flags = 0;
        externalInfo.usage = usage;
        externalInfo.handleType = kExternalMemoryHandleType;
        VkExternalBufferProperties externalProps = {
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(
            ctx.physicalDevice, &externalInfo, &externalProps);
        const VkExternalMemoryFeatureFlags features =
            externalProps.externalMemoryProperties.externalMemoryFeatures;
        TORCH_CHECK(
            (features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0,
            "Vulkan buffer usage is not external-memory exportable");
        dedicatedOnly =
            (features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    }

    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = size;
    bufCI.usage = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (cudaImportable) bufCI.pNext = &extBufCI;

    VK_CHECK(vkCreateBuffer(ctx.device, &bufCI, nullptr, &buf.buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &memReqs);

    VkExportMemoryAllocateInfo exportAI = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAI.handleTypes = kExternalMemoryHandleType;
    VkMemoryDedicatedAllocateInfo dedicatedAI = {
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAI.buffer = buf.buffer;
    if (dedicatedOnly)
        exportAI.pNext = &dedicatedAI;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.memProperties, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (cudaImportable) allocInfo.pNext = &exportAI;

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &buf.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0));

    if (cudaImportable) {
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
#if defined(_WIN32)
        VkMemoryGetWin32HandleInfoKHR getHandleInfo = {
            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        getHandleInfo.memory = buf.memory;
        getHandleInfo.handleType = kExternalMemoryHandleType;
        auto vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)
            vkGetDeviceProcAddr(ctx.device, "vkGetMemoryWin32HandleKHR");
        TORCH_CHECK(vkGetMemoryWin32HandleKHR,
            "vkGetMemoryWin32HandleKHR not available");
        HANDLE memoryHandle = nullptr;
        VK_CHECK(vkGetMemoryWin32HandleKHR(
            ctx.device, &getHandleInfo, &memoryHandle));
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        memDesc.handle.win32.handle = memoryHandle;
#else
        VkMemoryGetFdInfoKHR getHandleInfo = {
            VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        getHandleInfo.memory = buf.memory;
        getHandleInfo.handleType = kExternalMemoryHandleType;
        auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
            vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
        TORCH_CHECK(vkGetMemoryFdKHR, "vkGetMemoryFdKHR not available");
        int memoryFd = -1;
        VK_CHECK(vkGetMemoryFdKHR(ctx.device, &getHandleInfo, &memoryFd));
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memDesc.handle.fd = memoryFd;
#endif
        memDesc.size = memReqs.size;
        memDesc.flags = dedicatedOnly ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;
        CUresult cr = cuImportExternalMemory(&buf.cuExtMem, &memDesc);
#if defined(_WIN32)
        // Unlike an fd import, CUDA does not consume the Win32 NT handle.
        CloseHandle(memoryHandle);
#else
        if (cr != CUDA_SUCCESS) close(memoryFd);
#endif
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuImportExternalMemory failed: ", (int)cr);

        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufDesc = {};
        bufDesc.offset = 0;
        bufDesc.size = size;
        cr = cuExternalMemoryGetMappedBuffer(&buf.cuDevPtr, buf.cuExtMem, &bufDesc);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuExternalMemoryGetMappedBuffer failed: ", (int)cr);
    }

    return buf;
}

void destroyExternalBuffer(VkContext& ctx, VkExternalBuffer& buf)
{
    if (buf.cuDevPtr) { cuMemFree(buf.cuDevPtr); buf.cuDevPtr = 0; }
    if (buf.cuExtMem) { cuDestroyExternalMemory(buf.cuExtMem); buf.cuExtMem = 0; }
    if (buf.buffer) { vkDestroyBuffer(ctx.device, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; }
    if (buf.memory) { vkFreeMemory(ctx.device, buf.memory, nullptr); buf.memory = VK_NULL_HANDLE; }
    buf.size = 0;
}

void resizeExternalBuffer(
    VkContext& ctx,
    VkExternalBuffer& buf,
    VkDeviceSize newSize,
    VkBufferUsageFlags usage,
    bool cudaImportable)
{
    bool needsCuda = cudaImportable && (buf.cuDevPtr == 0);
    if (buf.size >= newSize && buf.buffer != VK_NULL_HANDLE && !needsCuda) return;
    destroyExternalBuffer(ctx, buf);
    buf = createExternalBuffer(ctx, newSize, usage, cudaImportable);
}

// ---------------------------------------------------------------------------
// Device-local native raster images. CUDA image import is intentionally absent.
// ---------------------------------------------------------------------------

VkDeviceImage createDeviceImage(
    VkContext& ctx,
    uint32_t width, uint32_t height, uint32_t layers,
    VkFormat format,
    VkImageUsageFlags usage,
    VkSampleCountFlagBits samples)
{
    VkDeviceImage img = {};
    img.width = width;
    img.height = height;
    img.layers = layers;
    img.format = format;

    VkImageCreateInfo imgCI = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCI.imageType = VK_IMAGE_TYPE_2D;
    imgCI.format = format;
    imgCI.extent = {width, height, 1};
    imgCI.mipLevels = 1;
    imgCI.arrayLayers = layers;
    imgCI.samples = samples;
    imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage = usage;
    imgCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(ctx.device, &imgCI, nullptr, &img.image));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(ctx.device, img.image, &memReqs);
    img.size = memReqs.size;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.memProperties, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, img.image, img.memory, 0));

    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    VkImageViewCreateInfo viewCI = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewCI.image = img.image;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewCI.format = format;
    viewCI.subresourceRange.aspectMask = aspect;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = layers;
    VK_CHECK(vkCreateImageView(ctx.device, &viewCI, nullptr, &img.imageView));

    return img;
}

void destroyDeviceImage(VkContext& ctx, VkDeviceImage& img)
{
    if (img.imageView) { vkDestroyImageView(ctx.device, img.imageView, nullptr); img.imageView = VK_NULL_HANDLE; }
    if (img.image) { vkDestroyImage(ctx.device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(ctx.device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// Command buffer helpers
// ---------------------------------------------------------------------------

VkCommandBuffer beginSingleTimeCommands(VkContext& ctx)
{
    VkCommandBufferAllocateInfo allocCI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCI.commandPool = ctx.commandPool;
    allocCI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &allocCI, &cmd));

    VkCommandBufferBeginInfo beginCI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginCI));
    return cmd;
}

void endSingleTimeCommands(VkContext& ctx, VkCommandBuffer cmd)
{
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence tempFence;
    VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(ctx.device, &fenceCI, nullptr, &tempFence));
    VK_CHECK(vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, tempFence));
    VK_CHECK(vkWaitForFences(ctx.device, 1, &tempFence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(ctx.device, tempFence, nullptr);
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
}

void transitionImageLayout(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t layerCount,
    VkImageAspectFlags aspect)
{
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layerCount;

    VkPipelineStageFlags srcStage, dstStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    } else {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// CUDA<->Vulkan handoff uses the exported timeline semaphore above. The
// single-use helpers below are reserved for initialization/control paths.
