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
#include <unistd.h>

// VK_CHECK / VK_DBG / ludus_vk_debug() live in vkutil.h (shared with the
// renderer translation unit).

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
// PCI bus-id based pairing with a CUDA device.
// ---------------------------------------------------------------------------

static bool matchCudaDevice(VkInstance /*instance*/, VkPhysicalDevice physDev, int cudaDeviceIdx)
{
    cudaDeviceProp cudaProps;
    if (cudaGetDeviceProperties(&cudaProps, cudaDeviceIdx) != cudaSuccess)
        return false;

    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pciInfo = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    props2.pNext = &pciInfo;
    vkGetPhysicalDeviceProperties2(physDev, &props2);

    return pciInfo.pciDomain == (uint32_t)cudaProps.pciDomainID &&
           pciInfo.pciBus    == (uint32_t)cudaProps.pciBusID &&
           pciInfo.pciDevice == (uint32_t)cudaProps.pciDeviceID;
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
        ctx.physicalDevice = physDevs[0];
        if (cudaDeviceIdx >= 0)
            fprintf(stderr, "[Vulkan] Could not match CUDA device %d, falling back to first Vulkan device\n", cudaDeviceIdx);
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
    ctx.hasExternalMemory = hasExt(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);

    TORCH_CHECK(ctx.hasMeshShader,
        "VK_EXT_mesh_shader is required but not supported by Vulkan device '",
        devProps.deviceName, "'");
    TORCH_CHECK(ctx.hasExternalMemory,
        "VK_KHR_external_memory_fd is required for CUDA interop");

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
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
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

    VkPhysicalDeviceMultiviewFeatures mvFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
    mvFeatures.multiview = VK_TRUE;
    maint4.pNext = &mvFeatures;

    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    barycentricFeatures.fragmentShaderBarycentric = VK_TRUE;
    mvFeatures.pNext = ctx.hasFragmentShaderBarycentric ? (void*)&barycentricFeatures : nullptr;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.features.multiDrawIndirect = VK_TRUE;
    features2.features.fillModeNonSolid = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
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
    allocCI.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &allocCI, &ctx.commandBuffer));

    VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(ctx.device, &fenceCI, nullptr, &ctx.fence));

    VK_DBG("[Vulkan] Context ready (mesh_shader=EXT, barycentric=%s)\n",
        ctx.hasFragmentShaderBarycentric ? "KHR" : "off");

    return ctx;
}

void destroyVkContext(VkContext& ctx)
{
    if (ctx.device) {
        vkDeviceWaitIdle(ctx.device);
        if (ctx.fence) vkDestroyFence(ctx.device, ctx.fence, nullptr);
        if (ctx.commandPool) vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        vkDestroyDevice(ctx.device, nullptr);
    }
    if (ctx.instance) vkDestroyInstance(ctx.instance, nullptr);
    memset(&ctx, 0, sizeof(VkContext));
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
    buf.memFd = -1;

    if (size == 0) return buf;

    VkExternalMemoryBufferCreateInfo extBufCI = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBufCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = size;
    bufCI.usage = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (cudaImportable) bufCI.pNext = &extBufCI;

    VK_CHECK(vkCreateBuffer(ctx.device, &bufCI, nullptr, &buf.buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &memReqs);

    VkExportMemoryAllocateInfo exportAI = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.memProperties, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (cudaImportable) allocInfo.pNext = &exportAI;

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &buf.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0));

    if (cudaImportable) {
        VkMemoryGetFdInfoKHR getFdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        getFdInfo.memory = buf.memory;
        getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
        TORCH_CHECK(vkGetMemoryFdKHR, "vkGetMemoryFdKHR not available");
        VK_CHECK(vkGetMemoryFdKHR(ctx.device, &getFdInfo, &buf.memFd));

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memDesc.handle.fd = buf.memFd;
        memDesc.size = memReqs.size;
        CUresult cr = cuImportExternalMemory(&buf.cuExtMem, &memDesc);
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
    if (buf.cuExtMem) { cuDestroyExternalMemory(buf.cuExtMem); buf.cuExtMem = 0; }
    if (buf.buffer) { vkDestroyBuffer(ctx.device, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; }
    if (buf.memory) { vkFreeMemory(ctx.device, buf.memory, nullptr); buf.memory = VK_NULL_HANDLE; }
    buf.cuDevPtr = 0;
    buf.size = 0;
    buf.memFd = -1;
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
// External image (color/depth with CUDA import for color only)
// ---------------------------------------------------------------------------

VkExternalImage createExternalImage(
    VkContext& ctx,
    uint32_t width, uint32_t height, uint32_t layers,
    VkFormat format,
    VkImageUsageFlags usage,
    VkSampleCountFlagBits samples,
    bool cudaImportable)
{
    VkExternalImage img = {};
    img.width = width;
    img.height = height;
    img.layers = layers;
    img.format = format;
    img.memFd = -1;

    VkExternalMemoryImageCreateInfo extImgCI = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extImgCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

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
    if (cudaImportable) imgCI.pNext = &extImgCI;

    VK_CHECK(vkCreateImage(ctx.device, &imgCI, nullptr, &img.image));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(ctx.device, img.image, &memReqs);
    img.size = memReqs.size;

    VkExportMemoryAllocateInfo exportAI = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.memProperties, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (cudaImportable) allocInfo.pNext = &exportAI;

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

    if (cudaImportable && aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
        auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR getFdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        getFdInfo.memory = img.memory;
        getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VK_CHECK(vkGetMemoryFdKHR(ctx.device, &getFdInfo, &img.memFd));

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memDesc.handle.fd = img.memFd;
        memDesc.size = memReqs.size;
        CUresult cr = cuImportExternalMemory(&img.cuExtMem, &memDesc);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuImportExternalMemory (image) failed: ", (int)cr);

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipDesc = {};
        mipDesc.offset = 0;
        mipDesc.arrayDesc.Width = width;
        mipDesc.arrayDesc.Height = height;
        mipDesc.arrayDesc.Depth = layers;
        // Layered images (arrayLayers > 1) need CUDA_ARRAY3D_LAYERED, else CUDA
        // reads Depth as a 3D volume and the import fails with INVALID_VALUE.
        if (layers > 1)
            mipDesc.arrayDesc.Flags = CUDA_ARRAY3D_LAYERED;
        if (format == VK_FORMAT_R8G8B8A8_UNORM) {
            mipDesc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            mipDesc.arrayDesc.NumChannels = 4;
        }
        mipDesc.numLevels = 1;
        cr = cuExternalMemoryGetMappedMipmappedArray(&img.cuMipArray, img.cuExtMem, &mipDesc);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuExternalMemoryGetMappedMipmappedArray failed: ", (int)cr);

        cr = cuMipmappedArrayGetLevel(&img.cuArray, img.cuMipArray, 0);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuMipmappedArrayGetLevel failed: ", (int)cr);
    }

    return img;
}

void destroyExternalImage(VkContext& ctx, VkExternalImage& img)
{
    if (img.cuMipArray) { cuMipmappedArrayDestroy(img.cuMipArray); img.cuMipArray = 0; }
    if (img.cuExtMem) { cuDestroyExternalMemory(img.cuExtMem); img.cuExtMem = 0; }
    if (img.imageView) { vkDestroyImageView(ctx.device, img.imageView, nullptr); img.imageView = VK_NULL_HANDLE; }
    if (img.image) { vkDestroyImage(ctx.device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(ctx.device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
    img.cuArray = 0;
    img.memFd = -1;
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

// CUDA<->Vulkan handoff is fence-only (cudaStreamSynchronize -> vkQueueSubmit
// -> vkWaitForFences); no external-semaphore path is used.
