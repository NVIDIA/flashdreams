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

#pragma once

#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// VkContext: instance + physical device + logical device + queue + command
// pool. Targets Vulkan 1.3 with VK_EXT_mesh_shader for hardware mesh shading.
// ---------------------------------------------------------------------------

struct VkContext
{
    VkInstance                  instance;
    VkPhysicalDevice            physicalDevice;
    VkDevice                    device;
    VkQueue                     graphicsQueue;
    uint32_t                    graphicsQueueFamily;
    VkCommandPool               commandPool;
    VkCommandBuffer             commandBuffer;
    VkFence                     fence;
    VkPhysicalDeviceMemoryProperties memProperties;
    int                         cudaDeviceIdx;
    bool                        validationEnabled;

    // Device capabilities (checked at init)
    bool                        hasMeshShader;             // VK_EXT_mesh_shader
    bool                        hasFragmentShaderBarycentric;
    bool                        hasShaderInt64;
    bool                        hasExternalMemory;

    // Cached EXT entry points (resolved at device creation)
    PFN_vkCmdDrawMeshTasksEXT           pfnCmdDrawMeshTasksEXT;
    PFN_vkCmdDrawMeshTasksIndirectEXT   pfnCmdDrawMeshTasksIndirectEXT;
};

// Vulkan buffer with optional CUDA-importable external memory backing.
// When cudaImportable is true the underlying allocation is exported as
// VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT and imported into CUDA so
// that PyTorch tensors can write directly into the buffer.
struct VkExternalBuffer
{
    VkBuffer                    buffer;
    VkDeviceMemory              memory;
    VkDeviceSize                size;
    CUexternalMemory            cuExtMem;
    CUdeviceptr                 cuDevPtr;
    int                         memFd;       // POSIX fd for external memory (-1 if not external)
};

// Vulkan image with optional CUDA-importable external memory backing.
struct VkExternalImage
{
    VkImage                     image;
    VkDeviceMemory              memory;
    VkImageView                 imageView;
    VkDeviceSize                size;
    uint32_t                    width;
    uint32_t                    height;
    uint32_t                    layers;
    VkFormat                    format;
    CUexternalMemory            cuExtMem;
    CUmipmappedArray            cuMipArray;
    CUarray                     cuArray;     // First level of mipmap
    int                         memFd;
};

// Context lifecycle.
VkContext   createVkContext(int cudaDeviceIdx);
void        destroyVkContext(VkContext& ctx);

// Buffer management with CUDA external memory.
VkExternalBuffer createExternalBuffer(
    VkContext& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    bool cudaImportable
);
void destroyExternalBuffer(VkContext& ctx, VkExternalBuffer& buf);
void resizeExternalBuffer(
    VkContext& ctx,
    VkExternalBuffer& buf,
    VkDeviceSize newSize,
    VkBufferUsageFlags usage,
    bool cudaImportable
);

// Image management with CUDA external memory.
VkExternalImage createExternalImage(
    VkContext& ctx,
    uint32_t width, uint32_t height, uint32_t layers,
    VkFormat format,
    VkImageUsageFlags usage,
    VkSampleCountFlagBits samples,
    bool cudaImportable
);
void destroyExternalImage(VkContext& ctx, VkExternalImage& img);

// Memory type helpers.
uint32_t findMemoryType(
    const VkPhysicalDeviceMemoryProperties& memProps,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties
);

// Single-use command buffer helpers.
VkCommandBuffer beginSingleTimeCommands(VkContext& ctx);
void endSingleTimeCommands(VkContext& ctx, VkCommandBuffer cmd);

// Image layout transitions.
void transitionImageLayout(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t layerCount,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT
);

// CUDA-Vulkan synchronization (binary external semaphore, opaque-fd).
// Currently unused in the simple fence-based path but plumbed for future use.
struct VkCudaSync
{
    VkSemaphore                 vkSemaphore;
    CUexternalSemaphore         cuSemaphore;
    uint64_t                    timelineValue;
};

VkCudaSync createCudaSync(VkContext& ctx);
void destroyCudaSync(VkContext& ctx, VkCudaSync& sync);
void signalVulkanWaitCuda(VkContext& ctx, VkCudaSync& sync, cudaStream_t stream);
void signalCudaWaitVulkan(VkContext& ctx, VkCudaSync& sync, cudaStream_t stream);
