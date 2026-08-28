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

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#endif

#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Shared helpers for the Vulkan backend translation units (vkutil.cpp and
// ludus_timestamped_vk.cpp). Defined here so they aren't copy-pasted per file.
// ---------------------------------------------------------------------------

// Raise via torch's TORCH_CHECK (from framework.h, included by each .cpp before
// this header) when a Vulkan call doesn't return VK_SUCCESS.
#define VK_CHECK(call) do {                                                     \
    VkResult _r = (call);                                                       \
    TORCH_CHECK(_r == VK_SUCCESS, #call " failed with VkResult ", (int)_r);     \
} while(0)

// Verbose "[Vulkan] ..." diagnostics, gated on LUDUS_VK_DEBUG=1 so device and
// per-frame traces stay out of production output.
inline bool ludus_vk_debug() {
    static int cached = -1;
    if (cached == -1) {
        const char* e = getenv("LUDUS_VK_DEBUG");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached != 0;
}

#define VK_DBG(...) do { if (ludus_vk_debug()) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ---------------------------------------------------------------------------
// VkContext: instance + physical device + logical device + queue + command
// pool. Targets Vulkan 1.3 with VK_EXT_mesh_shader for hardware mesh shading.
// ---------------------------------------------------------------------------

struct VkContext
{
    static constexpr uint32_t kFramesInFlight = 3;

    VkInstance                  instance;
    VkPhysicalDevice            physicalDevice;
    VkDevice                    device;
    VkQueue                     graphicsQueue;
    uint32_t                    graphicsQueueFamily;
    VkCommandPool               commandPool;
    VkCommandBuffer             commandBuffers[kFramesInFlight];
    VkFence                     fences[kFramesInFlight];
    uint32_t                    frameCursor;
    VkPhysicalDeviceMemoryProperties memProperties;
    int                         cudaDeviceIdx;

    // Device capabilities (checked at init)
    bool                        hasMeshShader;             // VK_EXT_mesh_shader
    bool                        hasFragmentShaderBarycentric;
    bool                        hasExternalMemory;
    bool                        hasExternalSemaphore;

    // Shared timeline: CUDA-ready -> Vulkan-done -> CUDA-release.
    VkSemaphore                 interopTimeline;
    CUexternalSemaphore         cuInteropTimeline;
    uint64_t                    nextTimelineValue;

    // Cached EXT entry point (resolved at device creation)
    PFN_vkCmdDrawMeshTasksEXT           pfnCmdDrawMeshTasksEXT;
};

// Vulkan buffer with optional CUDA-importable external memory backing.
// When cudaImportable is true the underlying allocation is exported through
// the platform-native Vulkan handle type (Win32 NT handle or Linux fd) and
// imported into CUDA so that PyTorch tensors can write directly into it. The
// transient OS handle is released/transferred immediately after import and is
// deliberately not retained in this long-lived resource.
struct VkExternalBuffer
{
    VkBuffer                    buffer;
    VkDeviceMemory              memory;
    VkDeviceSize                size;
    CUexternalMemory            cuExtMem;
    CUdeviceptr                 cuDevPtr;
};

// Device-local Vulkan image used by the native raster pipeline. This is not an
// interop allocation: optimal image tiling is intentionally opaque to CUDA and
// cannot be exposed as a row-major tensor. Layered output interop uses a
// separate exported linear buffer populated after fixed-function visibility
// has been resolved.
struct VkDeviceImage
{
    VkImage                     image;
    VkDeviceMemory              memory;
    VkImageView                 imageView;
    VkDeviceSize                size;
    uint32_t                    width;
    uint32_t                    height;
    uint32_t                    layers;
    VkFormat                    format;
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

// Device-local image management for native color/depth attachments.
VkDeviceImage createDeviceImage(
    VkContext& ctx,
    uint32_t width, uint32_t height, uint32_t layers,
    VkFormat format,
    VkImageUsageFlags usage,
    VkSampleCountFlagBits samples
);
void destroyDeviceImage(VkContext& ctx, VkDeviceImage& img);

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

// GPU-only CUDA/Vulkan timeline handoff helpers. Neither waits on the host.
uint64_t signalInteropTimelineFromCuda(VkContext& ctx, cudaStream_t stream);
void waitInteropTimelineOnCuda(
    VkContext& ctx,
    cudaStream_t stream,
    uint64_t value
);
