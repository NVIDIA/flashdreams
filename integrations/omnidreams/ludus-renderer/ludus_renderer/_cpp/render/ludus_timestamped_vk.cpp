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

//=============================================================================
// Vulkan Timestamped Renderer (VK_EXT_mesh_shader path).
//
// Port from the GL-based timestamped renderer to native Vulkan. Geometry is
// generated procedurally in task/mesh shaders. CUDA tensors are bridged in
// through VK_KHR_external_memory_fd: every SSBO that needs to receive CUDA
// writes is allocated as external memory and imported into CUDA, allowing
// PyTorch to copy directly into the Vulkan buffer with cudaMemcpyAsync.
//=============================================================================

#include "ludus_vk.h"
#include "shaders_spv.h"   // generated header with embedded SPIR-V byte arrays
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>

// VK_CHECK / VK_DBG / ludus_vk_debug() are shared from vkutil.h (included via
// ludus_vk.h).

//=============================================================================
// SPIR-V Shader Module Creation
//=============================================================================

static VkShaderModule createShaderModule(VkDevice device, const uint32_t* code, size_t bytes)
{
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module));
    return module;
}

//=============================================================================
// Render Pass
//=============================================================================

static VkRenderPass createTimestampedRenderPass(
    VkDevice device, VkFormat colorFormat, VkFormat depthFormat,
    VkSampleCountFlagBits samples)
{
    bool msaa = (samples != VK_SAMPLE_COUNT_1_BIT);

    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> colorRefs, resolveRefs, depthRefs;

    if (msaa) {
        VkAttachmentDescription msaaColor = {};
        msaaColor.format = colorFormat;
        msaaColor.samples = samples;
        msaaColor.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        msaaColor.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        msaaColor.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        msaaColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        msaaColor.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        msaaColor.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments.push_back(msaaColor);

        VkAttachmentDescription resolveColor = {};
        resolveColor.format = colorFormat;
        resolveColor.samples = VK_SAMPLE_COUNT_1_BIT;
        resolveColor.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolveColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        resolveColor.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolveColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        resolveColor.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        resolveColor.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        attachments.push_back(resolveColor);

        VkAttachmentDescription msaaDepth = {};
        msaaDepth.format = depthFormat;
        msaaDepth.samples = samples;
        msaaDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        msaaDepth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        msaaDepth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        msaaDepth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        msaaDepth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        msaaDepth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(msaaDepth);

        colorRefs.push_back({0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        resolveRefs.push_back({1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        depthRefs.push_back({2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});
    } else {
        VkAttachmentDescription color = {};
        color.format = colorFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        attachments.push_back(color);

        VkAttachmentDescription depth = {};
        depth.format = depthFormat;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depth);

        colorRefs.push_back({0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        depthRefs.push_back({1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = (uint32_t)colorRefs.size();
    subpass.pColorAttachments = colorRefs.data();
    subpass.pResolveAttachments = msaa ? resolveRefs.data() : nullptr;
    subpass.pDepthStencilAttachment = depthRefs.data();

    VkSubpassDependency dep = {};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpCI = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpCI.attachmentCount = (uint32_t)attachments.size();
    rpCI.pAttachments = attachments.data();
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies = &dep;

    VkRenderPass renderPass;
    VK_CHECK(vkCreateRenderPass(device, &rpCI, nullptr, &renderPass));
    return renderPass;
}

//=============================================================================
// Descriptor Set Layout (14 SSBOs matching the GL/CUDA backend binding slots)
//=============================================================================

static VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device)
{
    constexpr uint32_t kNumBindings = 14;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < kNumBindings; i++) {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = i;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                     | VK_SHADER_STAGE_MESH_BIT_EXT
                     | VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(b);
    }

    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = (uint32_t)bindings.size();
    ci.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
    return layout;
}

//=============================================================================
// Pipeline Layout
//=============================================================================

static VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout dsLayout)
{
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                         | VK_SHADER_STAGE_MESH_BIT_EXT
                         | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(LudusPushConstants);

    VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &dsLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pushRange;

    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &ci, nullptr, &layout));
    return layout;
}

//=============================================================================
// Mesh Pipeline (task + mesh + fragment)
//=============================================================================

static VkPipeline createMeshPipeline(
    VkDevice device,
    VkPipelineLayout layout,
    VkRenderPass renderPass,
    VkShaderModule taskModule,
    VkShaderModule meshModule,
    VkShaderModule fragModule,
    VkSampleCountFlagBits samples)
{
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    if (taskModule != VK_NULL_HANDLE) {
        VkPipelineShaderStageCreateInfo taskStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        taskStage.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        taskStage.module = taskModule;
        taskStage.pName = "main";
        stages.push_back(taskStage);
    }

    VkPipelineShaderStageCreateInfo meshStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    meshStage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshStage.module = meshModule;
    meshStage.pName = "main";
    stages.push_back(meshStage);

    VkPipelineShaderStageCreateInfo fragStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    stages.push_back(fragStage);

    VkPipelineViewportStateCreateInfo viewportState = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterState = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode = VK_CULL_MODE_NONE;
    rasterState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaaState = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaaState.rasterizationSamples = samples;

    VkPipelineDepthStencilStateCreateInfo depthState = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthState.depthTestEnable = VK_TRUE;
    depthState.depthWriteEnable = VK_TRUE;
    depthState.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blendState = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendState.attachmentCount = 1;
    blendState.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineCI = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineCI.stageCount = (uint32_t)stages.size();
    pipelineCI.pStages = stages.data();
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pRasterizationState = &rasterState;
    pipelineCI.pMultisampleState = &msaaState;
    pipelineCI.pDepthStencilState = &depthState;
    pipelineCI.pColorBlendState = &blendState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.layout = layout;
    pipelineCI.renderPass = renderPass;
    pipelineCI.subpass = 0;

    VkPipeline pipeline;
    VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);
    TORCH_CHECK(r == VK_SUCCESS, "vkCreateGraphicsPipelines failed for mesh pipeline: ", (int)r);
    return pipeline;
}

//=============================================================================
// Descriptor Set Update
//=============================================================================

static void updateDescriptorSet(VkDevice device, VkDescriptorSet& ds, LudusTimestampedVkState& s)
{
    struct BufInfo { uint32_t binding; VkBuffer buffer; VkDeviceSize size; };
    BufInfo bufs[] = {
        { 0, s.timestampsBuffer.buffer,       s.timestampsBuffer.size},
        { 1, s.int32Buffer.buffer,            s.int32Buffer.size},
        { 2, s.vertexBuffer.buffer,           s.vertexBuffer.size},
        { 3, s.triangleBuffer.buffer,         s.triangleBuffer.size},
        { 4, s.poseBuffer.buffer,             s.poseBuffer.size},
        { 5, s.floatBuffer.buffer,            s.floatBuffer.size},
        { 6, s.sceneBuffer.buffer,            s.sceneBuffer.size},
        { 7, s.polylinePoolBuffer.buffer,     s.polylinePoolBuffer.size},
        { 8, s.polygonPoolBuffer.buffer,      s.polygonPoolBuffer.size},
        { 9, s.obstaclePoolBuffer.buffer,     s.obstaclePoolBuffer.size},
        {10, s.colorPaletteBuffer.buffer,     s.colorPaletteBuffer.size},
        {11, s.cameraIntrinsicsBuffer.buffer, s.cameraIntrinsicsBuffer.size},
        {12, s.cameraPoseBuffer.buffer,       s.cameraPoseBuffer.size},
        {13, s.queryBuffer.buffer,            s.queryBuffer.size},
    };

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufInfos(14);

    for (int i = 0; i < 14; i++) {
        if (bufs[i].buffer == VK_NULL_HANDLE || bufs[i].size == 0)
            continue;

        bufInfos[i].buffer = bufs[i].buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range = bufs[i].size;

        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = ds;
        w.dstBinding = bufs[i].binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bufInfos[i];
        writes.push_back(w);
    }

    if (!writes.empty()) {
        // Reset and re-allocate to avoid stale-handle cache when a buffer was
        // resized and the underlying VkBuffer is reused at a new GPU address.
        vkResetDescriptorPool(device, s.descriptorPool, 0);
        VkDescriptorSetAllocateInfo realloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        realloc.descriptorPool = s.descriptorPool;
        realloc.descriptorSetCount = 1;
        realloc.pSetLayouts = &s.descriptorSetLayout;
        vkAllocateDescriptorSets(device, &realloc, &ds);
        for (auto& w : writes)
            w.dstSet = ds;
        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }
}

//=============================================================================
// Initialization
//=============================================================================

void ludusTimestampedInitVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int cudaDeviceIdx)
{
    (void)nvdr_ctx;
    memset(&s, 0, sizeof(s));

    // cuImportExternalMemory and friends require a current CUDA context.
    // Force the runtime to materialize a primary context on the requested
    // device before any external-memory imports happen during init.
    cudaSetDevice(cudaDeviceIdx);
    cudaFree(nullptr);

    s.vkctx = createVkContext(cudaDeviceIdx);
    s.hasMeshShader = s.vkctx.hasMeshShader ? 1 : 0;

    s.msaaSamples = 0;
    s.tessellationThreshold = 1.0f;
    s.maxTessellationLevelPolyline = 4;
    s.maxTessellationLevelPolygon = 3;
    s.maxTessellationLevelCube = 3;
    s.depthScaling = 1.0f;
    s.maxExtrapolationUs = 500000;
    s.cullRadiusScale = 1.5f;

    s.renderPass = createTimestampedRenderPass(
        s.vkctx.device, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT,
        VK_SAMPLE_COUNT_1_BIT
    );

    s.descriptorSetLayout = createDescriptorSetLayout(s.vkctx.device);
    s.pipelineLayout = createPipelineLayout(s.vkctx.device, s.descriptorSetLayout);

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 14;

    VkDescriptorPoolCreateInfo poolCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(s.vkctx.device, &poolCI, nullptr, &s.descriptorPool));

    VkDescriptorSetAllocateInfo dsAllocCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAllocCI.descriptorPool = s.descriptorPool;
    dsAllocCI.descriptorSetCount = 1;
    dsAllocCI.pSetLayouts = &s.descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(s.vkctx.device, &dsAllocCI, &s.descriptorSet));

    // Embedded SPIR-V (generated from shaders/*.spv at build time).
    struct ShaderBin { const uint32_t* code; size_t bytes; };
    const ShaderBin bins[9] = {
        {kSpv_ts_polyline_task, sizeof(kSpv_ts_polyline_task)},
        {kSpv_ts_polyline_mesh, sizeof(kSpv_ts_polyline_mesh)},
        {kSpv_ts_polyline_frag, sizeof(kSpv_ts_polyline_frag)},
        {kSpv_ts_polygon_task,  sizeof(kSpv_ts_polygon_task)},
        {kSpv_ts_polygon_mesh,  sizeof(kSpv_ts_polygon_mesh)},
        {kSpv_ts_polygon_frag,  sizeof(kSpv_ts_polygon_frag)},
        {kSpv_ts_obstacle_task, sizeof(kSpv_ts_obstacle_task)},
        {kSpv_ts_obstacle_mesh, sizeof(kSpv_ts_obstacle_mesh)},
        {kSpv_ts_obstacle_frag, sizeof(kSpv_ts_obstacle_frag)},
    };
    for (int i = 0; i < 9; i++)
        s.shaderModules[i] = createShaderModule(s.vkctx.device, bins[i].code, bins[i].bytes);

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    s.pipelinePolyline = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[0], s.shaderModules[1], s.shaderModules[2], samples);
    s.pipelinePolygon  = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[3], s.shaderModules[4], s.shaderModules[5], samples);
    s.pipelineObstacle = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[6], s.shaderModules[7], s.shaderModules[8], samples);
    s.renderPassSamples = 1;  // render pass + pipelines built single-sample above

    // Dummy buffers so every descriptor binding has a valid buffer before
    // the first upload_scene call. Most aren't CUDA-importable because they
    // hold placeholder data; the ones the uploader writes through CUDA are
    // marked importable.
    VkDeviceSize dummySize = 256;
    VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto makeDummy = [&](VkExternalBuffer& buf, bool needsCuda = false) {
        if (buf.buffer == VK_NULL_HANDLE)
            buf = createExternalBuffer(s.vkctx, dummySize, ssboUsage, needsCuda);
    };
    makeDummy(s.timestampsBuffer);
    makeDummy(s.int32Buffer);
    makeDummy(s.vertexBuffer);
    makeDummy(s.triangleBuffer);
    makeDummy(s.poseBuffer);
    makeDummy(s.floatBuffer);
    // Scene buffer: smaller dummy so uploadScene always creates a larger one
    s.sceneBuffer = createExternalBuffer(s.vkctx, 64, ssboUsage, false);
    makeDummy(s.polylinePoolBuffer, true);
    makeDummy(s.polygonPoolBuffer, true);
    makeDummy(s.obstaclePoolBuffer, true);
    makeDummy(s.colorPaletteBuffer);
    makeDummy(s.cameraIntrinsicsBuffer);
    makeDummy(s.cameraPoseBuffer);
    makeDummy(s.queryBuffer);
    updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);

    cudaStreamCreate(&s.copyStream);
    for (int i = 0; i < 2; i++) {
        cudaEventCreateWithFlags(&s.stagingReadyEvent[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.pinnedReadyEvent[i], cudaEventDisableTiming);
    }

    VK_DBG("[Vulkan] Timestamped renderer initialized\n");
}

//=============================================================================
// Buffer Resize Helpers
//=============================================================================

static const VkBufferUsageFlags SSBO_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

static void ensureBuffers(LudusTimestampedVkState& s, bool& changed)
{
    // We always want CUDA-importable buffers here so the uploader can write
    // directly through cudaMemcpyAsync. A dummy buffer that was created
    // without CUDA import (cuDevPtr == 0) must be recreated even if its
    // Vulkan-side size already exceeds the requested capacity, otherwise
    // upload_scene will hit the "no CUDA pointer" guard.
    auto resize = [&](VkExternalBuffer& buf, int capacity, size_t elemSize) {
        VkDeviceSize needed = (VkDeviceSize)capacity * elemSize;
        if (needed == 0) return;
        bool needs_cuda_upgrade = (buf.cuDevPtr == 0);
        bool needs_grow = (buf.buffer == VK_NULL_HANDLE || buf.size < needed);
        if (needs_grow || needs_cuda_upgrade) {
            VkDeviceSize target = std::max<VkDeviceSize>(needed, buf.size);
            resizeExternalBuffer(s.vkctx, buf, target, SSBO_USAGE, true);
            changed = true;
        }
    };

    resize(s.timestampsBuffer,       s.timestampsCapacity,    sizeof(int64_t));
    resize(s.int32Buffer,            s.int32Capacity,         sizeof(int32_t));
    resize(s.vertexBuffer,           s.vertexCapacity,        sizeof(Vertex));
    resize(s.triangleBuffer,         s.triangleCapacity,      sizeof(Triangle));
    resize(s.poseBuffer,             s.poseCapacity,          sizeof(CameraPose));
    resize(s.floatBuffer,            s.floatCapacity,         sizeof(float));
    resize(s.polylinePoolBuffer,     s.polylinePoolCapacity,  sizeof(TimestampedPolylinePool));
    resize(s.polygonPoolBuffer,      s.polygonPoolCapacity,   sizeof(TimestampedPolygonPool));
    resize(s.obstaclePoolBuffer,     s.obstaclePoolCapacity,  sizeof(ObstaclePool));
    resize(s.cameraIntrinsicsBuffer, s.cameraCapacity,        sizeof(FThetaCamera));
    resize(s.cameraPoseBuffer,       s.queryCapacity,         sizeof(CameraPose));
    resize(s.queryBuffer,            s.queryCapacity,         sizeof(RenderQuery));
}

// Rebuild the render pass + pipelines for a new sample count (their attachment
// layout and rasterizationSamples are fixed at creation). Caller must be idle.
static void rebuildRenderPassAndPipelines(LudusTimestampedVkState& s, VkSampleCountFlagBits samples)
{
    if (s.pipelinePolyline) { vkDestroyPipeline(s.vkctx.device, s.pipelinePolyline, nullptr); s.pipelinePolyline = VK_NULL_HANDLE; }
    if (s.pipelinePolygon)  { vkDestroyPipeline(s.vkctx.device, s.pipelinePolygon,  nullptr); s.pipelinePolygon  = VK_NULL_HANDLE; }
    if (s.pipelineObstacle) { vkDestroyPipeline(s.vkctx.device, s.pipelineObstacle, nullptr); s.pipelineObstacle = VK_NULL_HANDLE; }
    if (s.renderPass)       { vkDestroyRenderPass(s.vkctx.device, s.renderPass, nullptr); s.renderPass = VK_NULL_HANDLE; }

    s.renderPass = createTimestampedRenderPass(
        s.vkctx.device, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT, samples);

    s.pipelinePolyline = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[0], s.shaderModules[1], s.shaderModules[2], samples);
    s.pipelinePolygon  = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[3], s.shaderModules[4], s.shaderModules[5], samples);
    s.pipelineObstacle = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[6], s.shaderModules[7], s.shaderModules[8], samples);

    s.renderPassSamples = (samples == VK_SAMPLE_COUNT_1_BIT) ? 1 : (int)samples;
}

static void ensureFramebuffer(LudusTimestampedVkState& s, int width, int height, int layers)
{
    if (s.width == width && s.height == height && s.maxLayers >= layers)
        return;

    vkDeviceWaitIdle(s.vkctx.device);

    // Rebuild the render pass/pipelines if the sample count changed so they
    // match the framebuffer created below.
    int desiredSamplesInt = (s.msaaSamples > 1) ? s.msaaSamples : 1;
    if (s.renderPassSamples != desiredSamplesInt) {
        rebuildRenderPassAndPipelines(s, (VkSampleCountFlagBits)desiredSamplesInt);
    }

    if (s.framebuffer) { vkDestroyFramebuffer(s.vkctx.device, s.framebuffer, nullptr); s.framebuffer = VK_NULL_HANDLE; }
    destroyExternalImage(s.vkctx, s.colorImage);
    destroyExternalImage(s.vkctx, s.depthStencilImage);
    if (s.msaaSamples > 1) {
        destroyExternalImage(s.vkctx, s.colorImageMSAA);
        destroyExternalImage(s.vkctx, s.depthStencilImageMSAA);
    }

    s.width = width;
    s.height = height;
    s.maxLayers = layers;

    s.colorImage = createExternalImage(s.vkctx, width, height, layers,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_SAMPLE_COUNT_1_BIT, true);

    s.depthStencilImage = createExternalImage(s.vkctx, width, height, layers,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_SAMPLE_COUNT_1_BIT, false);

    std::vector<VkImageView> fbAttachments;
    if (s.msaaSamples > 1) {
        VkSampleCountFlagBits samples = (VkSampleCountFlagBits)s.msaaSamples;
        s.colorImageMSAA = createExternalImage(s.vkctx, width, height, layers,
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            samples, false);
        s.depthStencilImageMSAA = createExternalImage(s.vkctx, width, height, layers,
            VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            samples, false);
        fbAttachments = {s.colorImageMSAA.imageView, s.colorImage.imageView, s.depthStencilImageMSAA.imageView};
    } else {
        fbAttachments = {s.colorImage.imageView, s.depthStencilImage.imageView};
    }

    VkFramebufferCreateInfo fbCI = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbCI.renderPass = s.renderPass;
    fbCI.attachmentCount = (uint32_t)fbAttachments.size();
    fbCI.pAttachments = fbAttachments.data();
    fbCI.width = width;
    fbCI.height = height;
    fbCI.layers = layers;
    VK_CHECK(vkCreateFramebuffer(s.vkctx.device, &fbCI, nullptr, &s.framebuffer));

    size_t frameSize = (size_t)width * height * 4 * layers;
    if (frameSize > s.stagingBufferSize) {
        for (int i = 0; i < 2; i++) {
            if (s.stagingBuffer[i]) cudaFree(s.stagingBuffer[i]);
            cudaMalloc(&s.stagingBuffer[i], frameSize);
            s.stagingValid[i] = 0;
        }
        s.stagingBufferSize = frameSize;
    }
}

//=============================================================================
// Scene Upload
//=============================================================================

void ludusUploadCamerasVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const FThetaCamera* intrinsics, int numCameras)
{
    (void)nvdr_ctx;
    if (numCameras > s.cameraCapacity) {
        s.cameraCapacity = numCameras;
        resizeExternalBuffer(s.vkctx, s.cameraIntrinsicsBuffer,
            numCameras * sizeof(FThetaCamera), SSBO_USAGE, true);
    }
    s.numCameras = numCameras;

    cudaMemcpyAsync((void*)s.cameraIntrinsicsBuffer.cuDevPtr, intrinsics,
        numCameras * sizeof(FThetaCamera), cudaMemcpyDeviceToDevice, stream);
    s.sceneBuffersDirty = 1;

    updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);
}

void ludusUploadColorPaletteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const float* colors, int numColors)
{
    (void)nvdr_ctx;
    VkDeviceSize size = numColors * 4 * sizeof(float);
    resizeExternalBuffer(s.vkctx, s.colorPaletteBuffer, size, SSBO_USAGE, true);
    cudaMemcpyAsync((void*)s.colorPaletteBuffer.cuDevPtr, colors, size,
        cudaMemcpyDeviceToDevice, stream);
    s.colorPaletteSize = numColors;
    s.sceneBuffersDirty = 1;
    updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);
}

int ludusUploadSceneVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const TimestampedScene* sceneDesc,
    const TimestampedPolylinePool* polylinePools, int numPolylinePools,
    const TimestampedPolygonPool* polygonPools, int numPolygonPools,
    const ObstaclePool* obstaclePools, int numObstaclePools,
    int maxObstaclesInPool,
    int maxVarraysPerTsPolyline, int maxVarraysPerTsPolygon,
    const int64_t* timestamps, int numTimestamps,
    const int32_t* int32Data, int numInt32,
    const Vertex* vertices, int numVertices,
    const Triangle* triangles, int numTriangles,
    const CameraPose* poses, int numPoses,
    const float* floatData, int numFloats)
{
    (void)nvdr_ctx;
    int sceneId = s.numScenes++;

    s.timestampsCapacity        = std::max(s.timestampsCapacity,        s.timestampsUsed + numTimestamps);
    s.int32Capacity             = std::max(s.int32Capacity,             s.int32Used + numInt32);
    s.vertexCapacity            = std::max(s.vertexCapacity,            s.vertexUsed + numVertices);
    s.triangleCapacity          = std::max(s.triangleCapacity,          s.triangleUsed + numTriangles);
    s.poseCapacity              = std::max(s.poseCapacity,              s.poseUsed + numPoses);
    s.floatCapacity             = std::max(s.floatCapacity,             s.floatUsed + numFloats);
    s.polylinePoolCapacity      = std::max(s.polylinePoolCapacity,      s.polylinePoolUsed + numPolylinePools);
    s.polygonPoolCapacity       = std::max(s.polygonPoolCapacity,       s.polygonPoolUsed + numPolygonPools);
    s.obstaclePoolCapacity      = std::max(s.obstaclePoolCapacity,      s.obstaclePoolUsed + numObstaclePools);
    s.maxObstaclesPerPool       = std::max(s.maxObstaclesPerPool,       maxObstaclesInPool);
    s.maxPolylinePoolsPerScene  = std::max(s.maxPolylinePoolsPerScene,  numPolylinePools);
    s.maxPolygonPoolsPerScene   = std::max(s.maxPolygonPoolsPerScene,   numPolygonPools);
    s.maxCubePoolsPerScene      = std::max(s.maxCubePoolsPerScene,      numObstaclePools);
    s.maxVarraysPerTsPolyline   = std::max(s.maxVarraysPerTsPolyline,   maxVarraysPerTsPolyline);
    s.maxVarraysPerTsPolygon    = std::max(s.maxVarraysPerTsPolygon,    maxVarraysPerTsPolygon);

    int sceneCapNeeded = sceneId + 1;
    if (sceneCapNeeded > s.maxScenes) {
        s.maxScenes = sceneCapNeeded;
        resizeExternalBuffer(s.vkctx, s.sceneBuffer,
            s.maxScenes * sizeof(TimestampedScene), SSBO_USAGE, true);
    }

    bool changed = false;
    ensureBuffers(s, changed);

    auto copyAppend = [&](VkExternalBuffer& buf, const void* data, int count, size_t elemSize,
                          int& used, const char* tag) {
        if (count > 0 && data) {
            TORCH_CHECK(buf.cuDevPtr != 0,
                "upload_scene: buffer ", tag, " has no CUDA pointer (cudaImportable not set?)");
            size_t dstOff = (size_t)used * elemSize;
            size_t bytes = (size_t)count * elemSize;
            CUdeviceptr dst = buf.cuDevPtr + dstOff;
            cudaError_t e = cudaMemcpyAsync((void*)dst, data, bytes,
                cudaMemcpyDeviceToDevice, stream);
            TORCH_CHECK(e == cudaSuccess, "upload_scene: cudaMemcpyAsync ", tag,
                " failed: ", (int)e);
            VK_DBG("[Vulkan] upload %6s: count=%d bytes=%zu\n", tag, count, bytes);
        }
        int offset = used;
        used += count;
        return offset;
    };

    copyAppend(s.timestampsBuffer,   timestamps,    numTimestamps,    sizeof(int64_t),                  s.timestampsUsed, "ts");
    copyAppend(s.int32Buffer,        int32Data,     numInt32,         sizeof(int32_t),                  s.int32Used,      "i32");
    copyAppend(s.vertexBuffer,       vertices,      numVertices,      sizeof(Vertex),                   s.vertexUsed,     "vert");
    copyAppend(s.triangleBuffer,     triangles,     numTriangles,     sizeof(Triangle),                 s.triangleUsed,   "tri");
    copyAppend(s.poseBuffer,         poses,         numPoses,         sizeof(CameraPose),               s.poseUsed,       "pose");
    copyAppend(s.floatBuffer,        floatData,     numFloats,        sizeof(float),                    s.floatUsed,      "flt");
    copyAppend(s.polylinePoolBuffer, polylinePools, numPolylinePools, sizeof(TimestampedPolylinePool),  s.polylinePoolUsed, "pl");
    copyAppend(s.polygonPoolBuffer,  polygonPools,  numPolygonPools,  sizeof(TimestampedPolygonPool),   s.polygonPoolUsed,  "pg");
    copyAppend(s.obstaclePoolBuffer, obstaclePools, numObstaclePools, sizeof(ObstaclePool),             s.obstaclePoolUsed, "obs");

    TORCH_CHECK(s.sceneBuffer.cuDevPtr != 0,
        "upload_scene: sceneBuffer has no CUDA pointer (cudaImportable not set?)");
    cudaError_t e_scene = cudaMemcpyAsync(
        (void*)(s.sceneBuffer.cuDevPtr + sceneId * sizeof(TimestampedScene)),
        sceneDesc, sizeof(TimestampedScene), cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(e_scene == cudaSuccess,
        "upload_scene: cudaMemcpyAsync (scene desc) failed: ", (int)e_scene);

    s.sceneBuffersDirty = 1;
    updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);
    return sceneId;
}

void ludusRemoveSceneVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int sceneId, cudaStream_t stream)
{
    (void)nvdr_ctx;
    // Tombstone the scene descriptor's first int so shaders skip it. HostToDevice
    // (source is host memory) and synchronous so `zero` outlives the copy.
    const int zero = 0;
    cudaError_t e = cudaMemcpyAsync(
        (void*)(s.sceneBuffer.cuDevPtr + sceneId * sizeof(TimestampedScene)),
        &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(e == cudaSuccess, "remove_scene: cudaMemcpyAsync (tombstone) failed: ", (int)e);
    cudaStreamSynchronize(stream);
    s.sceneBuffersDirty = 1;
}

//=============================================================================
// Render Batch
//=============================================================================

void ludusRenderBatchVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const RenderQuery* queries, const CameraPose* cameraPoses,
    int numQueries, int width, int height)
{
    (void)nvdr_ctx;
    if (numQueries <= 0) return;

    VK_DBG("[Vulkan] renderBatch: nq=%d size=%dx%d\n", numQueries, width, height);

    ensureFramebuffer(s, width, height, numQueries);

    if (numQueries > s.queryCapacity) {
        s.queryCapacity = numQueries;
        resizeExternalBuffer(s.vkctx, s.queryBuffer,
            numQueries * sizeof(RenderQuery), SSBO_USAGE, true);
        resizeExternalBuffer(s.vkctx, s.cameraPoseBuffer,
            numQueries * sizeof(CameraPose), SSBO_USAGE, true);
        updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);
    }

    cudaError_t e1 = cudaMemcpyAsync((void*)s.queryBuffer.cuDevPtr, queries,
        numQueries * sizeof(RenderQuery), cudaMemcpyDeviceToDevice, stream);
    cudaError_t e2 = cudaMemcpyAsync((void*)s.cameraPoseBuffer.cuDevPtr, cameraPoses,
        numQueries * sizeof(CameraPose), cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(e1 == cudaSuccess, "renderBatch: cudaMemcpyAsync (queries) failed: ", (int)e1);
    TORCH_CHECK(e2 == cudaSuccess, "renderBatch: cudaMemcpyAsync (poses) failed: ", (int)e2);

    // Ensure all CUDA writes are visible before Vulkan reads. Simple
    // synchronous handoff: cudaStreamSynchronize -> vkQueueSubmit(fence) ->
    // vkWaitForFences. Future optimization: replace with timeline semaphore.
    cudaStreamSynchronize(stream);

    vkResetFences(s.vkctx.device, 1, &s.vkctx.fence);

    VkCommandBuffer cmd = s.vkctx.commandBuffer;
    VkCommandBufferBeginInfo beginCI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginCI));

    // CUDA->Vulkan coherence workaround for drivers where CUDA writes to
    // VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD memory are not made fully
    // visible to subsequent Vulkan shader reads even after a queue-family
    // ownership transfer. Reads each CUDA-imported buffer back to a host
    // staging copy and writes it back through vkCmdUpdateBuffer (which
    // goes through Vulkan's own coherent transfer path). Opt out via
    // LUDUS_VK_DIRECT_IMPORT=1 if the driver doesn't need this hack.
    // Not static: re-read each call so the env var isn't latched at first render.
    const bool kHostRoundtrip = (getenv("LUDUS_VK_DIRECT_IMPORT") == nullptr);
    if (kHostRoundtrip) {
        auto copyToVk = [&](VkExternalBuffer& buf) {
            if (buf.buffer == VK_NULL_HANDLE || buf.size == 0 || buf.cuDevPtr == 0) return;
            size_t sz = (size_t)buf.size;
            std::vector<uint8_t> host(sz);
            cudaMemcpy(host.data(), (void*)buf.cuDevPtr, sz, cudaMemcpyDeviceToHost);
            // vkCmdUpdateBuffer must be called in <=65536 byte chunks per Vulkan spec.
            for (size_t off = 0; off < sz; off += 65536) {
                size_t chunk = std::min((size_t)65536, sz - off);
                vkCmdUpdateBuffer(cmd, buf.buffer, off, chunk, host.data() + off);
            }
        };
        // Per-query buffers are rewritten by CUDA every render -> always push.
        copyToVk(s.cameraPoseBuffer);
        copyToVk(s.queryBuffer);
        // Scene/camera/palette buffers only change on upload/remove/clear, so
        // push them once after such an op instead of every frame.
        if (s.sceneBuffersDirty) {
            copyToVk(s.timestampsBuffer);
            copyToVk(s.int32Buffer);
            copyToVk(s.vertexBuffer);
            copyToVk(s.triangleBuffer);
            copyToVk(s.poseBuffer);
            copyToVk(s.floatBuffer);
            copyToVk(s.sceneBuffer);
            copyToVk(s.polylinePoolBuffer);
            copyToVk(s.polygonPoolBuffer);
            copyToVk(s.obstaclePoolBuffer);
            copyToVk(s.colorPaletteBuffer);
            copyToVk(s.cameraIntrinsicsBuffer);
            s.sceneBuffersDirty = 0;
        }

        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
              | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
              | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // Queue family ownership acquire from VK_QUEUE_FAMILY_EXTERNAL.
    // CUDA writes happen on an external queue family; this barrier transfers
    // ownership to the graphics queue and makes the writes visible to shaders.
    {
        std::vector<VkBufferMemoryBarrier> bufBarriers;
        auto addBarrier = [&](VkExternalBuffer& buf) {
            if (buf.buffer == VK_NULL_HANDLE || buf.size == 0 || buf.memFd < 0) return;
            VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            b.dstQueueFamilyIndex = s.vkctx.graphicsQueueFamily;
            b.buffer = buf.buffer;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(b);
        };
        addBarrier(s.timestampsBuffer);
        addBarrier(s.int32Buffer);
        addBarrier(s.vertexBuffer);
        addBarrier(s.triangleBuffer);
        addBarrier(s.poseBuffer);
        addBarrier(s.floatBuffer);
        addBarrier(s.sceneBuffer);
        addBarrier(s.polylinePoolBuffer);
        addBarrier(s.polygonPoolBuffer);
        addBarrier(s.obstaclePoolBuffer);
        addBarrier(s.colorPaletteBuffer);
        addBarrier(s.cameraIntrinsicsBuffer);
        addBarrier(s.cameraPoseBuffer);
        addBarrier(s.queryBuffer);

        if (!bufBarriers.empty()) {
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
                  | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
                  | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr,
                (uint32_t)bufBarriers.size(), bufBarriers.data(),
                0, nullptr);
        }
    }

    // Clear values must match attachment order from createTimestampedRenderPass.
    // Use a non-black diagnostic clear color when LUDUS_VK_CLEAR_RED is set so
    // we can distinguish "framebuffer was cleared but nothing was drawn on
    // top" from "the readback path returned uninitialised memory".
    float cr = 0.0f, cg = 0.0f, cb = 0.0f, ca = 0.0f;
    if (getenv("LUDUS_VK_CLEAR_RED")) {
        cr = 1.0f; ca = 1.0f;
    }
    VkClearValue clearValues[3] = {};
    if (s.msaaSamples >= 2) {
        clearValues[0].color = {{cr, cg, cb, ca}};          // MSAA color
        clearValues[1].color = {{cr, cg, cb, ca}};          // resolve color
        clearValues[2].depthStencil = {1.0f, 0};            // MSAA depth/stencil
    } else {
        clearValues[0].color = {{cr, cg, cb, ca}};          // color
        clearValues[1].depthStencil = {1.0f, 0};            // depth/stencil
    }

    VkRenderPassBeginInfo rpBegin = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass = s.renderPass;
    rpBegin.framebuffer = s.framebuffer;
    rpBegin.renderArea = {{0, 0}, {(uint32_t)width, (uint32_t)height}};
    rpBegin.clearValueCount = (s.msaaSamples > 1) ? 3 : 2;
    rpBegin.pClearValues = clearValues;
    // Refresh descriptors before the render pass: updateDescriptorSet may
    // reset/realloc the descriptor pool, which is illegal inside a render pass.
    updateDescriptorSet(s.vkctx.device, s.descriptorSet, s);

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Y-flip viewport: shaders use OpenGL-style NDC (+Y up); Vulkan framebuffer
    // origin is top-left. Negative height inverts Y so the math stays GL-style.
    VkViewport viewport = {0, (float)height, (float)width, -(float)height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {(uint32_t)width, (uint32_t)height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        s.pipelineLayout, 0, 1, &s.descriptorSet, 0, nullptr);

    LudusPushConstants pc = {};
    pc.u_width_polyline_regular    = s.widthPolylineRegular > 0 ? s.widthPolylineRegular : 7.0f;
    pc.u_width_polyline_bev        = s.widthPolylineBev     > 0 ? s.widthPolylineBev     : 4.0f;
    pc.u_width_ego_traj_regular    = s.widthEgoTrajRegular  > 0 ? s.widthEgoTrajRegular  : 12.0f;
    pc.u_width_ego_traj_bev        = s.widthEgoTrajBev      > 0 ? s.widthEgoTrajBev      : 5.0f;
    pc.u_width_wireframe           = s.widthWireframe       > 0 ? s.widthWireframe       : 2.0f;
    pc.u_resolution_scale          = s.resolutionScale      > 0 ? s.resolutionScale      : 1.0f;
    pc.u_depth_scaling             = s.depthScaling;
    pc.u_max_extrapolation_us      = s.maxExtrapolationUs;
    pc.u_color_palette_size        = s.colorPaletteSize;
    pc.u_num_queries               = numQueries;
    pc.u_tessellation_threshold    = s.tessellationThreshold;
    pc.u_max_tessellation_polyline = s.maxTessellationLevelPolyline;
    pc.u_max_tessellation_polygon  = s.maxTessellationLevelPolygon;
    pc.u_max_tessellation_cube     = s.maxTessellationLevelCube;
    pc.u_cull_radius_scale         = s.cullRadiusScale;
    pc.u_fog_enabled               = s.depthScaling;

    const VkShaderStageFlags pcStages = VK_SHADER_STAGE_TASK_BIT_EXT
                                      | VK_SHADER_STAGE_MESH_BIT_EXT
                                      | VK_SHADER_STAGE_FRAGMENT_BIT;

    const char* dbg = getenv("LUDUS_VK_PIPELINES");
    bool draw_polyline = !dbg || strchr(dbg, 'P');
    bool draw_polygon  = !dbg || strchr(dbg, 'G');
    bool draw_obstacle = !dbg || strchr(dbg, 'O');

    auto drawMeshTasks = s.vkctx.pfnCmdDrawMeshTasksEXT;

    if (draw_polyline && s.polylinePoolUsed > 0 && drawMeshTasks && s.pipelinePolyline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelinePolyline);
        pc.u_num_polyline_pools = std::max(1u, (uint32_t)s.maxPolylinePoolsPerScene);
        pc.u_max_varrays_per_pool = std::max(1u, (uint32_t)s.maxVarraysPerTsPolyline);
        pc.u_cube_pool_index = 0;
        vkCmdPushConstants(cmd, s.pipelineLayout, pcStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * pc.u_num_polyline_pools * pc.u_max_varrays_per_pool;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    if (draw_polygon && s.polygonPoolUsed > 0 && drawMeshTasks && s.pipelinePolygon) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelinePolygon);
        pc.u_num_polygon_pools = std::max(1u, (uint32_t)s.maxPolygonPoolsPerScene);
        pc.u_max_varrays_per_pool = std::max(1u, (uint32_t)s.maxVarraysPerTsPolygon);
        pc.u_cube_pool_index = 0;
        vkCmdPushConstants(cmd, s.pipelineLayout, pcStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * pc.u_num_polygon_pools * pc.u_max_varrays_per_pool;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    uint32_t maxObstacles = std::max(1u, (uint32_t)s.maxObstaclesPerPool);
    for (int poolIdx = 0; poolIdx < s.maxCubePoolsPerScene; poolIdx++) {
        if (!(draw_obstacle && drawMeshTasks && s.pipelineObstacle)) break;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelineObstacle);
        pc.u_max_obstacles = maxObstacles;
        pc.u_cube_pool_index = poolIdx;
        // Wireframe rendering is driven by the per-pool ObstaclePool.render_flags
        // SSBO field that the mesh shader reads (CUBE_FLAG_WIREFRAME bit).
        vkCmdPushConstants(cmd, s.pipelineLayout, pcStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * maxObstacles;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(s.vkctx.graphicsQueue, 1, &submitInfo, s.vkctx.fence));
    VK_CHECK(vkWaitForFences(s.vkctx.device, 1, &s.vkctx.fence, VK_TRUE, UINT64_MAX));
}

//=============================================================================
// Copy Results
//=============================================================================

void ludusCopyBatchResultsVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    uint8_t* outputPtr, int width, int height, int numQueries)
{
    (void)nvdr_ctx;
    size_t totalSize = (size_t)width * height * 4 * numQueries;

    // Pull rendered color image through a CUDA-importable readback buffer
    // owned by this state. Grow on demand.
    if (s.readbackBuffer.buffer == VK_NULL_HANDLE || s.readbackBuffer.size < totalSize) {
        if (s.readbackBuffer.buffer != VK_NULL_HANDLE)
            destroyExternalBuffer(s.vkctx, s.readbackBuffer);
        s.readbackBuffer = createExternalBuffer(s.vkctx, totalSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    }
    VkExternalBuffer& readbackBuf = s.readbackBuffer;

    VkCommandBuffer copyCmd = beginSingleTimeCommands(s.vkctx);

    VkMemoryBarrier fullBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fullBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    fullBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(copyCmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &fullBarrier, 0, nullptr, 0, nullptr);

    transitionImageLayout(copyCmd, s.colorImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        numQueries, VK_IMAGE_ASPECT_COLOR_BIT);

    std::vector<VkBufferImageCopy> regions(numQueries);
    for (int i = 0; i < numQueries; i++) {
        regions[i] = {};
        regions[i].bufferOffset = (VkDeviceSize)i * width * height * 4;
        regions[i].bufferRowLength = 0;
        regions[i].bufferImageHeight = 0;
        regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[i].imageSubresource.mipLevel = 0;
        regions[i].imageSubresource.baseArrayLayer = i;
        regions[i].imageSubresource.layerCount = 1;
        regions[i].imageOffset = {0, 0, 0};
        regions[i].imageExtent = {(uint32_t)width, (uint32_t)height, 1};
    }
    vkCmdCopyImageToBuffer(copyCmd, s.colorImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        readbackBuf.buffer, (uint32_t)regions.size(), regions.data());

    transitionImageLayout(copyCmd, s.colorImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        numQueries, VK_IMAGE_ASPECT_COLOR_BIT);

    endSingleTimeCommands(s.vkctx, copyCmd);

    cudaMemcpyAsync(outputPtr, (void*)readbackBuf.cuDevPtr, totalSize,
        cudaMemcpyDeviceToDevice, stream);
}

int ludusCopyBatchResultsToStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int width, int height, int numQueries)
{
    int idx = s.currentStagingIdx;
    s.currentStagingIdx = 1 - idx;

    size_t totalSize = (size_t)width * height * 4 * numQueries;
    if (totalSize > s.stagingBufferSize) {
        for (int i = 0; i < 2; i++) {
            if (s.stagingBuffer[i]) cudaFree(s.stagingBuffer[i]);
            cudaMalloc(&s.stagingBuffer[i], totalSize);
            s.stagingValid[i] = 0;
        }
        s.stagingBufferSize = totalSize;
    }

    ludusCopyBatchResultsVk(NVDR_CTX_PARAMS, s, stream, s.stagingBuffer[idx], width, height, numQueries);
    cudaEventRecord(s.stagingReadyEvent[idx], stream);

    s.stagingWidth = width;
    s.stagingHeight = height;
    s.stagingNumQueries = numQueries;
    s.stagingValid[idx] = 1;

    return idx;
}

void ludusCopyStagingToOutputVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx,
    uint8_t* outputPtr, int width, int height, int numQueries)
{
    (void)nvdr_ctx;
    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);
    size_t size = (size_t)width * height * 4 * numQueries;
    cudaMemcpy(outputPtr, s.stagingBuffer[stagingIdx], size, cudaMemcpyDeviceToDevice);
}

int ludusStartAsyncHostTransferVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx)
{
    (void)nvdr_ctx;
    if (!s.stagingValid[stagingIdx]) return -1;

    int pinnedIdx = s.currentPinnedIdx;
    s.currentPinnedIdx = 1 - pinnedIdx;

    size_t size = (size_t)s.stagingWidth * s.stagingHeight * 4 * s.stagingNumQueries;
    if (size > s.pinnedHostBufferSize) {
        for (int i = 0; i < 2; i++) {
            if (s.pinnedHostBuffer[i]) cudaFreeHost(s.pinnedHostBuffer[i]);
            cudaMallocHost(&s.pinnedHostBuffer[i], size);
            s.pinnedValid[i] = 0;
        }
        s.pinnedHostBufferSize = size;
    }

    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);
    cudaMemcpyAsync(s.pinnedHostBuffer[pinnedIdx], s.stagingBuffer[stagingIdx],
        size, cudaMemcpyDeviceToHost, s.copyStream);
    cudaEventRecord(s.pinnedReadyEvent[pinnedIdx], s.copyStream);

    s.pinnedWidth[pinnedIdx] = s.stagingWidth;
    s.pinnedHeight[pinnedIdx] = s.stagingHeight;
    s.pinnedNumQueries[pinnedIdx] = s.stagingNumQueries;
    s.pinnedValid[pinnedIdx] = 1;

    return pinnedIdx;
}

int ludusIsPinnedBufferReadyVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int pinnedIdx)
{
    (void)nvdr_ctx;
    if (!s.pinnedValid[pinnedIdx]) return 0;
    return (cudaEventQuery(s.pinnedReadyEvent[pinnedIdx]) == cudaSuccess) ? 1 : 0;
}

int ludusIsHostTransferCompleteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    int prev = 1 - s.currentPinnedIdx;
    return ludusIsPinnedBufferReadyVk(NVDR_CTX_PARAMS, s, prev);
}

int ludusEncodeJpegBatchStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx, int quality,
    std::vector<std::pair<uint8_t*, size_t>>& outJpegs)
{
    (void)nvdr_ctx;
    if (!s.nvjpegInitialized) {
        nvjpegCreateSimple(&s.nvjpegHandle);
        nvjpegEncoderStateCreate(s.nvjpegHandle, &s.nvjpegEncoderState, 0);
        nvjpegEncoderParamsCreate(s.nvjpegHandle, &s.nvjpegEncoderParams, 0);
        s.nvjpegInitialized = 1;
    }
    nvjpegEncoderParamsSetQuality(s.nvjpegEncoderParams, quality, 0);

    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);

    int w = s.stagingWidth;
    int h = s.stagingHeight;
    int n = s.stagingNumQueries;
    size_t layerSize = (size_t)w * h * 4;
    size_t rgbSize = (size_t)w * h * 3;

    if (rgbSize > s.jpegFlipBufferSize) {
        if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
        cudaMalloc(&s.jpegFlipBuffer, rgbSize);
        s.jpegFlipBufferSize = rgbSize;
    }

    outJpegs.resize(n);
    for (int i = 0; i < n; i++) {
        uint8_t* srcRgba = s.stagingBuffer[stagingIdx] + i * layerSize;
        launchRgbaToRgbFlip(srcRgba, s.jpegFlipBuffer, w, h, 0);
        cudaDeviceSynchronize();

        nvjpegImage_t img;
        memset(&img, 0, sizeof(img));
        img.channel[0] = s.jpegFlipBuffer;
        img.pitch[0] = w * 3;

        nvjpegEncodeImage(s.nvjpegHandle, s.nvjpegEncoderState, s.nvjpegEncoderParams,
            &img, NVJPEG_INPUT_RGBI, w, h, 0);

        size_t jpegSize = 0;
        nvjpegEncodeRetrieveBitstream(s.nvjpegHandle, s.nvjpegEncoderState, nullptr, &jpegSize, 0);

        if (jpegSize > s.jpegOutputBufferSize) {
            if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
            cudaMallocHost(&s.jpegOutputBuffer, jpegSize);
            s.jpegOutputBufferSize = jpegSize;
        }

        nvjpegEncodeRetrieveBitstream(s.nvjpegHandle, s.nvjpegEncoderState,
            s.jpegOutputBuffer, &jpegSize, 0);

        uint8_t* jpegCopy = (uint8_t*)malloc(jpegSize);
        memcpy(jpegCopy, s.jpegOutputBuffer, jpegSize);
        outJpegs[i] = {jpegCopy, jpegSize};
    }

    return n;
}

//=============================================================================
// Cleanup
//=============================================================================

void ludusClearScenesVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    (void)nvdr_ctx;
    s.numScenes = 0;
    s.timestampsUsed = s.int32Used = s.vertexUsed = s.triangleUsed = 0;
    s.poseUsed = s.floatUsed = 0;
    s.polylinePoolUsed = s.polygonPoolUsed = s.obstaclePoolUsed = 0;
    s.maxObstaclesPerPool = s.maxCubePoolsPerScene = 0;
    s.maxPolylinePoolsPerScene = s.maxPolygonPoolsPerScene = 0;
    s.maxVarraysPerTsPolyline = s.maxVarraysPerTsPolygon = 0;
    s.sceneBuffersDirty = 1;
}

void ludusTimestampedReleaseVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    (void)nvdr_ctx;
    if (s.vkctx.device) vkDeviceWaitIdle(s.vkctx.device);

    for (int i = 0; i < 2; i++) {
        if (s.stagingBuffer[i]) cudaFree(s.stagingBuffer[i]);
        if (s.pinnedHostBuffer[i]) cudaFreeHost(s.pinnedHostBuffer[i]);
        if (s.stagingReadyEvent[i]) cudaEventDestroy(s.stagingReadyEvent[i]);
        if (s.pinnedReadyEvent[i]) cudaEventDestroy(s.pinnedReadyEvent[i]);
    }
    if (s.copyStream) cudaStreamDestroy(s.copyStream);
    if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
    if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
    if (s.nvjpegInitialized) {
        nvjpegEncoderParamsDestroy(s.nvjpegEncoderParams);
        nvjpegEncoderStateDestroy(s.nvjpegEncoderState);
        nvjpegDestroy(s.nvjpegHandle);
    }

    for (int i = 0; i < 9; i++)
        if (s.shaderModules[i]) vkDestroyShaderModule(s.vkctx.device, s.shaderModules[i], nullptr);

    if (s.pipelinePolyline) vkDestroyPipeline(s.vkctx.device, s.pipelinePolyline, nullptr);
    if (s.pipelinePolygon)  vkDestroyPipeline(s.vkctx.device, s.pipelinePolygon, nullptr);
    if (s.pipelineObstacle) vkDestroyPipeline(s.vkctx.device, s.pipelineObstacle, nullptr);
    if (s.pipelineLayout)   vkDestroyPipelineLayout(s.vkctx.device, s.pipelineLayout, nullptr);
    if (s.descriptorPool)   vkDestroyDescriptorPool(s.vkctx.device, s.descriptorPool, nullptr);
    if (s.descriptorSetLayout) vkDestroyDescriptorSetLayout(s.vkctx.device, s.descriptorSetLayout, nullptr);

    if (s.framebuffer) vkDestroyFramebuffer(s.vkctx.device, s.framebuffer, nullptr);
    if (s.renderPass)  vkDestroyRenderPass(s.vkctx.device, s.renderPass, nullptr);

    destroyExternalImage(s.vkctx, s.colorImage);
    destroyExternalImage(s.vkctx, s.depthStencilImage);
    destroyExternalImage(s.vkctx, s.colorImageMSAA);
    destroyExternalImage(s.vkctx, s.depthStencilImageMSAA);

    destroyExternalBuffer(s.vkctx, s.timestampsBuffer);
    destroyExternalBuffer(s.vkctx, s.int32Buffer);
    destroyExternalBuffer(s.vkctx, s.vertexBuffer);
    destroyExternalBuffer(s.vkctx, s.triangleBuffer);
    destroyExternalBuffer(s.vkctx, s.poseBuffer);
    destroyExternalBuffer(s.vkctx, s.floatBuffer);
    destroyExternalBuffer(s.vkctx, s.sceneBuffer);
    destroyExternalBuffer(s.vkctx, s.polylinePoolBuffer);
    destroyExternalBuffer(s.vkctx, s.polygonPoolBuffer);
    destroyExternalBuffer(s.vkctx, s.obstaclePoolBuffer);
    destroyExternalBuffer(s.vkctx, s.colorPaletteBuffer);
    destroyExternalBuffer(s.vkctx, s.cameraIntrinsicsBuffer);
    destroyExternalBuffer(s.vkctx, s.cameraPoseBuffer);
    destroyExternalBuffer(s.vkctx, s.queryBuffer);
    destroyExternalBuffer(s.vkctx, s.readbackBuffer);

    destroyVkContext(s.vkctx);
    memset(&s, 0, sizeof(s));
}
